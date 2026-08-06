"""Stage 2 헤드룸 측정. 게이트 A1/A2/B/C/D 판정 (프로토콜 §4 Stage 2).

이 프로젝트의 분기점이다. RL 스택을 만들기 전에 네 가지를 답한다.

```text
A1  현재 상태에서 좋은 damping 이 존재하는가          absolute, H=1
A2  현실적 multiplier 로 접근 가능한가                narrow/wide, shrinking Q_max
B   행동 범위 제한의 손해                             absolute vs wide vs narrow, H=1
C1  쿼터 초기화의 시간 불일치                          shrinking vs fresh
C2  다단계 계획의 가치 (주 판정)                       shrinking vs one-step
C3  상태 피드백의 추가 가치                            shrinking vs committed
D   cost-to-target 헤드룸                             target 난이도별
```

주 컨트롤러는 ``shrinking-quota MPC`` 다 (프로토콜 D12). ``fresh`` 는 시간
불일치가 확인된 진단 baseline 이므로 판정에 쓰지 않는다. PPO 착수는 게이트 C
하나가 아니라 P1~P4 로 판단한다.

디바이스
--------
**CPU 가 기본이며 그것이 옳다.** 대상은 quadratic(d=32~100)과 Rosenbrock(d=2~10)
뿐이다. Stage 0 실측에서 10만 파라미터 MNIST MLP 조차 GPU 런치 오버헤드
지배(0.68 ms/gradient)였으므로 d=100 matvec 을 GPU 로 보내면 순손실이다.
GPU 는 Stage 3 이후에만 쓴다.

재개 가능
---------
run 하나가 끝나는 즉시 ``results/raw/<run>.jsonl`` 에 기록한다. 프로세스가
끊겨도 같은 명령을 다시 실행하면 완료된 조합을 건너뛴다.

사용법:

    # beam calibration (프로토콜 F). 이것을 먼저 하고 beam 을 고정한다.
    uv run python scripts/run_headroom.py --mode calibrate-beam

    # pilot: 예산/target 선정용. dev seed 만 사용
    uv run python scripts/run_headroom.py --mode pilot --beam 2

    # confirmatory: held-out seed. 프로토콜 freeze 이후에만
    uv run python scripts/run_headroom.py --mode confirmatory --beam 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from rl_newton.benchmark.metrics import TargetSpec
from rl_newton.benchmark.oracle import (
    DEV_SEEDS,
    HELD_OUT_SEEDS,
    SELECTION_SEEDS,
    HeadroomConfig,
    calibrate_beam_width,
    run_headroom,
)
from rl_newton.benchmark.store import ResultStore, environment_fingerprint
from rl_newton.optimizers.action_space import ABSOLUTE, NARROW, WIDE
from rl_newton.tasks.micro_neural import MicroNeuralSpec
from rl_newton.tasks.quadratics import QuadraticSpec
from rl_newton.tasks.rosenbrock import RosenbrockSpec
from rl_newton.utils.provenance import collect_provenance, config_hash, git_commit

# ---------------------------------------------------------------------------
# 프로토콜 D6 사전 등록 target. easy / medium / hard 3단계.
# **pilot 에서 확정한 뒤에는 바꾸지 않는다.** 변경 시 프로토콜 §9에 기록한다.
# ---------------------------------------------------------------------------
TARGETS: dict[str, dict[str, TargetSpec]] = {
    "spd": {
        "easy": TargetSpec("relative_loss", 1.0e-2),
        "medium": TargetSpec("relative_loss", 1.0e-4),
        "hard": TargetSpec("relative_loss", 1.0e-6),
    },
    "ill_conditioned": {
        "easy": TargetSpec("relative_loss", 1.0e-2),
        "medium": TargetSpec("relative_loss", 1.0e-4),
        "hard": TargetSpec("relative_loss", 1.0e-6),
    },
    "indefinite": {
        "easy": TargetSpec("relative_loss", 1.0e-1),
        "medium": TargetSpec("relative_loss", 1.0e-2),
        "hard": TargetSpec("relative_loss", 1.0e-3),
    },
    "rosenbrock": {
        "easy": TargetSpec("absolute_loss", 1.0e-1),
        "medium": TargetSpec("absolute_loss", 1.0e-2),
        "hard": TargetSpec("absolute_loss", 1.0e-4),
    },
    # micro-neural 은 teacher 가 학생보다 넓고 라벨 노이즈가 있어 loss 가 0 으로
    # 가지 않는다. 절대 target 을 쓰면 도달률이 0 이 되므로 **상대 target** 을 쓴다.
    # 달성 가능 상한은 참조 solver panel 로 별도 측정한다 (D25).
    "micro_neural": {
        "easy": TargetSpec("relative_loss", 5.0e-1),
        "medium": TargetSpec("relative_loss", 2.0e-1),
        "hard": TargetSpec("relative_loss", 1.0e-1),
    },
}


def pilot_specs() -> list:
    """pilot subset. 예산/target/beam 선정에만 쓴다."""
    return [
        QuadraticSpec(kind="spd", dimension=64, condition_number=1.0e2),
        QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e5),
        RosenbrockSpec(dimension=2),
    ]


def challenge_specs() -> list:
    """Challenge **selection** set. 프로토콜 D20 에서 freeze 했다.

    ``scripts/calibrate_challenge.py`` 의 baseline-only 측정 가능성 조건을 통과한
    spec 이다. **planner 결과를 보지 않고 선정했다.**

    ```text
    failure_rate = 0                     통과
    joint floor-hit rate <= 1/3          통과 (0/2)
    각 baseline median logΔ >= 1 nat     통과 (최소 8.33)
    median distance-to-ceiling >= 3 nat  통과 (최소 13.71)
    ```

    후보 5개가 전부 통과해 ``MAX_SPECS=4`` 를 초과했으므로 사전 등록된
    tie-break("``log10(κ)`` 간격을 가장 고르게 덮는다")를 적용했다. 아래 4개가
    ``log10(κ)`` 축을 간격 1로 균등하게 덮는다.

    **이 목록은 바꾸지 않는다.** 변경 시 프로토콜 §9 변경이력에 기록한다.
    """
    return [
        QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e3),
        QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e4),
        QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e5),
        QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e6),
    ]


def micro_neural_specs() -> list:
    """P4 micro-neural. **feedback 의 가치를 시험하는 것이 목적이다** (D24).

    같은 모델과 데이터를 두 regime 으로 나눈다. 핵심 질문은 "모델이 비선형인가" 가
    아니라 "초기 계획 시점에 미래 상태를 정확히 예측할 수 없는가" 다.

    ```text
    [R1] full_batch             전체 데이터로 gradient 와 HVP. 결정론적
    [R2] controlled_stochastic  고정 seed batch 시퀀스. step 마다 표본이 바뀜
    ```

    `C3 = shrinking − committed` 를 두 regime 에서 비교하면 `feedback 의 가치`와
    `예측 가능성`을 분리할 수 있다. D22 는 결정론적 quadratic 에서 `C3 = −0.044`
    였다. R1 도 비슷하고 R2 에서만 양수면 D24 의 두 번째 갈래가 성립한다.

    **모델과 데이터는 하나만 고정한다.** regime 만 다르다.
    """
    base = MicroNeuralSpec(
        input_dim=32,
        hidden_dim=128,
        n_classes=5,
        n_samples=512,
        teacher_hidden_dim=256,
        label_noise=0.05,
    )
    # 중간 batch 하나만 추가한다 (D29). 축을 전면 스캔하면 프로젝트가 끝없이
    # 늘어난다. 세 점이면 "모델 오차가 커질수록 planning 가치가 사라지는가" 라는
    # 전이 방향을 탐색적으로 볼 수 있다.
    return [
        base,
        replace(base, regime="controlled_stochastic", batch_size=128),
        replace(base, regime="controlled_stochastic", batch_size=64),
    ]


def nonlinear_diagnostic_specs() -> list:
    """비선형 진단 층. **설정 선택 점수에 넣지 않는다** (프로토콜 D20).

    ``rosen_d5`` 는 calibration 4개 조건을 모두 통과했으나 ``κ`` 축에 없어
    tie-break 규칙상 challenge selection set 에서 탈락했다.

    관측 사실은 이것이다.

    ```text
    best_static / best_open_loop / heuristic / C0 가 모두 정확히 1.8175 nat
    ```

    baseline panel 로는 controller 구분력을 확인하지 못했다. 다만 baseline 4종이
    같다고 planner 도 같다는 보장이 없으므로 "무가치한 task" 로 판정하지 않는다.

    사용 순서를 지킨다. quadratic 4개에서 설정 하나를 freeze **한 뒤** 그 설정만
    여기에 적용한다. 그러면 Rosenbrock 결과를 보고 설정을 조정했다는 문제가
    사라진다.
    """
    return [RosenbrockSpec(dimension=5)]


def confirmatory_specs() -> list:
    """confirmatory. pilot 과 다른 조건수와 차원을 포함한다."""
    return [
        QuadraticSpec(kind="spd", dimension=64, condition_number=1.0e2),
        QuadraticSpec(kind="spd", dimension=128, condition_number=1.0e3),
        QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e5),
        QuadraticSpec(kind="ill_conditioned", dimension=200, condition_number=1.0e6),
        RosenbrockSpec(dimension=2),
        RosenbrockSpec(dimension=10, randomize_start=True),
    ]


def _sha256(path: Path) -> str:
    """raw 결과 파일의 SHA-256. 대용량 raw 는 Git 에 넣지 않으므로 참조용이다."""
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    return obj


def build_config(args: argparse.Namespace) -> tuple[HeadroomConfig, dict]:
    if args.mode == "confirmatory":
        specs = confirmatory_specs()
        seeds = list(HELD_OUT_SEEDS)[: args.seeds]
        phase = "confirmatory"
    elif args.mode == "challenge":
        # D20. challenge selection set + selection seed. calibration seed(0,1) 와
        # 겹치지 않는다.
        specs = challenge_specs()
        seeds = list(SELECTION_SEEDS)[: args.seeds]
        phase = "challenge"
    elif args.mode == "challenge-heldout":
        # D24. 같은 challenge spec, held-out seed. **설정을 다시 고르지 않는다.**
        # 최종 효과 추정용이므로 phase 를 confirmatory 로 둔다.
        specs = challenge_specs()
        seeds = list(HELD_OUT_SEEDS)[: args.seeds]
        phase = "confirmatory"
    elif args.mode == "nonlinear-diagnostic":
        # D20. 설정 선택 점수에 넣지 않는다. quadratic 에서 freeze 한 설정만 적용한다.
        specs = nonlinear_diagnostic_specs()
        seeds = list(SELECTION_SEEDS)[: args.seeds]
        phase = "challenge"
    elif args.mode == "micro-neural":
        # D24 P4. 두 regime 에서 C3 를 비교해 feedback 의 가치를 시험한다.
        # **설정은 shrinking_Q4_narrow 로 freeze 됐다. 다시 고르지 않는다.**
        specs = micro_neural_specs()
        seeds = list(SELECTION_SEEDS)[: args.seeds]
        phase = "challenge"
    else:
        specs = pilot_specs()
        seeds = list(DEV_SEEDS)[: args.seeds]
        phase = "pilot"
    if args.max_tasks is not None:
        specs = specs[: args.max_tasks]

    if args.control_step_size:
        narrow, wide, absolute = NARROW, WIDE, ABSOLUTE
        condition = "step_size_controlled"
    else:
        narrow = NARROW.with_fixed_step_size(1.0)
        wide = WIDE.with_fixed_step_size(1.0)
        absolute = ABSOLUTE.with_fixed_step_size(1.0)
        condition = "step_size_fixed"

    config = HeadroomConfig(
        specs=specs,
        seeds=seeds,
        targets=TARGETS,
        cost_budget_ge=args.budget,
        max_steps=args.max_steps,
        quotas=tuple(args.quotas),
        beam_width=args.beam,
        max_plan_depth=args.max_plan_depth,
        fresh_diagnostic_seeds=args.fresh_seeds,
        run_fresh_wide=args.fresh_wide,
        execution_modes=tuple(args.modes),
        planner_spaces=tuple(args.planner_spaces),
        run_track_t=not args.skip_track_t,
        tuning_budget=args.tuning_budget,
        acceptance_loss=args.acceptance_loss,
        phase=phase,  # type: ignore[arg-type]
        primary_difficulty=args.difficulty,
        device="cpu",
    )
    meta = {
        "mode": args.mode,
        "condition": condition,
        "phase": phase,
        "seeds": seeds,
        "budget_ge": args.budget,
        "beam": args.beam,
        "quotas": list(args.quotas),
        "difficulty": args.difficulty,
        "n_specs": len(specs),
        "narrow_only": bool(args.narrow_only),
    }
    # **기본값이면 넣지 않는다.** `meta` 는 표시용 `config_hash` 를 만들고 그것이
    # raw 파일 경로를 정한다. 무조건 넣으면 기본 설정 실험의 파일 경로가 바뀌어
    # 기존 결과가 캐시에서 빠지고 전부 재실행된다. 정체성은 이미
    # `run_semantics_id` 가 담당한다 (D28).
    if args.acceptance_loss != "control":
        meta["acceptance_loss"] = args.acceptance_loss
    return config, meta | {"spaces": (narrow, wide, absolute)}


def build_parser() -> argparse.ArgumentParser:
    """CLI 파서. 테스트에서 sweep 커버리지 기본값을 검증하려고 분리했다.

    ``--modes`` 를 **생략한 경우와 빈 목록으로 준 경우**가 구별되어야 한다.
    재현 명령을 잘못 입력해 planner 전체가 조용히 빠지는 사고를 막는다.
    """
    parser = argparse.ArgumentParser(
        description="Stage 2 헤드룸 측정 (게이트 A1/A2/B/C1/C2/C3/D)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=[
            "calibrate-beam",
            "pilot",
            "challenge",
            "challenge-heldout",
            "nonlinear-diagnostic",
            "micro-neural",
            "confirmatory",
        ],
        default="pilot",
        help=(
            "challenge = D20 selection set (quadratic 4, seeds 2/3/4). "
            "challenge-heldout = 같은 spec, held-out seed. 설정을 다시 고르지 않는다 (D24). "
            "nonlinear-diagnostic = rosen_d5. 설정 선택에 쓰지 않는다. "
            "micro-neural = D24 P4. full_batch 와 controlled_stochastic 두 regime"
        ),
    )
    parser.add_argument("--seeds", type=int, default=3, help="사용할 seed 개수")
    parser.add_argument("--budget", type=float, default=600.0, help="GE 예산")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--beam", type=int, default=2)
    parser.add_argument(
        "--quotas",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0],
        help="게이트 C 쿼터 사다리 (c_max 배수)",
    )
    parser.add_argument(
        "--max-plan-depth",
        type=int,
        default=24,
        help="계획 길이 상한. 쿼터가 아니라 이것에 걸리면 사다리 비교가 훼손된다",
    )
    parser.add_argument(
        "--fresh-seeds",
        type=int,
        default=1,
        help="fresh(진단 baseline)를 돌릴 seed 수. 0 이면 제외. P1~P3 판정에 쓰지 않는다",
    )
    parser.add_argument(
        "--fresh-wide",
        action="store_true",
        help="fresh 를 wide 에서도 돌린다. 가장 비싼 조합이므로 기본은 끔",
    )
    # --- sweep 커버리지 (프로토콜 D13). run 정체성이 아니므로 기존 결과를 재사용한다.
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["shrinking", "committed", "fresh"],
        choices=["shrinking", "committed", "fresh"],
        help="돌릴 실행 방식. 빈 목록이면 planner 를 건너뛰고 baseline 만 확보한다",
    )
    parser.add_argument(
        "--planner-spaces",
        nargs="+",
        default=["narrow", "wide"],
        choices=["narrow", "wide"],
        help="planner 를 돌릴 행동 공간",
    )
    parser.add_argument(
        "--skip-track-t",
        action="store_true",
        help="Track T 를 건너뛴다. baseline 확보나 bridge 검증 단계에서 쓴다",
    )
    parser.add_argument(
        "--beams",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="calibrate-beam 에서 시험할 beam 폭",
    )
    parser.add_argument(
        "--cal-quotas",
        type=float,
        nargs="+",
        default=[1.0, 4.0],
        help="calibrate-beam 에서 시험할 쿼터 (c_max 배수)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="사용할 task spec 수 상한. dry run 용",
    )
    parser.add_argument(
        "--narrow-only",
        action="store_true",
        help="calibrate-beam 에서 narrow 만 사용. dry run 용",
    )
    parser.add_argument("--tuning-budget", type=int, default=None, help="N_tune")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument(
        "--control-step-size",
        action="store_true",
        help="step_size 축을 제어에 포함. 기본은 1.0 고정 (aliasing 회피)",
    )
    parser.add_argument(
        "--acceptance-loss",
        choices=["control", "fixed_eval"],
        default="control",
        help=(
            "수락 판정 목적함수 (D28). fixed_eval 은 step 마다 바뀌지 않는 목적함수로 "
            "판정한다. gradient/HVP 는 계속 minibatch 를 쓴다. 평가 forward 비용을 "
            "GE 회계에 포함한다"
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch CPU 스레드 수. wall-clock tie-break 안정화를 위해 고정한다",
    )
    parser.add_argument(
        "--concurrent-processes",
        type=int,
        default=1,
        help="동시 실행 중인 다른 실험 프로세스 수 (기록용)",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/summaries"))
    parser.add_argument("--fresh", action="store_true", help="캐시를 무시하고 새 파일에 기록")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config, meta = build_config(args)
    narrow, wide, absolute = meta.pop("spaces")

    commit, dirty = git_commit(".")
    cfg_hash = config_hash(_clean(meta))
    tag = f"{args.mode}_{meta['condition']}_b{args.beam}_{cfg_hash[:8]}"
    raw_path = args.raw_dir / f"headroom_{tag}.jsonl"
    if args.fresh and raw_path.exists():
        raw_path = raw_path.with_name(f"{raw_path.stem}_fresh{raw_path.suffix}")

    store = ResultStore(raw_path, git_commit=commit, config_hash=cfg_hash)

    # wall-clock 은 beam 선택의 마지막 tie-break 로만 쓰이지만, 다른 실험과 CPU
    # 를 공유하면 그 tie-break 가 흔들린다. 스레드를 고정하고 환경을 기록한다.
    env = environment_fingerprint(pin_threads=args.threads)
    meta["environment"] = env
    meta["concurrent_processes"] = args.concurrent_processes

    print("=" * 96)
    print(f"Stage 2 {args.mode}  조건={meta['condition']}  device=cpu (GPU 미사용)")
    print(f"  seeds={config.seeds}  GE 예산={config.cost_budget_ge:g}  beam={args.beam}")
    print(f"  git={commit}{' (dirty)' if dirty else ''}  config_hash={cfg_hash}")
    print(
        f"  torch threads={env['torch_num_threads']} "
        f"interop={env['torch_num_interop_threads']} cpu={env['cpu_count']}"
    )
    print(f"  raw={raw_path}  (이미 {len(store)}개 기록됨, 완료분은 건너뜀)")
    for space in (narrow, wide, absolute):
        print(
            f"  {space.name:<26} actions={len(space):>4} "
            f"HVP/sweep={space.hvp_per_sweep:>5} log10범위={space.log10_span:>6.2f}"
        )
    print("=" * 96)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "calibrate-beam":
        calibration = calibrate_beam_width(
            config,
            narrow=narrow,
            wide=narrow if args.narrow_only else wide,
            beams=tuple(args.beams),
            quotas=tuple(args.cal_quotas),
            store=store,
            code_dirty=dirty,
        )
        print("\n" + calibration.table())
        print(f"\n선택: beam {calibration.selected_beam}")
        print(f"  근거: {calibration.rationale}")
        path = args.out_dir / f"beam_calibration_{cfg_hash[:8]}.json"
        path.write_text(
            json.dumps(
                _clean(
                    {
                        "meta": meta,
                        "selected_beam": calibration.selected_beam,
                        "reference_beam": calibration.reference_beam,
                        "tolerance": calibration.tolerance,
                        "rationale": calibration.rationale,
                        "rows": {
                            f"{s}|H{h}|b{b}": row for (s, h, b), row in calibration.rows.items()
                        },
                    }
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"저장: {path}")
        return 0

    report = run_headroom(
        config,
        narrow=narrow,
        wide=wide,
        absolute=absolute,
        store=store,
        git_commit=commit,
        code_dirty=dirty,
        verbose=True,
    )

    print("\n" + "=" * 96)
    print("결과 요약 (Track E = logΔ, Track T = 도달률 + cost→τ)")
    print("=" * 96)
    print(report.summary_table())

    print("\n" + "=" * 96)
    print("Track E 쌍별 차이 (양수면 treatment 가 좋다, 단위 nat)")
    print("=" * 96)
    for d in report.track_e_deltas:
        print("  " + d.describe())

    print("\n" + "=" * 96)
    print("Track T 쌍별 비율 (1보다 크면 treatment 가 싸다)")
    print("=" * 96)
    for level, c in report.track_t_ratios.items():
        print(f"  [{level}] " + c.describe())

    print("\n" + "=" * 96)
    print("게이트 판정")
    print("=" * 96)
    for gate in report.gates:
        print("  " + gate.describe().replace("\n", "\n  "))

    verdicts = {g.name: g.verdict for g in report.gates}
    print("\n" + "-" * 96)
    print("  " + "  ".join(f"{name}={v}" for name, v in verdicts.items()))
    if args.mode == "pilot":
        print("\n  주의: pilot 결과다. 예산/target/beam 선정에만 쓰고 결론에 쓰지 않는다.")

    print(f"\n  {store.describe()}")
    failures = store.failures()
    if failures:
        print(f"  실패 {len(failures)}건:")
        for record in failures[:5]:
            print(f"    {record.key.as_str()}: {record.error}")

    path = args.out_dir / f"headroom_{tag}.json"
    payload = {
        "meta": meta,
        # 3계층 정체성 (프로토콜 D13). experiment_id 는 구판 호환용이다.
        "experiment_id": report.experiment_id,
        "sweep_id": report.sweep_id,
        "aggregation_id": report.aggregation_id,
        "identity": report.identity,
        "execution_provenance": report.provenance,
        # baseline 선택 근거 (프로토콜 D16). 라벨만 남기면 사후 선택이 된다.
        "selections": {name: m.to_json() for name, m in report.selections.items()},
        "n_instances": report.n_instances,
        "tuning_budget": report.tuning_budget,
        "tuning_runs": report.tuning_runs,
        "best_static_action": (
            asdict(report.best_static_action) if report.best_static_action else None
        ),
        "groups": {
            name: {k: v for k, v in asdict(g).items() if k != "runs"}
            for name, g in report.groups.items()
        },
        "track_e_deltas": [asdict(d) for d in report.track_e_deltas],
        "track_t_ratios": {k: asdict(v) for k, v in report.track_t_ratios.items()},
        "gates": [
            {
                "name": g.name,
                "track": g.track,
                "question": g.question,
                "statistic": g.statistic,
                "unit": g.unit,
                "verdict": g.verdict,
                "go_threshold": g.go_threshold,
                "pivot_threshold": g.pivot_threshold,
                "nonsaturated": g.nonsaturated,
                "detail": g.detail,
            }
            for g in report.gates
        ],
        "n_failures": len(failures),
        "raw_path": str(raw_path),
        # raw 는 gitignore 이므로 체크섬으로 참조 가능하게 한다 (프로토콜 D16).
        "raw_sha256": _sha256(raw_path),
        "raw_n_lines": (
            sum(1 for _ in raw_path.open(encoding="utf-8")) if raw_path.exists() else 0
        ),
        "provenance": collect_provenance(_clean(meta), include_diff=False).to_dict(),
    }
    path.write_text(json.dumps(_clean(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
