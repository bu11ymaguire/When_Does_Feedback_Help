"""Challenge set 선정: **비적응 baseline 만으로 측정 가능성**을 판정한다 (D20/D25).

**planner 결과를 보지 않는다.** 선정 기준은 성능 우열이 아니라 측정 가능성이다.
어떤 컨트롤러가 이기는지로 benchmark 를 고르면 결과를 본 뒤 유리한 task 를
추가한 것이 된다.

```text
사용 baseline:  best_static / best open_loop / heuristic / C0(onestep_narrow)
참조 solver:    lbfgs / adam / sgd_momentum / newton  (planner 아님)
비공개:         shrinking / committed / fresh / beam 결과
```

D25: ceiling 을 참조 solver panel 로 잰다
-----------------------------------------
초판(D20)은 ``ceiling = log(L0 / numerical_floor) = 31.44 nat`` 을 썼다. 이것은
전역최소점이 0 이고 도달 가능하다고 가정한다. `rosen_d5` 에서 그 가정이 깨졌다.

```text
D20 ceiling         31.44 nat   -> "여유 29.62 nat" 통과
실제 달성 가능 상한  1.8175 nat  -> 여유 0.0000 nat   (국소최소점 cap)
```

이제 ``J_achievable = log(L0) − log(max(L_ref, L_floor))`` 를 쓴다. ``L_ref`` 는
참조 solver panel 의 최소 final loss 다.

채택 조건 (D25 개정)
--------------------
```text
failure_rate = 0                              numerical failure 없음
joint floor-hit rate <= 1/3                   포화가 과도하지 않음
각 baseline median logΔ >= 1 nat              문제를 전혀 못 줄이는 조건 아님
J_achievable − median logΔ >= 3 nat           달성 가능 상한까지 여유
참조 solver 간 수렴점 산포 <= 0.5 nat          상한 추정이 안정적
seed 마다 실제로 다른 인스턴스                  seed 복제 금지
```

사용법:
    python scripts/calibrate_challenge.py [--budget 150] [--seeds 0 1]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace

import torch

from rl_newton.benchmark.eligibility import (
    REFERENCE_AGREEMENT_NAT,
    achievable_ceiling,
    check_seed_variation,
    reference_panel,
)
from rl_newton.benchmark.metrics import TargetSpec, summarize_run
from rl_newton.optimizers.action_space import NARROW
from rl_newton.optimizers.controllers import (
    FixedController,
    HeuristicController,
    OneStepEfficiencyController,
    make_open_loop_controller,
)
from rl_newton.optimizers.newton_cg import NewtonCGConfig, NewtonCGOptimizer
from rl_newton.tasks.micro_neural import MicroNeuralSpec, MicroNeuralTask
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask
from rl_newton.tasks.rosenbrock import RosenbrockSpec, RosenbrockTask

# --- 채택 조건 (사전 고정) ---
MAX_JOINT_FLOOR_RATE = 1.0 / 3.0
MIN_MEDIAN_LOG_IMPROVEMENT = 1.0
MIN_DISTANCE_TO_CEILING = 3.0
MAX_SPECS = 4

BASELINES = ("best_static", "best_open_loop", "heuristic", "onestep_narrow")

_MLP_BASE = MicroNeuralSpec(
    input_dim=32,
    hidden_dim=128,
    n_classes=5,
    n_samples=512,
    teacher_hidden_dim=256,
    label_noise=0.05,
)

# --- 후보군. conditioning 축을 촘촘히 (D20) ---
CANDIDATES: list[tuple[str, object]] = [
    ("quad_d100_k1e3", QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e3)),
    ("quad_d100_k1e4", QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e4)),
    ("quad_d100_k1e5", QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e5)),
    ("quad_d100_k1e6", QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e6)),
    ("rosen_d5", RosenbrockSpec(dimension=5)),
    ("rosen_d5_rand", RosenbrockSpec(dimension=5, randomize_start=True)),
    # D24 P4. 두 regime 이 같은 모델·데이터를 쓰고 optimizer 표본만 다르다.
    ("mlp_full_batch", _MLP_BASE),
    ("mlp_stochastic", replace(_MLP_BASE, regime="controlled_stochastic", batch_size=64)),
]

TARGET = TargetSpec("relative_loss", 1.0e-6)


def make(spec, seed: int, *, dtype: torch.dtype = torch.float32):
    if isinstance(spec, QuadraticSpec):
        return QuadraticTask(spec, seed=seed, dtype=dtype)
    if isinstance(spec, MicroNeuralSpec):
        return MicroNeuralTask(spec, seed=seed, dtype=dtype)
    return RosenbrockTask(spec, seed=seed, dtype=dtype)


def run_one(controller, spec, seed: int, budget: float):
    task = make(spec, seed)
    config = NewtonCGConfig(total_steps=300, cost_budget_ge=budget, initial_damping=1.0e-2)
    trace = NewtonCGOptimizer(task, controller, config, run_id="cal", seed=seed).run()
    return summarize_run(trace, TARGET, budget_ge=budget)


def baseline_panel(space, budget: float, spec, seed: int, n_tune: int):
    """비적응 baseline 만. planner 는 만들지 않는다."""
    out: dict[str, object] = {}

    # best_static: 후보를 균등 간격으로 n_tune 개 평가하고 최고를 고른다.
    best = None
    for i in range(min(n_tune, len(space))):
        flat = int(i * len(space) / min(n_tune, len(space)))
        s = run_one(FixedController(space.action_from_flat(flat)), spec, seed, budget)
        if best is None or s.log_improvement > best.log_improvement:
            best = s
    out["best_static"] = best

    # best open_loop: resource-clock 스케줄 (D17). 동일 n_tune.
    import random

    rng = random.Random(0)
    best_ol = None
    for _ in range(n_tune):
        flats = tuple(rng.randrange(len(space)) for _ in range(4))
        cuts = sorted(rng.uniform(0.05, 0.95) for _ in range(3))
        ctrl = make_open_loop_controller(space, flats, (*cuts, 1.0))
        s = run_one(ctrl, spec, seed, budget)
        if best_ol is None or s.log_improvement > best_ol.log_improvement:
            best_ol = s
    out["best_open_loop"] = best_ol

    out["heuristic"] = run_one(HeuristicController(space), spec, seed, budget)
    out["onestep_narrow"] = run_one(OneStepEfficiencyController(space), spec, seed, budget)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=150.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--n-tune", type=int, default=6, help="baseline 튜닝 후보 수")
    parser.add_argument(
        "--reference-iters",
        type=int,
        default=4000,
        help="참조 solver 반복 상한. 예산 제약이 상한을 만들지 않을 만큼 크게",
    )
    args = parser.parse_args()

    space = NARROW.with_fixed_step_size(1.0)
    print(f"Challenge set calibration  예산 {args.budget:g} GE  "
          f"calibration seeds={args.seeds}  N_tune={args.n_tune}")
    print("채택 조건 (D25):")
    print(f"  failure=0, joint floor<= {MAX_JOINT_FLOOR_RATE:.2f}, "
          f"median logΔ>= {MIN_MEDIAN_LOG_IMPROVEMENT:g}")
    print(f"  J_achievable 여유>= {MIN_DISTANCE_TO_CEILING:g} nat, "
          f"참조 산포<= {REFERENCE_AGREEMENT_NAT:g} nat, seed 복제 금지")
    print("**planner 결과는 열지 않는다.**\n")

    verdicts: list[tuple[str, bool, float, str]] = []
    for name, spec in CANDIDATES:
        # --- seed 복제 검사 (D23 원인 3) ---
        variation = check_seed_variation(
            lambda s, _spec=spec: make(_spec, s, dtype=torch.float64), args.seeds
        )

        # --- 달성 가능 상한 (D25) ---
        first = args.seeds[0]
        l0 = float(make(spec, first, dtype=torch.float64).initial_loss)
        panel = reference_panel(
            lambda _spec=spec, _seed=first: make(_spec, _seed, dtype=torch.float64),
            max_iter=args.reference_iters,
            extra_inits=(0.9,) if isinstance(spec, RosenbrockSpec) else (),
        )
        ceiling = achievable_ceiling(l0, panel)

        print(f"=== {name} ===")
        print(f"  {variation.describe()}")
        print(f"  {ceiling.describe()}")
        for run in ceiling.runs:
            mark = "수렴" if run.is_critical_point else "미수렴"
            print(
                f"    {run.name:<16} final={run.final_loss:.6e} "
                f"|grad|={run.grad_norm:.3e} {mark}"
            )

        rows = [
            baseline_panel(space, args.budget, spec, seed, args.n_tune)
            for seed in args.seeds
        ]

        ok = True
        reasons: list[str] = []
        if not variation.ok:
            ok = False
            reasons.append("seed 복제")
        if not ceiling.references_agree:
            ok = False
            reasons.append(f"참조 산포 {ceiling.reference_spread_nat:.2f} nat")

        print(f"  {'baseline':<18} {'median logΔ':>12} {'여유':>8} {'floor':>6} {'fail':>6}")
        worst_gap = math.inf
        for label in BASELINES:
            vals = [r[label].log_improvement for r in rows]  # type: ignore[index]
            finite = sorted(v for v in vals if math.isfinite(v))
            med = finite[len(finite) // 2] if finite else float("nan")
            n_floor = sum(1 for r in rows if r[label].floor_hit)  # type: ignore[index]
            n_fail = sum(1 for r in rows if r[label].failure_rate > 0.0)  # type: ignore[index]
            gap = ceiling.nats - med
            worst_gap = min(worst_gap, gap)
            print(
                f"  {label:<18} {med:>12.4f} {gap:>8.2f} "
                f"{n_floor}/{len(rows)}  {n_fail}/{len(rows)}"
            )
            if n_fail:
                ok = False
                reasons.append(f"{label} numerical failure")
            if n_floor / len(rows) > MAX_JOINT_FLOOR_RATE:
                ok = False
                reasons.append(f"{label} floor-hit {n_floor}/{len(rows)}")
            if not math.isfinite(med) or med < MIN_MEDIAN_LOG_IMPROVEMENT:
                ok = False
                reasons.append(f"{label} median logΔ {med:.3f} < {MIN_MEDIAN_LOG_IMPROVEMENT}")
            if not math.isfinite(gap) or gap < MIN_DISTANCE_TO_CEILING:
                ok = False
                reasons.append(f"{label} 달성가능 여유 {gap:.2f} < {MIN_DISTANCE_TO_CEILING}")

        verdict = "채택" if ok else "탈락"
        print(f"  -> {verdict}" + (f"  ({'; '.join(reasons[:3])})" if reasons else ""))
        print()
        verdicts.append((name, ok, worst_gap, "; ".join(reasons[:3])))

    accepted = [v for v in verdicts if v[1]]
    print("=== 결과 ===")
    for name, ok, gap, why in verdicts:
        print(f"  {name:<18} {'채택' if ok else '탈락':<4} 최소여유={gap:>7.2f}  {why}")
    print()
    print(f"  통과 {len(accepted)}개 / 후보 {len(verdicts)}개")
    if len(accepted) > MAX_SPECS:
        print(f"  최대 {MAX_SPECS}개 초과. log10(κ) 간격을 고르게 덮는 spec 을 선택한다.")
        print("  **planner 성능은 선택에 쓰지 않는다.**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
