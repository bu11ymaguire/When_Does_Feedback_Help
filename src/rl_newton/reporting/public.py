"""공개 CSV 를 `RunSummary` 로 되돌리고 paired 통계를 계산한다.

왜 `RunSummary` 로 되돌리는가
-----------------------------
`compare_paired_delta` 가 `RunSummary` 를 받는다. 공개 CSV 를 위한 별도 통계 구현을
만들면 두 경로가 갈릴 수 있고, 이 프로젝트는 그런 사고를 이미 겪었다 (median 규약
불일치, E10). **같은 함수를 쓰는 것이 목적이다.**

무엇이 복원되고 무엇이 복원되지 않는가
--------------------------------------
```text
복원된다      pairing 키, initial/final loss, log_improvement, floor_hit,
              비용, 거절률, 도달 여부
복원되지 않는다 run_id, HVP 카운트, CG 수렴률, residual/damping/trust 중앙값
```

복원되지 않는 필드는 원고의 paired 통계에 쓰이지 않는다. 그 자리에는 값이 없음을
드러내는 표식(`NaN`, 빈 문자열)을 넣는다. **0 으로 채우지 않는다.** 0 은 측정값처럼
보이기 때문이다.

`task_instance_id` 는 `f"{task_spec}_seed{seed}"` 로 재구성한다. 원본과 같은 문자열이며,
설령 다르더라도 `(task_spec, seed)` 가 원본 pairing 과 일대일이므로 짝짓기 결과가
같다.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from rl_newton.benchmark.metrics import PairedDelta, RunSummary, compare_paired_delta

__all__ = [
    "PUBLIC_COLUMNS",
    "load_public_grouped",
    "load_public_results",
    "paired",
    "positive_count",
    "public_roles",
    "split_by",
]

PUBLIC_COLUMNS = (
    "raw_source",
    "role",
    "acceptance_rule",
    "beam_width",
    "budget_ge",
    "task_spec",
    "condition_number",
    "batch_size",
    "seed",
    "controller",
    "controller_role",
    "initial_loss",
    "final_loss",
    "log_improvement",
    "floor_hit",
    "total_cost_ge",
    "search_cost_ge",
    "n_steps",
    "stop_reason",
    "rejection_rate",
    "failure_rate",
    "negative_curvature_rate",
    "target",
    "reached",
    "cost_to_target_ge",
    "steps_to_target",
)

_NAN = float("nan")


def _opt_float(text: str) -> float | None:
    return None if text == "" else float(text)


def _opt_int(text: str) -> int | None:
    return None if text == "" else int(text)


def load_public_results(
    path: str | Path, *, acceptance_rule: str | None = None
) -> list[RunSummary]:
    """공개 CSV 한 개를 `RunSummary` 목록으로 읽는다.

    Args:
        path: `results/public/*.csv` 경로.
        acceptance_rule: 지정하면 그 수락 규칙의 행만 남긴다. `micro_neural.csv` 는
            `control` 과 `fixed_eval` 을 한 파일에 담으므로 보통 지정해야 한다.

    Returns:
        `RunSummary` 목록.

    Raises:
        ValueError: CSV 열 구성이 기대와 다르다.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or tuple(reader.fieldnames) != PUBLIC_COLUMNS:
            raise ValueError(
                f"{path.name} 의 열 구성이 기대와 다르다. "
                f"기대 {len(PUBLIC_COLUMNS)}개, 실제 {reader.fieldnames}"
            )
        rows = [dict(row) for row in reader]

    out: list[RunSummary] = []
    for row in rows:
        if acceptance_rule is not None and row["acceptance_rule"] != acceptance_rule:
            continue
        seed = int(row["seed"])
        out.append(
            RunSummary(
                # 복원 불가 필드는 값이 없음을 드러낸다. 0 으로 채우지 않는다.
                run_id="",
                controller=row["controller"],
                task_instance_id=f"{row['task_spec']}_seed{seed}",
                seed=seed,
                target=row["target"],
                reached=bool(int(row["reached"])),
                cost_to_target_ge=_opt_float(row["cost_to_target_ge"]),
                steps_to_target=_opt_int(row["steps_to_target"]),
                hvp_to_target=None,
                initial_loss=float(row["initial_loss"]),
                final_loss=float(row["final_loss"]),
                total_cost_ge=float(row["total_cost_ge"]),
                total_hvp=-1,
                search_cost_ge=float(row["search_cost_ge"]),
                n_steps=int(row["n_steps"]),
                stop_reason=row["stop_reason"],
                rejection_rate=float(row["rejection_rate"]),
                failure_rate=float(row["failure_rate"]),
                negative_curvature_rate=float(row["negative_curvature_rate"]),
                cg_convergence_rate=_NAN,
                median_residual_ratio=_NAN,
                median_damping=_NAN,
                median_trust_ratio=_NAN,
            )
        )
    return out


def split_by(
    runs: Iterable[RunSummary], key: Callable[[RunSummary], str]
) -> dict[str, list[RunSummary]]:
    """`key` 로 run 을 묶는다. controller 나 spec 별 분리에 쓴다."""
    out: dict[str, list[RunSummary]] = {}
    for run in runs:
        out.setdefault(key(run), []).append(run)
    return out


def public_roles(
    path: str | Path, *, acceptance_rule: str | None = None
) -> dict[str, str]:
    """역할 이름 -> 실제 controller 라벨.

    `best_static` 은 컨트롤러가 아니라 튜닝 선택 결과이고, 선택된 후보는 실험마다
    다르다. 이 대응이 없으면 공개 CSV 만으로는 원고의 `A2` 비교를 재현할 수 없다.

    Returns:
        예: `{"best_static": "static[2]", "best_open_loop": "open_loop[4]"}`.

    Raises:
        ValueError: 한 역할에 서로 다른 라벨이 두 개 이상 붙어 있다.
    """
    path = Path(path)
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if acceptance_rule is not None and row["acceptance_rule"] != acceptance_rule:
                continue
            role = row["controller_role"]
            if not role:
                continue
            found = out.setdefault(role, row["controller"])
            if found != row["controller"]:
                raise ValueError(
                    f"{path.name} 에서 역할 {role!r} 이 두 라벨을 가리킨다: "
                    f"{found!r}, {row['controller']!r}. "
                    "acceptance_rule 로 분리해야 한다."
                )
    return out


def load_public_grouped(
    path: str | Path, *, acceptance_rule: str | None = None
) -> tuple[dict[str, list[RunSummary]], dict[str, str]]:
    """`(controller -> runs, 역할 -> 라벨)`.

    `scripts/make_report.py` 와 `scripts/make_figures.py` 의 `load()` 와 **같은 모양**을
    반환한다. 그래서 두 스크립트가 raw 대신 공개 CSV 를 읽을 때 집계 코드를 복사하지
    않고 이 함수만 갈아 끼우면 된다.

    공개 저장소에는 raw 가 없으므로, 이 경로가 없으면 표와 그림을 재생성할 수 없다.
    """
    runs = load_public_results(path, acceptance_rule=acceptance_rule)
    return split_by(runs, lambda r: r.controller), public_roles(
        path, acceptance_rule=acceptance_rule
    )


def _select(
    runs: Sequence[RunSummary],
    baseline: str,
    treatment: str,
    spec: str | None,
    roles: dict[str, str] | None,
) -> tuple[list[RunSummary], list[RunSummary]]:
    def keep(run: RunSummary) -> bool:
        return spec is None or run.task_instance_id.rsplit("_seed", 1)[0] == spec

    lookup = roles or {}
    by_controller = split_by((r for r in runs if keep(r)), lambda r: r.controller)
    out: list[list[RunSummary]] = []
    for name in (baseline, treatment):
        label = lookup.get(name, name)
        if label not in by_controller:
            raise KeyError(
                f"controller {label!r} (요청 {name!r}) 가 없다. "
                f"있는 것: {sorted(by_controller)}"
            )
        out.append(by_controller[label])
    return out[0], out[1]


def paired(
    runs: Sequence[RunSummary],
    baseline: str,
    treatment: str,
    *,
    spec: str | None = None,
    roles: dict[str, str] | None = None,
) -> PairedDelta:
    """`baseline` 대 `treatment` 의 쌍별 차이.

    원고와 **같은 함수**(`compare_paired_delta`)를 쓴다. 기본값 `n_boot=10000`,
    `seed=0` 이고 리샘플링이 Python 표준 라이브러리 난수라 NumPy/SciPy 판본에
    의존하지 않는다. 같은 환경에서 CI 가 재현된다. 판본이 다른 환경에서의 비트 수준
    일치는 주장하지 않는다.

    Args:
        runs: 한 CSV 에서 읽은 run 목록.
        baseline: 기준 controller 이름 또는 역할 이름.
        treatment: 비교 대상.
        spec: 지정하면 그 `task_spec` 으로 한정한다.
        roles: `public_roles` 의 결과. `best_static` 같은 역할 이름을 쓸 때 필요하다.

    Raises:
        KeyError: 요청한 controller 가 없다.
    """
    base, treat = _select(runs, baseline, treatment, spec, roles)
    return compare_paired_delta(base, treat, metric="log_improvement")


def positive_count(
    runs: Sequence[RunSummary],
    baseline: str,
    treatment: str,
    *,
    spec: str | None = None,
    roles: dict[str, str] | None = None,
) -> tuple[int, int]:
    """양수 쌍 수와 유한한 쌍 수. 원고 표의 `35/40` 형태를 만든다."""
    base_runs, treat_runs = _select(runs, baseline, treatment, spec, roles)
    base = {(r.task_instance_id, r.seed): r for r in base_runs}
    treat = {(r.task_instance_id, r.seed): r for r in treat_runs}
    deltas = [
        treat[k].log_improvement - base[k].log_improvement
        for k in sorted(set(base) & set(treat))
        if math.isfinite(treat[k].log_improvement)
        and math.isfinite(base[k].log_improvement)
    ]
    return sum(1 for v in deltas if v > 0.0), len(deltas)
