"""두 컨트롤러의 결과를 **쌍별로** 비교한다. 중앙값만 같은지 전부 같은지 구분한다.

재집계에서 ``committed_Q1_narrow`` 와 ``shrinking_Q1_narrow`` 의 median logΔ 가
소수점 4자리까지 같았다 (9.3148). 네 자리 표만으로는 다음을 구분할 수 없다.

```text
median 만 같음            우연
각 final_loss 가 근사적으로 같음   전략이 비슷함
각 final_loss 와 trace 가 완전히 같음   합법적 동률 또는 alias
```

Q1 에서는 쿼터가 작아 재계획으로 얻을 선택지가 거의 없으므로 실제 동률일 수
있다. 그때는 매 재계획에서 ``new_plan == previous_suffix`` 다.

사용법:
    python scripts/compare_controllers.py <raw.jsonl> <exp_prefix> <ctrlA> <ctrlB>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from rl_newton.benchmark.store import ResultStore


def main(argv: list[str]) -> int:
    if len(argv) < 5:
        print(__doc__)
        return 2
    path, prefix, name_a, name_b = Path(argv[1]), argv[2], argv[3], argv[4]

    store = ResultStore(path)
    a: dict[tuple[str, int], object] = {}
    b: dict[tuple[str, int], object] = {}
    for rec in store:
        if rec.status != "completed" or rec.summary is None:
            continue
        if not rec.key.experiment_id.startswith(prefix):
            continue
        key = (rec.key.task_instance_id, rec.key.seed)
        if rec.key.controller == name_a:
            a[key] = rec
        elif rec.key.controller == name_b:
            b[key] = rec

    shared = sorted(set(a) & set(b))
    print(f"{name_a}  vs  {name_b}   공통 쌍 {len(shared)}개\n")

    n_bitwise = 0
    n_close = 0
    n_diff = 0
    for key in shared:
        ra, rb = a[key], b[key]  # type: ignore[assignment]
        sa, sb = ra.summary, rb.summary  # type: ignore[attr-defined]
        same_loss = sa.final_loss == sb.final_loss
        same_cost = sa.total_cost_ge == sb.total_cost_ge
        same_steps = sa.n_steps == sb.n_steps
        same_actions = ra.action_counts == rb.action_counts  # type: ignore[attr-defined]
        same_depths = ra.chosen_depths == rb.chosen_depths  # type: ignore[attr-defined]
        identical = all((same_loss, same_cost, same_steps, same_actions, same_depths))
        if identical:
            n_bitwise += 1
            tag = "완전 동일"
        elif math.isclose(sa.final_loss, sb.final_loss, rel_tol=1e-9):
            n_close += 1
            tag = "loss 근사 동일, trace 다름"
        else:
            n_diff += 1
            tag = "다름"

        print(f"  {key[0]:<34} seed={key[1]}  -> {tag}")
        if identical:
            continue
        print(
            f"    final_loss   A={sa.final_loss!r}  B={sb.final_loss!r}  "
            f"동일={same_loss}"
        )
        print(
            f"    logΔ         A={sa.log_improvement:.6f}  B={sb.log_improvement:.6f}"
        )
        print(
            f"    object GE    A={sa.total_cost_ge:.2f}  B={sb.total_cost_ge:.2f}  "
            f"steps A={sa.n_steps} B={sb.n_steps}"
        )
        print(f"    chosen_depths A={ra.chosen_depths}  B={rb.chosen_depths}")  # type: ignore[attr-defined]
        if not same_actions:
            print(f"    action_counts A={ra.action_counts}")  # type: ignore[attr-defined]
            print(f"                  B={rb.action_counts}")  # type: ignore[attr-defined]
        pa = ra.planner_stats or {}  # type: ignore[attr-defined]
        pb = rb.planner_stats or {}  # type: ignore[attr-defined]
        print(f"    planner A={pa}")
        print(f"            B={pb}")

    print()
    print(f"  완전 동일 {n_bitwise} / loss만 동일 {n_close} / 다름 {n_diff}")
    if n_bitwise == len(shared) and shared:
        print("  -> 모든 쌍이 완전 동일. 실제 동률인지 alias 인지 추가 확인 필요.")
    elif n_diff:
        print("  -> 쌍마다 다르다. median 이 같았던 것은 우연이거나 분포 특성이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
