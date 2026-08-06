"""`rosen_d5` 에서 모든 컨트롤러가 동일한 결과를 낸 원인을 진단한다.

12개 컨트롤러가 `logΔ=1.8175` 로 소수점 4자리까지 같고, planner 가 depth>1 을
27% 채택했는데도 결과가 바뀌지 않았다. 두 가설을 구별해야 한다.

```text
가설 1  step 이 damping / CG budget 에 둔감하다 (제어 자체가 무력)
가설 2  optimizer 가 어딘가에서 정체한다 (safe_fallback, 거절, negative curvature)
```

사용법:
    python scripts/diagnose_rosen.py results/raw/<file>.jsonl [--seed 2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.raw.open(encoding="utf-8")
        if line.strip()
    ]
    rows = [
        r
        for r in records
        if r.get("summary") and r["key"]["seed"] == args.seed and r["status"] == "completed"
    ]
    if not rows:
        print("해당 seed 의 완료된 run 이 없다.")
        return 2

    print(f"{args.raw.name}  seed={args.seed}  run {len(rows)}개")
    print()
    header = (
        f"{'controller':<24}{'final_loss':>16}{'steps':>6}{'rej':>6}"
        f"{'negcurv':>8}{'cgconv':>7}{'resid':>8}{'damp':>10}"
        f"{'trust':>8}{'objGE':>7}  stop"
    )
    print(header)
    print("-" * len(header))
    for record in sorted(rows, key=lambda r: r["key"]["controller"]):
        s = record["summary"]
        print(
            f"{record['key']['controller']:<24}"
            f"{s['final_loss']:>16.10f}"
            f"{s['n_steps']:>6}"
            f"{s['rejection_rate']:>6.2f}"
            f"{s['negative_curvature_rate']:>8.2f}"
            f"{s['cg_convergence_rate']:>7.2f}"
            f"{s['median_residual_ratio']:>8.3f}"
            f"{s['median_damping']:>10.2e}"
            f"{s['median_trust_ratio']:>8.3f}"
            f"{s['total_cost_ge']:>7.1f}"
            f"  {s['stop_reason']}"
        )

    print()
    losses = {r["summary"]["final_loss"] for r in rows}
    print(f"서로 다른 final_loss 값 {len(losses)}개")
    if len(losses) <= 3:
        for value in sorted(losses):
            names = sorted(
                r["key"]["controller"] for r in rows if r["summary"]["final_loss"] == value
            )
            print(f"  {value:.12f}  <- {len(names)}개 컨트롤러")
            print(f"      {', '.join(names)}")

    print()
    print("action 분포와 채택 깊이 (planner 만)")
    for record in sorted(rows, key=lambda r: r["key"]["controller"]):
        counts = record.get("action_counts") or {}
        depths = record.get("chosen_depths") or {}
        if not depths:
            continue
        print(f"  {record['key']['controller']}")
        print(f"    action_counts {counts}")
        print(f"    chosen_depths {depths}")
        stats = record.get("planner_stats") or {}
        if stats:
            trimmed = {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()
            }
            print(f"    planner_stats {trimmed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
