"""진단: paired delta 가 0인 쌍이 어느 spec 에서 나오는가 (프로토콜 D19).

3층 보고에서 ``rosen_d2`` 를 뺐는데도 primary 6쌍 중 3쌍이 정확히 ``+0.000`` 이었다.

```text
A2 primary 개별 [+0.000, +0.000, +0.000, +0.737, +0.494, +0.072]
C2 primary 개별 [+0.000, +0.000, +0.000, +0.873, +0.542, -0.191]
C3 primary 개별 [+0.000, +0.000, +0.000, +0.520, +0.230, +0.370]
```

포화가 아니다(primary 에는 포화 쌍이 없다). quadratic 한 spec 의 3 seed 전부에서
planner 가 baseline 과 같은 선택을 한다는 뜻이다.

**어느 spec 이 0인지에 따라 해석이 정반대다.**

```text
SPD k=1e2 가 0        잘 conditioned 된 문제에서는 damping 조절 필요성이 작다.
                      K=20 Newton step 이 충분하고 planner 도 같은 선택을 한다.
                      -> adaptive headroom 이 없는 regime. **유효한 결과다.**

ill k=1e5 가 0        어려운 문제에서 planner 가 대응하지 못한다.
                      action grid / damping 범위 / beam 폭 / 예산을 조사해야 한다.
                      -> 더 우려스럽다.
```

**delta 0 이라고 그 spec 을 제거하지 않는다.** 난이도별로 나눠 보고한다.

사용법:
    python scripts/probe_regime.py <raw.jsonl> <baseline_label> <treatment_label>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from rl_newton.benchmark.store import ResultStore


def spec_family(task_instance_id: str) -> str:
    """``quad_spd_d64_k1e+02_seed0`` -> ``quad_spd_d64_k1e+02``."""
    return task_instance_id.rsplit("_seed", 1)[0]


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 2
    path, base_label, treat_label = Path(argv[1]), argv[2], argv[3]
    if not path.exists():
        print(f"없음: {path}")
        return 1

    base: dict[tuple[str, int], object] = {}
    treat: dict[tuple[str, int], object] = {}
    for rec in ResultStore(path):
        if rec.status != "completed" or rec.summary is None:
            continue
        key = (rec.summary.task_instance_id, rec.summary.seed)
        if rec.key.controller == base_label:
            base[key] = rec
        elif rec.key.controller == treat_label:
            treat[key] = rec

    shared = sorted(set(base) & set(treat))
    print(f"{treat_label}  −  {base_label}   공통 {len(shared)}쌍\n")

    by_family: dict[str, list[float]] = {}
    for key in shared:
        br, tr = base[key], treat[key]  # type: ignore[assignment]
        bs, ts = br.summary, tr.summary  # type: ignore[attr-defined]
        delta = ts.log_improvement - bs.log_improvement
        family = spec_family(key[0])
        by_family.setdefault(family, []).append(delta)

        identical = (
            bs.final_loss == ts.final_loss
            and bs.total_cost_ge == ts.total_cost_ge
            and bs.n_steps == ts.n_steps
            and br.action_counts == tr.action_counts  # type: ignore[attr-defined]
        )
        tag = "완전 동일" if identical else ("delta 0" if delta == 0.0 else "다름")
        print(f"  {key[0]:<40} seed={key[1]}  Δ={delta:+.6f}  {tag}")
        if delta == 0.0 and not identical:
            # delta 만 0 이고 trajectory 가 다르면 floor 나 우연이다.
            print(
                f"      loss {bs.final_loss!r} vs {ts.final_loss!r}  "
                f"GE {bs.total_cost_ge:.1f} vs {ts.total_cost_ge:.1f}  "
                f"steps {bs.n_steps} vs {ts.n_steps}"
            )
        if identical:
            print(
                f"      action_counts {br.action_counts}"  # type: ignore[attr-defined]
            )
            tstats = tr.planner_stats or {}  # type: ignore[attr-defined]
            if tstats:
                keep = {
                    k: tstats[k]
                    for k in ("max_depth_seen", "suffix_retention_rate", "n_replans")
                    if k in tstats
                }
                print(f"      treatment planner {keep}")

    print()
    print("=== spec 별 요약 (난이도 regime 별 보고, 프로토콜 D19) ===")
    print(f"  {'spec':<40} {'n':>3} {'median Δ':>10} {'양수':>5} {'0':>4} {'음수':>5}")
    for family in sorted(by_family):
        deltas = by_family[family]
        finite = sorted(d for d in deltas if math.isfinite(d))
        med = finite[len(finite) // 2] if finite else float("nan")
        print(
            f"  {family:<40} {len(deltas):>3} {med:>10.4f} "
            f"{sum(1 for d in deltas if d > 0):>5} "
            f"{sum(1 for d in deltas if d == 0.0):>4} "
            f"{sum(1 for d in deltas if d < 0):>5}"
        )
    print()
    print("  전체 median 하나로 요약하지 않는다. 난이도에 따라 headroom 이")
    print("  달라지는 것 자체가 결과다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
