"""micro-neural 두 regime 을 분리해 보고한다 (D24 P4).

전체 median 하나로 요약하면 regime 차이가 사라진다. D19 에서 spec 별 분해가
all-task median 과 반대 결론을 낸 것과 같은 문제다.

핵심 질문은 `C3` 가 regime 에 따라 달라지는가다.

```text
[R1] full_batch            결정론적. quadratic 처럼 planner 모델이 정확하다
[R2] controlled_stochastic step 마다 표본이 바뀐다. 초기 계획이 낡는다
```

`C3 > 0` 이 R2 에서만 나타나면 두 가지 해석이 가능하므로 **절대값을 함께 봐야
한다.**

```text
해석 A  shrinking 이 더 좋아졌다        -> feedback 이 이득을 만든다
해석 B  committed 가 더 나빠졌다        -> 낡은 계획이 손해를 만든다
```

둘은 다른 주장이다. B 라면 "feedback 이 가치를 창출한다" 가 아니라 "계획을
고수하면 손해다" 가 된다.

사용법:
    python scripts/analyze_regimes.py results/raw/<file>.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from rl_newton.benchmark.metrics import RunSummary, compare_paired_delta, median_of
from rl_newton.benchmark.store import ResultStore

_SEED_SUFFIX = re.compile(r"_seed\d+$")

# 보고 순서. 사다리를 그대로 따른다 (D26).
LADDER = (
    ("best_static", "best_static"),
    ("best_open_loop", "best_open_loop"),
    ("heuristic", "heuristic"),
    ("onestep_narrow", "onestep_narrow"),
    ("onestep_absolute", "onestep_absolute"),
    ("committed_Q4_narrow", "committed_Q4_narrow"),
    ("shrinking_Q4_narrow", "shrinking_Q4_narrow"),
)

COMPARISONS = (
    ("A2  shrinking − best_static", "best_static", "shrinking_Q4_narrow"),
    ("C2  shrinking − onestep", "onestep_narrow", "shrinking_Q4_narrow"),
    ("C3  shrinking − committed", "committed_Q4_narrow", "shrinking_Q4_narrow"),
    ("ref committed − onestep", "onestep_narrow", "committed_Q4_narrow"),
    ("ref open_loop − best_static", "best_static", "best_open_loop"),
)


def regime_of(run: RunSummary) -> str:
    base = _SEED_SUFFIX.sub("", run.task_instance_id)
    if base.endswith("_fb"):
        return "full_batch"
    tail = base.rsplit("_", 1)[-1]
    return f"controlled_stochastic({tail})" if tail.startswith("cs") else base


def median(values: list[float]) -> float:
    """프로젝트 단일 규약을 쓴다 (`metrics.median_of`)."""
    return median_of(values)


def resolve_alias(by_controller: dict[str, list[RunSummary]], summary: Path) -> dict[str, str]:
    """``best_static`` -> ``static[4]`` (프로토콜 D16)."""
    alias: dict[str, str] = {}
    if not summary.exists():
        return alias
    payload = json.loads(summary.read_text(encoding="utf-8"))
    for family, manifest in (payload.get("selections") or {}).items():
        label = manifest.get("selected_label")
        if label and label in by_controller:
            alias["best_static" if family == "static" else f"best_{family}"] = label
    return alias


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    by_controller: dict[str, list[RunSummary]] = {}
    for record in ResultStore(args.raw):
        if record.status != "completed" or record.summary is None:
            continue
        by_controller.setdefault(record.key.controller, []).append(record.summary)

    summary_path = args.summary or (
        args.raw.parents[1] / "summaries" / f"{args.raw.stem}.json"
    )
    alias = resolve_alias(by_controller, summary_path)

    regimes = sorted(
        {regime_of(r) for runs in by_controller.values() for r in runs}
    )

    print("=" * 96)
    print(f"regime 별 분해  {args.raw.name}")
    print("  전체 median 하나로 요약하면 regime 차이가 사라진다 (D19 교훈)")
    print("=" * 96)

    # --- 절대값 사다리 ---
    print()
    print("절대 median logΔ (높을수록 좋다)")
    header = f"  {'controller':<24}" + "".join(f"{r:>26}" for r in regimes)
    print(header)
    for label, name in LADDER:
        resolved = alias.get(name, name)
        runs = by_controller.get(resolved)
        if not runs:
            print(f"  {label:<24}" + "".join(f"{'없음':>26}" for _ in regimes))
            continue
        shown = f"{label} [{resolved}]" if resolved != name else label
        cells = []
        for regime in regimes:
            vals = [r.log_improvement for r in runs if regime_of(r) == regime]
            cells.append(f"{median(vals):>20.4f} n={len(vals)}")
        print(f"  {shown:<24}" + "".join(f"{c:>26}" for c in cells))

    # --- 거절률과 탐색 비용 ---
    print()
    print("거절률 / 탐색 GE (낡은 계획은 거절을 늘린다)")
    print(f"  {'controller':<24}" + "".join(f"{r:>26}" for r in regimes))
    for label, name in LADDER:
        resolved = alias.get(name, name)
        runs = by_controller.get(resolved)
        if not runs:
            continue
        cells = []
        for regime in regimes:
            sub = [r for r in runs if regime_of(r) == regime]
            if not sub:
                cells.append(f"{'-':>26}")
                continue
            rej = median([r.rejection_rate for r in sub])
            ge = median([r.search_cost_ge for r in sub])
            cells.append(f"{rej:>9.2f} {ge:>13.0f}")
        print(f"  {label:<24}" + "".join(f"{c:>26}" for c in cells))

    # --- paired delta ---
    print()
    print("=" * 96)
    print("regime 별 paired delta")
    print("=" * 96)
    for title, base, treat in COMPARISONS:
        rb = by_controller.get(alias.get(base, base))
        rt = by_controller.get(alias.get(treat, treat))
        print()
        print(f"  {title}")
        if not rb or not rt:
            print("    라벨 없음")
            continue
        for regime in ["ALL", *regimes]:
            b = rb if regime == "ALL" else [r for r in rb if regime_of(r) == regime]
            t = rt if regime == "ALL" else [r for r in rt if regime_of(r) == regime]
            if not b or not t:
                continue
            d = compare_paired_delta(b, t, metric="log_improvement")
            p = f"{d.p_value:.4f}" if math.isfinite(d.p_value) else "n/a"
            # 개별 delta 를 직접 계산해 함께 출력한다. n 이 작아 median 만으로는
            # 부호 일관성을 알 수 없다 (D19/D26).
            bmap = {(r.task_instance_id, r.seed): r for r in b}
            tmap = {(r.task_instance_id, r.seed): r for r in t}
            each = sorted(
                tmap[k].log_improvement - bmap[k].log_improvement
                for k in set(bmap) & set(tmap)
                if math.isfinite(tmap[k].log_improvement)
                and math.isfinite(bmap[k].log_improvement)
            )
            n_pos = sum(1 for v in each if v > 0.0)
            print(
                f"    {regime:<26} {d.median_delta:+8.3f} nat  "
                f"CI {d.delta_ci[0]:+7.3f}~{d.delta_ci[1]:+7.3f}  p={p:<7} "
                f"n={d.n_valid}  양수 {n_pos}/{len(each)}  "
                f"[{', '.join(f'{v:+.3f}' for v in each)}]"
            )
    print()
    print("주의: regime 당 n=3 이다. CI 와 p-value 가 거칠다. 부호 일관성을 함께 본다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
