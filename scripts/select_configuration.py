"""프로토콜 D21 의 설정 선택 규칙을 **결정론적으로** 적용한다.

무엇을 고르는가
---------------
``shrinking`` 의 ``(Q, action space)`` 하나다. ``committed`` 는 배포 가능한
컨트롤러가 아니라 초기 상태에 조건화된 비교 oracle(C3)이고 ``fresh`` 는 진단
baseline 이므로 선택 대상이 아니다.

선택 통계 (D21)
---------------
```text
median over challenge 인스턴스 of  shrinking 의 Track E log improvement
목표: 최대화.  baseline 과의 delta 로 고르지 않는다
```

baseline delta 를 쓰지 않는 이유는 strongest baseline 순위가 현재 표본에서
안정적이지 않기 때문이다 (`onestep_absolute` 와 `heuristic` 이 둘 다 31.438,
p=0.906). 불안정한 기준점으로 나눗셈을 하면 설정 선택이 baseline 잡음을 따라간다.

tie-break 사다리 (median 이 ``TIE_TOLERANCE`` 이내로 같을 때 순서대로)
```text
1  decision-search GE 가 적은 쪽
2  Q 가 작은 쪽
3  narrow 가 wide 보다 우선
```

함께 출력하되 **선택에 쓰지 않는 것**: spec 별 median, 개별 delta, A2/C2/C3.

사용법:
    python scripts/select_configuration.py results/raw/<file>.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from rl_newton.benchmark.metrics import RunSummary, compare_paired_delta, median_of
from rl_newton.benchmark.store import ResultStore

TIE_TOLERANCE = 0.05
"""nat. 이 안이면 동률로 보고 tie-break 사다리로 내려간다 (D21)."""

_LABEL = re.compile(r"^shrinking_Q(?P<q>[\d.]+)_(?P<space>narrow|wide)$")

_SEED_SUFFIX = re.compile(r"_seed\d+$")


def spec_of(run: RunSummary) -> str:
    """``quad_..._k1e+03_seed2`` -> ``quad_..._k1e+03``."""
    return _SEED_SUFFIX.sub("", run.task_instance_id)


def median(values: list[float]) -> float:
    """프로젝트 단일 규약을 쓴다 (`metrics.median_of`).

    D21 의 선택 통계가 게이트 통계와 다른 median 규약을 쓰면 안 된다.
    """
    return median_of(values)


def load(path: Path) -> dict[str, list[RunSummary]]:
    by_controller: dict[str, list[RunSummary]] = {}
    for record in ResultStore(path):
        if record.status != "completed" or record.summary is None:
            continue
        by_controller.setdefault(record.key.controller, []).append(record.summary)
    return by_controller


def resolve_tuned_labels(
    by_controller: dict[str, list[RunSummary]], summary_path: Path
) -> dict[str, str]:
    """``best_static`` -> ``static[2]`` 매핑 (프로토콜 D16).

    ``best_static`` / ``best_open_loop`` 는 컨트롤러가 아니라 **튜닝 결과**이므로
    raw 에는 선택된 후보 라벨로 저장된다. SelectionManifest 에서 되돌린다.
    라벨만 보고 최고 점수를 다시 고르면 선택 근거가 소실된다.
    """
    alias: dict[str, str] = {}
    if not summary_path.exists():
        return alias
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for family, manifest in (payload.get("selections") or {}).items():
        label = manifest.get("selected_label")
        if not label or label not in by_controller:
            continue
        alias["best_static" if family == "static" else f"best_{family}"] = label
    return alias


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="SelectionManifest 를 읽을 summary JSON. 기본은 raw 이름에서 유추",
    )
    args = parser.parse_args()

    by_controller = load(args.raw)
    summary_path = args.summary or (
        args.raw.parents[1] / "summaries" / f"{args.raw.stem}.json"
    )
    alias = resolve_tuned_labels(by_controller, summary_path)
    candidates = {
        name: runs for name, runs in by_controller.items() if _LABEL.match(name)
    }
    if not candidates:
        print("shrinking_Q*_{narrow,wide} 라벨이 없다. 실행이 끝났는지 확인해라.")
        return 2

    print("=" * 92)
    print(f"D21 설정 선택  {args.raw.name}")
    print("  선택 통계 = shrinking 자신의 median logΔ (최대화). delta 로 고르지 않는다")
    print(f"  동률 허용 = {TIE_TOLERANCE:g} nat, tie-break = search GE -> 작은 Q -> narrow")
    print("=" * 92)

    rows = []
    for name in sorted(candidates):
        m = _LABEL.match(name)
        assert m is not None
        runs = candidates[name]
        rows.append(
            {
                "name": name,
                "q": float(m.group("q")),
                "space": m.group("space"),
                "n": len(runs),
                "median_log": median([r.log_improvement for r in runs]),
                "median_search_ge": median([r.search_cost_ge for r in runs]),
                "n_floor": sum(1 for r in runs if r.floor_hit),
                "n_fail": sum(1 for r in runs if r.failure_rate > 0.0),
                "runs": runs,
            }
        )

    print()
    print(f"  {'configuration':<22} {'n':>3} {'median logΔ':>12} {'search GE':>12} "
          f"{'floor':>7} {'fail':>6}")
    for row in sorted(rows, key=lambda r: -r["median_log"]):
        print(
            f"  {row['name']:<22} {row['n']:>3} {row['median_log']:>12.4f} "
            f"{row['median_search_ge']:>12.1f} {row['n_floor']:>3}/{row['n']:<3} "
            f"{row['n_fail']:>2}/{row['n']:<3}"
        )

    best = max(r["median_log"] for r in rows)
    tied = [r for r in rows if best - r["median_log"] <= TIE_TOLERANCE]
    print()
    if len(tied) == 1:
        selected = tied[0]
        reason = f"median logΔ 최대 ({selected['median_log']:.4f} nat), 단독"
    else:
        print(f"  동률 {len(tied)}개 (최대 {best:.4f} 에서 {TIE_TOLERANCE:g} nat 이내):")
        for row in tied:
            print(f"    {row['name']:<22} {row['median_log']:.4f}")
        selected = min(
            tied,
            key=lambda r: (r["median_search_ge"], r["q"], 0 if r["space"] == "narrow" else 1),
        )
        reason = (
            f"동률 {len(tied)}개 -> tie-break: search GE {selected['median_search_ge']:.1f}, "
            f"Q={selected['q']:g}, {selected['space']}"
        )

    print(f"  선택: {selected['name']}")
    print(f"    근거: {reason}")

    # --- 이하 전부 참고 정보다. 선택에 쓰지 않는다 (D21) ---
    print()
    print("=" * 92)
    print("아래는 참고 정보다. **선택에 쓰지 않았다** (프로토콜 D21)")
    print("=" * 92)

    specs = sorted({spec_of(r) for r in selected["runs"]})
    print()
    print(f"  spec 별 median logΔ  ({selected['name']})")
    for spec in specs:
        vals = [r.log_improvement for r in selected["runs"] if spec_of(r) == spec]
        print(f"    {spec:<40} {median(vals):>9.4f}  n={len(vals)}")

    print()
    print(f"  개별 run  ({selected['name']})")
    for run in sorted(selected["runs"], key=lambda r: (spec_of(r), r.seed)):
        flags = []
        if run.floor_hit:
            flags.append("floor")
        if run.exact_zero:
            flags.append("exact0")
        print(
            f"    {spec_of(run):<40} seed{run.seed}  logΔ={run.log_improvement:>9.4f}  "
            f"obj GE={run.total_cost_ge:>6.1f}  {' '.join(flags)}"
        )

    comparisons = [
        ("A2  vs best_static", "best_static"),
        ("C2  vs onestep", f"onestep_{selected['space']}"),
        ("C3  vs committed", f"committed_Q{selected['q']:g}_{selected['space']}"),
        ("ref vs open_loop", "best_open_loop"),
        ("ref vs heuristic", "heuristic"),
    ]
    label_width = 40
    print()
    print("  paired delta (양수면 shrinking 이 좋다). 헤드룸 주장용이며 선택에 쓰지 않았다")
    for label, base in comparisons:
        resolved = alias.get(base, base)
        base_runs = by_controller.get(resolved)
        shown = f"{label} [{resolved}]" if resolved != base else label
        if not base_runs:
            print(f"    {shown:<{label_width}} 라벨 없음 ({base})")
            continue
        d = compare_paired_delta(base_runs, selected["runs"], metric="log_improvement")
        p = f"{d.p_value:.4f}" if math.isfinite(d.p_value) else "n/a"
        print(
            f"    {shown:<{label_width}} {d.median_delta:+8.4f} nat  "
            f"CI {d.delta_ci[0]:+7.3f}~{d.delta_ci[1]:+7.3f}  p={p:<7} "
            f"n={d.n_valid}/{d.n_pairs}  "
            f"joint={d.n_joint_saturated} one-sided={d.n_one_sided_saturated}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
