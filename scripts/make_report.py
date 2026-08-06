"""Stage 2 결과를 논문용 표로 모은다.

여러 raw 파일을 읽어 하나의 Markdown 표로 만든다. 손으로 숫자를 옮기다 생기는
오류를 없애는 것이 목적이다.

통계 표기 규칙 (D26)
--------------------
```text
p 는 0.0000 으로 쓰지 않는다. 부트스트랩/순열 기반이므로 p<0.0001 로 쓴다
equivalence margin 을 사전 등록하지 않았으므로 "효과가 0 이다" 를 주장하지 않는다
CI 가 좁게 0 을 포함하면 "실용적으로 큰 이득이 관측되지 않았다" 로 쓴다
```

사용법:
    python scripts/make_report.py --out docs/results_stage2.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from rl_newton.benchmark.metrics import RunSummary, compare_paired_delta, median_of
from rl_newton.benchmark.store import ResultStore

_SEED_SUFFIX = re.compile(r"_seed\d+$")


@dataclass(frozen=True, slots=True)
class Source:
    """리포트에 넣을 raw 결과 하나."""

    label: str
    raw: Path
    note: str


def spec_of(run: RunSummary) -> str:
    return _SEED_SUFFIX.sub("", run.task_instance_id)


def median(values: list[float]) -> float:
    """프로젝트 단일 규약을 쓴다 (`metrics.median_of`).

    자체 구현을 두면 보고 도구와 게이트 통계가 갈린다. 실제로 그런 적이 있다.
    """
    return median_of(values)


def fmt_p(value: float) -> str:
    """`p=0.0000` 을 금지한다 (D26)."""
    if not math.isfinite(value):
        return "n/a"
    if value < 1.0e-4:
        return "<0.0001"
    return f"{value:.4f}"


def load(path: Path) -> tuple[dict[str, list[RunSummary]], dict[str, str]]:
    by_controller: dict[str, list[RunSummary]] = {}
    for record in ResultStore(path):
        if record.status != "completed" or record.summary is None:
            continue
        by_controller.setdefault(record.key.controller, []).append(record.summary)

    alias: dict[str, str] = {}
    summary = path.parents[1] / "summaries" / f"{path.stem}.json"
    if summary.exists():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        for family, manifest in (payload.get("selections") or {}).items():
            label = manifest.get("selected_label")
            if label and label in by_controller:
                alias["best_static" if family == "static" else f"best_{family}"] = label
    return by_controller, alias


LADDER = (
    "best_static",
    "best_open_loop",
    "heuristic",
    "onestep_narrow",
    "onestep_absolute",
    "committed_Q4_narrow",
    "shrinking_Q4_narrow",
)

DELTAS = (
    ("A2", "best_static", "shrinking_Q4_narrow"),
    ("C2", "onestep_narrow", "shrinking_Q4_narrow"),
    ("C3", "committed_Q4_narrow", "shrinking_Q4_narrow"),
    ("open_loop", "best_static", "best_open_loop"),
    # 사다리 각 단계를 튜닝 상수와 직접 비교한다. **차이를 손으로 빼서 만들면 안 된다.**
    # 쌍별 차이의 median 은 선형이 아니므로 `A2 − C2` 가 `onestep − static` 이 아니다.
    ("ladder", "best_static", "onestep_narrow"),
    ("ladder", "best_static", "committed_Q4_narrow"),
    # 게이트 B 와 heuristic baseline. 원고 Table 2 가 이 값을 인용한다.
    ("B", "onestep_narrow", "onestep_absolute"),
    ("B_wide", "onestep_narrow", "onestep_wide"),
    ("heuristic", "best_static", "heuristic"),
)


def section_absolute(
    out: list[str], by_controller: dict[str, list[RunSummary]], alias: dict[str, str]
) -> None:
    specs = sorted(
        {spec_of(r) for runs in by_controller.values() for r in runs}
    )
    out.append("")
    out.append("절대 median logΔ (nat, 높을수록 좋다)")
    out.append("")
    out.append("| controller | " + " | ".join(specs) + " |")
    out.append("|---|" + "---|" * len(specs))
    for name in LADDER:
        runs = by_controller.get(alias.get(name, name))
        if not runs:
            continue
        cells = []
        for spec in specs:
            vals = [r.log_improvement for r in runs if spec_of(r) == spec]
            cells.append(f"{median(vals):.3f} (n={len(vals)})" if vals else "-")
        shown = name if alias.get(name, name) == name else f"{name}"
        out.append(f"| `{shown}` | " + " | ".join(cells) + " |")


def section_deltas(
    out: list[str], by_controller: dict[str, list[RunSummary]], alias: dict[str, str]
) -> None:
    specs = sorted(
        {spec_of(r) for runs in by_controller.values() for r in runs}
    )
    out.append("")
    out.append("paired delta (nat, 양수면 treatment 가 좋다)")
    out.append("")
    out.append("| 비교 | 범위 | median | 95% CI | p | n | 양수 |")
    out.append("|---|---|---|---|---|---|---|")
    for gate, base, treat in DELTAS:
        rb = by_controller.get(alias.get(base, base))
        rt = by_controller.get(alias.get(treat, treat))
        if not rb or not rt:
            continue
        scopes = ["ALL", *specs] if len(specs) > 1 else ["ALL"]
        for scope in scopes:
            b = rb if scope == "ALL" else [r for r in rb if spec_of(r) == scope]
            t = rt if scope == "ALL" else [r for r in rt if spec_of(r) == scope]
            if not b or not t:
                continue
            d = compare_paired_delta(b, t, metric="log_improvement")
            bmap = {(r.task_instance_id, r.seed): r for r in b}
            tmap = {(r.task_instance_id, r.seed): r for r in t}
            each = [
                tmap[k].log_improvement - bmap[k].log_improvement
                for k in set(bmap) & set(tmap)
                if math.isfinite(tmap[k].log_improvement)
                and math.isfinite(bmap[k].log_improvement)
            ]
            n_pos = sum(1 for v in each if v > 0.0)
            out.append(
                f"| {gate} `{treat}` − `{base}` | {scope} | "
                f"{d.median_delta:+.3f} | "
                f"[{d.delta_ci[0]:+.3f}, {d.delta_ci[1]:+.3f}] | "
                f"{fmt_p(d.p_value)} | {d.n_valid} | {n_pos}/{len(each)} |"
            )


def section_cost(
    out: list[str], by_controller: dict[str, list[RunSummary]], alias: dict[str, str]
) -> None:
    out.append("")
    out.append("탐색 비용과 거절률 (전체 인스턴스 median)")
    out.append("")
    out.append("| controller | decision-search GE | object GE | 거절률 |")
    out.append("|---|---|---|---|")
    for name in LADDER:
        runs = by_controller.get(alias.get(name, name))
        if not runs:
            continue
        out.append(
            f"| `{name}` | {median([r.search_cost_ge for r in runs]):,.0f} | "
            f"{median([r.total_cost_ge for r in runs]):.1f} | "
            f"{median([r.rejection_rate for r in runs]):.2f} |"
        )


def section_cost_by_spec(
    out: list[str], by_controller: dict[str, list[RunSummary]], alias: dict[str, str]
) -> None:
    """spec 별 탐색 비용과 거절률.

    pooled median 만 두면 원고에서 regime 별 비용비를 손으로 만들게 된다.
    실제로 그런 오류가 있었다. **비율은 여기 값을 쓴다.**
    """
    specs = sorted({spec_of(r) for runs in by_controller.values() for r in runs})
    if len(specs) < 2:
        return
    out.append("")
    out.append("spec 별 탐색 비용 (median decision-search GE) 과 거절률")
    out.append("")
    out.append("| controller | " + " | ".join(f"{s} GE / 거절률" for s in specs) + " |")
    out.append("|---|" + "---|" * len(specs))
    for name in LADDER:
        runs = by_controller.get(alias.get(name, name))
        if not runs:
            continue
        cells = []
        for spec in specs:
            sub = [r for r in runs if spec_of(r) == spec]
            if not sub:
                cells.append("-")
                continue
            cells.append(
                f"{median([r.search_cost_ge for r in sub]):,.0f} / "
                f"{median([r.rejection_rate for r in sub]):.2f}"
            )
        out.append(f"| `{name}` | " + " | ".join(cells) + " |")

    out.append("")
    out.append("spec 별 `shrinking` 대비 `onestep` 탐색 비용 배수 (planner / onestep)")
    out.append("")
    out.append("| spec | onestep GE | shrinking GE | 배수 |")
    out.append("|---|---|---|---|")
    one = by_controller.get(alias.get("onestep_narrow", "onestep_narrow"))
    shr = by_controller.get(alias.get("shrinking_Q4_narrow", "shrinking_Q4_narrow"))
    if not one or not shr:
        return
    for spec in specs:
        o = [r.search_cost_ge for r in one if spec_of(r) == spec]
        s = [r.search_cost_ge for r in shr if spec_of(r) == spec]
        if not o or not s:
            continue
        mo, ms = median(o), median(s)
        ratio = f"{ms / mo:,.1f}x" if mo > 0.0 else "n/a"
        out.append(f"| {spec} | {mo:,.0f} | {ms:,.0f} | {ratio} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--out", type=Path, default=Path("docs/results_stage2.md"))
    args = parser.parse_args()

    sources = [
        Source(
            "beam 4 dev pilot",
            args.raw_dir / "headroom_pilot_step_size_fixed_b4_9d725689.jsonl",
            "원래 dev subset 3 spec x 3 seed. 2개 spec 이 포화됐다 (D19)",
        ),
        Source(
            "beam 8 challenge dev",
            args.raw_dir / "headroom_challenge_step_size_fixed_b8_fed9aebd.jsonl",
            "challenge 4 spec x seeds 2/3/4. 설정 선택에 사용 (D21/D22)",
        ),
        Source(
            "beam 8 held-out confirmatory",
            args.raw_dir / "headroom_challenge-heldout_step_size_fixed_b8_9a18b6e9.jsonl",
            "같은 4 spec x seeds 100~109. 사전 고정 설정, 최종 효과 추정 (D26)",
        ),
        Source(
            "nonlinear diagnostic",
            args.raw_dir / "headroom_nonlinear-diagnostic_step_size_fixed_b8_2a09bd45.jsonl",
            "rosen_d5. 국소최소점 cap + seed 복제로 사용 불가 (D23)",
        ),
    ]
    sources += [
        Source(f"micro-neural {p.stem.split('_b8_')[-1]}", p, "D27/D28/D29")
        for p in sorted(args.raw_dir.glob("headroom_micro-neural_*.jsonl"))
    ]

    out: list[str] = [
        "# Stage 2 결과표",
        "",
        "`scripts/make_report.py` 가 raw 결과에서 생성한다. **손으로 수정하지 않는다.**",
        "",
        "## 통계 표기 규칙 (프로토콜 D26)",
        "",
        "```text",
        "p 를 0.0000 으로 쓰지 않는다. 부트스트랩/순열 기반이므로 p<0.0001 로 쓴다",
        "equivalence margin 을 사전 등록하지 않았으므로 \"효과가 0 이다\" 를 주장하지 않는다",
        "CI 가 좁게 0 을 포함하면 \"실용적으로 큰 이득이 관측되지 않았다\" 로 쓴다",
        "```",
    ]

    for source in sources:
        if not source.raw.exists():
            continue
        by_controller, alias = load(source.raw)
        if not by_controller:
            continue
        n_runs = sum(len(v) for v in by_controller.values())
        out.append("")
        out.append(f"## {source.label}")
        out.append("")
        out.append(f"{source.note}")
        out.append("")
        out.append("```text")
        out.append(f"raw        {source.raw.name}")
        out.append(f"완료 run   {n_runs}")
        for family, label in sorted(alias.items()):
            out.append(f"{family:<10} -> {label}")
        out.append("```")
        section_absolute(out, by_controller, alias)
        section_deltas(out, by_controller, alias)
        section_cost(out, by_controller, alias)
        section_cost_by_spec(out, by_controller, alias)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"저장: {args.out}  ({len(out)} 줄)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
