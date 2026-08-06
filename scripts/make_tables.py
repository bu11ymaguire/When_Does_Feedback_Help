"""제출용 LaTeX 표를 raw 결과에서 생성한다.

`docs/results_stage2.md` 와 **같은 로더·같은 median 규약**을 쓴다. `make_report` 를
import 하므로 두 산출물이 갈릴 수 없다. 규약이 갈렸던 사고가 실제로 있었다 (E10).

생성 파일은 `tabular` 조각만 담는다. `\\begin{table}`, caption, label 은
`paper/sections/*.tex` 가 제공한다. 그래야 문장은 사람이 고치고 숫자는 스크립트만
쓴다는 경계가 유지된다.

```text
paper/tables/heldout_absolute.tex      Table 1  절대 median J_E
paper/tables/heldout_paired.tex        Table 2  게이트별 쌍별 차이
paper/tables/heldout_ladder.tex        Table 3  사다리 각 단계 대 튜닝 상수 + 직접 증분
paper/tables/heldout_conditioning.tex  Table 4  kappa 별 A2
paper/tables/heldout_cost.tex          Table 5  탐색 비용
paper/tables/micro_absolute.tex        Table 6  micro-neural 절대값
paper/tables/micro_paired.tex          Table 7  micro-neural 쌍별 차이
paper/tables/acceptance_delta.tex      Table 8  수락 기준 ablation
paper/tables/acceptance_rejection.tex  Table 9  거절률
```

사용법:
    python scripts/make_tables.py --out-dir paper/tables
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

# **이 저장소의 src 를 우선한다.** 아래 project import 보다 먼저 와야 한다.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from make_report import fmt_p, load, median, spec_of  # noqa: E402

from rl_newton.benchmark.metrics import RunSummary, compare_paired_delta  # noqa: E402
from rl_newton.reporting import load_public_grouped  # noqa: E402

RAW_HELDOUT = "headroom_challenge-heldout_step_size_fixed_b8_9a18b6e9.jsonl"
RAW_MICRO_CONTROL = "headroom_micro-neural_step_size_fixed_b8_0bec1125.jsonl"
RAW_MICRO_FIXED_EVAL = "headroom_micro-neural_step_size_fixed_b8_9f3194be.jsonl"

# 공개 저장소에는 raw 가 없다. `(공개 파일, acceptance_rule)` 로 대체한다.
PUBLIC_EQUIVALENT = {
    RAW_HELDOUT: ("heldout_quadratic.csv", None),
    RAW_MICRO_CONTROL: ("micro_neural.csv", "control"),
    RAW_MICRO_FIXED_EVAL: ("micro_neural.csv", "fixed_eval"),
}

# 원고의 컨트롤러 표기. LaTeX 에서는 `\texttt` 로 감싼다.
LADDER_ROWS = (
    ("best_static", "best\\_static"),
    ("best_open_loop", "best\\_open\\_loop"),
    ("heuristic", "heuristic"),
    ("onestep_narrow", "onestep\\_narrow"),
    ("onestep_absolute", "onestep\\_absolute"),
    ("committed_Q4_narrow", "committed\\_Q4\\_narrow"),
    ("shrinking_Q4_narrow", "shrinking\\_Q4\\_narrow"),
)

_KAPPA = re.compile(r"quad_ill_conditioned_d(\d+)_k1e\+0?(\d+)")
_MICRO = re.compile(r"mlp_d\d+_h\d+_c\d+_n\d+_(fb|cs(\d+))")
# micro-neural regime 표시 순서. 원고 순서를 따른다 (결정론 -> 잡음 큰 쪽).
MICRO_ORDER = ("fb", "cs128", "cs64")


def pretty_spec(spec: str) -> str:
    m = _KAPPA.match(spec)
    if m:
        return f"$\\kappa = 10^{{{int(m.group(2))}}}$"
    m = _MICRO.match(spec)
    if m:
        return "R1 full-batch" if m.group(1) == "fb" else f"R2 batch {m.group(2)}"
    return spec.replace("_", "\\_")


def micro_key(spec: str) -> int:
    m = _MICRO.match(spec)
    tag = m.group(1) if m else spec
    return MICRO_ORDER.index(tag) if tag in MICRO_ORDER else len(MICRO_ORDER)


def specs_of(by_controller: dict[str, list[RunSummary]], *, micro: bool) -> list[str]:
    found = {spec_of(r) for runs in by_controller.values() for r in runs}
    return sorted(found, key=micro_key) if micro else sorted(found)


def resolve(
    by_controller: dict[str, list[RunSummary]], alias: dict[str, str], name: str
) -> list[RunSummary]:
    return by_controller.get(alias.get(name, name), [])


def paired(
    base: list[RunSummary], treat: list[RunSummary]
) -> tuple[float, tuple[float, float], float, int, int]:
    """쌍별 차이. median / CI / p / 쌍 수 / 양수 개수."""
    d = compare_paired_delta(base, treat, metric="log_improvement")
    bmap = {(r.task_instance_id, r.seed): r for r in base}
    tmap = {(r.task_instance_id, r.seed): r for r in treat}
    each = [
        tmap[k].log_improvement - bmap[k].log_improvement
        for k in set(bmap) & set(tmap)
        if math.isfinite(tmap[k].log_improvement) and math.isfinite(bmap[k].log_improvement)
    ]
    n_pos = sum(1 for v in each if v > 0.0)
    return d.median_delta, d.delta_ci, d.p_value, len(each), n_pos


def header(name: str, raw: str | tuple[str, ...]) -> list[str]:
    sources = (raw,) if isinstance(raw, str) else raw
    lines = [
        "% 이 파일은 scripts/make_tables.py 가 생성한다. **손으로 수정하지 않는다.**",
        f"% 표: {name}",
    ]
    lines += [f"% raw: {s}" for s in sources]
    lines.append("% raw 의 SHA-256 은 paper/evidence_map.md 에 있다.")
    lines.append(
        "% 공개 저장소에는 raw 가 없다. results/public/*.csv 에서 --public-dir 로 만들면"
    )
    lines.append("% **고정된 환경에서** 이 파일과 바이트 단위로 같아진다.")
    lines.append("% 판본이 다른 환경에서의 비트 수준 일치는 주장하지 않는다.")
    return lines


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {path}  ({len(lines)} 줄)")


# --- 표 생성 ---------------------------------------------------------------


def table_absolute(
    by_controller: dict[str, list[RunSummary]],
    alias: dict[str, str],
    *,
    micro: bool,
    raw: str,
    label: str,
) -> list[str]:
    specs = specs_of(by_controller, micro=micro)
    out = header(label, raw)
    out.append("\\begin{tabular}{l" + "r" * len(specs) + "}")
    out.append("\\toprule")
    out.append(
        "Controller & " + " & ".join(pretty_spec(s) for s in specs) + " \\\\"
    )
    out.append("\\midrule")
    for name, shown in LADDER_ROWS:
        runs = resolve(by_controller, alias, name)
        if not runs:
            continue
        cells = []
        for spec in specs:
            vals = [r.log_improvement for r in runs if spec_of(r) == spec]
            cells.append(f"{median(vals):.3f}" if vals else "--")
        out.append(f"\\texttt{{{shown}}} & " + " & ".join(cells) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return out


def _delta_row(
    gate: str,
    text: str,
    base: list[RunSummary],
    treat: list[RunSummary],
    *,
    with_p: bool = True,
) -> str:
    m, ci, p, n, pos = paired(base, treat)
    cells = [gate, text, f"${m:+.3f}$", f"$[{ci[0]:+.3f}, {ci[1]:+.3f}]$"]
    if with_p:
        cells.append(f"${fmt_p(p)}$".replace("<", "{<}"))
    cells.append(f"{pos}/{n}")
    return " & ".join(cells) + " \\\\"


def table_heldout_paired(
    by_controller: dict[str, list[RunSummary]], alias: dict[str, str], raw: str
) -> list[str]:
    g = lambda name: resolve(by_controller, alias, name)  # noqa: E731
    out = header("held-out paired delta", raw)
    out.append("\\begin{tabular}{llrrrr}")
    out.append("\\toprule")
    out.append("Gate & Comparison & Median & 95\\% CI & $p$ & Positive \\\\")
    out.append("\\midrule")
    out.append(
        _delta_row(
            "A2", "shrinking $-$ best\\_static", g("best_static"), g("shrinking_Q4_narrow")
        )
    )
    out.append(
        _delta_row(
            "C2", "shrinking $-$ onestep", g("onestep_narrow"), g("shrinking_Q4_narrow")
        )
    )
    out.append(
        _delta_row(
            "C3", "shrinking $-$ committed", g("committed_Q4_narrow"), g("shrinking_Q4_narrow")
        )
    )
    out.append(
        _delta_row(
            "B",
            "onestep\\_absolute $-$ onestep\\_narrow",
            g("onestep_narrow"),
            g("onestep_absolute"),
        )
    )
    if g("onestep_wide"):
        out.append(
            _delta_row(
                "B",
                "onestep\\_wide $-$ onestep\\_narrow",
                g("onestep_narrow"),
                g("onestep_wide"),
            )
        )
    out.append("\\midrule")
    out.append(
        _delta_row(
            "--", "open\\_loop $-$ best\\_static", g("best_static"), g("best_open_loop")
        )
    )
    out.append(
        _delta_row("--", "heuristic $-$ best\\_static", g("best_static"), g("heuristic"))
    )
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return out


def table_heldout_ladder(
    by_controller: dict[str, list[RunSummary]], alias: dict[str, str], raw: str
) -> list[str]:
    """사다리 각 단계 대 튜닝 상수, 그리고 **직접 측정한** 증분.

    이 표의 값을 서로 빼면 안 된다. 쌍별 median 은 선형이 아니다 (C26).
    """
    g = lambda name: resolve(by_controller, alias, name)  # noqa: E731
    out = header("ladder vs tuned constant (직접 측정만)", raw)
    out.append("\\begin{tabular}{llrrrr}")
    out.append("\\toprule")
    out.append("& Comparison & Median & 95\\% CI & $p$ & Positive \\\\")
    out.append("\\midrule")
    out.append("\\multicolumn{6}{l}{\\emph{Relative to the tuned constant setting}} \\\\")
    for shown, name in (
        ("open\\_loop", "best_open_loop"),
        ("onestep\\_narrow", "onestep_narrow"),
        ("committed\\_Q4\\_narrow", "committed_Q4_narrow"),
        ("shrinking\\_Q4\\_narrow", "shrinking_Q4_narrow"),
    ):
        out.append(_delta_row("", shown, g("best_static"), g(name)))
    out.append("\\midrule")
    out.append("\\multicolumn{6}{l}{\\emph{Directly measured increments}} \\\\")
    out.append(
        _delta_row(
            "C2", "shrinking $-$ onestep", g("onestep_narrow"), g("shrinking_Q4_narrow")
        )
    )
    out.append(
        _delta_row(
            "C3", "shrinking $-$ committed", g("committed_Q4_narrow"), g("shrinking_Q4_narrow")
        )
    )
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return out


def table_conditioning(
    by_controller: dict[str, list[RunSummary]], alias: dict[str, str], raw: str
) -> list[str]:
    specs = specs_of(by_controller, micro=False)
    base = resolve(by_controller, alias, "best_static")
    treat = resolve(by_controller, alias, "shrinking_Q4_narrow")
    out = header("A2 by condition number", raw)
    out.append("\\begin{tabular}{lrrr}")
    out.append("\\toprule")
    out.append("Spec & Median & 95\\% CI & Positive \\\\")
    out.append("\\midrule")
    for spec in specs:
        b = [r for r in base if spec_of(r) == spec]
        t = [r for r in treat if spec_of(r) == spec]
        if not b or not t:
            continue
        m, ci, _p, n, pos = paired(b, t)
        out.append(
            f"{pretty_spec(spec)} & ${m:+.3f}$ & "
            f"$[{ci[0]:+.3f}, {ci[1]:+.3f}]$ & {pos}/{n} \\\\"
        )
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return out


def table_cost(
    by_controller: dict[str, list[RunSummary]],
    alias: dict[str, str],
    raw: str,
    *,
    budget_ge: float,
) -> list[str]:
    out = header("decision-search cost", raw)
    out.append("\\begin{tabular}{lrr}")
    out.append("\\toprule")
    out.append("Controller & Decision-search GE & Relative to budget \\\\")
    out.append("\\midrule")
    for name, shown in LADDER_ROWS:
        runs = resolve(by_controller, alias, name)
        if not runs:
            continue
        ge = median([r.search_cost_ge for r in runs])
        ratio = "--" if ge <= 0.0 else f"${ge / budget_ge:,.1f}\\times$"
        out.append(f"\\texttt{{{shown}}} & {ge:,.0f} & {ratio} \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return out


def table_micro_paired(
    by_controller: dict[str, list[RunSummary]], alias: dict[str, str], raw: str
) -> list[str]:
    """regime 당 n=3 이므로 **CI 와 p 를 쓰지 않는다** (§10.4, L2)."""
    specs = specs_of(by_controller, micro=True)
    g = lambda name: resolve(by_controller, alias, name)  # noqa: E731
    comparisons = (
        ("shrinking $-$ best\\_static", "best_static", "shrinking_Q4_narrow"),
        ("shrinking $-$ onestep", "onestep_narrow", "shrinking_Q4_narrow"),
        ("shrinking $-$ committed", "committed_Q4_narrow", "shrinking_Q4_narrow"),
        ("committed $-$ onestep", "onestep_narrow", "committed_Q4_narrow"),
    )
    out = header("micro-neural paired delta (exploratory, n=3 per regime)", raw)
    out.append("\\begin{tabular}{l" + "r" * len(specs) + "}")
    out.append("\\toprule")
    out.append("Comparison & " + " & ".join(pretty_spec(s) for s in specs) + " \\\\")
    out.append("\\midrule")
    for text, base_name, treat_name in comparisons:
        cells = []
        for spec in specs:
            b = [r for r in g(base_name) if spec_of(r) == spec]
            t = [r for r in g(treat_name) if spec_of(r) == spec]
            if not b or not t:
                cells.append("--")
                continue
            m, _ci, _p, n, pos = paired(b, t)
            cells.append(f"${m:+.3f}$ ({pos}/{n})")
        out.append(f"{text} & " + " & ".join(cells) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return out


def table_acceptance_delta(
    control: tuple[dict[str, list[RunSummary]], dict[str, str]],
    fixed_eval: tuple[dict[str, list[RunSummary]], dict[str, str]],
    raw: tuple[str, str],
) -> list[str]:
    gates = (
        ("A2", "best_static", "shrinking_Q4_narrow"),
        ("C2", "onestep_narrow", "shrinking_Q4_narrow"),
        ("C3", "committed_Q4_narrow", "shrinking_Q4_narrow"),
    )
    specs = specs_of(control[0], micro=True)
    out = header("acceptance criterion ablation (exploratory, n=3 per regime)", raw)
    out.append("\\begin{tabular}{lllrr}")
    out.append("\\toprule")
    out.append("Gate & Regime & & Control & Fixed evaluation \\\\")
    out.append("\\midrule")
    for gate, base_name, treat_name in gates:
        for i, spec in enumerate(specs):
            cells = []
            for by_controller, alias in (control, fixed_eval):
                b = [
                    r
                    for r in resolve(by_controller, alias, base_name)
                    if spec_of(r) == spec
                ]
                t = [
                    r
                    for r in resolve(by_controller, alias, treat_name)
                    if spec_of(r) == spec
                ]
                if not b or not t:
                    cells.append("--")
                    continue
                m, _ci, _p, n, pos = paired(b, t)
                cells.append(f"${m:+.3f}$ ({pos}/{n})")
            gate_cell = gate if i == 0 else ""
            out.append(
                f"{gate_cell} & {pretty_spec(spec)} & & " + " & ".join(cells) + " \\\\"
            )
        if gate != gates[-1][0]:
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return out


def table_acceptance_rejection(
    control: tuple[dict[str, list[RunSummary]], dict[str, str]],
    fixed_eval: tuple[dict[str, list[RunSummary]], dict[str, str]],
    raw: tuple[str, str],
) -> list[str]:
    """거절률. 대체 기준이 **완화가 아니라 엄격화**였다는 근거다 (C14)."""
    specs = specs_of(control[0], micro=True)
    rows = ("committed_Q4_narrow", "shrinking_Q4_narrow", "onestep_narrow")
    shown = {
        "committed_Q4_narrow": "committed",
        "shrinking_Q4_narrow": "shrinking",
        "onestep_narrow": "onestep",
    }
    out = header("rejection rate under two acceptance criteria", raw)
    out.append("\\begin{tabular}{llrr}")
    out.append("\\toprule")
    out.append("Controller & Regime & Control & Fixed evaluation \\\\")
    out.append("\\midrule")
    for name in rows:
        for i, spec in enumerate(specs):
            cells = []
            for by_controller, alias in (control, fixed_eval):
                runs = [
                    r for r in resolve(by_controller, alias, name) if spec_of(r) == spec
                ]
                cells.append(
                    f"{median([r.rejection_rate for r in runs]):.2f}" if runs else "--"
                )
            label = f"\\texttt{{{shown[name]}}}" if i == 0 else ""
            out.append(f"{label} & {pretty_spec(spec)} & " + " & ".join(cells) + " \\\\")
        if name != rows[-1]:
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/tables"))
    parser.add_argument(
        "--budget-ge", type=float, default=150.0, help="배포 예산 GE. 비율 계산에만 쓴다"
    )
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=None,
        help="공개 CSV 에서 표를 만든다. raw 가 없는 공개 저장소에서 쓴다",
    )
    args = parser.parse_args()

    def source(raw_name: str) -> tuple[dict[str, list[RunSummary]], dict[str, str]]:
        """raw 또는 공개 CSV 에서 읽는다. 표 코드는 출처를 모른다."""
        if args.public_dir is None:
            return load(args.raw_dir / raw_name)
        name, rule = PUBLIC_EQUIVALENT[raw_name]
        return load_public_grouped(args.public_dir / name, acceptance_rule=rule)

    names = (RAW_HELDOUT, RAW_MICRO_CONTROL, RAW_MICRO_FIXED_EVAL)
    if args.public_dir is None:
        missing = [args.raw_dir / n for n in names if not (args.raw_dir / n).exists()]
        hint = "raw 결과가 없으면 표를 만들 수 없다. docs/reproduce.md 를 따른다."
    else:
        wanted = {args.public_dir / PUBLIC_EQUIVALENT[n][0] for n in names}
        missing = sorted(p for p in wanted if not p.exists())
        hint = "공개 CSV 가 없다. scripts/make_public_results.py 로 만든다."
    if missing:
        for p in missing:
            print(f"없음: {p}")
        print(hint)
        return 2

    print(f"소스: {'공개 CSV ' + str(args.public_dir) if args.public_dir else '비공개 raw'}")
    print("생성:")
    hb, ha = source(RAW_HELDOUT)
    write(
        args.out_dir / "heldout_absolute.tex",
        table_absolute(hb, ha, micro=False, raw=RAW_HELDOUT, label="held-out absolute"),
    )
    write(args.out_dir / "heldout_paired.tex", table_heldout_paired(hb, ha, RAW_HELDOUT))
    write(args.out_dir / "heldout_ladder.tex", table_heldout_ladder(hb, ha, RAW_HELDOUT))
    write(args.out_dir / "heldout_conditioning.tex", table_conditioning(hb, ha, RAW_HELDOUT))
    write(
        args.out_dir / "heldout_cost.tex",
        table_cost(hb, ha, RAW_HELDOUT, budget_ge=args.budget_ge),
    )

    cb, ca = source(RAW_MICRO_CONTROL)
    fb_, fa = source(RAW_MICRO_FIXED_EVAL)
    write(
        args.out_dir / "micro_absolute.tex",
        table_absolute(
            cb, ca, micro=True, raw=RAW_MICRO_CONTROL, label="micro-neural absolute"
        ),
    )
    write(args.out_dir / "micro_paired.tex", table_micro_paired(cb, ca, RAW_MICRO_CONTROL))
    raws = (RAW_MICRO_CONTROL, RAW_MICRO_FIXED_EVAL)
    write(
        args.out_dir / "acceptance_delta.tex",
        table_acceptance_delta((cb, ca), (fb_, fa), raws),
    )
    write(
        args.out_dir / "acceptance_rejection.tex",
        table_acceptance_rejection((cb, ca), (fb_, fa), raws),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
