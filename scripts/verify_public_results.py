"""공개 CSV 가 원고의 통계를 **정확히** 재현하는지 검사한다.

이 검사가 이 재현 패키지의 존재 이유다. 공개 데이터로 논문의 수치가 다시 나오지
않으면 패키지는 값이 없다.

방법
----
같은 통계를 두 경로로 계산해 **비트 단위로** 비교한다.

```text
경로 A  비공개 results/raw/*.jsonl  ->  ResultStore  ->  compare_paired_delta
경로 B  공개  results/public/*.csv  ->  load_public_results  ->  compare_paired_delta
```

두 경로가 같은 `compare_paired_delta` 를 쓰므로, 차이가 나면 공개 CSV 가 정보를
잃었다는 뜻이다. 허용 오차를 두지 않는다. `repr(float)` 로 비교한다.

`--raw-dir` 이 없으면 경로 A 를 만들 수 없으므로 건너뛴다. 그때는 공개 데이터만으로
값을 출력해 눈으로 원고와 대조할 수 있게 한다.

사용법
------
    python scripts/verify_public_results.py --raw-dir <비공개 raw 경로>
    python scripts/verify_public_results.py                # 공개 데이터만 출력
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

# **이 저장소의 src 를 우선한다.** 재현 패키지는 함께 배포된 코드로 계산해야 한다.
# 전역에 다른 판이 설치돼 있으면 조용히 그것을 쓰게 되고, 그러면 "공개 데이터로
# 재현했다" 는 말이 무엇을 재현한 것인지 불분명해진다.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rl_newton.benchmark.metrics import (  # noqa: E402
    RunSummary,
    compare_paired_delta,
    median_of,
)
from rl_newton.benchmark.store import ResultStore  # noqa: E402
from rl_newton.reporting import (  # noqa: E402
    load_public_results,
    public_roles,
    split_by,
)

_SEED_SUFFIX = re.compile(r"_seed\d+$")

HELD_OUT_RAW = "headroom_challenge-heldout_step_size_fixed_b8_9a18b6e9.jsonl"
MICRO_CONTROL_RAW = "headroom_micro-neural_step_size_fixed_b8_0bec1125.jsonl"
MICRO_FIXED_RAW = "headroom_micro-neural_step_size_fixed_b8_9f3194be.jsonl"

# 원고가 인용하는 held-out 비교. (라벨, baseline, treatment)
HELD_OUT_PAIRS = (
    ("A2  shrinking - static", "best_static", "shrinking_Q4_narrow"),
    ("C2  shrinking - onestep", "onestep_narrow", "shrinking_Q4_narrow"),
    ("C3  shrinking - committed", "committed_Q4_narrow", "shrinking_Q4_narrow"),
    ("B   absolute - narrow", "onestep_narrow", "onestep_absolute"),
    ("B   wide - narrow", "onestep_narrow", "onestep_wide"),
    ("    open_loop - static", "best_static", "best_open_loop"),
    ("    heuristic - static", "best_static", "heuristic"),
    ("    onestep - static", "best_static", "onestep_narrow"),
    ("    committed - static", "best_static", "committed_Q4_narrow"),
)

MICRO_PAIRS = (
    ("A2  shrinking - static", "best_static", "shrinking_Q4_narrow"),
    ("C2  shrinking - onestep", "onestep_narrow", "shrinking_Q4_narrow"),
    ("C3  shrinking - committed", "committed_Q4_narrow", "shrinking_Q4_narrow"),
    ("    committed - onestep", "onestep_narrow", "committed_Q4_narrow"),
)

LADDER = (
    "best_static",
    "best_open_loop",
    "heuristic",
    "onestep_narrow",
    "onestep_absolute",
    "committed_Q4_narrow",
    "shrinking_Q4_narrow",
)


def spec_of(run: RunSummary) -> str:
    return _SEED_SUFFIX.sub("", run.task_instance_id)


def load_raw(path: Path) -> list[RunSummary]:
    out: list[RunSummary] = []
    for record in ResultStore(path):
        if record.status == "completed" and record.summary is not None:
            out.append(record.summary)
    return out


def alias_of(path: Path) -> dict[str, str]:
    """summary JSON 의 selection 을 읽어 `best_static` 등을 실제 라벨로 옮긴다."""
    import json

    out: dict[str, str] = {}
    summary = path.parents[1] / "summaries" / f"{path.stem}.json"
    if summary.exists():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        for family, manifest in (payload.get("selections") or {}).items():
            label = manifest.get("selected_label")
            if label:
                out["best_static" if family == "static" else f"best_{family}"] = label
    return out


def resolve(runs: Sequence[RunSummary], alias: dict[str, str], name: str) -> list[RunSummary]:
    target = alias.get(name, name)
    return [r for r in runs if r.controller == target]


def stat_rows(
    runs: Sequence[RunSummary],
    alias: dict[str, str],
    pairs: Sequence[tuple[str, str, str]],
    *,
    by_spec: bool,
) -> dict[str, str]:
    """비교 라벨 -> 통계 문자열. 두 경로에서 같은 키가 나와야 한다."""
    out: dict[str, str] = {}
    specs = sorted({spec_of(r) for r in runs})
    scopes: list[str | None] = [None] + (list(specs) if by_spec else [])
    for label, base_name, treat_name in pairs:
        base_all = resolve(runs, alias, base_name)
        treat_all = resolve(runs, alias, treat_name)
        if not base_all or not treat_all:
            continue
        for scope in scopes:
            base = base_all if scope is None else [r for r in base_all if spec_of(r) == scope]
            treat = treat_all if scope is None else [r for r in treat_all if spec_of(r) == scope]
            if not base or not treat:
                continue
            d = compare_paired_delta(base, treat, metric="log_improvement")
            key = f"paired | {label} | {scope or 'ALL'}"
            out[key] = (
                f"median={d.median_delta!r} ci=({d.delta_ci[0]!r},{d.delta_ci[1]!r}) "
                f"p={d.p_value!r} n={d.n_valid}"
            )
    # 절대 median 과 비용도 대조한다. 표에 실리는 값이다.
    for name in LADDER:
        got = resolve(runs, alias, name)
        if not got:
            continue
        for scope in scopes:
            sub = got if scope is None else [r for r in got if spec_of(r) == scope]
            if not sub:
                continue
            tag = f"{name} | {scope or 'ALL'}"
            out[f"absolute | {tag}"] = repr(median_of([r.log_improvement for r in sub]))
            out[f"search_ge | {tag}"] = repr(median_of([r.search_cost_ge for r in sub]))
            out[f"reject | {tag}"] = repr(median_of([r.rejection_rate for r in sub]))
    return out


def compare(label: str, a: dict[str, str], b: dict[str, str]) -> list[str]:
    """두 경로의 통계를 비교하고 불일치를 반환한다."""
    problems: list[str] = []
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    for key in only_a:
        problems.append(f"{label}: raw 에만 있다  {key}")
    for key in only_b:
        problems.append(f"{label}: public 에만 있다  {key}")
    for key in sorted(set(a) & set(b)):
        if a[key] != b[key]:
            problems.append(f"{label}: 불일치  {key}\n    raw    {a[key]}\n    public {b[key]}")
    print(f"  {label}: 항목 {len(set(a) & set(b))}개 대조, 불일치 {len(problems)}건")
    return problems


def show(label: str, rows: dict[str, str]) -> None:
    print()
    print(f"--- {label} ---")
    for key in sorted(rows):
        if key.startswith("paired | ") and key.endswith("| ALL"):
            print(f"  {key[9:]}  {rows[key]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, default=Path("results/public"))
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="비공개 raw 경로. 주면 두 경로를 비교한다",
    )
    args = parser.parse_args()

    heldout_csv = args.public_dir / "heldout_quadratic.csv"
    micro_csv = args.public_dir / "micro_neural.csv"
    for path in (heldout_csv, micro_csv):
        if not path.exists():
            print(f"없음: {path}")
            print("먼저 scripts/make_public_results.py 를 실행한다.")
            return 2

    print("=" * 88)
    print("공개 결과 검증")
    print("=" * 88)

    # 역할 대응은 **공개 CSV 자체**에서 읽는다. summary JSON 에 의존하지 않는다.
    public_sets: dict[
        str, tuple[list[RunSummary], dict[str, str], Sequence[tuple[str, str, str]], bool]
    ] = {
        "held-out quadratic": (
            load_public_results(heldout_csv),
            public_roles(heldout_csv),
            HELD_OUT_PAIRS,
            True,
        ),
        "micro-neural control": (
            load_public_results(micro_csv, acceptance_rule="control"),
            public_roles(micro_csv, acceptance_rule="control"),
            MICRO_PAIRS,
            True,
        ),
        "micro-neural fixed_eval": (
            load_public_results(micro_csv, acceptance_rule="fixed_eval"),
            public_roles(micro_csv, acceptance_rule="fixed_eval"),
            MICRO_PAIRS,
            True,
        ),
    }
    for label, (runs, roles, _pairs, _by_spec) in public_sets.items():
        n_ctrl = len(split_by(runs, lambda r: r.controller))
        shown = ", ".join(f"{k}={v}" for k, v in sorted(roles.items())) or "없음"
        print(f"  {label}: {len(runs)} run, controller {n_ctrl}종, 역할 {shown}")

    if args.raw_dir is None:
        print()
        print("`--raw-dir` 가 없어 raw 대조를 건너뛴다. 공개 데이터만으로 값을 출력한다.")
        print("**이 출력만으로는 공개 CSV 가 원본과 같다고 말할 수 없다.**")
        for label, (runs, roles, pairs, by_spec) in public_sets.items():
            show(label, stat_rows(runs, roles, pairs, by_spec=by_spec))
        return 0

    raw_map = {
        "held-out quadratic": (args.raw_dir / HELD_OUT_RAW, HELD_OUT_PAIRS, True),
        "micro-neural control": (args.raw_dir / MICRO_CONTROL_RAW, MICRO_PAIRS, True),
        "micro-neural fixed_eval": (args.raw_dir / MICRO_FIXED_RAW, MICRO_PAIRS, True),
    }
    missing = [p for p, _, _ in raw_map.values() if not p.exists()]
    if missing:
        for path in missing:
            print(f"없음: {path}")
        return 2

    print()
    print("두 경로 대조 (허용 오차 없음)")
    problems: list[str] = []
    for label, (raw_path, pairs, by_spec) in raw_map.items():
        raw_runs = load_raw(raw_path)
        # 경로 A 는 summary JSON 의 selection 을, 경로 B 는 공개 CSV 의
        # controller_role 열을 쓴다. **두 출처가 일치해야 키가 맞는다.**
        alias = alias_of(raw_path)
        public_runs, roles, _p, _b = public_sets[label]
        if alias != roles:
            problems.append(
                f"{label}: 역할 대응 불일치\n    summary {alias}\n    csv     {roles}"
            )
        from_raw = stat_rows(raw_runs, alias, pairs, by_spec=by_spec)
        from_public = stat_rows(public_runs, roles, pairs, by_spec=by_spec)
        problems += compare(label, from_raw, from_public)

    print()
    print("=" * 88)
    if problems:
        print(f"실패 {len(problems)}건")
        for text in problems:
            print(f"  {text}")
        return 1
    print("통과: 공개 CSV 가 raw 와 같은 통계를 낸다 (median / CI / p / n / 절대값 / 비용)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
