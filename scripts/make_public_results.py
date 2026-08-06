"""공개용 행 단위 결과 파일을 raw 에서 만든다.

`results/raw/*.jsonl` 은 비공개다. 그러나 원고의 주장은 paired median, bootstrap CI,
Wilcoxon p 에 의존하므로 **그 통계를 다시 계산할 수 있는 행 단위 데이터**는 공개해야
한다. 이 스크립트가 그 최소 집합을 만든다.

무엇이 재현 가능한가
--------------------
`rl_newton.benchmark.metrics.compare_paired_delta` 는 `n_boot=10000`, `seed=0` 으로
결정론적이다. 따라서 공개 CSV 와 동봉된 `metrics.py` 만으로 median / CI / p 를 **정확히**
재현할 수 있다. `scripts/verify_public_results.py` 가 그것을 검사한다.

```text
공개한다      run 별 initial_loss, final_loss, 비용, 거절률, 식별자
공개하지 않는다  step 단위 궤적, 실행 환경 provenance, 장치 이름
```

개인정보
--------
`RunSummary` 에는 host 나 경로 필드가 없다. 따라서 이 CSV 는 **구성상** 개인정보를
담지 않는다. 그래도 값 검사를 한 번 더 한다 (`--forbid`).

사용법
------
    python scripts/make_public_results.py --raw-dir <비공개 raw 경로> --out-dir results/public
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# **이 저장소의 src 를 우선한다.** verify_public_results.py 와 같은 이유다.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rl_newton.benchmark.metrics import RunSummary  # noqa: E402
from rl_newton.benchmark.store import ResultStore  # noqa: E402

# public-export-allow-tokens: `--forbid` 기본값이 금지 문자열 **패턴**을 담는다.
# 실제 장치 이름이나 사용자 경로는 담지 않는다.

_SEED_SUFFIX = re.compile(r"_seed\d+$")
_KAPPA = re.compile(r"quad_\w*?_?d(\d+)_k([0-9.e+]+)$")
_MICRO_BATCH = re.compile(r"_cs(\d+)$")
_BEAM_FROM_NAME = re.compile(r"_b(\d+)_[0-9a-f]{8}\.jsonl$")

# CSV 열 순서. **행 단위 통계 재현에 필요한 것만** 넣는다.
COLUMNS = (
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
    # 튜닝으로 선택된 baseline 이 어느 것인지. 이것이 없으면 CSV 만으로는
    # "best_static" 이 static[2] 인지 static[4] 인지 알 수 없다. 선택 결과는
    # raw 파일마다 다르다 (§3.2).
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


@dataclass(frozen=True, slots=True)
class Source:
    """공개 CSV 하나에 들어갈 raw 결과."""

    raw: str
    out: str
    role: str
    acceptance_rule: str


# 원고가 실제로 인용하는 결과만 공개한다.
# 승계된 legacy pilot 두 개(8e9cdd02, 44e3242e)와 구 2-spec micro-neural(7aac1b26)은
# 원고가 인용하지 않으므로 넣지 않는다. 그 사실을 manifest 에 남긴다.
SOURCES = (
    Source(
        "headroom_challenge-heldout_step_size_fixed_b8_9a18b6e9.jsonl",
        "heldout_quadratic.csv",
        "held-out confirmation",
        "control",
    ),
    Source(
        "headroom_challenge_step_size_fixed_b8_fed9aebd.jsonl",
        "configuration_selection.csv",
        "configuration selection",
        "control",
    ),
    Source(
        "headroom_challenge_step_size_fixed_b8_fc78c2ad.jsonl",
        "configuration_selection.csv",
        "configuration selection (cost dry run, E1)",
        "control",
    ),
    Source(
        "headroom_micro-neural_step_size_fixed_b8_0bec1125.jsonl",
        "micro_neural.csv",
        "exploratory ablation",
        "control",
    ),
    Source(
        "headroom_micro-neural_step_size_fixed_b8_9f3194be.jsonl",
        "micro_neural.csv",
        "exploratory ablation",
        "fixed_eval",
    ),
    Source(
        "headroom_nonlinear-diagnostic_step_size_fixed_b8_2a09bd45.jsonl",
        "nonlinear_diagnostic.csv",
        "diagnostic, unusable benchmark",
        "control",
    ),
    Source(
        "headroom_pilot_step_size_fixed_b4_9d725689.jsonl",
        "dev_pilot.csv",
        "dev pilot",
        "control",
    ),
)

EXCLUDED_RAW = {
    "headroom_pilot_step_size_fixed_b4_8e9cdd02.jsonl": "legacy pilot. 원고가 인용하지 않는다",
    "headroom_pilot_step_size_fixed_b4_44e3242e.jsonl": "seed 1개 pilot. 원고가 인용하지 않는다",
    "headroom_micro-neural_step_size_fixed_b8_7aac1b26.jsonl": (
        "구 2-spec micro-neural. 3-spec 판(0bec1125)이 대체한다"
    ),
}


def spec_of(run: RunSummary) -> str:
    return _SEED_SUFFIX.sub("", run.task_instance_id)


def condition_number(spec: str) -> str:
    m = _KAPPA.search(spec)
    return m.group(2) if m else ""


def batch_size(spec: str) -> str:
    if spec.endswith("_fb"):
        return "full"
    m = _MICRO_BATCH.search(spec)
    return m.group(1) if m else ""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sweep_config(raw: Path) -> dict:
    """summary JSON 에서 budget 과 beam 을 읽는다. 없으면 파일명에서 beam 만 얻는다."""
    out: dict = {}
    summary = raw.parents[1] / "summaries" / f"{raw.stem}.json"
    if summary.exists():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        for section in payload.values():
            if not isinstance(section, dict):
                continue
            for key in ("budget_ge", "cost_budget_ge"):
                if key in section and "budget_ge" not in out:
                    out["budget_ge"] = section[key]
            for key in ("beam", "beam_width"):
                if key in section and "beam_width" not in out:
                    out["beam_width"] = section[key]
    if "beam_width" not in out:
        m = _BEAM_FROM_NAME.search(raw.name)
        if m:
            out["beam_width"] = int(m.group(1))
    return out


def selected_roles(raw: Path) -> dict[str, str]:
    """실제 controller 라벨 -> 역할 이름.

    `best_static` / `best_open_loop` 는 컨트롤러가 아니라 **튜닝 선택 결과**다
    (§3.2). 어느 후보가 뽑혔는지는 raw 파일마다 다르므로 CSV 에 담아야 공개
    데이터만으로 원고의 비교를 재현할 수 있다.
    """
    out: dict[str, str] = {}
    summary = raw.parents[1] / "summaries" / f"{raw.stem}.json"
    if not summary.exists():
        return out
    payload = json.loads(summary.read_text(encoding="utf-8"))
    for family, manifest in (payload.get("selections") or {}).items():
        label = manifest.get("selected_label")
        if label:
            role = "best_static" if family == "static" else f"best_{family}"
            out[label] = role
    return out


def rows_of(source: Source, raw: Path) -> list[dict]:
    cfg = sweep_config(raw)
    roles = selected_roles(raw)
    rows: list[dict] = []
    for record in ResultStore(raw):
        if record.status != "completed" or record.summary is None:
            continue
        r = record.summary
        spec = spec_of(r)
        rows.append(
            {
                "raw_source": raw.name,
                "role": source.role,
                "acceptance_rule": source.acceptance_rule,
                "beam_width": cfg.get("beam_width", ""),
                "budget_ge": cfg.get("budget_ge", ""),
                "task_spec": spec,
                "condition_number": condition_number(spec),
                "batch_size": batch_size(spec),
                "seed": r.seed,
                "controller": r.controller,
                "controller_role": roles.get(r.controller, ""),
                # log_improvement 는 파생값이다. initial/final 을 함께 내보내
                # 소비자가 같은 floor 규약으로 재계산해 대조할 수 있게 한다.
                "initial_loss": repr(r.initial_loss),
                "final_loss": repr(r.final_loss),
                "log_improvement": repr(r.log_improvement),
                "floor_hit": int(r.floor_hit),
                "total_cost_ge": repr(r.total_cost_ge),
                "search_cost_ge": repr(r.search_cost_ge),
                "n_steps": r.n_steps,
                "stop_reason": r.stop_reason,
                "rejection_rate": repr(r.rejection_rate),
                "failure_rate": repr(r.failure_rate),
                "negative_curvature_rate": repr(r.negative_curvature_rate),
                "target": r.target,
                "reached": int(r.reached),
                "cost_to_target_ge": (
                    "" if r.cost_to_target_ge is None else repr(r.cost_to_target_ge)
                ),
                "steps_to_target": (
                    "" if r.steps_to_target is None else r.steps_to_target
                ),
            }
        )
    rows.sort(key=lambda d: (d["task_spec"], d["seed"], d["controller"]))
    return rows


def private_commit() -> dict:
    """이 export 를 만든 소스 커밋.

    `dirty` 는 **추적 중인 파일이 수정됐는가**만 본다. 방금 만든 출력 파일은 아직
    추적되지 않으므로 `--untracked-files=no` 로 제외한다. 그렇게 하지 않으면 어떤
    export 든 항상 dirty 로 기록되어 이 필드가 쓸모없어진다.

    권장 순서는 코드를 먼저 커밋하고 `--require-clean` 으로 생성하는 것이다. 그러면
    `commit` 이 이 데이터를 만든 코드를 정확히 가리킨다.
    """

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/public"))
    parser.add_argument(
        "--forbid",
        nargs="*",
        default=["DESKTOP-", "jwkim", "OneDrive", "C:\\"],
        help="공개 CSV 에 나오면 실패하는 문자열",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="추적 파일이 수정돼 있으면 실패한다. 공개 릴리스 생성 시 쓴다",
    )
    args = parser.parse_args()

    # 아래 루프가 `source` 를 쓰므로 이름을 겹치지 않게 둔다.
    source_commit = private_commit()
    if args.require_clean and source_commit["dirty"]:
        print("**중단** 추적 중인 파일이 수정돼 있다.")
        print("  manifest 의 source.commit 이 이 데이터를 만든 코드를 가리키지 못한다.")
        print("  코드를 먼저 커밋한 뒤 다시 실행한다.")
        return 1

    missing = [s.raw for s in SOURCES if not (args.raw_dir / s.raw).exists()]
    if missing:
        for name in missing:
            print(f"없음: {args.raw_dir / name}")
        print("비공개 raw 가 있는 환경에서 실행해야 한다.")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict]] = {}
    per_source: list[dict] = []
    for source in SOURCES:
        raw = args.raw_dir / source.raw
        rows = rows_of(source, raw)
        grouped.setdefault(source.out, []).extend(rows)
        per_source.append(
            {
                "raw": source.raw,
                "sha256": sha256_of(raw),
                "role": source.role,
                "acceptance_rule": source.acceptance_rule,
                "public_file": source.out,
                "rows": len(rows),
            }
        )
        print(f"읽음: {source.raw}  {len(rows)} run  -> {source.out}")

    written: dict[str, str] = {}
    for name, rows in sorted(grouped.items()):
        path = args.out_dir / name
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(COLUMNS), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        text = path.read_text(encoding="utf-8")
        for token in args.forbid:
            if token in text:
                print(f"**중단** {path} 에 금지 문자열 '{token}' 이 있다")
                return 1
        written[name] = sha256_of(path)
        print(f"저장: {path}  {len(rows)} 행")

    manifest = {
        "export_version": "public-results-v1",
        "source": source_commit,
        "aggregation": {
            "median": "rl_newton.benchmark.metrics.median_of",
            "paired": "rl_newton.benchmark.metrics.compare_paired_delta",
            "bootstrap_n": 10000,
            "bootstrap_seed": 0,
            "wilcoxon": "rl_newton.benchmark.metrics.wilcoxon_signed_rank_p",
            "note": (
                "부트스트랩은 Python 표준 라이브러리의 random.Random(seed) 를 쓴다. "
                "NumPy 나 SciPy 에 의존하지 않는다. **고정된 환경에서** 이 CSV 와 "
                "동봉된 metrics.py 로 median / CI / p 를 다시 계산할 수 있고 "
                "scripts/verify_public_results.py 가 그것을 검사한다. "
                "판본이 다른 환경에서의 비트 수준 일치는 주장하지 않는다."
            ),
        },
        "log_improvement": {
            "formula": "log(initial_loss) - log(max(final_loss, floor))",
            "floor": "max(tiny, abs(initial_loss) * 100 * eps)   (프로토콜 D14)",
            "note": "파생값이므로 initial_loss / final_loss 를 함께 공개한다",
        },
        "sources": per_source,
        "excluded_raw": EXCLUDED_RAW,
        "files": written,
        "privacy": (
            "RunSummary 에는 host 나 경로 필드가 없다. 이 CSV 는 구성상 개인정보를 "
            "담지 않으며, 생성 시 금지 문자열 검사를 통과했다."
        ),
    }
    path = args.out_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
