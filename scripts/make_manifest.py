"""raw 결과의 checksum 과 정체성을 모아 `paper/evidence_map.md` 를 만든다.

**원고 수정 중 숫자가 수기로 변질되는 것을 막는 것이 목적이다.** 표와 그림이 어느
raw 결과에서 나왔는지 SHA-256 으로 고정한다.

사용법:
    python scripts/make_manifest.py --out paper/evidence_map.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# 논문에서 각 결과가 맡는 역할. 리뷰어가 사전에 지정했다.
ROLES = {
    "headroom_calibrate-beam": ("beam width calibration", "설정 선택 전 단계"),
    "headroom_pilot": ("dev pilot", "탐색적. 2개 spec 이 포화됐다 (D19)"),
    "headroom_challenge_": ("configuration selection", "설정 선택 근거 (D21/D22)"),
    "headroom_challenge-heldout": ("held-out confirmation", "최종 효과 추정 (D26)"),
    "headroom_nonlinear-diagnostic": ("diagnostic, unusable", "국소최소점 cap (D23)"),
    "headroom_micro-neural": ("exploratory ablation", "regime 및 수락 규칙 (D27/D28/D31)"),
}


@dataclass(frozen=True, slots=True)
class Entry:
    raw: Path
    role: str
    note: str
    sha256: str
    n_lines: int
    n_completed: int
    n_failed: int
    experiment_ids: tuple[str, ...]
    sweep_id: str
    aggregation_id: str
    git_commit: str
    code_dirty: bool
    selections: dict[str, str]
    specs: tuple[str, ...]
    seeds: tuple[int, ...]
    budget_ge: float
    beam: int
    acceptance_loss: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role_for(name: str) -> tuple[str, str]:
    for prefix, value in ROLES.items():
        if name.startswith(prefix):
            return value
    return ("unclassified", "")


def collect(raw: Path, summaries: Path) -> Entry | None:
    summary_path = summaries / f"{raw.stem}.json"
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = payload.get("meta") or {}
    identity = payload.get("identity") or {}
    prov = payload.get("execution_provenance") or {}

    experiment_ids: set[str] = set()
    # summary 의 execution_provenance.git_commit 이 빈 경우가 있다 (구 버전 버그).
    # raw 레코드는 run 마다 커밋을 기록하므로 그쪽을 권위 있는 값으로 쓴다.
    commits: set[str] = set()
    completed = failed = 0
    for line in raw.open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        experiment_ids.add(record["key"]["experiment_id"])
        if record.get("git_commit"):
            commits.add(str(record["git_commit"]))
        if record.get("status") == "completed":
            completed += 1
        else:
            failed += 1

    role, note = role_for(raw.name)
    return Entry(
        raw=raw,
        role=role,
        note=note,
        sha256=sha256_of(raw),
        n_lines=completed + failed,
        n_completed=completed,
        n_failed=failed,
        experiment_ids=tuple(sorted(experiment_ids)),
        sweep_id=str(payload.get("sweep_id") or identity.get("sweep_id") or ""),
        aggregation_id=str(payload.get("aggregation_id") or ""),
        git_commit=str(prov.get("git_commit") or "") or ", ".join(sorted(commits)),
        code_dirty=bool(prov.get("code_dirty")),
        selections={
            family: manifest.get("selected_label", "")
            for family, manifest in (payload.get("selections") or {}).items()
        },
        specs=tuple(str(s) for s in (meta.get("n_specs") and [] or [])),
        seeds=tuple(meta.get("seeds") or ()),
        budget_ge=float(meta.get("budget_ge") or 0.0),
        beam=int(meta.get("beam") or 0),
        acceptance_loss=str(meta.get("acceptance_loss") or "control"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--summary-dir", type=Path, default=Path("results/summaries"))
    parser.add_argument("--out", type=Path, default=Path("paper/evidence_map.md"))
    args = parser.parse_args()

    entries = [
        e
        for raw in sorted(args.raw_dir.glob("headroom_*.jsonl"))
        if (e := collect(raw, args.summary_dir)) is not None
    ]

    out: list[str] = [
        "# Evidence map",
        "",
        "`scripts/make_manifest.py` 가 생성한다. **손으로 수정하지 않는다.**",
        "",
        "원고의 모든 수치는 `docs/results_stage2.md` 에서 오고, 그 표는 아래 raw 결과에서",
        "생성된다. SHA-256 을 고정해 원고 수정 중 숫자가 수기로 변질되는 것을 막는다.",
        "",
        "## 결과별 역할",
        "",
        "리뷰어가 사전에 지정한 역할이다. **exploratory 결과로 primary 주장을 만들지 않는다.**",
        "",
        "| raw | 역할 | 완료 | 실패 | beam | 예산 GE | 수락 규칙 |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        out.append(
            f"| `{e.raw.name}` | {e.role} | {e.n_completed} | {e.n_failed} | "
            f"{e.beam or '-'} | {e.budget_ge:g} | `{e.acceptance_loss}` |"
        )

    out.append("")
    out.append("## Checksum 과 정체성")
    for e in entries:
        out.append("")
        out.append(f"### `{e.raw.name}`")
        out.append("")
        out.append(f"{e.note}" if e.note else "")
        out.append("")
        out.append("```text")
        out.append(f"sha256          {e.sha256}")
        out.append(f"완료 / 실패     {e.n_completed} / {e.n_failed}")
        out.append(f"sweep_id        {e.sweep_id}")
        out.append(f"aggregation_id  {e.aggregation_id}")
        out.append(f"git_commit      {e.git_commit}{' (dirty)' if e.code_dirty else ''}")
        if e.seeds:
            out.append(f"seeds           {list(e.seeds)}")
        out.append(f"experiment_id   {len(e.experiment_ids)}종")
        for exp in e.experiment_ids[:12]:
            out.append(f"                {exp}")
        if len(e.experiment_ids) > 12:
            out.append(f"                ... {len(e.experiment_ids) - 12}개 더")
        for family, label in sorted(e.selections.items()):
            out.append(f"selection       {family} -> {label}")
        out.append("```")

    out.append("")
    out.append("## 왜 `experiment_id` 가 여러 개인가")
    out.append("")
    out.append("정체성이 3계층으로 분리되어 있다 (프로토콜 D13). 컨트롤러마다 실제 쓰는")
    out.append("optimizer 설정이 다르므로 `run_semantics_id` 도 다르다. **정상 동작이다.**")
    out.append("")
    out.append("```text")
    out.append("run_semantics_id  이 컨트롤러가 실제 쓰는 optimizer 설정만")
    out.append("sweep_id          이번 실행이 요청한 run 집합")
    out.append("aggregation_id    집계 정책")
    out.append("```")
    out.append("")
    out.append("`git_commit` 과 `code_dirty` 는 어떤 ID 에도 넣지 않고")
    out.append("`execution_provenance` 로 분리한다. 문서만 고쳐도 해시가 바뀌면")
    out.append('"어떤 집합을 요청했는가" 라는 의미가 깨진다.')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"저장: {args.out}  (raw {len(entries)}개)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
