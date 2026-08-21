"""원고와 claim ledger 를 **기계적으로** 대조한다.

사람이 문장을 읽어 확인하면 누락이 생긴다. 원고에 `<!-- CLAIM: Cxx -->` 주석을 달고
아래를 검사한다. HTML 주석은 PDF 에 나오지 않으므로 원고 관리에만 쓰인다.

```text
1  SUPPORTED / LIMITED claim 이 draft 에 최소 1회 인용됐는가
2  NOT SUPPORTED claim 이 draft 에 인용되지 않았는가
3  금지 표현이 draft 에 없는가
4  숫자가 있는 claim 에 evidence source 가 있는가
5  draft 의 claim ID 가 ledger 에 존재하는가
6  draft 의 `[@key]` 가 references.bib 에 있고 서지정보가 TODO 가 아닌가
   `[CITATION NEEDED]` 가 남아 있지 않은가 (실패)
   `[TO BE FILLED: ...]` 가 남아 있지 않은가 (경고. 공개 전에 0 이어야 한다)
```

**[6] 이 검사하지 않는 것.** 인용한 논문의 *내용*이 우리 주장을 실제로 지지하는지는
기계로 확인할 수 없다. 그것은 `paper/CITATIONS.md` 의 사람 검토 체크리스트로 관리한다.

금지 표현은 리뷰어가 지정했다. 문장 안에 들어가면 주장 범위를 넘는다.

사용법:
    python scripts/check_claims.py
    python scripts/check_claims.py --strict     # 경고도 실패로 취급
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 리뷰어 지정 금지 표현. 대소문자를 구별하지 않는다.
# **단어 경계로 찾는다.** 부분 문자열로 찾으면 `provenance` 가 `prove` 로 잡힌다.
FORBIDDEN = (
    r"prove[sndrs]*",
    r"equivalent",
    r"equivalence",
    r"universally",
    r"feedback is useless",
    r"ppo failed",
    r"flop-matched",
    r"monotonic(?:ally)?",
    r"generalizes",
    r"state-of-the-art",
)

# 정당한 기술 용어. 금지 표현을 부분으로 포함하지만 주장이 아니다.
LEGITIMATE_COMPOUNDS = (
    "gradient-equivalent",  # GE 단위의 정식 이름
    "hvp-equivalent",       # 같은 단위를 HVP 기준으로 부른 이름 (§2.3 CPU 문구)
    "provenance",           # prove 를 부분으로 포함
    "monotonic acceptance",  # 수락 규칙의 이름. 리뷰어가 승인한 문장
    "monotonic descent",
    "equivalence margin",   # 없다는 것을 밝히는 문맥
    "equivalence testing",
)

# 금지 표현이 **부정문이나 인용으로** 나오는 것은 허용한다.
# 예: "we do not claim equivalence", "did not increase monotonically"
ALLOWED_CONTEXT = (
    "not increase monotonically",
    "did not increase monotonic",
    "쓰지 않는다",
    "쓰지 마",
    "주장하지 않는다",
    "주장하지 못",
    "금지",
    "not claim",
    "no equivalence",
    "equivalence margin",
    "did not establish",
    "cannot claim",
    "일반화할 수 없다",
    "일반화하지",
    "않는다",
)

CLAIM_RE = re.compile(r"<!--\s*CLAIM:\s*([A-Za-z0-9_]+)\s*-->")
# 원고의 인용 표기. pandoc 형식 `[@key]` / `[@a; @b]` 를 쓴다.
CITE_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_:.\-]*)")
BIB_ENTRY_RE = re.compile(r"^@\w+\{\s*([^,\s]+)\s*,", re.MULTILINE)
# `[CITATION NEEDED]` 가 남아 있으면 인용 작업이 끝나지 않았다.
CITATION_TODO_RE = re.compile(r"\[CITATION NEEDED\]")
# 아직 존재하지 않는 값의 자리 (공개 저장소 URL, 릴리스 태그 등).
FILL_TODO_RE = re.compile(r"\[TO BE FILLED:[^\]]*\]")
# 금지 주장을 **하지 않겠다고 밝히는** 참조. NOT SUPPORTED claim 은 이 형태만 허용한다.
AVOID_RE = re.compile(r"<!--\s*AVOID:\s*([A-Za-z0-9_]+)\s*-->")
LEDGER_HEADING = re.compile(r"^###\s+(C\d+)\.\s+(.*)$")
STATUS_HEADING = re.compile(r"^##\s+(SUPPORTED|LIMITED|EXPLORATORY|NOT SUPPORTED)")
NUMBER_RE = re.compile(r"[+-]?\d+\.\d{2,}")


@dataclass
class Claim:
    ident: str
    title: str
    status: str
    body: list[str] = field(default_factory=list)

    @property
    def has_numbers(self) -> bool:
        return any(NUMBER_RE.search(line) for line in self.body)

    @property
    def has_source(self) -> bool:
        text = "\n".join(self.body).lower()
        return any(k in text for k in ("source", "raw ", "tests", "tools", "protocol"))


def parse_ledger(path: Path) -> dict[str, Claim]:
    claims: dict[str, Claim] = {}
    status = "UNKNOWN"
    current: Claim | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m_status = STATUS_HEADING.match(line)
        if m_status:
            status = m_status.group(1)
            continue
        m = LEDGER_HEADING.match(line)
        if m:
            # `C1` -> `C01` 로 정규화한다. 원고 주석과 자리수를 맞춘다.
            raw = m.group(1)
            ident = f"C{int(raw[1:]):02d}"
            current = Claim(ident=ident, title=m.group(2).strip(), status=status)
            claims[ident] = current
            continue
        if current is not None:
            current.body.append(line)
    return claims


def normalize(ident: str) -> str:
    if re.fullmatch(r"C\d+", ident):
        return f"C{int(ident[1:]):02d}"
    return ident


def parse_bib(path: Path) -> dict[str, str]:
    """bib 파일에서 `{key: 항목 블록 전체}` 를 뽑는다.

    본격적인 BibTeX 파서가 아니다. `@type{key,` 로 항목을 나누고 그 블록의 텍스트를
    통째로 담는다. `[6]` 검사가 블록 안에 `TODO` 가 남아 있는지 보는 것이 목적이다.

    **주의.** 검증 상태는 `.bib` 에 두지 않는다. `plainnat` 이 `note` 필드를 조판
    결과에 인쇄하므로 내부 메모를 넣으면 참고문헌 목록에 나온다. 검증 기록은
    `paper/CITATIONS.md` 에 있다. 이 검사는 미완성 항목이 `.bib` 에 흘러드는 것을
    막는 안전망일 뿐이며, 통과가 검증을 뜻하지 않는다.
    """
    text = path.read_text(encoding="utf-8")
    entries: dict[str, str] = {}
    matches = list(BIB_ENTRY_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entries[m.group(1)] = text[m.end() : end]
    return entries


def scan_forbidden(path: Path) -> list[tuple[int, str, str]]:
    patterns = [(p, re.compile(rf"\b{p}\b", re.IGNORECASE)) for p in FORBIDDEN]
    hits: list[tuple[int, str, str]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        low = line.lower()
        if any(ctx.lower() in low for ctx in ALLOWED_CONTEXT):
            continue
        # 정당한 복합어를 먼저 지운 뒤 검사한다.
        probe = low
        for compound in LEGITIMATE_COMPOUNDS:
            probe = probe.replace(compound, " ")
        for name, rx in patterns:
            if rx.search(probe):
                hits.append((i, name, line.strip()[:110]))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("paper/claim_ledger.md"))
    parser.add_argument("--draft", type=Path, default=Path("paper/draft.md"))
    parser.add_argument("--bib", type=Path, default=Path("paper/references.bib"))
    parser.add_argument(
        "--also",
        nargs="*",
        type=Path,
        default=[Path("public/README.md")],
        help="금지 표현을 함께 검사할 문서. 공개 README 도 주장을 한다",
    )
    parser.add_argument("--strict", action="store_true", help="경고도 실패로 취급한다")
    args = parser.parse_args()

    for path in (args.ledger, args.draft, args.bib):
        if not path.exists():
            print(f"없음: {path}")
            return 2

    claims = parse_ledger(args.ledger)
    draft_text = args.draft.read_text(encoding="utf-8")
    cited = {normalize(m) for m in CLAIM_RE.findall(draft_text)}
    avoided = {normalize(m) for m in AVOID_RE.findall(draft_text)}

    by_status: dict[str, list[Claim]] = {}
    for claim in claims.values():
        by_status.setdefault(claim.status, []).append(claim)

    print("=" * 88)
    print(f"claim ledger  {args.ledger.name}   draft  {args.draft.name}")
    print(f"  claim {len(claims)}개, draft 인용 {len(cited)}종, 금지 표시 {len(avoided)}종")
    for status in sorted(by_status):
        print(f"  {status:<15} {len(by_status[status])}개")
    print("=" * 88)

    errors: list[str] = []
    warnings: list[str] = []

    # 1. SUPPORTED / LIMITED 는 최소 1회 인용
    print()
    print("[1] SUPPORTED / LIMITED claim 이 draft 에 인용됐는가")
    for status in ("SUPPORTED", "LIMITED"):
        for claim in by_status.get(status, []):
            mark = "인용" if claim.ident in cited else "**미인용**"
            print(f"  {claim.ident}  {status:<10} {mark}   {claim.title[:52]}")
            if claim.ident not in cited:
                warnings.append(f"{claim.ident} ({status}) 가 draft 에 인용되지 않았다")

    # 2. NOT SUPPORTED 는 인용 금지
    print()
    print("[2] NOT SUPPORTED claim 이 draft 에 주장으로 인용되지 않았는가")
    print("    `AVOID:` 표시는 허용한다. 하지 않겠다고 밝히는 참조다")
    for claim in by_status.get("NOT SUPPORTED", []):
        bad = claim.ident in cited
        note = "**주장으로 인용됨**" if bad else ("금지 표시" if claim.ident in avoided else "미인용")
        print(f"  {claim.ident}  {note:<16} {claim.title[:52]}")
        if bad:
            errors.append(f"{claim.ident} (NOT SUPPORTED) 가 draft 에 주장으로 인용됐다")

    # 3. 금지 표현. 원고와 **공개 문서**를 같은 목록으로 본다.
    # 공개 README 도 주장을 한다. 원고만 검사하면 README 로 과대주장이 새어 나간다.
    print()
    print("[3] 금지 표현 검사")
    targets = [args.draft] + [p for p in args.also if p.exists()]
    total_hits = 0
    for path in targets:
        for line_no, word, text in scan_forbidden(path):
            total_hits += 1
            print(f"  {path}:{line_no}  '{word}'  {text}")
            errors.append(f"{path}:{line_no} 금지 표현 '{word}'")
    print(f"  대상 {len(targets)}개" + ("" if total_hits else ", 위반 없음"))
    for path in args.also:
        if not path.exists():
            print(f"  건너뜀 (없음): {path}")

    # 4. 숫자가 있는 claim 은 근거 표기
    print()
    print("[4] 숫자를 담은 claim 에 evidence source 가 있는가")
    for claim in claims.values():
        if claim.status == "NOT SUPPORTED" or not claim.has_numbers:
            continue
        if not claim.has_source:
            print(f"  {claim.ident}  **근거 표기 없음**")
            errors.append(f"{claim.ident} 에 evidence source 가 없다")
    print("  검사 완료")

    # 5. draft 의 claim ID 가 ledger 에 존재
    print()
    print("[5] draft 의 claim ID 가 ledger 에 존재하는가")
    known = set(claims) | {"scope"}
    for ident in sorted((cited | avoided) - known):
        print(f"  **{ident}** 가 ledger 에 없다")
        errors.append(f"draft 의 claim ID {ident} 가 ledger 에 없다")
    print("  검사 완료")

    # 6. 인용 무결성. draft 의 `[@key]` 가 bib 에 있고 검증 상태가 TODO 가 아닌가
    print()
    print("[6] 인용 무결성")
    bib = parse_bib(args.bib)
    used = set(CITE_RE.findall(draft_text))
    print(f"  bib 항목 {len(bib)}개, draft 인용 키 {len(used)}종")
    for key in sorted(used - set(bib)):
        print(f"  **{key}** 가 {args.bib.name} 에 없다")
        errors.append(f"draft 가 인용한 bib 키 {key} 가 없다")
    for key in sorted(used & set(bib)):
        if "TODO" in bib[key]:
            print(f"  **{key}** 의 서지정보가 TODO 다")
            errors.append(f"서지정보가 TODO 인 {key} 를 draft 가 인용했다")
    todo_lines = [
        i
        for i, line in enumerate(draft_text.splitlines(), start=1)
        if CITATION_TODO_RE.search(line)
    ]
    for line_no in todo_lines:
        print(f"  {args.draft.name}:{line_no}  [CITATION NEEDED] 가 남아 있다")
        errors.append(f"{args.draft.name}:{line_no} [CITATION NEEDED] 미해결")
    # 아직 존재하지 않는 값의 자리. 인용과 달리 공개 직전에 채우므로 경고로 둔다.
    fill_lines = [
        (i, line.strip()[:90])
        for i, line in enumerate(draft_text.splitlines(), start=1)
        if FILL_TODO_RE.search(line)
    ]
    for line_no, text in fill_lines:
        print(f"  {args.draft.name}:{line_no}  채움 표시  {text}")
    if fill_lines:
        warnings.append(
            f"draft 에 채움 표시 {len(fill_lines)}개가 남아 있다. 공개 전에 0 이어야 한다"
        )
    unused = sorted(set(bib) - used)
    if unused:
        print(f"  미인용 bib 항목 {len(unused)}개: {', '.join(unused)}")
        warnings.append(f"bib 항목 {len(unused)}개가 draft 에서 인용되지 않았다")
    if not errors:
        print("  검사 완료")

    print()
    print("=" * 88)
    if errors:
        print(f"실패 {len(errors)}건")
        for e in errors:
            print(f"  {e}")
    if warnings:
        print(f"경고 {len(warnings)}건")
        for w in warnings:
            print(f"  {w}")
    if not errors and not warnings:
        print("통과: 모든 검사를 만족한다")
    elif not errors:
        print("통과 (경고 있음)")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
