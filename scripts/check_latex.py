r"""LaTeX 원고의 구조를 TeX 엔진 없이 빠르게 검사한다.

**깨지기 쉬운 참조만** 기계적으로 확인한다. 조판 자체는 이 검사의 대상이 아니고
문서화된 4-pass 빌드로 따로 검증한다.

    pdflatex main && bibtex main && pdflatex main && pdflatex main

Overfull/Underfull box, float 배치, 페이지 나눔 같은 조판 결과는 여기서 잡지 못한다.
빌드 로그를 직접 봐야 한다.

```text
1  \input{...} 대상 파일이 존재하는가
2  \includegraphics{...} 그림 파일이 존재하는가
3  \cite / \citep / \citet 의 키가 references.bib 에 있는가
4  \ref / \eqref 가 가리키는 \label 이 존재하는가
5  정의했지만 참조하지 않은 \label 이 있는가 (경고)
6  생성 파일(tables/*.tex)을 사람이 고친 흔적이 있는가
7  금지 표현이 LaTeX 본문에 없는가 (check_claims 와 같은 목록)
8  아직 채우지 않은 \PLACEHOLDER 가 남아 있는가
9  원고가 인용한 공개 저장소와 릴리스 태그가 **실제로 존재하는가** (--check-remote)
```

**공개 전 게이트는 `--strict --check-remote` 다.**

`[8]` 은 자리를 채웠는지만 본다. 이름을 채우는 것으로는 그 artifact 가 존재하는지 알 수
없으므로 `[9]` 가 원격을 조회한다. 존재하지 않는 태그나 URL 을 실재하는 artifact 처럼
제시하는 사고를 막는 장치다 (E12 와 같은 유형).

`[9]` 는 네트워크를 쓰므로 기본으로 켜지 않는다. 오프라인에서 조판 검사만 하려면
플래그 없이 실행한다.

`[7]` 은 `check_claims.py` 의 목록을 그대로 import 한다. 목록이 갈리면 markdown 은
통과하고 LaTeX 만 과대주장하는 상태가 생긴다.

LaTeX 인용부호 ``...'' 안의 문구는 검사에서 제외한다. `§14` 의 "우리가 주장하지 않는
것" 표는 금지 문구를 **인용해 부인하는** 자리이므로 그대로 두어야 한다.

사용법:
    python scripts/check_latex.py
    python scripts/check_latex.py --strict
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from check_claims import ALLOWED_CONTEXT, FORBIDDEN, LEGITIMATE_COMPOUNDS

INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite[a-zA-Z]*(?:\[[^\]]*\])*\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:eqref|[a-zA-Z]*ref)\{([^}]+)\}")
BIB_ENTRY_RE = re.compile(r"^@\w+\{\s*([^,\s]+)\s*,", re.MULTILINE)
GENERATED_MARK = "scripts/make_tables.py 가 생성한다"
# LaTeX 인용부호. 안쪽 문구는 부인하려고 옮겨 적은 것이므로 금지어 검사에서 뺀다.
LATEX_QUOTE_RE = re.compile(r"``.*?''", re.DOTALL)
# 아직 존재하지 않는 값의 자리. main.tex 의 \PLACEHOLDER 매크로.
PLACEHOLDER_RE = re.compile(r"\\PLACEHOLDER\{([^}]*)\}")
# 원고가 인용하는 공개 저장소와 릴리스 태그.
REPO_URL_RE = re.compile(r"\\url\{(https://github\.com/[^}]+)\}")
RELEASE_TAG_RE = re.compile(r"release tag\s*\n?\s*\\texttt\{([^}]+)\}")


def strip_comments(text: str) -> str:
    """주석을 지운다. `\\%` 는 남긴다."""
    out = []
    for line in text.splitlines():
        idx = 0
        while True:
            idx = line.find("%", idx)
            if idx < 0:
                out.append(line)
                break
            if idx > 0 and line[idx - 1] == "\\":
                idx += 1
                continue
            out.append(line[:idx])
            break
    return "\n".join(out)


def remote_refs(url: str) -> tuple[bool, set[str]]:
    """`(도달 가능한가, 태그 이름 집합)`.

    `git ls-remote` 를 쓴다. 저장소가 비어 있으면 도달은 되지만 ref 가 없다. 그
    구분이 중요하다. 원고가 URL 과 태그를 함께 인용하므로 둘을 따로 확인한다.
    """
    proc = subprocess.run(
        ["git", "ls-remote", "--tags", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, set()
    tags: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split("\trefs/tags/")
        if len(parts) == 2:
            tags.add(parts[1].removesuffix("^{}"))
    return True, tags


def scan_forbidden(body: str) -> list[tuple[int, str, str]]:
    """금지 표현을 찾는다. `check_claims.scan_forbidden` 과 같은 규칙이다."""
    patterns = [(p, re.compile(rf"\b{p}\b", re.IGNORECASE)) for p in FORBIDDEN]
    hits: list[tuple[int, str, str]] = []
    for i, line in enumerate(body.splitlines(), start=1):
        low = LATEX_QUOTE_RE.sub(" ", line).lower()
        if any(ctx.lower() in low for ctx in ALLOWED_CONTEXT):
            continue
        probe = low
        for compound in LEGITIMATE_COMPOUNDS:
            probe = probe.replace(compound, " ")
        for name, rx in patterns:
            if rx.search(probe):
                hits.append((i, name, line.strip()[:110]))
    return hits


def collect(root: Path, main: Path) -> tuple[dict[Path, str], list[str]]:
    """`main.tex` 에서 시작해 `\\input` 을 따라가며 본문을 모은다."""
    errors: list[str] = []
    bodies: dict[Path, str] = {}
    queue = [main]
    seen: set[Path] = set()
    while queue:
        path = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            errors.append(f"\\input 대상이 없다: {path}")
            continue
        body = strip_comments(path.read_text(encoding="utf-8"))
        bodies[path] = body
        for target in INPUT_RE.findall(body):
            name = target if target.endswith(".tex") else f"{target}.tex"
            queue.append(root / name)
    return bodies, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("paper"))
    parser.add_argument("--main", type=Path, default=None)
    parser.add_argument("--bib", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument(
        "--check-remote",
        action="store_true",
        help="원고가 인용한 공개 저장소와 릴리스 태그를 원격에서 조회한다. 공개 전 게이트",
    )
    parser.add_argument("--strict", action="store_true", help="경고도 실패로 취급한다")
    args = parser.parse_args()

    root: Path = args.root
    main_tex: Path = args.main or root / "main.tex"
    bib_path: Path = args.bib or root / "references.bib"
    fig_dir: Path = args.figures or root / "figures"

    if not main_tex.exists():
        print(f"없음: {main_tex}")
        return 2

    errors: list[str] = []
    warnings: list[str] = []

    bodies, input_errors = collect(root, main_tex)
    errors += input_errors

    print("=" * 88)
    print(f"LaTeX 구조 검사   main {main_tex}")
    print(f"  본문 파일 {len(bodies)}개")
    print("=" * 88)

    print()
    print("[1] \\input 대상 존재")
    for path in sorted(bodies):
        print(f"  {path.relative_to(root) if root in path.parents else path}")

    print()
    print("[2] 그림 파일 존재")
    figures = {f for body in bodies.values() for f in GRAPHICS_RE.findall(body)}
    for name in sorted(figures):
        target = fig_dir / name
        ok = target.exists()
        print(f"  {'있음' if ok else '**없음**'}  {target}")
        if not ok:
            errors.append(f"그림 파일이 없다: {target}")
    if not figures:
        print("  없음")

    print()
    print("[3] 인용 키")
    if not bib_path.exists():
        errors.append(f"bib 가 없다: {bib_path}")
        bib_keys: set[str] = set()
    else:
        bib_keys = set(BIB_ENTRY_RE.findall(bib_path.read_text(encoding="utf-8")))
    used: set[str] = set()
    for body in bodies.values():
        for group in CITE_RE.findall(body):
            used |= {k.strip() for k in group.split(",") if k.strip()}
    print(f"  bib 항목 {len(bib_keys)}개, 인용 키 {len(used)}종")
    for key in sorted(used - bib_keys):
        print(f"  **{key}** 가 {bib_path.name} 에 없다")
        errors.append(f"인용 키 {key} 가 bib 에 없다")
    unused = sorted(bib_keys - used)
    if unused:
        print(f"  LaTeX 미인용 {len(unused)}개: {', '.join(unused)}")
        warnings.append(f"bib 항목 {len(unused)}개가 LaTeX 본문에서 인용되지 않았다")

    print()
    print("[4] \\ref 대상 \\label 존재")
    labels: set[str] = set()
    refs: set[str] = set()
    for body in bodies.values():
        labels |= set(LABEL_RE.findall(body))
        refs |= set(REF_RE.findall(body))
    print(f"  label {len(labels)}개, ref {len(refs)}종")
    for name in sorted(refs - labels):
        print(f"  **{name}** 에 대응하는 \\label 이 없다")
        errors.append(f"\\ref{{{name}}} 의 label 이 없다")

    print()
    print("[5] 참조되지 않은 \\label (정보)")
    print("    절 anchor 는 참조 없이도 정상이다. 경고로 올리지 않는다.")
    orphan = sorted(labels - refs)
    print(f"  {len(orphan)}개" + (f": {', '.join(orphan)}" if orphan else ""))

    print()
    print("[6] 생성 표를 손으로 고쳤는지")
    table_dir = root / "tables"
    generated = sorted(table_dir.glob("*.tex")) if table_dir.exists() else []
    for path in generated:
        text = path.read_text(encoding="utf-8")
        if GENERATED_MARK not in text:
            print(f"  **{path.name}** 에 생성 표시가 없다")
            errors.append(f"{path} 에 생성 표시가 없다. 손으로 만든 표인지 확인한다")
    print(f"  검사 완료 ({len(generated)}개)")

    print()
    print("[7] 금지 표현 검사 (check_claims 와 같은 목록)")
    total = 0
    for path in sorted(bodies):
        for line_no, word, text in scan_forbidden(bodies[path]):
            total += 1
            print(f"  {path.name}:{line_no}  '{word}'  {text}")
            errors.append(f"{path.name}:{line_no} 금지 표현 '{word}'")
    if total == 0:
        print("  없음")

    # 8. 아직 채우지 않은 자리. **공개 전 게이트다.**
    print()
    print("[8] 미해결 채움 표시 (\\PLACEHOLDER)")
    print("    존재하지 않는 URL·태그를 실재하는 artifact 처럼 제시하는 사고를 막는다")
    pending = 0
    for path in sorted(bodies):
        for i, line in enumerate(bodies[path].splitlines(), start=1):
            for what in PLACEHOLDER_RE.findall(line):
                pending += 1
                print(f"  {path.name}:{i}  {what}")
    if pending:
        warnings.append(
            f"채움 표시 {pending}개가 남아 있다. 공개 전에 0 이어야 한다"
        )
    else:
        print("  없음")

    # 9. 인용한 artifact 가 실제로 존재하는가. **공개 전 게이트다.**
    print()
    print("[9] 인용한 공개 저장소와 릴리스 태그의 실재")
    joined = "\n".join(bodies.values())
    urls = sorted(set(REPO_URL_RE.findall(joined)))
    tags = sorted(set(RELEASE_TAG_RE.findall(joined)))
    print(f"  원고가 인용한 저장소 {len(urls)}개, 릴리스 태그 {len(tags)}개")
    for url in urls:
        print(f"    {url}")
    for tag in tags:
        print(f"    tag {tag}")
    if not args.check_remote:
        print("  건너뜀. --check-remote 로 실제 조회한다 (네트워크를 쓴다)")
        if urls or tags:
            warnings.append(
                "인용한 저장소와 태그의 실재를 확인하지 않았다. "
                "공개 전에 --check-remote 로 확인한다"
            )
    elif not urls:
        print("  인용한 저장소가 없다. 확인할 것이 없다")
    else:
        for url in urls:
            reachable, remote_tags = remote_refs(url)
            if not reachable:
                print(f"  **도달 불가** {url}")
                errors.append(f"원고가 인용한 저장소에 도달할 수 없다: {url}")
                continue
            print(f"  도달  {url}  (원격 태그 {len(remote_tags)}개)")
            for tag in tags:
                if tag in remote_tags:
                    print(f"    있음    {tag}")
                else:
                    print(f"    **없음**  {tag}")
                    errors.append(
                        f"원고가 인용한 릴리스 태그 {tag} 가 {url} 에 없다. "
                        "공개 전에 push 한다"
                    )

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
    print()
    print("**주의.** 이 검사는 조판 오류를 잡지 못한다. 공개 전에 TeX 환경에서")
    print("  pdflatex main && bibtex main && pdflatex main && pdflatex main")
    print("을 한 번 실행해야 한다.")
    print("공개 전 게이트: scripts/check_latex.py --strict --check-remote")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
