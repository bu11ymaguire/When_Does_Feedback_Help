"""검증된 private 트리에서 공개 저장소용 디렉터리를 만든다.

**allowlist 방식이다.** 무엇을 뺄지가 아니라 무엇을 넣을지 적는다. denylist 는 새
파일이 생길 때마다 조용히 새어 나가므로 공개 준비에 쓰면 안 된다.

핵심 제약
---------
```text
.git 을 복사하지 않는다          공개 저장소는 git init 으로 새로 시작한다
비공개 raw 를 복사하지 않는다     results/public/ 의 행 단위 결과만 나간다
대상이 이미 git 저장소면 거부한다  private 이력이 섞이는 사고를 막는다
금지 문자열을 검사한다            장치 이름, 사용자명, 로컬 절대경로
```

왜 `.git` 을 빼는 것이 중요한가. 같은 `.git` 을 새 원격에 push 하면 과거 커밋의 장치
이름과 중간 시행착오가 함께 공개된다. 이 저장소가 장치 이름을 정리한 이유가 사라진다.

무엇이 나가고 무엇이 남는가
---------------------------
원고 `§15` 가 artifact 로 인용하는 문서는 **전부 나간다.** 인용했는데 없으면 그
인용이 거짓이 된다. 반대로 연구 진행용 문서(초안, 개요, 검토 요청)는 남긴다.

사용법
------
    python scripts/export_public_repo.py --dest ..\\OPt_with_RL_public_stage
    python scripts/export_public_repo.py --dest <경로> --require-clean --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# 저장소 상대경로처럼 보이는 문자열. 최상위 디렉터리 이름으로 시작하는 것만 잡는다.
_REPO_PATH_RE = re.compile(
    r"(?:(?<=\s)|(?<=`)|(?<=^))"
    r"((?:src|scripts|tests|configs|results|paper|docs|notebooks)/[A-Za-z0-9_./*-]+)",
    re.MULTILINE,
)

# --- 공개할 것 -------------------------------------------------------------
# glob 패턴. 이 목록에 없는 것은 나가지 않는다.
ALLOWLIST = (
    # 패키징과 환경 고정
    "pyproject.toml",
    "uv.lock",
    # 구현
    "src/**/*.py",
    # 실험·집계·검증 도구. docs/reproduce.md 가 이들을 참조하므로 전부 나간다.
    "scripts/**/*.py",
    "tests/**/*.py",
    # 설정
    "configs/**/*.yaml",
    "configs/**/*.yml",
    "configs/**/*.json",
    # 결과. **행 단위 공개 결과만.** raw 는 나가지 않는다.
    "results/public/*.csv",
    "results/public/manifest.json",
    "results/summaries/*.json",
    "results/raw/.gitkeep",
    "results/checkpoints/.gitkeep",
    "results/figures/.gitkeep",
    # 원고. 제출용 소스와 생성된 표·그림.
    "paper/main.tex",
    "paper/references.bib",
    "paper/sections/*.tex",
    "paper/tables/*.tex",
    "paper/figures/*.png",
    # 원고 §15 가 artifact 로 인용하는 기록. 인용했으므로 반드시 나간다.
    "paper/claim_ledger.md",
    "paper/CITATIONS.md",
    "paper/evidence_map.md",
    "docs/reproduce.md",
    "docs/results_stage2.md",
    "docs/experiment_protocol.md",
    # 노트북
    "notebooks/*.ipynb",
)
# `docs/*.png` 를 넣지 않는다. 이 디렉터리에는 제3자의 이름과 얼굴이 찍힌 스크린샷이
# 들어온다. 와일드카드로 열어 두면 파일을 떨구는 순간 조용히 공개된다. 동의를 받지 않은
# 사적 서신이 이런 경로로 새는 것을 막는다.
#
# 공개할 이미지는 paper/figures/ 에 두거나, 승인한 파일을 위 목록에 하나씩 적는다.

# 이름을 바꿔 내보내는 것. private 루트의 README 를 덮지 않기 위한 장치다.
RENAMES = {
    "public/README.md": "README.md",
    "public/README.en.md": "README.en.md",
    "public/LICENSE": "LICENSE",
    "public/CITATION.cff": "CITATION.cff",
    "public/.gitignore": ".gitignore",
}

# 의도적으로 내보내지 않는 것. 이유를 manifest 에 남긴다.
WITHHELD = {
    ".git/": "공개 저장소는 git init 으로 새로 시작한다. private 이력을 공유하지 않는다",
    "results/raw/*.jsonl": (
        "step 단위 비공개 기록. 논문의 통계는 results/public/ 에서 재계산된다. "
        "SHA-256 은 paper/evidence_map.md 에 있다"
    ),
    "results/checkpoints/": "정책 체크포인트. PPO 를 실행하지 않았으므로 내용이 없다",
    "paper/draft.md": (
        "한국어 내용 source of truth. 원고와 같은 내용을 다른 언어로 중복 제시하면 "
        "독자를 혼란하게 한다. 원고 §15 가 artifact 로 인용하지 않는다"
    ),
    "paper/outline.md": "집필 계획 문서. 연구 진행용이며 인용되지 않는다",
    "paper/REVIEW_PACKAGE.md": "외부 검토 요청용 발췌. 연구 진행용이다",
    "PUBLIC_RELEASE_NOTES.md": (
        "private 릴리스 절차 기록. private 태그를 언급한다. 공개에 필요한 내용은 "
        "원고 §15 의 privacy note 가 담는다"
    ),
    "README.md (private)": "한국어 개발 문서. public/README.md 가 대체한다",
    ".kiro/": "개발 환경 설정. 추적되지 않는다",
}

# 공개 텍스트에 나오면 실패하는 문자열.
FORBIDDEN = ("DESKTOP-", "OneDrive", "C:\\Users", "/Users/jwkim")

# 저자 연락처가 나와도 되는 자리. 원고와 CITATION.cff 가 **의도적으로** 담는다.
# 그 외 파일에 나오면 사고다.
CONTACT_ALLOWED = {
    "CITATION.cff",
    "README.md",
    "README.en.md",
    "paper/main.tex",
}

# 개인정보 정리 도구는 금지 문자열 **패턴**을 담을 수밖에 없다. 예를 들어
# sanitize_public_artifacts.py 는 `DESKTOP-XXXX` 를 정규식 주석에 적는다.
# 그런 파일은 스스로 이 표식을 선언한다. 표식을 쓴 파일은 export 출력과 manifest 에
# 드러나므로 조용히 통과하지 않는다.
PUBLIC_REPOSITORY = "https://github.com/bu11ymaguire/When_Does_Feedback_Help"
"""선언된 공개 저장소. 갱신 모드에서 대상의 원격이 이것인지 확인한다."""

ALLOW_MARKER = "public-export-allow-tokens"
# public-export-allow-tokens: 이 파일 자체가 금지 문자열 목록을 담는다.
# 연락처는 여기 적지 않고 public/CITATION.cff 에서 읽는다 (contact_token 참조).
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".tex",
    ".bib",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".toml",
    ".cff",
    ".ipynb",
    ".lock",
    ".gitignore",
    "",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def readme_references(text: str) -> set[str]:
    """공개 README 가 가리키는 저장소 내부 경로.

    markdown 링크와, 본문·코드블록에 나오는 저장소 상대경로를 모은다. 외부 URL 과
    앵커는 제외한다.
    """
    out: set[str] = set()
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
        target = target.split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        out.add(target.lstrip("./"))
    for path in re.findall(_REPO_PATH_RE, text):
        out.add(path.rstrip(".,)"))
    return out


def check_references(text: str, exported: set[str]) -> list[str]:
    """README 가 가리키는 경로가 export 안에 있는가.

    **없는 파일을 문서가 가리키는 사고를 막는다.** 실제로 공개 README 가 아직 만들지
    않은 노트북을 가리킨 적이 있다. 사람 주의력에 맡기지 않는다.
    """
    problems: list[str] = []
    for ref in sorted(readme_references(text)):
        if "*" in ref:
            prefix = ref.split("*", 1)[0]
            if not any(item.startswith(prefix) for item in exported):
                problems.append(f"README 가 가리키는 {ref} 에 맞는 파일이 없다")
            continue
        if ref in exported:
            continue
        # 디렉터리 참조는 그 아래 파일이 하나라도 있으면 통과한다.
        if any(item.startswith(ref.rstrip("/") + "/") for item in exported):
            continue
        problems.append(f"README 가 가리키는 {ref} 가 export 에 없다")
    return problems


def contact_token(root: Path) -> str | None:
    """저자 연락처를 `public/CITATION.cff` 에서 읽는다.

    **이 스크립트에 이메일을 적지 않는다.** 적으면 그 리터럴 자체가 공개 파일에
    들어가고, 연락처 정의가 두 곳으로 갈린다. 정의는 CITATION.cff 한 곳에 두고
    검사기는 읽어서 쓴다.
    """
    path = root / "public" / "CITATION.cff"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("email:"):
            return stripped.split(":", 1)[1].strip() or None
    return None


def resolve_sources(root: Path) -> dict[str, Path]:
    """`대상 상대경로 -> 원본 경로`. allowlist 와 RENAMES 를 합친다."""
    out: dict[str, Path] = {}
    for pattern in ALLOWLIST:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                out[path.relative_to(root).as_posix()] = path
    for src, dest in RENAMES.items():
        path = root / src
        if path.is_file():
            out[dest] = path
    return out


def scan_forbidden(
    rel: str, path: Path, contact: str | None
) -> tuple[list[str], bool]:
    """`(위반 목록, 표식을 썼는가)`."""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return [], False
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [], False

    hits: list[str] = []
    marked = ALLOW_MARKER in text
    if not marked:
        hits += [token for token in FORBIDDEN if token in text]
    # 연락처는 표식으로 면제되지 않는다. 나갈 자리가 정해져 있다.
    if contact and contact in text and rel not in CONTACT_ALLOWED:
        hits.append("author contact")
    return hits, marked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dest", type=Path, required=True, help="공개 staging 디렉터리")
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="추적 파일이 수정돼 있으면 실패한다. 실제 릴리스 때 쓴다",
    )
    parser.add_argument(
        "--force", action="store_true", help="대상이 비어 있지 않아도 덮어쓴다"
    )
    args = parser.parse_args()

    root = args.root.resolve()
    dest = args.dest.resolve()

    # --- 안전 검사 --------------------------------------------------------
    if dest == root or root in dest.parents:
        print(f"**중단** 대상이 소스 트리 안이다: {dest}")
        print("  공개 staging 은 형제 디렉터리에 둔다. 소스 안에 두면 다음 export 가 자신을 삼킨다.")
        return 1
    # 대상에 `.git` 이 있으면 **그 원격이 선언된 공개 저장소인지**만 허용한다.
    # 무조건 거부하면 첫 공개 이후 갱신을 할 수 없고, 무조건 허용하면 private
    # 이력이 섞이는 사고를 막지 못한다. 원격을 확인하는 것이 정확한 기준이다.
    if (dest / ".git").exists():
        existing = subprocess.run(
            ["git", "-C", str(dest), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if existing.removesuffix(".git") != PUBLIC_REPOSITORY.removesuffix(".git"):
            print(f"**중단** 대상에 .git 이 있고 원격이 공개 저장소가 아니다: {dest}")
            print(f"  원격     {existing or '(없음)'}")
            print(f"  기대값   {PUBLIC_REPOSITORY}")
            print("  private 이력이 섞이는 사고를 막기 위해 거부한다.")
            return 1
        print(f"갱신 모드. 대상이 이미 공개 저장소를 가리킨다: {existing}")
    if dest.exists() and any(dest.iterdir()) and not args.force:
        print(f"**중단** 대상이 비어 있지 않다: {dest}")
        print("  내용을 갈아치우려면 --force 를 준다.")
        return 1

    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if args.require_clean and dirty:
        print("**중단** 추적 중인 파일이 수정돼 있다.")
        print("  manifest 의 source commit 이 이 export 를 가리키지 못한다.")
        return 1

    sources = resolve_sources(root)
    if not sources:
        print("**중단** allowlist 에 걸린 파일이 없다. --root 를 확인한다.")
        return 1

    # --- 개인정보 검사 ----------------------------------------------------
    contact = contact_token(root)
    if contact is None:
        print("**주의** public/CITATION.cff 에서 연락처를 읽지 못했다. 연락처 검사를 건너뛴다.")
    problems: list[str] = []
    marked_files: list[str] = []
    for rel, path in sorted(sources.items()):
        hits, marked = scan_forbidden(rel, path, contact)
        if marked:
            marked_files.append(rel)
        problems += [f"{rel}: 금지 문자열 {token!r}" for token in hits]
    if problems:
        print(f"**중단** 공개 대상에 금지 문자열이 있다 ({len(problems)}건)")
        for text in problems:
            print(f"  {text}")
        print()
        print("정리 도구처럼 패턴을 담을 수밖에 없는 파일이면 그 파일에")
        print(f"  {ALLOW_MARKER}")
        print("표식을 두고 이유를 적는다. 표식은 export 출력과 manifest 에 드러난다.")
        return 1

    # --- README 가 없는 파일을 가리키는지 --------------------------------
    # 한국어판과 영문판 **둘 다** 본다. 한쪽만 검사하면 다른 쪽의 깨진 경로가
    # 조용히 공개된다.
    dangling: list[str] = []
    checked_readmes: list[str] = []
    for name in ("README.md", "README.en.md"):
        readme = sources.get(name)
        if readme is None:
            continue
        checked_readmes.append(name)
        for text in check_references(readme.read_text(encoding="utf-8"), set(sources)):
            dangling.append(f"{name}: {text}")
    if dangling:
        print(f"**중단** 공개 README 가 없는 경로를 가리킨다 ({len(dangling)}건)")
        for text in dangling:
            print(f"  {text}")
        print()
        print("allowlist 에 넣거나, 그 파일을 만들거나, README 에서 그 참조를 지운다.")
        return 1

    # --- 복사 -------------------------------------------------------------
    if dest.exists() and args.force:
        for child in sorted(dest.iterdir()):
            if child.name == ".git":
                # 갱신 모드에서는 공개 저장소의 이력을 보존한다. 위에서 원격을
                # 확인했으므로 private 이력이 아니다.
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    dest.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    for rel, path in sorted(sources.items()):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files[rel] = sha256_of(target)

    # allowlist 가 `.git` 을 담을 수 없으므로 복사로 생길 일은 없다. 그래도 확인한다.
    if any(rel.startswith(".git/") for rel in files):
        print("**중단** export 가 .git 아래 파일을 복사했다.")
        return 1

    manifest = {
        "export_version": "public-v1",
        "source": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": dirty,
        },
        "public_repository": PUBLIC_REPOSITORY,
        "method": (
            "allowlist export. .git 을 복사하지 않으므로 공개 저장소는 private 이력을 "
            "공유하지 않는다. 아래 files 의 SHA-256 으로 두 트리를 한 방향으로 대조한다."
        ),
        "renamed": RENAMES,
        "withheld": WITHHELD,
        "token_scan": {
            "forbidden": list(FORBIDDEN),
            "contact_checked": contact is not None,
            "contact_allowed_in": sorted(CONTACT_ALLOWED),
            "marker": ALLOW_MARKER,
            "marked_files": marked_files,
            "note": (
                "marked_files 는 개인정보 정리 도구다. 금지 문자열 패턴을 담을 수밖에 "
                "없어 스스로 표식을 선언했다. 실제 장치 이름이나 사용자 경로는 담지 않는다."
            ),
        },
        "readme_reference_check": (
            f"통과. 검사한 README {', '.join(checked_readmes)} 가 가리키는 "
            "저장소 경로가 모두 export 안에 있다."
            if checked_readmes
            else "건너뜀. README 가 export 대상에 없다."
        ),
        "counts": {"files": len(files), "marked": len(marked_files)},
        "files": files,
    }
    (dest / "EXPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    by_top: dict[str, int] = {}
    for rel in files:
        by_top[rel.split("/", 1)[0] if "/" in rel else "(root)"] = (
            by_top.get(rel.split("/", 1)[0] if "/" in rel else "(root)", 0) + 1
        )

    print(f"대상: {dest}")
    print(f"소스: {manifest['source']['commit'][:8]} ({manifest['source']['branch']})")
    if dirty:
        print("  **주의** 추적 파일이 수정된 상태에서 만들었다. --require-clean 을 권한다.")
    print(f"파일 {len(files)}개")
    for top, count in sorted(by_top.items()):
        print(f"  {top:<12} {count}")
    if marked_files:
        print(f"금지어 검사 면제 {len(marked_files)}개 (정리 도구. 표식을 스스로 선언했다)")
        for rel in marked_files:
            print(f"  {rel}")
    print(f"저장: {dest / 'EXPORT_MANIFEST.json'}")
    print()
    print("다음 단계")
    print("  1. 이 디렉터리에서 설치와 검증을 다시 실행한다")
    print("  2. 통과하면 이 디렉터리에서만 git init 과 공개 원격 연결을 한다")
    print("  3. private 저장소의 원격은 건드리지 않는다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
