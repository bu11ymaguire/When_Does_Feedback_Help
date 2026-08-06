"""공개 산출물에서 장치 실제 이름을 안정적 별칭으로 치환한다.

공개 전 개인정보 정리다. **실험 프로토콜 이탈이 아니다.** 설정, 결과, 통계, 실행
의미를 바꾸지 않는다.

```text
바꾼다      git 추적 대상인 summary JSON 과 cost-model YAML 의 장치 이름
안 바꾼다   raw 수치 기록. evidence map 의 SHA-256 이 raw 를 가리킨다
```

`paper/evidence_map.md` 의 checksum 은 `results/raw/*.jsonl` 을 해시한다. raw 를
건드리지 않으므로 checksum 이 그대로 유효하다.

사용법:
    python scripts/sanitize_public_artifacts.py --check
    python scripts/sanitize_public_artifacts.py --apply
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from rl_newton.benchmark.store import HOST_ID_FALLBACK

# public-export-allow-tokens: 이 파일은 장치 이름 **패턴**을 정규식과 주석에 담는다.
# 실제 장치 이름은 담지 않는다. scripts/export_public_repo.py 의 금지어 검사가 이
# 표식을 보고 면제하며, 면제 사실을 export manifest 에 기록한다.

# 치환 대상. summary JSON 은 `hostname`, cost-model YAML 은 `host` 키를 쓴다.
TARGETS = (
    Path("results/summaries"),
    Path("configs"),
)
SUFFIXES = (".json", ".yaml", ".yml")

ALIAS = "host-a"
"""공개용 별칭. 모든 Stage 2 실행이 같은 기계에서 이루어졌으므로 하나로 충분하다."""

# `"hostname": "DESKTOP-XXXX"` 와 `host: DESKTOP-XXXX` 를 모두 잡는다.
JSON_RE = re.compile(r'("hostname"\s*:\s*")([^"]*)(")')
YAML_RE = re.compile(r"^(host:\s*)(\S+)\s*$", re.MULTILINE)

# 이미 정리된 값. 다시 치환하지 않는다.
SAFE = {ALIAS, HOST_ID_FALLBACK, ""}


def sanitize(text: str) -> tuple[str, list[str]]:
    found: list[str] = []

    def json_sub(m: re.Match[str]) -> str:
        if m.group(2) in SAFE:
            return m.group(0)
        found.append(m.group(2))
        return f"{m.group(1)}{ALIAS}{m.group(3)}"

    def yaml_sub(m: re.Match[str]) -> str:
        if m.group(2) in SAFE:
            return m.group(0)
        found.append(m.group(2))
        return f"{m.group(1)}{ALIAS}"

    text = JSON_RE.sub(json_sub, text)
    text = YAML_RE.sub(yaml_sub, text)
    return text, found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="파일을 실제로 수정한다")
    parser.add_argument("--check", action="store_true", help="검사만 한다 (기본)")
    args = parser.parse_args()

    paths = [
        p
        for root in TARGETS
        if root.exists()
        for p in sorted(root.rglob("*"))
        if p.suffix in SUFFIXES
    ]

    print(f"검사 대상 {len(paths)}개  별칭 '{ALIAS}'")
    dirty: list[tuple[Path, list[str]]] = []
    for path in paths:
        original = path.read_text(encoding="utf-8")
        updated, found = sanitize(original)
        if not found:
            continue
        dirty.append((path, sorted(set(found))))
        if args.apply:
            path.write_text(updated, encoding="utf-8")

    if not dirty:
        print("정리할 값이 없다. 모든 공개 산출물이 이미 별칭을 쓴다.")
        return 0

    for path, values in dirty:
        action = "치환" if args.apply else "발견"
        print(f"  {action}  {path}  {values}")
    print()
    if args.apply:
        print(f"{len(dirty)}개 파일을 치환했다.")
        print("raw 수치 기록은 건드리지 않았다. evidence map 의 SHA-256 이 유효하다.")
        return 0
    print(f"{len(dirty)}개 파일에 장치 이름이 남아 있다. --apply 로 치환한다.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
