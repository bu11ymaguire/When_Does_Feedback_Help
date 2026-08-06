"""테스트가 **이 저장소의 `src`** 를 쓰도록 고정한다.

두 가지 상황을 같은 방식으로 처리한다.

```text
개발 중    worktree 가 여러 개면 editable 설치가 다른 트리를 가리킬 수 있다
공개 후    clone 직후 설치 없이 pytest 를 돌리는 사람이 있다
```

둘 다 "지금 보고 있는 코드가 아닌 것을 테스트한다" 는 같은 사고로 이어진다.
`sys.path` 앞에 이 저장소의 `src` 를 넣어 그것을 막는다.

**테스트 전용 변경이다.** 실험 경로에는 영향이 없다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
