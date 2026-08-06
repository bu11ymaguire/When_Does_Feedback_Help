"""노트북이 공개 가능한 상태인지 검사한다.

이 환경에는 jupyter 도구가 없어 노트북을 **실행**하지 못한다. 그래서 실행 없이 확인할
수 있는 것만 본다.

```text
JSON 이 유효한가
모든 코드 셀이 컴파일되는가
저장된 출력이 없는가          공개 저장소에 실행 흔적을 남기지 않는다
로컬 절대경로가 없는가         다른 기계에서 돌지 않는 셀을 막는다
패키지 함수를 쓰는가           통계 구현을 노트북에 복사하지 않는다
전체 planner 를 돌리지 않는가   기본 실행이 몇 분을 넘으면 안 된다
```

실제 실행 검증은 jupyter 가 있는 환경에서 한 번 해야 한다. `docs/reproduce.md` 참조.

public-export-allow-tokens: 이 파일은 로컬 경로 **패턴**을 검사 대상으로 담는다.
실제 장치 이름이나 사용자 경로는 담지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "overview_and_reproduction.ipynb"
)

pytestmark = pytest.mark.skipif(not NOTEBOOK.exists(), reason="노트북이 없다")


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_cells(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


class TestStructure:
    def test_is_nbformat_4(self, nb: dict):
        assert nb["nbformat"] == 4
        assert isinstance(nb["cells"], list)
        assert nb["cells"], "셀이 없다"

    def test_has_prose_and_code(self, nb: dict):
        kinds = {c["cell_type"] for c in nb["cells"]}
        assert "markdown" in kinds
        assert "code" in kinds

    def test_every_code_cell_compiles(self, code_cells: list[str]):
        for i, source in enumerate(code_cells, start=1):
            compile(source, f"<notebook cell {i}>", "exec")


class TestPublishable:
    def test_no_stored_outputs(self, nb: dict):
        """실행 흔적을 커밋하지 않는다. 독자가 스스로 돌려야 한다."""
        for i, cell in enumerate(nb["cells"], start=1):
            if cell["cell_type"] != "code":
                continue
            assert not cell.get("outputs"), f"셀 {i} 에 저장된 출력이 있다"
            assert cell.get("execution_count") is None, f"셀 {i} 에 실행 번호가 있다"

    @pytest.mark.parametrize("token", ["C:\\Users", "/Users/", "OneDrive", "DESKTOP-"])
    def test_no_local_paths(self, code_cells: list[str], token: str):
        """절대경로가 박히면 다른 기계에서 돌지 않는다."""
        for i, source in enumerate(code_cells, start=1):
            assert token not in source, f"셀 {i} 에 로컬 경로 {token!r}"


class TestDiscipline:
    def test_uses_package_statistics(self, code_cells: list[str]):
        """통계를 노트북에 복사하지 않고 패키지에서 가져온다."""
        joined = "\n".join(code_cells)
        assert "from rl_newton.reporting import" in joined
        for name in ("paired", "positive_count", "public_roles", "load_public_results"):
            assert name in joined, f"{name} 을 쓰지 않는다"

    def test_does_not_reimplement_bootstrap(self, code_cells: list[str]):
        """부트스트랩이나 Wilcoxon 을 손으로 다시 구현하면 두 경로가 갈린다 (E10)."""
        joined = "\n".join(code_cells).lower()
        for banned in ("def bootstrap", "def wilcoxon", "np.percentile", "def _median"):
            assert banned not in joined, f"노트북이 통계를 재구현한다: {banned}"

    def test_does_not_run_the_full_planner(self, code_cells: list[str]):
        """전체 planner 스위트는 기본 실행에 넣지 않는다. 탐색 비용이 예산의 1,294배다."""
        joined = "\n".join(code_cells)
        for banned in ("run_headroom", "ShrinkingQuotaMPCController", "BudgetedMPCController"):
            assert banned not in joined, f"노트북이 전체 planner 를 돌린다: {banned}"

    def test_smoke_experiment_is_small(self, code_cells: list[str]):
        """end-to-end 예제는 작아야 한다. 차원과 예산을 확인한다."""
        joined = "\n".join(code_cells)
        assert "dimension=50" in joined
        assert "BUDGET_GE" in joined
