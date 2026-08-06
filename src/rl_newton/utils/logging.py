"""step 단위 JSONL 로거.

한 run = 두 개의 파일.

``<run_id>.meta.json``   config 스냅샷 + provenance (1회 기록)
``<run_id>.jsonl``       optimizer step마다 한 줄

NaN / Inf 처리
--------------
``json.dumps(allow_nan=True)`` 는 ``NaN`` / ``Infinity`` 리터럴을 뱉는데 이는
JSON 표준이 아니다. pandas는 읽을 수 있지만 다른 도구에서 깨진다. 이 프로젝트는
수치 실패를 자주 기록하므로 **비유한값은 ``null`` 로 직렬화**하고, 원래 실패
사실은 ``numerical_failure`` / ``failure_tag`` 필드로 따로 남긴다.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from rl_newton.types import StepRecord

__all__ = ["JsonlStepLogger", "sanitize_for_json"]


def sanitize_for_json(value: Any) -> Any:
    """JSON 표준에 맞게 값을 변환한다.

    - 비유한 float (``NaN``, ``inf``, ``-inf``) -> ``None``
    - dict / list / tuple -> 재귀 처리
    - dataclass -> dict
    - Path -> str
    - 그 외 JSON 비호환 객체 -> ``repr``
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [sanitize_for_json(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return sanitize_for_json(asdict(value))
    if isinstance(value, Path):
        return str(value)
    return repr(value)


class JsonlStepLogger:
    """append-only JSONL 로거.

    Args:
        output_dir: 로그를 쓸 디렉터리. 없으면 만든다.
        run_id: 파일 이름 접두사. run 식별자.
        metadata: config 스냅샷과 provenance. ``<run_id>.meta.json`` 에 기록된다.
        flush_every: 이 개수만큼 쓸 때마다 디스크로 flush한다. 크래시 시
            손실을 제한한다. 1로 두면 매 step flush (느리지만 가장 안전).
        overwrite: ``False`` 이면 기존 파일이 있을 때 ``FileExistsError``.
            실패한 run을 덮어쓰지 않는다는 원칙(README §15)을 강제한다.

    Example:
        >>> import tempfile
        >>> from rl_newton.types import StepRecord
        >>> with tempfile.TemporaryDirectory() as d:
        ...     with JsonlStepLogger(d, "demo") as log:
        ...         log.write(StepRecord(run_id="demo", seed=0, optimizer="fixed",
        ...                              step=0, train_loss_before=1.0,
        ...                              train_loss_after=0.9))
        ...     print(log.n_written)
        1
    """

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        flush_every: int = 20,
        overwrite: bool = False,
    ) -> None:
        if flush_every < 1:
            raise ValueError(f"flush_every must be >= 1, got {flush_every}")

        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.flush_every = flush_every
        self._n_written = 0
        self._handle = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.output_dir / f"{run_id}.jsonl"
        self.meta_path = self.output_dir / f"{run_id}.meta.json"

        if not overwrite and self.steps_path.exists():
            raise FileExistsError(
                f"{self.steps_path} already exists. 실패한 run도 보존한다는 원칙에 따라 "
                "덮어쓰지 않는다. run_id 를 바꾸거나 overwrite=True 를 명시하라."
            )

        meta = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            **(metadata or {}),
        }
        self.meta_path.write_text(
            json.dumps(sanitize_for_json(meta), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # --- 컨텍스트 관리 ----------------------------------------------------

    def __enter__(self) -> JsonlStepLogger:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def open(self) -> None:
        if self._handle is None:
            self._handle = self.steps_path.open("a", encoding="utf-8", newline="\n")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    # --- 쓰기 -------------------------------------------------------------

    @property
    def n_written(self) -> int:
        """지금까지 쓴 레코드 수."""
        return self._n_written

    def write(self, record: StepRecord | dict[str, Any]) -> None:
        """레코드 한 개를 JSONL 한 줄로 쓴다."""
        if self._handle is None:
            self.open()
        assert self._handle is not None  # open() 이 보장

        payload = asdict(record) if isinstance(record, StepRecord) else dict(record)
        line = json.dumps(sanitize_for_json(payload), ensure_ascii=False, separators=(",", ":"))
        self._handle.write(line + "\n")
        self._n_written += 1

        if self._n_written % self.flush_every == 0:
            self._handle.flush()

    def write_summary(self, summary: dict[str, Any]) -> None:
        """run 종료 후 집계 결과를 meta 파일에 병합한다."""
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        meta["summary"] = sanitize_for_json(summary)
        meta["finished_at"] = datetime.now(UTC).isoformat()
        meta["n_steps_logged"] = self._n_written
        self.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def __repr__(self) -> str:
        return f"JsonlStepLogger(run_id={self.run_id!r}, n_written={self._n_written})"
