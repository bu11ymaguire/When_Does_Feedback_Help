"""JSONL 로거 테스트.

수치 실패를 자주 기록하는 프로젝트이므로 NaN/Inf 직렬화 정책과
"실패한 run을 덮어쓰지 않는다"는 원칙(README §15)을 테스트로 고정한다.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from rl_newton.types import StepRecord
from rl_newton.utils.logging import JsonlStepLogger, sanitize_for_json


def make_record(step: int = 0, **overrides) -> StepRecord:
    defaults = {
        "run_id": "test_run",
        "seed": 0,
        "optimizer": "fixed_newton_cg",
        "step": step,
        "train_loss_before": 1.0,
        "train_loss_after": 0.8,
    }
    return StepRecord(**{**defaults, **overrides})


class TestSanitize:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats_become_null(self, value):
        assert sanitize_for_json(value) is None

    def test_finite_floats_pass_through(self):
        assert sanitize_for_json(0.125) == 0.125

    def test_nested_structures_are_processed(self):
        out = sanitize_for_json({"a": [1.0, float("nan")], "b": {"c": float("inf")}})
        assert out == {"a": [1.0, None], "b": {"c": None}}

    def test_bools_are_not_coerced_to_int(self):
        assert sanitize_for_json(True) is True


class TestJsonlStepLogger:
    def test_writes_one_valid_json_line_per_record(self, tmp_path):
        with JsonlStepLogger(tmp_path, "run_a", flush_every=1) as log:
            for i in range(3):
                log.write(make_record(step=i))

        lines = (tmp_path / "run_a.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        parsed = [json.loads(line) for line in lines]
        assert [p["step"] for p in parsed] == [0, 1, 2]

    def test_output_is_standard_json_even_with_nan_fields(self):
        """NaN / Infinity 리터럴 없이 직렬화되어야 한다 (JSON 표준 준수)."""
        record = make_record(grad_norm=float("nan"), trust_ratio=float("inf"))
        payload = json.dumps(sanitize_for_json(asdict(record)), allow_nan=False)
        assert "NaN" not in payload
        assert "Infinity" not in payload

    def test_nan_round_trips_as_none(self, tmp_path):
        with JsonlStepLogger(tmp_path, "run_nan", flush_every=1) as log:
            log.write(
                make_record(grad_norm=float("nan"), numerical_failure=True, failure_tag="nan")
            )

        line = (tmp_path / "run_nan.jsonl").read_text(encoding="utf-8").strip()
        parsed = json.loads(line)  # allow_nan 없이 파싱 가능해야 한다
        assert parsed["grad_norm"] is None
        assert parsed["numerical_failure"] is True
        assert parsed["failure_tag"] == "nan"

    def test_writes_metadata_sidecar_with_config(self, tmp_path):
        meta = {"config": {"lr": 0.01}, "git_commit": "abc123"}
        with JsonlStepLogger(tmp_path, "run_meta", metadata=meta) as log:
            log.write(make_record())

        parsed = json.loads((tmp_path / "run_meta.meta.json").read_text(encoding="utf-8"))
        assert parsed["run_id"] == "run_meta"
        assert parsed["config"] == {"lr": 0.01}
        assert parsed["git_commit"] == "abc123"
        assert "created_at" in parsed

    def test_refuses_to_overwrite_existing_run(self, tmp_path):
        """실패한 run 도 보존한다는 원칙을 코드로 강제한다."""
        with JsonlStepLogger(tmp_path, "run_dup") as log:
            log.write(make_record())

        with pytest.raises(FileExistsError, match="보존"):
            JsonlStepLogger(tmp_path, "run_dup")

    def test_overwrite_flag_allows_reuse(self, tmp_path):
        with JsonlStepLogger(tmp_path, "run_ow") as log:
            log.write(make_record())
        with JsonlStepLogger(tmp_path, "run_ow", overwrite=True) as log:
            log.write(make_record(step=99))

        lines = (tmp_path / "run_ow.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2  # append 이므로 두 줄

    def test_write_summary_merges_into_metadata(self, tmp_path):
        with JsonlStepLogger(tmp_path, "run_sum") as log:
            log.write(make_record())
            log.write_summary({"cost_to_target_ge": 412.5, "reached": True})

        parsed = json.loads((tmp_path / "run_sum.meta.json").read_text(encoding="utf-8"))
        assert parsed["summary"]["cost_to_target_ge"] == 412.5
        assert parsed["n_steps_logged"] == 1
        assert "finished_at" in parsed

    def test_rejects_invalid_flush_interval(self, tmp_path):
        with pytest.raises(ValueError, match="flush_every"):
            JsonlStepLogger(tmp_path, "run_bad", flush_every=0)

    def test_accepts_plain_dict_records(self, tmp_path):
        with JsonlStepLogger(tmp_path, "run_dict", flush_every=1) as log:
            log.write({"step": 0, "custom": 1.5})

        parsed = json.loads((tmp_path / "run_dict.jsonl").read_text(encoding="utf-8").strip())
        assert parsed == {"step": 0, "custom": 1.5}


class TestStepRecord:
    def test_rejects_unknown_failure_tag(self):
        with pytest.raises(ValueError, match="unknown failure_tag"):
            make_record(failure_tag="something_else")

    @pytest.mark.parametrize(
        "tag", ["nan", "divergence", "budget_exhausted", "cg_breakdown", "oom"]
    )
    def test_accepts_documented_failure_tags(self, tag):
        assert make_record(failure_tag=tag).failure_tag == tag
