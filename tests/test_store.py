"""재개 가능한 결과 저장소 테스트.

프로세스가 끊겨도 계산한 결과를 잃지 않아야 한다. Stage 2는 수백~수천 run 이라
이 성질이 없으면 실험을 완주할 수 없다.

가장 위험한 실패 모드는 재개 기능 자체가 **낡은 결과를 새 결과로 착각**하는
것이다. 설정이 바뀌었는데 같은 조합으로 보고 건너뛰면, 재개가 실험을 오염시킨다.
그래서 실험 정체성 테스트를 가장 중요하게 다룬다.
"""

from __future__ import annotations

import json
import math

import pytest

from rl_newton.benchmark.metrics import RunSummary
from rl_newton.benchmark.store import (
    OPTIMIZER_SEMANTICS_VERSION,
    PLANNER_SEMANTICS_VERSION,
    SELECTION_SEMANTICS_VERSION,
    ResultStore,
    RunKey,
    RunRecord,
    environment_fingerprint,
    experiment_id,
    run_semantics_id,
)

EXP = "exp0001"


def make_summary(
    *,
    controller: str = "best_static",
    instance: str = "quad_spd_d64_k1e+02_seed0",
    seed: int = 0,
    target: str = "relative_loss<=0.0001",
    reached: bool = True,
    cost: float | None = 123.5,
) -> RunSummary:
    return RunSummary(
        run_id=f"{controller}|{instance}",
        controller=controller,
        task_instance_id=instance,
        seed=seed,
        target=target,
        reached=reached,
        cost_to_target_ge=cost,
        steps_to_target=7 if reached else None,
        hvp_to_target=90 if reached else None,
        initial_loss=1.0,
        final_loss=1.0e-5,
        total_cost_ge=600.0,
        total_hvp=520,
        search_cost_ge=0.0,
        n_steps=30,
        stop_reason="cost_budget",
        rejection_rate=0.0,
        failure_rate=0.0,
        negative_curvature_rate=0.0,
        cg_convergence_rate=0.1,
        median_residual_ratio=0.2,
        median_damping=1.0e-2,
        median_trust_ratio=1.0,
    )


# ---------------------------------------------------------------------------
# 실험 정체성: 재개가 실험을 오염시키지 않도록 하는 핵심 장치
# ---------------------------------------------------------------------------


class TestExperimentIdentity:
    def test_same_payload_gives_same_id(self):
        payload = {"beam": 2, "budget": 600.0, "cg_budgets": [3, 5, 10, 20]}
        assert experiment_id(payload) == experiment_id(dict(payload))

    def test_key_order_does_not_matter(self):
        a = {"beam": 2, "budget": 600.0}
        b = {"budget": 600.0, "beam": 2}
        assert experiment_id(a) == experiment_id(b)

    @pytest.mark.parametrize(
        "change",
        [
            {"beam": 4},
            {"budget": 1200.0},
            {"cg_budgets": [5, 20]},
            {"horizons": [1, 3]},
            {"damping_values": [0.5, 1.0, 2.0]},
            {"max_steps": 400},
            {"protocol_version": "stage2-v2"},
            {"code_dirty": True},
        ],
    )
    def test_any_config_change_gives_new_id(self, change):
        """설정이 하나라도 바뀌면 다른 실험이다.

        ``(controller, task, seed, target)`` 만으로 완료를 판단하면 beam, horizon,
        GE 예산, action space, CG budget 중 무엇이 바뀌어도 낡은 결과를 재사용한다.
        """
        base = {
            "beam": 2,
            "budget": 600.0,
            "cg_budgets": [3, 5, 10, 20],
            "horizons": [1, 3, 5],
            "damping_values": [1 / 3, 1.0, 3.0],
            "max_steps": 200,
            "protocol_version": "stage2-v1",
            "code_dirty": False,
        }
        assert experiment_id(base) != experiment_id(base | change)

    def test_key_includes_experiment_id(self):
        a = RunKey("expA", "fixed", "inst", 0, "t")
        b = RunKey("expB", "fixed", "inst", 0, "t")
        assert a.as_str() != b.as_str()
        assert a.as_str().startswith("expA|")

    def test_completed_run_is_not_skipped_after_config_change(self, tmp_path):
        """가장 위험한 회귀. 설정이 바뀌면 반드시 다시 실행되어야 한다."""
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        summary = make_summary()
        store.record_success(summary, "exp_beam2", wall_clock_sec=1.0)

        old_key = RunKey.from_summary(summary, "exp_beam2")
        new_key = RunKey.from_summary(summary, "exp_beam4")

        assert store.is_completed(old_key)
        assert not store.is_completed(new_key)


class TestRunKey:
    def test_key_string_is_stable_and_unique(self):
        a = RunKey(EXP, "fixed", "inst", 0, "t")
        b = RunKey(EXP, "fixed", "inst", 0, "t")
        c = RunKey(EXP, "fixed", "inst", 1, "t")

        assert a.as_str() == b.as_str()
        assert a.as_str() != c.as_str()

    def test_key_from_summary_round_trips(self):
        summary = make_summary()
        key = RunKey.from_summary(summary, EXP)

        assert key.experiment_id == EXP
        assert key.controller == summary.controller
        assert key.task_instance_id == summary.task_instance_id
        assert key.seed == summary.seed
        assert key.target == summary.target


class TestResume:
    def test_completed_run_is_skipped_on_reopen(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        summary = make_summary()
        store.record_success(summary, EXP, wall_clock_sec=1.5)

        reopened = ResultStore(path)
        assert reopened.is_completed(RunKey.from_summary(summary, EXP))
        assert len(reopened) == 1

    def test_no_duplicate_records_for_same_key(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        summary = make_summary()
        store.record_success(summary, EXP, wall_clock_sec=1.0)
        store.record_success(summary, EXP, wall_clock_sec=2.0)

        reopened = ResultStore(path)
        assert len(reopened) == 1  # 인덱스는 마지막 것만 유지
        assert len(reopened.summaries()) == 1

    def test_failed_run_is_retried(self, tmp_path):
        """실패는 건너뛰지 않는다. 원인은 보존한다 (README §15)."""
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        key = RunKey(EXP, "mpc_H3_narrow", "inst", 0, "t")
        store.record_failure(key, "RuntimeError: boom")

        reopened = ResultStore(path)
        assert not reopened.is_completed(key)
        failures = reopened.failures()
        assert len(failures) == 1
        assert "boom" in str(failures[0].error)

    def test_failed_run_is_excluded_from_summaries(self, tmp_path):
        """실패가 평균에 섞이면 안 된다."""
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(), EXP, wall_clock_sec=1.0)
        store.record_failure(RunKey(EXP, "other", "inst", 1, "t"), "err")

        assert len(store.summaries()) == 1
        assert len(ResultStore(path).summaries()) == 1

    def test_retry_after_failure_marks_completed(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        summary = make_summary()
        key = RunKey.from_summary(summary, EXP)

        store.record_failure(key, "transient")
        assert not store.is_completed(key)

        store.record_success(summary, EXP, wall_clock_sec=2.0)
        assert store.is_completed(key)
        assert ResultStore(path).is_completed(key)

    def test_summary_survives_round_trip(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        original = make_summary()
        ResultStore(path).record_success(original, EXP, wall_clock_sec=3.0)

        restored = ResultStore(path).summaries()[0]

        assert restored.controller == original.controller
        assert restored.reached is True
        assert restored.cost_to_target_ge == pytest.approx(123.5)
        assert restored.total_cost_ge == pytest.approx(600.0)
        assert restored.n_steps == 30
        assert restored.stop_reason == "cost_budget"

    def test_unreached_run_keeps_none_cost(self, tmp_path):
        """절단 규칙: 미도달은 큰 값으로도, NaN 으로도 대입되지 않는다 (D6).

        ``None`` 은 "목표에 도달하지 못했다", NaN 은 "값을 모른다"다. 둘을
        섞으면 도달률과 cost-to-target 집계가 오염된다.
        """
        path = tmp_path / "runs.jsonl"
        summary = make_summary(reached=False, cost=None)
        ResultStore(path).record_success(summary, EXP, wall_clock_sec=1.0)

        restored = ResultStore(path).summaries()[0]
        assert restored.reached is False
        assert restored.cost_to_target_ge is None
        assert restored.steps_to_target is None

    def test_non_finite_floats_round_trip_as_nan(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        summary = make_summary()
        summary.median_trust_ratio = float("nan")
        ResultStore(path).record_success(summary, EXP, wall_clock_sec=float("nan"))

        restored = ResultStore(path).summaries()[0]
        assert math.isnan(restored.median_trust_ratio)

    def test_output_is_valid_json_per_line(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(), EXP, wall_clock_sec=1.0)
        store.record_failure(RunKey(EXP, "x", "i", 1, "t"), "err")

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # allow_nan 없이 파싱 가능해야 한다

    def test_corrupt_trailing_line_is_skipped(self, tmp_path):
        """프로세스가 쓰는 중 끊기면 마지막 줄이 깨질 수 있다."""
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(), EXP, wall_clock_sec=1.0)
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"key": {"controller": "broken"')  # 미완성 JSON

        reopened = ResultStore(path)
        assert len(reopened) == 1
        assert reopened.is_completed(RunKey.from_summary(make_summary(), EXP))


class TestMetadata:
    def test_provenance_is_attached(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path, git_commit="abc1234", config_hash="deadbeef")
        store.record_success(make_summary(), EXP, wall_clock_sec=1.0)

        record = next(iter(ResultStore(path)))
        assert record.git_commit == "abc1234"
        assert record.config_hash == "deadbeef"
        assert record.recorded_at

    def test_action_counts_and_depths_are_preserved(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(
            make_summary(),
            EXP,
            wall_clock_sec=1.0,
            action_counts={"m=3,k=20,a=1": 5},
            chosen_depths={"1": 3, "3": 2},
        )

        record = next(iter(ResultStore(path)))
        assert record.action_counts == {"m=3,k=20,a=1": 5}
        assert record.chosen_depths == {"1": 3, "3": 2}

    def test_wall_clock_is_recorded_separately_from_ge(self, tmp_path):
        """wall-clock 은 GE 와 별개로 기록한다 (프로토콜 D1)."""
        path = tmp_path / "runs.jsonl"
        ResultStore(path).record_success(make_summary(), EXP, wall_clock_sec=12.75)

        record = next(iter(ResultStore(path)))
        assert record.wall_clock_sec == pytest.approx(12.75)
        assert record.summary is not None
        assert record.summary.total_cost_ge == pytest.approx(600.0)

    def test_filter_summaries_by_controller(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(controller="a"), EXP, wall_clock_sec=1.0)
        store.record_success(make_summary(controller="b"), EXP, wall_clock_sec=1.0)

        assert len(store.summaries()) == 2
        assert len(store.summaries(controller="a")) == 1
        assert store.controllers() == ["a", "b"]

    def test_describe_reports_counts(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        store = ResultStore(path)
        store.record_success(make_summary(), EXP, wall_clock_sec=1.0)
        store.record_failure(RunKey(EXP, "x", "i", 1, "t"), "err")

        text = store.describe()
        assert "완료 1" in text
        assert "실패 1" in text


class TestEnvironmentFingerprint:
    def test_pins_threads_and_records_them(self):
        """wall-clock tie-break 가 CPU 경쟁에 흔들리지 않게 고정한다."""
        info = environment_fingerprint(pin_threads=1)

        assert info["torch_num_threads"] == 1
        assert info["pinned"] == 1
        assert info["cpu_count"]
        assert "platform" in info

    def test_can_skip_pinning(self):
        info = environment_fingerprint(pin_threads=None)
        assert info["pinned"] is None


class TestRunRecordSerialization:
    def test_failure_record_has_no_summary(self):
        record = RunRecord(
            key=RunKey(EXP, "c", "i", 0, "t"), status="failed", error="ValueError: x"
        )
        payload = record.to_json()

        assert payload["summary"] is None
        assert payload["status"] == "failed"

        restored = RunRecord.from_json(payload)
        assert restored.summary is None
        assert restored.error == "ValueError: x"
        assert restored.key.experiment_id == EXP


# ---------------------------------------------------------------------------
# 3계층 정체성 (프로토콜 D13)
# ---------------------------------------------------------------------------


class TestThreeLayerIdentity:
    """run semantics / sweep coverage / aggregation 을 분리했는지 검증한다.

    집계 코드만 바꿨는데 ``experiment_id`` 가 갈려 423 run 이 다시 돌았다.
    무관한 변경이 고비용 optimizer run 을 무효화하면 안 된다.
    """

    def _config(self, **kwargs):
        from rl_newton.benchmark.metrics import TargetSpec
        from rl_newton.benchmark.oracle import HeadroomConfig
        from rl_newton.tasks.quadratics import QuadraticSpec

        params = {
            "specs": (QuadraticSpec(dimension=32, condition_number=1.0e3),),
            "seeds": (0, 1, 2),
            "targets": {
                "quadratic": {
                    "easy": TargetSpec("relative_loss", 1.0e-2),
                    "medium": TargetSpec("relative_loss", 1.0e-4),
                    "hard": TargetSpec("relative_loss", 1.0e-6),
                }
            },
            "cost_budget_ge": 150.0,
            "quotas": (1.0, 4.0),
            "beam_width": 4,
        }
        params.update(kwargs)
        return HeadroomConfig(**params)

    def _sem(self, config, controller, **kwargs):
        from rl_newton.benchmark.store import run_semantics_id

        return run_semantics_id(
            config.run_semantics_payload(controller=controller, **kwargs)
        )

    def test_sweep_coverage_change_preserves_run_semantics(self):
        """``fresh_diagnostic_seeds`` 만 바꾸면 baseline / planner ID 가 유지된다."""
        from rl_newton.optimizers.action_space import NARROW

        space = NARROW.with_fixed_step_size(1.0)
        a = self._config(fresh_diagnostic_seeds=1)
        b = self._config(fresh_diagnostic_seeds=3, run_fresh_wide=True)

        for controller, extra in (
            ("best_static", {}),
            ("heuristic", {}),
            ("onestep", {}),
        ):
            assert self._sem(a, controller, space=space, **extra) == self._sem(
                b, controller, space=space, **extra
            )
        assert self._sem(a, "budgeted_mpc", space=space, quota=4.0) == self._sem(
            b, "budgeted_mpc", space=space, quota=4.0
        )

    def test_beam_change_preserves_non_planner_ids(self):
        """beam 을 바꾸면 planner 만 달라지고 static / heuristic / C0 는 유지된다."""
        from rl_newton.optimizers.action_space import NARROW

        space = NARROW.with_fixed_step_size(1.0)
        a = self._config(beam_width=4)
        b = self._config(beam_width=8)

        for controller in ("best_static", "heuristic", "onestep"):
            assert self._sem(a, controller, space=space) == self._sem(
                b, controller, space=space
            )
        assert self._sem(a, "budgeted_mpc", space=space, quota=4.0) != self._sem(
            b, "budgeted_mpc", space=space, quota=4.0
        )

    def test_aggregation_change_does_not_touch_run_ids(self):
        """집계 규칙이 바뀌면 ``aggregation_id`` 만 달라진다."""
        from rl_newton.benchmark.store import aggregation_id

        config = self._config()
        base = config.aggregation_payload()
        changed = dict(base)
        changed["relative_loss_floor"] = 1.0e-12
        assert aggregation_id(base) != aggregation_id(changed)
        # run semantics payload 에는 집계 항목이 없다.
        payload = config.run_semantics_payload(controller="best_static")
        assert "relative_loss_floor" not in payload
        assert "aggregation_version" not in payload

    def test_optimizer_semantics_change_invalidates_all(self):
        """실행 의미가 바뀌면 모든 컨트롤러 ID 가 달라져야 한다."""
        config = self._config()
        payload = config.run_semantics_payload(controller="best_static")
        assert payload["optimizer_semantics"] == OPTIMIZER_SEMANTICS_VERSION
        bumped = dict(payload)
        bumped["optimizer_semantics"] = OPTIMIZER_SEMANTICS_VERSION + 1
        assert run_semantics_id(payload) != run_semantics_id(bumped)

    def test_planner_semantics_only_in_planner_payload(self):
        """planner 의미 버전은 planner 계열에만 들어간다."""
        config = self._config()
        static = config.run_semantics_payload(controller="best_static")
        planner = config.run_semantics_payload(controller="budgeted_mpc", quota=4.0)
        assert "planner_semantics" not in static
        assert planner["planner_semantics"] == PLANNER_SEMANTICS_VERSION

    def test_track_e_excludes_target_track_t_includes_it(self):
        config = self._config()
        track_e = config.run_semantics_payload(controller="budgeted_mpc", quota=4.0)
        track_t = config.run_semantics_payload(
            controller="budgeted_mpc", quota=4.0, uses_target=True
        )
        assert "targets" not in track_e
        assert "targets" in track_t
        assert run_semantics_id(track_e) != run_semantics_id(track_t)

    def test_serialization_order_and_defaults_do_not_matter(self):
        config = self._config()
        payload = config.run_semantics_payload(controller="best_static")
        shuffled = dict(reversed(list(payload.items())))
        assert run_semantics_id(payload) == run_semantics_id(shuffled)

    def test_different_effective_configs_never_collide(self):
        """실제로 다른 설정이 같은 ID 를 만들면 안 된다."""
        from rl_newton.optimizers.action_space import NARROW, WIDE

        narrow = NARROW.with_fixed_step_size(1.0)
        wide = WIDE.with_fixed_step_size(1.0)
        config = self._config()
        ids = {
            self._sem(config, "best_static", space=narrow),
            self._sem(config, "best_static", space=wide),
            self._sem(config, "heuristic", space=narrow),
            self._sem(config, "onestep", space=narrow),
            self._sem(config, "budgeted_mpc", space=narrow, quota=1.0),
            self._sem(config, "budgeted_mpc", space=narrow, quota=4.0),
            self._sem(config, "budgeted_mpc", space=wide, quota=4.0),
            self._sem(
                config,
                "budgeted_mpc",
                space=narrow,
                quota=4.0,
                extra={"execution_mode": "committed"},
            ),
            self._sem(
                config,
                "budgeted_mpc",
                space=narrow,
                quota=4.0,
                extra={"execution_mode": "shrinking"},
            ),
        }
        assert len(ids) == 9

    def test_code_dirty_is_provenance_only(self):
        """git dirty 는 **어떤 ID 에도** 들어가지 않는다 (프로토콜 D13).

        ``sweep_id`` 가 "어떤 run 집합을 요청했는가" 를 뜻한다면, 문서만 수정해도
        ID 가 달라지는 것은 의미가 어긋난다.
        """
        from rl_newton.benchmark.store import execution_provenance

        config = self._config()
        assert "code_dirty" not in config.run_semantics_payload(controller="best_static")
        assert "code_dirty" not in config.sweep_payload(controllers=["best_static"])
        assert "code_dirty" not in config.aggregation_payload()
        prov = execution_provenance(git_commit="abc123", code_dirty=True)
        assert prov["code_dirty"] is True
        assert prov["git_commit"] == "abc123"

    def test_same_request_gives_same_sweep_id_across_commits(self):
        """같은 run 집합 요청이면 commit 이 달라도 ``sweep_id`` 가 같다."""
        from rl_newton.benchmark.store import sweep_id

        config = self._config()
        a = sweep_id(config.sweep_payload(controllers=["best_static", "onestep"]))
        b = sweep_id(config.sweep_payload(controllers=["onestep", "best_static"]))
        assert a == b

    def test_same_semantics_run_shared_across_sweeps(self):
        """같은 semantics run 은 sweep 이 달라도 한 번만 실행되고 양쪽에서 참조된다."""
        from rl_newton.benchmark.store import sweep_id

        a = self._config(fresh_diagnostic_seeds=1)
        b = self._config(fresh_diagnostic_seeds=3)
        sweep_a = sweep_id(a.sweep_payload(controllers=["best_static"]))
        sweep_b = sweep_id(b.sweep_payload(controllers=["best_static"]))
        assert sweep_a != sweep_b
        assert self._sem(a, "best_static") == self._sem(b, "best_static")


# ---------------------------------------------------------------------------
# baseline 선택 manifest (프로토콜 D16)
# ---------------------------------------------------------------------------


class TestSelectionManifest:
    """``best_static`` / ``best_open_loop`` 는 컨트롤러가 아니라 튜닝 결과다.

    라벨만 ``static[7]`` -> ``best_static`` 으로 바꾸면 어떤 설정이 왜 선택됐는지
    사라진다. evaluation 결과를 보고 역추정하면 사후 선택이다.
    """

    def _manifest(self, **kwargs):
        from rl_newton.benchmark.store import SelectionManifest

        params = {
            "selection_id": "sel0001",
            "family": "static",
            "candidate_labels": ["static[0]", "static[3]", "static[7]"],
            "candidate_scores": {"static[0]": 6.8, "static[3]": 9.3, "static[7]": 9.3},
            "selected_label": "static[3]",
            "selected_config": {"flat_index": 3, "cg_budget": 5},
            "n_tune": 3,
        }
        params.update(kwargs)
        return SelectionManifest(**params)

    def test_manifest_records_candidates_and_tie_break(self):
        m = self._manifest()
        assert m.resolved
        assert len(m.candidate_labels) == 3
        assert m.selected_label in m.candidate_scores
        assert m.tie_break_rule == "lowest_flat_index"
        # 동률이면 낮은 인덱스. static[3] 과 static[7] 이 같은 점수다.
        assert m.candidate_scores["static[3]"] == m.candidate_scores["static[7]"]
        assert m.selected_label == "static[3]"

    def test_manifest_roundtrips_to_json(self):
        m = self._manifest()
        payload = m.to_json()
        assert payload["selected_label"] == "static[3]"
        assert payload["selected_config"]["flat_index"] == 3
        assert payload["semantics_version"] == SELECTION_SEMANTICS_VERSION

    def test_constant_open_loop_schedule_is_detected(self):
        """모든 구간 action 이 같으면 open-loop 이 static 으로 퇴화한 것이다."""
        action = {"flat_index": 4, "cg_budget": 5, "step_size": 1.0}
        m = self._manifest(
            family="open_loop",
            selected_label="open_loop[4]",
            selected_config={"schedule": [action, action, action, action]},
        )
        assert m.is_constant_schedule

    def test_varying_open_loop_schedule_is_not_constant(self):
        m = self._manifest(
            family="open_loop",
            selected_label="open_loop[4]",
            selected_config={
                "schedule": [
                    {"flat_index": 4, "cg_budget": 5},
                    {"flat_index": 9, "cg_budget": 20},
                ]
            },
        )
        assert not m.is_constant_schedule

    def test_static_family_is_never_constant_schedule(self):
        assert not self._manifest().is_constant_schedule

    def test_unresolved_manifest_is_marked(self):
        """튜닝 기록이 없으면 추측하지 않고 legacy_unresolved 로 둔다."""
        m = self._manifest(resolved=False, candidate_scores={}, selected_label="")
        assert not m.resolved
        assert "legacy_unresolved" in m.describe()

    def test_selection_id_changes_with_tuning_scope(self):
        """튜닝 범위가 달라지면 선택 정체성도 달라진다."""
        from rl_newton.benchmark.store import selection_id
        from rl_newton.optimizers.action_space import NARROW

        space = NARROW.with_fixed_step_size(1.0)
        base = {
            "selection_family": "static",
            "selection_semantics_version": SELECTION_SEMANTICS_VERSION,
            "n_tune": 12,
            "tuning_seeds": [0, 1, 2],
            "space": {"cg_budgets": list(space.cg_budgets)},
        }
        changed = dict(base, n_tune=24)
        assert selection_id(base) != selection_id(changed)


class TestSweepCoverageCli:
    """``--modes`` 생략과 빈 목록을 구별해야 한다 (프로토콜 D13).

    재현 명령을 잘못 입력해 planner 전체가 조용히 빠지는 사고를 막는다.
    """

    def _parse(self, argv):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import run_headroom

        parser = run_headroom.build_parser()
        return parser.parse_args(argv)

    def test_omitted_modes_gives_all_three(self):
        args = self._parse([])
        assert set(args.modes) == {"shrinking", "committed", "fresh"}

    def test_empty_modes_disables_planner(self):
        args = self._parse(["--modes"])
        assert args.modes == []

    def test_explicit_subset_is_respected(self):
        args = self._parse(["--modes", "shrinking", "committed"])
        assert args.modes == ["shrinking", "committed"]

    def test_planner_spaces_default_and_subset(self):
        assert set(self._parse([]).planner_spaces) == {"narrow", "wide"}
        assert self._parse(["--planner-spaces", "narrow"]).planner_spaces == ["narrow"]

    def test_track_t_default_on(self):
        assert not self._parse([]).skip_track_t
        assert self._parse(["--skip-track-t"]).skip_track_t

    def test_coverage_options_are_sweep_only(self):
        """커버리지 옵션은 run 정체성에 들어가지 않는다."""
        helper = TestThreeLayerIdentity()
        a = helper._config(execution_modes=("shrinking",), run_track_t=False)
        b = helper._config(
            execution_modes=("shrinking", "committed", "fresh"), run_track_t=True
        )
        assert run_semantics_id(
            a.run_semantics_payload(controller="best_static")
        ) == run_semantics_id(b.run_semantics_payload(controller="best_static"))
        assert "execution_modes" not in a.run_semantics_payload(controller="best_static")


class TestChallengeSetFreeze:
    """Challenge set 과 seed 역할을 실행 전에 고정한다 (프로토콜 D20).

    이 목록을 조용히 바꾸면 "사전 등록" 이 무의미해진다.
    """

    def _module(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import run_headroom

        return run_headroom

    def test_selection_set_is_four_quadratics_even_in_log_kappa(self):
        import math

        specs = self._module().challenge_specs()
        assert len(specs) == 4
        assert {s.kind for s in specs} == {"ill_conditioned"}
        assert {s.dimension for s in specs} == {100}
        logs = sorted(math.log10(s.condition_number) for s in specs)
        assert logs == [3.0, 4.0, 5.0, 6.0]
        gaps = [b - a for a, b in zip(logs[:-1], logs[1:], strict=True)]
        assert gaps == [1.0, 1.0, 1.0], "tie-break 은 log10(κ) 균등 포괄이다"

    def test_nonlinear_diagnostic_is_separate_from_selection(self):
        mod = self._module()
        diagnostic = mod.nonlinear_diagnostic_specs()
        assert len(diagnostic) == 1
        assert diagnostic[0].dimension == 5
        # 설정 선택 점수에 섞이면 안 된다.
        assert all(d not in mod.challenge_specs() for d in diagnostic)

    def test_calibration_and_selection_seeds_are_disjoint(self):
        from rl_newton.benchmark.oracle import (
            CALIBRATION_SEEDS,
            HELD_OUT_SEEDS,
            SELECTION_SEEDS,
        )

        assert set(CALIBRATION_SEEDS).isdisjoint(SELECTION_SEEDS)
        assert set(CALIBRATION_SEEDS).isdisjoint(HELD_OUT_SEEDS)
        assert set(SELECTION_SEEDS).isdisjoint(HELD_OUT_SEEDS)

    def test_challenge_mode_uses_selection_seeds(self):
        mod = self._module()
        args = mod.build_parser().parse_args(["--mode", "challenge", "--seeds", "3"])
        config, meta = mod.build_config(args)
        meta.pop("spaces")
        assert config.phase == "challenge"
        assert list(config.seeds) == [2, 3, 4]
        assert list(config.specs) == mod.challenge_specs()

    def test_diagnostic_mode_carries_only_rosenbrock(self):
        mod = self._module()
        args = mod.build_parser().parse_args(["--mode", "nonlinear-diagnostic"])
        config, meta = mod.build_config(args)
        meta.pop("spaces")
        assert list(config.specs) == mod.nonlinear_diagnostic_specs()

    def test_challenge_and_pilot_are_different_sweeps(self):
        from rl_newton.benchmark.store import sweep_id

        mod = self._module()
        ids = set()
        for mode in ("pilot", "challenge", "nonlinear-diagnostic"):
            args = mod.build_parser().parse_args(["--mode", mode])
            config, meta = mod.build_config(args)
            meta.pop("spaces")
            ids.add(sweep_id(config.sweep_payload(controllers=["heuristic"])))
        assert len(ids) == 3

    def test_phase_does_not_change_run_semantics(self):
        """phase 는 sweep 라벨이다. 저장된 run 을 무효화하면 안 된다."""
        helper = TestThreeLayerIdentity()
        a = helper._config(phase="pilot")
        b = helper._config(phase="challenge")
        assert run_semantics_id(
            a.run_semantics_payload(controller="best_static")
        ) == run_semantics_id(b.run_semantics_payload(controller="best_static"))


class TestRealizedSegmentReporting:
    """구간 미계측(캐시 재사용)과 구간 미실행을 구별해야 한다 (프로토콜 D17)."""

    def _manifest(self, **kwargs):
        from rl_newton.benchmark.store import SelectionManifest

        action = {"flat_index": 4, "cg_budget": 5}
        params = {
            "selection_id": "sel0002",
            "family": "open_loop",
            "selected_label": "open_loop[7]",
            "selected_config": {"schedule": [action, action, action, action]},
            "progress_clock": "object_ge_fraction",
        }
        params.update(kwargs)
        return SelectionManifest(**params)

    def test_uncounted_is_not_reported_as_unused(self):
        text = self._manifest(realized_segment_counts={}).describe()
        assert "미계측" in text
        assert "실행되지 않았다" not in text

    def test_partial_usage_is_warned(self):
        text = self._manifest(realized_segment_counts={"0": 4, "1": 20}).describe()
        assert "실행된 구간 2/4" in text
        assert "실행되지 않았다" in text

    def test_full_usage_has_no_warning(self):
        text = self._manifest(
            realized_segment_counts={"0": 4, "1": 20, "2": 1, "3": 11}
        ).describe()
        assert "실행된 구간 4/4" in text
        assert "실행되지 않았다" not in text
        assert "36 step" in text
