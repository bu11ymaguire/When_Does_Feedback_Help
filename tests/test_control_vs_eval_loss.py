"""control loss 와 evaluation loss 분리가 **기존 결과를 바꾸지 않아야 한다** (D24).

D24 의 micro-neural regime 을 지원하려고 optimizer 에 세 가지를 바꿨다.

```text
evaluate_loss_at / _handle_rejection / _run_step 이 curvature_loss 를 쓴다
실제 step 뒤에 advance_batch() 를 호출한다 (있으면)
trace.final_loss 를 마지막 기록 대신 loss() 재평가로 채운다
```

결정론적 task 에서는 ``loss() is curvature_loss()`` 이므로 **bitwise 동일**해야
한다. 그것이 깨지면 D18 bridge 로 확정한 기존 raw 결과 전체가 무효가 된다.

`OPTIMIZER_SEMANTICS_VERSION` 을 올리지 않은 근거가 이 테스트다.
"""

from __future__ import annotations

import math

import pytest
import torch

from rl_newton.benchmark.metrics import TargetSpec, summarize_run
from rl_newton.optimizers.action_space import NARROW
from rl_newton.optimizers.controllers import (
    FixedController,
    HeuristicController,
    OneStepEfficiencyController,
    ShrinkingQuotaMPCController,
)
from rl_newton.optimizers.newton_cg import NewtonCGConfig, NewtonCGOptimizer
from rl_newton.tasks.micro_neural import MicroNeuralSpec, MicroNeuralTask
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask
from rl_newton.tasks.rosenbrock import RosenbrockSpec, RosenbrockTask

SPACE = NARROW.with_fixed_step_size(1.0)
TARGET = TargetSpec("relative_loss", 1.0e-6)


def quad(seed: int = 0, **kw):
    spec = QuadraticSpec(
        kind=kw.pop("kind", "ill_conditioned"),
        dimension=kw.pop("dimension", 20),
        condition_number=kw.pop("condition_number", 1.0e4),
    )
    return QuadraticTask(spec, seed=seed, dtype=torch.float64)


def rosen(seed: int = 0):
    return RosenbrockTask(RosenbrockSpec(dimension=5), seed=seed, dtype=torch.float64)


def micro(regime: str = "full_batch", seed: int = 0):
    spec = MicroNeuralSpec(
        input_dim=8,
        hidden_dim=16,
        n_classes=3,
        n_samples=64,
        teacher_hidden_dim=32,
        regime=regime,
        batch_size=16,
    )
    return MicroNeuralTask(spec, seed=seed, dtype=torch.float64)


def optimize(task, controller, *, budget: float = 60.0, steps: int = 40):
    config = NewtonCGConfig(
        total_steps=steps, cost_budget_ge=budget, initial_damping=1.0e-2
    )
    return NewtonCGOptimizer(task, controller, config, run_id="t", seed=0).run()


CONTROLLERS = {
    "fixed": lambda: FixedController(SPACE.action_from_flat(6)),
    "heuristic": lambda: HeuristicController(SPACE),
    "onestep": lambda: OneStepEfficiencyController(SPACE),
    "shrinking": lambda: ShrinkingQuotaMPCController(
        SPACE, quota_multiplier=4.0, beam_width=4
    ),
}


class TestDeterministicTasksUnchanged:
    """결정론적 task 에서 두 loss 가 같으므로 결과가 바뀌면 안 된다."""

    @pytest.mark.parametrize("factory", [quad, rosen])
    def test_curvature_loss_equals_loss(self, factory):
        task = factory()
        assert float(task.curvature_loss().detach()) == float(task.loss().detach())

    @pytest.mark.parametrize("name", sorted(CONTROLLERS))
    @pytest.mark.parametrize("factory", [quad, rosen])
    def test_final_loss_matches_last_record_bitwise(self, name, factory):
        """`trace.final_loss` 재평가가 마지막 기록과 bitwise 동일해야 한다.

        이것이 기존 raw 결과를 보존하는 핵심 불변식이다.
        """
        trace = optimize(factory(), CONTROLLERS[name]())
        assert trace.records
        last = trace.records[-1].train_loss_after
        assert math.isfinite(last)
        assert trace.final_loss == last, (
            f"{name}: final_loss={trace.final_loss!r} != last={last!r}"
        )

    @pytest.mark.parametrize("factory", [quad, rosen])
    def test_no_advance_batch_attribute(self, factory):
        """기존 task 에는 `advance_batch` 가 없어야 한다. 훅이 no-op 이어야 한다."""
        assert not hasattr(factory(), "advance_batch")

    def test_repeated_runs_are_bitwise_identical(self):
        a = optimize(quad(), CONTROLLERS["shrinking"]())
        b = optimize(quad(), CONTROLLERS["shrinking"]())
        assert a.final_loss == b.final_loss
        assert a.total_cost_ge == b.total_cost_ge
        assert a.search_cost_ge == b.search_cost_ge
        assert len(a.records) == len(b.records)


class TestMicroNeuralFullBatch:
    """R1 은 결정론적이므로 기존 task 와 같은 불변식을 만족해야 한다."""

    def test_curvature_loss_equals_loss(self):
        task = micro("full_batch")
        assert float(task.curvature_loss().detach()) == float(task.loss().detach())

    @pytest.mark.parametrize("name", sorted(CONTROLLERS))
    def test_final_loss_matches_last_record(self, name):
        trace = optimize(micro("full_batch"), CONTROLLERS[name]())
        assert trace.final_loss == trace.records[-1].train_loss_after

    def test_optimizer_makes_progress(self):
        trace = optimize(micro("full_batch"), CONTROLLERS["onestep"]())
        assert trace.final_loss < trace.initial_loss

    def test_batch_index_stays_zero_in_full_batch_effect(self):
        """`advance_batch` 는 호출되지만 R1 에서는 결과에 영향이 없다."""
        task = micro("full_batch")
        trace = optimize(task, CONTROLLERS["fixed"]())
        assert task.step_index == len(trace.records)
        assert trace.final_loss == trace.records[-1].train_loss_after


class TestMicroNeuralStochastic:
    """R2 에서는 control loss 와 evaluation loss 가 갈려야 한다."""

    def test_batch_advances_once_per_recorded_step(self):
        task = micro("controlled_stochastic")
        trace = optimize(task, CONTROLLERS["fixed"]())
        assert task.step_index == len(trace.records), (
            "실제 step 마다 정확히 한 번 전진해야 한다"
        )

    def test_planner_search_does_not_advance_batch(self):
        """planner 시뮬레이션이 batch 를 전진시키면 미래 데이터 oracle 이 된다."""
        task = micro("controlled_stochastic")
        trace = optimize(task, CONTROLLERS["shrinking"]())
        assert task.step_index == len(trace.records)

    def test_final_loss_is_full_dataset_not_minibatch(self):
        """Track E 가 regime 간 비교 가능해야 한다."""
        task = micro("controlled_stochastic")
        trace = optimize(task, CONTROLLERS["fixed"]())
        # 마지막 기록은 minibatch 값이므로 전체 데이터 값과 다를 것이 정상이다.
        assert trace.final_loss != trace.records[-1].train_loss_after
        assert trace.final_loss == pytest.approx(float(task.loss()), rel=0, abs=0)

    def test_initial_loss_is_full_dataset(self):
        task = micro("controlled_stochastic")
        fresh = micro("controlled_stochastic")
        trace = optimize(task, CONTROLLERS["fixed"]())
        assert trace.initial_loss == pytest.approx(
            float(fresh.loss()), rel=0, abs=0
        )

    def test_summary_is_computable(self):
        """`summarize_run` 이 R2 에서도 유한한 logΔ 를 낸다."""
        trace = optimize(micro("controlled_stochastic"), CONTROLLERS["onestep"]())
        summary = summarize_run(trace, TARGET, budget_ge=60.0)
        assert math.isfinite(summary.log_improvement)
        assert summary.failure_rate == 0.0

    def test_paired_design_holds_across_controllers(self):
        """같은 seed 면 컨트롤러가 달라도 같은 batch 시퀀스를 본다 (프로토콜 D7)."""
        seen = {}
        for name in ("fixed", "onestep"):
            task = micro("controlled_stochastic", seed=3)
            optimize(task, CONTROLLERS[name](), steps=5, budget=1.0e9)
            task.reset()
            seen[name] = [
                float(task.curvature_loss()) for _ in range(3)
            ]
        # reset 후 같은 파라미터/같은 batch 이므로 값이 일치해야 한다.
        assert seen["fixed"] == seen["onestep"]

    def test_regimes_share_the_same_problem(self):
        """모델과 데이터가 같아야 C3 의 regime 간 비교가 성립한다 (D24)."""
        a, b = micro("full_batch", seed=1), micro("controlled_stochastic", seed=1)
        assert a.instance_id != b.instance_id, "run 은 구별되어야 한다"
        assert a.initial_loss == b.initial_loss, "문제는 같아야 한다"

    def test_stochastic_is_actually_harder_to_predict(self):
        """R2 에서 control loss 가 step 마다 흔들려야 feedback 검증이 성립한다."""
        task = micro("controlled_stochastic")
        values = []
        for _ in range(8):
            values.append(float(task.curvature_loss()))
            task.advance_batch()
        assert len(set(values)) > 1


class TestFixedEvalAcceptance:
    """`fixed_eval` 수락 규칙이 기존 결과를 바꾸지 않아야 한다 (D28).

    `_accept` 는 단조 감소를 요구한다. minibatch 목적함수에서는 참 목적함수를
    개선하는 step 도 표본 잡음 때문에 거절될 수 있다. 그 교란을 분리하는
    ablation 이지만, **기본값 경로는 bitwise 보존**되어야 한다.
    """

    def _optimize(self, task, controller, *, acceptance: str):
        config = NewtonCGConfig(
            total_steps=40,
            cost_budget_ge=60.0,
            initial_damping=1.0e-2,
            acceptance_loss=acceptance,
        )
        return NewtonCGOptimizer(task, controller, config, run_id="t", seed=0).run()

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            NewtonCGConfig(acceptance_loss="full_batch")

    @pytest.mark.parametrize("factory", [quad, rosen])
    def test_deterministic_tasks_are_unaffected(self, factory):
        """결정론적 task 는 `acceptance_loss` 를 제공하지 않으므로 무시된다."""
        a = self._optimize(factory(), CONTROLLERS["shrinking"](), acceptance="control")
        b = self._optimize(
            factory(), CONTROLLERS["shrinking"](), acceptance="fixed_eval"
        )
        assert a.final_loss == b.final_loss
        assert a.total_cost_ge == b.total_cost_ge
        assert a.search_cost_ge == b.search_cost_ge
        assert len(a.records) == len(b.records)

    def test_fallback_when_task_lacks_hook(self):
        opt = NewtonCGOptimizer(
            quad(),
            CONTROLLERS["fixed"](),
            NewtonCGConfig(total_steps=3, acceptance_loss="fixed_eval"),
            run_id="t",
            seed=0,
        )
        assert not opt.uses_fixed_eval_acceptance

    def test_enabled_for_micro_neural(self):
        opt = NewtonCGOptimizer(
            micro("controlled_stochastic"),
            CONTROLLERS["fixed"](),
            NewtonCGConfig(total_steps=3, acceptance_loss="fixed_eval"),
            run_id="t",
            seed=0,
        )
        assert opt.uses_fixed_eval_acceptance

    def test_forward_count_stays_integer(self):
        """`StepRecord.forward_count` 는 호출 횟수다. 비용 단위와 섞지 않는다."""
        trace = self._optimize(
            micro("controlled_stochastic"), CONTROLLERS["fixed"](), acceptance="fixed_eval"
        )
        for record in trace.records:
            assert isinstance(record.forward_count, int)
            assert record.forward_count >= 1

    def test_acceptance_forward_cost_is_charged(self):
        """평가 forward 비용을 숨기지 않는다. 같은 step 수라면 GE 가 더 커야 한다."""
        task = micro("controlled_stochastic")
        # 테스트 헬퍼는 n_samples=64, batch_size=16 이다.
        assert task.acceptance_forward_units == pytest.approx(64 / 16)
        cheap = self._optimize(
            micro("controlled_stochastic"), CONTROLLERS["fixed"](), acceptance="control"
        )
        dear = self._optimize(
            micro("controlled_stochastic"), CONTROLLERS["fixed"](), acceptance="fixed_eval"
        )
        per_step_cheap = cheap.total_cost_ge / len(cheap.records)
        per_step_dear = dear.total_cost_ge / len(dear.records)
        assert per_step_dear > per_step_cheap

    def test_full_batch_units_are_one(self):
        assert micro("full_batch").acceptance_forward_units == 1.0

    def test_acceptance_loss_equals_full_data_loss(self):
        task = micro("controlled_stochastic")
        assert float(task.acceptance_loss().detach()) == float(task.loss().detach())
        assert float(task.acceptance_loss().detach()) != float(task.curvature_loss().detach())

    def test_gradient_still_comes_from_minibatch(self):
        """수락 판정만 바꾼다. gradient / HVP 는 계속 minibatch 다."""
        task = micro("controlled_stochastic")
        opt = NewtonCGOptimizer(
            task,
            CONTROLLERS["fixed"](),
            NewtonCGConfig(total_steps=3, acceptance_loss="fixed_eval"),
            run_id="t",
            seed=0,
        )
        assert opt.control_loss() != opt.step_objective()


class TestAcceptanceIdentity:
    """`acceptance_loss` 기본값은 `run_semantics_id` 를 바꾸지 않아야 한다 (D28)."""

    def _config(self, **kwargs):
        from rl_newton.benchmark.metrics import TargetSpec as _TargetSpec
        from rl_newton.benchmark.oracle import HeadroomConfig
        from rl_newton.tasks.quadratics import QuadraticSpec as _QSpec

        params = {
            "specs": [_QSpec(kind="spd", dimension=8, condition_number=10.0)],
            "seeds": [0],
            "targets": {"spd": {"medium": _TargetSpec("relative_loss", 1.0e-4)}},
            "cost_budget_ge": 150.0,
        }
        params.update(kwargs)
        return HeadroomConfig(**params)  # type: ignore[arg-type]

    def test_default_keeps_hash_stable(self):
        from rl_newton.benchmark.store import run_semantics_id

        base = self._config()
        payload = base.run_semantics_payload(controller="best_static")
        assert "acceptance_loss" not in payload
        explicit = self._config(acceptance_loss="control")
        assert run_semantics_id(payload) == run_semantics_id(
            explicit.run_semantics_payload(controller="best_static")
        )

    def test_fixed_eval_changes_hash(self):
        from rl_newton.benchmark.store import run_semantics_id

        a = self._config().run_semantics_payload(controller="best_static")
        b = self._config(acceptance_loss="fixed_eval").run_semantics_payload(
            controller="best_static"
        )
        assert "acceptance_loss" in b
        assert run_semantics_id(a) != run_semantics_id(b)

    def test_invalid_value_rejected_at_optimizer_config(self):
        with pytest.raises(ValueError):
            self._config(acceptance_loss="nonsense").optimizer_config()
