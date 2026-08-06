"""집계 지표 테스트.

여기서 검증하는 것은 **비교의 공정성**이다. 수치 계산이 맞는지보다, 서로 다른
컨트롤러를 같은 조건에서 비교하고 있는지가 결론을 좌우한다.

프로토콜 D11 (Track E 예산 절단)
--------------------------------
optimizer 루프는 ``spent >= budget`` 에서 종료하므로 **마지막 step 이 예산을
초과한다.** 초과량은 컨트롤러가 고른 action 크기에 비례하므로, 고정 예산
비교에서 큰 step 을 고르는 컨트롤러가 공짜로 이득을 본다.

```text
C0 (평균 k=17.9)   150 GE 예산에 실제 171 GE 소모   <- 큰 step 하나가 공짜
Q=4 (평균 k=3.3)   150 GE 예산에 실제 154 GE 소모
```

이 편향은 "쿼터를 키우면 planner 가 싼 action 을 고르고 성능이 나빠진다"는
관측에 그대로 섞여 있었다. 집계에서 예산을 넘지 않는 prefix 로 잘라 제거한다.
"""

from __future__ import annotations

import math

import pytest

from rl_newton.benchmark.metrics import (
    RELATIVE_LOSS_FLOOR,
    TargetSpec,
    budget_respecting_prefix,
    summarize_run,
)
from rl_newton.optimizers.newton_cg import OptimizationTrace
from rl_newton.types import StepRecord


def make_trace(costs, losses, *, initial_loss: float = 1.0) -> OptimizationTrace:
    trace = OptimizationTrace(run_id="t", controller="c", task_instance_id="i", seed=0)
    trace.initial_loss = initial_loss
    for i, (cost, loss) in enumerate(zip(costs, losses, strict=True)):
        trace.records.append(
            StepRecord(
                run_id="t",
                seed=0,
                optimizer="newton_cg",
                step=i,
                train_loss_before=initial_loss if i == 0 else losses[i - 1],
                train_loss_after=loss,
                cost_ge=cost,
                hvp_count=3,
                cg_budget=3,
            )
        )
    trace.final_loss = losses[-1]
    trace.total_cost_ge = sum(costs)
    # ``n_steps`` 는 records 에서 파생되는 property 다. 직접 설정하지 않는다.
    return trace


class TestBudgetRespectingPrefix:
    def test_overshooting_step_is_excluded(self):
        # 20 + 20 = 40 <= 50. 세 번째 step 을 더하면 60 > 50 이므로 제외한다.
        trace = make_trace([20.0, 20.0, 20.0], [0.5, 0.2, 0.01])
        loss, cost, steps = budget_respecting_prefix(trace, 50.0)
        assert steps == 2
        assert cost == pytest.approx(40.0)
        assert loss == pytest.approx(0.2)

    def test_cost_never_exceeds_budget(self):
        trace = make_trace([21.3] * 10, [0.5**i for i in range(1, 11)])
        _loss, cost, _steps = budget_respecting_prefix(trace, 150.0)
        assert cost <= 150.0

    def test_big_step_controller_loses_its_free_overshoot(self):
        """절단 전에는 큰 step 컨트롤러가 예산을 초과해 쓰고 있었다."""
        big = make_trace([21.3] * 8, [10.0**-i for i in range(1, 9)])
        assert big.total_cost_ge > 150.0
        _loss, big_cost, _steps = budget_respecting_prefix(big, 150.0)
        assert big_cost <= 150.0

        small = make_trace([4.3] * 40, [10.0 ** -(i * 0.2) for i in range(1, 41)])
        _l, small_cost, _s = budget_respecting_prefix(small, 150.0)
        assert small_cost <= 150.0
        # 두 컨트롤러의 예산 사용량 차이가 한 step 비용 이내로 줄어든다.
        assert abs(big_cost - small_cost) <= 21.3

    def test_single_step_over_budget_yields_initial_loss(self):
        """첫 step 조차 예산을 넘으면 아무 진행도 인정하지 않는다."""
        trace = make_trace([100.0], [0.1])
        loss, cost, steps = budget_respecting_prefix(trace, 50.0)
        assert steps == 0
        assert cost == pytest.approx(0.0)
        assert loss == pytest.approx(trace.initial_loss)

    def test_none_budget_leaves_trace_untouched(self):
        trace = make_trace([20.0, 20.0, 20.0], [0.5, 0.2, 0.01])
        assert budget_respecting_prefix(trace, None) == (
            trace.final_loss,
            trace.total_cost_ge,
            trace.n_steps,
        )

    def test_nan_step_stops_the_prefix(self):
        trace = make_trace([10.0, 10.0, 10.0], [0.5, float("nan"), 0.01])
        loss, _cost, steps = budget_respecting_prefix(trace, 100.0)
        assert steps == 2
        assert loss != loss  # NaN 은 결과 자체다. 조용히 건너뛰지 않는다.


class TestSummarizeRunFairness:
    def test_summarize_run_applies_the_prefix(self):
        trace = make_trace([20.0, 20.0, 20.0], [0.5, 0.2, 0.01])
        summary = summarize_run(
            trace, TargetSpec(metric="relative_loss", value=1.0e-6), budget_ge=50.0
        )
        assert summary.total_cost_ge == pytest.approx(40.0)
        assert summary.final_loss == pytest.approx(0.2)
        assert summary.n_steps == 2

    def test_track_t_metrics_are_not_truncated(self):
        """cost-to-target 은 도달 시점으로 정의되므로 예산 절단과 무관하다.

        Track E 를 공정하게 만드는 수정이 Track T 의 정의를 바꾸면 안 된다.
        """
        trace = make_trace([20.0, 20.0, 20.0], [0.5, 0.2, 1.0e-9])
        summary = summarize_run(
            trace, TargetSpec(metric="relative_loss", value=1.0e-6), budget_ge=50.0
        )
        assert summary.reached
        assert summary.cost_to_target_ge == pytest.approx(60.0)

    def test_without_budget_behaviour_is_unchanged(self):
        trace = make_trace([20.0, 20.0, 20.0], [0.5, 0.2, 0.01])
        summary = summarize_run(trace, TargetSpec(metric="relative_loss", value=1.0e-6))
        assert summary.final_loss == pytest.approx(0.01)
        assert summary.n_steps == 3


# ---------------------------------------------------------------------------
# 수치 하한과 조용한 제외 금지 (프로토콜 D14)
# ---------------------------------------------------------------------------


def make_summary(
    *,
    controller: str,
    instance: str = "inst",
    seed: int = 0,
    initial_loss: float = 24.2,
    final_loss: float = 1.0,
    reached: bool = False,
    cost_to_target: float | None = None,
    total_cost_ge: float = 100.0,
):
    """최소 ``RunSummary``. Track E / Track T 분리를 검증하는 데 쓴다."""
    from rl_newton.benchmark.metrics import RunSummary

    return RunSummary(
        run_id="r",
        controller=controller,
        task_instance_id=instance,
        seed=seed,
        target="absolute_loss<=0.01",
        reached=reached,
        cost_to_target_ge=cost_to_target,
        steps_to_target=None,
        hvp_to_target=None,
        initial_loss=initial_loss,
        final_loss=final_loss,
        total_cost_ge=total_cost_ge,
        total_hvp=10,
        search_cost_ge=0.0,
        n_steps=10,
        stop_reason="cost_budget",
        rejection_rate=0.0,
        failure_rate=0.0,
        negative_curvature_rate=0.0,
        cg_convergence_rate=1.0,
        median_residual_ratio=0.0,
        median_damping=1.0e-2,
        median_trust_ratio=1.0,
    )


class TestLossFloorPolicy:
    """``final_loss = 0`` 을 조용히 버리면 최적점에 도달한 run 이 제거된다.

    beam 4 pilot 실측: ``rosen_d2`` 에서 ``onestep_absolute`` 와 ``heuristic`` 이
    ``final_loss=0.0`` 이라 3쌍씩 빠졌고, 게이트 A1 과 B 가 낮게 잡혔다.
    """

    def test_relative_floor_is_scale_invariant(self):
        small = make_summary(controller="c", initial_loss=1.0, final_loss=0.0)
        large = make_summary(controller="c", initial_loss=1.0e6, final_loss=0.0)
        # floor 가 초기 loss 에 비례하므로 최대 logΔ 가 같다.
        assert small.log_improvement == pytest.approx(large.log_improvement)

    def test_floor_is_not_finfo_tiny(self):
        """``tiny`` 를 쓰면 최대 logΔ 가 708 nat 까지 커져 통계를 지배한다."""
        run = make_summary(controller="c", initial_loss=24.2, final_loss=0.0)
        assert run.log_improvement < 40.0
        assert run.log_improvement == pytest.approx(
            math.log(24.2) - math.log(24.2 * RELATIVE_LOSS_FLOOR)
        )

    def test_exact_zero_is_capped_not_dropped(self):
        run = make_summary(controller="c", final_loss=0.0)
        assert run.exact_zero
        assert run.floor_hit
        assert math.isfinite(run.log_improvement)

    def test_small_negative_is_roundoff_and_capped(self):
        run = make_summary(controller="c", initial_loss=24.2, final_loss=-1.0e-15)
        assert run.negative_roundoff
        assert run.floor_hit
        assert math.isfinite(run.log_improvement)

    def test_large_negative_is_numerical_failure(self):
        """모든 음수를 clamp 하면 실제 계산 오류를 숨긴다."""
        run = make_summary(controller="c", initial_loss=24.2, final_loss=-1.0)
        assert not run.negative_roundoff
        assert not math.isfinite(run.log_improvement)

    def test_nonfinite_final_loss_is_not_capped(self):
        for bad in (float("nan"), float("inf")):
            run = make_summary(controller="c", final_loss=bad)
            assert not math.isfinite(run.log_improvement)


class TestTrackSeparation:
    def test_unreached_run_is_included_in_track_e(self):
        """Track E 는 target 도달 여부와 무관하다."""
        from rl_newton.benchmark.metrics import compare_paired_delta

        base = [make_summary(controller="b", final_loss=1.0, reached=False)]
        treat = [make_summary(controller="t", final_loss=0.1, reached=False)]
        d = compare_paired_delta(base, treat)
        assert d.n_valid == 1
        assert d.median_delta == pytest.approx(math.log(10.0))

    def test_same_run_is_censored_in_track_t(self):
        """같은 run 이 Track T 에서는 미도달로 절단된다."""
        from rl_newton.benchmark.metrics import compare_paired

        base = [make_summary(controller="b", reached=True, cost_to_target=50.0)]
        treat = [make_summary(controller="t", reached=False, cost_to_target=None)]
        c = compare_paired(base, treat, metric="cost_to_target_ge")
        assert c.n_both_reached == 0


class TestExclusionIsRecorded:
    def test_excluded_pairs_records_reason(self):
        """조용한 dropna 금지. 빠진 쌍의 task/seed/사유가 남아야 한다."""
        from rl_newton.benchmark.metrics import compare_paired_delta

        base = [make_summary(controller="b", instance="i0", final_loss=1.0)]
        treat = [make_summary(controller="t", instance="i0", final_loss=float("nan"))]
        d = compare_paired_delta(base, treat)
        assert d.n_pairs == 1
        assert d.n_valid == 0
        assert len(d.excluded_pairs) == 1
        task, seed, why = d.excluded_pairs[0]
        assert (task, seed) == ("i0", 0)
        assert "final_loss" in why

    def test_joint_and_one_sided_saturation_are_counted(self):
        from rl_newton.benchmark.metrics import compare_paired_delta

        base = [
            make_summary(controller="b", instance="i0", final_loss=0.0),
            make_summary(controller="b", instance="i1", final_loss=1.0),
        ]
        treat = [
            make_summary(controller="t", instance="i0", final_loss=0.0),
            make_summary(controller="t", instance="i1", final_loss=0.0),
        ]
        d = compare_paired_delta(base, treat)
        assert d.n_valid == 2
        assert d.n_joint_saturated == 1
        assert d.n_one_sided_saturated == 1

    def test_joint_saturation_gives_zero_delta(self):
        """양쪽 모두 하한이면 terminal objective 상 실제 동률이다."""
        from rl_newton.benchmark.metrics import compare_paired_delta

        base = [make_summary(controller="b", final_loss=0.0)]
        treat = [make_summary(controller="t", final_loss=0.0)]
        assert compare_paired_delta(base, treat).median_delta == pytest.approx(0.0)

    def test_drop_saturated_is_sensitivity_only(self):
        """비포화 필터가 포화 쌍만 정확히 제거한다."""
        from rl_newton.benchmark.metrics import drop_saturated_pairs

        base = [
            make_summary(controller="b", instance="i0", final_loss=0.0),
            make_summary(controller="b", instance="i1", final_loss=1.0),
        ]
        treat = [
            make_summary(controller="t", instance="i0", final_loss=0.5),
            make_summary(controller="t", instance="i1", final_loss=0.5),
        ]
        b, t = drop_saturated_pairs(base, treat)
        assert [r.task_instance_id for r in b] == ["i1"]
        assert len(t) == 1


# ---------------------------------------------------------------------------
# 3층 보고: primary / all-task / saturation diagnostic (프로토콜 D14)
# ---------------------------------------------------------------------------


class TestTaskFamilySplit:
    """포화 task 를 primary 에서 분리하되 **버리지 않는다.**

    `rosen_d2` 는 150 GE 에서 여러 컨트롤러가 정확히 0 에 도달해 paired delta 를
    기계적으로 0 으로 만든다. n=6 primary 만 내고 n=9 를 숨기면 선택적 제외다.
    """

    def _runs(self):
        return [
            make_summary(controller="c", instance="quad_spd_d64_seed0", final_loss=1.0),
            make_summary(controller="c", instance="quad_ill_d100_seed0", final_loss=0.5),
            make_summary(controller="c", instance="rosen_d2_s100_seed0", final_loss=0.0),
        ]

    def test_split_separates_by_prefix(self):
        from rl_newton.benchmark.metrics import split_by_task_family

        primary, excluded = split_by_task_family(self._runs(), exclude_prefixes=("rosen_d2",))
        assert [r.task_instance_id for r in primary] == [
            "quad_spd_d64_seed0",
            "quad_ill_d100_seed0",
        ]
        assert [r.task_instance_id for r in excluded] == ["rosen_d2_s100_seed0"]

    def test_nothing_is_dropped(self):
        from rl_newton.benchmark.metrics import split_by_task_family

        runs = self._runs()
        primary, excluded = split_by_task_family(runs, exclude_prefixes=("rosen_d2",))
        assert len(primary) + len(excluded) == len(runs)

    def test_empty_prefix_list_keeps_everything(self):
        from rl_newton.benchmark.metrics import split_by_task_family

        primary, excluded = split_by_task_family(self._runs(), exclude_prefixes=())
        assert len(primary) == 3
        assert excluded == []

    def test_saturation_report_measures_zero_rate(self):
        from rl_newton.benchmark.metrics import saturation_report

        runs = [
            make_summary(controller="c", instance="rosen_d2_a", final_loss=0.0),
            make_summary(controller="c", instance="rosen_d2_b", final_loss=0.0),
            make_summary(controller="c", instance="rosen_d2_c", final_loss=1.0),
        ]
        rep = saturation_report(runs)
        assert rep["n"] == 3
        assert rep["exact_zero_rate"] == pytest.approx(2 / 3)
        assert rep["floor_hit_rate"] == pytest.approx(2 / 3)

    def test_saturation_report_on_empty_is_empty(self):
        from rl_newton.benchmark.metrics import saturation_report

        assert saturation_report([]) == {}

    def test_saturated_pairs_pin_the_all_task_median_to_zero(self):
        """실측 재현: 포화 쌍이 다수면 all-task median 이 기계적으로 0 이 된다.

        beam 4 pilot 에서 9쌍 중 3쌍이 joint saturation 이었고 A2·C2·C3 가 모두
        정확히 `+0.000` 으로 나왔다. 포화 쌍의 delta 가 0 이므로 median 을
        차지하면 실제 개선이 보이지 않는다.
        """
        from rl_newton.benchmark.metrics import compare_paired_delta, split_by_task_family

        # quad 2쌍은 개선, rosen 3쌍은 양쪽 포화 -> delta 0 이 과반이 된다.
        base = [
            make_summary(controller="b", instance=f"quad_{i}", final_loss=1.0) for i in range(2)
        ]
        treat = [
            make_summary(controller="t", instance=f"quad_{i}", final_loss=0.1) for i in range(2)
        ]
        for i in range(3):
            base.append(make_summary(controller="b", instance=f"rosen_d2_{i}", final_loss=0.0))
            treat.append(make_summary(controller="t", instance=f"rosen_d2_{i}", final_loss=0.0))

        all_task = compare_paired_delta(base, treat)
        pb, _ = split_by_task_family(base, exclude_prefixes=("rosen_d2",))
        pt, _ = split_by_task_family(treat, exclude_prefixes=("rosen_d2",))
        primary = compare_paired_delta(pb, pt)

        assert all_task.n_valid == 5
        assert all_task.n_joint_saturated == 3
        # 포화 쌍이 과반이라 median 이 0 으로 고정된다. 실측과 같은 현상이다.
        assert all_task.median_delta == pytest.approx(0.0)
        # primary 는 실제 개선을 드러낸다.
        assert primary.n_valid == 2
        assert primary.median_delta == pytest.approx(math.log(10.0))
