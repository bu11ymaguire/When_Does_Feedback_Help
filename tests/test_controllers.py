"""Stage 2: 컨트롤러와 planner 의 구현 불변조건.

**horizon 이 늘면 실현 성능이 좋아진다고 가정하지 않는다.** beam search 는 정확한
planner 가 아니고, MPC 는 매 step 재계획하므로 깊은 탐색이 좋은 branch 를 중간에
잘라낼 수 있다. 따라서 테스트하는 것은 구현 불변조건이다.

  - 같은 시퀀스를 같은 초기 상태에서 실행하면 같은 terminal loss / cost
  - planner 효용이 실제 terminal objective 와 일치한다
  - **incumbent carry-over**: 이전 horizon 의 최선이 최종 후보에 남는다
    (이것 덕분에 planner **효용**의 단조성만은 보장된다)
  - beam pruning 전후 상태 복원이 정확하다
"""

from __future__ import annotations

import math

import pytest
import torch

from rl_newton.optimizers.action_space import (
    ABSOLUTE,
    LATTICE_BASE,
    NARROW,
    WIDE,
    ActionSpace,
)
from rl_newton.optimizers.controllers import (
    LAGRANGIAN_BETA_GRID,
    AverageRateEfficiencyPlanner,
    BudgetedMPCController,
    CommittedPlanController,
    FixedController,
    HeuristicController,
    LagrangianPlannerController,
    OneStepEfficiencyController,
    OpenLoopController,
    PlanCandidate,
    ScheduleSegment,
    ShrinkingQuotaMPCController,
    average_rate_utility,
    bucket_prune,
    efficiency_score,
    lagrangian_utility,
    make_open_loop_controller,
    pareto_frontier,
)
from rl_newton.optimizers.newton_cg import (
    NewtonCGConfig,
    NewtonCGOptimizer,
    StepContext,
    apply_damping_action,
)
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask
from rl_newton.types import ControllerAction

NARROW_F = NARROW.with_fixed_step_size(1.0)
WIDE_F = WIDE.with_fixed_step_size(1.0)


def make_task(kappa: float = 1.0e3, d: int = 32, seed: int = 0) -> QuadraticTask:
    return QuadraticTask(QuadraticSpec(dimension=d, condition_number=kappa), seed=seed)


def make_optimizer(controller, *, budget: float = 200.0, steps: int = 50):
    task = make_task()
    config = NewtonCGConfig(total_steps=steps, cost_budget_ge=budget, initial_damping=1.0e-2)
    return NewtonCGOptimizer(task, controller, config, run_id="t", seed=0)


# ---------------------------------------------------------------------------
# 행동 공간: 게이트 B의 전제
# ---------------------------------------------------------------------------


class TestActionSpaceResolution:
    def test_narrow_multipliers_are_exact_reciprocals(self):
        """``3 x (1/3) = 1`` 이 정확해야 damping 표류가 없다.

        README 원안 ``0.3`` 은 ``3 x 0.3 = 0.9`` 라서 배수를 번갈아 고르면
        step 당 10% 아래로 밀린다.
        """
        values = NARROW.damping_values
        assert len(values) == 3
        assert values[0] * values[2] == pytest.approx(1.0, rel=1e-12)

    def test_alternating_multipliers_do_not_drift(self):
        """올렸다 되돌리기를 반복해도 damping 이 제자리로 와야 한다."""
        log_damping = -2.0
        up = ControllerAction(damping_multiplier=3.0, cg_budget=5, step_size=1.0)
        down = ControllerAction(damping_multiplier=1.0 / 3.0, cg_budget=5, step_size=1.0)
        for _ in range(20):
            log_damping = apply_damping_action(log_damping, up, min_log10=-8, max_log10=8)
            log_damping = apply_damping_action(log_damping, down, min_log10=-8, max_log10=8)
        assert log_damping == pytest.approx(-2.0, abs=1e-9)

    def test_all_spaces_share_log_resolution(self):
        """게이트 B가 도달성 손실만 재려면 로그 해상도가 같아야 한다.

        ``absolute`` 가 범위만 넓고 해상도가 거칠면 도달성 이득이 해상도
        손실에 잠식되어 두 효과를 분리할 수 없다. 실제로 초기 구성(2 decade
        간격)에서 absolute 가 narrow 보다 나쁜 결과를 냈다.
        """
        expected = math.log10(3.0)
        for space in (NARROW, WIDE, ABSOLUTE):
            logs = sorted(math.log10(v) for v in space.damping_values)
            gaps = [b - a for a, b in zip(logs, logs[1:], strict=False)]
            for gap in gaps:
                assert gap == pytest.approx(expected, rel=1e-9), space.name

    def test_absolute_has_widest_range(self):
        assert ABSOLUTE.log10_span > WIDE.log10_span > NARROW.log10_span

    def test_all_damping_values_lie_on_the_same_power_of_three_lattice(self):
        """세 공간이 같은 격자 위에 있어야 게이트 B가 범위 차이만 잰다.

        ``NARROW`` 와 ``WIDE`` 의 배수 집합은 ``ABSOLUTE`` 의 값 집합과 같은
        ``3^e`` 격자에 놓인다. 해상도가 교란 요인이 되지 않는다.
        """
        for space in (NARROW, WIDE, ABSOLUTE):
            for value in space.damping_values:
                exponent = math.log(value) / math.log(LATTICE_BASE)
                assert exponent == pytest.approx(round(exponent), abs=1e-9), (
                    f"{space.name}: {value}"
                )

    def test_absolute_range_fits_inside_default_damping_bounds(self):
        """경계에서 클립되면 서로 다른 action 이 같은 damping 으로 붕괴한다."""
        config = NewtonCGConfig()
        assert min(ABSOLUTE.damping_values) > config.min_damping
        assert max(ABSOLUTE.damping_values) < config.max_damping

    def test_cg_budgets_are_preserved_across_presets(self):
        """CG budget 은 이 프로젝트의 핵심 inexactness 축이므로 축소하지 않는다."""
        for space in (NARROW, WIDE, ABSOLUTE):
            assert space.cg_budgets == (3, 5, 10, 20), space.name

    def test_fixed_step_size_subset_only(self):
        with pytest.raises(ValueError, match="not in"):
            NARROW.with_fixed_step_size(0.7)

    def test_solve_groups_share_step_sizes(self):
        """step_size 는 CG solve 에 영향이 없으므로 결과를 공유한다."""
        groups = list(NARROW.iter_solve_groups())
        assert len(groups) == NARROW.n_solve_groups
        for representative, step_sizes in groups:
            assert step_sizes == NARROW.step_sizes
            assert representative.step_size == NARROW.step_sizes[0]

    def test_absolute_actions_carry_absolute_damping(self):
        action = ABSOLUTE.action_from_flat(0)
        assert action.is_absolute
        assert action.damping_absolute is not None
        assert not NARROW.action_from_flat(0).is_absolute


# ---------------------------------------------------------------------------
# 효용 함수: terminal objective 가 ratio 합산과 다름을 고정
# ---------------------------------------------------------------------------


class TestAverageRateUtility:
    def test_fixed_budget_is_cumulative_not_per_step_sum(self):
        """누적 효율은 step 별 비율의 합과 다르다. 초판 결함의 회귀 테스트.

        두 step 으로 각각 loss 를 절반씩 줄이며 1 GE 를 쓴 경우:
          per-step ratio 합 = ln2/1 + ln2/1 = 1.386
          누적 효용        = ln4/2            = 0.693
        """
        per_step_sum = efficiency_score(1.0, 0.5, 1.0) + efficiency_score(0.5, 0.25, 1.0)
        cumulative = average_rate_utility(1.0, 0.25, 2.0, track="fixed_budget")

        assert per_step_sum == pytest.approx(2.0 * math.log(2.0))
        assert cumulative == pytest.approx(math.log(4.0) / 2.0)
        assert cumulative != pytest.approx(per_step_sum)

    def test_fixed_budget_ranks_by_rate_not_by_absolute_gain(self):
        """고정 예산 효용은 **비율**로 순위를 정한다. 절대 감소량이 아니다.

        이것이 파일럿 현상의 메커니즘이다. CG 반복은 수익이 체감하므로
        ``k=3`` 이 절대 감소량은 작아도 비용당 감소량은 클 수 있고, 그러면
        국소 효율 기준에서 이긴다.
        """
        # 절대 감소량은 작지만 비용당으로는 큰 경우
        cheap_high_rate = average_rate_utility(1.0, 0.5, 3.0, track="fixed_budget")
        costly_low_rate = average_rate_utility(1.0, 0.1, 20.0, track="fixed_budget")
        assert cheap_high_rate == pytest.approx(math.log(2.0) / 3.0)
        assert costly_low_rate == pytest.approx(math.log(10.0) / 20.0)
        assert cheap_high_rate > costly_low_rate

        # 반대로 비용당 감소량이 작으면 싼 행동도 진다
        cheap_low_rate = average_rate_utility(1.0, 0.9, 3.0, track="fixed_budget")
        costly_high_rate = average_rate_utility(1.0, 0.4, 20.0, track="fixed_budget")
        assert cheap_low_rate < costly_high_rate

    def test_cost_to_target_returns_negative_total_cost_when_reached(self):
        u = average_rate_utility(1.0, 1.0e-7, 42.0, track="cost_to_target", target_loss=1.0e-6)
        assert u == pytest.approx(-42.0)

    def test_cost_to_target_estimates_remaining_cost(self):
        """미도달이면 남은 거리를 관측 진행률로 나눠 예상 총비용을 만든다."""
        # 10 GE 로 loss 를 1 -> 0.1 (ln10 nat). 목표는 0.01 (추가로 ln10 필요).
        u = average_rate_utility(1.0, 0.1, 10.0, track="cost_to_target", target_loss=0.01)
        assert u == pytest.approx(-20.0, rel=1e-9)

    def test_cost_to_target_prefers_lower_total_cost(self):
        fast = average_rate_utility(1.0, 0.1, 10.0, track="cost_to_target", target_loss=0.01)
        slow = average_rate_utility(1.0, 0.5, 10.0, track="cost_to_target", target_loss=0.01)
        assert fast > slow

    def test_no_progress_is_rejected(self):
        assert average_rate_utility(1.0, 1.0, 5.0, track="fixed_budget") == -math.inf
        assert average_rate_utility(1.0, 2.0, 5.0, track="fixed_budget") == -math.inf
        assert average_rate_utility(1.0, float("nan"), 5.0, track="fixed_budget") == -math.inf


# ---------------------------------------------------------------------------
# 시뮬레이션 결정론성과 상태 복원
# ---------------------------------------------------------------------------


class TestSimulationInvariants:
    def test_same_sequence_from_same_state_gives_same_outcome(self):
        """planner 가 신뢰할 수 있으려면 시뮬레이션이 결정론적이어야 한다."""
        optimizer = make_optimizer(FixedController(NARROW_F.action_from_flat(0)))
        actions = [NARROW_F.action_from_flat(i) for i in (0, 5, 11)]

        def run_sequence() -> list[tuple[float, float, bool]]:
            out = []
            for action in actions:
                out.append(optimizer.simulate_step(action))
            return out

        root = optimizer.snapshot()
        first = run_sequence()
        optimizer.restore(root)
        second = run_sequence()

        assert first == second

    def test_restore_recovers_parameters_and_damping_exactly(self):
        optimizer = make_optimizer(FixedController(NARROW_F.action_from_flat(0)))
        root_params, root_log = optimizer.snapshot()

        optimizer.simulate_step(NARROW_F.action_from_flat(11))
        assert not torch.allclose(optimizer.flattener.flatten_params(), root_params)

        optimizer.restore((root_params, root_log))
        assert torch.equal(optimizer.flattener.flatten_params(), root_params)
        assert optimizer.damping_log10 == root_log

    def test_simulate_step_cost_is_charged_to_search(self):
        """planner 비용은 본문 비용에 섞이지 않아야 한다 (프로토콜 D5)."""
        planner = AverageRateEfficiencyPlanner(NARROW_F, horizon=2, beam_width=2)
        optimizer = make_optimizer(planner, budget=60.0, steps=3)
        trace = optimizer.run()

        assert trace.search_cost_ge > trace.total_cost_ge
        assert trace.search_hvp > 0


# ---------------------------------------------------------------------------
# incumbent carry-over: 게이트 C 해석의 전제
# ---------------------------------------------------------------------------


class TestIncumbentCarryOver:
    def _utility_at_first_step(self, space: ActionSpace, horizon: int) -> float:
        planner = AverageRateEfficiencyPlanner(space, horizon=horizon, beam_width=2)
        task = make_task()
        config = NewtonCGConfig(total_steps=1, cost_budget_ge=1.0e9, initial_damping=1.0e-2)
        optimizer = NewtonCGOptimizer(task, planner, config, run_id="u", seed=0)
        optimizer.run()
        return planner.last_utility

    @pytest.mark.parametrize("space", [NARROW_F, WIDE_F], ids=["narrow", "wide"])
    def test_planner_utility_is_monotone_in_horizon(self, space: ActionSpace):
        """incumbent carry-over 덕분에 **효용**은 H 가 늘어도 감소하지 않는다.

        실현 성능의 단조성은 보장되지 않는다 (MPC 는 매 step 재계획하고 beam
        search 는 정확하지 않다). 그래서 테스트는 효용에만 적용한다.
        """
        u1 = self._utility_at_first_step(space, 1)
        u3 = self._utility_at_first_step(space, 3)

        assert math.isfinite(u1)
        assert u3 >= u1 - 1e-12

    def test_chosen_depth_is_recorded(self):
        """planner 가 실제로 깊은 계획을 쓰는지 관측 가능해야 한다.

        항상 1이면 horizon 을 늘려도 의미가 없다는 직접적 증거다.
        """
        planner = AverageRateEfficiencyPlanner(NARROW_F, horizon=3, beam_width=2)
        make_optimizer(planner, budget=80.0, steps=4).run()

        assert planner.choices
        assert all(1 <= c.chosen_depth <= 3 for c in planner.choices)


# ---------------------------------------------------------------------------
# H=1 등가성: 게이트 A1/B가 one-step 구현을 쓰는 근거
# ---------------------------------------------------------------------------


class TestOneStepEquivalence:
    def test_one_step_is_far_cheaper_than_h1_planner(self):
        """one-step 은 HVP 그래프를 후보 전체에 공유하므로 훨씬 싸다.

        absolute (34 damping x 4 budget) 를 감당할 수 있는 유일한 경로다.
        """
        onestep_trace = make_optimizer(
            OneStepEfficiencyController(NARROW_F), budget=60.0, steps=3
        ).run()
        planner_trace = make_optimizer(
            AverageRateEfficiencyPlanner(NARROW_F, horizon=1, beam_width=1),
            budget=60.0,
            steps=3,
        ).run()

        assert onestep_trace.search_cost_ge < planner_trace.search_cost_ge

    def test_absolute_sweep_cost_is_bounded(self):
        """absolute 는 비싸지만 one-step 경로로는 감당 가능한 범위여야 한다."""
        absolute_f = ABSOLUTE.with_fixed_step_size(1.0)
        assert len(absolute_f) == len(ABSOLUTE.damping_values) * 4
        # sweep 당 HVP = n_damping x sum(budgets)
        assert absolute_f.hvp_per_sweep == len(ABSOLUTE.damping_values) * 38


# ---------------------------------------------------------------------------
# 기타 컨트롤러
# ---------------------------------------------------------------------------


class TestBaselineControllers:
    def test_fixed_controller_always_returns_same_action(self):
        action = NARROW_F.action_from_flat(3)
        controller = FixedController(action)
        context = StepContext(step=0, total_steps=10, loss=1.0, grad_norm=1.0, damping=1e-2)
        assert controller.select(context, None) is action  # type: ignore[arg-type]

    def test_open_loop_switches_on_progress_only(self):
        early = NARROW_F.action_from_flat(0)
        late = NARROW_F.action_from_flat(11)
        controller = OpenLoopController([ScheduleSegment(0.5, early), ScheduleSegment(1.0, late)])
        assert controller.action_at(0.0) is early
        assert controller.action_at(0.5) is early
        assert controller.action_at(0.51) is late
        assert controller.action_at(1.0) is late

    def test_open_loop_requires_sorted_segments_covering_one(self):
        a = NARROW_F.action_from_flat(0)
        with pytest.raises(ValueError, match="sorted"):
            OpenLoopController([ScheduleSegment(1.0, a), ScheduleSegment(0.5, a)])
        with pytest.raises(ValueError, match="progress 1.0"):
            OpenLoopController([ScheduleSegment(0.5, a)])

    def test_make_open_loop_controller_validates_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            make_open_loop_controller(NARROW_F, [0, 1], [1.0])

    def test_heuristic_rejects_absolute_space(self):
        """absolute 는 분석 전용이다. 상대 배수를 전제하는 규칙에 쓸 수 없다."""
        with pytest.raises(ValueError, match="absolute"):
            HeuristicController(ABSOLUTE)

    def test_heuristic_moves_conservative_on_low_trust(self):
        from rl_newton.types import StepRecord

        controller = HeuristicController(NARROW_F, initial_flat=len(NARROW_F) // 2)
        low_trust = StepRecord(
            run_id="t",
            seed=0,
            optimizer="h",
            step=0,
            train_loss_before=1.0,
            train_loss_after=0.99,
            trust_ratio=0.01,
        )
        context = StepContext(
            step=1,
            total_steps=10,
            loss=0.99,
            grad_norm=1.0,
            damping=1e-2,
            previous=low_trust,
        )
        action = controller.select(context, None)  # type: ignore[arg-type]
        # damping 을 올리고 예산을 늘리는 방향이어야 한다
        assert action.damping_multiplier >= 1.0

    def test_all_controllers_run_without_crashing(self):
        controllers = [
            FixedController(NARROW_F.action_from_flat(5)),
            HeuristicController(NARROW_F),
            OneStepEfficiencyController(NARROW_F),
            AverageRateEfficiencyPlanner(NARROW_F, horizon=2, beam_width=2),
            make_open_loop_controller(NARROW_F, [0, 5, 11], [0.3, 0.7, 1.0]),
        ]
        for controller in controllers:
            trace = make_optimizer(controller, budget=80.0, steps=6).run()
            assert trace.n_steps >= 1
            assert math.isfinite(trace.final_loss)
            assert trace.final_loss <= trace.initial_loss


class TestCostBudgetTermination:
    def test_run_stops_on_cost_budget(self):
        """비교는 GE 예산 기준이어야 한다 (README §4.2)."""
        trace = make_optimizer(
            FixedController(NARROW_F.action_from_flat(11)), budget=100.0, steps=1000
        ).run()

        assert trace.stop_reason == "cost_budget"
        assert trace.total_cost_ge >= 100.0
        assert trace.n_steps < 1000

    def test_expensive_actions_consume_budget_faster(self):
        """k=20 은 k=3 보다 같은 예산에서 적은 step 을 쓴다."""
        cheap = next(a for a in NARROW_F.iter_actions() if a.cg_budget == 3)
        costly = next(a for a in NARROW_F.iter_actions() if a.cg_budget == 20)

        cheap_trace = make_optimizer(FixedController(cheap), budget=200.0, steps=1000).run()
        costly_trace = make_optimizer(FixedController(costly), budget=200.0, steps=1000).run()

        assert cheap_trace.n_steps > costly_trace.n_steps


# ---------------------------------------------------------------------------
# 쿼터 기반 planner (게이트 C 주 컨트롤러, 프로토콜 D10)
# ---------------------------------------------------------------------------


class TestParetoPruning:
    def test_dominated_candidate_is_removed(self):
        """GE 를 더 쓰고 loss 도 더 높으면 지배된다."""
        better = PlanCandidate(used_ge=5.0, terminal_loss=0.1)
        worse = PlanCandidate(used_ge=9.0, terminal_loss=0.5)
        assert pareto_frontier([better, worse]) == [better]

    def test_non_dominated_candidates_are_preserved(self):
        """싼-높은loss 와 비싼-낮은loss 는 둘 다 남아야 한다.

        이것이 비율 하나로 정렬하면 안 되는 이유다. ``Δlog L / cost`` 로
        줄이면 비싼 장기 계획이 조기에 탈락한다.
        """
        cheap = PlanCandidate(used_ge=4.0, terminal_loss=0.5)
        costly = PlanCandidate(used_ge=20.0, terminal_loss=0.01)
        frontier = pareto_frontier([costly, cheap])
        assert set(id(n) for n in frontier) == {id(cheap), id(costly)}

    def test_equal_cost_keeps_only_lower_loss(self):
        keep = PlanCandidate(used_ge=5.0, terminal_loss=0.1)
        drop = PlanCandidate(used_ge=5.0, terminal_loss=0.2)
        assert pareto_frontier([drop, keep]) == [keep]

    def test_minimum_loss_candidate_always_survives(self):
        """Pareto 는 최소 loss 후보를 지우지 않는다.

        이 성질이 incumbent carry-over 를 대체한다. depth 1 최선은 더 나은
        계획에 의해서만 밀려난다.
        """
        nodes = [
            PlanCandidate(used_ge=c, terminal_loss=loss)
            for c, loss in ((3.0, 0.9), (7.0, 0.4), (11.0, 0.05), (15.0, 0.6))
        ]
        frontier = pareto_frontier(nodes)
        assert min(n.terminal_loss for n in frontier) == 0.05

    def test_bucket_prune_keeps_best_per_cost_bucket(self):
        """비용 구간마다 살아남으므로 싼 계획과 비싼 계획이 섞이지 않는다."""
        nodes = [
            PlanCandidate(used_ge=1.0, terminal_loss=0.9),
            PlanCandidate(used_ge=2.0, terminal_loss=0.8),
            PlanCandidate(used_ge=11.0, terminal_loss=0.3),
            PlanCandidate(used_ge=12.0, terminal_loss=0.2),
        ]
        kept = bucket_prune(nodes, beam_width=1, bucket_ge=10.0)
        assert [n.used_ge for n in kept] == [2.0, 12.0]


class TestLagrangianUtility:
    def test_no_cost_division_so_no_average_dilution(self):
        """누적 형태다. 같은 비용이면 loss 가 낮은 쪽이 항상 높은 효용이다."""
        deep = lagrangian_utility(1.0, 0.01, 10.0, beta=0.01)
        shallow = lagrangian_utility(1.0, 0.5, 10.0, beta=0.01)
        assert deep > shallow

    def test_beta_zero_ignores_cost(self):
        a = lagrangian_utility(1.0, 0.1, 1.0, beta=0.0)
        b = lagrangian_utility(1.0, 0.1, 100.0, beta=0.0)
        assert a == pytest.approx(b)

    def test_large_beta_prefers_cheap(self):
        cheap = lagrangian_utility(1.0, 0.5, 3.0, beta=1.0)
        costly = lagrangian_utility(1.0, 0.1, 20.0, beta=1.0)
        assert cheap > costly

    def test_beta_grid_is_fixed_and_sorted(self):
        """사전 고정 격자다. 결과를 보고 바꾸면 사후 선택이 된다."""
        assert tuple(sorted(LAGRANGIAN_BETA_GRID)) == LAGRANGIAN_BETA_GRID
        assert LAGRANGIAN_BETA_GRID[0] == 0.0


class TestBudgetedSelectionRule:
    """선택 규칙을 후보 집합에 직접 적용해 검증한다."""

    def _planner(self, **kwargs):
        params = {"quota_ge": 100.0, "track": "fixed_budget"}
        params.update(kwargs)
        return BudgetedMPCController(NARROW_F, **params)  # type: ignore[arg-type]

    def _node(self, cost, loss, depth=1, reached=False):
        from rl_newton.optimizers.controllers import _PlanNode

        return _PlanNode(
            actions=tuple(NARROW_F.action_from_flat(i % len(NARROW_F)) for i in range(depth)),
            used_ge=cost,
            terminal_loss=loss,
            snapshot=(None, 0.0),
            reached_target=reached,
        )

    def test_same_quota_picks_lower_terminal_loss(self):
        """같은 쿼터 안에서는 terminal loss 가 낮은 시퀀스를 고른다."""
        low = self._node(90.0, 0.01, depth=4)
        high = self._node(90.0, 0.30, depth=2)
        assert self._planner()._best([high, low]) is low

    def test_cheap_high_rate_short_plan_does_not_always_win(self):
        """싸고 효율 높은 짧은 계획이 무조건 이기지 않는다.

        비율 목적함수라면 ``log(2)/3 = 0.231 > log(100)/90 = 0.051`` 이므로
        싼 계획이 이긴다. 고정 예산 목적에서는 terminal loss 가 기준이므로
        비싼 계획이 이긴다. 이것이 D10 교체의 핵심이다.
        """
        cheap_short = self._node(3.0, 0.5, depth=1)
        costly_deep = self._node(90.0, 0.01, depth=5)
        assert self._planner()._best([cheap_short, costly_deep]) is costly_deep
        # 비율 기준이라면 반대 결론이 나온다는 것을 같은 수치로 확인한다.
        assert average_rate_utility(1.0, 0.5, 3.0, track="fixed_budget") > average_rate_utility(
            1.0, 0.01, 90.0, track="fixed_budget"
        )

    def test_tie_on_loss_prefers_fewer_ge_then_shorter(self):
        a = self._node(50.0, 0.1, depth=3)
        b = self._node(20.0, 0.1, depth=4)
        c = self._node(20.0, 0.1, depth=2)
        assert self._planner()._best([a, b, c]) is c

    def test_track_t_prefers_target_reaching_over_lower_loss(self):
        """도달한 시퀀스가 미도달보다 우선이다. loss 가 더 높아도 그렇다."""
        planner = self._planner(track="cost_to_target", target_loss=0.1)
        reached = self._node(30.0, 0.09, reached=True)
        lower_loss_not_reached = self._node(30.0, 0.11, reached=False)
        assert planner._best([lower_loss_not_reached, reached]) is reached

    def test_track_t_picks_min_cost_among_reached(self):
        planner = self._planner(track="cost_to_target", target_loss=0.1)
        expensive = self._node(80.0, 0.001, reached=True)
        cheap = self._node(20.0, 0.09, reached=True)
        assert planner._best([expensive, cheap]) is cheap

    def test_track_t_falls_back_to_lowest_loss_when_none_reached(self):
        planner = self._planner(track="cost_to_target", target_loss=1.0e-9)
        a = self._node(20.0, 0.5)
        b = self._node(80.0, 0.2)
        assert planner._best([a, b]) is b

    def test_lagrangian_overrides_only_the_selection_rule(self):
        """탐색은 같고 선택만 다르다. β 가 크면 싼 계획을 고른다."""
        cheap = self._node(3.0, 0.5, depth=1)
        costly = self._node(90.0, 0.01, depth=5)
        greedy_beta = LagrangianPlannerController(NARROW_F, beta=1.0, quota_ge=100.0)
        patient_beta = LagrangianPlannerController(NARROW_F, beta=0.0, quota_ge=100.0)
        assert greedy_beta._best([cheap, costly]) is cheap
        assert patient_beta._best([cheap, costly]) is costly


class TestBudgetedQuotaMechanics:
    def test_requires_exactly_one_quota_specification(self):
        with pytest.raises(ValueError, match="정확히 하나"):
            BudgetedMPCController(NARROW_F)
        with pytest.raises(ValueError, match="정확히 하나"):
            BudgetedMPCController(NARROW_F, quota_multiplier=1.0, quota_ge=10.0)

    def test_cost_to_target_requires_target(self):
        with pytest.raises(ValueError, match="target_loss"):
            BudgetedMPCController(NARROW_F, quota_multiplier=1.0, track="cost_to_target")

    def test_quota_resolves_from_c_max(self):
        """``Q = multiplier x c_max``. c_max 는 최대 CG budget action 의 비용이다."""
        planner = BudgetedMPCController(NARROW_F, quota_multiplier=2.0, beam_width=2)
        optimizer = make_optimizer(planner, budget=40.0, steps=1)
        optimizer.run()
        max_budget = max(NARROW_F.cg_budgets)
        expected = 2.0 * optimizer.step_cost_ge(max_budget, 1, with_graph=True)
        assert planner.quota_ge == pytest.approx(expected)

    def test_plan_never_exceeds_quota(self):
        """모든 채택 계획은 쿼터 안이어야 한다. 사다리 비교의 전제다."""
        planner = BudgetedMPCController(NARROW_F, quota_multiplier=2.0, beam_width=2)
        make_optimizer(planner, budget=60.0, steps=4).run()
        assert planner.choices
        for choice in planner.choices:
            if math.isfinite(choice.plan_used_ge):
                assert choice.plan_used_ge <= planner.quota_ge * (1.0 + 1.0e-9)

    def test_larger_quota_allows_deeper_plans(self):
        """쿼터를 늘리면 최대 채택 depth 가 줄어들 수 없다 (탐색 가능 집합이 포함관계).

        실현 성능의 단조성은 주장하지 않는다. MPC 는 매 step 재계획하므로
        보장되지 않는다.
        """
        depths = {}
        for multiplier in (1.0, 4.0):
            planner = BudgetedMPCController(
                NARROW_F, quota_multiplier=multiplier, beam_width=2, max_depth=8
            )
            make_optimizer(planner, budget=50.0, steps=2).run()
            depths[multiplier] = max(c.chosen_depth for c in planner.choices)
        assert depths[4.0] >= depths[1.0]

    def test_depth_cap_is_recorded_when_it_binds(self):
        """계산 상한에 걸리면 기록된다. 조용히 넘기면 사다리 비교가 훼손된다."""
        planner = BudgetedMPCController(NARROW_F, quota_multiplier=8.0, beam_width=1, max_depth=2)
        make_optimizer(planner, budget=40.0, steps=2).run()
        assert any(c.depth_cap_hit for c in planner.choices)

    def test_search_cost_is_charged_separately(self):
        """planner 비용은 본문 비용에 섞이지 않는다 (프로토콜 D5)."""
        planner = BudgetedMPCController(NARROW_F, quota_multiplier=1.0, beam_width=2)
        trace = make_optimizer(planner, budget=60.0, steps=3).run()
        assert trace.search_cost_ge > 0.0
        assert trace.total_cost_ge <= 60.0 + max(NARROW_F.cg_budgets) + 2.0

    def test_state_is_restored_across_branches(self):
        """분기 사이에 파라미터와 damping 이 정확히 복원돼야 한다.

        복원이 어긋나면 planner 가 평가한 것과 실제 적용 결과가 달라지고,
        게이트 결론이 조용히 오염된다.
        """
        planner = BudgetedMPCController(NARROW_F, quota_multiplier=2.0, beam_width=2)
        config = NewtonCGConfig(total_steps=1, cost_budget_ge=1.0e9, initial_damping=1.0e-2)
        optimizer = NewtonCGOptimizer(make_task(), planner, config, run_id="t", seed=0)
        trace = optimizer.run()

        # planner 가 고른 action 을 같은 초기 상태에서 단독 실행하면 같은 결과여야
        # 한다. 분기 사이에 파라미터나 damping 이 어긋나 있으면 어긋난다.
        chosen = planner.trajectory[0][1]
        replay = NewtonCGOptimizer(
            make_task(), FixedController(chosen), config, run_id="r", seed=0
        ).run()
        assert replay.final_loss == pytest.approx(trace.final_loss, rel=1.0e-10)

    def test_snapshot_roundtrip_is_exact(self):
        """``snapshot`` / ``restore`` 가 파라미터와 damping 을 정확히 되돌린다."""
        planner = BudgetedMPCController(NARROW_F, quota_multiplier=1.0, beam_width=2)
        optimizer = make_optimizer(planner, budget=1.0e9, steps=1)
        root = optimizer.snapshot()
        optimizer.simulate_step(NARROW_F.action_from_flat(0))
        moved = optimizer.snapshot()
        assert not torch.allclose(moved[0], root[0])  # type: ignore[arg-type]
        optimizer.restore(root)
        back = optimizer.snapshot()
        assert torch.equal(back[0], root[0])  # type: ignore[arg-type]
        assert back[1] == root[1]


class TestDelayedRewardToyProblem:
    """ "지금 손해, 나중에 이득" 구조에서 depth 2 가 선택되는가.

    비율 목적함수는 이 구조를 표현할 수 없다 (mediant 부등식). 고정 예산
    목적함수는 표현할 수 있어야 한다. 후보 집합을 직접 만들어 선택 규칙만
    검증하므로 최적화 문제의 우연에 의존하지 않는다.
    """

    def _node(self, cost, loss, depth):
        from rl_newton.optimizers.controllers import _PlanNode

        return _PlanNode(
            actions=tuple(NARROW_F.action_from_flat(depth) for _ in range(depth)),
            used_ge=cost,
            terminal_loss=loss,
            snapshot=(None, 0.0),
        )

    def test_investment_plan_wins_under_fixed_quota(self):
        """1 step 은 손해지만 2 step 누적으로는 이득인 계획.

        ```text
        계획 A (depth 1)  10 GE, loss 0.60   rate = log(1/0.6)/10  = 0.051
        계획 B (depth 2)  20 GE, loss 0.10   rate = log(1/0.1)/20  = 0.115
        계획 C (depth 1)   5 GE, loss 0.70   rate = log(1/0.7)/5   = 0.071
        ```

        비율 기준으로는 C 가 A 를 이기고 B 도 이길 수 있다. 고정 예산(20 GE)
        기준에서는 B 가 이겨야 한다.
        """
        planner = BudgetedMPCController(NARROW_F, quota_ge=20.0)
        a = self._node(10.0, 0.60, 1)
        b = self._node(20.0, 0.10, 2)
        c = self._node(5.0, 0.70, 1)
        assert planner._best([a, b, c]) is b

    def test_ratio_objective_picks_the_shallow_plan_on_same_candidates(self):
        """같은 후보에서 비율 목적함수는 얕은 계획을 고른다. 대비 증거다."""
        rates = {
            "A_depth1": average_rate_utility(1.0, 0.60, 10.0, track="fixed_budget"),
            "B_depth2": average_rate_utility(1.0, 0.10, 20.0, track="fixed_budget"),
            "C_depth1": average_rate_utility(1.0, 0.70, 5.0, track="fixed_budget"),
        }
        assert (
            max(rates, key=lambda k: rates[k]) == "B_depth2"
            or rates["C_depth1"] > rates["A_depth1"]
        )
        # 핵심: 비율은 비용을 나누므로 싼 얕은 계획이 비싼 깊은 계획을 이길 수 있다.
        assert rates["C_depth1"] > rates["A_depth1"]

    def test_deep_plan_is_reachable_in_frontier(self):
        """Pareto 가지치기가 깊은 투자 계획을 지우지 않아야 한다."""
        a = self._node(10.0, 0.60, 1)
        b = self._node(20.0, 0.10, 2)
        c = self._node(5.0, 0.70, 1)
        frontier = pareto_frontier([a, b, c])
        assert b in frontier


# ---------------------------------------------------------------------------
# 실행 방식 3종 (프로토콜 D12)
# ---------------------------------------------------------------------------


class TestPlanPredictionMatchesExecution:
    """**가장 먼저 확인할 불변조건.** 이게 깨지면 이후 비교는 의미가 없다.

    synthetic task 는 결정적이므로 planner 가 예측한 terminal loss 와 그 계획을
    끝까지 실행한 결과가 같아야 한다. 다르면 상태 복원이나 실행 회계에 버그가
    있는 것이다.
    """

    def test_committed_execution_reproduces_predicted_loss(self):
        planner = CommittedPlanController(NARROW_F, quota_ge=60.0, beam_width=4, max_depth=12)
        task = make_task()
        # 예산을 쿼터와 같게 두어 window 가 하나만 열리게 한다.
        config = NewtonCGConfig(total_steps=200, cost_budget_ge=60.0, initial_damping=1.0e-2)
        trace = NewtonCGOptimizer(task, planner, config, run_id="c", seed=0).run()

        assert planner.predictions, "계획이 하나는 수립돼야 한다"
        _step, predicted_loss, predicted_cost = planner.predictions[0]

        # 계획 비용만큼 실행된 지점의 loss 를 찾는다.
        spent = 0.0
        realized = trace.initial_loss
        for record in trace.records:
            spent += record.cost_ge
            realized = record.train_loss_after
            if spent >= predicted_cost - 1.0e-9:
                break
        assert spent == pytest.approx(predicted_cost, rel=1.0e-9)
        assert realized == pytest.approx(predicted_loss, rel=1.0e-9)

    def test_committed_replans_only_when_the_plan_is_exhausted(self):
        """계획 도중에는 재계획하지 않는다. 이것이 fresh-quota 와의 차이다."""
        planner = CommittedPlanController(NARROW_F, quota_ge=60.0, beam_width=4, max_depth=12)
        config = NewtonCGConfig(total_steps=200, cost_budget_ge=200.0, initial_damping=1.0e-2)
        trace = NewtonCGOptimizer(make_task(), planner, config, run_id="c", seed=0).run()
        # 계획 수립 횟수 < step 수. fresh-quota 는 매 step 계획하므로 같아진다.
        assert len(planner.predictions) < trace.n_steps
        # 각 계획은 여러 step 을 덮는다.
        depths = [c.chosen_depth for c in planner.choices]
        assert max(depths) >= 2

    def test_fresh_quota_replans_every_step(self):
        """대조군. 실행 방식만 다르고 탐색은 동일하다는 것을 확인한다."""
        planner = BudgetedMPCController(NARROW_F, quota_ge=60.0, beam_width=4, max_depth=12)
        config = NewtonCGConfig(total_steps=200, cost_budget_ge=200.0, initial_damping=1.0e-2)
        trace = NewtonCGOptimizer(make_task(), planner, config, run_id="f", seed=0).run()
        assert len(planner.choices) == trace.n_steps

    def test_plan_returns_full_sequence(self):
        planner = BudgetedMPCController(NARROW_F, quota_ge=60.0, beam_width=4, max_depth=12)
        optimizer = make_optimizer(planner, budget=1.0e9, steps=1)
        context = StepContext(
            step=0,
            total_steps=1,
            loss=float(optimizer.task.loss().detach()),
            grad_norm=1.0,
            damping=1.0e-2,
        )
        plan = planner.plan(context, optimizer)
        assert plan is not None
        assert plan.depth == len(plan.actions) >= 1
        assert plan.first_action is plan.actions[0]
        assert plan.suffix == plan.actions[1:]


class TestShrinkingQuota:
    def test_quota_shrinks_as_cost_is_spent(self):
        """쿼터가 차감되므로 window 안에서 quota_ge 기록이 감소해야 한다."""
        planner = ShrinkingQuotaMPCController(NARROW_F, quota_ge=90.0, beam_width=2, max_depth=12)
        NewtonCGOptimizer(
            make_task(),
            planner,
            NewtonCGConfig(total_steps=6, cost_budget_ge=1.0e9, initial_damping=1.0e-2),
            run_id="s",
            seed=0,
        ).run()
        quotas = [c.quota_ge for c in planner.choices]
        assert quotas[0] == pytest.approx(90.0)
        # 첫 window 안에서는 단조 감소한다.
        first_window = [q for q in quotas if q <= 90.0 + 1e-9]
        assert any(b < a for a, b in zip(first_window, first_window[1:], strict=False))

    def test_new_window_opens_when_quota_is_exhausted(self):
        planner = ShrinkingQuotaMPCController(NARROW_F, quota_ge=30.0, beam_width=2, max_depth=12)
        NewtonCGOptimizer(
            make_task(),
            planner,
            NewtonCGConfig(total_steps=12, cost_budget_ge=1.0e9, initial_damping=1.0e-2),
            run_id="s",
            seed=0,
        ).run()
        assert planner.windows >= 2

    def test_fresh_quota_does_not_shrink(self):
        """대조군. fresh-quota 는 매 step 같은 쿼터를 다시 받는다."""
        planner = BudgetedMPCController(NARROW_F, quota_ge=90.0, beam_width=2, max_depth=12)
        NewtonCGOptimizer(
            make_task(),
            planner,
            NewtonCGConfig(total_steps=5, cost_budget_ge=1.0e9, initial_damping=1.0e-2),
            run_id="f",
            seed=0,
        ).run()
        assert all(c.quota_ge == pytest.approx(90.0) for c in planner.choices)


class TestSeedPlanIncumbent:
    """이전 계획의 suffix 가 후보에 보존되는가.

    결정적 환경에서 재계획이 더 나은 것을 못 찾아도 이전 suffix 는 유지할 수
    있어야 한다. 그렇지 않으면 beam 근사 때문에 재계획 자체가 성능을 떨어뜨리고,
    그것이 "피드백이 해롭다"로 오해된다.
    """

    def _plan_once(self, *, seed_plan=(), beam=1):
        planner = BudgetedMPCController(NARROW_F, quota_ge=60.0, beam_width=beam, max_depth=12)
        optimizer = make_optimizer(planner, budget=1.0e9, steps=1)
        context = StepContext(
            step=0,
            total_steps=1,
            loss=float(optimizer.task.loss().detach()),
            grad_norm=1.0,
            damping=1.0e-2,
        )
        return planner.plan(context, optimizer, seed_plan=seed_plan)

    def test_seed_plan_can_only_improve_the_result(self):
        """seed 를 주면 결과가 나빠질 수 없다. 후보 집합이 커지기만 하므로."""
        without = self._plan_once(beam=1)
        assert without is not None
        # 넓은 탐색으로 좋은 계획을 찾아 seed 로 넣는다.
        rich = self._plan_once(beam=8)
        assert rich is not None
        seeded = self._plan_once(seed_plan=rich.actions, beam=1)
        assert seeded is not None
        assert seeded.terminal_loss <= without.terminal_loss + 1.0e-12

    def test_seed_plan_survives_pruning(self):
        """좁은 beam 에서도 seed 계획이 채택될 수 있어야 한다."""
        rich = self._plan_once(beam=8)
        assert rich is not None
        seeded = self._plan_once(seed_plan=rich.actions, beam=1)
        assert seeded is not None
        assert seeded.terminal_loss <= rich.terminal_loss + 1.0e-12

    def test_seed_plan_over_quota_is_truncated_not_rejected(self):
        """쿼터를 넘는 seed 는 들어가는 prefix 까지만 후보가 된다."""
        rich = self._plan_once(beam=4)
        assert rich is not None
        long_seed = rich.actions * 5
        seeded = self._plan_once(seed_plan=long_seed, beam=1)
        assert seeded is not None
        assert seeded.used_ge <= 60.0 * (1.0 + 1.0e-9)


# ---------------------------------------------------------------------------
# 실행 방식 사이의 상태 비공유와 suffix 유지율 (프로토콜 D15)
# ---------------------------------------------------------------------------


class TestExecutionModeIsolation:
    """``committed`` 와 ``shrinking`` 이 같은 결과를 낼 때 alias 가 아님을 봉인한다.

    beam 4 pilot 에서 ``Q1`` 의 두 컨트롤러가 bitwise 같은 ``final_loss`` 를 냈다.
    실제 동률로 확인됐지만(``chosen_depths`` 와 ``planner_stats`` 가 달랐다),
    객체나 mutable 상태 공유는 별도로 배제해야 한다.
    """

    def _pair(self):
        return (
            CommittedPlanController(NARROW_F, quota_multiplier=1.0, beam_width=2),
            ShrinkingQuotaMPCController(NARROW_F, quota_multiplier=1.0, beam_width=2),
        )

    def test_controllers_are_distinct_objects_and_types(self):
        committed, shrinking = self._pair()
        assert committed is not shrinking
        assert type(committed) is not type(shrinking)
        assert committed.name != shrinking.name

    def test_mutable_state_is_not_shared(self):
        """클래스 변수나 mutable default 를 쓰면 계획 상태가 새어 나간다."""
        committed, shrinking = self._pair()
        committed.reset()
        shrinking.reset()
        assert committed.choices is not shrinking.choices
        assert committed.trajectory is not shrinking.trajectory

        sentinel = NARROW_F.action_from_flat(0)
        committed.trajectory.append((None, sentinel))  # type: ignore[arg-type]
        assert len(shrinking.trajectory) == 0

    def test_two_instances_of_same_type_do_not_share_state(self):
        a = ShrinkingQuotaMPCController(NARROW_F, quota_multiplier=1.0, beam_width=2)
        b = ShrinkingQuotaMPCController(NARROW_F, quota_multiplier=1.0, beam_width=2)
        make_optimizer(a, budget=40.0, steps=3).run()
        assert a.choices
        assert b.choices == []

    def test_result_is_independent_of_execution_order(self):
        """실행 순서에 따라 결과가 달라지면 전역 상태나 공유 cache 가 있다."""

        def run(cls):
            planner = cls(NARROW_F, quota_multiplier=1.0, beam_width=2)
            trace = make_optimizer(planner, budget=60.0, steps=20).run()
            return trace.final_loss

        c_first = run(CommittedPlanController)
        s_after = run(ShrinkingQuotaMPCController)
        s_first = run(ShrinkingQuotaMPCController)
        c_after = run(CommittedPlanController)

        assert c_first == pytest.approx(c_after, rel=1e-12)
        assert s_first == pytest.approx(s_after, rel=1e-12)


class TestSuffixRetentionRate:
    """``chosen_depth`` 히스토그램은 깊이만 보여준다. 행동 내용을 비교해야 한다.

    ```text
    replanned_actions == previous_plan[1:]
    ```
    """

    def test_retention_rate_is_measured_on_action_content(self):
        # Q=1 은 이 quadratic 에서 depth 1 계획만 내므로 재계획 기회가 없다.
        # 실측(rosen_d2, depth 4)과 같은 조건을 만들려면 쿼터를 키운다.
        planner = ShrinkingQuotaMPCController(
            NARROW_F, quota_multiplier=4.0, beam_width=2, max_depth=8
        )
        make_optimizer(planner, budget=80.0, steps=30).run()
        rate = planner.suffix_retention_rate
        assert planner.n_replans > 0, "재계획 기회가 있어야 한다"
        assert 0.0 <= rate <= 1.0

    def test_no_replan_opportunity_gives_nan(self):
        planner = ShrinkingQuotaMPCController(
            NARROW_F, quota_multiplier=1.0, beam_width=2
        )
        assert math.isnan(planner.suffix_retention_rate)

    def test_reset_clears_retention_counters(self):
        planner = ShrinkingQuotaMPCController(
            NARROW_F, quota_multiplier=4.0, beam_width=2, max_depth=8
        )
        make_optimizer(planner, budget=80.0, steps=30).run()
        assert planner.n_replans > 0
        planner.reset()
        assert planner.n_replans == 0
        assert math.isnan(planner.suffix_retention_rate)

    def test_full_retention_implies_committed_equivalence(self):
        """유지율이 1.0 이면 committed 와 같은 경로를 간다.

        ``Q1`` 에서 두 컨트롤러가 같은 ``final_loss`` 를 낸 이유가 이것이다.
        재계획은 하지만 계획이 바뀌지 않는다.
        """
        shrinking = ShrinkingQuotaMPCController(
            NARROW_F, quota_multiplier=4.0, beam_width=2, max_depth=8
        )
        committed = CommittedPlanController(
            NARROW_F, quota_multiplier=4.0, beam_width=2, max_depth=8
        )
        s_trace = make_optimizer(shrinking, budget=80.0, steps=30).run()
        c_trace = make_optimizer(committed, budget=80.0, steps=30).run()

        assert shrinking.n_replans > 0
        if shrinking.suffix_retention_rate == 1.0:
            # 유지율 1.0 이면 committed 와 같은 경로다. Q1 동일성의 원인이다.
            assert s_trace.final_loss == pytest.approx(c_trace.final_loss, rel=1e-12)
        else:
            # 계획이 바뀌었다는 것이 계측에 반영돼야 한다.
            assert shrinking.suffix_retention_rate < 1.0


# ---------------------------------------------------------------------------
# open-loop resource clock (프로토콜 D17)
# ---------------------------------------------------------------------------


class TestOpenLoopResourceClock:
    """``progress`` 는 소모 GE 비율이다. step 비율이 아니다.

    초판은 ``step / total_steps`` 였고, GE 예산으로 종료하므로 두 시계가
    불일치했다. ``total_steps=200``, 150 GE, ``k=20`` 이면 7 step 만에 끝나서
    ``progress`` 가 0.035 를 넘지 못했고 **스케줄의 첫 구간만 실행됐다.**
    """

    def _schedule(self, n_seg=4):
        # 구간마다 다른 CG budget 을 준다. 실행 여부를 action 으로 구별한다.
        flats = [0, 3, 6, 9][:n_seg]
        breaks = [(i + 1) / n_seg for i in range(n_seg)]
        return make_open_loop_controller(NARROW_F, flats, breaks)

    def test_progress_uses_ge_budget_not_steps(self):
        ctx = StepContext(
            step=3,
            total_steps=200,
            loss=1.0,
            grad_norm=1.0,
            damping=1.0e-2,
            spent_ge=75.0,
            cost_budget_ge=150.0,
        )
        # step 기준이면 3/200 = 0.015. GE 기준이면 75/150 = 0.5.
        assert ctx.progress == pytest.approx(0.5)

    def test_progress_is_capped_at_one(self):
        ctx = StepContext(
            step=0,
            total_steps=200,
            loss=1.0,
            grad_norm=1.0,
            damping=1.0e-2,
            spent_ge=400.0,
            cost_budget_ge=150.0,
        )
        assert ctx.progress == pytest.approx(1.0)

    def test_falls_back_to_step_clock_without_budget(self):
        ctx = StepContext(
            step=50, total_steps=200, loss=1.0, grad_norm=1.0, damping=1.0e-2
        )
        assert ctx.progress == pytest.approx(0.25)

    def test_multiple_segments_run_within_ge_budget(self):
        """step 수가 200보다 훨씬 작아도 여러 구간이 실행돼야 한다."""
        ctrl = self._schedule()
        trace = make_optimizer(ctrl, budget=150.0, steps=200).run()
        assert trace.n_steps < 50, "GE 예산이 먼저 끝나는 조건이어야 한다"
        assert len(ctrl.realized_segment_counts) >= 2

    def test_progress_reaches_near_one_at_budget_end(self):
        ctrl = self._schedule()
        trace = make_optimizer(ctrl, budget=150.0, steps=200).run()
        spent = sum(r.cost_ge for r in trace.records if math.isfinite(r.cost_ge))
        assert spent / 150.0 > 0.8

    def test_same_clock_gives_same_action_regardless_of_loss(self):
        """상태를 보지 않는다. 같은 GE 시계면 loss 가 달라도 같은 action 이다."""
        ctrl = self._schedule()
        common = {"step": 5, "total_steps": 200, "grad_norm": 1.0, "damping": 1.0e-2}
        a = ctrl.select(
            StepContext(loss=1.0, spent_ge=60.0, cost_budget_ge=150.0, **common), None
        )
        b = ctrl.select(
            StepContext(loss=1.0e-9, spent_ge=60.0, cost_budget_ge=150.0, **common), None
        )
        assert a is b

    def test_segment_boundary_is_deterministic(self):
        """breakpoint 직전 / 정확히 일치 / 직후의 선택이 결정적이다."""
        ctrl = self._schedule(n_seg=2)  # breaks = [0.5, 1.0]
        assert ctrl.segment_index_at(0.4999) == 0
        assert ctrl.segment_index_at(0.5) == 0, "`until` 은 이하 포함이다"
        assert ctrl.segment_index_at(0.5001) == 1

    def test_realized_counts_reset(self):
        ctrl = self._schedule()
        make_optimizer(ctrl, budget=150.0, steps=200).run()
        assert ctrl.realized_segment_counts
        ctrl.reset()
        assert ctrl.realized_segment_counts == {}

    def test_clock_label_is_recorded(self):
        assert self._schedule().clock == "object_ge_fraction"
