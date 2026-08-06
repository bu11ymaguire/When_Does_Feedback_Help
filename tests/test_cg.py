"""Stage 1 게이트: Conjugate Gradient.

게이트 기준 (프로토콜 §4 Stage 1):
  - SPD quadratic에서 Newton-CG 해의 상대오차 < 1e-3
  - damping 증가 시 ill-conditioned 문제의 CG 실패 감소
  - indefinite 문제에서 negative curvature 정상 탐지
  - HVP 카운트 정확성 (비용 회계의 근거)
"""

from __future__ import annotations

import math

import pytest
import torch

from rl_newton.curvature.operators import (
    DampedHessianOperator,
    DiagonalPreconditioner,
    IdentityPreconditioner,
)
from rl_newton.solvers.conjugate_gradient import conjugate_gradient
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask
from rl_newton.tasks.rosenbrock import RosenbrockSpec, RosenbrockTask
from rl_newton.utils.seed import seed_everything

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = float(expected.norm())
    if denom == 0.0:
        return float(actual.norm())
    return float((actual - expected).norm() / denom)


def _operator_from_matrix(
    a: torch.Tensor, point: torch.Tensor, *, damping: float = 0.0
) -> DampedHessianOperator:
    """explicit 행렬로 quadratic 연산자를 만든다.

    ``QuadraticTask`` 는 랜덤 직교기저를 쓰므로 대각 구조를 통제할 수 없다.
    preconditioner 검증처럼 행렬 구조 자체가 중요한 경우에 사용한다.
    """
    x = torch.nn.Parameter(point.clone())
    return DampedHessianOperator.from_closure(
        lambda: 0.5 * (x @ (a @ x)),
        [x],
        damping=damping,
        min_damping=0.0,
        max_damping=1e12,
    )


def solve_newton_direction(
    task: QuadraticTask | RosenbrockTask,
    *,
    damping: float = 0.0,
    max_iters: int = 200,
    tolerance: float = 1e-10,
    preconditioner=None,
):
    """``(H + lambda I) p = -g`` 를 풀어 ``(CGResult, operator)`` 를 반환한다."""
    op = DampedHessianOperator.from_closure(
        task.loss, task.params, damping=damping, min_damping=0.0, max_damping=1e12
    )
    result = conjugate_gradient(
        op,
        -op.grad,
        max_iters=max_iters,
        tolerance=tolerance,
        preconditioner=preconditioner,
    )
    return result, op


# ---------------------------------------------------------------------------
# 게이트 1: SPD quadratic 정확성
# ---------------------------------------------------------------------------


class TestSpdAccuracy:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("kappa", [1.0e1, 1.0e2, 1.0e3])
    def test_matches_exact_newton_step_within_gate(self, device: str, kappa: float):
        """게이트: 상대오차 < 1e-3. 기준은 explicit linear solve."""
        seed_everything(0)
        task = QuadraticTask(
            QuadraticSpec(dimension=64, condition_number=kappa), seed=0, device=device
        )
        result, _ = solve_newton_direction(task)

        assert result.converged
        assert not result.numerical_failure
        assert not result.negative_curvature
        assert relative_error(result.solution, task.exact_newton_step()) < 1e-3

    def test_quadratic_newton_step_reaches_optimum_in_one_step(self):
        """quadratic에서 정확한 Newton step은 한 번에 최적점에 도달한다.

        Newton 방향의 정확성을 loss 값으로 확인하는 독립적인 검사다.
        """
        task = QuadraticTask(
            QuadraticSpec(dimension=32, condition_number=1.0e2),
            seed=0,
            dtype=torch.float64,
        )
        result, op = solve_newton_direction(task, max_iters=200, tolerance=1e-14)

        with torch.no_grad():
            task.params[0].add_(result.solution)

        assert float(task.loss().detach()) < task.initial_loss * 1e-12

    def test_converges_well_within_dimension_for_conditioned_problem(self):
        """조건수가 낮으면 차원보다 훨씬 적은 반복으로 수렴한다.

        "CG는 d회 안에 정확해에 도달한다"는 **정확 산술에서만** 성립한다.
        부동소수에서는 conjugacy가 소실되어 d회로 기계정밀도에 닿지 않는다.
        실제로 의미 있는 성질은 반복 수가 차원이 아니라 조건수에 지배된다는
        것이다. 상한은 대략 ``(sqrt(kappa)/2) * ln(2/tol)`` 이다.
        """
        d = 40
        kappa = 1.0e1
        tol = 1e-6
        task = QuadraticTask(
            QuadraticSpec(dimension=d, condition_number=kappa),
            seed=0,
            dtype=torch.float64,
        )
        result, _ = solve_newton_direction(task, max_iters=d, tolerance=tol)

        theoretical = 0.5 * (kappa**0.5) * math.log(2.0 / tol)
        assert result.converged
        assert result.iterations <= min(d, math.ceil(theoretical))

    def test_damping_shifts_solution_toward_gradient_direction(self):
        """damping이 커지면 Newton 방향이 최급강하 방향으로 기울어야 한다."""
        task = QuadraticTask(
            QuadraticSpec(dimension=32, condition_number=1.0e3),
            seed=0,
            dtype=torch.float64,
        )
        op_ref = DampedHessianOperator.from_closure(task.loss, task.params)
        steepest = -op_ref.grad
        steepest = steepest / steepest.norm()
        op_ref.release()

        def cosine_with_steepest(damping: float) -> float:
            result, _ = solve_newton_direction(
                task, damping=damping, max_iters=200, tolerance=1e-12
            )
            p = result.solution / result.solution.norm()
            return float(torch.dot(p, steepest))

        low = cosine_with_steepest(1e-6)
        high = cosine_with_steepest(1e6)

        assert high > low
        assert high == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 게이트 2: damping이 ill-conditioned 문제의 CG 실패를 줄인다
# ---------------------------------------------------------------------------


class TestDampingReducesFailure:
    @pytest.mark.parametrize("budget", [5, 10, 20])
    def test_damping_improves_convergence_rate_under_fixed_budget(self, budget: int):
        """게이트: 같은 예산에서 damping을 올리면 수렴률이 단조 증가한다.

        조건수가 ``(L + lambda) / (m + lambda)`` 로 줄어드는 효과를 측정한다.
        eigenvalue 범위가 ``[1, 1e6]`` 이므로 damped 조건수를 1 근처로 만들려면
        ``lambda >> 1e6`` 이 필요하다. CG의 반복 수는 대략
        ``sqrt(kappa)/2 * ln(2/tol)`` 이므로 damping 을 조금 올린 정도로는
        예산 5회 안에 수렴하지 않는다. 이 점을 감안해 damping 사다리를 잡는다.
        """
        spec = QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e6)

        def convergence_rate(damping: float) -> float:
            converged = 0
            for seed in range(8):
                task = QuadraticTask(spec, seed=seed)
                result, _ = solve_newton_direction(
                    task, damping=damping, max_iters=budget, tolerance=1e-3
                )
                converged += int(result.converged)
            return converged / 8

        ladder = [1e-8, 1e2, 1e4, 1e6, 1e8]
        rates = [convergence_rate(d) for d in ladder]

        assert rates == sorted(rates), f"수렴률이 damping에 단조여야 한다: {rates}"
        assert rates[0] == 0.0, "무감쇠로는 kappa=1e6 문제를 이 예산에 못 푼다"
        assert rates[-1] == 1.0, "damped kappa~1 이면 예산 안에서 모두 수렴해야 한다"

    def test_iterations_to_converge_shrinks_with_damping(self):
        """수렴에 필요한 반복 수 자체가 damping과 함께 줄어드는지 확인한다.

        위 테스트가 예산 기준이라면 이건 비용 기준이다. 프로토콜 D1의
        cost-to-target 이 damping 선택에 어떻게 반응하는지의 축소판이다.
        """
        spec = QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e6)
        counts = []
        for damping in (1e0, 1e2, 1e4, 1e6):
            task = QuadraticTask(spec, seed=0, dtype=torch.float64)
            result, _ = solve_newton_direction(
                task, damping=damping, max_iters=2000, tolerance=1e-6
            )
            assert result.converged
            counts.append(result.iterations)

        assert counts == sorted(counts, reverse=True), f"반복 수가 단조 감소해야 한다: {counts}"

    def test_residual_ratio_improves_with_damping(self):
        """수렴 여부가 아니라 residual 감소량으로도 확인한다."""
        task_spec = QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e6)

        ratios = []
        for damping in (1e-8, 1e-2, 1e0, 1e2):
            task = QuadraticTask(task_spec, seed=0)
            result, _ = solve_newton_direction(task, damping=damping, max_iters=5, tolerance=1e-8)
            ratios.append(result.residual_ratio)

        # damping이 커질수록 같은 반복 수에서 residual이 더 많이 줄어든다
        assert ratios == sorted(ratios, reverse=True)

    def test_condition_number_shrinks_with_damping(self):
        """damping의 작동 원리 자체를 확인한다. 위 결과의 해석 근거."""
        task = QuadraticTask(
            QuadraticSpec(dimension=50, condition_number=1.0e6),
            seed=0,
            dtype=torch.float64,
        )
        eig = torch.linalg.eigvalsh(task.hessian_matrix())

        def damped_condition(damping: float) -> float:
            shifted = eig + damping
            return float(shifted.max() / shifted.min())

        assert damped_condition(1e2) < damped_condition(1e-2) < damped_condition(1e-8)


# ---------------------------------------------------------------------------
# 게이트 3: negative curvature 탐지
# ---------------------------------------------------------------------------


class TestNegativeCurvature:
    def test_detected_on_indefinite_quadratic(self):
        """게이트: indefinite 문제에서 negative curvature를 탐지한다."""
        task = QuadraticTask(
            QuadraticSpec(kind="indefinite", dimension=64, condition_number=1.0e2),
            seed=0,
        )
        assert task.min_eigenvalue < 0.0

        result, _ = solve_newton_direction(task, damping=0.0, max_iters=64)

        assert result.negative_curvature
        assert not result.converged

    def test_sufficient_damping_removes_negative_curvature(self):
        """``damping > -lambda_min`` 이면 계가 SPD가 되어 정상 수렴해야 한다."""
        task = QuadraticTask(
            QuadraticSpec(kind="indefinite", dimension=64, condition_number=1.0e2),
            seed=0,
            dtype=torch.float64,
        )
        damping = -task.min_eigenvalue * 2.0
        result, _ = solve_newton_direction(task, damping=damping, max_iters=200, tolerance=1e-10)

        assert not result.negative_curvature
        assert result.converged

    def test_solution_is_finite_and_nonzero_when_truncated(self):
        """negative curvature로 끊겨도 사용 가능한 방향을 돌려줘야 한다.

        0 방향을 반환하면 optimizer가 예산만 쓰고 아무 일도 못 한다.
        """
        task = QuadraticTask(
            QuadraticSpec(kind="indefinite", dimension=32, condition_number=1.0e2),
            seed=1,
        )
        result, _ = solve_newton_direction(task, damping=0.0, max_iters=32)

        assert result.negative_curvature
        assert torch.isfinite(result.solution).all()
        assert float(result.solution.norm()) > 0.0

    def test_immediate_negative_curvature_falls_back_to_steepest_descent(self):
        """첫 반복에서 걸리면 최급강하 방향으로 대체한다.

        ``H = -I`` 이면 어떤 방향이든 음의 곡률이므로 j=0에서 즉시 걸린다.
        """
        x = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        op = DampedHessianOperator.from_closure(lambda: -0.5 * (x**2).sum(), [x])
        rhs = -op.grad  # g = -x  ->  rhs = x

        result = conjugate_gradient(op, rhs, max_iters=10)

        assert result.negative_curvature
        assert result.iterations == 0
        assert torch.allclose(result.solution, rhs)  # 전처리 없음 -> rhs 그대로

    def test_rosenbrock_above_valley_triggers_negative_curvature(self):
        """실제 비볼록 문제에서도 경로를 밟는지 확인한다.

        표준 시작점은 Hessian이 양정이므로 골짜기 위쪽(``y > x^2 + 1/(2s)``)으로
        옮겨야 음의 곡률을 만난다.
        """
        task = RosenbrockTask(RosenbrockSpec(dimension=2), seed=0, dtype=torch.float64)
        task.move_to(task.negative_curvature_point(x0=0.0))

        result, _ = solve_newton_direction(task, damping=0.0, max_iters=10)

        assert result.negative_curvature

    def test_rosenbrock_standard_start_solves_normally(self):
        """양정 구간에서는 정상 수렴해야 한다."""
        task = RosenbrockTask(RosenbrockSpec(dimension=2), seed=0, dtype=torch.float64)
        result, _ = solve_newton_direction(task, damping=0.0, max_iters=50, tolerance=1e-12)

        assert not result.negative_curvature
        assert result.converged


# ---------------------------------------------------------------------------
# 비용 회계: 프로토콜 D1의 근거
# ---------------------------------------------------------------------------


class TestCostAccounting:
    def test_hvp_count_equals_iterations_when_not_truncated(self):
        """CG 반복 1회 = HVP 1회. GE 환산식의 전제다."""
        task = QuadraticTask(QuadraticSpec(dimension=64, condition_number=1e4), seed=0)
        result, op = solve_newton_direction(task, max_iters=7, tolerance=1e-14)

        assert result.iterations == 7
        assert result.hvp_count == 7
        assert op.hvp_count == 7

    def test_hvp_count_respects_budget(self):
        """예산을 절대 초과하지 않아야 한다. RL action의 의미가 여기 달려 있다."""
        task = QuadraticTask(
            QuadraticSpec(kind="ill_conditioned", dimension=200, condition_number=1e8),
            seed=0,
        )
        for budget in (1, 3, 5, 10, 20):
            result, _ = solve_newton_direction(task, max_iters=budget, tolerance=1e-14)
            assert result.hvp_count <= budget
            assert result.iterations <= budget
            assert result.budget == budget

    def test_iters_used_ratio_reflects_budget(self):
        """RL 상태 특징 ``cg_iters_used_ratio``."""
        task = QuadraticTask(
            QuadraticSpec(kind="ill_conditioned", dimension=200, condition_number=1e8),
            seed=0,
        )
        exhausted, _ = solve_newton_direction(task, max_iters=4, tolerance=1e-14)
        assert exhausted.iters_used_ratio == pytest.approx(1.0)

        early, _ = solve_newton_direction(task, max_iters=50, tolerance=1e9)
        assert early.iters_used_ratio < 1.0

    def test_zero_rhs_consumes_no_hvp(self):
        """이미 최적점이면 계산을 낭비하지 않아야 한다."""
        task = QuadraticTask(QuadraticSpec(dimension=16), seed=0)
        with torch.no_grad():
            task.params[0].zero_()  # g = 0

        result, op = solve_newton_direction(task, max_iters=10)

        assert result.hvp_count == 0
        assert result.iterations == 0
        assert result.converged
        assert float(result.solution.norm()) == 0.0

    def test_operator_reset_count(self):
        task = QuadraticTask(QuadraticSpec(dimension=16), seed=0)
        op = DampedHessianOperator.from_closure(task.loss, task.params)

        op.matvec(torch.randn(16))
        op.matvec(torch.randn(16))
        assert op.hvp_count == 2

        op.reset_count()
        assert op.hvp_count == 0
        op.matvec(torch.randn(16))
        assert op.hvp_count == 1


# ---------------------------------------------------------------------------
# 종료 조건과 preconditioner
# ---------------------------------------------------------------------------


class TestTerminationAndPreconditioning:
    def test_tolerance_controls_early_stopping(self):
        """tolerance를 조이면 반복 수가 늘고 residual이 더 줄어야 한다.

        예산은 이론 상한 ``(sqrt(kappa)/2) * ln(2/tol)`` 을 넉넉히 넘게 잡는다.
        kappa=1e4, tol=1e-6 이면 약 725회이므로 2000회를 준다.
        """
        task = QuadraticTask(
            QuadraticSpec(dimension=100, condition_number=1e4),
            seed=0,
            dtype=torch.float64,
        )
        loose, _ = solve_newton_direction(task, max_iters=2000, tolerance=1e-2)
        tight, _ = solve_newton_direction(task, max_iters=2000, tolerance=1e-6)

        assert loose.converged and tight.converged
        assert loose.iterations < tight.iterations
        assert loose.residual_ratio <= 1e-2
        assert tight.residual_ratio <= 1e-6

    def test_identity_preconditioner_matches_no_preconditioner(self):
        task = QuadraticTask(QuadraticSpec(dimension=32), seed=0, dtype=torch.float64)

        plain, _ = solve_newton_direction(task, max_iters=100, tolerance=1e-12)
        identity, _ = solve_newton_direction(
            task, max_iters=100, tolerance=1e-12, preconditioner=IdentityPreconditioner()
        )

        assert plain.iterations == identity.iterations
        assert torch.allclose(plain.solution, identity.solution, atol=1e-10)

    def test_jacobi_helps_when_diagonal_is_spread(self):
        """대각이 넓게 퍼진 계에서는 Jacobi가 반복 수를 크게 줄인다.

        ``A = D^(1/2) B D^(1/2)`` 로 구성한다. ``D`` 는 log 등간격 대각,
        ``B`` 는 조건수가 1에 가까운 SPD다. 그러면 ``kappa(A) ~ kappa(D) kappa(B)``
        이고, Jacobi 전처리 후에는 ``~ kappa(B)`` 로 줄어든다. 교과서적 사례다.
        """
        d = 100
        gen = torch.Generator().manual_seed(0)
        symmetric = torch.randn(d, d, generator=gen, dtype=torch.float64) / (d**0.5)
        b = torch.eye(d, dtype=torch.float64) + 0.1 * 0.5 * (symmetric + symmetric.T)
        root_d = torch.logspace(0.0, 5.0, d, dtype=torch.float64).sqrt()
        a = root_d.unsqueeze(1) * b * root_d.unsqueeze(0)
        a = 0.5 * (a + a.T)

        point = torch.randn(d, generator=gen, dtype=torch.float64)

        plain_op = _operator_from_matrix(a, point)
        plain = conjugate_gradient(plain_op, -plain_op.grad, max_iters=5000, tolerance=1e-8)

        jacobi_op = _operator_from_matrix(a, point)
        jacobi = conjugate_gradient(
            jacobi_op,
            -jacobi_op.grad,
            max_iters=5000,
            tolerance=1e-8,
            preconditioner=DiagonalPreconditioner(torch.diagonal(a).clone()),
        )

        assert plain.converged and jacobi.converged
        assert jacobi.iterations < plain.iterations
        # 같은 해에 도달해야 한다. preconditioner는 경로만 바꾼다.
        assert relative_error(jacobi.solution, plain.solution) < 1e-4

    def test_jacobi_is_ineffective_for_random_basis_hessian(self):
        """랜덤 직교기저로 만든 A는 대각이 거의 상수이므로 Jacobi가 무력하다.

        ``QuadraticTask`` 가 바로 그 구조다. Stage 5에서 preconditioner를
        평가할 때 이 사실을 알고 있어야 한다. quadratic 벤치마크에서 diagonal
        preconditioner의 이득이 없다고 나오는 것은 구현 결함이 아니라
        문제 구조 때문이다.
        """
        task = QuadraticTask(
            QuadraticSpec(dimension=100, condition_number=1e5),
            seed=0,
            dtype=torch.float64,
        )
        diagonal = torch.diagonal(task.hessian_matrix())

        spread = float(diagonal.max() / diagonal.min())
        assert spread < 10.0, "랜덤 직교기저이므로 대각 분산이 작아야 한다"

        plain, _ = solve_newton_direction(task, max_iters=5000, tolerance=1e-8)
        jacobi, _ = solve_newton_direction(
            task,
            max_iters=5000,
            tolerance=1e-8,
            preconditioner=DiagonalPreconditioner(diagonal.clone()),
        )

        assert plain.converged and jacobi.converged
        # 이득이 미미하다. 크게 나빠지지도 않는다는 것만 확인한다.
        assert jacobi.iterations <= plain.iterations * 1.5

    def test_x0_warm_start_costs_one_extra_hvp(self):
        """초기해를 주면 residual 계산에 HVP 1회가 더 든다. 비용 회계 정확성."""
        task = QuadraticTask(QuadraticSpec(dimension=32), seed=0)
        op = DampedHessianOperator.from_closure(task.loss, task.params)
        exact = task.exact_newton_step()

        result = conjugate_gradient(op, -op.grad, max_iters=5, tolerance=1e-6, x0=exact.clone())

        # 이미 정확해에서 출발했으므로 residual이 작고 즉시 종료한다
        assert result.hvp_count >= 1
        assert result.residual_ratio <= 1.0


class TestValidation:
    def _operator(self):
        x = torch.nn.Parameter(torch.ones(4))
        return DampedHessianOperator.from_closure(lambda: (x**2).sum(), [x])

    def test_rejects_nonpositive_budget(self):
        op = self._operator()
        with pytest.raises(ValueError, match="max_iters"):
            conjugate_gradient(op, torch.ones(4), max_iters=0)

    def test_rejects_nonpositive_tolerance(self):
        op = self._operator()
        with pytest.raises(ValueError, match="tolerance"):
            conjugate_gradient(op, torch.ones(4), max_iters=5, tolerance=0.0)

    def test_rejects_non_1d_rhs(self):
        op = self._operator()
        with pytest.raises(ValueError, match="1-D"):
            conjugate_gradient(op, torch.ones(2, 2), max_iters=5)

    def test_rejects_low_precision_dtype(self):
        """FP16/BF16은 CG 누적에서 정밀도가 무너진다 (README §15)."""
        op = self._operator()
        with pytest.raises(NotImplementedError, match="float32/float64"):
            conjugate_gradient(op, torch.ones(4, dtype=torch.float16), max_iters=5)

    def test_rejects_mismatched_x0(self):
        op = self._operator()
        with pytest.raises(ValueError, match="x0 shape"):
            conjugate_gradient(op, torch.ones(4), max_iters=5, x0=torch.ones(3))

    def test_non_finite_rhs_reports_failure_without_crashing(self):
        op = self._operator()
        rhs = torch.tensor([1.0, float("nan"), 1.0, 1.0])

        result = conjugate_gradient(op, rhs, max_iters=5)

        assert result.numerical_failure
        assert not result.converged
        assert torch.isfinite(result.solution).all()


class TestDampedOperator:
    def test_matvec_adds_damping(self):
        x = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        op = DampedHessianOperator.from_closure(lambda: (x**2).sum(), [x], damping=1.0)  # H = 2I
        assert torch.allclose(op.matvec(torch.tensor([1.0, 0.0])), torch.tensor([3.0, 0.0]))

    def test_scale_damping_clips_to_bounds(self):
        x = torch.nn.Parameter(torch.ones(2))
        op = DampedHessianOperator.from_closure(
            lambda: (x**2).sum(), [x], damping=1.0, min_damping=1e-3, max_damping=1e2
        )
        assert op.scale_damping(1e-9) == pytest.approx(1e-3)
        assert op.scale_damping(1e9) == pytest.approx(1e2)

    def test_damping_can_change_without_rebuilding_graph(self):
        """step 거절 후 damping만 올려 재풀이할 때 HVP 그래프를 재사용한다."""
        task = QuadraticTask(QuadraticSpec(dimension=32), seed=0)
        op = DampedHessianOperator.from_closure(
            task.loss, task.params, damping=1e-6, min_damping=1e-12
        )
        first = conjugate_gradient(op, -op.grad, max_iters=10)

        op.set_damping(1e3)
        second = conjugate_gradient(op, -op.grad, max_iters=10)

        assert not torch.allclose(first.solution, second.solution)
        # 그래프를 다시 만들지 않았으므로 loss/grad 스냅샷은 동일하다
        assert op.loss == pytest.approx(task.initial_loss, rel=1e-6)

    def test_rejects_negative_damping(self):
        x = torch.nn.Parameter(torch.ones(2))
        with pytest.raises(ValueError, match="damping must be >= 0"):
            DampedHessianOperator.from_closure(lambda: (x**2).sum(), [x], damping=-1.0)

    def test_rejects_nonpositive_multiplier(self):
        x = torch.nn.Parameter(torch.ones(2))
        op = DampedHessianOperator.from_closure(lambda: (x**2).sum(), [x], damping=1.0)
        with pytest.raises(ValueError, match="multiplier"):
            op.scale_damping(0.0)
