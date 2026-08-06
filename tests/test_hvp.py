"""Stage 1 게이트: HVP 정확성.

프로젝트 전체가 double-backward HVP에 의존한다. 여기서 틀리면 CG 해도,
trust ratio도, RL 상태 특징도 전부 조용히 틀린다. 그래서 explicit Hessian과
직접 대조한다.

게이트 기준 (프로토콜 §4 Stage 1):
  - explicit Hessian 대조 상대오차 < 1e-5
  - ill-conditioned FP32 에서는 1e-4 허용
  - 파라미터 shape 보존
  - unused parameter 처리
"""

from __future__ import annotations

import pytest
import torch

from rl_newton.curvature.hvp import HvpGraph, hessian_vector_product
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask
from rl_newton.tasks.rosenbrock import RosenbrockSpec, RosenbrockTask
from rl_newton.utils.flatten import ParameterFlattener
from rl_newton.utils.seed import seed_everything

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = float(expected.norm())
    if denom == 0.0:
        return float(actual.norm())
    return float((actual - expected).norm() / denom)


# ---------------------------------------------------------------------------
# quadratic: Hessian이 정확히 A 이므로 가장 엄격한 대조가 가능하다
# ---------------------------------------------------------------------------


class TestQuadraticExactMatch:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("kappa", [1.0e1, 1.0e2, 1.0e3])
    def test_matches_explicit_hessian_within_gate(self, device: str, kappa: float):
        """게이트: 상대오차 < 1e-5."""
        seed_everything(0)
        task = QuadraticTask(
            QuadraticSpec(dimension=64, condition_number=kappa), seed=0, device=device
        )
        graph = HvpGraph(task.loss, task.params)

        v = torch.randn(graph.numel, device=device)
        hv = graph.matvec(v)
        expected = task.hessian_matrix() @ v

        assert relative_error(hv, expected) < 1e-5

    @pytest.mark.parametrize("device", DEVICES)
    def test_ill_conditioned_within_relaxed_gate(self, device: str):
        """게이트: ill-conditioned FP32 에서는 1e-4 허용."""
        seed_everything(0)
        task = QuadraticTask(
            QuadraticSpec(kind="ill_conditioned", dimension=64, condition_number=1.0e5),
            seed=0,
            device=device,
        )
        graph = HvpGraph(task.loss, task.params)

        v = torch.randn(graph.numel, device=device)
        assert relative_error(graph.matvec(v), task.hessian_matrix() @ v) < 1e-4

    def test_float64_is_far_more_accurate(self):
        """FP64에서는 오차가 거의 사라진다. 오차원이 부동소수라는 확인."""
        task = QuadraticTask(
            QuadraticSpec(dimension=64, condition_number=1.0e5),
            seed=0,
            dtype=torch.float64,
        )
        graph = HvpGraph(task.loss, task.params)
        v = torch.randn(graph.numel, dtype=torch.float64)

        assert relative_error(graph.matvec(v), task.hessian_matrix() @ v) < 1e-12

    def test_indefinite_hessian_reproduced_including_sign(self):
        """음의 곡률 방향에서 부호까지 맞는지 확인한다."""
        task = QuadraticTask(
            QuadraticSpec(kind="indefinite", dimension=32, condition_number=1.0e2),
            seed=3,
        )
        assert task.min_eigenvalue < 0.0

        graph = HvpGraph(task.loss, task.params)
        eigvals, eigvecs = torch.linalg.eigh(task.hessian_matrix())
        v = eigvecs[:, 0].contiguous()  # 최소 고유값 방향

        hv = graph.matvec(v)
        assert relative_error(hv, eigvals[0] * v) < 1e-4
        assert float(torch.dot(v, hv)) < 0.0  # v^T H v < 0

    def test_linearity_in_v(self):
        """``H(av + bw) = a Hv + b Hw``. 연산자가 선형인지 확인한다."""
        task = QuadraticTask(QuadraticSpec(dimension=32), seed=1)
        graph = HvpGraph(task.loss, task.params)

        v = torch.randn(graph.numel)
        w = torch.randn(graph.numel)
        combined = graph.matvec(2.0 * v - 3.0 * w)
        separate = 2.0 * graph.matvec(v) - 3.0 * graph.matvec(w)

        assert relative_error(combined, separate) < 1e-5

    def test_symmetry(self):
        """``v^T H w == w^T H v``. CG는 A의 대칭성을 가정한다."""
        task = QuadraticTask(QuadraticSpec(dimension=32), seed=1)
        graph = HvpGraph(task.loss, task.params)

        v = torch.randn(graph.numel)
        w = torch.randn(graph.numel)
        vhw = float(torch.dot(v, graph.matvec(w)))
        whv = float(torch.dot(w, graph.matvec(v)))

        assert vhw == pytest.approx(whv, rel=1e-4)


# ---------------------------------------------------------------------------
# Rosenbrock: Hessian이 위치에 의존하는 경우
# ---------------------------------------------------------------------------


class TestRosenbrock:
    @pytest.mark.parametrize("dimension", [2, 10])
    def test_matches_autograd_hessian(self, dimension: int):
        """``autograd.functional.hessian`` 은 우리 구현과 독립적인 대조군이다."""
        task = RosenbrockTask(RosenbrockSpec(dimension=dimension), seed=0, dtype=torch.float64)
        graph = HvpGraph(task.loss, task.params)

        v = torch.randn(graph.numel, dtype=torch.float64)
        assert relative_error(graph.matvec(v), task.hessian_matrix() @ v) < 1e-10

    def test_standard_start_is_positive_definite(self):
        """표준 시작점 (-1.2, 1.0) 은 골짜기 아래쪽이라 Hessian이 양정이다.

        det H = 8 s^2 (x^2 - y) + 4 s 이고 x^2 = 1.44 > y = 1.0 이므로 det > 0.
        시작점부터 음의 곡률을 밟는다고 가정하면 안 된다.
        """
        task = RosenbrockTask(RosenbrockSpec(dimension=2), seed=0, dtype=torch.float64)
        eigvals = torch.linalg.eigvalsh(task.hessian_matrix())

        assert float(eigvals.min()) > 0.0

    def test_above_valley_has_negative_curvature(self):
        """``y > x^2 + 1/(2s)`` 영역에서는 Hessian이 indefinite 하다.

        negative curvature 경로가 실제로 존재한다는 근거다.
        """
        task = RosenbrockTask(RosenbrockSpec(dimension=2), seed=0, dtype=torch.float64)
        task.move_to(task.negative_curvature_point(x0=0.0))
        eigvals = torch.linalg.eigvalsh(task.hessian_matrix())

        assert float(eigvals.min()) < 0.0

    def test_hvp_matches_hessian_at_indefinite_point(self):
        """음의 곡률 지점에서도 HVP가 정확한지 확인한다."""
        task = RosenbrockTask(RosenbrockSpec(dimension=4), seed=0, dtype=torch.float64)
        task.move_to(task.negative_curvature_point(x0=0.5))

        graph = HvpGraph(task.loss, task.params)
        v = torch.randn(graph.numel, dtype=torch.float64)

        assert relative_error(graph.matvec(v), task.hessian_matrix() @ v) < 1e-10

    def test_hessian_changes_with_position(self):
        """quadratic과 달리 위치마다 Hessian이 달라야 한다."""
        task = RosenbrockTask(RosenbrockSpec(dimension=2), seed=0, dtype=torch.float64)
        v = torch.tensor([1.0, 0.0], dtype=torch.float64)

        with HvpGraph(task.loss, task.params) as graph:
            hv_start = graph.matvec(v)

        with torch.no_grad():
            task.params[0].copy_(task.minimizer)
        with HvpGraph(task.loss, task.params) as graph:
            hv_min = graph.matvec(v)

        assert not torch.allclose(hv_start, hv_min)


# ---------------------------------------------------------------------------
# 그래프 재사용: 이 프로젝트의 비용 모델이 여기 의존한다
# ---------------------------------------------------------------------------


class TestGraphReuse:
    def test_repeated_matvec_is_consistent(self):
        """같은 그래프로 여러 번 호출해도 결과가 동일해야 한다.

        CG는 한 solve 안에서 같은 A를 반복 적용한다. 이게 흔들리면 CG가
        푸는 선형계가 반복마다 달라진다.
        """
        task = QuadraticTask(QuadraticSpec(dimension=32), seed=2)
        graph = HvpGraph(task.loss, task.params)
        v = torch.randn(graph.numel)

        first = graph.matvec(v)
        for _ in range(5):
            assert torch.equal(graph.matvec(v), first)

    def test_loss_closure_is_called_exactly_once(self):
        """curvature batch 고정이 구조로 보장되는지 확인한다.

        클로저가 여러 번 호출되면 minibatch가 바뀔 수 있고, README §15의
        "한 CG solve 안에서 동일한 curvature batch" 원칙이 깨진다.
        """
        task = QuadraticTask(QuadraticSpec(dimension=16), seed=0)
        calls = 0

        def counting_loss():
            nonlocal calls
            calls += 1
            return task.loss()

        graph = HvpGraph(counting_loss, task.params)
        for _ in range(10):
            graph.matvec(torch.randn(graph.numel))

        assert calls == 1

    def test_hvp_count_tracks_matvec_calls(self):
        """비용 회계(프로토콜 D1)의 기본 카운터."""
        task = QuadraticTask(QuadraticSpec(dimension=16), seed=0)
        graph = HvpGraph(task.loss, task.params)

        assert graph.hvp_count == 0
        for expected in range(1, 4):
            graph.matvec(torch.randn(graph.numel))
            assert graph.hvp_count == expected

    def test_grad_and_loss_snapshot_are_available(self):
        """CG의 우변은 ``-grad`` 다. 그래프가 값을 함께 들고 있어야 한다."""
        task = QuadraticTask(QuadraticSpec(dimension=16), seed=0)
        graph = HvpGraph(task.loss, task.params)

        expected_grad = task.hessian_matrix() @ task.params[0].detach()
        assert relative_error(graph.grad, expected_grad) < 1e-5
        assert graph.loss == pytest.approx(task.initial_loss, rel=1e-6)
        assert not graph.grad.requires_grad

    def test_release_prevents_further_matvec(self):
        task = QuadraticTask(QuadraticSpec(dimension=16), seed=0)
        graph = HvpGraph(task.loss, task.params)
        graph.release()

        with pytest.raises(RuntimeError, match="released"):
            graph.matvec(torch.zeros(graph.numel))

    def test_context_manager_releases(self):
        task = QuadraticTask(QuadraticSpec(dimension=16), seed=0)
        with HvpGraph(task.loss, task.params) as graph:
            graph.matvec(torch.zeros(graph.numel))
        with pytest.raises(RuntimeError, match="released"):
            graph.matvec(torch.zeros(graph.numel))


# ---------------------------------------------------------------------------
# 다중 파라미터 텐서와 unused parameter
# ---------------------------------------------------------------------------


class TestMultiTensorModels:
    def test_preserves_dimension_across_parameter_tensors(self):
        seed_everything(0)
        model = torch.nn.Sequential(torch.nn.Linear(8, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))
        flat = ParameterFlattener(model.parameters())
        inputs = torch.randn(16, 8)
        targets = torch.randint(0, 3, (16,))

        def loss():
            return torch.nn.functional.cross_entropy(model(inputs), targets)

        graph = HvpGraph(loss, flat.params, flattener=flat)
        hv = graph.matvec(torch.randn(flat.numel))

        assert hv.shape == (flat.numel,)
        assert torch.isfinite(hv).all()
        assert hv.norm().item() > 0.0

    def test_matches_explicit_hessian_for_small_mlp(self):
        """작은 MLP의 explicit Hessian을 열 단위로 만들어 대조한다."""
        seed_everything(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 2), torch.nn.Tanh(), torch.nn.Linear(2, 2)
        ).double()
        flat = ParameterFlattener(model.parameters())
        inputs = torch.randn(8, 3, dtype=torch.float64)
        targets = torch.randint(0, 2, (8,))

        def loss():
            return torch.nn.functional.cross_entropy(model(inputs), targets)

        # 기준 Hessian: 각 기저 벡터에 대해 일회성 HVP를 독립적으로 계산
        columns = [
            hessian_vector_product(loss, flat.params, e)
            for e in torch.eye(flat.numel, dtype=torch.float64)
        ]
        reference = torch.stack(columns, dim=1)

        graph = HvpGraph(loss, flat.params, flattener=flat)
        v = torch.randn(flat.numel, dtype=torch.float64)

        assert relative_error(graph.matvec(v), reference @ v) < 1e-10
        # 기준 행렬 자체가 대칭이어야 한다 (Hessian의 필요조건)
        assert relative_error(reference, reference.T) < 1e-10

    def test_unused_parameter_yields_zero_block(self):
        """loss에 기여하지 않는 파라미터의 HVP 블록은 0이고 차원은 유지된다."""
        used = torch.nn.Parameter(torch.ones(3))
        unused = torch.nn.Parameter(torch.ones(4))
        flat = ParameterFlattener([used, unused])

        graph = HvpGraph(lambda: (used**3).sum(), flat.params, flattener=flat)
        hv = graph.matvec(torch.ones(7))

        assert hv.shape == (7,)
        # d^2/dx^2 x^3 = 6x, x=1 -> 6
        assert torch.allclose(hv[:3], torch.full((3,), 6.0))
        assert torch.allclose(hv[3:], torch.zeros(4))

    def test_zero_hessian_for_linear_loss(self):
        """loss가 선형이면 Hessian은 0이다. autograd 예외 대신 0을 반환해야 한다."""
        x = torch.nn.Parameter(torch.ones(5))
        coeffs = torch.arange(5.0)

        graph = HvpGraph(lambda: (coeffs * x).sum(), [x])
        hv = graph.matvec(torch.randn(5))

        assert torch.equal(hv, torch.zeros(5))
        assert graph.hvp_count == 1


# ---------------------------------------------------------------------------
# 입력 검증
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_empty_params(self):
        with pytest.raises(ValueError, match="at least one parameter"):
            HvpGraph(lambda: torch.tensor(1.0), [])

    def test_rejects_param_without_grad(self):
        x = torch.randn(3)
        with pytest.raises(ValueError, match="requires_grad=False"):
            HvpGraph(lambda: (x**2).sum(), [x])

    def test_rejects_non_scalar_loss(self):
        x = torch.nn.Parameter(torch.randn(3))
        with pytest.raises(RuntimeError, match="scalar"):
            HvpGraph(lambda: x**2, [x])

    def test_rejects_detached_loss(self):
        x = torch.nn.Parameter(torch.randn(3))
        with pytest.raises(RuntimeError, match="no grad_fn"):
            HvpGraph(lambda: (x**2).sum().detach(), [x])

    def test_rejects_wrong_vector_shape(self):
        x = torch.nn.Parameter(torch.randn(3))
        graph = HvpGraph(lambda: (x**2).sum(), [x])

        with pytest.raises(ValueError, match="must have shape"):
            graph.matvec(torch.randn(4))
        with pytest.raises(ValueError, match="must have shape"):
            graph.matvec(torch.randn(3, 1))


class TestOneShotHelper:
    def test_matches_graph_based_result(self):
        task = QuadraticTask(QuadraticSpec(dimension=16), seed=0)
        v = torch.randn(16)

        one_shot = hessian_vector_product(task.loss, task.params, v)
        with HvpGraph(task.loss, task.params) as graph:
            reused = graph.matvec(v)

        assert torch.allclose(one_shot, reused, atol=1e-6)
