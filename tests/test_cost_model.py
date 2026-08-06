"""GE 비용 모델 테스트 (프로토콜 D1).

측정된 계수의 절대값은 하드웨어와 OS 스케줄링에 따라 흔들리므로 단정하지 않는다.
대신 구조적 성질을 고정한다: 계수가 양수인지, 비용이 CG 반복 수에 선형인지,
저장/복원이 왕복하는지, 그리고 잘못된 클로저를 조기에 거부하는지.
"""

from __future__ import annotations

import pytest
import torch

from rl_newton.benchmark.cost_model import CostModel, measure_cost_model


@pytest.fixture
def quadratic_problem():
    """작은 SPD quadratic. ``L(x) = 0.5 x^T A x``, ``L* = 0``."""
    torch.manual_seed(0)
    d = 128
    m = torch.randn(d, d)
    a = m @ m.T + d * torch.eye(d)
    x = torch.nn.Parameter(torch.randn(d))

    def loss_fn() -> torch.Tensor:
        return 0.5 * x @ (a @ x)

    return [x], loss_fn


class TestNewtonStepGE:
    @pytest.fixture
    def model(self) -> CostModel:
        return CostModel(
            model_id="synthetic",
            c_grad_graph=1.4,
            c_hvp=2.0,
            c_fwd=0.3,
            t_grad_ms=1.0,
            t_grad_graph_ms=1.4,
            t_hvp_ms=2.0,
            t_fwd_ms=0.3,
            n_params=100,
        )

    def test_cost_is_linear_in_cg_iterations(self, model):
        assert model.newton_step_ge(0) == pytest.approx(1.7)
        assert model.newton_step_ge(10) == pytest.approx(21.7)
        assert model.newton_step_ge(20) == pytest.approx(41.7)

    def test_acceptance_forward_can_be_excluded(self, model):
        assert model.newton_step_ge(5, include_acceptance=False) == pytest.approx(11.4)

    def test_first_order_step_is_one_ge_by_definition(self, model):
        assert model.first_order_step_ge() == pytest.approx(1.0)

    def test_rejects_negative_cg_iterations(self, model):
        with pytest.raises(ValueError, match="cg_iters"):
            model.newton_step_ge(-1)

    def test_documented_budget_arithmetic_holds(self, model):
        """프로토콜 D1의 예산 계산이 코드와 일치하는지 확인한다.

        c_hvp≈2, k=10 이면 Newton step 1회가 약 20~26 GE 이고,
        따라서 200 step 은 AdamW 수천 step 에 해당한다.
        """
        per_step = model.newton_step_ge(10)
        assert 20.0 <= per_step <= 26.0
        assert per_step * 200 > 4000

    def test_launch_bound_flag_follows_measured_gradient_time(self):
        fast = CostModel(
            model_id="fast",
            c_grad_graph=1.0,
            c_hvp=1.0,
            c_fwd=0.3,
            t_grad_ms=0.2,
            t_grad_graph_ms=0.3,
            t_hvp_ms=0.2,
            t_fwd_ms=0.05,
            n_params=10,
        )
        slow = CostModel(
            model_id="slow",
            c_grad_graph=1.4,
            c_hvp=2.0,
            c_fwd=0.3,
            t_grad_ms=25.0,
            t_grad_graph_ms=35.0,
            t_hvp_ms=50.0,
            t_fwd_ms=8.0,
            n_params=10**7,
        )
        assert fast.is_launch_bound is True
        assert slow.is_launch_bound is False


class TestMeasurement:
    def test_measures_positive_coefficients_on_cpu(self, quadratic_problem):
        params, loss_fn = quadratic_problem
        cm = measure_cost_model(params, loss_fn, model_id="test_quadratic", n_warmup=2, n_repeat=5)

        assert cm.model_id == "test_quadratic"
        assert cm.n_params == 128
        assert cm.c_grad == 1.0
        assert cm.c_grad_graph > 0.0
        assert cm.c_hvp > 0.0
        assert cm.c_fwd > 0.0
        assert cm.t_grad_ms > 0.0
        assert cm.device == "cpu"
        assert cm.n_repeat == 5

    def test_forward_only_is_cheaper_than_gradient(self, quadratic_problem):
        """forward < forward+backward. 이게 깨지면 측정이 노이즈에 묻힌 것이다."""
        params, loss_fn = quadratic_problem
        cm = measure_cost_model(params, loss_fn, model_id="ordering", n_warmup=5, n_repeat=25)
        assert cm.c_fwd < 1.0

    def test_rejects_empty_params(self, quadratic_problem):
        _, loss_fn = quadratic_problem
        with pytest.raises(ValueError, match="must not be empty"):
            measure_cost_model([], loss_fn, model_id="x")

    def test_rejects_params_without_grad(self):
        x = torch.randn(4)  # requires_grad=False

        with pytest.raises(ValueError, match="requires_grad"):
            measure_cost_model([x], lambda: (x**2).sum(), model_id="x")

    def test_rejects_non_scalar_loss(self):
        x = torch.nn.Parameter(torch.randn(4))
        with pytest.raises(RuntimeError, match="scalar"):
            measure_cost_model([x], lambda: x**2, model_id="x")

    def test_rejects_closure_that_does_not_build_graph(self):
        """detach된 loss를 주면 조용히 이상한 계수가 나오는 대신 즉시 실패한다."""
        x = torch.nn.Parameter(torch.randn(4))
        with pytest.raises(RuntimeError, match="no grad_fn"):
            measure_cost_model([x], lambda: (x**2).sum().detach(), model_id="x")


class TestSerialization:
    def test_save_load_round_trip(self, tmp_path, quadratic_problem):
        params, loss_fn = quadratic_problem
        cm = measure_cost_model(
            params,
            loss_fn,
            model_id="roundtrip",
            grad_batch_size=512,
            curvature_batch_size=256,
            n_warmup=1,
            n_repeat=3,
        )

        path = cm.save(tmp_path / "sub" / "cost_model.yaml")
        assert path.exists()

        restored = CostModel.load(path)
        assert restored.model_id == cm.model_id
        assert restored.c_hvp == pytest.approx(cm.c_hvp)
        assert restored.c_fwd == pytest.approx(cm.c_fwd)
        assert restored.n_params == cm.n_params
        assert restored.grad_batch_size == 512
        assert restored.curvature_batch_size == 256

    def test_load_ignores_derived_fields(self, tmp_path):
        cm = CostModel(
            model_id="derived",
            c_grad_graph=1.0,
            c_hvp=2.0,
            c_fwd=0.3,
            t_grad_ms=1.0,
            t_grad_graph_ms=1.0,
            t_hvp_ms=2.0,
            t_fwd_ms=0.3,
            n_params=5,
        )
        path = cm.save(tmp_path / "cm.yaml")
        assert "is_launch_bound" in path.read_text(encoding="utf-8")

        restored = CostModel.load(path)  # 파생 필드가 있어도 실패하지 않아야 한다
        assert restored.model_id == "derived"
