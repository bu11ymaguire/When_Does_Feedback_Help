"""Stage 0 게이트: 환경이 이 프로젝트를 실행할 수 있는지 검증한다.

이 프로젝트 전체가 double-backward(``create_graph=True``)에 의존한다.
그것이 대상 하드웨어에서 정확하게 동작하지 않으면 나머지는 의미가 없으므로,
가장 먼저 확인한다.
"""

from __future__ import annotations

import pytest
import torch

import rl_newton
from rl_newton.utils.flatten import ParameterFlattener


def test_package_imports_with_version():
    assert rl_newton.__version__


def test_core_dependencies_are_importable():
    import gymnasium
    import numpy
    import scipy
    import stable_baselines3

    assert gymnasium.__version__
    assert stable_baselines3.__version__
    assert numpy.__version__
    assert scipy.__version__


def test_gymnasium_multidiscrete_action_space_matches_protocol():
    """README §5.2 action space: damping x cg_budget x step_size = 3 x 4 x 3."""
    from gymnasium import spaces

    space = spaces.MultiDiscrete([3, 4, 3])
    assert space.nvec.tolist() == [3, 4, 3]
    assert int(space.nvec.prod()) == 36  # 프로토콜 D5의 탐색 예산 기준값


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_double_backward_hvp_matches_explicit_hessian(device: str):
    """``Hv = grad(g^T v)`` 가 explicit Hessian 곱과 일치하는지 확인한다.

    SPD quadratic ``L = 0.5 x^T A x`` 의 Hessian은 정확히 ``A`` 이므로
    ``Hv`` 를 ``A v`` 와 직접 대조할 수 있다.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA 사용 불가")

    torch.manual_seed(0)
    d = 64
    m = torch.randn(d, d, device=device)
    a = m @ m.T + d * torch.eye(d, device=device)  # SPD, 조건수 적당
    x = torch.nn.Parameter(torch.randn(d, device=device))

    loss = 0.5 * x @ (a @ x)
    (g,) = torch.autograd.grad(loss, [x], create_graph=True)

    v = torch.randn(d, device=device)
    (hv,) = torch.autograd.grad(g @ v, [x])

    expected = a @ v
    rel_err = ((hv - expected).norm() / expected.norm()).item()
    assert rel_err < 1e-5, f"HVP 상대오차 {rel_err:.2e} 가 허용치 1e-5 를 초과"


def test_hvp_preserves_parameter_shapes_across_multiple_tensors():
    """여러 파라미터 텐서에 걸친 HVP가 차원을 보존하는지 확인한다."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 5),
        torch.nn.Tanh(),  # 2차 미분이 0이 아닌 활성함수
        torch.nn.Linear(5, 3),
    )
    flat = ParameterFlattener(model.parameters())

    inputs = torch.randn(16, 8)
    targets = torch.randint(0, 3, (16,))
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)

    grads = torch.autograd.grad(loss, flat.params, create_graph=True)
    g = flat.flatten(grads)
    assert g.numel() == flat.numel

    v = torch.randn(flat.numel)
    hv = flat.flatten(torch.autograd.grad(g @ v, flat.params, allow_unused=True))

    assert hv.shape == (flat.numel,)
    assert torch.isfinite(hv).all()
    # Hessian은 0이 아니어야 한다 (Tanh 때문에 곡률이 존재)
    assert hv.norm().item() > 0.0


@pytest.mark.cuda
def test_cuda_is_available_with_expected_capability():
    """프로토콜 §1 기준 환경 확인. CUDA가 없으면 Phase 1 이후를 실행할 수 없다."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA 사용 불가")

    major, _ = torch.cuda.get_device_capability(0)
    assert major >= 7, "CUDA 13.0 wheel은 Turing(7.5) 이상을 지원한다"

    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    assert total_mb > 4000, f"VRAM {total_mb:.0f}MB 로는 Phase 1 실험이 어렵다"


@pytest.mark.cuda
def test_peak_memory_tracking_works():
    """peak VRAM 로깅(README §15)이 실제로 동작하는지 확인한다."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA 사용 불가")

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.max_memory_allocated()
    buf = torch.empty(1024 * 1024, device="cuda")  # 4 MB
    after = torch.cuda.max_memory_allocated()

    assert after > before
    del buf
