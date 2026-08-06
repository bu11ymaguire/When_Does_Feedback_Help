"""flatten / unflatten 유틸 테스트.

CG는 파라미터 전체를 하나의 벡터로 다룬다. 차원이 step마다 흔들리면 solver가
조용히 깨지므로, unused parameter 정책과 왕복 변환의 정확성을 여기서 고정한다.
"""

from __future__ import annotations

import pytest
import torch

from rl_newton.utils.flatten import (
    ParameterFlattener,
    flatten_tensors,
    unflatten_like,
)


def test_flatten_unflatten_roundtrip_preserves_values_and_shapes():
    ref = [torch.randn(3, 4), torch.randn(5), torch.randn(2, 2, 2)]
    flat = flatten_tensors(ref)

    assert flat.dim() == 1
    assert flat.numel() == 12 + 5 + 8

    back = unflatten_like(flat, ref)
    assert len(back) == len(ref)
    for original, restored in zip(ref, back, strict=True):
        assert restored.shape == original.shape
        assert torch.equal(restored, original)


def test_flatten_fills_none_with_zeros_using_reference():
    """unused parameter 정책: gradient가 None이면 0으로 채우고 차원을 보존한다."""
    ref = [torch.randn(2, 3), torch.randn(4)]
    grads = [torch.ones(2, 3), None]

    flat = flatten_tensors(grads, reference=ref)

    assert flat.numel() == 6 + 4
    assert torch.equal(flat[:6], torch.ones(6))
    assert torch.equal(flat[6:], torch.zeros(4))


def test_flatten_none_without_reference_raises():
    with pytest.raises(ValueError, match="no reference"):
        flatten_tensors([torch.randn(2), None])


def test_flatten_rejects_mismatched_reference_length():
    with pytest.raises(ValueError, match="reference length"):
        flatten_tensors([torch.randn(2)], reference=[torch.randn(2), torch.randn(3)])


def test_unflatten_rejects_size_mismatch():
    ref = [torch.randn(3), torch.randn(4)]
    with pytest.raises(ValueError, match="size mismatch"):
        unflatten_like(torch.zeros(6), ref)


def test_unflatten_rejects_non_1d_input():
    with pytest.raises(ValueError, match="must be 1-D"):
        unflatten_like(torch.zeros(2, 3), [torch.randn(6)])


def test_unflatten_returns_views_not_copies():
    """복사를 만들지 않는다. Newton step마다 수십 번 호출되는 경로다."""
    ref = [torch.randn(4), torch.randn(2, 3)]
    flat = torch.zeros(10)
    parts = unflatten_like(flat, ref)

    parts[0][0] = 7.0
    assert flat[0].item() == pytest.approx(7.0)


class TestParameterFlattener:
    def test_captures_dimension_of_linear_layer(self):
        model = torch.nn.Linear(3, 2, bias=True)
        flat = ParameterFlattener(model.parameters())

        assert flat.numel == 3 * 2 + 2
        assert len(flat) == 8
        assert len(flat.params) == 2

    def test_rejects_empty_parameter_list(self):
        with pytest.raises(ValueError, match="at least one parameter"):
            ParameterFlattener([])

    def test_flatten_grads_handles_unused_parameters(self):
        """loss에 기여하지 않은 파라미터는 grad가 None이지만 차원은 유지된다."""
        used = torch.nn.Parameter(torch.ones(3))
        unused = torch.nn.Parameter(torch.ones(4))
        flat = ParameterFlattener([used, unused])

        (used.sum() * 2.0).backward()

        g = flat.flatten_grads()
        assert g.numel() == 7
        assert torch.equal(g[:3], torch.full((3,), 2.0))
        assert torch.equal(g[3:], torch.zeros(4))

    def test_add_applies_direction_in_place_with_alpha(self):
        p1 = torch.nn.Parameter(torch.zeros(2))
        p2 = torch.nn.Parameter(torch.zeros(3))
        flat = ParameterFlattener([p1, p2])

        flat.add_(torch.ones(5), alpha=0.5)

        assert torch.allclose(p1, torch.full((2,), 0.5))
        assert torch.allclose(p2, torch.full((3,), 0.5))

    def test_copy_from_restores_parameters_for_step_rejection(self):
        """step 거절 시 파라미터를 되돌리는 경로."""
        p = torch.nn.Parameter(torch.randn(5))
        flat = ParameterFlattener([p])
        snapshot = flat.flatten_params()

        flat.add_(torch.ones(5), alpha=10.0)
        assert not torch.allclose(p, snapshot)

        flat.copy_from_(snapshot)
        assert torch.allclose(p, snapshot)

    def test_flatten_params_returns_detached_copy(self):
        p = torch.nn.Parameter(torch.ones(3))
        flat = ParameterFlattener([p])
        snapshot = flat.flatten_params()

        assert not snapshot.requires_grad
        with torch.no_grad():
            p.mul_(5.0)
        assert torch.allclose(snapshot, torch.ones(3))

    def test_rejects_parameters_on_multiple_devices(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA가 없어 멀티 디바이스 검증을 건너뛴다")
        cpu_p = torch.nn.Parameter(torch.ones(2))
        gpu_p = torch.nn.Parameter(torch.ones(2, device="cuda"))
        with pytest.raises(ValueError, match="multiple devices"):
            ParameterFlattener([cpu_p, gpu_p])

    def test_zeros_matches_linear_system_dimension(self):
        model = torch.nn.Linear(10, 4)
        flat = ParameterFlattener(model.parameters())
        z = flat.zeros()

        assert z.shape == (flat.numel,)
        assert z.device == flat.device
        assert z.dtype == flat.dtype
        assert torch.equal(z, torch.zeros(flat.numel))
