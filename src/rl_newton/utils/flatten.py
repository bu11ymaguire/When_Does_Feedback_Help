"""파라미터 리스트와 단일 1차원 벡터 사이의 변환.

Newton-CG는 ``(H + lambda*I) p = -g`` 라는 하나의 선형계를 푼다. 그런데 PyTorch
모델의 파라미터는 여러 개의 서로 다른 shape을 가진 텐서다. CG 내부에서
inner product와 axpy를 다루려면 전체를 하나의 벡터로 보는 뷰가 필요하다.

unused parameter 정책
---------------------
``requires_grad=True`` 인데 gradient가 ``None`` 인 파라미터는 **0으로 채운다.**
이는 "해당 파라미터가 이번 loss에 기여하지 않았다"는 뜻이고, Newton 방향에서도
0이 되는 것이 올바른 처리다. 조용히 건너뛰면 벡터 차원이 step마다 달라져
CG가 깨지므로 차원은 항상 보존한다.

``allow_unused=True`` 로 얻은 HVP 결과에도 동일한 정책을 적용한다.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor

__all__ = ["flatten_tensors", "unflatten_like", "ParameterFlattener"]


def flatten_tensors(
    tensors: Iterable[Tensor | None], reference: Sequence[Tensor] | None = None
) -> Tensor:
    """텐서 리스트를 하나의 1차원 텐서로 이어붙인다.

    Args:
        tensors: 이어붙일 텐서들. ``None`` 원소는 0으로 채운다.
        reference: ``None`` 원소의 shape/dtype/device를 결정하기 위한 참조 리스트.
            ``tensors`` 에 ``None`` 이 있으면 필수다.

    Returns:
        1차원 텐서. 입력이 비어 있으면 길이 0 텐서.

    Raises:
        ValueError: ``None`` 원소가 있는데 ``reference`` 가 주어지지 않은 경우,
            또는 ``reference`` 길이가 맞지 않는 경우.
    """
    items = list(tensors)
    if reference is not None and len(reference) != len(items):
        raise ValueError(f"reference length {len(reference)} != tensors length {len(items)}")

    pieces: list[Tensor] = []
    for i, t in enumerate(items):
        if t is None:
            if reference is None:
                raise ValueError(
                    f"tensors[{i}] is None but no reference was provided; "
                    "cannot determine shape for the unused-parameter zero fill"
                )
            ref = reference[i]
            pieces.append(torch.zeros(ref.numel(), dtype=ref.dtype, device=ref.device))
        else:
            pieces.append(t.reshape(-1))

    if not pieces:
        return torch.zeros(0)
    return torch.cat(pieces)


def unflatten_like(flat: Tensor, reference: Sequence[Tensor]) -> list[Tensor]:
    """1차원 텐서를 ``reference`` 와 같은 shape의 리스트로 되돌린다.

    Args:
        flat: 1차원 텐서.
        reference: 목표 shape을 제공하는 텐서 리스트.

    Returns:
        ``reference`` 와 원소별로 같은 shape을 갖는 텐서 리스트.
        반환 텐서는 ``flat`` 의 view이므로 복사가 일어나지 않는다.

    Raises:
        ValueError: 원소 수 총합이 ``flat.numel()`` 과 다른 경우.
    """
    if flat.dim() != 1:
        raise ValueError(f"flat must be 1-D, got shape {tuple(flat.shape)}")

    total = sum(r.numel() for r in reference)
    if total != flat.numel():
        raise ValueError(
            f"size mismatch: flat has {flat.numel()} elements but reference needs {total}"
        )

    out: list[Tensor] = []
    offset = 0
    for r in reference:
        n = r.numel()
        out.append(flat[offset : offset + n].view_as(r))
        offset += n
    return out


class ParameterFlattener:
    """모델 파라미터의 shape을 한 번 캡처해두고 반복 변환에 재사용한다.

    매 optimizer step마다 shape을 다시 훑는 비용을 없애고, 차원이 도중에
    바뀌지 않는다는 것을 보장한다.

    Example:
        >>> import torch
        >>> model = torch.nn.Linear(3, 2, bias=True)
        >>> flat = ParameterFlattener(model.parameters())
        >>> flat.numel
        8
        >>> v = flat.zeros()
        >>> len(flat.unflatten(v))
        2
    """

    def __init__(self, params: Iterable[Tensor]) -> None:
        self._params: list[Tensor] = list(params)
        if not self._params:
            raise ValueError("ParameterFlattener requires at least one parameter")

        self._shapes: list[torch.Size] = [p.shape for p in self._params]
        self._numels: list[int] = [p.numel() for p in self._params]
        self._numel: int = sum(self._numels)
        self._device: torch.device = self._params[0].device
        self._dtype: torch.dtype = self._params[0].dtype

        devices = {p.device for p in self._params}
        if len(devices) > 1:
            raise ValueError(f"parameters span multiple devices: {devices}")

    # --- 속성 -------------------------------------------------------------

    @property
    def params(self) -> list[Tensor]:
        """캡처된 파라미터 리스트 (원본 참조)."""
        return self._params

    @property
    def numel(self) -> int:
        """전체 파라미터 개수. 선형계의 차원."""
        return self._numel

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # --- 변환 -------------------------------------------------------------

    def flatten(self, tensors: Iterable[Tensor | None]) -> Tensor:
        """``None`` 을 0으로 채우며 1차원으로 이어붙인다."""
        return flatten_tensors(tensors, reference=self._params)

    def unflatten(self, flat: Tensor) -> list[Tensor]:
        """1차원 텐서를 파라미터 shape 리스트로 되돌린다."""
        return unflatten_like(flat, self._params)

    def flatten_params(self) -> Tensor:
        """현재 파라미터 값을 1차원으로 복사해 반환한다 (detach + clone)."""
        return torch.cat([p.detach().reshape(-1) for p in self._params])

    def flatten_grads(self) -> Tensor:
        """현재 ``.grad`` 를 1차원으로 반환한다. ``None`` 은 0으로 채운다."""
        return self.flatten([p.grad for p in self._params])

    def zeros(self) -> Tensor:
        """선형계 차원의 0 벡터."""
        return torch.zeros(self._numel, dtype=self._dtype, device=self._device)

    # --- 파라미터 갱신 ----------------------------------------------------

    @torch.no_grad()
    def add_(self, direction: Tensor, alpha: float = 1.0) -> None:
        """``params += alpha * direction`` 을 in-place로 적용한다."""
        for p, d in zip(self._params, self.unflatten(direction), strict=True):
            p.add_(d, alpha=alpha)

    @torch.no_grad()
    def copy_from_(self, flat: Tensor) -> None:
        """``flat`` 의 값을 파라미터에 그대로 써넣는다. step rejection 시 복원용."""
        for p, v in zip(self._params, self.unflatten(flat), strict=True):
            p.copy_(v)

    def __len__(self) -> int:
        return self._numel

    def __repr__(self) -> str:
        return (
            f"ParameterFlattener(n_tensors={len(self._params)}, numel={self._numel}, "
            f"device={self._device}, dtype={self._dtype})"
        )
