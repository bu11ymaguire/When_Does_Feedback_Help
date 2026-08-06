"""Hessian-vector product via double backward (Pearlmutter trick).

전체 Hessian을 만들지 않고 다음 항등식만 사용한다.

    H v = grad_theta ( g^T v ),   g = grad_theta L

핵심 설계: 그래프를 한 번만 만든다
------------------------------------
Newton-CG step 하나는 CG 반복 k회 동안 같은 Hessian에 대해 k번의 matvec을 한다.
매번 loss를 다시 계산하면 비용이 두 배가 되고, 더 나쁜 것은 minibatch가 바뀌면
**CG가 푸는 선형계 자체가 반복마다 달라진다.**

그래서 ``HvpGraph`` 는 생성 시점에 ``create_graph=True`` 로 gradient를 한 번
계산해 그래프를 붙잡아 두고, 이후 ``matvec`` 은 그 그래프를 재사용한다.
결과적으로 "한 CG solve 안에서는 동일한 curvature batch를 유지한다"는
README §15 원칙이 규율이 아니라 **구조로** 보장된다. 다른 배치를 쓰려면
새 ``HvpGraph`` 를 만들어야 하고, 그건 명시적인 행동이다.

비용도 이 구조를 따른다 (프로토콜 D1).

    Newton-CG step (k iters) = c_grad_graph + k * c_hvp + c_fwd   [GE]

unused parameter 정책
---------------------
``allow_unused=True`` 로 받고 ``None`` 은 0으로 채운다. 차원은 항상 보존한다
(``utils.flatten`` 참조). ReLU 처럼 2차 미분이 거의 모든 곳에서 0인 경우
일부 블록이 ``None`` 으로 오는 것은 정상이며 0이 수학적으로 옳은 값이다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from rl_newton.utils.flatten import ParameterFlattener

__all__ = ["HvpGraph", "hessian_vector_product"]

LossClosure = Callable[[], Tensor]


class HvpGraph:
    """gradient 그래프를 붙잡아 두고 HVP를 반복 수행한다.

    Args:
        loss_closure: 스칼라 loss를 반환하는 클로저. **정확히 한 번** 호출된다.
            따라서 이 그래프의 수명 동안 curvature batch는 고정된다.
        params: 미분 대상 파라미터.
        flattener: 재사용할 ``ParameterFlattener``. 없으면 새로 만든다.

    Raises:
        ValueError: ``params`` 가 비었거나 ``requires_grad=False`` 인 원소가 있는 경우.
        RuntimeError: loss가 스칼라가 아니거나 그래프가 없는 경우.

    Example:
        >>> import torch
        >>> x = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        >>> graph = HvpGraph(lambda: (x ** 2).sum(), [x])   # H = 2I
        >>> graph.matvec(torch.tensor([1.0, 0.0]))
        tensor([2., 0.])
        >>> graph.hvp_count
        1
    """

    def __init__(
        self,
        loss_closure: LossClosure,
        params: Sequence[Tensor],
        *,
        flattener: ParameterFlattener | None = None,
    ) -> None:
        plist = list(params)
        if not plist:
            raise ValueError("HvpGraph requires at least one parameter")
        for i, p in enumerate(plist):
            if not p.requires_grad:
                raise ValueError(f"params[{i}] has requires_grad=False; HVP is undefined")

        self._flat = flattener if flattener is not None else ParameterFlattener(plist)
        if self._flat.numel != sum(p.numel() for p in plist):
            raise ValueError("provided flattener does not match params")

        loss = loss_closure()
        if loss.dim() != 0:
            raise RuntimeError(f"loss_closure must return a scalar, got shape {tuple(loss.shape)}")
        if loss.grad_fn is None:
            raise RuntimeError(
                "loss_closure returned a tensor with no grad_fn; "
                "cannot build a differentiable graph for HVP"
            )

        grads = torch.autograd.grad(loss, plist, create_graph=True, allow_unused=True)
        self._loss_value = float(loss.detach())
        self._grad_graph = self._flat.flatten(grads)
        self._grad = self._grad_graph.detach().clone()
        self._hvp_count = 0
        self._released = False

        # loss가 파라미터에 대해 선형이면 gradient가 상수이므로 그래프가 붙지 않는다.
        # 이때 Hessian은 정확히 0이고, matvec은 0을 반환해야 한다. autograd에
        # 넘기면 "does not require grad" 예외가 나므로 여기서 구분해 둔다.
        self._zero_curvature = not self._grad_graph.requires_grad

    # --- 속성 -------------------------------------------------------------

    @property
    def flattener(self) -> ParameterFlattener:
        return self._flat

    @property
    def numel(self) -> int:
        """선형계 차원."""
        return self._flat.numel

    @property
    def loss(self) -> float:
        """그래프를 만든 시점의 loss 값."""
        return self._loss_value

    @property
    def grad(self) -> Tensor:
        """그래프를 만든 시점의 flatten된 gradient (detached copy)."""
        return self._grad

    @property
    def hvp_count(self) -> int:
        """이 그래프에서 수행한 HVP 횟수."""
        return self._hvp_count

    # --- 연산 -------------------------------------------------------------

    def matvec(self, v: Tensor) -> Tensor:
        """``H v`` 를 계산한다. damping은 포함하지 않는다.

        Args:
            v: 길이 ``numel`` 의 1차원 텐서.

        Returns:
            ``H v``. 1차원, ``v`` 와 같은 shape. 그래프에서 분리된 상태.

        Raises:
            RuntimeError: 그래프가 이미 해제된 경우.
            ValueError: ``v`` 의 shape이 맞지 않는 경우.
        """
        if self._released:
            raise RuntimeError("HvpGraph has been released; construct a new one")
        if v.shape != (self._flat.numel,):
            raise ValueError(f"v must have shape ({self._flat.numel},), got {tuple(v.shape)}")

        if self._zero_curvature:
            self._hvp_count += 1
            return torch.zeros_like(v)

        # retain_graph=True: 같은 그래프로 CG 반복 내내 재사용한다.
        # grad_outputs=v 는 v^T (dg/dtheta) = H v 를 계산한다 (H 대칭).
        # g @ v 노드를 매 반복 새로 만들지 않아 그만큼 싸다.
        hv = torch.autograd.grad(
            self._grad_graph,
            self._flat.params,
            grad_outputs=v,
            retain_graph=True,
            allow_unused=True,
        )
        self._hvp_count += 1
        return self._flat.flatten(hv).detach()

    def release(self) -> None:
        """그래프 참조를 놓아 메모리를 회수한다.

        Newton step이 끝나면 호출한다. 이후 ``matvec`` 은 실패한다.
        """
        self._grad_graph = self._grad  # 그래프 없는 텐서로 교체
        self._released = True

    def __enter__(self) -> HvpGraph:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def __repr__(self) -> str:
        state = "released" if self._released else "live"
        return f"HvpGraph(numel={self._flat.numel}, hvp_count={self._hvp_count}, {state})"


def hessian_vector_product(
    loss_closure: LossClosure,
    params: Sequence[Tensor],
    v: Tensor,
) -> Tensor:
    """``H v`` 를 한 번 계산한다. 편의 및 참조 구현.

    매 호출마다 그래프를 새로 만들므로 CG 루프 안에서 쓰면 안 된다.
    반복 계산에는 ``HvpGraph`` 를 쓴다. 이 함수는 단위 테스트와
    일회성 진단용이다.

    Args:
        loss_closure: 스칼라 loss를 반환하는 클로저.
        params: 미분 대상 파라미터.
        v: 길이가 전체 파라미터 수와 같은 1차원 텐서.

    Returns:
        ``H v``. 1차원 텐서.
    """
    graph = HvpGraph(loss_closure, params)
    try:
        return graph.matvec(v)
    finally:
        graph.release()
