"""``(H + lambda*I)`` 연산자와 preconditioner.

CG solver는 이 연산자만 본다. 행렬을 반환하는 메서드가 없으므로 전체 Hessian을
만들 방법이 타입 수준에서 존재하지 않는다 (README §15).

damping의 역할
--------------
``H`` 의 고유값을 ``lambda`` 만큼 이동시켜 ``(H + lambda*I)`` 의 조건수를
``(L + lambda) / (m + lambda)`` 로 만든다. ``lambda`` 를 키우면 조건수가
1에 가까워져 CG가 빨리 수렴하지만, 방향은 Newton step에서 gradient step 쪽으로
치우친다. RL 컨트롤러가 조절하는 대상이 정확히 이 trade-off다.

indefinite Hessian(``m < 0``)에서는 ``lambda > -m`` 이어야 계가 SPD가 되고
CG가 정의된다. 그보다 작으면 CG가 negative curvature를 만나며, 이는 실패가
아니라 탐지해서 보고해야 하는 정상 사건이다 (``solvers.conjugate_gradient``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from rl_newton.curvature.hvp import HvpGraph, LossClosure
from rl_newton.utils.flatten import ParameterFlattener

__all__ = [
    "DampedHessianOperator",
    "IdentityPreconditioner",
    "DiagonalPreconditioner",
]


class DampedHessianOperator:
    """``types.CurvatureOperator`` 구현체. ``(H + lambda*I) v`` 를 제공한다.

    ``HvpGraph`` 를 감싸므로 생성 시 loss 클로저가 한 번만 호출된다.
    즉 이 연산자의 수명 동안 curvature batch가 고정된다.

    damping은 생성 후에도 바꿀 수 있다. Newton step 하나 안에서 step이 거절되어
    damping을 올리고 다시 풀 때, HVP 그래프를 재사용하면서 damping만 갱신하면
    forward/backward를 다시 하지 않아도 된다. 실질적인 비용 절약이다.

    Args:
        graph: 이미 만들어진 ``HvpGraph``.
        damping: 초기 damping ``lambda``. 0 이상.
        min_damping: ``set_damping`` / ``scale_damping`` 의 하한. 0 을 허용한다.
            무감쇠 Newton-CG(``lambda = 0``)는 Stage 1 정확성 검증에서
            explicit Newton solve와 대조하기 위해 반드시 필요한 설정이다.
            단 하한이 0이면 ``scale_damping`` 이 0에서 벗어날 수 없다.
        max_damping: 상한.

    Example:
        >>> import torch
        >>> x = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        >>> op = DampedHessianOperator.from_closure(lambda: (x ** 2).sum(), [x], damping=1.0)
        >>> op.matvec(torch.tensor([1.0, 0.0]))   # (2I + 1I) e_0
        tensor([3., 0.])
    """

    def __init__(
        self,
        graph: HvpGraph,
        *,
        damping: float = 0.0,
        min_damping: float = 1.0e-6,
        max_damping: float = 1.0e3,
    ) -> None:
        if damping < 0.0:
            raise ValueError(f"damping must be >= 0, got {damping}")
        if min_damping < 0.0 or min_damping > max_damping:
            raise ValueError(
                f"require 0 <= min_damping <= max_damping, got {min_damping}, {max_damping}"
            )

        self._graph = graph
        self._damping = float(damping)
        self._min_damping = float(min_damping)
        self._max_damping = float(max_damping)
        # 그래프의 누적 카운트에서 이 값을 빼서 "이 연산자 기준" 횟수를 만든다.
        self._count_offset = graph.hvp_count

    @classmethod
    def from_closure(
        cls,
        loss_closure: LossClosure,
        params: Sequence[Tensor],
        *,
        damping: float = 0.0,
        min_damping: float = 1.0e-6,
        max_damping: float = 1.0e3,
        flattener: ParameterFlattener | None = None,
    ) -> DampedHessianOperator:
        """loss 클로저로부터 그래프를 만들어 연산자를 생성한다."""
        graph = HvpGraph(loss_closure, params, flattener=flattener)
        return cls(graph, damping=damping, min_damping=min_damping, max_damping=max_damping)

    # --- CurvatureOperator 프로토콜 ---------------------------------------

    @property
    def damping(self) -> float:
        """현재 damping 계수 ``lambda``."""
        return self._damping

    @property
    def hvp_count(self) -> int:
        """누적 HVP 호출 횟수. 비용 회계의 기본 단위다 (프로토콜 D1)."""
        return self._graph.hvp_count - self._count_offset

    def matvec(self, v: Tensor) -> Tensor:
        """``(H + lambda*I) v``."""
        hv = self._graph.matvec(v)
        if self._damping != 0.0:
            hv = hv + self._damping * v
        return hv

    def reset_count(self) -> None:
        """누적 HVP 횟수를 0으로 되돌린다."""
        self._count_offset = self._graph.hvp_count

    # --- 부가 정보 --------------------------------------------------------

    @property
    def graph(self) -> HvpGraph:
        return self._graph

    @property
    def numel(self) -> int:
        """선형계 차원."""
        return self._graph.numel

    @property
    def grad(self) -> Tensor:
        """그래프 생성 시점의 flatten된 gradient. CG의 우변은 ``-grad`` 다."""
        return self._graph.grad

    @property
    def loss(self) -> float:
        """그래프 생성 시점의 loss."""
        return self._graph.loss

    def set_damping(self, damping: float) -> float:
        """damping을 설정하고 ``[min_damping, max_damping]`` 으로 클립한다.

        Returns:
            클립 후 실제 적용된 값.
        """
        if not (damping == damping):  # NaN 검사
            raise ValueError("damping must not be NaN")
        self._damping = float(min(max(damping, self._min_damping), self._max_damping))
        return self._damping

    def scale_damping(self, multiplier: float) -> float:
        """damping에 배수를 적용한다. RL action의 기본 형태다.

        Returns:
            클립 후 실제 적용된 값.
        """
        if multiplier <= 0.0:
            raise ValueError(f"multiplier must be > 0, got {multiplier}")
        return self.set_damping(self._damping * multiplier)

    def release(self) -> None:
        """HVP 그래프를 해제한다. Newton step 종료 시 호출한다."""
        self._graph.release()

    def __enter__(self) -> DampedHessianOperator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def __repr__(self) -> str:
        return (
            f"DampedHessianOperator(numel={self.numel}, damping={self._damping:.3e}, "
            f"hvp_count={self.hvp_count})"
        )


class IdentityPreconditioner:
    """항등 preconditioner. 기본 실험에서 사용한다.

    ``types.Preconditioner`` 프로토콜을 만족한다. CG 코드에 preconditioner
    분기를 두지 않기 위해 존재한다.
    """

    def apply(self, r: Tensor) -> Tensor:
        return r

    def __repr__(self) -> str:
        return "IdentityPreconditioner()"


class DiagonalPreconditioner:
    """대각 preconditioner. ``M^{-1} r = r / d``.

    Stage 5(README Phase 3) 확장 지점이다. Adam second moment나 Hutchinson
    Hessian diagonal을 ``diagonal`` 로 넣어 쓴다.

    Args:
        diagonal: 양수 대각 성분. 1차원, 선형계 차원과 같아야 한다.
        eps: 0 나눗셈 방지 하한.
    """

    def __init__(self, diagonal: Tensor, *, eps: float = 1.0e-8) -> None:
        if diagonal.dim() != 1:
            raise ValueError(f"diagonal must be 1-D, got shape {tuple(diagonal.shape)}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self._inv = 1.0 / torch.clamp(diagonal, min=eps)

    def apply(self, r: Tensor) -> Tensor:
        return r * self._inv

    def __repr__(self) -> str:
        return f"DiagonalPreconditioner(numel={self._inv.numel()})"
