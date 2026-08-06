"""Truncated preconditioned Conjugate Gradient.

``(H + lambda*I) p = -g`` 를 근사적으로 푼다. Newton 방향을 정확히 구하지 않고
**의도적으로 조기에 끊는다.** 이 절단 지점이 RL 컨트롤러의 제어 대상 중 하나다
(``cg_budget``).

반환값이 왜 이렇게 많은가
-------------------------
``CGResult`` 는 해뿐 아니라 비용(``hvp_count``)과 실패 신호를 함께 돌려준다.
이것이 없으면 프로토콜의 주 지표(GE 기준 cost-to-target)를 계산할 수 없고,
RL 상태 특징(``cg_residual_ratio``, ``cg_iters_used_ratio``)도 만들 수 없다.

negative curvature 처리
-----------------------
``p^T A p <= eps * ||p||^2`` 이면 A가 그 방향으로 양정이 아니므로 CG의 전제가
깨진다. 표준 truncated Newton 방식대로 **현재까지 누적한 반복해를 반환하고
중단**한다.

판정을 상대 기준으로 하는 것이 중요하다. ``p^T A p`` 는 ``||p||^2`` 에 비례하므로
절대 임계값을 쓰면 수렴이 진행되어 ``p`` 가 작아질 때 양정 행렬에서도 조건이
성립해 버린다. 곡률이 아니라 스케일을 재게 된다.

단, 첫 반복(j=0)에서 발생하면 누적해가 0이라 방향이 없다. 이 경우
전처리된 최급강하 방향 ``M^{-1} b`` 를 반환한다. 0 방향을 돌려주면 optimizer가
아무 일도 못 하면서 예산만 소모하기 때문이다. 어느 쪽이든
``negative_curvature=True`` 로 보고하므로, damping을 올리거나 step을 거절하는
결정은 상위 계층(optimizer / RL 컨트롤러)이 한다.

수치 정밀도
-----------
residual과 내적 누적은 FP32 이상에서만 수행한다 (README §15). FP16/BF16은
CG 내부 누적에서 정밀도가 무너지므로 명시적으로 거부한다. mixed precision은
baseline이 안정화된 뒤 도입한다.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from rl_newton.types import CGResult, CurvatureOperator, Preconditioner

__all__ = ["conjugate_gradient", "SUPPORTED_DTYPES"]

SUPPORTED_DTYPES = (torch.float32, torch.float64)
"""CG 내부 누적에 허용하는 dtype. README §15 (curvature 누적은 FP32 기본)."""


def _is_finite(x: float) -> bool:
    return math.isfinite(x)


def conjugate_gradient(
    operator: CurvatureOperator,
    rhs: Tensor,
    *,
    max_iters: int,
    tolerance: float = 1.0e-3,
    pap_eps: float = 1.0e-12,
    preconditioner: Preconditioner | None = None,
    x0: Tensor | None = None,
) -> CGResult:
    """``A x = rhs`` 를 truncated PCG로 푼다. ``A = operator``.

    Newton-CG에서는 ``rhs = -g`` 로 호출한다.

    Args:
        operator: ``matvec`` 을 제공하는 curvature 연산자. ``(H + lambda*I)``.
        rhs: 우변 ``b``. 1차원 텐서.
        max_iters: 최대 반복 수. RL action의 ``cg_budget`` 이 여기 들어온다.
            HVP 횟수의 상한이기도 하다.
        tolerance: 상대 residual 종료 기준. ``||r_k|| <= tolerance * ||r_0||``.
        pap_eps: negative curvature 판정 임계값. **상대 기준**으로 쓰인다.
            ``p^T A p <= pap_eps * ||p||^2`` 이면 탐지된 것으로 본다.
            절대 기준을 쓰면 수렴 근처에서 ``p`` 가 작아질 때 양정 행렬에서도
            오탐이 발생한다.
        preconditioner: ``M^{-1}`` 적용자. ``None`` 이면 항등.
        x0: 초기해. ``None`` 이면 0. 0이 아니면 초기 residual 계산에
            HVP 1회가 추가로 소모된다.

    Returns:
        ``CGResult``. ``solution`` 은 항상 유한한 값을 담는다. 수치 붕괴가
        발생하면 붕괴 이전의 마지막 유효 반복해를 반환하고
        ``numerical_failure=True`` 로 표시한다.

    Raises:
        ValueError: ``max_iters < 1``, ``tolerance <= 0``, ``rhs`` 가 1차원이
            아니거나 연산자 차원과 맞지 않는 경우.
        NotImplementedError: dtype이 FP32/FP64가 아닌 경우.

    Example:
        >>> import torch
        >>> from rl_newton.curvature.operators import DampedHessianOperator
        >>> x = torch.nn.Parameter(torch.zeros(2))
        >>> op = DampedHessianOperator.from_closure(lambda: (x ** 2).sum(), [x])
        >>> res = conjugate_gradient(op, torch.tensor([2.0, 4.0]), max_iters=5)
        >>> res.converged, res.solution.tolist()   # H = 2I  ->  x = b / 2
        (True, [1.0, 2.0])
    """
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1, got {max_iters}")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}")
    if rhs.dim() != 1:
        raise ValueError(f"rhs must be 1-D, got shape {tuple(rhs.shape)}")
    if rhs.dtype not in SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"CG accumulation requires float32/float64, got {rhs.dtype}. "
            "mixed precision은 baseline 안정화 이후에 도입한다 (README §15)."
        )

    start_hvp = operator.hvp_count

    def used_hvp() -> int:
        return operator.hvp_count - start_hvp

    # --- 초기화 ---------------------------------------------------------
    if x0 is None:
        x = torch.zeros_like(rhs)
        r = rhs.clone()
    else:
        if x0.shape != rhs.shape:
            raise ValueError(f"x0 shape {tuple(x0.shape)} != rhs shape {tuple(rhs.shape)}")
        x = x0.clone()
        r = rhs - operator.matvec(x)

    initial_residual = float(r.norm())

    # 우변이 0이면 해도 0이다. HVP를 쓰지 않고 즉시 반환한다.
    if initial_residual == 0.0:
        return CGResult(
            solution=x,
            iterations=0,
            hvp_count=used_hvp(),
            budget=max_iters,
            initial_residual=0.0,
            final_residual=0.0,
            converged=True,
            negative_curvature=False,
            numerical_failure=False,
        )

    if not _is_finite(initial_residual):
        return CGResult(
            solution=torch.zeros_like(rhs),
            iterations=0,
            hvp_count=used_hvp(),
            budget=max_iters,
            initial_residual=initial_residual,
            final_residual=initial_residual,
            converged=False,
            negative_curvature=False,
            numerical_failure=True,
        )

    stop_threshold = tolerance * initial_residual

    z = preconditioner.apply(r) if preconditioner is not None else r.clone()
    p = z.clone()
    rz = float(torch.dot(r, z))

    # 전처리기가 양정이 아니면 rz <= 0 이 될 수 있다. CG의 전제가 깨진다.
    if not _is_finite(rz) or rz <= 0.0:
        return CGResult(
            solution=z if _is_finite(float(z.norm())) else torch.zeros_like(rhs),
            iterations=0,
            hvp_count=used_hvp(),
            budget=max_iters,
            initial_residual=initial_residual,
            final_residual=initial_residual,
            converged=False,
            negative_curvature=False,
            numerical_failure=True,
        )

    residual_norm = initial_residual
    iterations = 0
    negative_curvature = False
    numerical_failure = False
    converged = False

    # --- 반복 -----------------------------------------------------------
    for _ in range(max_iters):
        ap = operator.matvec(p)
        pap = float(torch.dot(p, ap))

        if not _is_finite(pap):
            numerical_failure = True
            break

        # 판정은 반드시 상대 기준으로 한다. pap 은 ||p||^2 에 비례하므로
        # 절대 임계값을 쓰면 수렴 근처에서 p 가 작아질 때 양정 행렬에서도
        # 오탐이 발생한다 (curvature가 아니라 스케일을 재는 셈).
        p_norm_sq = float(torch.dot(p, p))
        if pap <= pap_eps * max(p_norm_sq, 1.0e-300):
            # A가 이 방향으로 양정이 아니다. 실패가 아니라 탐지 대상 사건이다.
            negative_curvature = True
            if iterations == 0:
                # 누적해가 0이므로 방향이 없다. 전처리된 최급강하 방향으로 대체한다.
                x = z.clone()
                residual_norm = float((rhs - operator.matvec(x)).norm())
                if not _is_finite(residual_norm):
                    x = torch.zeros_like(rhs)
                    residual_norm = initial_residual
                    numerical_failure = True
            break

        alpha = rz / pap
        if not _is_finite(alpha):
            numerical_failure = True
            break

        x_next = x + alpha * p
        r_next = r - alpha * ap
        next_norm = float(r_next.norm())

        if not _is_finite(next_norm) or not bool(torch.isfinite(x_next).all()):
            # 붕괴 이전의 마지막 유효 반복해를 유지한다.
            numerical_failure = True
            break

        x = x_next
        r = r_next
        residual_norm = next_norm
        iterations += 1

        if residual_norm <= stop_threshold:
            converged = True
            break

        z_next = preconditioner.apply(r) if preconditioner is not None else r
        rz_next = float(torch.dot(r, z_next))

        if not _is_finite(rz_next) or rz_next <= 0.0:
            numerical_failure = True
            break

        beta = rz_next / rz
        p = z_next + beta * p
        rz = rz_next
        z = z_next

    return CGResult(
        solution=x,
        iterations=iterations,
        hvp_count=used_hvp(),
        budget=max_iters,
        initial_residual=initial_residual,
        final_residual=residual_norm,
        converged=converged,
        negative_curvature=negative_curvature,
        numerical_failure=numerical_failure,
    )
