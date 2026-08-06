"""Curvature 연산자 — matrix-free Hessian 접근.

원칙 (README §15): 전체 Hessian을 생성하거나 역행렬을 계산하지 않는다.
``torch.linalg.inv`` 를 optimizer update 경로에 쓰지 않는다.

구현 예정 (Stage 1)
-------------------
``hvp.py``
    Pearlmutter trick 기반 Hessian-vector product.
    ``Hv = grad(g^T v)``, ``torch.autograd.grad(..., create_graph=True)``.
    unused parameter는 ``allow_unused=True`` + 0 채움 (``utils.flatten`` 정책).
    검증: explicit Hessian 대조 상대오차 < 1e-5 (FP32 ill-conditioned는 1e-4).

``operators.py``
    ``types.CurvatureOperator`` 구현체. ``(H + lambda*I)v`` 제공, HVP 횟수 누적.
    한 CG solve 내부에서 curvature batch를 고정한다 (README §15).
    GGN / Fisher-vector product 확장 지점.

구현 예정 (Stage 5, 선택)
-------------------------
``diagonal.py``
    Hutchinson estimator 기반 Hessian diagonal. diagonal preconditioner용.
"""

from rl_newton.curvature.hvp import HvpGraph, hessian_vector_product
from rl_newton.curvature.operators import (
    DampedHessianOperator,
    DiagonalPreconditioner,
    IdentityPreconditioner,
)

__all__ = [
    "HvpGraph",
    "hessian_vector_product",
    "DampedHessianOperator",
    "DiagonalPreconditioner",
    "IdentityPreconditioner",
]
