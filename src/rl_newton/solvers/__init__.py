"""선형계 solver — truncated Conjugate Gradient.

구현 완료 (Stage 1)
-------------------
``conjugate_gradient.py``
    ``(H + lambda*I) p = -g`` 를 근사적으로 푼다. ``types.CGResult`` 를 반환한다.

    구현된 기능:
      - residual 기반 early stopping (``||r_k|| <= tol * ||r_0||``)
      - maximum iteration budget (RL action이 지정)
      - negative curvature 탐지 (``p^T A p <= pap_eps * ||p||^2``, **상대 기준**)
      - NaN/Inf 탐지 및 마지막 유효 반복해 보존
      - HVP 횟수, 초기/최종 residual, 예산 반환
      - optional preconditioner 인터페이스

    residual 누적은 FP32 이상만 허용한다 (README §15). FP16/BF16은 거부한다.

구현 예정 (Stage 3)
-------------------
``line_search.py``
    step acceptance와 backtracking. loss가 비유한값이거나 허용 배수 이상
    증가하면 step을 거절하고 damping을 올린다. RL agent가 fallback을
    악용하지 못하도록 실패 penalty가 보상에 반영된다 (프로토콜 D3).

구현 예정 (Stage 5, 선택)
-------------------------
``preconditioners.py``
    Adam second moment / Hutchinson Hessian diagonal / low-rank Lanczos.
    identity 와 diagonal 은 ``curvature.operators`` 에 이미 있다.

    주의: ``tasks.quadratics`` 는 랜덤 직교기저로 ``A`` 를 만들기 때문에 대각이
    거의 상수다. 따라서 quadratic 벤치마크에서 diagonal preconditioner 의
    이득이 없게 나오는 것은 정상이다. 대각이 퍼진 계에서 평가해야 한다
    (``tests/test_cg.py::test_jacobi_helps_when_diagonal_is_spread`` 참조).
"""

from rl_newton.solvers.conjugate_gradient import conjugate_gradient

__all__ = ["conjugate_gradient"]
