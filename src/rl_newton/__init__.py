"""RL-Controlled Hessian-Free Newton Optimization.

강화학습 에이전트가 Newton-CG solver의 계산 예산(CG 반복 수)과 안정화
파라미터(damping, step size)를 매 optimizer step마다 선택하도록 하여,
고정 하이퍼파라미터보다 나은 cost-to-quality를 달성할 수 있는지 검증한다.

에이전트는 업데이트 방향을 직접 출력하지 않는다. Newton-CG solver의
컨트롤러 역할만 한다.

패키지 구조
-----------
``curvature``   Hessian-vector product 및 curvature 연산자
``solvers``     Conjugate Gradient, line search
``optimizers``  fixed / heuristic / RL-controlled Newton-CG
``rl``          Gymnasium 환경, 상태 특징, 보상, 정책 학습
``tasks``       quadratic / Rosenbrock / 데이터셋 / 모델
``benchmark``   비용 모델, 실행기, 지표, 그림
``utils``       seed, flatten, 로깅, provenance

실험 절차는 ``docs/experiment_protocol.md`` 를 따른다.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
