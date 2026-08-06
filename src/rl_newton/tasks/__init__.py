"""최적화 task — synthetic 문제와 신경망 학습 문제.

모든 task는 ``seed`` 로부터 **결정론적으로** 인스턴스를 만든다. paired design
(프로토콜 D7)에서 ``seed=s`` 일 때 모든 optimizer가 동일한 문제와 동일한
minibatch 순서를 보아야 하기 때문이다. ``utils.seed.spawn_seed`` 를 쓴다.

구현 완료 (Stage 1)
-------------------
``quadratics.py``
    ``L(x) = 0.5 x^T A x``. ``L* = 0`` 으로 구성해 ``log L`` 보상이
    정의되도록 한다 (프로토콜 D3). 고유값 분해로 직접 구성하므로 조건수를
    **정확히** 지정할 수 있다.
      - SPD, 지정 조건수 kappa
      - ill-conditioned (kappa ~ 1e4 이상)
      - indefinite: **진단 전용**. 아래로 유계가 아니므로 cost-to-target
        집계와 log 보상에서 제외한다. ``is_bounded_below`` 로 확인한다.

``rosenbrock.py``
    비볼록, 좁은 곡선 골짜기. 2D 및 N차원.
    negative curvature 는 ``y > x^2 + 1/(2 scale)`` 즉 골짜기 **위쪽**에서만
    발생한다. 표준 시작점 (-1.2, 1.0) 은 Hessian이 양정이다.
    ``negative_curvature_point()`` 로 음의 곡률 지점을 얻는다.

구현 예정 (Stage 3)
-------------------
``datasets.py``
    MNIST / Fashion-MNIST / CIFAR-10 subset 로더.
    gradient batch와 curvature batch를 분리 제어한다 (프로토콜 D2:
    ``B_c/B_g in {1/4, 1/2, 1}``, 기본은 동일 배치).
    ``data/`` 에 캐시하며 git에는 추적하지 않는다.

``models.py``
    MLP 784-128-10 (약 101,770 파라미터), 깊은 MLP, small CNN.
    width/depth/init scale 랜덤화로 meta-train task 분포를 만든다.
    Stage 5에서 wall-clock 검증용 수백만 파라미터급 CNN을 추가한다
    (프로토콜 D1: 작은 모델은 런치 오버헤드 지배로 wall-clock 해석이 불가).
"""

from rl_newton.tasks.quadratics import QuadraticKind, QuadraticSpec, QuadraticTask
from rl_newton.tasks.rosenbrock import RosenbrockSpec, RosenbrockTask

__all__ = [
    "QuadraticKind",
    "QuadraticSpec",
    "QuadraticTask",
    "RosenbrockSpec",
    "RosenbrockTask",
]
