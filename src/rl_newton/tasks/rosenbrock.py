"""Rosenbrock 함수 task.

    L(x) = sum_{i=1}^{d-1} [ 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]

quadratic과 달리 **Hessian이 위치에 따라 변한다.** 좁고 휘어진 골짜기를 따라
곡률이 급격히 바뀌므로, "구간에 따라 필요한 damping과 CG 정확도가 다르다"는
README §2 중심 가설을 시험할 수 있는 가장 저렴한 비볼록 문제다.

최소값은 ``x = (1, ..., 1)`` 에서 ``L = 0`` 이다. 따라서 프로토콜 D3의
log-loss 보상이 그대로 정의된다.

곡률 특성 (2D, scale = s)
-------------------------
Hessian은 다음과 같다.

    H = [[12 s x^2 - 4 s y + 2,  -4 s x],
         [-4 s x,                 2 s  ]]

    det H = 8 s^2 (x^2 - y) + 4 s

따라서 **negative curvature는 ``y > x^2 + 1/(2s)`` 인 영역에서만 발생한다.**
즉 포물선 골짜기의 *위쪽*이다. s = 100 이면 경계가 ``y = x^2 + 0.005`` 다.

표준 시작점 ``(-1.2, 1.0)`` 은 ``y = 1.0 < x^2 = 1.44`` 이므로 골짜기 아래쪽이고
Hessian이 양정이다 (고유값 약 23.6, 1506). 시작점부터 음의 곡률을 밟는 것은
아니지만, 최적화가 골짜기를 따라 움직이며 경계를 넘나들기 때문에 실행 중에는
negative curvature 경로를 거치게 된다.

d >= 4 에서는 국소 최소값이 추가로 존재한다고 알려져 있어, 단순히 "0에
도달했는가"만 보면 안 되고 도달률을 함께 봐야 한다 (프로토콜 D6 절단 규칙).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_newton.utils.seed import torch_generator

__all__ = ["RosenbrockSpec", "RosenbrockTask"]


@dataclass(frozen=True, slots=True)
class RosenbrockSpec:
    """Rosenbrock 인스턴스 명세.

    Attributes:
        dimension: 차원 ``d``. 2 이상.
        scale: 골짜기 급경사 계수. 표준값 100.
        randomize_start: ``True`` 면 표준 시작점에 결정론적 잡음을 더한다.
            paired design에서 seed별로 다른 시작점을 만들 때 쓴다.
        start_noise: 잡음 스케일.
    """

    dimension: int = 2
    scale: float = 100.0
    randomize_start: bool = False
    start_noise: float = 0.1

    def __post_init__(self) -> None:
        if self.dimension < 2:
            raise ValueError(f"Rosenbrock requires dimension >= 2, got {self.dimension}")
        if self.scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {self.scale}")
        if self.start_noise < 0.0:
            raise ValueError(f"start_noise must be >= 0, got {self.start_noise}")

    def instance_id(self, seed: int) -> str:
        tag = "rand" if self.randomize_start else "std"
        return f"rosen_d{self.dimension}_s{self.scale:.0f}_{tag}_seed{seed}"


class RosenbrockTask:
    """Rosenbrock 최소화 task.

    Args:
        spec: 문제 명세.
        seed: 실험 조건 식별자.
        device: 텐서 디바이스.
        dtype: FP32 또는 FP64.

    Example:
        >>> task = RosenbrockTask(RosenbrockSpec(dimension=2), seed=0)
        >>> task.optimal_loss
        0.0
        >>> round(float(task.loss()), 2)   # 표준 시작점 (-1.2, 1.0)
        24.2
    """

    def __init__(
        self,
        spec: RosenbrockSpec,
        seed: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"dtype must be float32 or float64, got {dtype}")

        self._spec = spec
        self._seed = seed
        self._device = torch.device(device)
        self._dtype = dtype

        start = self._standard_start(spec.dimension)
        if spec.randomize_start:
            gen = torch_generator(seed, "rosenbrock", "start", spec.instance_id(seed))
            start = start + spec.start_noise * torch.randn(
                spec.dimension, generator=gen, dtype=torch.float64
            )

        self._initial_point = start.to(device=self._device, dtype=dtype)
        self._x = torch.nn.Parameter(self._initial_point.clone())
        self._initial_loss = float(self.loss().detach())

    @staticmethod
    def _standard_start(d: int) -> Tensor:
        """표준 시작점. 첫 좌표를 -1.2로 두어 골짜기 밖에서 출발한다."""
        start = torch.ones(d, dtype=torch.float64)
        start[0] = -1.2
        return start

    # --- task 인터페이스 --------------------------------------------------

    @property
    def spec(self) -> RosenbrockSpec:
        return self._spec

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def instance_id(self) -> str:
        return self._spec.instance_id(self._seed)

    @property
    def params(self) -> list[Tensor]:
        return [self._x]

    @property
    def is_bounded_below(self) -> bool:
        return True

    @property
    def optimal_loss(self) -> float:
        return 0.0

    @property
    def initial_loss(self) -> float:
        return self._initial_loss

    @property
    def minimizer(self) -> Tensor:
        """전역 최소점 ``(1, ..., 1)``."""
        return torch.ones(self._spec.dimension, device=self._device, dtype=self._dtype)

    def negative_curvature_point(self, x0: float = 0.0) -> Tensor:
        """Hessian이 indefinite인 지점을 반환한다.

        모듈 docstring의 조건 ``y > x^2 + 1/(2 scale)`` 을 만족하는 점을 만든다.
        여유를 두어 부동소수 오차로 경계에 걸치지 않게 한다.

        negative curvature 탐지와 damping 동작을 결정론적으로 시험할 때 쓴다.
        표준 시작점은 골짜기 아래쪽이라 Hessian이 양정이므로 이 용도로는 쓸 수 없다.

        Args:
            x0: 첫 좌표. 나머지 좌표는 ``x0^2 + margin`` 으로 채운다.

        Returns:
            길이 ``dimension`` 의 점.
        """
        margin = 1.0 / self._spec.scale  # 경계값 1/(2s) 의 2배
        point = torch.full(
            (self._spec.dimension,),
            x0 * x0 + margin,
            device=self._device,
            dtype=self._dtype,
        )
        point[0] = x0
        return point

    def move_to(self, point: Tensor) -> None:
        """파라미터를 지정 위치로 옮긴다. 곡률 진단용이다."""
        if point.shape != (self._spec.dimension,):
            raise ValueError(
                f"point must have shape ({self._spec.dimension},), got {tuple(point.shape)}"
            )
        with torch.no_grad():
            self._x.copy_(point)

    def loss(self) -> Tensor:
        """Rosenbrock 값. 미분 가능한 스칼라."""
        x = self._x
        head = x[:-1]
        tail = x[1:]
        return (self._spec.scale * (tail - head**2) ** 2 + (1.0 - head) ** 2).sum()

    def curvature_loss(self) -> Tensor:
        """minibatch가 없으므로 ``loss`` 와 동일하다."""
        return self.loss()

    def reset(self) -> None:
        with torch.no_grad():
            self._x.copy_(self._initial_point)

    # --- 검증 보조 --------------------------------------------------------

    def hessian_matrix(self) -> Tensor:
        """현재 위치의 explicit Hessian. 테스트 전용이다.

        ``torch.autograd.functional.hessian`` 으로 계산하므로 HVP 구현과
        독립적인 대조군이 된다.
        """
        scale = self._spec.scale

        def fn(x: Tensor) -> Tensor:
            head = x[:-1]
            tail = x[1:]
            return (scale * (tail - head**2) ** 2 + (1.0 - head) ** 2).sum()

        return torch.autograd.functional.hessian(fn, self._x.detach())

    def __repr__(self) -> str:
        return (
            f"RosenbrockTask({self.instance_id}, d={self._spec.dimension}, "
            f"L0={self._initial_loss:.6e})"
        )
