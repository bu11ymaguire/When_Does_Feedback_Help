"""Quadratic 최적화 task.

    L(x) = 0.5 * x^T A x

``A`` 를 고유값 분해로 직접 구성하므로 조건수를 **정확히 지정**할 수 있다.
Hessian이 ``A`` 이고 상수이므로 HVP와 CG의 정확성을 explicit 행렬 곱과 직접
대조할 수 있다. Stage 1 검증의 기준 문제다.

``L* = 0`` 인 이유
------------------
프로토콜 D3의 보상은 ``log L_t - log L_{t+1}`` 이다. 최적값이 0이어야
``L -> 0`` 일 때 log가 정의되고 상대 진행도가 의미를 갖는다. 그래서 선형항과
상수항 없이 순수 이차형식으로 둔다.

indefinite quadratic은 진단 전용이다
------------------------------------
음의 고유값이 있으면 ``inf(L) = -inf`` 이므로 target 도달 개념이 성립하지 않고
log 보상도 정의되지 않는다. 따라서 indefinite 인스턴스는

  - negative curvature 탐지 검증 (Stage 1)
  - step rejection / fallback 동작 검증 (Stage 3)

에만 쓰고, cost-to-target 집계와 RL meta-training 분포에서는 제외한다.
``QuadraticTask.is_bounded_below`` 로 구분한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from rl_newton.utils.seed import torch_generator

__all__ = ["QuadraticKind", "QuadraticSpec", "QuadraticTask"]

QuadraticKind = Literal["spd", "ill_conditioned", "indefinite"]


@dataclass(frozen=True, slots=True)
class QuadraticSpec:
    """quadratic 인스턴스를 완전히 결정하는 명세.

    ``(spec, seed)`` 쌍이 인스턴스를 유일하게 결정한다. paired design에서
    모든 optimizer가 같은 문제를 보게 하는 근거다 (프로토콜 D7).

    Attributes:
        kind: 문제 종류.
        dimension: 차원 ``d``.
        condition_number: ``lambda_max / |lambda|_min``. 1 이상.
        negative_fraction: ``kind="indefinite"`` 일 때 음의 고유값 비율.
            (0, 1) 범위.
        initial_scale: 초기점 norm 스케일.
    """

    kind: QuadraticKind = "spd"
    dimension: int = 100
    condition_number: float = 1.0e2
    negative_fraction: float = 0.2
    initial_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {self.dimension}")
        if self.condition_number < 1.0:
            raise ValueError(f"condition_number must be >= 1, got {self.condition_number}")
        if self.kind == "indefinite" and not 0.0 < self.negative_fraction < 1.0:
            raise ValueError(f"negative_fraction must be in (0, 1), got {self.negative_fraction}")
        if self.initial_scale <= 0.0:
            raise ValueError(f"initial_scale must be > 0, got {self.initial_scale}")

    def instance_id(self, seed: int) -> str:
        """로그와 집계에 쓰는 인스턴스 식별자."""
        base = f"quad_{self.kind}_d{self.dimension}_k{self.condition_number:.0e}"
        if self.kind == "indefinite":
            base += f"_neg{self.negative_fraction:.2f}"
        return f"{base}_seed{seed}"


class QuadraticTask:
    """``L(x) = 0.5 x^T A x`` 최적화 task.

    Args:
        spec: 문제 명세.
        seed: 실험 조건 식별자. 같은 ``(spec, seed)`` 면 항상 같은 ``A`` 와
            같은 초기점이 나온다.
        device: 텐서 디바이스.
        dtype: 텐서 dtype. FP32 또는 FP64.

    Example:
        >>> spec = QuadraticSpec(kind="spd", dimension=10, condition_number=100.0)
        >>> task = QuadraticTask(spec, seed=0)
        >>> a = QuadraticTask(spec, seed=0)
        >>> bool(torch.allclose(task.matrix, a.matrix))   # 결정론적 재현
        True
        >>> task.optimal_loss
        0.0
    """

    def __init__(
        self,
        spec: QuadraticSpec,
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

        d = spec.dimension
        # 난수 스트림을 용도별로 분리한다. 전역 시드를 오염시키지 않는다.
        gen_matrix = torch_generator(seed, "quadratic", "matrix", spec.instance_id(seed))
        gen_init = torch_generator(seed, "quadratic", "init", spec.instance_id(seed))

        eigenvalues = self._build_eigenvalues(spec, gen_matrix)
        basis = self._random_orthogonal(d, gen_matrix)

        # A = Q diag(eig) Q^T
        self._eigenvalues = eigenvalues.to(device=self._device, dtype=dtype)
        self._matrix = (basis * eigenvalues) @ basis.T
        self._matrix = self._matrix.to(device=self._device, dtype=dtype)
        # 부동소수 비대칭을 제거한다. CG는 A가 대칭이라고 가정한다.
        self._matrix = 0.5 * (self._matrix + self._matrix.T)

        x0 = torch.randn(d, generator=gen_init, dtype=torch.float64)
        x0 = spec.initial_scale * x0 / x0.norm() * (d**0.5)
        self._initial_point = x0.to(device=self._device, dtype=dtype)
        self._x = torch.nn.Parameter(self._initial_point.clone())
        self._initial_loss = float(self.loss().detach())

    # --- 구성 -------------------------------------------------------------

    @staticmethod
    def _build_eigenvalues(spec: QuadraticSpec, gen: torch.Generator) -> Tensor:
        """조건수를 정확히 만족하는 고유값을 만든다.

        크기는 ``[1, kappa]`` 에 log 등간격으로 배치한다. 양 끝값을 반드시
        포함시켜 조건수가 지정값과 정확히 일치하도록 한다.
        """
        d = spec.dimension
        kappa = spec.condition_number

        if d == 1:
            magnitudes = torch.tensor([kappa], dtype=torch.float64)
        else:
            magnitudes = torch.logspace(
                0.0,
                float(torch.log10(torch.tensor(kappa, dtype=torch.float64))),
                d,
                dtype=torch.float64,
            )

        if spec.kind != "indefinite":
            return magnitudes

        # 음의 고유값을 섞는다. 크기 분포는 유지하므로 조건수도 유지된다.
        n_negative = max(1, min(d - 1, int(round(d * spec.negative_fraction))))
        signs = torch.ones(d, dtype=torch.float64)
        # 어느 인덱스를 음수로 할지도 결정론적으로 고른다.
        picked = torch.randperm(d, generator=gen)[:n_negative]
        signs[picked] = -1.0
        return magnitudes * signs

    @staticmethod
    def _random_orthogonal(d: int, gen: torch.Generator) -> Tensor:
        """QR 분해로 직교행렬을 만든다. FP64로 계산해 직교성을 확보한다."""
        m = torch.randn(d, d, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(m)
        # QR의 부호 모호성을 제거해 결정론성을 확보한다.
        return q * torch.sign(torch.diagonal(r)).unsqueeze(0)

    # --- task 인터페이스 --------------------------------------------------

    @property
    def spec(self) -> QuadraticSpec:
        return self._spec

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def instance_id(self) -> str:
        return self._spec.instance_id(self._seed)

    @property
    def params(self) -> list[Tensor]:
        """최적화 대상. quadratic은 단일 텐서다."""
        return [self._x]

    @property
    def matrix(self) -> Tensor:
        """``A``. 검증용이며 optimizer 경로에서는 쓰지 않는다."""
        return self._matrix

    @property
    def eigenvalues(self) -> Tensor:
        """``A`` 의 고유값. 부호를 포함한다."""
        return self._eigenvalues

    @property
    def condition_number(self) -> float:
        """실제 조건수 ``|lambda|_max / |lambda|_min``."""
        mags = self._eigenvalues.abs()
        return float(mags.max() / mags.min())

    @property
    def min_eigenvalue(self) -> float:
        """최소 고유값. 음수면 indefinite이고, CG를 안정화하려면
        ``damping > -min_eigenvalue`` 가 필요하다."""
        return float(self._eigenvalues.min())

    @property
    def is_bounded_below(self) -> bool:
        """아래로 유계인지. ``False`` 면 target 도달 개념이 성립하지 않는다."""
        return self.min_eigenvalue > 0.0

    @property
    def optimal_loss(self) -> float:
        """SPD면 0. indefinite면 ``-inf``."""
        return 0.0 if self.is_bounded_below else float("-inf")

    @property
    def initial_loss(self) -> float:
        """초기점에서의 loss. 상대 target 판정의 기준값이다."""
        return self._initial_loss

    def loss(self) -> Tensor:
        """``0.5 x^T A x``. 미분 가능한 스칼라를 반환한다."""
        return 0.5 * (self._x @ (self._matrix @ self._x))

    def curvature_loss(self) -> Tensor:
        """quadratic에는 minibatch가 없으므로 ``loss`` 와 동일하다."""
        return self.loss()

    def reset(self) -> None:
        """파라미터를 초기점으로 되돌린다."""
        with torch.no_grad():
            self._x.copy_(self._initial_point)

    # --- 검증 보조 --------------------------------------------------------

    def hessian_matrix(self) -> Tensor:
        """explicit Hessian. quadratic에서는 ``A`` 와 같다. 테스트 전용이다."""
        return self._matrix

    def exact_newton_step(self, damping: float = 0.0) -> Tensor:
        """``-(A + lambda I)^{-1} A x``. 테스트 전용 참조 해다.

        optimizer 경로에서는 절대 쓰지 않는다 (README §15: ``torch.linalg.inv``
        금지). CG 해와 대조하기 위한 기준값이다.
        """
        d = self._spec.dimension
        eye = torch.eye(d, device=self._device, dtype=self._dtype)
        grad = self._matrix @ self._x.detach()
        return torch.linalg.solve(self._matrix + damping * eye, -grad)

    def __repr__(self) -> str:
        return (
            f"QuadraticTask({self.instance_id}, kappa={self.condition_number:.3e}, "
            f"lambda_min={self.min_eigenvalue:.3e}, L0={self._initial_loss:.6e})"
        )
