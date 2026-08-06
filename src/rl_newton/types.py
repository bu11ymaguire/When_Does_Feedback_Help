"""프로젝트 전역 인터페이스 정의.

여기 있는 dataclass와 Protocol은 모듈 간 계약이다. Stage 1 이후의 구현은
모두 이 계약을 따른다. 계약을 바꿀 때는 ``docs/experiment_protocol.md`` §9
변경 이력에 기록한다.

설계 의도
---------
- ``CGResult`` / ``StepRecord`` 는 **비용과 실패를 반드시 보고**하도록 필드를 강제한다.
  HVP 횟수나 실패 플래그를 빼먹으면 프로토콜의 주 지표(GE)를 계산할 수 없다.
- ``CurvatureOperator`` 는 전체 Hessian을 만들지 않는다는 원칙을 타입으로 표현한다.
  matvec만 제공하며, 행렬을 반환하는 메서드가 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from torch import Tensor

__all__ = [
    "ControllerAction",
    "CurvatureOperator",
    "Preconditioner",
    "CGResult",
    "StepRecord",
    "FailureTag",
    "SolverConfig",
]


# ---------------------------------------------------------------------------
# 실패 분류
# ---------------------------------------------------------------------------

FailureTag = str
"""실패 원인 태그. 절단(censoring) 집계에서 사용한다.

허용값:
    ``"nan"``               loss 또는 gradient에 NaN/Inf 발생
    ``"divergence"``        loss가 허용 배수 이상 증가
    ``"budget_exhausted"``  예산 내 target 미도달
    ``"cg_breakdown"``      CG 내부 수치 붕괴 (pAp <= eps 등)
    ``"oom"``               VRAM 부족
"""

VALID_FAILURE_TAGS: frozenset[str] = frozenset(
    {"nan", "divergence", "budget_exhausted", "cg_breakdown", "oom"}
)


# ---------------------------------------------------------------------------
# 컨트롤러 행동
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ControllerAction:
    """Newton-CG solver에 전달되는 제어값 하나.

    RL 에이전트, heuristic 규칙, 고정 baseline, open-loop 스케줄이 모두
    이 타입을 출력한다. 덕분에 비교군 사이에 action space가 동일하다는 점을
    구조적으로 보장한다 (프로토콜 D4).

    damping 의 두 가지 모드
    -----------------------
    기본은 **상대 모드**다. ``damping_multiplier`` 가 현재 damping 에 곱해지며,
    damping 은 step 사이에 누적되는 지속 상태다. 실제 컨트롤러(heuristic, RL)가
    쓰는 방식이다.

    ``damping_absolute`` 를 지정하면 **절대 모드**가 되어 현재 damping 과
    무관하게 그 값으로 즉시 이동한다. 이는 분석 전용이다. 도달성 제약이 없는
    오라클(프로토콜 게이트 A)을 만들 때만 쓴다. 절대 모드 오라클과 상대 모드
    오라클의 차이가 "배수 전이와 도달성 때문에 잃는 양"이다.

    Attributes:
        damping_multiplier: 현재 damping에 곱할 계수. 절대 모드에서는 무시된다.
        cg_budget: 이번 step에서 허용할 CG 최대 반복 수.
        step_size: Newton 방향에 적용할 step size.
        damping_absolute: 지정되면 이 값으로 damping 을 직접 설정한다.
            분석용 오라클 전용이며 학습 정책은 쓰지 않는다.
    """

    damping_multiplier: float
    cg_budget: int
    step_size: float
    damping_absolute: float | None = None

    def __post_init__(self) -> None:
        if self.damping_multiplier <= 0.0:
            raise ValueError(f"damping_multiplier must be > 0, got {self.damping_multiplier}")
        if self.cg_budget < 1:
            raise ValueError(f"cg_budget must be >= 1, got {self.cg_budget}")
        if self.step_size <= 0.0:
            raise ValueError(f"step_size must be > 0, got {self.step_size}")
        if self.damping_absolute is not None and self.damping_absolute <= 0.0:
            raise ValueError(
                f"damping_absolute must be > 0 when given, got {self.damping_absolute}"
            )

    @property
    def is_absolute(self) -> bool:
        """절대 damping 모드인지. ``True`` 면 분석용 오라클 action 이다."""
        return self.damping_absolute is not None


# ---------------------------------------------------------------------------
# Curvature 연산자
# ---------------------------------------------------------------------------


@runtime_checkable
class CurvatureOperator(Protocol):
    """``(H + lambda*I) v`` 를 matrix-free로 계산하는 연산자.

    전체 Hessian을 생성하거나 반환하는 메서드를 의도적으로 두지 않는다
    (README §15 구현 원칙).

    구현체는 호출 횟수를 스스로 누적해 ``hvp_count`` 로 노출해야 한다.
    CG solver가 비용을 보고할 때 이 값을 사용한다.
    """

    @property
    def damping(self) -> float:
        """현재 damping 계수 lambda."""
        ...

    @property
    def hvp_count(self) -> int:
        """생성 이후 누적 HVP 호출 횟수."""
        ...

    def matvec(self, v: Tensor) -> Tensor:
        """flatten된 벡터 ``v`` 에 ``(H + lambda*I)`` 를 곱한다."""
        ...

    def reset_count(self) -> None:
        """누적 HVP 횟수를 0으로 되돌린다."""
        ...


@runtime_checkable
class Preconditioner(Protocol):
    """CG preconditioner. ``M^{-1} r`` 을 계산한다.

    Stage 5(README Phase 3) 확장 지점이다. 기본 실험에서는 항등 연산을 쓴다.
    """

    def apply(self, r: Tensor) -> Tensor: ...


# ---------------------------------------------------------------------------
# Solver 결과
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """CG solver 설정. RL action과 무관하게 고정되는 값들."""

    tolerance: float = 1.0e-3
    max_iters: int = 20
    min_damping: float = 1.0e-6
    max_damping: float = 1.0e3
    pap_eps: float = 1.0e-12
    """negative curvature 판정 임계값. **상대 기준**이다.

    ``p^T A p <= pap_eps * ||p||^2`` 이면 탐지된 것으로 본다. 절대 기준을 쓰면
    ``p`` 가 작아지는 수렴 구간에서 양정 행렬에도 오탐이 생긴다.
    """


@dataclass(frozen=True, slots=True)
class CGResult:
    """Conjugate Gradient 실행 결과.

    ``hvp_count`` 와 실패 플래그는 필수 필드다. 비용 회계와 실패율 집계가
    전부 이 값에 의존한다.

    Attributes:
        solution: 근사해 ``p``. flatten된 1차원 텐서.
        iterations: 실제 수행한 CG 반복 수.
        hvp_count: 소모한 HVP 횟수.
        budget: 허용된 최대 반복 수. RL action의 ``cg_budget``.
        initial_residual: 초기 residual norm ``||r_0||``.
        final_residual: 최종 residual norm ``||r_k||``.
        converged: tolerance 기준 수렴 여부.
        negative_curvature: negative curvature 탐지 여부.
        numerical_failure: NaN/Inf 또는 CG 붕괴 발생 여부.
    """

    solution: Tensor
    iterations: int
    hvp_count: int
    budget: int
    initial_residual: float
    final_residual: float
    converged: bool
    negative_curvature: bool
    numerical_failure: bool

    @property
    def residual_ratio(self) -> float:
        """``||r_k|| / ||r_0||``. 0 나눗셈은 1.0으로 처리한다.

        RL 상태 특징 ``cg_residual_ratio``. 작을수록 선형계를 정확히 풀었다는 뜻이다.
        """
        if self.initial_residual <= 0.0:
            return 1.0
        return self.final_residual / self.initial_residual

    @property
    def iters_used_ratio(self) -> float:
        """사용한 반복 수 / 허용 예산.

        RL 상태 특징 ``cg_iters_used_ratio``. 1.0 이면 예산을 다 썼다는
        뜻이므로 예산이 부족했을 가능성을 시사한다. 1.0 미만이면 tolerance
        기준으로 조기 종료했거나 negative curvature 로 중단된 것이다.
        """
        if self.budget <= 0:
            return 0.0
        return self.iterations / self.budget


# ---------------------------------------------------------------------------
# step 단위 로그 레코드
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StepRecord:
    """optimizer step 하나에 대한 로그 레코드.

    ``docs/experiment_protocol.md`` §3 스키마를 따른다. JSONL 한 줄에 대응한다.
    README §13 스키마에 비용/provenance 필드가 추가되어 있다.
    """

    # --- 식별 ---
    run_id: str
    seed: int
    optimizer: str
    step: int

    # --- 품질 ---
    train_loss_before: float
    train_loss_after: float
    validation_loss: float | None = None
    validation_accuracy: float | None = None

    # --- 최적화 상태 ---
    grad_norm: float = float("nan")
    damping: float = float("nan")
    step_size: float = float("nan")
    cg_budget: int = 0
    cg_iterations: int = 0
    initial_residual: float = float("nan")
    final_residual: float = float("nan")
    trust_ratio: float = float("nan")
    predicted_reduction: float = float("nan")
    actual_reduction: float = float("nan")

    # --- 비용 (프로토콜 D1) ---
    hvp_count: int = 0
    forward_count: int = 0
    backward_count: int = 0
    cost_ge: float = float("nan")
    """grad-equivalent 환산 비용. 주 지표."""
    step_wall_time_sec: float = float("nan")
    peak_vram_mb: float = float("nan")

    # --- 안정성 ---
    step_accepted: bool = True
    negative_curvature: bool = False
    numerical_failure: bool = False
    failure_tag: FailureTag | None = None

    # --- provenance (프로토콜 §3) ---
    cost_model_id: str = ""
    git_commit: str = ""
    config_hash: str = ""
    task_instance_id: str = ""

    # --- 자유 확장 ---
    extra: dict[str, float | int | bool | str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.failure_tag is not None and self.failure_tag not in VALID_FAILURE_TAGS:
            raise ValueError(
                f"unknown failure_tag {self.failure_tag!r}; allowed: {sorted(VALID_FAILURE_TAGS)}"
            )
