"""컨트롤러가 주도하는 Newton-CG optimizer.

이 모듈에는 optimizer가 **하나만** 있다. fixed / heuristic / open_loop /
oracle / RL 의 차이는 optimizer 가 아니라 **누가 action 을 고르는가** 뿐이다.
그래서 ``Controller`` 프로토콜을 주입받는 단일 루프로 구현한다. 비교군 사이에
구현 차이로 인한 교란이 원천적으로 없다 (프로토콜 D4).

한 step 의 구조
---------------
```text
1. HvpGraph 생성            gradient + create_graph          (c_grad_graph GE)
2. 컨트롤러가 action 선택    damping / cg_budget / step_size
3. damping 적용 후 CG solve  (H + lambda I) p = -g            (k * c_hvp GE)
4. quadratic model 예측      p^T H p                          (1 * c_hvp GE)
5. candidate step 평가       L(theta + alpha p)               (c_fwd GE)
6. 수락 또는 거절            거절 시 damping 상향 + fallback
```

damping 은 로그 공간의 지속 상태다
----------------------------------
``u = log10(lambda)`` 를 상태로 들고, 상대 모드에서는 ``u <- u + log10(m)``,
절대 모드에서는 ``u <- log10(v)`` 로 갱신한 뒤 ``[u_min, u_max]`` 로 클립한다.

로그 공간을 쓰는 이유:

- 배수 누적이 덧셈이 되어 오버플로와 언더플로가 없다
- 클립 경계가 "몇 자릿수까지 허용"이라는 해석을 갖는다
- RL 상태 특징도 ``log10(lambda)`` 를 쓰므로 표현이 일치한다

경계 기본값은 ``[1e-8, 1e8]`` 이다. Stage 1 에서 ``kappa = 1e6`` 문제가
damping ``~1e6`` 을 요구한다는 것이 확인되었으므로, 이전 기본값 ``1e3`` 은
그 자체가 병목이었다.

step acceptance 를 두는 이유
----------------------------
RL agent 가 위험한 action 을 골라도 학습이 즉시 붕괴하지 않아야 한다. 다만
fallback 이 너무 관대하면 agent 가 그것을 악용한다. 그래서 실패는 반드시
``StepRecord`` 에 기록되고 보상에서 패널티로 반영된다 (프로토콜 D3).

높은 damping 의 함정
--------------------
``lambda`` 가 매우 크면 ``(H + lambda I)^{-1} g ~ g / lambda`` 이므로 update 가
``-(alpha/lambda) g`` 로 근사된다. 즉 **CG residual 은 빠르게 줄지만 실제
objective 감소는 느려질 수 있다.** 그래서 오라클이 최적화하는 목적은 CG 성공
여부가 아니라 ``objective 감소 / 비용`` 이어야 한다. 두 지표를 모두 기록해
"높은 damping 이 CG 만 쉽게 만들고 최적화를 망치는가"를 별도로 확인한다.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor

from rl_newton.curvature.hvp import HvpGraph
from rl_newton.curvature.operators import DampedHessianOperator
from rl_newton.solvers.conjugate_gradient import conjugate_gradient
from rl_newton.types import CGResult, ControllerAction, StepRecord
from rl_newton.utils.flatten import ParameterFlattener

if TYPE_CHECKING:
    # 런타임 import 를 피한다. ``benchmark.__init__`` 이 ``metrics`` 를 불러오고
    # ``metrics`` 가 다시 이 모듈을 참조하므로, 여기서 실제로 import 하면
    # ``optimizers`` 를 먼저 불러올 때 순환 import 가 된다.
    # ``from __future__ import annotations`` 덕분에 타입 힌트는 문자열이므로
    # 런타임에 필요하지 않다.
    from rl_newton.benchmark.cost_model import CostModel

__all__ = [
    "Controller",
    "StepContext",
    "NewtonCGConfig",
    "NewtonCGOptimizer",
    "OptimizationTrace",
    "Candidate",
    "TaskLike",
    "apply_damping_action",
]


class TaskLike(Protocol):
    """optimizer 가 필요로 하는 task 인터페이스.

    ``loss`` 와 ``curvature_loss`` 의 역할이 다르다.

    ```text
    curvature_loss()  optimizer 가 보는 것. gradient / HVP / 수락 판정
    loss()            평가용. Track E 점수
    ```

    결정론적 task 에서는 둘이 같다. minibatch task 에서만 갈린다 (D24).

    ``advance_batch()`` 는 **선택 사항**이다. 있으면 실제 step 뒤에 호출된다.
    """

    @property
    def params(self) -> list[Tensor]: ...
    @property
    def instance_id(self) -> str: ...
    @property
    def initial_loss(self) -> float: ...
    def loss(self) -> Tensor: ...
    def curvature_loss(self) -> Tensor: ...
    def reset(self) -> None: ...


def apply_damping_action(
    damping_log10: float,
    action: ControllerAction,
    *,
    min_log10: float,
    max_log10: float,
) -> float:
    """action 을 적용한 새 ``log10(damping)`` 을 반환한다.

    상대 모드는 덧셈, 절대 모드는 대입이다. 어느 쪽이든 경계로 클립한다.

    Example:
        >>> from rl_newton.types import ControllerAction
        >>> a = ControllerAction(damping_multiplier=3.0, cg_budget=5, step_size=1.0)
        >>> round(apply_damping_action(-2.0, a, min_log10=-8, max_log10=8), 6)
        -1.522879
        >>> b = ControllerAction(damping_multiplier=1.0, cg_budget=5, step_size=1.0,
        ...                      damping_absolute=1e6)
        >>> apply_damping_action(-2.0, b, min_log10=-8, max_log10=8)
        6.0
    """
    if action.damping_absolute is not None:
        target = math.log10(action.damping_absolute)
    else:
        target = damping_log10 + math.log10(action.damping_multiplier)
    return min(max(target, min_log10), max_log10)


@dataclass(frozen=True, slots=True)
class StepContext:
    """컨트롤러가 action 을 고를 때 볼 수 있는 정보.

    RL 상태 특징(README §5.1)의 원재료다. 여기 없는 정보는 컨트롤러가 쓸 수
    없다. ``progress`` 를 포함하되, 이것만으로 결정하는 컨트롤러가
    ``open_loop`` baseline 이다 (프로토콜 D4).

    Attributes:
        step: 현재 step 인덱스 (0-based).
        total_steps: 전체 step 예산.
        loss: 현재 loss.
        grad_norm: 현재 gradient norm.
        damping: 현재 damping ``lambda``.
        previous: 직전 step 의 레코드. 첫 step 에서는 ``None``.
        history: 지금까지의 레코드 전체 (읽기 전용으로 취급).
    """

    step: int
    total_steps: int
    loss: float
    grad_norm: float
    damping: float
    previous: StepRecord | None = None
    history: Sequence[StepRecord] = ()
    spent_ge: float = 0.0
    """이 step 시작 시점까지 소모한 object-level GE."""
    cost_budget_ge: float | None = None
    """GE 예산. ``None`` 이면 step 예산으로 종료한다."""

    @property
    def progress(self) -> float:
        """**소모 GE 비율**. ``open_loop`` 컨트롤러의 유일한 입력이다 (프로토콜 D17).

        초판은 ``step / total_steps`` 였다. 그런데 종료는 GE 예산으로 하므로
        (D1) 두 시계가 불일치했다. ``total_steps=200``, 예산 150 GE, ``k=20`` 이면
        약 7 step 만에 끝나서 ``progress`` 가 0.035 를 넘지 못했고, **스케줄의
        첫 구간만 실행됐다.**

        실측: ``best_open_loop`` 이 4구간 스케줄을 골랐는데 결과가
        ``best_static`` 과 9쌍 전부 bitwise 동일했다 (delta CI ``+0.000~+0.000``).
        비정적 스케줄이 static 으로 퇴화한 것이 아니라 **뒤쪽 구간이 실행될 수
        없었던 baseline 구현 결함**이다.

        지금은 예산 소모 비율을 쓴다. 이것을 **budget-indexed open-loop schedule**
        (resource-clock schedule) 이라 부른다. 실제 GE 소비는 CG 조기 종료 등으로
        변하지만, 컨트롤러가 loss / gradient / Hessian 상태를 **관찰하지 않고**
        계산 예산 시계만 쓰므로 적응 제어와 구별되는 강한 단순 baseline 이다.

        GE 예산이 없으면 step 비율로 되돌아간다.
        """
        if self.cost_budget_ge is not None and self.cost_budget_ge > 0.0:
            return min(1.0, self.spent_ge / self.cost_budget_ge)
        if self.total_steps <= 0:
            return 0.0
        return self.step / self.total_steps

    @property
    def log_damping(self) -> float:
        """``log10(lambda)``. RL 상태 특징에는 raw damping 대신 이 값을 쓴다."""
        return math.log10(max(self.damping, 1e-300))


class Controller(Protocol):
    """action 을 고르는 주체.

    ``select`` 는 파라미터를 영구히 바꾸지 않아야 한다. 후보를 실제로 평가해야
    하는 컨트롤러는 ``NewtonCGOptimizer`` 가 제공하는 경로로만 평가한다. 그래야
    비용이 회계되고 파라미터가 복원된다.
    """

    @property
    def name(self) -> str: ...

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction: ...

    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class NewtonCGConfig:
    """optimizer 설정. 컨트롤러가 고르지 않는 값들이다."""

    total_steps: int = 50
    """step 수 상한. ``cost_budget_ge`` 가 주어지면 그쪽이 실질적 종료 조건이다."""
    cost_budget_ge: float | None = None
    """GE 예산. 누적 비용이 이 값에 도달하면 종료한다.

    **비교는 반드시 이 기준으로 해야 한다.** Newton-CG 는 한 step 의 비용이
    action 에 따라 크게 달라지므로(k=3 과 k=20 은 6배 이상 차이), step 수를
    맞추고 최종 loss 를 비교하면 비용이 다른 것들을 비교하게 된다
    (README §4.2). ``None`` 이면 step 수로만 끊는다 (진단 용도).
    """
    initial_damping: float = 1.0e-2
    min_damping: float = 1.0e-8
    max_damping: float = 1.0e8
    """Stage 1 에서 kappa=1e6 이 damping ~1e6 을 요구함이 확인되었다.
    이전 기본값 1e3 은 그 자체로 병목이었다."""
    cg_tolerance: float = 1.0e-3
    pap_eps: float = 1.0e-12
    max_loss_increase_ratio: float = 1.5
    """candidate loss 가 현재 loss 의 이 배수를 넘으면 발산으로 본다."""
    reject_damping_multiplier: float = 3.0
    nan_damping_multiplier: float = 10.0
    safe_fallback: str = "gradient"
    """``none`` | ``gradient``. 거절 후 clipped gradient step 을 적용할지."""
    fallback_step_size: float = 1.0e-3
    fallback_grad_clip: float = 1.0
    compute_trust_ratio: bool = True
    """``True`` 면 ``p^T H p`` 계산에 HVP 1회를 더 쓴다. 비용에 포함된다."""
    acceptance_loss: str = "control"
    """``control`` | ``fixed_eval``. 수락 판정을 어떤 목적함수로 하는가 (D28).

    ```text
    control     gradient / HVP 와 같은 표본. 결정론적 task 에서는 유일한 선택
    fixed_eval  step 마다 바뀌지 않는 목적함수. task 가 acceptance_loss() 를
                제공할 때만 유효하다
    ```

    `_accept` 는 단조 감소를 요구한다. minibatch 목적함수에서는 참 목적함수를
    개선하는 step 도 표본 잡음 때문에 거절될 수 있다. `fixed_eval` 은 그 교란을
    분리하는 ablation 이다. gradient 와 HVP 는 계속 minibatch 를 쓴다.

    **비용을 숨기지 않는다.** 고정 평가 forward 는 `acceptance_forward_units` 배로
    회계에 들어간다.
    """

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {self.total_steps}")
        if self.acceptance_loss not in ("control", "fixed_eval"):
            raise ValueError(f"unknown acceptance_loss: {self.acceptance_loss!r}")
        if self.cost_budget_ge is not None and self.cost_budget_ge <= 0.0:
            raise ValueError(f"cost_budget_ge must be > 0 when given, got {self.cost_budget_ge}")
        if self.min_damping <= 0.0:
            raise ValueError(f"min_damping must be > 0 for log-space state, got {self.min_damping}")
        if self.min_damping > self.max_damping:
            raise ValueError(
                f"require min_damping <= max_damping, got {self.min_damping}, {self.max_damping}"
            )
        if not self.min_damping <= self.initial_damping <= self.max_damping:
            raise ValueError(
                f"initial_damping {self.initial_damping} must lie in "
                f"[{self.min_damping}, {self.max_damping}]"
            )
        if self.max_loss_increase_ratio <= 1.0:
            raise ValueError(
                f"max_loss_increase_ratio must be > 1, got {self.max_loss_increase_ratio}"
            )
        if self.safe_fallback not in ("none", "gradient"):
            raise ValueError(f"unknown safe_fallback: {self.safe_fallback!r}")

    @property
    def min_damping_log10(self) -> float:
        return math.log10(self.min_damping)

    @property
    def max_damping_log10(self) -> float:
        return math.log10(self.max_damping)

    @property
    def initial_damping_log10(self) -> float:
        return math.log10(self.initial_damping)


@dataclass(slots=True)
class Candidate:
    """전수 탐색에서 평가한 하나의 후보."""

    action: ControllerAction
    cg: CGResult
    candidate_loss: float
    cost_ge: float
    """이 후보를 **채택했을 때** 소모될 비용. 탐색 비용은 별도로 회계한다."""
    applied_damping: float = float("nan")

    @property
    def is_finite(self) -> bool:
        return math.isfinite(self.candidate_loss)


@dataclass(slots=True)
class OptimizationTrace:
    """한 번의 최적화 run 결과."""

    run_id: str
    controller: str
    task_instance_id: str
    seed: int
    records: list[StepRecord] = field(default_factory=list)
    initial_loss: float = float("nan")
    final_loss: float = float("nan")
    total_cost_ge: float = 0.0
    total_hvp: int = 0
    search_cost_ge: float = 0.0
    """컨트롤러가 후보 탐색에 쓴 추가 비용. oracle 은 크고 나머지는 0이다.

    프로토콜 D5(탐색 예산 회계)에 따라 별도로 보고한다. 오라클의 비용을
    본문 비용에 섞으면 헤드룸이 과소평가된다.
    """
    search_hvp: int = 0
    stop_reason: str = "step_budget"
    """``step_budget`` | ``cost_budget`` | ``nan``. 절단 규칙 집계에 쓴다."""

    @property
    def n_steps(self) -> int:
        return len(self.records)

    @property
    def n_rejected(self) -> int:
        return sum(1 for r in self.records if not r.step_accepted)

    @property
    def n_failures(self) -> int:
        return sum(1 for r in self.records if r.numerical_failure)

    @property
    def n_negative_curvature(self) -> int:
        return sum(1 for r in self.records if r.negative_curvature)

    @property
    def n_cg_converged(self) -> int:
        """CG 가 tolerance 를 만족한 step 수.

        loss 감소와 **분리해서** 본다. 높은 damping 은 CG 를 쉽게 만들지만
        최적화를 느리게 할 수 있다 (모듈 docstring 참조).
        """
        return sum(
            1 for r in self.records if float(r.extra.get("cg_residual_ratio", 1.0)) <= 1.0e-3
        )

    def loss_curve(self) -> list[float]:
        """step 별 loss (step 후 값)."""
        return [r.train_loss_after for r in self.records]

    def cumulative_cost_ge(self) -> list[float]:
        """step 별 누적 GE 비용. 주 그림의 x축이다 (프로토콜 §8)."""
        out: list[float] = []
        total = 0.0
        for r in self.records:
            total += r.cost_ge if math.isfinite(r.cost_ge) else 0.0
            out.append(total)
        return out

    def damping_curve(self) -> list[float]:
        return [r.damping for r in self.records]


@dataclass(slots=True)
class _Outcome:
    """``_execute_step`` 의 반환값. 내부용."""

    loss_after: float
    cost_ge: float
    accepted: bool
    record: StepRecord | None = None


class _StepScope:
    """HvpGraph 의 생성과 해제를 관리한다.

    그래프는 step 당 1회만 만드는 것이 원칙이다 (프로토콜 D1의 비용식).
    다만 look-ahead 시뮬레이션은 파라미터를 바꾸며 진행하므로 매번 새 그래프가
    필요하다. ``force_new`` 가 그 경우를 처리하고, 바깥 스코프의 연산자를
    저장해 두었다가 복원한다.
    """

    def __init__(self, optimizer: NewtonCGOptimizer, *, force_new: bool) -> None:
        self._opt = optimizer
        self._force_new = force_new
        self._owned = False
        self._saved_operator: DampedHessianOperator | None = None
        self._saved_params: Tensor | None = None

    def __enter__(self) -> bool:
        opt = self._opt
        if opt._operator is not None and not self._force_new:
            return True

        self._saved_operator = opt._operator
        self._saved_params = opt._base_params

        graph = HvpGraph(opt.task.curvature_loss, opt._flat.params, flattener=opt._flat)
        operator = DampedHessianOperator(
            graph,
            damping=opt.damping,
            min_damping=opt.config.min_damping,
            max_damping=opt.config.max_damping,
        )
        opt._operator = operator
        opt._base_params = opt._flat.flatten_params()
        self._owned = True
        return bool(torch.isfinite(operator.grad).all()) and math.isfinite(operator.loss)

    def __exit__(self, *exc_info: object) -> None:
        if not self._owned:
            return
        opt = self._opt
        if opt._operator is not None:
            opt._operator.release()
        opt._operator = self._saved_operator
        opt._base_params = self._saved_params


class NewtonCGOptimizer:
    """단일 Newton-CG 루프. 컨트롤러를 주입받는다.

    Args:
        task: 최적화 대상.
        controller: action 선택자.
        config: optimizer 설정.
        cost_model: GE 환산 계수. ``None`` 이면 HVP 횟수를 단위로 쓴다
            (synthetic task 에서는 이게 더 해석이 쉽다).
        run_id: 로그 식별자.
        seed: 실험 조건 식별자.

    Example:
        >>> from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask
        >>> from rl_newton.optimizers.controllers import FixedController
        >>> from rl_newton.optimizers.action_space import NARROW
        >>> task = QuadraticTask(QuadraticSpec(dimension=16), seed=0)
        >>> ctrl = FixedController(NARROW.action_from_flat(0))
        >>> opt = NewtonCGOptimizer(task, ctrl, NewtonCGConfig(total_steps=5))
        >>> trace = opt.run()
        >>> trace.n_steps
        5
        >>> trace.final_loss < trace.initial_loss
        True
    """

    def __init__(
        self,
        task: TaskLike,
        controller: Controller,
        config: NewtonCGConfig | None = None,
        *,
        cost_model: CostModel | None = None,
        run_id: str = "run",
        seed: int = 0,
    ) -> None:
        self.task = task
        self.controller = controller
        self.config = config or NewtonCGConfig()
        self.cost_model = cost_model
        self.run_id = run_id
        self.seed = seed

        self._flat = ParameterFlattener(task.params)
        self._damping_log10 = self.config.initial_damping_log10
        self._operator: DampedHessianOperator | None = None

        # D28. task 가 고정 평가 목적함수를 제공하지 않으면 조용히 `control` 로
        # 되돌린다. 결정론적 task 에서는 두 목적함수가 같으므로 차이가 없다.
        self._fixed_eval = self.config.acceptance_loss == "fixed_eval" and hasattr(
            task, "acceptance_loss"
        )
        self._accept_units = (
            float(getattr(task, "acceptance_forward_units", 1.0))
            if self._fixed_eval
            else 1.0
        )
        self._base_params: Tensor | None = None
        self._search_hvp = 0
        self._search_forward: float = 0.0
        self._search_graph_count = 0
        self._graph_dirty = False
        """파라미터가 in-place 로 변경되어 현재 HVP 그래프가 무효한지 여부."""

    # --- 상태 -------------------------------------------------------------

    @property
    def damping(self) -> float:
        """현재 damping ``lambda``."""
        return 10.0**self._damping_log10

    @property
    def damping_log10(self) -> float:
        return self._damping_log10

    @property
    def flattener(self) -> ParameterFlattener:
        return self._flat

    def set_damping_log10(self, value: float) -> float:
        """damping 상태를 직접 설정한다. look-ahead 시뮬레이션 복원용이다."""
        cfg = self.config
        self._damping_log10 = min(max(value, cfg.min_damping_log10), cfg.max_damping_log10)
        return self._damping_log10

    # --- 비용 회계 --------------------------------------------------------

    def step_cost_ge(self, hvp: int, forward: float, *, with_graph: bool = True) -> float:
        """GE 환산 비용. cost_model 이 없으면 HVP 를 단위로 쓴다.

        ``forward`` 는 실수다. 고정 평가 목적함수 forward 는 control forward 보다
        비싸므로 배수로 센다 (D28).
        """
        if self.cost_model is None:
            # synthetic task 에서는 "HVP 등가 횟수" 로 해석한다.
            # forward 는 HVP 보다 훨씬 싸므로 0.3 상당으로 근사한다.
            return hvp + 0.3 * forward + (1.0 if with_graph else 0.0)
        cm = self.cost_model
        total = hvp * cm.c_hvp + forward * cm.c_fwd
        if with_graph:
            total += cm.c_grad_graph
        return total

    # --- 후보 평가 (컨트롤러가 호출) --------------------------------------

    @property
    def operator(self) -> DampedHessianOperator:
        """현재 step 의 curvature 연산자.

        Raises:
            RuntimeError: step 밖에서 호출한 경우.
        """
        if self._operator is None:
            raise RuntimeError("operator is only available inside a step")
        return self._operator

    def solve_direction(self, action: ControllerAction) -> tuple[CGResult, float]:
        """주어진 action 으로 CG 를 풀어 ``(결과, 적용된 damping)`` 을 반환한다.

        연산자의 HVP 그래프를 재사용하므로 forward/backward 를 다시 하지 않는다.
        ``self._damping_log10`` 은 바꾸지 않는다 (탐색은 상태를 오염시키지 않는다).
        """
        cfg = self.config
        op = self.operator
        target_log = apply_damping_action(
            self._damping_log10,
            action,
            min_log10=cfg.min_damping_log10,
            max_log10=cfg.max_damping_log10,
        )
        applied = 10.0**target_log
        previous = op.damping
        op.set_damping(applied)
        try:
            result = conjugate_gradient(
                op,
                -op.grad,
                max_iters=action.cg_budget,
                tolerance=cfg.cg_tolerance,
                pap_eps=cfg.pap_eps,
            )
        finally:
            op.set_damping(previous)
        return result, applied

    def control_loss(self) -> float:
        """**optimizer 가 보는 loss.** gradient / HVP 와 같은 표본이어야 한다.

        `_StepScope` 가 `HvpGraph(task.curvature_loss, ...)` 로 그래프를 만들므로
        `operator.loss` 는 `curvature_loss` 값이다. candidate 평가와 거절 처리도
        같은 표본을 써야 `loss_before` 와 비교가 성립한다.

        결정론적 task 에서는 `loss()` 와 `curvature_loss()` 가 같은 값이므로 이
        구분이 무의미하다. minibatch task (D24 R2) 에서만 갈린다.
        """
        with torch.no_grad():
            return float(self.task.curvature_loss().detach())

    @property
    def uses_fixed_eval_acceptance(self) -> bool:
        """수락 판정이 고정 평가 목적함수를 쓰는가 (D28)."""
        return self._fixed_eval

    def step_objective(self) -> float:
        """**수락 판정에 쓰는 목적함수 값.**

        `control` 모드에서는 `control_loss()` 와 같다. `fixed_eval` 모드에서만
        `task.acceptance_loss()` 로 갈린다 (D28).
        """
        if not self._fixed_eval:
            return self.control_loss()
        with torch.no_grad():
            return float(self.task.acceptance_loss().detach())  # type: ignore[attr-defined]

    def evaluate_loss_at(self, direction: Tensor, step_size: float) -> float:
        """``L(theta + step_size * direction)``. 파라미터는 값으로 복원한다.

        **주의: 이 호출은 HVP 그래프를 무효화한다.** 파라미터를 in-place 로
        바꾸면 autograd 의 버전 카운터가 올라가고, 그 파라미터를 leaf 로 갖는
        그래프는 이후 backward 에서 ``RuntimeError`` 를 던진다. 값을 되돌려도
        버전은 되돌아가지 않는다.

        그래서 ``_graph_dirty`` 를 세운다. 이후 curvature 가 필요한 연산은
        새 그래프를 만들어야 한다.

        **수락 판정에 쓰는 목적함수를 쓴다.** `loss_before` 와 같은 목적함수여야
        비교가 성립한다 (D28).
        """
        assert self._base_params is not None
        self._flat.add_(direction, alpha=step_size)
        self._graph_dirty = True
        try:
            return self.step_objective()
        finally:
            self._flat.copy_from_(self._base_params)

    def evaluate_candidates(
        self, solve_groups: Iterable[tuple[ControllerAction, tuple[float, ...]]]
    ) -> list[Candidate]:
        """전수 탐색. ``ActionSpace.iter_solve_groups`` 의 출력을 받는다.

        **두 단계로 분리해야 한다.** CG solve 는 살아 있는 HVP 그래프를 쓰고,
        candidate loss 평가는 파라미터를 in-place 로 바꿔 그 그래프를 무효화한다.
        섞으면 두 번째 solve 에서 autograd 가 실패한다. 그래서

        ```text
        1단계  모든 CG solve      (그래프 필요, 파라미터 불변)
        2단계  모든 loss 평가      (파라미터 변경, 그래프 불필요)
        ```

        같은 ``(damping, budget)`` 그룹의 step_size 들이 하나의 CG 결과를
        공유하므로 CG solve 는 ``damping x budget`` 회만 수행된다.

        탐색에 든 비용은 ``search_cost_ge`` 로 누적되며 본문 비용과 분리된다.
        소모한 그래프 1개의 비용도 탐색 비용에 포함한다.
        """
        op = self.operator
        hvp_before = op.hvp_count

        # --- 1단계: 모든 CG solve ---
        solved: list[tuple[ControllerAction, tuple[float, ...], CGResult, float]] = []
        for representative, step_sizes in solve_groups:
            cg, applied = self.solve_direction(representative)
            solved.append((representative, step_sizes, cg, applied))
        self._search_hvp += op.hvp_count - hvp_before
        # 이 탐색이 소모한 그래프 비용. 실제 step 은 새 그래프를 만들어야 한다.
        self._search_graph_count += 1

        # --- 2단계: 모든 candidate loss 평가 ---
        candidates: list[Candidate] = []
        for representative, step_sizes, cg, applied in solved:
            for step_size in step_sizes:
                action = ControllerAction(
                    damping_multiplier=representative.damping_multiplier,
                    cg_budget=representative.cg_budget,
                    step_size=step_size,
                    damping_absolute=representative.damping_absolute,
                )
                candidate_loss = self.evaluate_loss_at(cg.solution, step_size)
                self._search_forward += self._accept_units
                candidates.append(
                    Candidate(
                        action=action,
                        cg=cg,
                        candidate_loss=candidate_loss,
                        cost_ge=self.step_cost_ge(cg.hvp_count, 1),
                        applied_damping=applied,
                    )
                )
        return candidates

    # --- look-ahead 지원 --------------------------------------------------

    def snapshot(self) -> tuple[Tensor, float]:
        """``(파라미터, log10 damping)`` 스냅샷. look-ahead 시뮬레이션 복원용."""
        return self._flat.flatten_params(), self._damping_log10

    def restore(self, snapshot: tuple[Tensor, float]) -> None:
        params, damping_log10 = snapshot
        self._flat.copy_from_(params)
        self._damping_log10 = damping_log10

    def simulate_step(self, action: ControllerAction) -> tuple[float, float, bool]:
        """action 을 실제로 적용한다. ``(loss_after, cost_ge, accepted)`` 반환.

        로깅하지 않고 컨트롤러도 호출하지 않는 순수 실행 경로다. look-ahead
        오라클이 beam 을 전개할 때 사용한다. 파라미터와 damping 상태를
        **변경하므로** 호출자가 ``snapshot`` / ``restore`` 로 관리해야 한다.

        **항상 새 HvpGraph 를 만든다.** 시뮬레이션이 진행되면 파라미터가
        바뀌므로 바깥 step 의 그래프를 재사용하면 낡은 Hessian 을 쓰게 된다.
        비용은 탐색 비용으로 회계된다.
        """
        outcome = self._execute_step(action, record=False, reuse_operator=False)
        self._graph_dirty = True
        return outcome.loss_after, outcome.cost_ge, outcome.accepted

    # --- 메인 루프 --------------------------------------------------------

    def run(self, *, reset_task: bool = True) -> OptimizationTrace:
        """``total_steps`` 만큼 최적화를 수행한다."""
        if reset_task:
            self.task.reset()
        self.controller.reset()
        self._damping_log10 = self.config.initial_damping_log10
        self._search_hvp = 0
        self._search_forward = 0.0
        self._search_graph_count = 0
        self._graph_dirty = False

        trace = OptimizationTrace(
            run_id=self.run_id,
            controller=self.controller.name,
            task_instance_id=self.task.instance_id,
            seed=self.seed,
        )
        trace.initial_loss = float(self.task.loss().detach())

        budget = self.config.cost_budget_ge
        spent = 0.0
        for step in range(self.config.total_steps):
            record = self._run_step(step, trace)
            trace.records.append(record)
            # 실제 step 뒤에만 batch 를 전진시킨다 (D24). planner 의 look-ahead
            # 시뮬레이션은 `_execute_step(record=False)` 로 돌아 여기 오지 않으므로,
            # planner 는 현재 batch 로 미래를 예측한다. 미래 batch 를 미리 보면
            # 데이터 oracle 이 되어 feedback 검증이 무의미해진다.
            advance = getattr(self.task, "advance_batch", None)
            if advance is not None:
                advance()
            if math.isfinite(record.cost_ge):
                spent += record.cost_ge
            if not math.isfinite(record.train_loss_after):
                trace.stop_reason = "nan"
                break
            if budget is not None and spent >= budget:
                trace.stop_reason = "cost_budget"
                break
        else:
            trace.stop_reason = "step_budget"

        # **평가 loss 로 점수를 매긴다.** 결정론적 task 에서는 마지막 기록과
        # bitwise 동일하다 (같은 파라미터에서 같은 식을 다시 계산). minibatch task
        # 에서는 control loss 가 표본마다 다르므로 전체 데이터 값을 써야 Track E 가
        # regime 간에 비교 가능하다 (D24).
        #
        # 마지막 기록이 비유한값이면 기존 동작을 유지한다. 실패한 run 의 지표를
        # 조용히 유한값으로 바꾸면 실패가 숨는다.
        if not trace.records:
            trace.final_loss = trace.initial_loss
        elif math.isfinite(trace.records[-1].train_loss_after):
            with torch.no_grad():
                trace.final_loss = float(self.task.loss().detach())
        else:
            trace.final_loss = trace.records[-1].train_loss_after
        trace.total_cost_ge = sum(r.cost_ge for r in trace.records if math.isfinite(r.cost_ge))
        trace.total_hvp = sum(r.hvp_count for r in trace.records)
        trace.search_hvp = self._search_hvp
        # 탐색이 소모한 그래프 비용도 포함한다. cost_model 이 없으면 그래프 1개는
        # 1.0 (gradient 1회 상당) 으로 센다.
        graph_cost = self._search_graph_count * (
            1.0 if self.cost_model is None else self.cost_model.c_grad_graph
        )
        trace.search_cost_ge = (
            self.step_cost_ge(self._search_hvp, self._search_forward, with_graph=False) + graph_cost
        )
        return trace

    def _run_step(self, step: int, trace: OptimizationTrace) -> StepRecord:
        # 컨트롤러가 보는 loss 는 수락 판정에 쓰는 목적함수와 같아야 한다. 자신이
        # 판정받는 기준을 봐야 하기 때문이다. 결정론적 task 에서는 `loss()` 와 동일하다.
        loss_before = self.step_objective()
        if not math.isfinite(loss_before):
            return self._failed_record(step, loss_before, loss_before, "nan")

        with _StepScope(self, force_new=True) as ok:
            if not ok:
                return self._failed_record(step, loss_before, loss_before, "nan")
            operator = self.operator
            context = StepContext(
                step=step,
                total_steps=self.config.total_steps,
                loss=loss_before,
                grad_norm=float(operator.grad.norm()),
                damping=self.damping,
                previous=trace.records[-1] if trace.records else None,
                history=trace.records,
                # step 시작 시점의 누적 비용. resource clock 의 분자다 (D17).
                spent_ge=sum(
                    r.cost_ge for r in trace.records if math.isfinite(r.cost_ge)
                ),
                cost_budget_ge=self.config.cost_budget_ge,
            )
            self._graph_dirty = False
            action = self.controller.select(context, self)
            # 컨트롤러가 후보를 평가했다면 파라미터가 in-place 로 변경되어
            # 그래프가 무효화됐다. 값은 복원되었지만 autograd 버전 카운터는
            # 되돌아가지 않으므로 새 그래프가 필요하다.
            outcome = self._execute_step(
                action,
                record=True,
                step=step,
                reuse_operator=not self._graph_dirty,
            )
            assert outcome.record is not None
            return outcome.record

    # --- step 실행 코어 ---------------------------------------------------

    def _execute_step(
        self,
        action: ControllerAction,
        *,
        record: bool,
        step: int = -1,
        reuse_operator: bool = True,
    ) -> _Outcome:
        """action 하나를 실제로 적용한다.

        Args:
            action: 적용할 제어값.
            record: ``StepRecord`` 를 만들지 여부. ``False`` 면 비용이 탐색
                비용으로 회계된다 (look-ahead 시뮬레이션).
            step: 로그용 step 인덱스.
            reuse_operator: ``True`` 면 이미 열려 있는 HvpGraph 를 재사용한다.
                ``False`` 면 현재 파라미터에서 새 그래프를 만든다. 시뮬레이션이
                진행되어 파라미터가 바뀐 경우 반드시 ``False`` 여야 한다.
        """
        cfg = self.config
        with _StepScope(self, force_new=not reuse_operator) as ok:
            operator = self.operator
            # `control` 모드에서는 그래프가 이미 계산한 값을 쓴다 (추가 비용 0).
            # `fixed_eval` 모드에서는 별도 forward 가 필요하고 그 비용을 센다 (D28).
            loss_before = operator.loss if not self._fixed_eval else self.step_objective()
            if not ok or not math.isfinite(loss_before):
                return _Outcome(
                    loss_after=float("nan"),
                    cost_ge=self.step_cost_ge(0, 0),
                    accepted=False,
                    record=(
                        self._failed_record(step, loss_before, float("nan"), "nan")
                        if record
                        else None
                    ),
                )

            grad = operator.grad
            grad_norm = float(grad.norm())
            hvp_before = operator.hvp_count
            # **호출 횟수와 비용 단위를 분리한다** (D28).
            #   forward_calls  기록용 정수. `StepRecord.forward_count` 계약을 지킨다
            #   forward_units  GE 회계용 실수. 고정 평가 forward 는 더 비싸다
            # `control` 모드에서는 두 값이 같으므로 기존 기록이 그대로 보존된다.
            forward_calls = 1 if self._fixed_eval else 0
            forward_units: float = self._accept_units if self._fixed_eval else 0.0

            # --- damping 갱신 (지속 상태) ---
            self._damping_log10 = apply_damping_action(
                self._damping_log10,
                action,
                min_log10=cfg.min_damping_log10,
                max_log10=cfg.max_damping_log10,
            )
            applied_damping = 10.0**self._damping_log10
            operator.set_damping(applied_damping)

            # --- CG solve ---
            cg = conjugate_gradient(
                operator,
                -grad,
                max_iters=action.cg_budget,
                tolerance=cfg.cg_tolerance,
                pap_eps=cfg.pap_eps,
            )
            direction = cg.solution

            # --- quadratic model 예측 감소량 ---
            directional = float(torch.dot(grad, direction))
            curvature = float("nan")
            predicted = float("nan")
            if cfg.compute_trust_ratio:
                hp = operator.matvec(direction)
                curvature = float(torch.dot(direction, hp)) - applied_damping * float(
                    torch.dot(direction, direction)
                )
                predicted = -(
                    action.step_size * directional + 0.5 * action.step_size**2 * curvature
                )

            # --- candidate 평가와 수락 판정 ---
            candidate_loss = self.evaluate_loss_at(direction, action.step_size)
            forward_calls += 1
            forward_units += self._accept_units
            accepted, failure_tag = self._accept(loss_before, candidate_loss, cg)

            if accepted:
                self._flat.add_(direction, alpha=action.step_size)
                loss_after = candidate_loss
            else:
                loss_after = self._handle_rejection(grad, failure_tag)
                forward_calls += 1
                forward_units += self._accept_units

            hvp_used = operator.hvp_count - hvp_before
            cost_ge = self.step_cost_ge(hvp_used, forward_units)

            if not record:
                self._search_hvp += hvp_used
                self._search_forward += forward_units
                return _Outcome(loss_after=loss_after, cost_ge=cost_ge, accepted=accepted)

            actual = loss_before - loss_after
            trust_ratio = (
                actual / predicted
                if math.isfinite(predicted) and abs(predicted) > 1e-300
                else float("nan")
            )
            step_record = StepRecord(
                run_id=self.run_id,
                seed=self.seed,
                optimizer=self.controller.name,
                step=step,
                train_loss_before=loss_before,
                train_loss_after=loss_after,
                grad_norm=grad_norm,
                damping=applied_damping,
                step_size=action.step_size,
                cg_budget=action.cg_budget,
                cg_iterations=cg.iterations,
                initial_residual=cg.initial_residual,
                final_residual=cg.final_residual,
                trust_ratio=trust_ratio,
                predicted_reduction=predicted,
                actual_reduction=actual,
                hvp_count=hvp_used,
                forward_count=forward_calls,
                backward_count=1,
                cost_ge=cost_ge,
                step_accepted=accepted,
                negative_curvature=cg.negative_curvature,
                numerical_failure=cg.numerical_failure or not math.isfinite(loss_after),
                failure_tag=failure_tag,
                task_instance_id=self.task.instance_id,
                extra={
                    "directional_derivative": directional,
                    "curvature": curvature,
                    "damping_multiplier": action.damping_multiplier,
                    "damping_absolute": action.damping_absolute,
                    "log_damping": self._damping_log10,
                    "cg_residual_ratio": cg.residual_ratio,
                    "cg_iters_used_ratio": cg.iters_used_ratio,
                    "cg_converged": cg.converged,
                },
            )
            return _Outcome(
                loss_after=loss_after,
                cost_ge=cost_ge,
                accepted=accepted,
                record=step_record,
            )

    # --- step acceptance --------------------------------------------------

    def _accept(
        self, loss_before: float, candidate_loss: float, cg: CGResult
    ) -> tuple[bool, str | None]:
        """수락 여부와 실패 태그를 판정한다.

        negative curvature 자체는 거절 사유가 아니다. truncated CG 는 그 경우에도
        사용 가능한 하강 방향을 돌려주며, 실제로 loss 가 줄면 받아들이는 것이 옳다.

        Stage 3 주의: 여기서는 **단조 감소**를 요구한다. deterministic synthetic
        task 에서는 옳은 정책이지만, minibatch loss 에는 노이즈가 있으므로
        신경망 task 에서 그대로 쓰면 정상 step 도 대량 거절된다. Stage 3 에서는
        고정 평가 배치로 판정하거나 trust ratio 기준으로 완화해야 한다.
        """
        if not math.isfinite(candidate_loss):
            return False, "nan"
        if cg.numerical_failure:
            return False, "cg_breakdown"
        if candidate_loss > abs(loss_before) * self.config.max_loss_increase_ratio:
            return False, "divergence"
        if candidate_loss > loss_before:
            # 개선이 아니면 받지 않는다. 다만 발산은 아니므로 실패로 집계하지
            # 않는다. failure_tag 는 run 단위 실패 원인 분류용이고, 단순
            # 비개선은 step rejection 비율로만 보고한다.
            return False, None
        return True, None

    def _handle_rejection(self, grad: Tensor, failure_tag: str | None) -> float:
        """거절 처리. damping 을 올리고 선택적으로 gradient fallback 을 적용한다.

        ``self._damping_log10`` 은 이미 컨트롤러의 선택이 반영된 값이다. 거절은
        그 위에 추가 상향을 얹는다.
        """
        cfg = self.config
        multiplier = (
            cfg.nan_damping_multiplier if failure_tag == "nan" else cfg.reject_damping_multiplier
        )
        self._damping_log10 = min(
            max(
                self._damping_log10 + math.log10(multiplier),
                cfg.min_damping_log10,
            ),
            cfg.max_damping_log10,
        )

        if cfg.safe_fallback == "none":
            return self.step_objective()

        norm = float(grad.norm())
        if not math.isfinite(norm) or norm == 0.0:
            return self.step_objective()
        scale = min(1.0, cfg.fallback_grad_clip / norm)
        assert self._base_params is not None
        self._flat.add_(grad, alpha=-cfg.fallback_step_size * scale)
        value = self.step_objective()
        if not math.isfinite(value):
            self._flat.copy_from_(self._base_params)
            return self.step_objective()
        return value

    def _failed_record(
        self, step: int, loss_before: float, loss_after: float, tag: str
    ) -> StepRecord:
        return StepRecord(
            run_id=self.run_id,
            seed=self.seed,
            optimizer=self.controller.name,
            step=step,
            train_loss_before=loss_before,
            train_loss_after=loss_after,
            damping=self.damping,
            cost_ge=self.step_cost_ge(0, 0),
            step_accepted=False,
            numerical_failure=True,
            failure_tag=tag,
            task_instance_id=self.task.instance_id,
        )
