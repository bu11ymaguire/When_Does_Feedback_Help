"""action 을 고르는 컨트롤러들.

모두 같은 ``Controller`` 프로토콜을 구현하고 같은 ``ActionSpace`` 에서 고른다.
차이는 **무엇을 보고 고르는가** 뿐이다.

```text
FixedController               아무것도 안 본다. 항상 같은 action
OpenLoopController            progress 만 본다 (step / total_steps)
HeuristicController           trust ratio 를 본다
OneStepEfficiencyController   현재 step 결과를 본다 (전수 시도, 국소 효율 최대)
AverageRateEfficiencyPlanner  H step 앞을 본다. 누적 평균 효율 최대화 (진단용)
BudgetedMPCController         동일 미래 GE 쿼터 안에서 terminal loss 최소화 (게이트 C)
LagrangianPlannerController   ``Δlog L - β·Σc`` 최대화 (보조 민감도 분석)
```

어느 것도 전역 상한이 아니다
----------------------------
초판은 ``greedy_oracle`` / ``lookahead_oracle`` 이라는 이름을 쓰고 이를 헤드룸의
상한으로 해석했다. **파일럿에서 그 해석이 틀렸음이 확인됐다.** 국소 효율을 매 step
최대화한 컨트롤러가 고정 설정보다 cost-to-target 에서 나빴다 (비율 0.967x).

```text
행동 A:  3 GE 로 loss 10% 감소     -> 순간 효율 높음
행동 B: 20 GE 로 loss 60% 감소     -> 목표까지 총비용은 더 적을 수 있음
```

국소 효율 최대화와 총비용 최소화는 다른 문제다. 그래서 이름과 해석을 정정했다
(프로토콜 D9). ``OneStepEfficiencyController`` 는 상한이 아니라 **비교군의 하나**이고,
planner 들은 유한 쿼터와 beam 폭에 제한된 근사다.

비율 목적함수는 장기 투자를 검출하지 못한다 (프로토콜 D10)
----------------------------------------------------------
``AverageRateEfficiencyPlanner`` 는 시퀀스를
``(log L_start - log L_terminal) / cumulative_cost`` 로 평가했다. 이는 step 별
rate 의 **비용 가중 평균**이므로 mediant 부등식이 적용된다.

```text
min(r1, r2) <= (g1+g2)/(c1+c2) <= max(r1, r2)
```

depth 1 에서 이미 최대 rate ``R*`` 를 골랐으면, depth 2 가 이기려면
``r2 > R*`` 여야 한다. 즉 두 번째 step 이 **지금 당장 가능한 모든 행동보다**
효율적이어야 한다. 수익 체감이 일반적인 환경에서는 드물다.

실측 (quadratic, seed 0, beam 3, 150 GE):

```text
SPD k=1e2   H=1/3/5 전부 logΔ=59.8636, depth 히스토그램 {1: 8}
ill k=1e5   H=1 -> 10.4998 {1:10} / H=3,5 -> 10.5116 {1:9, 2:1}
게이트 C 효과크기 0.025 nat (GO 0.3, pivot 0.05)
```

depth 3 이상은 H=5 에서도 한 번도 채택되지 않았다. 이것은 버그가 아니라
목적함수가 푸는 문제가 달랐던 것이다. 따라서 **이 결과를 "lookahead 가
불필요하다"는 근거로 쓸 수 없다.** 증명되는 것은 다음뿐이다.

> 누적 평균 효율을 최대화하는 목적에서는 짧은 계획이 유리하다.

Track E 의 실제 연구 질문은 고정 예산 문제다.

```text
max  log(L_t / L_{t+m})   s.t.   sum_{i} c_i <= Q
```

여기에 비용으로 나누는 비율은 들어가지 않는다. 그래서 게이트 C 의 주
컨트롤러를 ``BudgetedMPCController`` 로 교체했다. 기존 planner 는 "RL 보상을
ratio 로 설계하면 생기는 함정"의 진단 baseline 으로 보존한다.

이 계층이 프로토콜 게이트를 구성한다
------------------------------------
```text
게이트 A  absolute MPC planner vs best_static  (Track E, 고정 GE 예산)
          -> 적응 제어의 내재적 여지. 작으면 연구를 접거나 음성 결과로 정리.

게이트 B  absolute vs wide vs narrow planner   (로그 해상도를 맞춘 상태에서)
          -> 도달성/행동범위 손실. 크면 행동 공간을 고친다.

게이트 C  미래 GE 쿼터 사다리 (프로토콜 D10 개정)
          C0  OneStepEfficiencyController        (비율 baseline)
          C1  BudgetedMPC  Q = 1 x c_max
          C2  BudgetedMPC  Q = 2 x c_max
          C3  BudgetedMPC  Q = 4 x c_max
          -> 큰 쿼터가 유의미한 개선을 만들고 **동시에** depth >= 2 를 실제로
             채택할 때만 장기 계획에 가치가 있다고 판단한다. 둘 중 하나만
             만족하면 근거가 되지 않는다.

게이트 D  best_static vs planner               (Track T, target 난이도별)
          -> cost-to-target 헤드룸. 게이트 A와 결론이 다를 수 있고
             그 불일치 자체가 결과다.
```

``OpenLoopController`` 는 별도로 결정적이다 (프로토콜 D4). RL 이 fixed 는
이기고 open_loop 는 못 이기면, 학습된 것은 **적응 제어가 아니라 스케줄**이다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from rl_newton.optimizers.action_space import ActionSpace
from rl_newton.optimizers.newton_cg import (
    Candidate,
    NewtonCGOptimizer,
    StepContext,
)
from rl_newton.types import ControllerAction

__all__ = [
    "FixedController",
    "OpenLoopController",
    "HeuristicController",
    "OneStepEfficiencyController",
    "AverageRateEfficiencyPlanner",
    "BudgetedMPCController",
    "CommittedPlanController",
    "ShrinkingQuotaMPCController",
    "LagrangianPlannerController",
    "PlannerTrack",
    "ScheduleSegment",
    "PlannerChoice",
    "PlanCandidate",
    "make_open_loop_controller",
    "efficiency_score",
    "average_rate_utility",
    "lagrangian_utility",
    "pareto_frontier",
    "bucket_prune",
    "LAGRANGIAN_BETA_GRID",
]

PlannerTrack = Literal["fixed_budget", "cost_to_target"]
"""planner 가 최적화하는 트랙 (프로토콜 D9).

``fixed_budget``    Track E. 동일 GE 예산에서 terminal loss 최소화
``cost_to_target``  Track T. 목표 loss 도달까지 총 GE 최소화
"""


def efficiency_score(
    loss_before: float, loss_after: float, cost_ge: float, *, loss_floor: float = 1.0e-30
) -> float:
    """``(log L_before - log L_after) / cost_GE``. 국소 효율.

    **주의: 이것을 매 step 최대화하면 총비용이 나빠질 수 있다.** 파일럿에서
    이 목적으로 매 step 최선을 고른 컨트롤러가 고정 설정보다 cost-to-target 에서
    나빴다 (비율 0.967x). 싸고 작은 행동(``k=3``)을 반복하는 유인이 생기고,
    그 행동이 다음 step 의 상태를 나쁘게 만드는 것을 보지 못한다.

    따라서 이 함수는 ``OneStepEfficiencyController`` 전용이며, 그 컨트롤러는
    **상한이 아니라 비교군의 하나**다 (프로토콜 D9). 게이트 C 사다리에서는
    ``C0`` 에 해당한다.

    **CG 수렴 여부가 아니라 objective 감소를 본다.** 높은 damping 은
    ``(H + lambda I)^{-1} g ~ g / lambda`` 로 CG 를 쉽게 만들지만 실제 감소는
    느려질 수 있다. 실측에서 damping ``1e6`` 은 CG 를 30/30 수렴시키고도 최종
    loss 가 1500배 나빴다.

    개선이 없으면 ``-inf`` 를 반환한다. optimizer 가 어차피 거절할 후보다.
    """
    if not math.isfinite(loss_after) or loss_after >= loss_before:
        return -math.inf
    before = max(loss_before, loss_floor)
    after = max(loss_after, loss_floor)
    return (math.log(before) - math.log(after)) / max(cost_ge, 1e-12)


def average_rate_utility(
    loss_start: float,
    loss_terminal: float,
    cumulative_cost: float,
    *,
    track: PlannerTrack,
    target_loss: float | None = None,
    loss_floor: float = 1.0e-30,
) -> float:
    """시퀀스의 **누적 평균 효율**. ``AverageRateEfficiencyPlanner`` 전용이다.

    per-step ratio 의 합은 아니다 (그건 초판의 결함이었다). 그러나 여전히
    비용으로 나누는 비율이므로 **장기 투자 행동을 검출하지 못한다.**

    ``fixed_budget`` 트랙에서 이 값은 step 별 rate 의 비용 가중 평균이고,
    mediant 부등식에 의해 ``min(r_i) <= U <= max(r_i)`` 다. depth 1 에서 최대
    rate 를 골랐으면 depth 2 는 평균을 희석시킬 뿐이다. 실측에서 depth 3 이상은
    H=5 에서도 한 번도 채택되지 않았다 (프로토콜 D10).

    따라서 게이트 C 의 주 지표로 쓰지 않는다. 주 지표는
    ``BudgetedMPCController`` 이고, 이 함수는 "ratio 보상의 함정" 진단용이다.

    Args:
        loss_start: 시퀀스 시작 시점 loss.
        loss_terminal: 시퀀스 종료 시점 loss.
        cumulative_cost: 시퀀스가 소모한 총 GE.
        track: ``fixed_budget`` 또는 ``cost_to_target``.
        target_loss: ``cost_to_target`` 트랙에서 목표 loss. 없으면 ``fixed_budget``
            처럼 동작한다.
        loss_floor: ``log`` 하한.

    Returns:
        클수록 좋은 효용. 진행이 없으면 ``-inf``.

    두 트랙의 효용
    --------------
    ``fixed_budget`` (Track E): 남은 예산을 가장 효율적으로 쓰는 것이 목적이므로
    **누적** 로그 감소를 **누적** 비용으로 나눈다. horizon 이 길어지면 실제
    목적(총 감소 / 총 예산)에 수렴한다.

    ```text
    U = (log L_start - log L_terminal) / cumulative_cost
    ```

    ``cost_to_target`` (Track T): 목표까지의 **예상 총비용**을 최소화한다.
    시퀀스 안에서 도달했다면 실제 누적 비용이고, 아니면 남은 로그 거리를
    관측된 진행률로 나눈 값을 더한다. 전형적인 cost-to-go 추정이다.

    ```text
    rate = (log L_start - log L_terminal) / cumulative_cost
    U    = -(cumulative_cost + remaining_log_distance / rate)
    ```
    """
    if not math.isfinite(loss_terminal) or loss_terminal >= loss_start:
        return -math.inf
    start = max(loss_start, loss_floor)
    terminal = max(loss_terminal, loss_floor)
    gain = math.log(start) - math.log(terminal)
    cost = max(cumulative_cost, 1e-12)

    if track == "fixed_budget" or target_loss is None:
        return gain / cost

    # cost_to_target: 목표까지의 예상 총비용을 최소화한다.
    target = max(target_loss, loss_floor)
    if terminal <= target:
        return -cost
    rate = gain / cost
    if rate <= 0.0:
        return -math.inf
    remaining = math.log(terminal) - math.log(target)
    return -(cost + remaining / rate)


class FixedController:
    """항상 같은 action 을 고른다.

    프로토콜 D4의 ``best_static`` baseline 은 이 컨트롤러로 행동 공간의 모든
    조합을 각각 돌려 최고를 고른 것이다. "동일 탐색 예산" 원칙의 정직한 구현이다.
    """

    def __init__(self, action: ControllerAction, *, name: str | None = None) -> None:
        self._action = action
        if name is not None:
            self._name = name
        elif action.is_absolute:
            self._name = (
                f"fixed(lam={action.damping_absolute:g},k={action.cg_budget},"
                f"a={action.step_size:g})"
            )
        else:
            self._name = (
                f"fixed(m={action.damping_multiplier:g},k={action.cg_budget},"
                f"a={action.step_size:g})"
            )

    @property
    def name(self) -> str:
        return self._name

    @property
    def action(self) -> ControllerAction:
        return self._action

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction:
        return self._action

    def reset(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"FixedController({self._action})"


@dataclass(frozen=True, slots=True)
class ScheduleSegment:
    """open-loop 스케줄의 한 구간.

    Attributes:
        until: 이 구간이 적용되는 progress 상한 (이하).
        action: 해당 구간에서 쓸 action.
    """

    until: float
    action: ControllerAction


class OpenLoopController:
    """``progress`` 만 보고 고른다. 상태를 전혀 보지 않는다.

    프로토콜 D4의 결정적 baseline 이다. 이 컨트롤러가 얻는 이득은 전부
    "시간에 따른 스케줄"로 설명된다. curvature 나 residual 같은
    optimizer-level 신호는 쓰지 않는다.

    Example:
        >>> from rl_newton.optimizers.action_space import NARROW
        >>> early = NARROW.action_from_flat(0)
        >>> late = NARROW.action_from_flat(len(NARROW) - 1)
        >>> ctrl = OpenLoopController([ScheduleSegment(0.5, early),
        ...                            ScheduleSegment(1.0, late)])
        >>> ctrl.action_at(0.1) is early
        True
        >>> ctrl.action_at(0.9) is late
        True
    """

    _CLOCK = "object_ge_fraction"
    """시계 정의 (프로토콜 D17). ``run_semantics_id`` 에 들어간다."""

    def __init__(self, segments: Sequence[ScheduleSegment], *, name: str = "open_loop") -> None:
        if not segments:
            raise ValueError("segments must not be empty")
        uppers = [s.until for s in segments]
        if uppers != sorted(uppers):
            raise ValueError(f"segments must be sorted by `until`, got {uppers}")
        if uppers[-1] < 1.0:
            raise ValueError(f"last segment must cover progress 1.0, got {uppers[-1]}")
        self._segments = tuple(segments)
        self._name = name
        self._realized_counts: dict[int, int] = {}
        self._realized_ge: dict[int, float] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def clock(self) -> str:
        return self._CLOCK

    @property
    def segments(self) -> tuple[ScheduleSegment, ...]:
        return self._segments

    def action_at(self, progress: float) -> ControllerAction:
        for segment in self._segments:
            if progress <= segment.until:
                return segment.action
        return self._segments[-1].action

    def segment_index_at(self, progress: float) -> int:
        for i, segment in enumerate(self._segments):
            if progress <= segment.until:
                return i
        return len(self._segments) - 1

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction:
        # resource clock (프로토콜 D17). progress 는 소모 GE 비율이다.
        index = self.segment_index_at(context.progress)
        self._realized_counts[index] = self._realized_counts.get(index, 0) + 1
        self._realized_ge[index] = self._realized_ge.get(index, 0.0) + (
            context.previous.cost_ge
            if context.previous is not None and math.isfinite(context.previous.cost_ge)
            else 0.0
        )
        return self._segments[index].action

    @property
    def realized_segment_counts(self) -> dict[int, int]:
        """구간별 실행 step 수 (프로토콜 D17).

        비싼 action 하나가 breakpoint 를 건너뛰면 특정 구간이 실행되지 않을 수
        있다. 오류는 아니지만 **스케줄이 실제로 얼마나 쓰였는지** 보여야 한다.
        """
        return dict(self._realized_counts)

    @property
    def realized_ge_by_segment(self) -> dict[int, float]:
        """구간별 소모 GE. 직전 step 비용을 누적하므로 마지막 step 은 빠진다."""
        return dict(self._realized_ge)

    def reset(self) -> None:
        self._realized_counts = {}
        self._realized_ge = {}

    def __repr__(self) -> str:
        return (
            f"OpenLoopController(n_segments={len(self._segments)}, "
            f"clock=object_ge_fraction)"
        )


class HeuristicController:
    """trust ratio 기반 규칙 (README §4.3).

    ```text
    rho < rho_low             -> damping 증가, step size 감소, budget 증가
    rho_low <= rho < rho_high -> 유지
    rho >= rho_high           -> damping 감소, step size 증가
    ```

    RL 과 **같은 행동 공간**에서 고르므로, "RL 이 단순 적응 규칙보다 나은가"를
    행동 공간 차이 없이 비교할 수 있다.
    """

    def __init__(
        self,
        space: ActionSpace,
        *,
        rho_low: float = 0.25,
        rho_high: float = 0.75,
        initial_flat: int | None = None,
        name: str = "heuristic",
    ) -> None:
        if not 0.0 < rho_low < rho_high:
            raise ValueError(f"require 0 < rho_low < rho_high, got {rho_low}, {rho_high}")
        if space.is_absolute:
            raise ValueError(
                "HeuristicController 는 상대 damping 모드를 전제한다. "
                "absolute 공간은 분석용 오라클 전용이다."
            )
        self._space = space
        self._rho_low = rho_low
        self._rho_high = rho_high
        self._name = name

        nd, nb, ns = space.nvec
        self._initial = (
            space.indices_from_flat(initial_flat)
            if initial_flat is not None
            else (nd // 2, nb // 2, ns - 1)
        )
        self._current = self._initial

    @property
    def name(self) -> str:
        return self._name

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction:
        previous = context.previous
        if previous is not None:
            rho = previous.trust_ratio
            d, b, s = self._current
            nd, nb, ns = self._space.nvec

            if not math.isfinite(rho) or rho < self._rho_low:
                d = min(d + 1, nd - 1)
                s = max(s - 1, 0)
                b = min(b + 1, nb - 1)
            elif rho >= self._rho_high:
                d = max(d - 1, 0)
                s = min(s + 1, ns - 1)
            self._current = (d, b, s)

        return self._space.action_from_indices(self._current)

    def reset(self) -> None:
        self._current = self._initial

    def __repr__(self) -> str:
        return (
            f"HeuristicController(rho_low={self._rho_low}, rho_high={self._rho_high}, "
            f"space={self._space.name})"
        )


@dataclass(slots=True)
class PlannerChoice:
    """컨트롤러가 한 step 에서 내린 선택의 기록.

    정책 분석과 Stage 4 behavior cloning 데이터셋의 원재료다.
    """

    step: int
    chosen_flat: int
    chosen_score: float
    best_loss: float
    worst_loss: float
    n_finite: int
    n_candidates: int
    damping_before: float = float("nan")
    damping_after: float = float("nan")
    n_cg_converged: int = 0
    """CG 가 수렴한 후보 수. loss 감소와 분리해서 본다."""
    chosen_depth: int = 1
    """채택된 계획의 시퀀스 길이. planner 가 실제로 깊은 계획을 쓰는지 분석용.

    항상 1이면 horizon 이나 쿼터를 늘려도 의미가 없다는 직접적 증거다.
    """
    plan_used_ge: float = float("nan")
    """채택된 계획이 쿼터 중 실제로 소모한 GE. ``BudgetedMPCController`` 전용."""
    quota_ge: float = float("nan")
    """이 step 에서 각 후보에게 부여된 미래 GE 쿼터."""
    n_simulations: int = 0
    """이 step 의 계획 수립에 쓴 ``simulate_step`` 호출 수. 탐색 비용 진단용."""
    depth_cap_hit: bool = False
    """``max_depth`` 때문에 확장이 끊겼는지.

    ``True`` 면 쿼터를 다 쓰지 못한 계획이 있으므로 **쿼터 사다리 비교가
    훼손된다.** 게이트 C 보고에 반드시 포함해야 한다.
    """
    reached_target: bool = False
    """채택된 계획이 쿼터 안에서 target 에 도달했는지. Track T 진단용."""


class OneStepEfficiencyController:
    """매 step 전수 시도 후 ``Δlog L / cost_GE`` 가 최대인 action 을 고른다.

    **이것은 상한이 아니다** (프로토콜 D9). 이름이 ``greedy_oracle`` 이었을 때
    상한으로 오해됐으나, 파일럿에서 고정 설정보다 cost-to-target 이 나빴다
    (비율 0.967x). 국소 효율 최대화가 총비용 최소화와 다른 문제이기 때문이다.

    비교군의 하나로서 답하는 질문은 이것이다.

    > 매 step 즉시 효율이 가장 좋은 행동을 고르면 어떻게 되는가?

    이 컨트롤러가 나쁘게 나오는 것 자체가 결과다. RL 보상을 per-step ratio 로
    설계하면 같은 함정에 빠진다는 증거이므로, 프로토콜 D3의 보상 설계 근거가 된다.

    비용
    ----
    ``ActionSpace.iter_solve_groups`` 를 쓰므로 CG solve 는
    ``damping x budget`` 회만 수행된다. NARROW 기준 한 step 당

    ```text
    HvpGraph 생성    1회
    CG solve        12회  (3 damping x 4 budget)
    HVP 합계        3 x (3+5+10+20) = 114회
    forward 평가     36회  (12 방향 x 3 step size)
    ```

    순진한 구현(36 CG solve)보다 6배 이상 싸다. 이 비용은
    ``OptimizationTrace.search_cost_ge`` 로 **본문 비용과 분리해** 회계된다
    (프로토콜 D5).
    """

    def __init__(
        self,
        space: ActionSpace,
        *,
        loss_floor: float = 1.0e-30,
        name: str | None = None,
    ) -> None:
        self._space = space
        self._loss_floor = loss_floor
        self._name = name or f"one_step_efficiency({space.name})"
        self._choices: list[PlannerChoice] = []
        self._trajectory: list[tuple[StepContext, ControllerAction]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def space(self) -> ActionSpace:
        return self._space

    @property
    def choices(self) -> list[PlannerChoice]:
        return self._choices

    @property
    def trajectory(self) -> list[tuple[StepContext, ControllerAction]]:
        """``(state, action)`` 쌍. Stage 4의 behavior cloning 데이터셋이다."""
        return self._trajectory

    def _score(self, loss_before: float, candidate: Candidate) -> float:
        return efficiency_score(
            loss_before,
            candidate.candidate_loss,
            candidate.cost_ge,
            loss_floor=self._loss_floor,
        )

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction:
        candidates = optimizer.evaluate_candidates(self._space.iter_solve_groups())

        best_index = 0
        best_score = -math.inf
        for i, candidate in enumerate(candidates):
            score = self._score(context.loss, candidate)
            if score > best_score:
                best_score = score
                best_index = i

        finite = [c.candidate_loss for c in candidates if c.is_finite]
        chosen = candidates[best_index]

        self._choices.append(
            PlannerChoice(
                step=context.step,
                chosen_flat=best_index,
                chosen_score=best_score,
                best_loss=min(finite) if finite else float("nan"),
                worst_loss=max(finite) if finite else float("nan"),
                n_finite=len(finite),
                n_candidates=len(candidates),
                damping_before=context.damping,
                damping_after=chosen.applied_damping,
                n_cg_converged=sum(1 for c in candidates if c.cg.converged),
            )
        )
        self._trajectory.append((context, chosen.action))
        return chosen.action

    def reset(self) -> None:
        self._choices = []
        self._trajectory = []

    def __repr__(self) -> str:
        return (
            f"OneStepEfficiencyController(space={self._space.name}, n_steps={len(self._choices)})"
        )


@dataclass(slots=True)
class _BeamNode:
    """beam search 의 한 노드.

    ``cumulative_cost`` 와 ``loss`` 를 함께 들고 다닌다. 효용은 시퀀스 끝에서
    이 두 값으로 한 번에 계산한다. **step 별 점수를 누적하지 않는다** —
    그것이 초판의 결함이었다.
    """

    first_action: ControllerAction
    cumulative_cost: float
    loss: float
    snapshot: tuple[object, float]
    depth: int = 1
    """이 노드가 대응하는 시퀀스 길이. 어느 depth 의 계획이 채택됐는지 분석용."""


class AverageRateEfficiencyPlanner:
    """``horizon`` step 앞을 beam search 로 보고 **누적 평균 효율**을 최대화한다.

    **게이트 C 의 주 컨트롤러가 아니다.** 진단 baseline 이다 (프로토콜 D10).

    무엇을 보여주는 결과인가
    ------------------------
    이 planner 는 버그가 아니다. 다만 푸는 문제가 Track E 의 연구 질문과 달랐다.

    ```text
    이 planner:   max (log L_start - log L_terminal) / cumulative_cost
    Track E:      max  log L_t - log L_{t+m}    s.t.  sum c_i <= Q
    ```

    앞쪽은 비용으로 나누므로 step 별 rate 의 가중 평균이 되고, mediant 부등식에
    의해 depth 1 incumbent 가 지나치게 강해진다. 실측에서 그 결과는 다음이었다.

    ```text
    SPD k=1e2   H=1/3/5 전부 동일, depth 히스토그램 {1: 8}
    ill k=1e5   H=1 -> 10.4998 / H=3,5 -> 10.5116  (차이 0.0118 nat)
    depth 3 이상은 H=5 에서도 채택 0회
    search 비용은 H=5 에서 본문의 약 100배
    ```

    이것은 **RL 보상을 ``Δlog L / GE`` 비율로 설계하면 생기는 함정**의 증거다.
    Mediant 부등식 때문에 깊은 계획이 수학적으로 절대 불가능한 것은 아니다.
    두 번째 상태에서 더 효율적인 행동이 열리면 이길 수 있고, 실제로
    ill-conditioned 문제에서 depth 2 가 간헐적으로 채택됐다. 그러나 수익 체감이
    일반적인 환경에서는 **장기 투자 행동을 검출하는 목적함수로 부적합**하다.

    따라서 이 planner 의 음성 결과는 "lookahead 가 불필요하다"의 근거가 될 수
    없다. 게이트 C 는 ``BudgetedMPCController`` 로 판정한다.

    구현
    ----
    model predictive control 방식이다. 매 실제 step 에서 ``horizon`` 만큼
    앞을 보고 첫 action 만 적용한 뒤 다음 step 에서 다시 계획한다.

    전체 조합은 ``|A|^horizon`` 으로 폭발하므로 beam search 로 상위
    ``beam_width`` 개 궤적만 유지한다. 시뮬레이션은 ``optimizer.simulate_step``
    을 쓰고 ``snapshot`` / ``restore`` 로 상태를 되돌린다. 각 시뮬레이션 step 은
    새 HvpGraph 를 만든다 (파라미터가 바뀌므로).

    비용
    ----
    실제 step 당 시뮬레이션 횟수는 대략 ``|A| + (horizon-1) * beam_width * |A|`` 다.
    ``NARROW.with_fixed_step_size()`` 는 12 action 이므로 horizon 3, beam 4 에서
    step 당 약 108 회다. 전부 ``search_cost_ge`` 로 회계되며 본문 비용에
    섞이지 않는다.

    Args:
        space: 행동 공간. 비교군과 **동일**해야 게이트가 성립한다.
        horizon: 앞을 보는 step 수. 1이면 one-step 과 같은 선택을 한다.
        beam_width: 유지할 궤적 수.
        track: ``fixed_budget`` (Track E) 또는 ``cost_to_target`` (Track T).
        target_loss: ``cost_to_target`` 트랙의 목표 loss. 절대값이다.
        loss_floor: ``log`` 하한.
    """

    def __init__(
        self,
        space: ActionSpace,
        *,
        horizon: int = 3,
        beam_width: int = 4,
        track: PlannerTrack = "fixed_budget",
        target_loss: float | None = None,
        loss_floor: float = 1.0e-30,
        name: str | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        if track == "cost_to_target" and target_loss is None:
            raise ValueError("cost_to_target track requires target_loss")
        self._space = space
        self._horizon = horizon
        self._beam_width = beam_width
        self._track: PlannerTrack = track
        self._target_loss = target_loss
        self._loss_floor = loss_floor
        self._name = name or f"avgrate_H{horizon}_{track}({space.name})"
        self._choices: list[PlannerChoice] = []
        self._trajectory: list[tuple[StepContext, ControllerAction]] = []
        self._last_utility = float("nan")

    @property
    def name(self) -> str:
        return self._name

    @property
    def space(self) -> ActionSpace:
        return self._space

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def track(self) -> PlannerTrack:
        return self._track

    @property
    def choices(self) -> list[PlannerChoice]:
        return self._choices

    @property
    def trajectory(self) -> list[tuple[StepContext, ControllerAction]]:
        return self._trajectory

    @property
    def last_utility(self) -> float:
        """직전 ``select`` 에서 채택한 계획의 효용.

        구현 불변조건 검증용이다. **incumbent carry-over 덕분에 같은 상태에서
        H 를 늘리면 이 값은 감소할 수 없다.** 실현 성능의 단조성은 보장되지
        않는다 (MPC 는 매 step 재계획하므로). 따라서 테스트는 이 값에 대해서만
        단조성을 주장한다.
        """
        return self._last_utility

    def _utility(self, loss_start: float, node: _BeamNode) -> float:
        return average_rate_utility(
            loss_start,
            node.loss,
            node.cumulative_cost,
            track=self._track,
            target_loss=self._target_loss,
            loss_floor=self._loss_floor,
        )

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction:
        root = optimizer.snapshot()
        actions = list(self._space.iter_actions())
        loss_start = context.loss

        # --- depth 0 ---
        beam: list[_BeamNode] = []
        for action in actions:
            optimizer.restore(root)
            loss_after, cost_ge, _ = optimizer.simulate_step(action)
            if not math.isfinite(loss_after):
                continue
            beam.append(
                _BeamNode(
                    first_action=action,
                    cumulative_cost=cost_ge,
                    loss=loss_after,
                    snapshot=optimizer.snapshot(),
                )
            )
        optimizer.restore(root)

        if not beam:
            return self._fallback(context, optimizer, root, actions)

        beam.sort(key=lambda n: self._utility(loss_start, n), reverse=True)
        # incumbent: 지금까지 본 모든 depth 중 최선. depth 확장이 실패해도
        # 이것을 잃지 않는다.
        incumbent = beam[0]
        beam = beam[: self._beam_width]

        # --- depth 1..horizon-1 ---
        for depth in range(2, self._horizon + 1):
            expanded: list[_BeamNode] = []
            for node in beam:
                for action in actions:
                    optimizer.restore(node.snapshot)  # type: ignore[arg-type]
                    loss_after, cost_ge, _ = optimizer.simulate_step(action)
                    if not math.isfinite(loss_after):
                        continue
                    expanded.append(
                        _BeamNode(
                            first_action=node.first_action,
                            cumulative_cost=node.cumulative_cost + cost_ge,
                            loss=loss_after,
                            snapshot=optimizer.snapshot(),
                            depth=depth,
                        )
                    )
            optimizer.restore(root)
            if not expanded:
                break
            # 효용은 시퀀스 끝에서 terminal loss 와 누적 비용으로 계산한다.
            expanded.sort(key=lambda n: self._utility(loss_start, n), reverse=True)
            # **incumbent carry-over.** depth 를 늘렸다고 이전 depth 의 최선을
            # 버리면 안 된다. beam search 는 정확한 planner 가 아니므로 깊은
            # 탐색이 좋은 branch 를 중간에 잘라낼 수 있다. 이 처리가 없으면
            # H 를 늘렸을 때 오히려 나빠질 수 있고, 게이트 C 의 해석이 불가능해진다.
            #
            # 효용은 길이로 정규화되어 있으므로(fixed_budget: gain/cost) 서로 다른
            # 길이의 시퀀스를 비교하는 것이 타당하다.
            if self._utility(loss_start, expanded[0]) > self._utility(loss_start, incumbent):
                incumbent = expanded[0]
            beam = expanded[: self._beam_width]

        optimizer.restore(root)
        best = incumbent
        self._last_utility = self._utility(loss_start, best)

        self._choices.append(
            PlannerChoice(
                step=context.step,
                chosen_flat=actions.index(best.first_action),
                chosen_score=self._last_utility,
                best_loss=min(n.loss for n in beam),
                worst_loss=max(n.loss for n in beam),
                n_finite=len(beam),
                n_candidates=len(actions),
                damping_before=context.damping,
                chosen_depth=best.depth,
            )
        )
        self._trajectory.append((context, best.first_action))
        return best.first_action

    def _fallback(
        self,
        context: StepContext,
        optimizer: NewtonCGOptimizer,
        root: tuple[object, float],
        actions: Sequence[ControllerAction],
    ) -> ControllerAction:
        """유한한 결과를 내는 action 이 없을 때. 가장 loss 가 낮은 것을 고른다."""
        best_action = actions[0]
        best_loss = math.inf
        for action in actions:
            optimizer.restore(root)  # type: ignore[arg-type]
            loss_after, _, _ = optimizer.simulate_step(action)
            if math.isfinite(loss_after) and loss_after < best_loss:
                best_loss = loss_after
                best_action = action
        optimizer.restore(root)  # type: ignore[arg-type]
        self._last_utility = -math.inf
        self._choices.append(
            PlannerChoice(
                step=context.step,
                chosen_flat=list(actions).index(best_action),
                chosen_score=-math.inf,
                best_loss=best_loss,
                worst_loss=float("nan"),
                n_finite=0,
                n_candidates=len(actions),
                damping_before=context.damping,
            )
        )
        self._trajectory.append((context, best_action))
        return best_action

    def reset(self) -> None:
        self._choices = []
        self._trajectory = []

    def __repr__(self) -> str:
        return (
            f"AverageRateEfficiencyPlanner(space={self._space.name}, "
            f"horizon={self._horizon}, beam={self._beam_width}, track={self._track})"
        )


LAGRANGIAN_BETA_GRID: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
"""보조 민감도 분석용 β 격자. **사전 고정이며 결과를 보고 바꾸지 않는다.**

Lagrangian planner 는 주 결과가 아니다 (프로토콜 D10). 특정 β 하나를 골라
게이트 C 결론으로 쓰면 사후 선택이 되므로, 이 격자 **전체**를 보고한다.
"""


def lagrangian_utility(
    loss_start: float,
    loss_terminal: float,
    cumulative_cost: float,
    *,
    beta: float,
    loss_floor: float = 1.0e-30,
) -> float:
    """``(log L_start - log L_terminal) - beta * cumulative_cost``.

    비용으로 나누지 않으므로 ``average_rate_utility`` 의 평균 희석이 없다.
    깊은 계획이 더 많은 진행을 만들면 이길 수 있다. 대신 β 에 따라 결론이
    바뀌므로 **보조 분석 전용**이다 (프로토콜 D10).

    ``beta=0`` 은 비용을 무시하고 terminal loss 만 본다. 쿼터 제약이 없으면
    항상 가장 긴 계획을 고르므로, 이 함수는 쿼터 안에서만 쓴다.
    """
    if not math.isfinite(loss_terminal):
        return -math.inf
    start = max(loss_start, loss_floor)
    terminal = max(loss_terminal, loss_floor)
    return (math.log(start) - math.log(terminal)) - beta * cumulative_cost


class _CostLoss(Protocol):
    """``(used_ge, terminal_loss)`` 를 가진 후보. 둘 다 작을수록 좋다."""

    used_ge: float
    terminal_loss: float


def pareto_frontier[N: _CostLoss](candidates: Sequence[N]) -> list[N]:
    """``(used_ge, terminal_loss)`` 의 비지배 후보만 남긴다.

    **비율 하나로 정렬하면 안 되는 이유** (프로토콜 D10): ``Δlog L / cost`` 로
    가지치기하면 mediant 문제가 beam pruning 안에서 그대로 재발한다. 비싼
    장기 계획이 싼 단기 계획과 스칼라 하나로 섞여 조기에 탈락한다.

    후보 A 가 B 보다 GE 를 같거나 적게 쓰고 terminal loss 도 같거나 낮으면
    B 를 제거한다.

    Returns:
        비용 오름차순으로 정렬된 비지배 후보. **terminal loss 최소 후보는 항상
        포함된다** (비용이 더 적으면서 loss 가 더 낮은 후보는 존재할 수 없으므로).
        이것이 incumbent carry-over 를 대체한다. depth 1 최선이 더 나은 계획에
        의해서만 밀려난다.
    """
    order = sorted(candidates, key=lambda n: (n.used_ge, n.terminal_loss))
    out: list[N] = []
    best_loss = math.inf
    for node in order:
        if node.terminal_loss < best_loss:
            out.append(node)
            best_loss = node.terminal_loss
    return out


def bucket_prune[N: _CostLoss](
    candidates: Sequence[N], *, beam_width: int, bucket_ge: float
) -> list[N]:
    """GE 비용 구간별로 상위 ``beam_width`` 개만 남긴다.

    Pareto frontier 는 크기 상한이 없어서 탐색량을 제한하지 못한다. 그래서
    비용을 ``bucket_ge`` 폭의 구간으로 나누고 구간마다 terminal loss 가 좋은
    후보를 ``beam_width`` 개 남긴다. 이렇게 하면 "싼 단기 계획"과 "비싼 장기
    계획"이 각자의 구간에서 살아남으므로, 스칼라 하나로 조기에 섞이지 않는다.

    Args:
        candidates: 후보들.
        beam_width: 구간당 유지 수.
        bucket_ge: 구간 폭 (GE). 0 이하면 전체를 한 구간으로 본다.
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")
    buckets: dict[int, list[N]] = {}
    for node in candidates:
        key = int(node.used_ge // bucket_ge) if bucket_ge > 0.0 else 0
        buckets.setdefault(key, []).append(node)
    out: list[N] = []
    for key in sorted(buckets):
        ranked = sorted(buckets[key], key=lambda n: (n.terminal_loss, n.used_ge))
        out.extend(ranked[:beam_width])
    return out


@dataclass(slots=True)
class PlanCandidate:
    """계획 후보의 공개 표현. 스냅샷을 들지 않으므로 기록/테스트에 쓴다."""

    used_ge: float
    terminal_loss: float
    depth: int = 1
    reached_target: bool = False


@dataclass(slots=True)
class _PlanNode:
    """쿼터 기반 탐색의 한 노드. ``used_ge`` 와 ``terminal_loss`` 로 비교된다.

    **시퀀스 전체를 들고 다닌다.** 첫 action 만 들면 committed 실행과
    shrinking 재계획에서 남은 suffix 를 쓸 수 없다 (프로토콜 D12).
    """

    actions: tuple[ControllerAction, ...]
    used_ge: float
    terminal_loss: float
    snapshot: tuple[object, float]
    reached_target: bool = False

    @property
    def first_action(self) -> ControllerAction:
        return self.actions[0]

    @property
    def depth(self) -> int:
        return len(self.actions)

    @property
    def suffix(self) -> tuple[ControllerAction, ...]:
        """첫 action 을 실행한 뒤 남는 계획."""
        return self.actions[1:]


class BudgetedMPCController:
    """각 후보에게 **동일한 미래 GE 쿼터** ``Q`` 를 주고 terminal loss 를 겨룬다.

    프로토콜 게이트 C 의 주 컨트롤러다 (D10 개정). Track E 의 연구 질문과 형태가
    일치한다.

    ```text
    max  log L_t - log L_{t+m}     s.t.  sum_{i=t}^{t+m-1} c_i <= Q
    ```

    **비용으로 나누는 비율이 들어가지 않는다.** 그래서
    ``AverageRateEfficiencyPlanner`` 의 평균 희석 문제가 없다. 비싼 한 방과 싼
    여러 방이 같은 예산 안에서 공정하게 비교된다. 이것이 게이트 C 가 원래
    물으려던 것이다.

    > 같은 미래 예산을 쓴다면, 지금 손해를 보고 나중에 이득을 보는 것이
    > 이득인가?

    쿼터 사다리
    -----------
    ``c_max`` 를 단일 action 최대 비용이라 하면

    ```text
    C1  Q = 1 x c_max    비싼 action 1회 vs 싼 action 여러 회
    C2  Q = 2 x c_max
    C3  Q = 4 x c_max
    ```

    쿼터를 늘려도 실제 episode 의 terminal loss 가 개선되지 않거나, 개선돼도
    ``chosen_depth`` 가 계속 1 이면 장기 계획의 실질 가치가 없다. **두 조건을
    모두** 만족해야 temporal planning 근거가 된다.

    확장 규칙
    ---------
    - depth 1 후보는 **쿼터와 무관하게 항상 생성한다.** planner 는 반드시 행동
      해야 하고, ``Q < c_max`` 여도 비싼 action 을 후보에서 배제하면 행동 공간이
      쿼터에 따라 달라져 게이트 B 와 혼동된다.
    - depth >= 2 확장은 ``used_ge + c <= Q`` 일 때만 유효하다. 초과하면 그 확장을
      버리고 부모를 leaf 로 둔다. 실제 관측 비용을 쓰므로 CG 조기 수렴이
      반영된다.
    - 더 확장할 수 있는 노드가 없거나 ``max_depth`` 에 닿으면 멈춘다.

    선택 규칙
    ---------
    Pareto/bucket 가지치기는 **탐색 중**에만 쓴다. 최종 선택은 다르다.

    ```text
    fixed_budget (Track E)
        terminal loss 최소 -> 동률이면 GE 적은 것 -> 그래도 동률이면 짧은 것

    cost_to_target (Track T)  lexicographic
        1. target 도달 후보가 있으면 그 중 누적 GE 최소
        2. 아무도 못 도달하면 terminal loss 최소
        3. 동률이면 GE 적은 것, 그다음 짧은 것
    ```

    Track T 에 임의의 실패 벌점이나 비율을 넣지 않는다. 도달 여부가 우선이고
    비용이 그다음이라는 것을 순서로 표현한다.

    Args:
        space: 행동 공간. 비교군과 동일해야 게이트가 성립한다.
        quota_multiplier: ``Q = quota_multiplier * c_max``. ``quota_ge`` 와
            정확히 하나만 준다.
        quota_ge: 절대 GE 쿼터.
        beam_width: 비용 구간당 유지할 후보 수.
        bucket_ge: 비용 구간 폭. ``None`` 이면 단일 action 최소 비용 ``c_min``.
        track: ``fixed_budget`` (Track E) 또는 ``cost_to_target`` (Track T).
        target_loss: Track T 의 절대 목표 loss.
        max_depth: 계획 길이 상한. 계산량 안전장치다. 이것에 걸리면
            ``depth_cap_hit`` 가 기록되고 쿼터 사다리 비교가 훼손되므로
            보고에 포함해야 한다.
        max_open: 한 라운드에서 확장할 노드 수 상한. ``None`` 이면
            ``4 * beam_width``.
    """

    def __init__(
        self,
        space: ActionSpace,
        *,
        quota_multiplier: float | None = None,
        quota_ge: float | None = None,
        beam_width: int = 4,
        bucket_ge: float | None = None,
        track: PlannerTrack = "fixed_budget",
        target_loss: float | None = None,
        max_depth: int = 6,
        max_open: int | None = None,
        loss_floor: float = 1.0e-30,
        name: str | None = None,
    ) -> None:
        if (quota_multiplier is None) == (quota_ge is None):
            raise ValueError("quota_multiplier 와 quota_ge 중 정확히 하나를 지정해야 한다")
        if quota_multiplier is not None and quota_multiplier <= 0.0:
            raise ValueError(f"quota_multiplier must be > 0, got {quota_multiplier}")
        if quota_ge is not None and quota_ge <= 0.0:
            raise ValueError(f"quota_ge must be > 0, got {quota_ge}")
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if track == "cost_to_target" and target_loss is None:
            raise ValueError("cost_to_target track requires target_loss")
        self._space = space
        self._quota_multiplier = quota_multiplier
        self._quota_ge = quota_ge
        self._beam_width = beam_width
        self._bucket_ge = bucket_ge
        self._track: PlannerTrack = track
        self._target_loss = target_loss
        self._max_depth = max_depth
        self._max_open = max_open if max_open is not None else 4 * beam_width
        self._loss_floor = loss_floor
        label = (
            f"Q{quota_multiplier:g}xcmax" if quota_multiplier is not None else f"Q{quota_ge:g}ge"
        )
        self._name = name or f"budgeted_{label}_{track}({space.name})"
        self._choices: list[PlannerChoice] = []
        self._trajectory: list[tuple[StepContext, ControllerAction]] = []
        self._resolved_quota = float("nan")
        self._resolved_bucket = float("nan")
        self._last_plan: _PlanNode | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def space(self) -> ActionSpace:
        return self._space

    @property
    def track(self) -> PlannerTrack:
        return self._track

    @property
    def choices(self) -> list[PlannerChoice]:
        return self._choices

    @property
    def trajectory(self) -> list[tuple[StepContext, ControllerAction]]:
        return self._trajectory

    @property
    def quota_ge(self) -> float:
        """해석된 쿼터. 첫 ``select`` 이전에는 ``nan``."""
        return self._resolved_quota

    def _action_cost(self, optimizer: NewtonCGOptimizer, action: ControllerAction) -> float:
        """단일 action 의 예측 비용. 쿼터 정의에만 쓴다 (결정적이어야 하므로)."""
        return optimizer.step_cost_ge(action.cg_budget, 1, with_graph=True)

    def _resolve(self, optimizer: NewtonCGOptimizer, actions: Sequence[ControllerAction]) -> None:
        if math.isfinite(self._resolved_quota):
            return
        costs = [self._action_cost(optimizer, a) for a in actions]
        if self._quota_ge is not None:
            self._resolved_quota = self._quota_ge
        else:
            assert self._quota_multiplier is not None
            self._resolved_quota = self._quota_multiplier * max(costs)
        self._resolved_bucket = self._bucket_ge if self._bucket_ge is not None else min(costs)

    def _reached(self, loss: float) -> bool:
        return self._target_loss is not None and loss <= self._target_loss

    def _prune(self, nodes: Sequence[_PlanNode]) -> list[_PlanNode]:
        return bucket_prune(
            pareto_frontier(nodes), beam_width=self._beam_width, bucket_ge=self._resolved_bucket
        )

    def _best(self, nodes: Sequence[_PlanNode]) -> _PlanNode:
        if self._track == "cost_to_target":
            reached = [n for n in nodes if n.reached_target]
            if reached:
                # 1. 도달한 것 중 누적 GE 최소
                return min(reached, key=lambda n: (n.used_ge, n.depth, n.terminal_loss))
        # 2. 아무도 못 도달했거나 Track E: terminal loss 최소
        return min(nodes, key=lambda n: (n.terminal_loss, n.used_ge, n.depth))

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction:
        plan = self.plan(context, optimizer)
        if plan is None:
            return self._fallback_action(context, optimizer)
        return plan.first_action

    def plan(
        self,
        context: StepContext,
        optimizer: NewtonCGOptimizer,
        *,
        quota: float | None = None,
        seed_plan: Sequence[ControllerAction] = (),
    ) -> _PlanNode | None:
        """쿼터 ``quota`` 안에서 최선의 action 시퀀스를 찾는다.

        ``select`` 는 첫 action 만 쓰지만, committed / shrinking 실행 방식은
        시퀀스 전체가 필요하다 (프로토콜 D12). 그래서 탐색과 실행을 분리했다.

        Args:
            context: 현재 step 정보.
            optimizer: 시뮬레이션에 쓸 optimizer. 호출 후 상태는 복원된다.
            quota: 쓸 쿼터. ``None`` 이면 해석된 전체 쿼터.
            seed_plan: **반드시 후보에 포함할 계획.** shrinking 재계획에서
                이전 계획의 남은 suffix 를 넣는다. 결정적 환경에서는 재계획이
                더 나은 것을 못 찾아도 이전 suffix 를 유지할 수 있어야 한다.
                이것이 없으면 beam 근사 때문에 재계획 자체가 성능을 떨어뜨린다.

        Returns:
            최선 노드. 유한한 결과를 내는 action 이 하나도 없으면 ``None``.
        """
        root = optimizer.snapshot()
        actions = list(self._space.iter_actions())
        self._resolve(optimizer, actions)
        budget = self._resolved_quota if quota is None else quota
        # 부동소수 비교 여유. 쿼터 경계에서 확장이 임의로 갈리지 않게 한다.
        slack = abs(budget) * 1.0e-9
        n_sims = 0

        # --- depth 1: 쿼터와 무관하게 전부 생성 ---
        # 쿼터가 작아도 비싼 action 을 배제하지 않는다. 그러면 행동 공간이
        # 쿼터에 따라 달라져 게이트 B 와 혼동된다. planner 는 반드시 행동해야 한다.
        frontier: list[_PlanNode] = []
        for action in actions:
            optimizer.restore(root)
            loss_after, cost_ge, _ = optimizer.simulate_step(action)
            n_sims += 1
            if not math.isfinite(loss_after):
                continue
            frontier.append(
                _PlanNode(
                    actions=(action,),
                    used_ge=cost_ge,
                    terminal_loss=loss_after,
                    snapshot=optimizer.snapshot(),
                    reached_target=self._reached(loss_after),
                )
            )
        optimizer.restore(root)

        if not frontier:
            return None

        # --- seed_plan 을 보장 후보로 넣는다 (incumbent) ---
        seed_nodes, seed_sims = self._rollout(optimizer, root, seed_plan, budget, slack)
        n_sims += seed_sims
        frontier.extend(seed_nodes)

        open_nodes = self._open(frontier, budget, slack)

        # --- depth 2..: 쿼터를 넘지 않는 확장만 ---
        depth = 1
        depth_cap_hit = False
        while open_nodes:
            if depth >= self._max_depth:
                depth_cap_hit = True
                break
            depth += 1
            expanded: list[_PlanNode] = []
            for node in open_nodes:
                for action in actions:
                    optimizer.restore(node.snapshot)  # type: ignore[arg-type]
                    loss_after, cost_ge, _ = optimizer.simulate_step(action)
                    n_sims += 1
                    if not math.isfinite(loss_after):
                        continue
                    total = node.used_ge + cost_ge
                    if total > budget + slack:
                        # 쿼터 초과 계획은 이 사다리 단계에서 유효하지 않다.
                        continue
                    expanded.append(
                        _PlanNode(
                            actions=(*node.actions, action),
                            used_ge=total,
                            terminal_loss=loss_after,
                            snapshot=optimizer.snapshot(),
                            reached_target=self._reached(loss_after),
                        )
                    )
            optimizer.restore(root)
            if not expanded:
                break
            # 모든 depth 를 한 frontier 에 모아 가지치기한다. Pareto 는 terminal
            # loss 최소 후보를 지우지 않으므로 depth 1 최선과 seed 가 보존된다.
            frontier = self._prune(frontier + expanded)
            open_nodes = self._open(expanded, budget, slack)

        optimizer.restore(root)
        best = self._best(frontier)

        losses = [n.terminal_loss for n in frontier]
        self._choices.append(
            PlannerChoice(
                step=context.step,
                chosen_flat=actions.index(best.first_action),
                chosen_score=-best.terminal_loss,
                best_loss=min(losses),
                worst_loss=max(losses),
                n_finite=len(frontier),
                n_candidates=len(actions),
                damping_before=context.damping,
                chosen_depth=best.depth,
                plan_used_ge=best.used_ge,
                quota_ge=budget,
                n_simulations=n_sims,
                depth_cap_hit=depth_cap_hit,
                reached_target=best.reached_target,
            )
        )
        self._trajectory.append((context, best.first_action))
        self._last_plan = best
        return best

    def _open(self, nodes: Sequence[_PlanNode], budget: float, slack: float) -> list[_PlanNode]:
        """더 확장할 노드. Track T 에서 이미 도달한 계획은 확장하지 않는다."""
        return [
            n
            for n in self._prune(nodes)
            if n.used_ge < budget - slack
            and not (self._track == "cost_to_target" and n.reached_target)
        ][: self._max_open]

    def _rollout(
        self,
        optimizer: NewtonCGOptimizer,
        root: tuple[object, float],
        plan: Sequence[ControllerAction],
        budget: float,
        slack: float,
    ) -> tuple[list[_PlanNode], int]:
        """``plan`` 을 root 에서 그대로 굴려 prefix 노드들을 만든다.

        쿼터를 넘는 지점에서 멈춘다. 결정적 환경이므로 이전 계획의 suffix 를
        여기에 넣으면 정확히 같은 궤적이 재현된다.
        """
        nodes: list[_PlanNode] = []
        if not plan:
            return nodes, 0
        optimizer.restore(root)
        used = 0.0
        seq: list[ControllerAction] = []
        n_sims = 0
        for action in plan:
            loss_after, cost_ge, _ = optimizer.simulate_step(action)
            n_sims += 1
            if not math.isfinite(loss_after) or used + cost_ge > budget + slack:
                break
            used += cost_ge
            seq.append(action)
            nodes.append(
                _PlanNode(
                    actions=tuple(seq),
                    used_ge=used,
                    terminal_loss=loss_after,
                    snapshot=optimizer.snapshot(),
                    reached_target=self._reached(loss_after),
                )
            )
        optimizer.restore(root)
        return nodes, n_sims

    def _fallback_action(
        self, context: StepContext, optimizer: NewtonCGOptimizer
    ) -> ControllerAction:
        return self._fallback(
            context, optimizer, optimizer.snapshot(), list(self._space.iter_actions())
        )

    def _fallback(
        self,
        context: StepContext,
        optimizer: NewtonCGOptimizer,
        root: tuple[object, float],
        actions: Sequence[ControllerAction],
    ) -> ControllerAction:
        """유한한 결과를 내는 action 이 없을 때. loss 가 가장 낮은 것을 고른다."""
        best_action = actions[0]
        best_loss = math.inf
        for action in actions:
            optimizer.restore(root)
            loss_after, _, _ = optimizer.simulate_step(action)
            if math.isfinite(loss_after) and loss_after < best_loss:
                best_loss = loss_after
                best_action = action
        optimizer.restore(root)
        self._choices.append(
            PlannerChoice(
                step=context.step,
                chosen_flat=list(actions).index(best_action),
                chosen_score=-math.inf,
                best_loss=best_loss,
                worst_loss=float("nan"),
                n_finite=0,
                n_candidates=len(actions),
                damping_before=context.damping,
                quota_ge=self._resolved_quota,
            )
        )
        self._trajectory.append((context, best_action))
        return best_action

    @property
    def last_plan(self) -> _PlanNode | None:
        """직전 ``plan`` 이 고른 노드. 예측값 검증용이다."""
        return self._last_plan

    def reset(self) -> None:
        self._choices = []
        self._trajectory = []
        self._last_plan = None

    def __repr__(self) -> str:
        return (
            f"BudgetedMPCController(space={self._space.name}, quota={self._resolved_quota:g}, "
            f"beam={self._beam_width}, track={self._track})"
        )


class CommittedPlanController(BudgetedMPCController):
    """계획을 찾으면 **끝까지 그대로 실행**한다. 재계획하지 않는다.

    프로토콜 D12 의 세 실행 방식 중 하나다. ``BudgetedMPCController`` 와 탐색은
    완전히 동일하고 **실행 방식만** 다르므로, 차이가 탐색 품질 차이와 섞이지
    않는다.

    ```text
    계획: [a1, a2, a3]
    실행: a1 -> a2 -> a3   (중간에 다시 계획하지 않음)
    ```

    시퀀스를 다 쓰면 그 지점에서 새 쿼터로 다시 계획한다 (committed window 반복).
    episode 예산이 쿼터보다 크면 여러 window 가 생긴다.

    **가장 먼저 확인할 불변조건** (프로토콜 D12):

    ```text
    J_predicted_plan  ~=  J_committed_execution
    ```

    synthetic task 는 결정적이므로 planner 가 예측한 terminal loss 와 실제
    실행 결과가 거의 같아야 한다. 다르면 상태 복원이나 실행 회계에 버그가
    있는 것이고, 그러면 이후 비교는 의미가 없다.
    """

    def __init__(self, space: ActionSpace, **kwargs: object) -> None:
        name = kwargs.pop("name", None)
        super().__init__(space, **kwargs)  # type: ignore[arg-type]
        self._name = name or self._name.replace("budgeted_", "committed_")  # type: ignore[assignment]
        self._pending: list[ControllerAction] = []
        self._predictions: list[tuple[int, float, float]] = []
        """``(계획 수립 step, 예측 terminal loss, 계획 비용)``. 검증용이다."""

    @property
    def predictions(self) -> list[tuple[int, float, float]]:
        return self._predictions

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction:
        if not self._pending:
            plan = self.plan(context, optimizer)
            if plan is None:
                return self._fallback_action(context, optimizer)
            self._pending = list(plan.actions)
            self._predictions.append((context.step, plan.terminal_loss, plan.used_ge))
        return self._pending.pop(0)

    def reset(self) -> None:
        super().reset()
        self._pending = []
        self._predictions = []

    def __repr__(self) -> str:
        return (
            f"CommittedPlanController(space={self._space.name}, "
            f"quota={self._resolved_quota:g}, beam={self._beam_width})"
        )


class ShrinkingQuotaMPCController(BudgetedMPCController):
    """쿼터에서 **쓴 비용을 차감**하며 재계획한다. horizon 을 새로 연장하지 않는다.

    프로토콜 D12 의 세 실행 방식 중 하나다. fresh-quota 방식(기본
    ``BudgetedMPCController``)은 매 step 마다 미래 예산 ``Q`` 를 새로 지급하므로,
    "나중에 이득을 얻을 준비 행동"을 계속 고르면서 실제 payoff 를 무한히 뒤로
    미룰 수 있다. 시간 불일치다.

    ```text
    fresh:      step1 Q=90, step2 Q=90, step3 Q=90 ...
    shrinking:  step1 Q=90, 20 사용 -> step2 Q=70, 10 사용 -> step3 Q=60 ...
    ```

    쿼터가 소진되면 새 window 를 열고 다시 ``Q`` 를 지급한다.

    **이전 계획의 남은 suffix 를 반드시 후보로 보존한다.** 결정적 환경에서
    재계획이 더 나은 것을 못 찾아도 이전 suffix 는 유지할 수 있어야 한다.
    그렇지 않으면 beam 근사 때문에 재계획 자체가 성능을 떨어뜨리고, 그것이
    "피드백이 해롭다"로 오해된다.
    """

    def __init__(self, space: ActionSpace, **kwargs: object) -> None:
        name = kwargs.pop("name", None)
        super().__init__(space, **kwargs)  # type: ignore[arg-type]
        self._name = name or self._name.replace("budgeted_", "shrinking_")  # type: ignore[assignment]
        self._remaining = float("nan")
        self._suffix: tuple[ControllerAction, ...] = ()
        self._last_loss = float("nan")
        self._windows = 0
        self._n_replans = 0
        self._n_suffix_retained = 0

    @property
    def windows(self) -> int:
        """열린 window 수. 쿼터 소진 횟수다."""
        return self._windows

    @property
    def suffix_retention_rate(self) -> float:
        """재계획 결과가 이전 계획의 suffix 와 **정확히 같았던** 비율.

        ```text
        replanned_actions == previous_plan[1:]
        ```

        1.0 이면 재계획이 계획을 한 번도 바꾸지 않았다는 뜻이고, committed 실행과
        같은 경로를 간다. ``chosen_depth`` 히스토그램만으로는 깊이만 같고 내용이
        다를 수 있으므로 **행동 내용까지 비교해** 계측한다 (프로토콜 D15).

        ``nan`` 이면 재계획 기회가 없었다 (window 하나에 step 하나).
        """
        if self._n_replans == 0:
            return float("nan")
        return self._n_suffix_retained / self._n_replans

    @property
    def n_replans(self) -> int:
        """이전 계획이 남아 있는 상태에서 재계획한 횟수. 분모다."""
        return self._n_replans

    def select(self, context: StepContext, optimizer: NewtonCGOptimizer) -> ControllerAction:
        actions = list(self._space.iter_actions())
        self._resolve(optimizer, actions)
        cheapest = self._resolved_bucket

        # **직전 step 의 실제 비용**을 차감한다. 예측 비용을 쓰면 CG 가 조기
        # 수렴한 만큼 쿼터가 과도하게 줄어들어 window 가 일찍 닫히고, 그것이
        # "shrinking 이 나쁘다"로 오해된다.
        if context.previous is not None and math.isfinite(self._remaining):
            spent = context.previous.cost_ge
            if math.isfinite(spent):
                self._remaining -= spent

        # 첫 step 이거나 쿼터가 가장 싼 action 보다 적게 남으면 새 window 를 연다.
        if not math.isfinite(self._remaining) or self._remaining < cheapest:
            self._remaining = self._resolved_quota
            self._suffix = ()
            self._windows += 1

        previous = self._suffix
        plan = self.plan(context, optimizer, quota=self._remaining, seed_plan=previous)
        if plan is None:
            return self._fallback_action(context, optimizer)

        # 재계획이 이전 계획을 실제로 바꿨는지 계측한다 (프로토콜 D15).
        # 깊이가 아니라 **행동 내용**을 비교한다.
        if previous:
            self._n_replans += 1
            if plan.actions == previous:
                self._n_suffix_retained += 1

        # 남은 suffix 를 다음 재계획의 보장 후보로 넘긴다. **탐색에서 살아남는
        # 것은 보장되지만 채택이 보장되는 것은 아니다.** 목적함수가 남은 쿼터
        # 안에서 더 낮은 terminal loss 를 찾으면 계획을 버린다. 그 이탈이
        # 국소적으로는 개선이어도 episode 전체로는 손해일 수 있다.
        self._suffix = plan.suffix
        return plan.first_action

    def reset(self) -> None:
        super().reset()
        self._remaining = float("nan")
        self._suffix = ()
        self._windows = 0
        self._n_replans = 0
        self._n_suffix_retained = 0

    def __repr__(self) -> str:
        return (
            f"ShrinkingQuotaMPCController(space={self._space.name}, "
            f"quota={self._resolved_quota:g}, beam={self._beam_width})"
        )


class LagrangianPlannerController(BudgetedMPCController):
    """``(Δlog L) - β·Σc`` 를 최대화한다. **보조 민감도 분석 전용**이다.

    쿼터 기반 탐색과 Pareto 가지치기는 ``BudgetedMPCController`` 와 동일하고,
    **최종 선택 규칙만** 다르다. 그래서 두 목적함수의 차이가 탐색 품질 차이와
    섞이지 않는다.

    β 하나를 골라 게이트 C 주 결과로 쓰지 않는다. ``LAGRANGIAN_BETA_GRID``
    전체를 보고한다 (프로토콜 D10). 이유는 두 가지다.

    - β 에 따라 결론이 뒤집힐 수 있다.
    - discrete action 에서 Lagrangian 완화는 고정 예산 문제와 정확히 같지 않다
      (duality gap 이 0 이라는 보장이 없다).
    """

    def __init__(self, space: ActionSpace, *, beta: float, **kwargs: object) -> None:
        if beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        name = kwargs.pop("name", None)
        super().__init__(space, **kwargs)  # type: ignore[arg-type]
        self._beta = float(beta)
        self._name = name or f"lagrangian_b{beta:g}_{self._track}({space.name})"  # type: ignore[assignment]

    @property
    def beta(self) -> float:
        return self._beta

    def _best(self, nodes: Sequence[_PlanNode]) -> _PlanNode:
        if self._track == "cost_to_target":
            reached = [n for n in nodes if n.reached_target]
            if reached:
                return min(reached, key=lambda n: (n.used_ge, n.depth, n.terminal_loss))
        # loss_start 는 상수이므로 후보 간 비교에서 상쇄된다. 1.0 을 넣어도 순서가
        # 같다. 명시적으로 남겨 목적함수를 읽을 수 있게 한다.
        return max(
            nodes,
            key=lambda n: (
                lagrangian_utility(
                    1.0, n.terminal_loss, n.used_ge, beta=self._beta, loss_floor=self._loss_floor
                ),
                -n.used_ge,
                -n.depth,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"LagrangianPlannerController(space={self._space.name}, beta={self._beta:g}, "
            f"quota={self._resolved_quota:g}, track={self._track})"
        )


def make_open_loop_controller(
    space: ActionSpace, flats: Sequence[int], breakpoints: Sequence[float]
) -> OpenLoopController:
    """스케줄 파라미터로 ``OpenLoopController`` 를 만든다.

    Args:
        space: 행동 공간.
        flats: 각 구간에서 쓸 action 의 flat 인덱스.
        breakpoints: 구간 상한. ``len(flats)`` 와 같아야 하고 오름차순이며
            마지막은 1.0 이상이어야 한다.
    """
    if len(flats) != len(breakpoints):
        raise ValueError(
            f"flats and breakpoints must have equal length, got {len(flats)} and {len(breakpoints)}"
        )
    segments = [
        ScheduleSegment(until=float(b), action=space.action_from_flat(int(f)))
        for f, b in zip(flats, breakpoints, strict=True)
    ]
    label = "-".join(str(int(f)) for f in flats)
    return OpenLoopController(segments, name=f"open_loop[{label}]")
