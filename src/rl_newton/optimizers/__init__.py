"""Newton-CG optimizer와 컨트롤러들.

optimizer는 **하나뿐**이다 (``newton_cg.NewtonCGOptimizer``). fixed / heuristic /
open_loop / greedy_oracle / RL 의 차이는 optimizer 가 아니라 **누가 action 을
고르는가** 뿐이다. 컨트롤러를 주입받는 단일 루프이므로 비교군 사이에 구현
차이로 인한 교란이 원천적으로 없다 (프로토콜 D4).

```text
fixed              아무것도 안 본다. 항상 같은 action
open_loop          progress 만 본다          <- RL이 "적응 제어"인지 "스케줄"인지 판별
heuristic          trust ratio 를 본다
one_step_efficiency 후보를 전수 시도한다      <- 상한이 아니라 비교군 (프로토콜 D9)
budgeted_mpc       동일 GE 쿼터 안의 계획을 겨룬다  <- 게이트 C 주 컨트롤러
rl                 상태 특징을 본다 (Stage 4)
```

행동 공간
--------
``action_space.NARROW`` 가 프로토콜 원안(36 조합), ``WIDE`` 가 damping 배수를
넓힌 것(72 조합)이다. Stage 1에서 ``kappa=1e6`` 문제가 damping ``1e6`` 을
요구한다는 것이 드러났고, ``x3`` 배수로는 18 step 이 걸린다. 두 프리셋의 헤드룸
차이가 "행동 공간이 병목인가"에 대한 답이 된다.

구현 예정 (Stage 4)
-------------------
``controlled_newton_cg.py``
    RL 정책을 ``Controller`` 로 감싸는 얇은 어댑터. 환경과 평가 스크립트가
    같은 optimizer 루프를 공유한다.
"""

from rl_newton.optimizers.action_space import (
    ABSOLUTE,
    NARROW,
    PRESETS,
    WIDE,
    ActionSpace,
)
from rl_newton.optimizers.controllers import (
    AverageRateEfficiencyPlanner,
    BudgetedMPCController,
    CommittedPlanController,
    FixedController,
    HeuristicController,
    LagrangianPlannerController,
    OneStepEfficiencyController,
    OpenLoopController,
    PlannerChoice,
    PlannerTrack,
    ScheduleSegment,
    ShrinkingQuotaMPCController,
    average_rate_utility,
    bucket_prune,
    efficiency_score,
    lagrangian_utility,
    make_open_loop_controller,
    pareto_frontier,
)
from rl_newton.optimizers.newton_cg import (
    Candidate,
    Controller,
    NewtonCGConfig,
    NewtonCGOptimizer,
    OptimizationTrace,
    StepContext,
    apply_damping_action,
)

__all__ = [
    "ActionSpace",
    "NARROW",
    "WIDE",
    "ABSOLUTE",
    "PRESETS",
    "Candidate",
    "Controller",
    "NewtonCGConfig",
    "NewtonCGOptimizer",
    "OptimizationTrace",
    "StepContext",
    "apply_damping_action",
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
    "PlannerChoice",
    "ScheduleSegment",
    "efficiency_score",
    "average_rate_utility",
    "lagrangian_utility",
    "pareto_frontier",
    "bucket_prune",
    "make_open_loop_controller",
]
