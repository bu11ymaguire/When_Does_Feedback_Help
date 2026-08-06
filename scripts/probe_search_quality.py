"""진단: beam search 가 도달 가능한 좋은 계획을 버리고 있는가.

쿼터를 키울수록 실제 episode 성능이 **단조롭게 나빠졌다**. 쿼터가 episode
예산 전체를 덮어 truncation 이 없는 조건에서도 그랬다. 두 가지 해석이 가능하다.

```text
(a) planning 자체가 가치 없다
(b) beam search 가 너무 약해서 깊은 계획을 제대로 못 찾는다
```

구분하는 방법
-------------
one-step efficiency 컨트롤러의 궤적은 **탐색 트리 안에 존재하는 실현 가능한
계획**이다. 같은 행동 공간에서 같은 dynamics 로 생성되기 때문이다. 따라서
같은 비용 예산 안에서 beam search 가 찾은 최선이 그 궤적보다 나쁘면,
가지치기가 도달 가능한 계획을 버렸다는 뜻이다. 목적함수 문제가 아니라
탐색 문제다.

**동일 비용으로 비교해야 한다 (초판 진단의 버그).**
episode 는 ``spent >= budget`` 에서 종료하므로 마지막 step 이 예산을 초과한다.
초판은 ``cost_budget_ge=Q`` 로 돌린 one-step 의 최종 loss 를 그대로 썼는데,
실제 소모 비용이 Q=30 에서 49.9 GE, Q=60 에서 72.2 GE 였다. planner 의 쿼터는
초과를 엄격히 금지하므로 참조 쪽이 1.2~1.66배 예산을 더 쓴 비교였다.
그 상태의 "탐색 손실" 판정은 무효다.

지금은 one-step 궤적을 **누적비용이 Q 이하인 마지막 prefix** 에서 잘라 쓴다.
그 prefix 는 쿼터 제약을 만족하는 실현 가능한 계획이다.

부수 확인
---------
``simulate_step`` 의 비용과 실제 step 의 비용이 같은지도 본다. 시뮬레이션이
더 비싸게 회계되면 쿼터 Q 가 실제 episode 의 같은 GE 보다 적은 action 을
허용하므로, planner 가 구조적으로 불리해진다.
"""

from __future__ import annotations

import math

from rl_newton.optimizers.action_space import NARROW
from rl_newton.optimizers.controllers import BudgetedMPCController, OneStepEfficiencyController
from rl_newton.optimizers.newton_cg import NewtonCGConfig, NewtonCGOptimizer
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask

SPACE = NARROW.with_fixed_step_size(1.0)
SPECS = [
    ("SPD k=1e2 d=64", QuadraticSpec(dimension=64, condition_number=1.0e2)),
    (
        "ill k=1e5 d=100",
        QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e5),
    ),
]


def greedy_prefix(spec: QuadraticSpec, quota: float) -> tuple[float, int, float]:
    """one-step efficiency 궤적을 누적비용 <= quota 인 마지막 prefix 에서 자른다.

    Returns:
        ``(prefix 최종 loss, prefix step 수, prefix 누적비용)``.
        어떤 step 도 쿼터에 안 들어가면 ``(초기 loss, 0, 0.0)``.
    """
    task = QuadraticTask(spec, seed=0)
    # 쿼터보다 넉넉히 돌려서 prefix 를 확보한다.
    config = NewtonCGConfig(total_steps=200, cost_budget_ge=quota * 3.0, initial_damping=1.0e-2)
    trace = NewtonCGOptimizer(
        task, OneStepEfficiencyController(SPACE), config, run_id="g", seed=0
    ).run()

    spent = 0.0
    loss = trace.initial_loss
    steps = 0
    for record in trace.records:
        if not math.isfinite(record.cost_ge) or not math.isfinite(record.train_loss_after):
            break
        if spent + record.cost_ge > quota:
            break
        spent += record.cost_ge
        loss = record.train_loss_after
        steps += 1
    return loss, steps, spent


def planner_frontier_best(
    spec: QuadraticSpec, quota: float, beam: int
) -> tuple[float, float, int, int]:
    """step 0 에서 planner 가 찾은 frontier 최소 loss.

    Returns:
        ``(최소 terminal loss, 채택 계획 비용, 채택 depth, 시뮬레이션 수)``.
    """
    task = QuadraticTask(spec, seed=0)
    planner = BudgetedMPCController(SPACE, quota_ge=quota, beam_width=beam, max_depth=40)
    # 1 step 만 돌려 step 0 의 계획 수립 결과만 본다.
    config = NewtonCGConfig(total_steps=1, cost_budget_ge=1.0e9, initial_damping=1.0e-2)
    NewtonCGOptimizer(task, planner, config, run_id="p", seed=0).run()
    choice = planner.choices[0]
    return choice.best_loss, choice.plan_used_ge, choice.chosen_depth, choice.n_simulations


def check_cost_accounting(spec: QuadraticSpec) -> None:
    """``simulate_step`` 비용과 실제 step 비용이 같은지. 다르면 쿼터가 왜곡된다."""
    action = SPACE.action_from_flat(0)
    config = NewtonCGConfig(total_steps=1, cost_budget_ge=1.0e9, initial_damping=1.0e-2)

    from rl_newton.optimizers.controllers import FixedController

    real = NewtonCGOptimizer(
        QuadraticTask(spec, seed=0), FixedController(action), config, run_id="r", seed=0
    ).run()
    real_cost = real.records[0].cost_ge

    opt = NewtonCGOptimizer(
        QuadraticTask(spec, seed=0), FixedController(action), config, run_id="s", seed=0
    )
    opt.task.reset()
    _loss, sim_cost, _ok = opt.simulate_step(action)

    verdict = "일치" if abs(sim_cost - real_cost) < 1e-9 else "불일치 (쿼터 왜곡)"
    print(
        f"  비용 회계 점검: 실제 step={real_cost:.4f} GE  "
        f"simulate_step={sim_cost:.4f} GE  -> {verdict}"
    )


def main() -> int:
    print("A = one-step efficiency 궤적을 누적비용 <= Q 인 prefix 에서 자른 loss")
    print("B = BudgetedMPC 가 step 0 에서 쿼터 Q 로 찾은 frontier 최소 loss")
    print("A 는 탐색 트리 안의 실현 가능한 계획이므로 B <= A 여야 한다.")
    print("B > A 이면 가지치기가 도달 가능한 계획을 버린 것이다 (탐색 문제).\n")
    for name, spec in SPECS:
        l0 = float(QuadraticTask(spec, seed=0).initial_loss)
        print(f"=== {name} | L0={l0:.4e} ===")
        check_cost_accounting(spec)
        for quota in (30.0, 60.0, 90.0):
            ref_loss, ref_steps, ref_cost = greedy_prefix(spec, quota)
            print(f"\n  Q={quota:g} GE")
            print(
                f"    A one-step prefix: loss={ref_loss:.6e}  "
                f"logΔ={math.log(l0) - math.log(max(ref_loss, 1e-300)):.4f}  "
                f"steps={ref_steps}  cost={ref_cost:.1f}"
            )
            print(
                f"    {'beam':>5} {'B loss':>14} {'B logΔ':>9} {'B cost':>7} "
                f"{'depth':>6} {'sims':>7}  판정"
            )
            for beam in (2, 4, 8):
                best, used, depth, sims = planner_frontier_best(spec, quota, beam)
                verdict = "탐색 손실" if best > ref_loss * (1.0 + 1e-12) else "OK"
                print(
                    f"    {beam:>5} {best:>14.6e} "
                    f"{math.log(l0) - math.log(max(best, 1e-300)):>9.4f} "
                    f"{used:>7.1f} {depth:>6} {sims:>7}  {verdict}"
                )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
