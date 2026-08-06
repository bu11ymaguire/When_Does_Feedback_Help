"""진단: horizon 이 planner 선택을 실제로 바꾸는가.

dry run 에서 ``depth>1`` 비율이 0이고 beam 1/2 의 결과가 소수점까지 같았다.
H=3 인데 깊은 계획을 한 번도 채택하지 않았다는 뜻이다. 구현 버그인지
효용 함수의 구조적 성질인지 구분해야 한다.

의심되는 원인
-------------
Track E 효용은 ``(log L_start - log L_terminal) / cumulative_cost`` 다. 이는
step 별 rate 의 **비용 가중 평균**이다. mediant 부등식에 의해

    min(rate_1, rate_2) <= (g1+g2)/(c1+c2) <= max(rate_1, rate_2)

이므로, depth 1 에서 이미 최대 rate 를 골랐다면 depth 2 를 더해도 평균이
희석될 뿐이다. depth 2 가 이기려면 두 번째 step 의 rate 가 **현재 가능한
모든 rate 보다 높아야** 한다. 수익 체감이 있는 문제에서는 드물다.

Track T 효용은 ``-(cost + 남은거리/rate)`` 로 누적 형태이므로 다를 수 있다.

확인 항목
---------
1. H=1/3/5 의 최종 loss 와 선택 action 이 실제로 다른가
2. chosen_depth 분포
3. Track T 에서는 depth>1 이 나오는가
"""

from __future__ import annotations

import math

from rl_newton.optimizers.action_space import NARROW, WIDE
from rl_newton.optimizers.controllers import AverageRateEfficiencyPlanner
from rl_newton.optimizers.newton_cg import NewtonCGConfig, NewtonCGOptimizer
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask


def depth_histogram(planner: AverageRateEfficiencyPlanner) -> dict[int, int]:
    counts: dict[int, int] = {}
    for choice in planner.choices:
        counts[choice.chosen_depth] = counts.get(choice.chosen_depth, 0) + 1
    return dict(sorted(counts.items()))


def run(space, horizon, track, target_loss, spec, budget=150.0):
    task = QuadraticTask(spec, seed=0)
    planner = AverageRateEfficiencyPlanner(
        space,
        horizon=horizon,
        beam_width=3,
        track=track,
        target_loss=target_loss,
    )
    config = NewtonCGConfig(total_steps=200, cost_budget_ge=budget, initial_damping=1.0e-2)
    trace = NewtonCGOptimizer(task, planner, config, run_id="p", seed=0).run()
    log_delta = math.log(trace.initial_loss) - math.log(max(trace.final_loss, 1e-300))
    actions = [(r.extra["damping_multiplier"], r.cg_budget) for r in trace.records[:6]]
    return {
        "log_delta": log_delta,
        "n_steps": trace.n_steps,
        "cost": trace.total_cost_ge,
        "search": trace.search_cost_ge,
        "depths": depth_histogram(planner),
        "first_actions": actions,
    }


def main() -> int:
    specs = [
        ("SPD k=1e2 d=64", QuadraticSpec(dimension=64, condition_number=1.0e2)),
        (
            "ill k=1e5 d=100",
            QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e5),
        ),
    ]
    space = NARROW.with_fixed_step_size(1.0)

    for name, spec in specs:
        print(f"\n=== {name} | Track E (fixed_budget) | narrow, beam 3 ===")
        print(f"  {'H':>2} {'logΔ(nat)':>11} {'steps':>6} {'cost':>7} {'search':>9} depths")
        baseline = None
        for horizon in (1, 3, 5):
            r = run(space, horizon, "fixed_budget", None, spec)
            if baseline is None:
                baseline = r["log_delta"]
            same = "동일" if abs(r["log_delta"] - baseline) < 1e-9 else "다름"
            print(
                f"  {horizon:>2} {r['log_delta']:>11.4f} {r['n_steps']:>6} "
                f"{r['cost']:>7.1f} {r['search']:>9.1f} {r['depths']}  {same}"
            )

        # Track T: 절대 target = 1e-4 * L0
        task0 = QuadraticTask(spec, seed=0)
        target = 1.0e-4 * task0.initial_loss
        print(f"\n=== {name} | Track T (cost_to_target, τ={target:.3e}) ===")
        print(f"  {'H':>2} {'logΔ(nat)':>11} {'steps':>6} {'cost':>7} {'search':>9} depths")
        for horizon in (1, 3, 5):
            r = run(space, horizon, "cost_to_target", target, spec)
            print(
                f"  {horizon:>2} {r['log_delta']:>11.4f} {r['n_steps']:>6} "
                f"{r['cost']:>7.1f} {r['search']:>9.1f} {r['depths']}"
            )

    print("\n=== wide 공간에서도 같은가 (Track E, ill k=1e5) ===")
    wide = WIDE.with_fixed_step_size(1.0)
    spec = specs[1][1]
    print(f"  {'space':<8} {'H':>2} {'logΔ(nat)':>11} depths")
    for label, sp in (("narrow", space), ("wide", wide)):
        for horizon in (1, 5):
            r = run(sp, horizon, "fixed_budget", None, spec)
            print(f"  {label:<8} {horizon:>2} {r['log_delta']:>11.4f} {r['depths']}")

    print("\n=== 첫 6 step 의 선택 action (H=1 vs H=5, ill k=1e5, Track E) ===")
    for horizon in (1, 5):
        r = run(space, horizon, "fixed_budget", None, specs[1][1])
        print(f"  H={horizon}: {r['first_actions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
