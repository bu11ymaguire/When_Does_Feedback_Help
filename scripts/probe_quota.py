"""진단: 쿼터 사다리가 실제로 쿼터에 의해 구속되는가.

dry run 에서 두 가지가 관측됐다.

```text
narrow Q=1  logΔ=58.7554  depth>1=0.00  cap=0.00
narrow Q=4  logΔ=40.6091  depth>1=1.00  cap=1.00   <- 더 나쁘다
```

``cap=1.00`` 은 ``max_depth`` 계산 상한이 **모든 step 에서** 걸렸다는 뜻이다.
그러면 계획이 쿼터를 다 쓰기 전에 잘리므로 "동일 예산" 전제가 깨진다. 싼
action 계획은 6 step 에서 잘려 쿼터의 일부만 쓰고, 비싼 action 계획은 4 step 만에
쿼터를 거의 다 쓴다. terminal loss 로 비교하면 비싼 쪽이 유리해진다.

확인 항목
---------
1. ``max_depth`` 를 충분히 올리면 ``cap`` 이 0 이 되는가
2. 그때도 Q=4 가 Q=1 보다 나쁜가 (= MPC 자체의 문제인가)
3. ``quota_used_fraction`` 이 1 에 가까운가 (쿼터가 실제로 구속하는가)
4. 계산 비용이 감당 가능한가
"""

from __future__ import annotations

import math
import time

from rl_newton.benchmark.metrics import budget_respecting_prefix
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
BUDGET = 150.0


def _run(controller, spec):
    task = QuadraticTask(spec, seed=0)
    config = NewtonCGConfig(total_steps=200, cost_budget_ge=BUDGET, initial_damping=1.0e-2)
    started = time.perf_counter()
    trace = NewtonCGOptimizer(task, controller, config, run_id="p", seed=0).run()
    elapsed = time.perf_counter() - started
    # 예산 초과 step 을 잘라야 공정하다 (프로토콜 D11). 이것 없이는 큰 step 을
    # 고르는 컨트롤러가 최대 한 step 만큼 예산을 공짜로 더 쓴다.
    final_loss, spent, n_steps = budget_respecting_prefix(trace, BUDGET)
    log_delta = math.log(trace.initial_loss) - math.log(max(final_loss, 1e-300))
    depths: dict[int, int] = {}
    for choice in controller.choices:
        depths[choice.chosen_depth] = depths.get(choice.chosen_depth, 0) + 1
    caps = [c.depth_cap_hit for c in controller.choices if hasattr(c, "depth_cap_hit")]
    used = [
        c.plan_used_ge / c.quota_ge
        for c in controller.choices
        if math.isfinite(getattr(c, "plan_used_ge", float("nan"))) and c.quota_ge > 0
    ]
    sims = [float(getattr(c, "n_simulations", 0)) for c in controller.choices]
    budgets = [r.cg_budget for r in trace.records[:n_steps]]
    return {
        "log_delta": log_delta,
        "steps": n_steps,
        "spent": spent,
        "search": trace.search_cost_ge,
        "depths": dict(sorted(depths.items())),
        "cap": (sum(caps) / len(caps)) if caps else float("nan"),
        "used": (sum(used) / len(used)) if used else float("nan"),
        "sims": (sum(sims) / len(sims)) if sims else 0.0,
        "wall": elapsed,
        "mean_k": sum(budgets) / max(len(budgets), 1),
    }


def main() -> int:
    for name, spec in SPECS:
        print(f"\n=== {name} | narrow, beam 2, {BUDGET:g} GE ===")
        base = _run(OneStepEfficiencyController(SPACE), spec)
        print(
            f"  {'controller':<26} {'logΔ(nat)':>10} {'steps':>6} {'소모GE':>7} {'d>1':>5} "
            f"{'cap':>5} {'Q사용':>6} {'sims':>6} {'평균k':>6} {'search':>9} {'wall(s)':>8}"
        )
        print(
            f"  {'C0 one_step_efficiency':<26} {base['log_delta']:>10.4f} "
            f"{base['steps']:>6} {base['spent']:>7.1f} {'-':>5} {'-':>5} {'-':>6} {'-':>6} "
            f"{base['mean_k']:>6.1f} {base['search']:>9.0f} {base['wall']:>8.1f}"
        )
        for quota, max_depth in ((1.0, 6), (2.0, 6), (4.0, 6), (1.0, 24), (2.0, 24), (4.0, 24)):
            planner = BudgetedMPCController(
                SPACE, quota_multiplier=quota, beam_width=2, max_depth=max_depth
            )
            r = _run(planner, spec)
            deep = 1.0 - r["depths"].get(1, 0) / max(sum(r["depths"].values()), 1)
            label = f"Q={quota:g} maxdepth={max_depth}"
            print(
                f"  {label:<26} {r['log_delta']:>10.4f} {r['steps']:>6} {r['spent']:>7.1f} "
                f"{deep:>5.2f} {r['cap']:>5.2f} {r['used']:>6.2f} {r['sims']:>6.0f} "
                f"{r['mean_k']:>6.1f} {r['search']:>9.0f} {r['wall']:>8.1f}"
            )
            print(f"    depths={r['depths']}  quota={planner.quota_ge:.1f} GE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def probe_full_horizon() -> None:
    """쿼터가 남은 episode 예산 전체를 덮으면 planning 이 이기는가.

    위 결과에서 Q 를 키울수록 나빠졌다. 그런데 Q=4xc_max=85 GE 는 episode
    예산 150 GE 의 57% 다. 즉 planner 는 episode 의 절반만 보고 첫 action 을
    고른 뒤 매 step 재계획한다. 이 truncation 편향이 원인일 수 있다.

    예산을 줄여 ``Q >= 남은 예산`` 이 되게 하면 truncation 이 없어진다. 그때
    planner 는 "고정 예산에서 terminal loss 최소화" 문제의 (beam 근사) 해를
    직접 푸는 것이므로, 여기서도 one-step 을 못 이기면 planning 가치 부족은
    truncation 탓이 아니다.
    """
    print("\n" + "=" * 78)
    print("=== 끝까지 계획 (Q >= episode 예산) ===")
    for name, spec in SPECS:
        for budget in (30.0, 60.0):
            print(f"\n  {name} | 예산 {budget:g} GE | narrow, beam 2")
            global BUDGET
            saved = BUDGET
            BUDGET = budget
            try:
                base = _run(OneStepEfficiencyController(SPACE), spec)
                print(
                    f"    {'controller':<24} {'logΔ(nat)':>10} {'steps':>6} {'d>1':>5} "
                    f"{'cap':>5} {'Q사용':>6} {'평균k':>6} {'search':>9} {'wall(s)':>8}"
                )
                print(
                    f"    {'C0 one_step':<24} {base['log_delta']:>10.4f} {base['steps']:>6} "
                    f"{'-':>5} {'-':>5} {'-':>6} {base['mean_k']:>6.1f} "
                    f"{base['search']:>9.0f} {base['wall']:>8.1f}"
                )
                for quota_ge in (budget, budget * 1.5):
                    planner = BudgetedMPCController(
                        SPACE, quota_ge=quota_ge, beam_width=2, max_depth=40
                    )
                    r = _run(planner, spec)
                    deep = 1.0 - r["depths"].get(1, 0) / max(sum(r["depths"].values()), 1)
                    label = f"Q={quota_ge:g} GE (전체)"
                    print(
                        f"    {label:<24} {r['log_delta']:>10.4f} {r['steps']:>6} "
                        f"{deep:>5.2f} {r['cap']:>5.2f} {r['used']:>6.2f} "
                        f"{r['mean_k']:>6.1f} {r['search']:>9.0f} {r['wall']:>8.1f}"
                    )
                    print(f"      depths={r['depths']}")
            finally:
                BUDGET = saved


if __name__ == "__main__":
    probe_full_horizon()
