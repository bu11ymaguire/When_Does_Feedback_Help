"""진단: 세 실행 방식 비교 (프로토콜 D12).

쿼터를 키우면 실제 episode 성능이 나빠졌다. 그런데 planner 가 찾은 계획 자체는
동일 비용의 greedy 궤적보다 좋았다. 따라서 문제는 목적함수나 탐색이 아니라
**실행 방식**일 수 있다. 세 가지를 구분한다.

```text
1. committed        계획을 끝까지 실행. 재계획 없음
2. fresh-quota      매 step 미래 예산 Q 를 새로 지급 (현재 방식)
3. shrinking-quota  쓴 비용을 차감. horizon 을 새로 연장하지 않음
```

동일 planner, 동일 쿼터, 동일 GE 예산에서 실행 방식만 바꾼다. 그래야 차이가
탐색 품질 차이와 섞이지 않는다. beam 은 4 와 8 만 쓴다. beam 2 에서 이미
탐색 손실이 한 번 관측됐으므로 실행 방식과 탐색 부족을 섞지 않기 위한 것이다.

해석표 (리뷰 제공)
------------------
```text
Committed > C0, Shrinking ~= Committed, Fresh < Committed
    -> 쿼터 초기화에 의한 시간 불일치

Committed > C0, Shrinking < Committed
    -> 재계획 또는 beam pruning 이 좋은 suffix 를 파괴

Shrinking > Committed > C0
    -> 피드백 기반 재계획에 실질적 가치가 있음

Committed > C0, Fresh/Shrinking <= C0
    -> 좋은 open-loop schedule 은 있지만 feedback controller 로 회수 못 함

셋 다 <= C0
    -> 해당 조건에서 temporal planning headroom 이 거의 없음

예측값 != committed 실행값
    -> planner simulator 또는 실행 회계 버그 (다른 해석 전에 이것부터)
```

이 진단은 **pilot 기록 전용**이다. 기존 게이트 C 통계를 사후 변경하지 않는다.
"""

from __future__ import annotations

import math
import time

from rl_newton.benchmark.metrics import budget_respecting_prefix
from rl_newton.optimizers.action_space import NARROW
from rl_newton.optimizers.controllers import (
    BudgetedMPCController,
    CommittedPlanController,
    OneStepEfficiencyController,
    ShrinkingQuotaMPCController,
)
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
MAX_DEPTH = 24


def run(controller, spec) -> dict[str, float]:
    task = QuadraticTask(spec, seed=0)
    config = NewtonCGConfig(total_steps=300, cost_budget_ge=BUDGET, initial_damping=1.0e-2)
    started = time.perf_counter()
    trace = NewtonCGOptimizer(task, controller, config, run_id="x", seed=0).run()
    wall = time.perf_counter() - started
    # 예산 초과 step 절단 (프로토콜 D11). 없으면 큰 step 쪽이 공짜로 이득본다.
    final_loss, spent, n_steps = budget_respecting_prefix(trace, BUDGET)
    log_delta = math.log(trace.initial_loss) - math.log(max(final_loss, 1e-300))

    choices = getattr(controller, "choices", []) or []
    depths = [c.chosen_depth for c in choices]
    caps = [bool(getattr(c, "depth_cap_hit", False)) for c in choices]
    ks = [r.cg_budget for r in trace.records[:n_steps]]
    return {
        "log_delta": log_delta,
        "steps": n_steps,
        "spent": spent,
        "n_plans": len(choices),
        "max_depth": max(depths) if depths else 0,
        "cap": (sum(caps) / len(caps)) if caps else float("nan"),
        "mean_k": (sum(ks) / len(ks)) if ks else float("nan"),
        "search": trace.search_cost_ge,
        "wall": wall,
    }


def prediction_check(spec, quota: float, beam: int) -> str:
    """committed 실행이 planner 예측과 일치하는가. 다른 해석 전에 이것부터."""
    planner = CommittedPlanController(SPACE, quota_ge=quota, beam_width=beam, max_depth=MAX_DEPTH)
    task = QuadraticTask(spec, seed=0)
    config = NewtonCGConfig(total_steps=300, cost_budget_ge=quota, initial_damping=1.0e-2)
    trace = NewtonCGOptimizer(task, planner, config, run_id="v", seed=0).run()
    if not planner.predictions:
        return "계획 없음"
    _step, predicted, plan_cost = planner.predictions[0]
    spent = 0.0
    realized = trace.initial_loss
    for record in trace.records:
        spent += record.cost_ge
        realized = record.train_loss_after
        if spent >= plan_cost - 1.0e-9:
            break
    rel = abs(realized - predicted) / max(abs(predicted), 1.0e-300)
    tag = "일치" if rel < 1.0e-9 else f"불일치 rel={rel:.2e}"
    return f"예측 {predicted:.6e} / 실행 {realized:.6e} -> {tag}"


def main() -> int:
    for name, spec in SPECS:
        print(f"\n{'=' * 100}")
        print(f"=== {name} | narrow, {BUDGET:g} GE, max_depth={MAX_DEPTH} ===")
        base = run(OneStepEfficiencyController(SPACE), spec)
        print(
            f"  C0 one_step_efficiency:  logΔ={base['log_delta']:.4f}  "
            f"steps={base['steps']}  소모={base['spent']:.1f} GE  "
            f"평균k={base['mean_k']:.1f}  search={base['search']:.0f}"
        )
        for beam in (4, 8):
            for quota in (1.0, 2.0, 4.0):
                planners = {
                    "committed": CommittedPlanController(
                        SPACE, quota_multiplier=quota, beam_width=beam, max_depth=MAX_DEPTH
                    ),
                    "fresh": BudgetedMPCController(
                        SPACE, quota_multiplier=quota, beam_width=beam, max_depth=MAX_DEPTH
                    ),
                    "shrinking": ShrinkingQuotaMPCController(
                        SPACE, quota_multiplier=quota, beam_width=beam, max_depth=MAX_DEPTH
                    ),
                }
                print(f"\n  --- beam {beam}, Q={quota:g} x c_max ---")
                q_abs = float("nan")
                for mode, planner in planners.items():
                    r = run(planner, spec)
                    q_abs = planner.quota_ge
                    delta = r["log_delta"] - base["log_delta"]
                    print(
                        f"    {mode:<10} logΔ={r['log_delta']:>9.4f} "
                        f"({delta:+7.4f} vs C0)  steps={r['steps']:>3} "
                        f"소모={r['spent']:>6.1f}  계획수={r['n_plans']:>3} "
                        f"최대depth={r['max_depth']:>3}  cap={r['cap']:.2f}  "
                        f"평균k={r['mean_k']:>4.1f}  search={r['search']:>8.0f}  "
                        f"{r['wall']:>5.1f}s"
                    )
                print(f"    쿼터={q_abs:.1f} GE   검증: {prediction_check(spec, q_abs, beam)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
