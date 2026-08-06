"""Stage 2 중간 점검. 행동 공간과 컨트롤러가 의도대로 동작하는지 확인한다.

일회성 진단 도구다. 본 판정은 ``scripts/run_headroom.py`` 가 두 트랙과 게이트
A~D로 수행한다 (프로토콜 D9).

확인 대상:
  - 세 행동 공간의 로그 해상도가 정렬됐는가 (게이트 B의 전제)
  - damping 이 로그 공간에서 누적되고 경계에서 클립되는가
  - H=1 planner 가 one-step efficiency 와 다른 선택을 하는가
    (terminal objective 로 바꾼 효과)
  - 높은 damping 이 CG 는 쉽게 만들지만 loss 감소는 느려지는 현상의 재현성
"""

from __future__ import annotations

import math

from rl_newton.optimizers.action_space import ABSOLUTE, NARROW, WIDE
from rl_newton.optimizers.controllers import (
    AverageRateEfficiencyPlanner,
    FixedController,
    HeuristicController,
    OneStepEfficiencyController,
)
from rl_newton.optimizers.newton_cg import NewtonCGConfig, NewtonCGOptimizer
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask

HEADER = (
    f"  {'controller':<30} {'logΔ(nat)':>10} {'final/L0':>11} {'cost_GE':>8} "
    f"{'HVP':>6} {'rej':>4} {'cgconv':>7} {'search_GE':>10}"
)


def run_one(task, controller, config, label: str) -> None:
    opt = NewtonCGOptimizer(task, controller, config, run_id=label, seed=0)
    trace = opt.run()
    ratio = trace.final_loss / trace.initial_loss if trace.initial_loss else float("nan")
    log_delta = (
        math.log(trace.initial_loss) - math.log(max(trace.final_loss, 1e-300))
        if trace.initial_loss > 0
        else float("nan")
    )
    print(
        f"  {label:<30} {log_delta:>10.3f} {ratio:>11.3e} "
        f"{trace.total_cost_ge:>8.1f} {trace.total_hvp:>6} {trace.n_rejected:>4} "
        f"{trace.n_cg_converged:>7} {trace.search_cost_ge:>10.1f}"
    )


def main() -> int:
    print("=== 행동 공간: 로그 해상도가 정렬됐는가 (게이트 B의 전제) ===")
    print(
        f"  {'name':<10} {'mode':<9} {'n_damp':>7} {'log10 간격':>11} "
        f"{'log10 범위':>11} {'actions':>8} {'HVP/sweep':>10}"
    )
    for space in (NARROW, WIDE, ABSOLUTE):
        logs = sorted(math.log10(v) for v in space.damping_values)
        gaps = [b - a for a, b in zip(logs, logs[1:], strict=False)]
        gap = sum(gaps) / len(gaps) if gaps else float("nan")
        fixed = space.with_fixed_step_size(1.0)
        print(
            f"  {space.name:<10} {space.damping_mode:<9} "
            f"{len(space.damping_values):>7} {gap:>11.3f} "
            f"{space.log10_span:>11.2f} {len(fixed):>8} {fixed.hvp_per_sweep:>10}"
        )
    print(f"  (기준: log10(3) = {math.log10(3.0):.3f})")

    spec_spd = QuadraticSpec(dimension=64, condition_number=1.0e3)
    spec_ill = QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e6)
    # GE 예산으로 끊는다. step 수를 맞추면 비용이 다른 것을 비교하게 된다.
    config = NewtonCGConfig(total_steps=200, cost_budget_ge=400.0, initial_damping=1.0e-2)

    n_fixed = NARROW.with_fixed_step_size(1.0)
    w_fixed = WIDE.with_fixed_step_size(1.0)
    a_fixed = ABSOLUTE.with_fixed_step_size(1.0)

    for name, spec in (
        ("SPD kappa=1e3 d=64", spec_spd),
        ("ill kappa=1e6 d=100", spec_ill),
    ):
        print(f"\n=== {name}, GE 예산 400, step_size 고정 ===")
        print(HEADER)
        print("  " + "-" * (len(HEADER) - 2))
        run_one(
            QuadraticTask(spec, seed=0),
            FixedController(n_fixed.action_from_flat(len(n_fixed) - 1)),
            config,
            "fixed(m=3,k=20)",
        )
        run_one(QuadraticTask(spec, seed=0), HeuristicController(n_fixed), config, "heuristic")
        run_one(
            QuadraticTask(spec, seed=0),
            OneStepEfficiencyController(n_fixed),
            config,
            "one_step_efficiency",
        )
        for horizon in (1, 3):
            run_one(
                QuadraticTask(spec, seed=0),
                AverageRateEfficiencyPlanner(
                    n_fixed, horizon=horizon, beam_width=3, track="fixed_budget"
                ),
                config,
                f"mpc_H{horizon}(narrow)",
            )
        run_one(
            QuadraticTask(spec, seed=0),
            AverageRateEfficiencyPlanner(a_fixed, horizon=3, beam_width=3, track="fixed_budget"),
            config,
            "mpc_H3(absolute)",
        )
        run_one(
            QuadraticTask(spec, seed=0),
            AverageRateEfficiencyPlanner(w_fixed, horizon=3, beam_width=3, track="fixed_budget"),
            config,
            "mpc_H3(wide)",
        )

    print("\n=== 높은 damping: CG는 쉬워지고 loss 감소는 느려지는가 (재현성) ===")
    print(f"  {'task':<28} {'damping':>9} {'logΔ':>8} {'CG수렴/step':>12} {'평균resid':>11}")
    cases = [
        ("SPD kappa=1e2", QuadraticSpec(dimension=64, condition_number=1e2)),
        ("SPD kappa=1e4", QuadraticSpec(dimension=64, condition_number=1e4)),
        (
            "ill kappa=1e6",
            QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1e6),
        ),
    ]
    action = next(
        a for a in n_fixed.iter_actions() if a.damping_multiplier == 1.0 and a.cg_budget == 20
    )
    for case_name, spec in cases:
        for damping in (1e-2, 1e2, 1e6):
            task = QuadraticTask(spec, seed=0)
            cfg = NewtonCGConfig(total_steps=200, cost_budget_ge=400.0, initial_damping=damping)
            trace = NewtonCGOptimizer(task, FixedController(action), cfg, run_id="d", seed=0).run()
            residuals = [
                float(r.extra.get("cg_residual_ratio", float("nan"))) for r in trace.records
            ]
            finite = [r for r in residuals if math.isfinite(r)]
            log_delta = math.log(trace.initial_loss) - math.log(max(trace.final_loss, 1e-300))
            conv = trace.n_cg_converged / max(trace.n_steps, 1)
            print(
                f"  {case_name:<28} {damping:>9.0e} {log_delta:>8.3f} "
                f"{conv:>12.2f} {(sum(finite) / len(finite) if finite else float('nan')):>11.3e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
