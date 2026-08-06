"""micro-neural 두 regime 의 사전 점검 (D24/D25).

full 실행 전에 다음을 확인한다. **컨트롤러 우열은 보지 않는다.**

```text
파라미터 수와 L0 규모
J_achievable (참조 solver panel).  국소최소점 cap 이 있는지
seed 가 실제로 다른 인스턴스를 만드는지
두 regime 이 같은 데이터와 초기점을 쓰는지
R2 에서 control loss 가 실제로 흔들리는지
150 GE 안에서 baseline 이 의미 있게 줄이는지
```
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace

import torch

from rl_newton.benchmark.eligibility import (
    achievable_ceiling,
    check_seed_variation,
    reference_panel,
)
from rl_newton.benchmark.metrics import TargetSpec, summarize_run
from rl_newton.optimizers.action_space import NARROW
from rl_newton.optimizers.controllers import (
    FixedController,
    HeuristicController,
    OneStepEfficiencyController,
)
from rl_newton.optimizers.newton_cg import NewtonCGConfig, NewtonCGOptimizer
from rl_newton.tasks.micro_neural import MicroNeuralSpec, MicroNeuralTask

TARGET = TargetSpec("relative_loss", 2.0e-1)


def build(spec: MicroNeuralSpec, seed: int) -> MicroNeuralTask:
    return MicroNeuralTask(spec, seed=seed, dtype=torch.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=150.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--reference-iters", type=int, default=3000)
    args = parser.parse_args()

    base = MicroNeuralSpec(
        input_dim=32,
        hidden_dim=128,
        n_classes=5,
        n_samples=512,
        teacher_hidden_dim=256,
        label_noise=0.05,
    )
    stochastic = replace(base, regime="controlled_stochastic", batch_size=64)

    print(f"파라미터 수 {base.n_parameters}")
    print(f"data_key 동일 여부  {base.data_key == stochastic.data_key}")
    print()

    space = NARROW.with_fixed_step_size(1.0)
    seed0 = args.seeds[0]

    # --- D25 eligibility ---
    variation = check_seed_variation(lambda s: build(base, s), args.seeds)
    print(f"seed 변동  {variation.describe()}")

    l0 = float(build(base, seed0).initial_loss)
    panel = reference_panel(
        lambda: build(base, seed0), max_iter=args.reference_iters
    )
    ceiling = achievable_ceiling(l0, panel)
    print(f"달성 가능 상한  {ceiling.describe()}")
    for run in panel:
        mark = "수렴" if run.is_critical_point else "미수렴"
        print(f"  {run.name:<16} final={run.final_loss:.6e} |grad|={run.grad_norm:.3e} {mark}")
    print()

    # --- 두 regime 이 같은 문제인지 ---
    a, b = build(base, seed0), build(stochastic, seed0)
    same = a.initial_loss == b.initial_loss
    print(f"두 regime 초기 loss 동일  {same}  ({a.initial_loss:.10f} / {b.initial_loss:.10f})")

    probe = build(stochastic, seed0)
    seen = []
    for _ in range(6):
        seen.append(float(probe.curvature_loss().detach()))
        probe.advance_batch()
    spread = max(seen) - min(seen)
    print(f"R2 control loss 변동  {spread:.6f}  (표본 {len(set(seen))}종)")
    print(f"  전체 데이터 loss {float(probe.loss().detach()):.6f}")
    print()

    # --- baseline 이 150 GE 안에서 의미 있게 줄이는가 ---
    print(f"{'regime':<22}{'controller':<18}{'logΔ':>10}{'final':>12}{'acc':>8}{'steps':>7}{'rej':>6}")
    for label, spec in (("full_batch", base), ("controlled_stochastic", stochastic)):
        for name, factory in (
            ("fixed[6]", lambda: FixedController(space.action_from_flat(6))),
            ("heuristic", lambda: HeuristicController(space)),
            ("onestep", lambda: OneStepEfficiencyController(space)),
        ):
            task = build(spec, seed0)
            config = NewtonCGConfig(
                total_steps=300, cost_budget_ge=args.budget, initial_damping=1.0e-2
            )
            trace = NewtonCGOptimizer(
                task, factory(), config, run_id="probe", seed=seed0
            ).run()
            s = summarize_run(trace, TARGET, budget_ge=args.budget)
            rej = sum(1 for r in trace.records if not r.step_accepted) / max(
                1, len(trace.records)
            )
            print(
                f"{label:<22}{name:<18}{s.log_improvement:>10.4f}"
                f"{s.final_loss:>12.6f}{task.accuracy():>8.3f}"
                f"{s.n_steps:>7}{rej:>6.2f}"
            )
    print()
    print("주의: 컨트롤러 우열을 보려는 것이 아니다. 측정 가능성만 확인한다.")
    print(f"참고 J_achievable = {ceiling.nats:.4f} nat, 제한 = {ceiling.limited_by}")
    if math.isfinite(ceiling.nats):
        print(f"  baseline median logΔ 가 {ceiling.nats - 3.0:.4f} nat 이하여야 D25 여유 조건을 만족한다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
