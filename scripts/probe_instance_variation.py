"""seed 가 실제로 서로 다른 인스턴스를 만드는지 확인한다.

`rosen_d5` 진단에서 seed 2 와 seed 4 의 모든 run 이 bitwise 동일하게 나왔다.
`RosenbrockSpec(dimension=5)` 는 `randomize_start=False` 이므로 seed 가 시작점을
바꾸지 않는다. challenge set 의 quadratic 도 같은 문제가 있는지 확인한다.

또한 quadratic 이 볼록(SPD)인지 확인한다. 국소최소점이 있으면 `rosen_d5` 와 같은
cap 이 생긴다.
"""

from __future__ import annotations

import math

import torch

from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask
from rl_newton.tasks.rosenbrock import RosenbrockSpec, RosenbrockTask

CHALLENGE = [
    QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e3),
    QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e4),
    QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e5),
    QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e6),
]

SEEDS = (2, 3, 4)


def refine_to_critical_point(task: RosenbrockTask) -> float:
    """LBFGS 로 임계점까지 정밀 수렴시키고 그 loss 를 반환한다.

    D23 이 제안하는 ``achievable_ceiling`` 의 ``L_ref`` 계산에 해당한다.
    """
    x = task.params[0]
    opt = torch.optim.LBFGS(
        [x],
        max_iter=5000,
        tolerance_grad=1e-16,
        tolerance_change=1e-18,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        value = task.loss()
        value.backward()
        return value

    opt.step(closure)
    return float(task.loss().detach())


def main() -> int:
    print("=== challenge set quadratic: seed 별 인스턴스 구별 ===")
    print(f"  {'spec':<34}{'seed':>5}{'L0':>16}{'|x0|':>12}{'eig min':>14}")
    for spec in CHALLENGE:
        losses = []
        for seed in SEEDS:
            task = QuadraticTask(spec, seed=seed, dtype=torch.float64)
            x0 = task.params[0].detach()
            hess = task.hessian_matrix() if hasattr(task, "hessian_matrix") else None
            eig = float(torch.linalg.eigvalsh(hess).min()) if hess is not None else float("nan")
            losses.append(task.initial_loss)
            label = f"d{spec.dimension}_k{spec.condition_number:.0e}"
            print(
                f"  {label:<34}{seed:>5}{task.initial_loss:>16.6f}"
                f"{float(x0.norm()):>12.6f}{eig:>14.6e}"
            )
        distinct = len({round(v, 10) for v in losses})
        print(f"    -> 서로 다른 L0 {distinct}/{len(SEEDS)}개")
    print()

    print("=== rosen_d5: seed 별 인스턴스 구별 ===")
    for randomize in (False, True):
        spec = RosenbrockSpec(dimension=5, randomize_start=randomize)
        losses = []
        for seed in SEEDS:
            task = RosenbrockTask(spec, seed=seed, dtype=torch.float64)
            x0 = task.params[0].detach()
            losses.append(task.initial_loss)
            print(
                f"  randomize_start={randomize!s:<5} seed{seed}  "
                f"L0={task.initial_loss:>12.6f}  "
                f"x0={[round(float(v), 5) for v in x0]}"
            )
        distinct = len({round(v, 10) for v in losses})
        print(f"    -> 서로 다른 L0 {distinct}/{len(SEEDS)}개")
        print()

    print("=== randomize_start=True 로 바꿔도 같은 basin 인가 ===")
    print("  국소최소점(loss=3.930839)에 수렴하면 cap 은 그대로다.")
    spec = RosenbrockSpec(dimension=5, randomize_start=True)
    for seed in SEEDS:
        task = RosenbrockTask(spec, seed=seed, dtype=torch.float64)
        l0 = task.initial_loss
        terminal = refine_to_critical_point(task)
        ceiling = math.log(l0) - math.log(max(terminal, 1e-300))
        basin = "국소최소점" if terminal > 1e-6 else "전역최소점"
        print(
            f"  seed{seed}  L0={l0:>10.4f} -> 수렴 loss={terminal:>14.10f}  "
            f"{basin}  달성가능 ceiling={ceiling:>6.3f} nat"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
