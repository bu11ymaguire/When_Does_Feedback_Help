"""`rosen_d5` 의 정체점이 국소최소점인지 확인한다.

모든 컨트롤러가 `loss ~ 3.9308` 에서 멈췄다. 두 가설을 구별한다.

```text
가설 1  수치 floor 나 예산 부족   -> 예산을 늘리면 더 내려간다
가설 2  국소최소점               -> |grad| ~ 0, Hessian PSD, 더 돌려도 그대로
```

`rosenbrock.py` 모듈 docstring 은 이미 "d >= 4 에서는 국소 최소값이 추가로
존재한다" 고 적어 두었다. 그것이 실제로 관측된 정체점인지 수치로 확인한다.
"""

from __future__ import annotations

import torch

from rl_newton.tasks.rosenbrock import RosenbrockSpec, RosenbrockTask


def report(label: str, task: RosenbrockTask) -> float:
    x = task.params[0]
    loss = task.loss()
    (grad,) = torch.autograd.grad(loss, x, create_graph=True)
    rows = []
    for i in range(x.numel()):
        (row,) = torch.autograd.grad(grad[i], x, retain_graph=True)
        rows.append(row.detach())
    hess = torch.stack(rows)
    eigs = torch.linalg.eigvalsh(0.5 * (hess + hess.T))
    value = float(loss)
    print(f"  {label}")
    print(f"    x        {[round(float(v), 8) for v in x.detach()]}")
    print(f"    loss     {value:.12f}")
    print(f"    |grad|   {float(grad.detach().norm()):.6e}")
    print(f"    eig min  {float(eigs.min()):+.6e}   eig max {float(eigs.max()):+.6e}")
    print(f"    PSD      {bool(eigs.min() >= 0)}")
    return value


def refine(task: RosenbrockTask, start: torch.Tensor) -> None:
    """LBFGS 로 정밀 수렴시켜 임계점을 찾는다."""
    task.move_to(start.to(dtype=task.params[0].dtype))
    x = task.params[0]
    opt = torch.optim.LBFGS(
        [x], max_iter=5000, tolerance_grad=1e-16, tolerance_change=1e-18,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        value = task.loss()
        value.backward()
        return value

    opt.step(closure)


def main() -> int:
    spec = RosenbrockSpec(dimension=5)
    task = RosenbrockTask(spec, seed=2, dtype=torch.float64)
    print(f"spec  {spec}")
    print(f"dtype {task.params[0].dtype}")
    print(f"L0    {task.initial_loss:.6f}")
    print("전역최소점 x=(1,...,1), loss=0")
    print()

    report("표준 시작점 (-1.2, 1, 1, 1, 1)", task)
    print()

    d = spec.dimension
    starts = {
        "(-1, 1, 1, 1, 1) 에서 정밀 수렴": torch.tensor([-1.0] + [1.0] * (d - 1)),
        "표준 시작점에서 정밀 수렴": torch.tensor([-1.2] + [1.0] * (d - 1)),
        "전역최소점 근처에서 정밀 수렴": torch.full((d,), 0.9),
    }
    for label, start in starts.items():
        refine(task, start)
        report(label, task)
        print()

    print("관측된 컨트롤러 정체 loss (seed 2): 3.9308388 ~ 3.9310777, 서로 다른 값 14개")
    print("14개가 모두 같은 임계점 근방이면 정체 원인은 국소최소점이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
