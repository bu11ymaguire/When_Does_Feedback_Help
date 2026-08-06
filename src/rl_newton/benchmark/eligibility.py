"""Benchmark eligibility: **달성 가능한** 상한을 참조 solver panel 로 잰다 (D25).

왜 필요한가
-----------
D20 은 ``ceiling`` 을 이렇게 계산했다.

```text
ceiling = log(L0) − log(L0 x RELATIVE_LOSS_FLOOR) = 31.44 nat
```

이것은 **전역최소점이 0 이고 도달 가능하다**고 가정한다. `rosen_d5` 에서 이
가정이 깨졌다 (D23).

```text
D20 이 쓴 ceiling         31.44 nat   -> "여유 29.62 nat" 으로 통과
실제 달성 가능 상한        1.8175 nat  -> 여유 0.0000 nat
```

표준 시작점의 basin 에 strict 국소최소점(``loss=3.930839434``, ``|grad|=1e-8``,
Hessian 양정)이 있어 모든 baseline 이 거기 도달했다. calibration 이 이것을
놓쳤다.

참조 solver 하나만 쓰지 않는다
------------------------------
단일 solver 의 수렴점을 절대 상한으로 삼으면 그 solver 의 약점이 상한으로
굳는다. 여러 참조 방법을 긴 예산으로 돌려 **최소값**을 쓴다.

```text
L_ref = min over reference runs of L_final
J_achievable = log(L0) − log(max(L_ref, L_floor))
```

이 값은 **컨트롤러 평가 점수가 아니다.** calibration 지표로만 쓴다.

```text
baseline 이 이미 도달 가능한 최적점에 포화됐는가
여전히 비교할 수 있는 headroom 이 남았는가
```

참조 solver 는 planner 를 포함하지 않는다. 포함하면 spec 선정에 planner 결과가
새어 든다 (D20).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from rl_newton.benchmark.metrics import RELATIVE_LOSS_FLOOR

__all__ = [
    "ReferenceRun",
    "AchievableCeiling",
    "SeedVariationReport",
    "REFERENCE_MAX_ITERS",
    "REFERENCE_AGREEMENT_NAT",
    "reference_panel",
    "achievable_ceiling",
    "check_seed_variation",
]

REFERENCE_MAX_ITERS = 20000
"""참조 solver 반복 상한. 예산 제약이 상한을 만들지 않을 만큼 크게 둔다."""

REFERENCE_AGREEMENT_NAT = 0.5
"""참조 solver 간 수렴점이 이 이상 벌어지면 상한 추정이 불안정하다고 본다."""


@dataclass(frozen=True, slots=True)
class ReferenceRun:
    """참조 solver 한 번의 실행 결과.

    Attributes:
        from_task_start: task 자신의 시작점에서 출발했는가. **``L_ref`` 계산에는
            이것이 ``True`` 인 run 만 쓴다.**

            컨트롤러는 항상 task 의 시작점에서 출발한다. 다른 초기화에서 더 좋은
            점을 찾았다는 사실은 그 시작점의 basin 이 전역최적이 아니라는 **진단**
            이지, 컨트롤러가 도달할 수 있는 상한이 아니다. 섞으면 `rosen_d5` 처럼
            국소최소점에 갇힌 task 가 통과한다.
    """

    name: str
    final_loss: float
    grad_norm: float
    n_iters: int
    from_task_start: bool = True

    @property
    def is_critical_point(self) -> bool:
        """수렴했다고 볼 수 있는가. gradient norm 기준이다."""
        return math.isfinite(self.grad_norm) and self.grad_norm < 1.0e-5


@dataclass(frozen=True, slots=True)
class AchievableCeiling:
    """``(spec, seed)`` 하나의 달성 가능 상한.

    Attributes:
        initial_loss: ``L0``.
        reference_loss: ``L_ref``. 참조 run 중 최소 final loss.
        loss_floor: 수치 하한 (``L0 x RELATIVE_LOSS_FLOOR``).
        runs: 참조 run 전체. 어느 solver 가 최소를 냈는지 남긴다.
    """

    initial_loss: float
    reference_loss: float
    loss_floor: float
    runs: tuple[ReferenceRun, ...] = field(default_factory=tuple)

    @property
    def effective_floor(self) -> float:
        """상한을 정하는 실제 하한. 국소최소점과 수치 하한 중 큰 쪽이다."""
        return max(self.reference_loss, self.loss_floor)

    @property
    def nats(self) -> float:
        """``J_achievable = log(L0) − log(max(L_ref, L_floor))``."""
        if self.initial_loss <= 0.0 or self.effective_floor <= 0.0:
            return float("nan")
        return math.log(self.initial_loss) - math.log(self.effective_floor)

    @property
    def limited_by(self) -> str:
        """상한을 정한 원인. 보고할 때 이 구분이 중요하다."""
        if not math.isfinite(self.reference_loss):
            return "reference_failed"
        if self.reference_loss > self.loss_floor:
            return "critical_point"
        return "numerical_floor"

    @property
    def start_runs(self) -> tuple[ReferenceRun, ...]:
        """task 시작점에서 출발한 run 만. ``L_ref`` 의 근거다."""
        return tuple(r for r in self.runs if r.from_task_start)

    @property
    def off_start_runs(self) -> tuple[ReferenceRun, ...]:
        """다른 초기화에서 출발한 run. **진단용이며 상한을 올리지 않는다.**"""
        return tuple(r for r in self.runs if not r.from_task_start)

    @property
    def off_start_best(self) -> float:
        values = [
            r.final_loss for r in self.off_start_runs if math.isfinite(r.final_loss)
        ]
        return min(values) if values else float("nan")

    @property
    def start_basin_is_suboptimal(self) -> bool:
        """다른 초기화가 시작점 basin 보다 유의미하게 더 좋은 점을 찾았는가.

        `True` 면 이 task 의 시작점은 국소최소점의 basin 이다. 컨트롤러 비교에는
        영향이 없지만 **결과 해석에 반드시 함께 보고해야 한다.** "방법이 전역
        최적에 도달했다" 고 쓰면 틀린 주장이 된다.
        """
        best_off = self.off_start_best
        if not math.isfinite(best_off) or not math.isfinite(self.reference_loss):
            return False
        if best_off <= 0.0 or self.reference_loss <= 0.0:
            return best_off < self.reference_loss
        return math.log(self.reference_loss) - math.log(best_off) > REFERENCE_AGREEMENT_NAT

    @property
    def n_converged(self) -> int:
        return sum(1 for r in self.start_runs if r.is_critical_point)

    @property
    def reference_spread_nat(self) -> float:
        """**수렴한** 참조 run 들의 final loss 가 log 축에서 얼마나 벌어졌는가.

        미수렴 run 을 포함하면 "느린 solver" 와 "다른 임계점에 갇힌 solver" 를
        구별할 수 없다. 실측에서 micro-neural 의 Adam/SGD 가 예산 안에 수렴하지
        못해 산포가 32 nat 로 나왔지만, LBFGS 가 하한에 도달했으므로 상한 추정
        자체는 모호하지 않았다. 따라서 수렴한 run 만 본다.
        """
        values = [
            r.final_loss
            for r in self.start_runs
            if r.is_critical_point and math.isfinite(r.final_loss) and r.final_loss > 0.0
        ]
        if len(values) < 2:
            return 0.0
        return math.log(max(values)) - math.log(min(values))

    @property
    def references_agree(self) -> bool:
        """수렴한 참조 solver 들이 비슷한 값에 도달했는가.

        크게 벌어지면 어떤 solver 는 더 나쁜 임계점에 갇혔다는 뜻이고, 상한 추정이
        불안정하다. 단 `numerical_floor` 로 제한된 경우는 floor cap 때문에 값이
        갈리므로 이 검사를 적용하지 않는다.
        """
        if self.limited_by != "critical_point":
            return True
        return self.reference_spread_nat <= REFERENCE_AGREEMENT_NAT

    def describe(self) -> str:
        best = min(
            (r for r in self.start_runs if math.isfinite(r.final_loss)),
            key=lambda r: r.final_loss,
            default=None,
        )
        who = best.name if best is not None else "없음"
        text = (
            f"L0={self.initial_loss:.4e} L_ref={self.reference_loss:.6e} "
            f"({who}) floor={self.loss_floor:.3e} "
            f"J_achievable={self.nats:.4f} nat 제한={self.limited_by} "
            f"참조 산포={self.reference_spread_nat:.3f} nat"
        )
        if self.start_basin_is_suboptimal:
            text += (
                f"  **시작점 basin 이 전역최적 아님** "
                f"(다른 초기화 최선={self.off_start_best:.3e})"
            )
        return text


@dataclass(frozen=True, slots=True)
class SeedVariationReport:
    """seed 가 실제로 다른 인스턴스를 만드는지 (D23 원인 3).

    `RosenbrockSpec(dimension=5)` 는 `randomize_start=False` 가 기본이라 seed
    2/3/4 가 같은 인스턴스였다. `n=3` 이 실제로는 `n=1` 이었고 CI 와 p-value 가
    전부 무의미했다. calibration 단계에서 자동으로 걸러야 한다.
    """

    n_seeds: int
    n_distinct_initial_loss: int
    n_distinct_start_point: int

    @property
    def ok(self) -> bool:
        return (
            self.n_distinct_initial_loss == self.n_seeds
            and self.n_distinct_start_point == self.n_seeds
        )

    def describe(self) -> str:
        verdict = "정상" if self.ok else "**seed 복제**"
        return (
            f"seed {self.n_seeds}개 -> 서로 다른 L0 {self.n_distinct_initial_loss}개, "
            f"서로 다른 시작점 {self.n_distinct_start_point}개  {verdict}"
        )


def _flat_params(task) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in task.params])


def _grad_norm(task) -> float:
    loss = task.loss()
    grads = torch.autograd.grad(loss, list(task.params), allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().pow(2).sum())
    return math.sqrt(total)


def _run_lbfgs(task, *, max_iter: int) -> ReferenceRun:
    params = list(task.params)
    opt = torch.optim.LBFGS(
        params,
        max_iter=max_iter,
        tolerance_grad=1.0e-16,
        tolerance_change=1.0e-18,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        value = task.loss()
        value.backward()
        return value

    opt.step(closure)
    return ReferenceRun(
        name="lbfgs",
        final_loss=float(task.loss().detach()),
        grad_norm=_grad_norm(task),
        n_iters=max_iter,
    )


def _run_first_order(task, *, name: str, lr: float, steps: int) -> ReferenceRun:
    params = list(task.params)
    opt = (
        torch.optim.Adam(params, lr=lr)
        if name.startswith("adam")
        else torch.optim.SGD(params, lr=lr, momentum=0.9)
    )
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        value = task.loss()
        value.backward()
        opt.step()
    return ReferenceRun(
        name=name,
        final_loss=float(task.loss().detach()),
        grad_norm=_grad_norm(task),
        n_iters=steps,
    )


def _run_damped_newton(task, *, steps: int) -> ReferenceRun:
    """긴 예산의 damped Newton. explicit Hessian solve 를 쓴다.

    HVP + CG 구현과 독립적인 대조군이어야 하므로 직접 선형계를 푼다. 차원이
    커지면 비싸므로 `d <= 512` 에서만 호출한다.
    """
    params = list(task.params)
    if len(params) != 1:
        return ReferenceRun("newton", float("nan"), float("nan"), 0)
    x = params[0]
    n = x.numel()
    damping = 1.0e-8
    for _ in range(steps):
        loss = task.loss()
        (grad,) = torch.autograd.grad(loss, x, create_graph=True)
        rows = []
        for i in range(n):
            (row,) = torch.autograd.grad(grad.reshape(-1)[i], x, retain_graph=True)
            rows.append(row.detach().reshape(-1))
        hess = torch.stack(rows)
        hess = 0.5 * (hess + hess.T)
        g = grad.detach().reshape(-1)
        if float(g.norm()) < 1.0e-14:
            break
        eye = torch.eye(n, dtype=hess.dtype, device=hess.device)
        # Hessian 을 양정으로 만들 만큼 damping 을 올린다.
        shift = damping
        for _ in range(60):
            try:
                step = torch.linalg.solve(hess + shift * eye, -g)
                if bool(torch.isfinite(step).all()):
                    break
            except RuntimeError:
                pass
            shift = max(shift * 10.0, 1.0e-12)
        else:
            break
        before = float(loss.detach())
        with torch.no_grad():
            scale = 1.0
            for _ in range(40):
                trial = x.detach() + scale * step.reshape(x.shape)
                x.copy_(trial)
                after = float(task.loss().detach())
                if math.isfinite(after) and after <= before:
                    break
                scale *= 0.5
            else:
                break
    return ReferenceRun(
        name="newton",
        final_loss=float(task.loss().detach()),
        grad_norm=_grad_norm(task),
        n_iters=steps,
    )


def reference_panel(
    make_task: Callable[[], object],
    *,
    max_iter: int = REFERENCE_MAX_ITERS,
    newton_dim_limit: int = 512,
    extra_inits: Sequence[float] = (),
) -> tuple[ReferenceRun, ...]:
    """planner 를 제외한 강한 참조 방법들을 긴 예산으로 돌린다.

    Args:
        make_task: 매번 **새 task 인스턴스**를 주는 팩토리. 각 solver 가 독립된
            파라미터에서 출발해야 한다.
        max_iter: LBFGS 반복 상한.
        newton_dim_limit: explicit Newton 을 쓸 최대 차원.
        extra_inits: 추가 초기화 스케일. 시작점을 이 값으로 채워 다른 basin 을
            시험한다. 예: ``(0.0, 0.5)``.

    Returns:
        참조 run 들. 실패한 solver 도 ``final_loss=nan`` 으로 포함한다.
    """
    runs: list[ReferenceRun] = []

    task = make_task()
    runs.append(_run_lbfgs(task, max_iter=max_iter))

    task = make_task()
    runs.append(_run_first_order(task, name="adam", lr=1.0e-2, steps=max_iter // 2))

    task = make_task()
    runs.append(_run_first_order(task, name="sgd_momentum", lr=1.0e-4, steps=max_iter // 2))

    task = make_task()
    if sum(p.numel() for p in task.params) <= newton_dim_limit:
        runs.append(_run_damped_newton(task, steps=200))

    # 다른 초기화는 **진단용**이다. `from_task_start=False` 로 표시해 `L_ref` 에
    # 들어가지 않게 한다. 컨트롤러는 task 시작점에서만 출발하므로 다른 basin 의
    # 최적값은 도달 가능한 상한이 아니다.
    for scale in extra_inits:
        task = make_task()
        if hasattr(task, "move_to"):
            n = sum(p.numel() for p in task.params)
            task.move_to(
                torch.full((n,), float(scale), dtype=task.params[0].dtype)
            )
            run = _run_lbfgs(task, max_iter=max_iter)
            runs.append(
                ReferenceRun(
                    name=f"lbfgs@init{scale:g}",
                    final_loss=run.final_loss,
                    grad_norm=run.grad_norm,
                    n_iters=run.n_iters,
                    from_task_start=False,
                )
            )
    return tuple(runs)


def achievable_ceiling(
    initial_loss: float, runs: Sequence[ReferenceRun]
) -> AchievableCeiling:
    """참조 run 들로부터 달성 가능 상한을 만든다.

    `L_ref` 는 **task 시작점에서 출발한 run 중 최소 final loss** 다. 다른 초기화의
    결과는 진단으로만 쓰고 상한을 올리지 않는다.

    수렴하지 않은 run 도 값이 유한하면 후보에 넣는다. 상한을 과소평가하는 쪽이
    안전하기 때문이다.
    """
    floor = max(
        torch.finfo(torch.float64).tiny, abs(initial_loss) * RELATIVE_LOSS_FLOOR
    )
    finite = [
        r.final_loss
        for r in runs
        if r.from_task_start and math.isfinite(r.final_loss)
    ]
    reference = min(finite) if finite else float("nan")
    return AchievableCeiling(
        initial_loss=initial_loss,
        reference_loss=reference,
        loss_floor=floor,
        runs=tuple(runs),
    )


def check_seed_variation(
    make_task_for_seed: Callable[[int], object], seeds: Sequence[int]
) -> SeedVariationReport:
    """seed 가 실제로 다른 인스턴스를 만드는지 확인한다 (D23 원인 3).

    `initial_loss` 만 보면 우연히 같을 수 있으므로 시작점 벡터도 함께 본다.
    """
    losses: set[float] = set()
    starts: set[tuple[float, ...]] = set()
    for seed in seeds:
        task = make_task_for_seed(seed)
        losses.add(round(float(task.initial_loss), 12))
        flat = _flat_params(task).to(dtype=torch.float64)
        starts.add(tuple(round(float(v), 12) for v in flat[: min(64, flat.numel())]))
    return SeedVariationReport(
        n_seeds=len(seeds),
        n_distinct_initial_loss=len(losses),
        n_distinct_start_point=len(starts),
    )
