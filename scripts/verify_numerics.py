"""Stage 1 게이트 검증. HVP와 Newton-CG의 수치 정확성을 실제 수치로 보고한다.

프로토콜 §4 Stage 1 게이트:
  1. explicit Hessian 대조 상대오차 < 1e-5 (FP32 ill-conditioned는 1e-4)
  2. SPD quadratic에서 Newton-CG 해의 상대오차 < 1e-3
  3. damping 증가 시 ill-conditioned 문제의 CG 실패 감소
  4. indefinite 문제에서 negative curvature 정상 탐지

테스트는 통과/실패만 알려준다. 이 스크립트는 **얼마나 정확한가**를 숫자로
남긴다. 하드웨어나 torch 버전이 바뀌었을 때 여유가 얼마나 남았는지 알 수 있다.

사용법:

    uv run python scripts/verify_numerics.py
    uv run python scripts/verify_numerics.py --device cuda --dtype float64
"""

from __future__ import annotations

import argparse

import torch

from rl_newton.curvature.hvp import HvpGraph
from rl_newton.curvature.operators import DampedHessianOperator
from rl_newton.solvers.conjugate_gradient import conjugate_gradient
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask
from rl_newton.tasks.rosenbrock import RosenbrockSpec, RosenbrockTask
from rl_newton.utils.seed import seed_everything

DTYPES = {"float32": torch.float32, "float64": torch.float64}

# 게이트 임계값 (프로토콜 §4 Stage 1)
GATE_HVP = 1.0e-5
GATE_HVP_ILL = 1.0e-4
GATE_NEWTON = 1.0e-3


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = float(expected.norm())
    if denom == 0.0:
        return float(actual.norm())
    return float((actual - expected).norm() / denom)


def _operator(task, damping: float) -> DampedHessianOperator:
    return DampedHessianOperator.from_closure(
        task.loss, task.params, damping=damping, min_damping=0.0, max_damping=1e12
    )


# ---------------------------------------------------------------------------
# 게이트 1: HVP 정확성
# ---------------------------------------------------------------------------


def gate_hvp(device: str, dtype: torch.dtype) -> bool:
    print("[게이트 1] HVP vs explicit Hessian")
    print(f"  {'문제':<34} {'상대오차':>12} {'임계값':>10}  판정")

    rows: list[tuple[str, float, float]] = []
    for kappa in (1.0e1, 1.0e2, 1.0e3, 1.0e5):
        threshold = GATE_HVP_ILL if kappa >= 1.0e5 else GATE_HVP
        seed_everything(0)
        task = QuadraticTask(
            QuadraticSpec(dimension=64, condition_number=kappa),
            seed=0,
            device=device,
            dtype=dtype,
        )
        with HvpGraph(task.loss, task.params) as graph:
            v = torch.randn(graph.numel, device=device, dtype=dtype)
            err = relative_error(graph.matvec(v), task.hessian_matrix() @ v)
        rows.append((f"quadratic kappa={kappa:.0e}", err, threshold))

    for d in (2, 10):
        task = RosenbrockTask(RosenbrockSpec(dimension=d), seed=0, device=device, dtype=dtype)
        with HvpGraph(task.loss, task.params) as graph:
            v = torch.randn(graph.numel, device=device, dtype=dtype)
            err = relative_error(graph.matvec(v), task.hessian_matrix() @ v)
        rows.append((f"rosenbrock d={d}", err, GATE_HVP))

    ok = True
    for name, err, threshold in rows:
        passed = err < threshold
        ok &= passed
        print(f"  {name:<34} {err:>12.3e} {threshold:>10.0e}  {'PASS' if passed else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# 게이트 2: Newton-CG 해가 explicit Newton solve와 일치
# ---------------------------------------------------------------------------


def gate_newton_direction(device: str, dtype: torch.dtype) -> bool:
    print("\n[게이트 2] Newton-CG 방향 vs explicit linear solve")
    print(f"  {'kappa':>8} {'iters':>6} {'resid ratio':>12} {'상대오차':>12} {'임계값':>10}  판정")

    ok = True
    for kappa in (1.0e1, 1.0e2, 1.0e3, 1.0e4):
        seed_everything(0)
        task = QuadraticTask(
            QuadraticSpec(dimension=64, condition_number=kappa),
            seed=0,
            device=device,
            dtype=dtype,
        )
        op = _operator(task, 0.0)
        result = conjugate_gradient(op, -op.grad, max_iters=2000, tolerance=1e-8)
        op.release()

        err = relative_error(result.solution, task.exact_newton_step())
        passed = err < GATE_NEWTON and result.converged
        ok &= passed
        print(
            f"  {kappa:>8.0e} {result.iterations:>6} {result.residual_ratio:>12.3e} "
            f"{err:>12.3e} {GATE_NEWTON:>10.0e}  {'PASS' if passed else 'FAIL'}"
        )
    return ok


# ---------------------------------------------------------------------------
# 게이트 3: damping이 ill-conditioned 문제의 CG 실패를 줄인다
# ---------------------------------------------------------------------------


def gate_damping_helps(device: str, dtype: torch.dtype) -> bool:
    print("\n[게이트 3] damping이 CG 실패를 줄인다 (kappa=1e6, d=100, 8 seeds)")
    print(f"  {'damping':>9} {'예산=5':>9} {'예산=10':>9} {'예산=20':>9} {'중앙 resid':>12}")

    spec = QuadraticSpec(kind="ill_conditioned", dimension=100, condition_number=1.0e6)
    budgets = (5, 10, 20)
    dampings = (1e-8, 1e2, 1e4, 1e6, 1e8)
    table: dict[float, tuple[list[float], float]] = {}

    for damping in dampings:
        rates: list[float] = []
        residuals: list[float] = []
        for budget in budgets:
            converged = 0
            for seed in range(8):
                task = QuadraticTask(spec, seed=seed, device=device, dtype=dtype)
                op = _operator(task, damping)
                result = conjugate_gradient(op, -op.grad, max_iters=budget, tolerance=1e-3)
                op.release()
                converged += int(result.converged)
                if budget == 10:
                    residuals.append(result.residual_ratio)
            rates.append(converged / 8)
        residuals.sort()
        table[damping] = (rates, residuals[len(residuals) // 2])

    for damping, (rates, median_resid) in table.items():
        cells = " ".join(f"{r:>9.2f}" for r in rates)
        print(f"  {damping:>9.0e} {cells} {median_resid:>12.3e}")

    # 판정: 예산별 수렴률이 damping에 단조 비감소이고, 최대 damping에서 1.0
    ok = True
    for i, budget in enumerate(budgets):
        series = [table[d][0][i] for d in dampings]
        monotone = series == sorted(series)
        reaches_full = series[-1] == 1.0
        ok &= monotone and reaches_full
        print(
            f"  예산 {budget:>2}: 단조={'O' if monotone else 'X'} "
            f"최대damping수렴률={series[-1]:.2f}  {'PASS' if monotone and reaches_full else 'FAIL'}"
        )
    return ok


# ---------------------------------------------------------------------------
# 게이트 4: negative curvature 탐지
# ---------------------------------------------------------------------------


def gate_negative_curvature(device: str, dtype: torch.dtype) -> bool:
    print("\n[게이트 4] negative curvature 탐지")
    print(f"  {'문제':<28} {'lambda_min':>12} {'damping':>10} {'탐지':>6} {'수렴':>6}  판정")

    ok = True

    for kappa in (1.0e2, 1.0e4):
        spec = QuadraticSpec(
            kind="indefinite", dimension=64, condition_number=kappa, negative_fraction=0.2
        )
        task = QuadraticTask(spec, seed=0, device=device, dtype=dtype)
        lam_min = task.min_eigenvalue

        # damping 없음 -> 탐지되어야 한다
        op = _operator(task, 0.0)
        undamped = conjugate_gradient(op, -op.grad, max_iters=64, tolerance=1e-6)
        op.release()

        # damping 충분 -> SPD가 되어 정상 수렴해야 한다
        task.reset()
        op = _operator(task, -lam_min * 2.0)
        damped = conjugate_gradient(op, -op.grad, max_iters=2000, tolerance=1e-6)
        op.release()

        detected = undamped.negative_curvature
        recovered = damped.converged and not damped.negative_curvature
        finite_direction = bool(torch.isfinite(undamped.solution).all()) and (
            float(undamped.solution.norm()) > 0.0
        )
        passed = detected and recovered and finite_direction
        ok &= passed
        print(
            f"  {'indefinite kappa=' + f'{kappa:.0e}':<28} {lam_min:>12.3e} {0.0:>10.0e} "
            f"{'O' if detected else 'X':>6} {'-':>6}  {'PASS' if passed else 'FAIL'}"
        )
        print(
            f"  {'  + damping = 2|lambda_min|':<28} {'':>12} {-lam_min * 2.0:>10.3e} "
            f"{'X' if not damped.negative_curvature else 'O':>6} "
            f"{'O' if damped.converged else 'X':>6}"
        )

    # Rosenbrock 골짜기 위쪽
    task = RosenbrockTask(RosenbrockSpec(dimension=2), seed=0, device=device, dtype=dtype)
    task.move_to(task.negative_curvature_point(x0=0.0))
    lam_min = float(torch.linalg.eigvalsh(task.hessian_matrix()).min())
    op = _operator(task, 0.0)
    result = conjugate_gradient(op, -op.grad, max_iters=10, tolerance=1e-6)
    op.release()

    passed = result.negative_curvature and lam_min < 0.0
    ok &= passed
    print(
        f"  {'rosenbrock (골짜기 위)':<28} {lam_min:>12.3e} {0.0:>10.0e} "
        f"{'O' if result.negative_curvature else 'X':>6} {'-':>6}  "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return ok


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1 수치 게이트 검증 (프로토콜 §4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 를 쓸 수 없다. --device cpu 로 실행하라.")
        return 1

    dtype = DTYPES[args.dtype]
    label = args.device
    if args.device == "cuda":
        label = f"cuda ({torch.cuda.get_device_name(0)})"
    print(f"device={label}  dtype={args.dtype}  torch={torch.__version__}")
    print("=" * 92)

    results = {
        "HVP 정확성": gate_hvp(args.device, dtype),
        "Newton-CG 방향": gate_newton_direction(args.device, dtype),
        "damping 효과": gate_damping_helps(args.device, dtype),
        "negative curvature": gate_negative_curvature(args.device, dtype),
    }

    print("\n" + "=" * 92)
    print("Stage 1 게이트 요약")
    for name, passed in results.items():
        print(f"  {name:<24} {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\n결과: {'전체 통과' if all_passed else '실패 항목 있음'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
