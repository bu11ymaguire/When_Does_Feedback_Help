"""대상 하드웨어에서 grad-equivalent(GE) 환산 계수를 실측한다 (프로토콜 D1).

주 비용 지표가 wall-clock 이 아니라 GE 인 이유는 작은 모델에서 GPU 시간이
FLOP 대신 커널 런치 오버헤드에 지배되기 때문이다. 이 스크립트는 그 사실을
직접 확인하고, 벤치마크가 사용할 환산 계수를 산출한다.

사용법:

    # 진단: 배치 크기별로 오버헤드 지배 구간을 확인한다
    uv run python scripts/measure_cost_model.py --model mnist_mlp

    # 실측값을 config 로 저장한다 (Stage 1 산출물)
    uv run python scripts/measure_cost_model.py --model mnist_mlp \
        --batch-size 512 --save configs/cost_model.mnist_mlp.yaml

    # 데이터셋 없이 quadratic 로 빠르게 확인
    uv run python scripts/measure_cost_model.py --model quadratic --device cpu

측정에는 실제 데이터가 필요하지 않다. 비용은 텐서 shape 에만 의존하므로
같은 shape 의 난수를 쓴다. 데이터셋 다운로드 없이 실행할 수 있다.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from rl_newton.benchmark.cost_model import CostModel, measure_cost_model
from rl_newton.utils.seed import seed_everything

Problem = tuple[Sequence[Tensor], Callable[[], Tensor]]


# ---------------------------------------------------------------------------
# 측정 대상 구성
# ---------------------------------------------------------------------------


def build_mnist_mlp(batch_size: int, device: str) -> Problem:
    """MNIST MLP 784-128-10. 약 101,770 파라미터."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.Tanh(),
        nn.Linear(128, 10),
    ).to(device)
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    return params, lambda: nn.functional.cross_entropy(model(x), y)


def build_deep_mlp(batch_size: int, device: str) -> Problem:
    """더 깊은 MLP 784-256-256-128-10. meta-test 용 (프로토콜 Stage 5)."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 128),
        nn.Tanh(),
        nn.Linear(128, 10),
    ).to(device)
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    return params, lambda: nn.functional.cross_entropy(model(x), y)


def build_small_cnn(batch_size: int, device: str) -> Problem:
    """CIFAR-10 small CNN. 수백만 파라미터 규모.

    프로토콜 D1: 작은 MLP 에서는 wall-clock 이 오버헤드 지배라 해석이 불가하다.
    FLOP 지배 구간에서 결론이 유지되는지 확인하려면 이 규모가 필요하다.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.Tanh(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.Tanh(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256),
        nn.Tanh(),
        nn.Linear(256, 10),
    ).to(device)
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    return params, lambda: nn.functional.cross_entropy(model(x), y)


def build_quadratic(batch_size: int, device: str) -> Problem:
    """SPD quadratic ``L = 0.5 x^T A x``. batch_size 를 차원으로 해석한다."""
    d = batch_size
    m = torch.randn(d, d, device=device)
    a = m @ m.T + d * torch.eye(d, device=device)
    x = nn.Parameter(torch.randn(d, device=device))
    return [x], lambda: 0.5 * x @ (a @ x)


BUILDERS: dict[str, Callable[[int, str], Problem]] = {
    "mnist_mlp": build_mnist_mlp,
    "deep_mlp": build_deep_mlp,
    "small_cnn": build_small_cnn,
    "quadratic": build_quadratic,
}

DEFAULT_BATCH_GRID: dict[str, tuple[int, ...]] = {
    "mnist_mlp": (256, 512, 1024, 4096),
    "deep_mlp": (256, 512, 1024, 4096),
    "small_cnn": (32, 64, 128, 256),
    "quadratic": (100, 200, 500, 1000),
}


# ---------------------------------------------------------------------------
# 보고
# ---------------------------------------------------------------------------

_HEADER = (
    f"{'batch':>7} {'n_params':>10} {'t_fwd':>8} {'t_grad':>8} {'t_hvp':>8} "
    f"{'c_hvp':>7} {'c_fwd':>7} {'GE(k=10)':>9} {'regime':>13}"
)


def _row(cm: CostModel, batch: int) -> str:
    regime = "launch-bound" if cm.is_launch_bound else "flop-bound"
    return (
        f"{batch:>7} {cm.n_params:>10,} {cm.t_fwd_ms:>8.3f} {cm.t_grad_ms:>8.3f} "
        f"{cm.t_hvp_ms:>8.3f} {cm.c_hvp:>7.2f} {cm.c_fwd:>7.2f} "
        f"{cm.newton_step_ge(10):>9.1f} {regime:>13}"
    )


def _scaling_diagnosis(results: list[tuple[int, CostModel]]) -> str:
    """배치 크기가 2배 될 때 시간이 2배 되는지로 오버헤드 지배를 판정한다.

    시간이 배치에 비례하지 않으면 연산이 아니라 고정 오버헤드가 지배한다는
    직접적인 증거다. 프로토콜 D1 의 근거를 실측으로 확인한다.
    """
    lines = ["", "배치 스케일링 진단 (시간이 배치에 비례하면 FLOP 지배):"]
    for (b0, c0), (b1, c1) in zip(results, results[1:], strict=False):
        batch_ratio = b1 / b0
        time_ratio = c1.t_grad_ms / c0.t_grad_ms
        efficiency = time_ratio / batch_ratio
        verdict = "FLOP 지배" if efficiency > 0.7 else "오버헤드 지배"
        lines.append(
            f"  batch {b0:>5} -> {b1:>5} (x{batch_ratio:.1f}) : "
            f"시간 x{time_ratio:.2f}  (비례계수 {efficiency:.2f})  {verdict}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GE 환산 계수 실측 (프로토콜 D1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(BUILDERS), default="mnist_mlp")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="지정하면 이 배치만 측정한다. 생략하면 진단용 배치 grid 를 훑는다.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--curvature-batch-ratio", type=float, default=1.0, help="B_c / B_g (프로토콜 D2)"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="측정 결과를 YAML 로 저장한다. --batch-size 와 함께 쓴다.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 를 쓸 수 없다. --device cpu 로 실행하라.")
        return 1
    if args.save is not None and args.batch_size is None:
        print("--save 는 --batch-size 와 함께 써야 한다 (계수는 배치 구성에 종속).")
        return 1
    if not 0.0 < args.curvature_batch_ratio <= 1.0:
        print(f"--curvature-batch-ratio 는 (0, 1] 범위여야 한다: {args.curvature_batch_ratio}")
        return 1

    seed_everything(args.seed)
    builder = BUILDERS[args.model]
    batches = (args.batch_size,) if args.batch_size else DEFAULT_BATCH_GRID[args.model]

    device_label = args.device
    if args.device == "cuda":
        device_label = f"cuda ({torch.cuda.get_device_name(0)})"
    print(f"model={args.model}  device={device_label}  B_c/B_g={args.curvature_batch_ratio}")
    print(f"warmup={args.warmup}  repeat={args.repeat}  (중앙값, CUDA 동기화 포함)")
    print()
    print(_HEADER)
    print("-" * len(_HEADER))

    results: list[tuple[int, CostModel]] = []
    for batch in batches:
        params, loss_fn_grad = builder(batch, args.device)

        loss_fn_curv = None
        if args.curvature_batch_ratio < 1.0:
            curv_batch = max(1, int(batch * args.curvature_batch_ratio))
            # curvature batch 는 gradient batch 와 같은 모델을 공유해야 한다.
            # builder 가 모델을 새로 만들므로 여기서는 같은 파라미터를 쓰는
            # 별도 배치 클로저가 필요하다. quadratic 은 배치 개념이 없어 제외한다.
            if args.model == "quadratic":
                print("  (quadratic 에는 curvature batch 개념이 없어 비율을 무시한다)")
            else:
                _, loss_fn_curv = builder(curv_batch, args.device)

        curv_batch_size = (
            None
            if args.curvature_batch_ratio == 1.0
            else max(1, int(batch * args.curvature_batch_ratio))
        )
        cm = measure_cost_model(
            params,
            loss_fn_grad,
            loss_fn_curv,
            model_id=f"{args.model}_b{batch}_{args.device}",
            grad_batch_size=batch,
            curvature_batch_size=curv_batch_size or batch,
            n_warmup=args.warmup,
            n_repeat=args.repeat,
        )
        print(_row(cm, batch))
        results.append((batch, cm))

    if len(results) > 1:
        print(_scaling_diagnosis(results))

    launch_bound = [b for b, cm in results if cm.is_launch_bound]
    if launch_bound:
        print()
        print(
            f"경고: batch {launch_bound} 는 런치 오버헤드 지배 구간이다.\n"
            "      이 구간의 wall-clock 결론은 단독 근거로 쓸 수 없다.\n"
            "      프로토콜 D1 에 따라 GE 기준 결론을 주로 삼고, wall-clock 은\n"
            "      FLOP 지배 규모(small_cnn)에서 별도 검증한다."
        )

    if args.save is not None:
        cm = results[0][1]
        path = cm.save(args.save)
        print()
        print(f"저장: {path}")
        print(f"  c_grad_graph={cm.c_grad_graph:.3f}  c_hvp={cm.c_hvp:.3f}  c_fwd={cm.c_fwd:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
