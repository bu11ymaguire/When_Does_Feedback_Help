"""Paired design: ``seed`` 를 실험 조건 식별자로 사용한다 (프로토콜 D7).

``seed`` 는 난수 시드가 아니다. **실험 조건의 이름**이다. ``seed=s`` 일 때
모든 optimizer가 동일한 task 인스턴스, 동일한 초기점, 동일한 minibatch 순서를
본다. 이 전제가 깨지면 쌍별 비교 통계(Wilcoxon signed-rank, 비율 부트스트랩)
전체가 무의미해진다.

왜 굳이 팩토리를 두는가
-----------------------
"각 optimizer가 알아서 같은 시드를 쓰면 되지 않나"로 충분하지 않다. optimizer마다
난수 소비량이 다르기 때문이다. 예를 들어 RL 컨트롤러는 정책 샘플링으로 난수를
소비하고 fixed baseline은 하지 않는다. 전역 시드를 공유하면 그 차이가 task
인스턴스나 배치 순서까지 오염시킨다.

그래서 task 생성용 난수 스트림을 optimizer 실행 스트림과 **완전히 분리**한다.
``utils.seed.torch_generator`` 가 namespace별 독립 스트림을 주고, 이 팩토리는
task를 만들 때 그 스트림만 쓴다. 결과적으로 optimizer가 난수를 얼마나 쓰든
task 인스턴스는 동일하다.

분산 감소 효과는 공짜다. 같은 문제에서 두 optimizer를 비교하면 문제 난이도
차이가 상쇄되므로, 독립 표본 비교보다 훨씬 적은 seed로 같은 검정력을 얻는다.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace

import torch

from rl_newton.tasks.micro_neural import MicroNeuralSpec, MicroNeuralTask
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask
from rl_newton.tasks.rosenbrock import RosenbrockSpec, RosenbrockTask
from rl_newton.utils.seed import spawn_seed

__all__ = [
    "TaskSpec",
    "SyntheticTask",
    "PairedTaskFactory",
    "make_task",
    "quadratic_meta_train_specs",
    "quadratic_meta_test_specs",
]

TaskSpec = QuadraticSpec | RosenbrockSpec | MicroNeuralSpec
SyntheticTask = QuadraticTask | RosenbrockTask | MicroNeuralTask


def make_task(
    spec: TaskSpec,
    seed: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SyntheticTask:
    """``(spec, seed)`` 로부터 task 인스턴스를 결정론적으로 만든다.

    같은 인자를 주면 프로세스와 실행 시점에 무관하게 항상 동일한 인스턴스가
    나온다. 이것이 paired design의 기본 보장이다.

    Args:
        spec: task 명세.
        seed: 실험 조건 식별자.
        device: 텐서 디바이스.
        dtype: FP32 또는 FP64.

    Returns:
        새로 만들어진 task. 호출마다 별개 객체이며 파라미터도 독립이다.

    Raises:
        TypeError: 지원하지 않는 spec 타입.
    """
    if isinstance(spec, QuadraticSpec):
        return QuadraticTask(spec, seed, device=device, dtype=dtype)
    if isinstance(spec, RosenbrockSpec):
        return RosenbrockTask(spec, seed, device=device, dtype=dtype)
    if isinstance(spec, MicroNeuralSpec):
        return MicroNeuralTask(spec, seed, device=device, dtype=dtype)
    raise TypeError(f"unsupported task spec: {type(spec).__name__}")


@dataclass(frozen=True, slots=True)
class PairedTaskFactory:
    """하나의 task suite에 대해 optimizer별로 동일한 인스턴스를 공급한다.

    Args:
        specs: task 명세 목록.
        seeds: 실험 조건 식별자 목록.
        device: 텐서 디바이스.
        dtype: FP32 또는 FP64.

    Example:
        >>> specs = [QuadraticSpec(dimension=10, condition_number=100.0)]
        >>> factory = PairedTaskFactory(specs=specs, seeds=[0, 1])
        >>> len(list(factory.iter_instances()))
        2
        >>> a = factory.build(specs[0], 0)
        >>> b = factory.build(specs[0], 0)
        >>> bool(torch.allclose(a.matrix, b.matrix))
        True
    """

    specs: Sequence[TaskSpec]
    seeds: Sequence[int]
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("specs must not be empty")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"seeds must be unique, got {list(self.seeds)}")

    def build(self, spec: TaskSpec, seed: int) -> SyntheticTask:
        """``(spec, seed)`` 인스턴스를 새로 만든다."""
        return make_task(spec, seed, device=self.device, dtype=self.dtype)

    def iter_instances(self) -> Iterator[tuple[TaskSpec, int]]:
        """``(spec, seed)`` 쌍을 순회한다. 순서는 결정론적이다.

        optimizer 실행 루프는 이 순서를 그대로 따라야 한다. 그래야 로그가
        쌍별로 정렬되어 통계 처리가 단순해진다.
        """
        for spec in self.specs:
            for seed in self.seeds:
                yield spec, seed

    def n_instances(self) -> int:
        return len(self.specs) * len(self.seeds)

    def batch_order_seed(self, spec: TaskSpec, seed: int) -> int:
        """이 인스턴스의 minibatch 순서용 파생 시드.

        Stage 3의 신경망 task에서 사용한다. optimizer가 무엇이든 같은
        ``(spec, seed)`` 면 같은 값이 나오므로 배치 순서가 일치한다.
        """
        return spawn_seed(seed, "batch_order", _spec_key(spec))

    def model_init_seed(self, spec: TaskSpec, seed: int) -> int:
        """이 인스턴스의 모델 초기화용 파생 시드."""
        return spawn_seed(seed, "model_init", _spec_key(spec))


def _spec_key(spec: TaskSpec) -> str:
    """spec을 파생 시드용 문자열 키로 변환한다.

    ``instance_id`` 는 seed를 포함하므로 쓰지 않는다. seed는 ``spawn_seed`` 의
    첫 인자로 이미 들어가 있어 중복되면 스트림이 얽힌다.
    """
    if isinstance(spec, QuadraticSpec):
        return (
            f"quad|{spec.kind}|{spec.dimension}|{spec.condition_number:.6e}"
            f"|{spec.negative_fraction:.4f}|{spec.initial_scale:.4f}"
        )
    if isinstance(spec, RosenbrockSpec):
        return (
            f"rosen|{spec.dimension}|{spec.scale:.4f}"
            f"|{int(spec.randomize_start)}|{spec.start_noise:.4f}"
        )
    if isinstance(spec, MicroNeuralSpec):
        return (
            f"mlp|{spec.input_dim}|{spec.hidden_dim}|{spec.n_classes}"
            f"|{spec.n_samples}|{spec.regime}|{spec.batch_size}"
            f"|{spec.teacher_hidden_dim}|{spec.label_noise:.4f}|{spec.init_scale:.4f}"
        )
    raise TypeError(f"unsupported task spec: {type(spec).__name__}")


# ---------------------------------------------------------------------------
# 프로토콜 Stage 5의 meta-train / meta-test 분포
# ---------------------------------------------------------------------------


def quadratic_meta_train_specs(
    *,
    dimensions: Iterable[int] = (50, 100, 200),
    condition_numbers: Iterable[float] = (1.0e1, 1.0e2, 1.0e3, 1.0e4),
    include_indefinite: bool = True,
) -> list[QuadraticSpec]:
    """meta-train 분포의 quadratic 명세들.

    프로토콜 Stage 5 기준: ``kappa`` 는 ``[1e1, 1e4]``, ``d`` 는 {50, 100, 200}.
    ``include_indefinite`` 는 negative curvature 경로를 학습 분포에 포함할지를
    결정한다. indefinite 인스턴스는 아래로 유계가 아니므로 cost-to-target
    집계에서는 제외해야 한다 (``QuadraticTask.is_bounded_below`` 확인).
    """
    specs: list[QuadraticSpec] = []
    for d in dimensions:
        for kappa in condition_numbers:
            kind = "ill_conditioned" if kappa >= 1.0e4 else "spd"
            specs.append(QuadraticSpec(kind=kind, dimension=d, condition_number=kappa))
    if include_indefinite:
        for d in dimensions:
            specs.append(
                QuadraticSpec(
                    kind="indefinite",
                    dimension=d,
                    condition_number=1.0e3,
                    negative_fraction=0.2,
                )
            )
    return specs


def quadratic_meta_test_specs(
    *,
    dimension: int = 500,
    condition_number: float = 1.0e5,
) -> list[QuadraticSpec]:
    """meta-test 분포. 학습 분포 밖의 조건수와 차원으로 외삽을 시험한다."""
    base = QuadraticSpec(
        kind="ill_conditioned",
        dimension=dimension,
        condition_number=condition_number,
    )
    return [base, replace(base, kind="spd", condition_number=1.0e3)]
