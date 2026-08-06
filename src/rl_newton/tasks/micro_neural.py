"""Micro-neural task. **feedback 의 가치가 있는 regime 을 만드는 것이 목적이다** (D24).

왜 이 task 가 필요한가
----------------------
D22 에서 결정론적 quadratic 의 헤드룸이 feedback 이 아니라 sequence 에 있다는 것이
확인됐다 (`C3 = −0.044 nat, p=0.38`). 초기 상태에서 한 번 계획하고 맹목 실행하는
`committed` 가 매 step 재계획하는 `shrinking` 과 같거나 나았다.

quadratic 은 결정론적이라 초기 상태와 행동 시퀀스가 궤적을 완전히 결정한다. 즉
planner 의 내부 모델이 정확했다. 따라서 두 해석을 구별할 수 없다.

```text
해석 1  feedback 은 원래 가치가 없다
해석 2  이 task 족이 예측 가능해서 feedback 이 필요 없었다
```

핵심 질문은 "모델이 비선형인가" 가 아니다 (D24).

```text
초기 계획 시점에 미래 상태를 정확히 예측할 수 없는가
```

그래서 **같은 모델과 데이터**를 두 regime 으로 나눈다.

```text
[R1] full_batch            전체 데이터로 gradient 와 HVP. 결정론적
[R2] controlled_stochastic 고정 seed 의 batch 시퀀스. gradient/HVP batch 가 step 마다 변함
```

`R2` 에서만 관측 상태가 초기 계획의 예상과 달라질 수 있다. `C3` 를 두 regime 에서
비교하면 `feedback 의 가치`와 `예측 가능성`을 분리할 수 있다.

**두 regime 은 평가 목적함수가 같다.** `loss()` 는 항상 전체 데이터로 계산한다.
달라지는 것은 optimizer 가 보는 표본이다. 그래야 Track E 의 `logΔ` 가 regime 간에
비교 가능하다.

optimizer 인터페이스의 실제 동작
--------------------------------
`HvpGraph` 는 `curvature_loss` 를 **정확히 한 번** 호출해 gradient 그래프를
붙잡는다. 따라서 gradient, HVP, 그리고 step 시작 loss 가 **모두 같은 표본**에서
나온다. 별도의 gradient 표본 훅은 없다.

```text
curvature_loss()  optimizer 가 보는 것. gradient + HVP + loss_before
loss()            평가용. 항상 full dataset. Track E 점수에만 쓴다
```

이 구조는 R2 에 그대로 맞는다. optimizer 는 minibatch 로 step 을 정하고, 점수는
전체 데이터로 매긴다.

batch 는 **실제 step 에서만** 전진한다
--------------------------------------
`advance_batch()` 는 기록되는 실제 step 뒤에만 호출된다. planner 의 look-ahead
시뮬레이션 중에는 전진하지 않는다.

```text
전진함    실제 step. 다음 step 은 다음 batch 를 본다
전진 안 함  planner 시뮬레이션. 현재 batch 로 미래를 예측한다
```

이것이 의도된 설계다. planner 가 미래 batch 를 미리 보면 데이터에 대한 oracle
지식을 갖게 되어 R2 가 다시 예측 가능해진다. 그러면 feedback 의 가치를 시험하려던
목적이 무너진다.

따라서 R2 에서 `committed` 는 batch 0 만 보고 전체 계획을 세우고, `shrinking` 은
매 step 현재 batch 로 재계획한다. **C3 가 바로 이 차이를 잰다.**

데이터를 다운로드하지 않는다
----------------------------
외부 데이터셋을 받으면 네트워크 의존과 재현성 문제가 생긴다. 대신 **고정된 teacher
network** 로 라벨을 만든다. 결정론적이고, 선형 분리 불가능하며, 학생 모델의 구조와
독립적으로 잡을 수 있다.

```text
x ~ N(0, I)                          입력
y = argmax teacher(x)                라벨. teacher 는 학생과 다른 난수 스트림
```

teacher 를 학생보다 넓게 두어 학생이 teacher 함수를 정확히 표현할 수 없게 한다.

**다만 그것이 floor saturation 을 막아주지는 않는다.** 유한한 데이터셋은 과매개화된
신경망이 암기할 수 있다. 실측(`n=512`, 4869 파라미터)에서 `onestep` 이 정확도
1.000, loss `3.6e-9` 에 도달했다. 수치 하한 `4.1e-14` 보다는 위여서 150 GE 예산에서는
포화가 발생하지 않았지만, **구조적으로 보장된 것이 아니다.**

측정 가능성은 참조 solver panel 로 판정한다 (D25). `label_noise` 나 teacher 폭을
근거로 삼지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from rl_newton.utils.seed import torch_generator

__all__ = ["MicroNeuralSpec", "MicroNeuralTask", "Regime"]

Regime = Literal["full_batch", "controlled_stochastic"]


@dataclass(frozen=True, slots=True)
class MicroNeuralSpec:
    """Micro-neural 인스턴스 명세.

    Attributes:
        input_dim: 입력 차원.
        hidden_dim: 은닉층 폭. 파라미터 수를 지배한다.
        n_classes: 분류 클래스 수.
        n_samples: 데이터셋 크기. 고정 subset 이다.
        regime: ``full_batch`` 또는 ``controlled_stochastic``.
        batch_size: ``controlled_stochastic`` 에서 optimizer 가 보는 표본 크기.
            gradient 와 HVP 가 같은 batch 를 쓴다 (``HvpGraph`` 구조상 분리 불가).
        teacher_hidden_dim: teacher 폭. 학생보다 넓게 두어 표현 불가능하게 만든다.
        label_noise: 라벨을 무작위로 뒤집을 비율. 문제를 덜 매끄럽게 만든다.

            **이것은 floor saturation 을 막지 못한다.** 서로 다른 `x` 에 붙은
            뒤집힌 라벨은 과매개화된 신경망이 그대로 암기할 수 있다. 실측에서
            `n=512`, 4869 파라미터, `label_noise=0.05` 일 때 `onestep` 이 정확도
            1.000 에 도달했다. 측정 가능성은 참조 solver panel 로 판정한다 (D25).
        init_scale: 학생 초기화 스케일 배수.
    """

    input_dim: int = 32
    hidden_dim: int = 128
    n_classes: int = 5
    n_samples: int = 512
    regime: Regime = "full_batch"
    batch_size: int = 64
    teacher_hidden_dim: int = 256
    label_noise: float = 0.05
    init_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self.input_dim}")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {self.n_classes}")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if self.regime not in ("full_batch", "controlled_stochastic"):
            raise ValueError(f"unknown regime: {self.regime!r}")
        if self.regime == "controlled_stochastic" and not 1 <= self.batch_size <= self.n_samples:
            raise ValueError(
                f"batch_size must be in [1, {self.n_samples}], got {self.batch_size}"
            )
        if not 0.0 <= self.label_noise < 0.5:
            raise ValueError(f"label_noise must be in [0, 0.5), got {self.label_noise}")
        if self.init_scale <= 0.0:
            raise ValueError(f"init_scale must be > 0, got {self.init_scale}")

    @property
    def n_parameters(self) -> int:
        """학생 파라미터 수. 2-layer MLP with bias."""
        first = self.input_dim * self.hidden_dim + self.hidden_dim
        second = self.hidden_dim * self.n_classes + self.n_classes
        return first + second

    @property
    def data_key(self) -> str:
        """데이터셋과 초기 파라미터를 결정하는 키. **regime 을 포함하지 않는다.**

        D24 는 "모델과 데이터는 하나만 고정한다" 를 요구한다. 두 regime 이 다른
        데이터를 받으면 `C3` 의 regime 간 비교가 데이터 차이에 오염된다.

        ```text
        같아야 함   데이터셋, teacher, 초기 파라미터
        달라도 됨   optimizer 가 보는 표본 (batch 시퀀스)
        ```
        """
        return (
            f"mlp|{self.input_dim}|{self.hidden_dim}|{self.n_classes}"
            f"|{self.n_samples}|{self.teacher_hidden_dim}"
            f"|{self.label_noise:.6f}|{self.init_scale:.6f}"
        )

    def instance_id(self, seed: int) -> str:
        """저장소에서 run 을 구별하는 이름. regime 을 포함한다.

        `data_key` 와 달리 여기에는 regime 이 들어간다. 같은 데이터의 두 regime 이
        서로 다른 run 으로 기록되어야 하기 때문이다.
        """
        tag = "fb" if self.regime == "full_batch" else f"cs{self.batch_size}"
        return (
            f"mlp_d{self.input_dim}_h{self.hidden_dim}_c{self.n_classes}"
            f"_n{self.n_samples}_{tag}_seed{seed}"
        )


class MicroNeuralTask:
    """고정 teacher 라벨을 맞추는 2-layer MLP.

    `loss()` 는 **항상 전체 데이터**로 계산한다 (평가용). optimizer 가 보는
    gradient / HVP 표본만 regime 에 따라 달라진다.

    Args:
        spec: 문제 명세.
        seed: 실험 조건 식별자 (프로토콜 D7). 난수 시드가 아니라 조건 이름이다.
        device: 텐서 디바이스.
        dtype: FP32 또는 FP64.

    Example:
        >>> spec = MicroNeuralSpec(input_dim=8, hidden_dim=16, n_classes=3, n_samples=64)
        >>> task = MicroNeuralTask(spec, seed=0, dtype=torch.float64)
        >>> spec.n_parameters
        195
        >>> task.optimal_loss
        0.0
        >>> bool(task.initial_loss > 0.0)
        True
    """

    def __init__(
        self,
        spec: MicroNeuralSpec,
        seed: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"dtype must be float32 or float64, got {dtype}")

        self._spec = spec
        self._seed = seed
        self._device = torch.device(device)
        self._dtype = dtype

        # 데이터와 초기 파라미터는 **regime 을 포함하지 않는 키**로 만든다 (D24).
        # 두 regime 이 같은 문제를 풀어야 C3 비교가 성립한다.
        data_key = spec.data_key
        self._inputs, self._labels = self._build_dataset(spec, seed, data_key)
        self._initial_params = self._build_initial_params(spec, seed, data_key)
        self._params = [torch.nn.Parameter(p.clone()) for p in self._initial_params]

        # batch 순서는 optimizer 와 독립된 스트림에서 미리 뽑는다 (프로토콜 D7).
        # optimizer 가 난수를 얼마나 쓰든 같은 (spec, seed) 면 같은 순서를 본다.
        self._batch_stream = self._build_batch_stream(spec, seed, spec.instance_id(seed))
        self._step_index = 0

        self._initial_loss = float(self.loss().detach())

    # --- 구성 -------------------------------------------------------------

    def _build_dataset(  # noqa: C901
        self, spec: MicroNeuralSpec, seed: int, data_key: str
    ) -> tuple[Tensor, Tensor]:
        gen = torch_generator(seed, "micro_neural", "data", data_key)
        inputs = torch.randn(
            spec.n_samples, spec.input_dim, generator=gen, dtype=torch.float64
        )

        # teacher 는 학생보다 넓다. 학생이 teacher 함수를 정확히 표현할 수 없다.
        # 다만 유한 데이터셋 암기는 여전히 가능하다 (docstring 참조).
        t_gen = torch_generator(seed, "micro_neural", "teacher", data_key)
        h = spec.teacher_hidden_dim
        w1 = torch.randn(spec.input_dim, h, generator=t_gen, dtype=torch.float64)
        w1 = w1 / max(1.0, spec.input_dim**0.5)
        b1 = torch.randn(h, generator=t_gen, dtype=torch.float64) * 0.1
        w2 = torch.randn(h, spec.n_classes, generator=t_gen, dtype=torch.float64)
        w2 = w2 / max(1.0, h**0.5)
        b2 = torch.randn(spec.n_classes, generator=t_gen, dtype=torch.float64) * 0.1
        logits = torch.tanh(inputs @ w1 + b1) @ w2 + b2
        labels = logits.argmax(dim=1)

        if spec.label_noise > 0.0:
            n_flip = int(round(spec.label_noise * spec.n_samples))
            if n_flip > 0:
                idx = torch.randperm(spec.n_samples, generator=t_gen)[:n_flip]
                shift = torch.randint(
                    1, spec.n_classes, (n_flip,), generator=t_gen
                )
                labels = labels.clone()
                labels[idx] = (labels[idx] + shift) % spec.n_classes

        return (
            inputs.to(device=self._device, dtype=self._dtype),
            labels.to(device=self._device),
        )

    def _build_initial_params(
        self, spec: MicroNeuralSpec, seed: int, data_key: str
    ) -> list[Tensor]:
        gen = torch_generator(seed, "micro_neural", "init", data_key)
        d, h, c = spec.input_dim, spec.hidden_dim, spec.n_classes
        # He 초기화. tanh 은닉층이지만 곡률 규모를 안정적으로 두려고 표준값을 쓴다.
        w1 = torch.randn(d, h, generator=gen, dtype=torch.float64) * (2.0 / d) ** 0.5
        b1 = torch.zeros(h, dtype=torch.float64)
        w2 = torch.randn(h, c, generator=gen, dtype=torch.float64) * (2.0 / h) ** 0.5
        b2 = torch.zeros(c, dtype=torch.float64)
        scaled = [t * spec.init_scale for t in (w1, b1, w2, b2)]
        return [t.to(device=self._device, dtype=self._dtype) for t in scaled]

    def _build_batch_stream(
        self, spec: MicroNeuralSpec, seed: int, instance: str
    ) -> tuple[Tensor, ...]:
        """미리 정한 batch 인덱스 시퀀스.

        `full_batch` 에서는 비어 있다. `controlled_stochastic` 에서는 **모든
        컨트롤러가 동일한 시퀀스**를 본다. 이것이 paired design 의 전제다
        (프로토콜 D7).
        """
        if spec.regime == "full_batch":
            return ()
        gen = torch_generator(seed, "micro_neural", "batch_order", instance)
        # 넉넉하게 뽑아 둔다. step 수가 이보다 많으면 순환한다.
        n_batches = 512
        return tuple(
            torch.randperm(spec.n_samples, generator=gen)[: spec.batch_size].to(self._device)
            for _ in range(n_batches)
        )

    # --- task 인터페이스 --------------------------------------------------

    @property
    def spec(self) -> MicroNeuralSpec:
        return self._spec

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def instance_id(self) -> str:
        return self._spec.instance_id(self._seed)

    @property
    def params(self) -> list[Tensor]:
        return self._params

    @property
    def is_bounded_below(self) -> bool:
        """cross-entropy 는 0 이상이다."""
        return True

    @property
    def optimal_loss(self) -> float:
        """이론적 하한. **도달 가능하다는 뜻이 아니다.**

        teacher 가 학생보다 넓고 라벨 노이즈가 있으므로 실제 달성 가능 상한은
        참조 solver panel 로 재야 한다 (D25).
        """
        return 0.0

    @property
    def initial_loss(self) -> float:
        return self._initial_loss

    @property
    def n_samples(self) -> int:
        return self._spec.n_samples

    @property
    def step_index(self) -> int:
        """지금까지 소비한 batch 수. `controlled_stochastic` 진단용이다."""
        return self._step_index

    # --- loss 계열 --------------------------------------------------------

    def _forward(self, inputs: Tensor) -> Tensor:
        w1, b1, w2, b2 = self._params
        return torch.tanh(inputs @ w1 + b1) @ w2 + b2

    def _cross_entropy(self, index: Tensor | None) -> Tensor:
        if index is None:
            inputs, labels = self._inputs, self._labels
        else:
            inputs, labels = self._inputs[index], self._labels[index]
        return torch.nn.functional.cross_entropy(self._forward(inputs), labels)

    def loss(self) -> Tensor:
        """**평가용 목적함수. 항상 전체 데이터다.**

        Track E 의 `logΔ` 가 regime 간에 비교 가능해야 하므로 여기서 표본을
        바꾸지 않는다.
        """
        return self._cross_entropy(None)

    def curvature_loss(self) -> Tensor:
        """**optimizer 가 보는 것.** gradient, HVP, step 시작 loss 가 여기서 나온다.

        `HvpGraph` 가 이 클로저를 정확히 한 번 호출해 gradient 그래프를 붙잡으므로
        세 값이 같은 표본에서 나온다.

        `full_batch` 면 `loss()` 와 같다. `controlled_stochastic` 이면 미리 정한
        batch 시퀀스에서 현재 batch 를 쓴다.
        """
        if self._spec.regime == "full_batch":
            return self._cross_entropy(None)
        index = self._batch_stream[self._step_index % len(self._batch_stream)]
        return self._cross_entropy(index)

    def acceptance_loss(self) -> Tensor:
        """**고정 평가 목적함수.** `fixed_eval` 수락 규칙에서 쓴다 (D28).

        R2 에서 `_accept` 가 minibatch loss 의 단조 감소를 요구하면, 참 목적함수를
        개선하는 step 도 표본 잡음 때문에 거절될 수 있다. 그 교란을 분리하려고
        수락 판정만 **step 마다 바뀌지 않는** 목적함수로 옮긴다.

        gradient 와 HVP 는 계속 minibatch 에서 나온다. 바뀌는 것은 수락 판정뿐이다.

        전체 데이터를 쓴다. `batch_size` 크기의 고정 부분집합보다 참 목적함수에
        가깝고, 이 ablation 의 목적이 "표본 잡음 제거" 이므로 그쪽이 맞다.
        """
        return self._cross_entropy(None)

    @property
    def acceptance_forward_units(self) -> float:
        """`acceptance_loss` 한 번이 control forward 몇 개에 상당하는가.

        **비용을 숨기지 않는다.** 전체 데이터 forward 는 minibatch forward 보다
        `n_samples / batch_size` 배 비싸고, 그 값이 GE 회계에 들어가야 공정한
        비교가 된다.
        """
        if self._spec.regime == "full_batch":
            return 1.0
        return float(self._spec.n_samples) / float(self._spec.batch_size)

    def advance_batch(self) -> None:
        """다음 step 의 batch 로 넘어간다.

        **실제 step 뒤에만 호출된다.** planner 의 look-ahead 시뮬레이션 중에는
        전진하지 않으므로, planner 는 현재 batch 로 미래를 예측한다. 미래 batch 를
        미리 보면 데이터 oracle 이 되어 R2 의 목적이 무너진다.

        `full_batch` 에서는 아무 효과가 없다.
        """
        self._step_index += 1

    def accuracy(self) -> float:
        """전체 데이터 정확도. 보고용이며 최적화 목적이 아니다."""
        with torch.no_grad():
            pred = self._forward(self._inputs).argmax(dim=1)
            return float((pred == self._labels).double().mean())

    # --- 진단 보조 --------------------------------------------------------

    def move_to(self, point: Tensor) -> None:
        """평평한 벡터를 파라미터로 되돌린다. 참조 solver 다중 초기화용이다."""
        expected = self._spec.n_parameters
        if point.numel() != expected:
            raise ValueError(f"point must have {expected} elements, got {point.numel()}")
        offset = 0
        with torch.no_grad():
            for p in self._params:
                n = p.numel()
                p.copy_(point[offset : offset + n].reshape(p.shape).to(p.dtype))
                offset += n

    def reset(self) -> None:
        with torch.no_grad():
            for p, init in zip(self._params, self._initial_params, strict=True):
                p.copy_(init)
        self._step_index = 0

    def __repr__(self) -> str:
        return (
            f"MicroNeuralTask({self.instance_id}, "
            f"params={self._spec.n_parameters}, regime={self._spec.regime}, "
            f"L0={self._initial_loss:.6e})"
        )
