"""컨트롤러의 행동 공간.

모든 컨트롤러(fixed / heuristic / open_loop / oracle / RL)가 **같은** 타입의
행동 공간에서 고른다. 비교의 공정성이 코드 구조로 보장된다 (프로토콜 D4).

```text
MultiDiscrete([n_damping, n_budget, n_step_size])
```

damping 축의 두 모드
--------------------
``relative`` (기본)
    ``lambda <- lambda * m``. damping 은 step 사이에 누적되는 지속 상태다.
    실제 정책이 쓰는 방식이며, 도달성 제약이 있다.

``absolute`` (분석 전용)
    ``lambda <- v``. 현재 값과 무관하게 즉시 이동한다. 도달성 제약이 없는
    오라클을 만들기 위한 것이다. 학습 정책은 이 모드를 쓰지 않는다.

프리셋 세 개가 프로토콜 게이트 A/B를 나눈다
--------------------------------------------
```text
ABSOLUTE - best_static  = 내재적 one-step 헤드룸       (게이트 A)
ABSOLUTE - WIDE         = 배수 전이/도달성 손실        (게이트 B)
WIDE     - NARROW       = 행동 범위가 좁아 생기는 손실 (게이트 B)
```

셋 다 greedy 오라클과 함께 쓰면 여전히 **one-step 헤드룸**만 측정한다.
장기 의사결정의 필요성은 별도로 look-ahead 오라클로 본다 (게이트 C).

왜 정확한 역수쌍인가
--------------------
README 원안은 ``{0.3, 1.0, 3.0}`` 인데 ``3 x 0.3 = 0.9 != 1`` 이다. 배수를
번갈아 고르면 damping 이 step 당 10%씩 아래로 표류한다. 30 step 이면 ``0.9^15``
배, 약 0.2배로 밀린다. 컨트롤러가 "올렸다 되돌리기"를 할 수 없다는 뜻이다.

정확한 역수쌍 ``{1/3, 1, 3}`` 이면 ``3 x (1/3) = 1`` 이 정확하므로 표류가 없다.
로그 공간에서도 대칭이다. 프로토콜 §9에 변경 이유를 기록했다.

step_size 는 왜 분리 가능한가
-----------------------------
``(H + lambda I) p = -g`` 를 푸는 데 step_size 는 쓰이지 않는다. 따라서

1. 전수 탐색에서 서로 다른 step_size 는 **CG 결과를 공유**한다
   (``iter_solve_groups``). naive 대비 6배 이상 절약된다.
2. ``with_fixed_step_size()`` 로 damping x CG budget 축만 분석할 수 있다.

2번이 중요한 이유는 **action aliasing** 이다. damping 이 큰 구간에서는
``(H + lambda I)^{-1} g ~ g / lambda`` 이므로 update 가 ``-(alpha/lambda) g`` 로
근사된다. 즉 ``(lambda=1e4, alpha=1)`` 과 ``(lambda=1e5, alpha=10)`` 이 거의
같은 결과를 낸다. 서로 다른 action 이 같은 결과를 내면 어떤 축이 이득을
만들었는지 귀속시킬 수 없고, RL 학습도 어려워진다. 그래서 헤드룸 측정은
step_size 고정 조건을 **기본**으로 하고, 제어 추가 조건을 부가로 돌린다.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from rl_newton.types import ControllerAction

__all__ = [
    "ActionSpace",
    "DampingMode",
    "NARROW",
    "WIDE",
    "ABSOLUTE",
    "PRESETS",
    "LATTICE_BASE",
    "LATTICE_STEP_LOG10",
]

DampingMode = Literal["relative", "absolute"]


@dataclass(frozen=True, slots=True)
class ActionSpace:
    """이산 행동 공간. 세 축의 곱집합이다.

    Attributes:
        name: 프리셋 이름. 로그와 결과 표에 기록된다.
        damping_values: ``relative`` 면 배수, ``absolute`` 면 damping 값 자체.
        cg_budgets: 허용할 CG 최대 반복 수들.
        step_sizes: Newton 방향에 적용할 step size 들.
        damping_mode: ``relative`` (정책용) 또는 ``absolute`` (분석용).
    """

    name: str
    damping_values: tuple[float, ...]
    cg_budgets: tuple[int, ...]
    step_sizes: tuple[float, ...]
    damping_mode: DampingMode = "relative"

    def __post_init__(self) -> None:
        if not self.damping_values:
            raise ValueError("damping_values must not be empty")
        if not self.cg_budgets:
            raise ValueError("cg_budgets must not be empty")
        if not self.step_sizes:
            raise ValueError("step_sizes must not be empty")
        if any(v <= 0.0 for v in self.damping_values):
            raise ValueError(f"damping values must be > 0: {self.damping_values}")
        if any(b < 1 for b in self.cg_budgets):
            raise ValueError(f"cg budgets must be >= 1: {self.cg_budgets}")
        if any(s <= 0.0 for s in self.step_sizes):
            raise ValueError(f"step sizes must be > 0: {self.step_sizes}")
        if self.damping_mode not in ("relative", "absolute"):
            raise ValueError(f"unknown damping_mode: {self.damping_mode!r}")

    # --- 크기 -------------------------------------------------------------

    @property
    def nvec(self) -> tuple[int, int, int]:
        """``gymnasium.spaces.MultiDiscrete`` 에 넘길 차원."""
        return (len(self.damping_values), len(self.cg_budgets), len(self.step_sizes))

    def __len__(self) -> int:
        """전체 조합 수. 프로토콜 D5의 탐색 예산 기준값이다."""
        d, b, s = self.nvec
        return d * b * s

    @property
    def n_solve_groups(self) -> int:
        """서로 다른 CG solve 의 수. ``damping x budget``."""
        d, b, _ = self.nvec
        return d * b

    @property
    def hvp_per_sweep(self) -> int:
        """전수 탐색 1회에 드는 HVP 수. step_size 공유 효과가 반영된 값이다."""
        return len(self.damping_values) * sum(self.cg_budgets)

    @property
    def is_absolute(self) -> bool:
        return self.damping_mode == "absolute"

    @property
    def log10_span(self) -> float:
        """damping 값의 로그 범위. 도달성 논의의 기준이다."""
        logs = [math.log10(v) for v in self.damping_values]
        return max(logs) - min(logs)

    # --- 변환 -------------------------------------------------------------

    def _make_action(self, d: int, b: int, s: int) -> ControllerAction:
        value = self.damping_values[d]
        if self.is_absolute:
            return ControllerAction(
                damping_multiplier=1.0,
                cg_budget=self.cg_budgets[b],
                step_size=self.step_sizes[s],
                damping_absolute=value,
            )
        return ControllerAction(
            damping_multiplier=value,
            cg_budget=self.cg_budgets[b],
            step_size=self.step_sizes[s],
        )

    def action_from_indices(self, indices: Sequence[int]) -> ControllerAction:
        """``MultiDiscrete`` 인덱스를 ``ControllerAction`` 으로 바꾼다.

        Raises:
            ValueError: 길이가 3이 아니거나 인덱스가 범위를 벗어난 경우.
        """
        if len(indices) != 3:
            raise ValueError(f"expected 3 indices, got {len(indices)}")
        d, b, s = (int(i) for i in indices)
        nd, nb, ns = self.nvec
        if not (0 <= d < nd and 0 <= b < nb and 0 <= s < ns):
            raise ValueError(f"indices {tuple(indices)} out of range for nvec {self.nvec}")
        return self._make_action(d, b, s)

    def indices_from_flat(self, flat: int) -> tuple[int, int, int]:
        """단일 categorical 인덱스를 세 축 인덱스로 분해한다."""
        if not 0 <= flat < len(self):
            raise ValueError(f"flat index {flat} out of range [0, {len(self)})")
        _, nb, ns = self.nvec
        s = flat % ns
        b = (flat // ns) % nb
        d = flat // (ns * nb)
        return d, b, s

    def action_from_flat(self, flat: int) -> ControllerAction:
        """단일 categorical 인덱스를 ``ControllerAction`` 으로 바꾼다."""
        return self.action_from_indices(self.indices_from_flat(flat))

    # --- 열거 -------------------------------------------------------------

    def iter_actions(self) -> Iterator[ControllerAction]:
        """전체 조합을 결정론적 순서로 순회한다."""
        for flat in range(len(self)):
            yield self.action_from_flat(flat)

    def iter_solve_groups(self) -> Iterator[tuple[ControllerAction, tuple[float, ...]]]:
        """``(대표 action, step_sizes)`` 를 순회한다.

        같은 그룹의 step_size 들은 **하나의 CG 결과를 공유**한다. 대표 action 은
        그룹의 damping 과 budget 을 담고 step_size 는 첫 값이다. 전수 탐색
        구현은 이 순회를 써야 한다. 그러지 않으면 동일한 선형계를
        step_size 개수만큼 반복해서 푼다.
        """
        nd, nb, _ = self.nvec
        for d in range(nd):
            for b in range(nb):
                yield self._make_action(d, b, 0), self.step_sizes

    # --- 변형 -------------------------------------------------------------

    def with_fixed_step_size(self, step_size: float = 1.0) -> ActionSpace:
        """step_size 축을 단일값으로 고정한 변형을 반환한다.

        damping x CG budget 만 제어하는 조건이다. action aliasing 을 피하고
        어떤 축이 이득을 만들었는지 귀속시키기 위해 헤드룸 측정의 **기본**
        조건으로 쓴다 (모듈 docstring 참조).
        """
        if step_size not in self.step_sizes:
            raise ValueError(
                f"step_size {step_size} not in {self.step_sizes}; "
                "고정값은 원래 공간의 부분집합이어야 비교가 공정하다"
            )
        return replace(self, name=f"{self.name}+fixed_a{step_size:g}", step_sizes=(step_size,))

    def with_budgets(self, budgets: Sequence[int]) -> ActionSpace:
        """CG budget 축을 교체한 변형. look-ahead 오라클 비용 절감에 쓴다."""
        return replace(self, name=f"{self.name}+k{len(budgets)}", cg_budgets=tuple(budgets))

    def __repr__(self) -> str:
        return (
            f"ActionSpace({self.name!r}, mode={self.damping_mode}, nvec={self.nvec}, "
            f"n_actions={len(self)}, hvp_per_sweep={self.hvp_per_sweep})"
        )


# ---------------------------------------------------------------------------
# 프리셋
# ---------------------------------------------------------------------------

NARROW = ActionSpace(
    name="narrow",
    damping_values=(1.0 / 3.0, 1.0, 3.0),
    cg_budgets=(3, 5, 10, 20),
    step_sizes=(0.25, 0.5, 1.0),
)
"""프로토콜 원안 (README §4.4). 36 조합, CG solve 12회, sweep 당 114 HVP.

README 의 ``0.3`` 을 정확한 ``1/3`` 로 바꿨다. ``3 x 0.3 = 0.9`` 라서 배수를
번갈아 고르면 damping 이 아래로 표류하기 때문이다. 프로토콜 §9 참조.

도달성: ``log10`` 범위 ``0.95``. ``1e-2 -> 1e6`` 은 ``x3`` 을 17회 연속 골라야
한다 (``1e-2 x 3^17 = 1.29e6``). 30 step episode 의 절반 이상이다.
"""

WIDE = ActionSpace(
    name="wide",
    damping_values=tuple(3.0**e for e in range(-3, 4)),
    cg_budgets=(3, 5, 10, 20),
    step_sizes=(0.25, 0.5, 1.0),
)
"""damping 배수를 넓힌 프리셋. ``{1/27, 1/9, 1/3, 1, 3, 9, 27}``.

3의 거듭제곱이므로 **역수 대칭이면서 로그 균등**이다. 두 성질이 모두 필요하다.

- 역수 대칭: ``3 x (1/3) = 1`` 이 정확해 damping 표류가 없다
- 로그 균등: 간격이 모두 ``log10(3)`` 이므로 ``NARROW`` 와 해상도가 같다

초기 구성 ``{1/30, 1/10, 1/3, 1, 3, 10, 30}`` 은 역수 대칭이었으나 로그
균등이 아니었다. ``1/10 -> 1/3`` 이 3배가 아니라 3.33배여서 간격이
``0.477, 0.523, 0.477, ...`` 로 흔들렸다. 게이트 B는 세 공간의 해상도를
통제한 상태에서 **범위 차이만** 재야 하므로 이것을 고정해야 한다.

``x27`` 이면 ``1e-2 -> 1e6`` 에 6 step 이면 도달한다 (``27^6 = 3.9e8``).
"""

LATTICE_BASE = 3.0
"""세 프리셋이 공유하는 damping 격자의 밑.

``NARROW``, ``WIDE``, ``ABSOLUTE`` 의 damping 값이 모두 ``3^e`` 형태다.
따라서 로그 해상도가 정확히 같고, ``NARROW`` 와 ``WIDE`` 의 배수 집합은
``ABSOLUTE`` 의 값 집합과 같은 격자 위에 있다. 게이트 B가 **범위 차이만**
재려면 이 정렬이 필요하다.
"""

LATTICE_STEP_LOG10 = math.log10(LATTICE_BASE)
"""격자의 로그 간격. ``log10(3) ~ 0.477``."""

ABSOLUTE = ActionSpace(
    name="absolute",
    damping_values=tuple(LATTICE_BASE**e for e in range(-16, 17)),
    cg_budgets=(3, 5, 10, 20),
    step_sizes=(0.25, 0.5, 1.0),
    damping_mode="absolute",
)
"""도달성 제약이 없는 **분석 전용** 프리셋. ``{3^-16 ... 3^16}``, 33점.

현재 damping 과 무관하게 지정값으로 즉시 이동한다. 학습 정책은 이 모드를
쓰지 않는다. 프로토콜 게이트 A1/B의 기준이다.

해상도를 반드시 맞춘다
----------------------
게이트 B는 ``ABSOLUTE`` 와 ``WIDE`` / ``NARROW`` 의 격차를 "도달성 손실" 로
해석한다. 그 해석이 성립하려면 **로그 해상도가 같아야** 한다.

초기 구성은 세 번 틀렸다.

```text
1차  {1e-6, 1e-4, ..., 1e6}       7점,  2 decade 간격   -> narrow 대비 4배 거침
2차  {1e-6, ..., 1e6}            13점,  1 decade 간격   -> narrow 대비 2배 거침
3차  [-8, 8] 을 log10(3) 간격     34점,  마지막 간격만 0.255 -> 상한 강제 삽입 탓
현재 {3^-16 ... 3^16}            33점,  모든 간격 log10(3) -> narrow 와 동일
```

1차 구성에서는 absolute 쪽이 범위가 4배 넓은데도 narrow 보다 **나쁜** 결과를
냈다. 도달성 이득이 해상도 손실에 잠식된 것이다. 그 상태로는 게이트 B가
두 효과를 분리하지 못한다.

범위는 ``[2.3e-8, 4.3e7]`` 로 optimizer 기본 경계 ``[1e-8, 1e8]`` 안에 들어간다.
경계에서 클립되면 서로 다른 action 이 같은 damping 으로 붕괴하므로 피한다.

``ABSOLUTE`` 와 ``WIDE`` 의 차이가 "배수 전이와 도달성 때문에 잃는 양"이다.
후보 수가 많으므로 Stage 2 기본 조건은 ``with_fixed_step_size()`` 로
``damping x CG budget`` 만 본다. H=3, 5 planning 에는 쓰지 않는다 —
absolute 는 damping ramp-up 자체를 없애므로 게이트 C의 질문과 무관하고
비용도 감당할 수 없다.
"""

PRESETS: dict[str, ActionSpace] = {
    NARROW.name: NARROW,
    WIDE.name: WIDE,
    ABSOLUTE.name: ABSOLUTE,
}
