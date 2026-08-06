"""헤드룸 측정과 게이트 A~D 판정 (프로토콜 Stage 2).

이 프로젝트의 분기점이다. RL 스택을 만들기 전에 "적응 제어에 여지가 있는가",
"행동 공간이 병목인가", "장기 의사결정이 필요한가"를 먼저 답한다.

두 트랙 (프로토콜 D9)
---------------------
초판은 단일 헤드룸을 재려 했고 실패했다. 국소 효율 목적(``Δlog L / cost``)으로
매 step 최선을 고른 컨트롤러가 고정 설정보다 cost-to-target 에서 나빴다
(0.967x). 서로 다른 두 최적화 문제를 섞고 있었기 때문이다.

```text
Track E  max log(L_0 / L_B)  s.t.  Σ c ≤ B      헤드룸 = 차이 [nat]
Track T  min Σ c             s.t.  L ≤ τ        헤드룸 = 비율 [배수]
```

두 트랙이 같은 답을 주지 않을 수 있고, **그 불일치 자체가 결과다.**

게이트
------
```text
A  Track E  absolute planner vs best_static      적응 제어의 내재적 여지
B  Track E  absolute vs wide vs narrow           도달성/행동범위 손실
C1 Track E  shrinking vs fresh quota MPC         쿼터 초기화의 시간 불일치
C2 Track E  shrinking vs one-step efficiency     다단계 계획의 가치 (주 판정)
C3 Track E  shrinking vs committed plan          상태 피드백의 추가 가치
D  Track T  best_static vs planner (target별)     cost-to-target 헤드룸
```

pilot / confirmatory (프로토콜 D6)
----------------------------------
예산과 target을 결과를 본 뒤 고치면 사후 선택이 된다. ``phase`` 로 구분하며,
``pilot`` 은 dev seed 로 예산/target 선정에만 쓰고 ``confirmatory`` 는 held-out
seed 로 최종 판정한다.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from rl_newton.benchmark.metrics import (
    RELATIVE_LOSS_FLOOR,
    GroupSummary,
    PairedComparison,
    PairedDelta,
    RunSummary,
    TargetSpec,
    compare_paired,
    compare_paired_delta,
    drop_saturated_pairs,
    median_of,
    saturation_report,
    split_by_task_family,
    summarize_group,
    summarize_run,
)
from rl_newton.benchmark.paired import SyntheticTask, TaskSpec, make_task
from rl_newton.benchmark.store import (
    AGGREGATION_VERSION,
    OPEN_LOOP_SEMANTICS_VERSION,
    OPTIMIZER_SEMANTICS_VERSION,
    PLANNER_SEMANTICS_VERSION,
    SELECTION_SEMANTICS_VERSION,
    TASK_SEMANTICS_VERSION,
    ResultStore,
    RunKey,
    SelectionManifest,
    aggregation_id,
    execution_provenance,
    experiment_id,
    run_semantics_id,
    selection_id,
    sweep_id,
)
from rl_newton.optimizers.action_space import ActionSpace
from rl_newton.optimizers.controllers import (
    BudgetedMPCController,
    CommittedPlanController,
    FixedController,
    HeuristicController,
    OneStepEfficiencyController,
    OpenLoopController,
    ShrinkingQuotaMPCController,
    make_open_loop_controller,
)
from rl_newton.optimizers.newton_cg import (
    Controller,
    NewtonCGConfig,
    NewtonCGOptimizer,
    OptimizationTrace,
)
from rl_newton.tasks.micro_neural import MicroNeuralSpec
from rl_newton.tasks.quadratics import QuadraticSpec
from rl_newton.tasks.rosenbrock import RosenbrockSpec
from rl_newton.types import ControllerAction

__all__ = [
    "HeadroomConfig",
    "GateVerdict",
    "HeadroomReport",
    "Phase",
    "run_controller",
    "search_best_static",
    "search_best_open_loop",
    "run_headroom",
    "spec_kind_label",
]

UTILITY_TOLERANCE = 0.02
"""beam 선택의 상대 효용 허용 오차 (프로토콜 F). **실행 전에 고정된 값이다.**"""

UTILITY_EPSILON = 1.0e-6
"""상대 오차 분모의 하한. ``|J_b - J_4| / max(|J_4|, eps)``.

``J_4`` 가 0 근처면 상대 오차가 폭발한다. 그래서 분모를 ``+ eps`` 가 아니라
``max(|J_4|, eps)`` 로 둔다. 값도 config 에 고정해 사후 조정을 막는다.
"""

DEEP_FRACTION_TOLERANCE = 0.05
"""``chosen_depth > 1`` 비율의 허용 차이 (프로토콜 F).

효용이 비슷해도 깊은 계획을 쓰는 빈도가 달라지면 게이트 C의 해석이 바뀐다.
"크게 다르면 제외"를 사후에 판단하지 않기 위해 수치로 고정한다.
"""

PROTOCOL_VERSION = "stage2-v1"
"""프로토콜 버전. 실험 정체성에 포함된다.

C2 protocol freeze 시 태그와 함께 올린다. confirmatory 중 버그를 발견해
결과를 폐기해야 하면 이 값을 증가시켜 이전 결과와 섞이지 않게 한다.
"""

ControllerFactory = Callable[["SyntheticTask", TargetSpec], Controller]
"""컨트롤러 생성자. task 와 target 을 받는다.

Track T planner 는 절대 target loss 를 알아야 한다. 상대 target ``L/L_0 <= v``
의 절대값은 ``v * L_0`` 이므로 **인스턴스마다 다르다.** 그래서 팩토리가
task 를 받도록 한다.
"""

Phase = Literal["pilot", "challenge", "confirmatory"]

DEV_SEEDS = (0, 1, 2)
"""pilot 국면의 dev seed. 예산/target 선정에만 쓴다."""

CALIBRATION_SEEDS = (0, 1)
"""challenge spec **선정**에만 쓰는 seed (프로토콜 D20).

``scripts/calibrate_challenge.py`` 가 이 seed 로 baseline-only 측정 가능성을
판정했다. 설정 선택에 재사용하면 benchmark tuning 이 된다.
"""

SELECTION_SEEDS = (2, 3, 4)
"""challenge set 에서 **설정(Q, space, beam)을 고르는** seed (프로토콜 D20).

``CALIBRATION_SEEDS`` 와 겹치지 않는다. spec 을 고른 seed 로 설정을 고르면
같은 표본을 두 번 쓰게 된다.
"""

HELD_OUT_SEEDS = tuple(range(100, 110))
"""confirmatory 국면의 held-out seed. 최종 판정에만 쓴다.

calibration(0,1) 과 selection(2,3,4) 모두와 겹치지 않는다. 프로토콜 D20 의 권고
범위는 5~14 였으나 기존에 100~109 로 고정해 두었고 이미 분리 조건을 만족하므로
그대로 유지한다.
"""


@dataclass(frozen=True, slots=True)
class HeadroomConfig:
    """헤드룸 실험 설정."""

    specs: Sequence[TaskSpec]
    seeds: Sequence[int]
    targets: dict[str, dict[str, TargetSpec]]
    """``targets[spec_kind][difficulty]``. difficulty 는 easy/medium/hard."""
    cost_budget_ge: float = 600.0
    """모든 컨트롤러에 동일하게 주는 GE 예산 (Track E의 B)."""
    max_steps: int = 200
    initial_damping: float = 1.0e-2
    tuning_budget: int | None = None
    """N_tune. ``None`` 이면 narrow 행동 공간 크기를 쓴다. 모든 baseline 동일."""
    quotas: Sequence[float] = (1.0, 2.0, 4.0)
    """게이트 C의 미래 GE 쿼터 사다리. ``c_max`` 배수다 (프로토콜 D10).

    ```text
    C0  OneStepEfficiencyController    비율 baseline (별도 계산)
    C1  Q = 1 x c_max
    C2  Q = 2 x c_max
    C3  Q = 4 x c_max
    ```

    이전 판은 ``horizons=(1,3,5)`` 였다. horizon step 수로 나누면 비싼 action 과
    싼 action 의 계획이 서로 다른 비용을 쓰므로 공정 비교가 아니었고, 효용을
    비용으로 나눠 보정하려다 평균 희석 문제가 생겼다. 쿼터로 맞추면 나눗셈이
    필요 없다.
    """
    beam_width: int = 4
    fresh_diagnostic_seeds: int = 1
    """``fresh`` 실행 방식을 몇 개 seed 에서만 돌릴지 (프로토콜 D12).

    ``fresh`` 는 시간 불일치가 확인된 **진단 baseline** 이고 P1~P3 판정에
    쓰지 않는다. 그런데 탐색 비용이 가장 크다 (Q=4 wide 에서 인스턴스당
    1.4M GE). 전체 dev 에 반복할 과학적 가치가 낮으므로 seed 부분집합에서만
    돌려 C1 의 방향이 반복되는지만 확인한다.

    ``0`` 이면 ``fresh`` 를 아예 돌리지 않는다 (beam 8 확인 단계에서 쓴다).
    """
    run_fresh_wide: bool = False
    """``fresh`` 를 wide 행동 공간에서도 돌릴지.

    가장 비싼 조합이고 진단 목적에는 narrow 만으로 충분하다.
    """
    saturated_task_prefixes: Sequence[str] = ("rosen_d2",)
    """포화 **진단 표**를 따로 낼 task (프로토콜 D19).

    ``rosen_d2`` 는 150 GE 에서 여러 컨트롤러가 정확히 ``loss = 0`` 에 도달한다.

    **이것으로 primary 게이트를 정의하지 않는다.** 포화는 task 이름이 아니라
    실제 ``floor_hit`` 으로 발생하며, ``quad_spd`` 도 floor 아래로 내려가
    컨트롤러를 구분하지 못한다. 이 목록은 진단 표 대상을 고르는 데만 쓴다.
    """
    execution_modes: Sequence[str] = ("shrinking", "committed", "fresh")
    """돌릴 실행 방식 (프로토콜 D12). **sweep 커버리지이므로 run 정체성이 아니다.**

    단계별 실행에 쓴다. baseline 만 먼저 확보하려면 빈 튜플을 준다.
    """
    planner_spaces: Sequence[str] = ("narrow", "wide")
    """planner 를 돌릴 행동 공간. sweep 커버리지다."""
    run_track_t: bool = True
    """Track T (cost-to-target) 를 돌릴지. sweep 커버리지다."""
    max_plan_depth: int = 24
    """계획 길이 상한. 계산량 안전장치다.

    쿼터가 아니라 이 값에 걸리면 계획이 쿼터를 다 쓰지 못하므로 사다리 비교가
    훼손된다. ``planner_stats["depth_cap_hit"]`` 로 감시한다 (프로토콜 D10).
    """
    tuning_seed: int = 0
    n_schedule_segments: int = 4
    acceptance_loss: str = "control"
    """``control`` | ``fixed_eval``. 수락 판정 목적함수 (D28).

    기본값이 아닐 때만 ``run_semantics_id`` 에 들어간다. 기존 결과를 보존한다.
    """
    phase: Phase = "pilot"
    primary_difficulty: str = "medium"
    """게이트 D의 주 target 난이도."""
    device: str = "cpu"
    """텐서 디바이스.

    **Stage 2 는 CPU 가 기본이며 그것이 옳다.** 대상은 quadratic(d=32~100)과
    Rosenbrock(d=2~10)뿐이다. Stage 0 실측에서 10만 파라미터 MNIST MLP 조차
    GPU 런치 오버헤드 지배(0.68 ms/gradient)였으므로, d=100 matvec 을 GPU 로
    보내면 순손실이다. GPU 는 Stage 3 이후에만 쓴다.
    """

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("specs must not be empty")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if not self.targets:
            raise ValueError("targets must not be empty")

    def optimizer_config(self) -> NewtonCGConfig:
        return NewtonCGConfig(
            total_steps=self.max_steps,
            cost_budget_ge=self.cost_budget_ge,
            initial_damping=self.initial_damping,
            acceptance_loss=self.acceptance_loss,
        )

    def _space_payload(self, space: ActionSpace) -> dict[str, object]:
        return {
            "mode": space.damping_mode,
            "damping_values": [float(v) for v in space.damping_values],
            "cg_budgets": list(space.cg_budgets),
            "step_sizes": [float(s) for s in space.step_sizes],
        }

    def _core_payload(self) -> dict[str, object]:
        """모든 컨트롤러가 공유하는 실행 의미. optimizer 루프 자체의 설정이다.

        **기본값인 설정은 키를 넣지 않는다.** 새 옵션을 추가할 때 무조건 키를 넣으면
        기존 run 전체의 ``run_semantics_id`` 가 바뀌어 재실행된다. 기본값은 이전과
        같은 의미이므로 해시도 같아야 한다 (D28).
        """
        optimizer = self.optimizer_config()
        payload: dict[str, object] = {
            "protocol_version": PROTOCOL_VERSION,
            "optimizer_semantics": OPTIMIZER_SEMANTICS_VERSION,
            "task_semantics": TASK_SEMANTICS_VERSION,
            "device": self.device,
            "cost_budget_ge": self.cost_budget_ge,
            "max_steps": self.max_steps,
            "initial_damping": self.initial_damping,
            "min_damping": optimizer.min_damping,
            "max_damping": optimizer.max_damping,
            "cg_tolerance": optimizer.cg_tolerance,
            "pap_eps": optimizer.pap_eps,
            "max_loss_increase_ratio": optimizer.max_loss_increase_ratio,
            "safe_fallback": optimizer.safe_fallback,
            "compute_trust_ratio": optimizer.compute_trust_ratio,
        }
        if optimizer.acceptance_loss != "control":
            payload["acceptance_loss"] = optimizer.acceptance_loss
        return payload

    def run_semantics_payload(
        self,
        *,
        controller: str,
        space: ActionSpace | None = None,
        quota: float | None = None,
        uses_target: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """**이 컨트롤러가 실제 쓰는 설정만** 담는다 (프로토콜 D13).

        무관한 설정 변경으로 baseline 이 무효화되지 않게 한다.

        ```text
        best_static / open_loop / heuristic   beam, quota, planner 정책 제외
        one-step efficiency                   quota, suffix 정책 제외
        planner 계열                          quota, beam, max_plan_depth 포함
        Track E                               평가용 target 제외
        Track T                               target 포함
        ```

        Args:
            controller: 컨트롤러 종류 라벨. 튜닝으로 선택된 개별 설정
                (``static[3]`` 등)이 아니라 종류를 넣는다.
            space: 그 컨트롤러가 쓰는 행동 공간. ``None`` 이면 넣지 않는다.
            quota: planner 쿼터 배수. ``None`` 이면 planner 가 아니다.
            uses_target: target 이 **종료 조건**으로 쓰이는가. Track E 는 ``False``.
            extra: 컨트롤러별 추가 의미 (예: ``execution_mode``).
        """
        payload = self._core_payload()
        payload["controller"] = controller
        if space is not None:
            payload["space"] = self._space_payload(space)
        if quota is not None:
            payload["planner_semantics"] = PLANNER_SEMANTICS_VERSION
            payload["quota"] = float(quota)
            payload["beam_width"] = self.beam_width
            payload["max_plan_depth"] = self.max_plan_depth
        if controller == "best_open_loop":
            # open-loop 만 progress 시계를 쓴다. 다른 컨트롤러는 영향받지 않는다 (D17).
            payload["open_loop_semantics"] = OPEN_LOOP_SEMANTICS_VERSION
            payload["progress_clock"] = OpenLoopController._CLOCK
        if uses_target:
            # **이 실행이 실제로 쓰는 spec 종류의 target 만 넣는다.**
            #
            # 초판은 `self.targets` 전체를 넣었다. 그래서 관계없는 task 족의 target
            # 을 추가하기만 해도 **모든 Track T run 의 `run_semantics_id` 가 바뀌어
            # 재실행됐다.** micro-neural target 을 추가했을 때 quadratic held-out 의
            # Track T 240 run 이 무효화됐다. D13 이 막으려던 실패 그대로다.
            #
            # 개별 target 문자열은 `RunKey.target` 에 이미 들어 있으므로 여기서는
            # "이 spec 종류에 어떤 난이도 사다리를 썼는가" 만 고정하면 된다.
            kinds = {spec_kind_label(spec) for spec in self.specs}
            payload["targets"] = {
                kind: {level: target.label for level, target in levels.items()}
                for kind, levels in self.targets.items()
                if kind in kinds
            }
        if extra:
            payload.update(dict(extra))
        return payload

    def sweep_payload(self, *, controllers: Sequence[str]) -> dict[str, object]:
        """이번 실행이 **어떤 run 집합을 요청했는지** (프로토콜 D13).

        ``run_semantics_id`` 와 분리되므로 여기가 바뀌어도 기존 run 을 재사용한다.

        **git commit 과 code_dirty 는 넣지 않는다.** 같은 run 집합을 요청했는데
        문서만 수정해도 ``sweep_id`` 가 달라지면 "어떤 집합을 요청했는가" 라는
        의미가 깨진다. 그것들은 ``execution_provenance`` 로 분리한다.
        """
        return {
            "protocol_version": PROTOCOL_VERSION,
            "phase": self.phase,
            "controllers": sorted(controllers),
            "specs": [str(s) for s in self.specs],
            "seeds": list(self.seeds),
            "quotas": [float(q) for q in self.quotas],
            "beam_width": self.beam_width,
            "fresh_diagnostic_seeds": self.fresh_diagnostic_seeds,
            "run_fresh_wide": self.run_fresh_wide,
            "execution_modes": sorted(self.execution_modes),
            "planner_spaces": sorted(self.planner_spaces),
            "run_track_t": self.run_track_t,
            "tuning_budget": self.tuning_budget,
            "n_schedule_segments": self.n_schedule_segments,
            "tuning_seed": self.tuning_seed,
            "primary_difficulty": self.primary_difficulty,
        }

    def selection_payload(
        self, *, family: str, space: ActionSpace, n_tune: int
    ) -> dict[str, object]:
        """baseline 선택(튜닝) 과정의 정체성 (프로토콜 D16).

        ``best_static`` / ``best_open_loop`` 는 컨트롤러가 아니라 **튜닝 결과**다.
        같은 후보 집합·같은 지표·같은 tie-break 면 선택도 결정론적이므로, 이 해시가
        같으면 저장된 선택 결과를 재사용할 수 있다.
        """
        payload = self._core_payload()
        payload.update(
            {
                "selection_family": family,
                "selection_semantics_version": SELECTION_SEMANTICS_VERSION,
                "space": self._space_payload(space),
                "n_tune": n_tune,
                "tuning_seed": self.tuning_seed,
                "tuning_specs": [str(s) for s in self.specs],
                "tuning_seeds": list(self.seeds),
                "selection_metric": "median_log_improvement",
                "tie_break": "lowest_flat_index",
            }
        )
        if family == "open_loop":
            payload["n_schedule_segments"] = self.n_schedule_segments
            payload["open_loop_semantics"] = OPEN_LOOP_SEMANTICS_VERSION
            payload["progress_clock"] = OpenLoopController._CLOCK
        return payload

    def aggregation_payload(self) -> dict[str, object]:
        """집계 규칙 정체성. 바뀌면 재집계만 한다 (프로토콜 D13/D14).

        `targets` 는 **이 실행이 실제로 쓰는 spec 종류만** 담는다. 전체를 담으면
        무관한 task 족의 target 을 추가하기만 해도 `aggregation_id` 가 바뀐다 (D32).
        run 을 무효화하지는 않지만 보고 라벨이 이유 없이 달라진다.
        """
        kinds = {spec_kind_label(spec) for spec in self.specs}
        return {
            "aggregation_version": AGGREGATION_VERSION,
            "relative_loss_floor": RELATIVE_LOSS_FLOOR,
            "saturated_task_prefixes": sorted(self.saturated_task_prefixes),
            "utility_tolerance": UTILITY_TOLERANCE,
            "utility_epsilon": UTILITY_EPSILON,
            "deep_fraction_tolerance": DEEP_FRACTION_TOLERANCE,
            "targets": {
                kind: {level: target.label for level, target in levels.items()}
                for kind, levels in self.targets.items()
                if kind in kinds
            },
        }

    def identity_payload(
        self,
        spaces: Mapping[str, ActionSpace],
        *,
        code_dirty: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """**구판 통합 정체성.** 신규 코드는 ``run_semantics_payload`` 를 쓴다.

        D13 이전 결과와의 호환 확인용으로만 남긴다. 이 payload 는 sweep 커버리지와
        집계 정책까지 섞여 있어서, 무관한 변경이 모든 run 을 무효화한다.

        **여기에 빠진 항목은 재개 시 낡은 결과를 재사용하게 만든다.**
        beam, horizon, GE 예산, action space 정의(damping 값, CG budget,
        step size), damping 경계, target 정의, protocol 버전을 모두 넣는다.

        ``code_dirty`` 를 포함하는 이유: git commit 만으로는 커밋되지 않은
        변경을 구분할 수 없다.
        """
        optimizer = self.optimizer_config()
        return {
            "protocol_version": PROTOCOL_VERSION,
            "phase": self.phase,
            "device": self.device,
            "cost_budget_ge": self.cost_budget_ge,
            "max_steps": self.max_steps,
            "initial_damping": self.initial_damping,
            "min_damping": optimizer.min_damping,
            "max_damping": optimizer.max_damping,
            "cg_tolerance": optimizer.cg_tolerance,
            "pap_eps": optimizer.pap_eps,
            "max_loss_increase_ratio": optimizer.max_loss_increase_ratio,
            "safe_fallback": optimizer.safe_fallback,
            "compute_trust_ratio": optimizer.compute_trust_ratio,
            "quotas": [float(q) for q in self.quotas],
            "beam_width": self.beam_width,
            "max_plan_depth": self.max_plan_depth,
            "fresh_diagnostic_seeds": self.fresh_diagnostic_seeds,
            "run_fresh_wide": self.run_fresh_wide,
            "tuning_budget": self.tuning_budget,
            "n_schedule_segments": self.n_schedule_segments,
            "tuning_seed": self.tuning_seed,
            "primary_difficulty": self.primary_difficulty,
            "specs": [str(s) for s in self.specs],
            "seeds": list(self.seeds),
            "targets": {
                kind: {level: spec.label for level, spec in levels.items()}
                for kind, levels in self.targets.items()
            },
            "spaces": {
                name: {
                    "mode": space.damping_mode,
                    "damping_values": [float(v) for v in space.damping_values],
                    "cg_budgets": list(space.cg_budgets),
                    "step_sizes": [float(s) for s in space.step_sizes],
                }
                for name, space in spaces.items()
            },
            "code_dirty": code_dirty,
            **dict(extra or {}),
        }

    def target_for(self, spec: TaskSpec, difficulty: str) -> TargetSpec:
        return self.targets[spec_kind_label(spec)][difficulty]

    def difficulties(self) -> list[str]:
        first = next(iter(self.targets.values()))
        return list(first)


def spec_kind_label(spec: TaskSpec) -> str:
    """target 조회용 spec 분류 키.

    **타입으로 명시 분기한다.** 예전에는 ``getattr(spec, "kind", None)`` 이 없으면
    ``"rosenbrock"`` 으로 떨어뜨렸다. 새 spec 을 추가하면 조용히 Rosenbrock target
    을 쓰게 되므로 위험하다.
    """
    if isinstance(spec, QuadraticSpec):
        return str(spec.kind)
    if isinstance(spec, RosenbrockSpec):
        return "rosenbrock"
    if isinstance(spec, MicroNeuralSpec):
        return "micro_neural"
    raise TypeError(f"unsupported task spec: {type(spec).__name__}")


def _is_eligible(task: SyntheticTask) -> bool:
    """집계 대상인지. indefinite 는 아래로 유계가 아니라 제외된다."""
    return bool(getattr(task, "is_bounded_below", True))


def absolute_target_loss(task: SyntheticTask, target: TargetSpec) -> float:
    """상대 target 을 절대 loss 값으로 환산한다.

    ``relative_loss`` 는 ``v * L_0``, ``absolute_loss`` 는 그대로다.
    Track T planner 가 cost-to-go 를 추정할 때 필요하다.
    """
    if target.metric == "absolute_loss":
        return target.value
    return target.value * task.initial_loss


def _action_counts(trace: OptimizationTrace) -> dict[str, int]:
    """선택한 action 의 빈도. 정책 분석용 (README §8 action heatmap)."""
    counts: dict[str, int] = {}
    for record in trace.records:
        key = f"m={record.extra.get('damping_multiplier')},k={record.cg_budget},a={record.step_size:g}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _chosen_depths(controller: Controller) -> dict[str, int] | None:
    """planner 가 채택한 계획 길이의 빈도.

    거의 항상 1이면 horizon 을 늘려도 실질적 가치가 없다는 직접적 증거다
    (프로토콜 게이트 C).
    """
    choices = getattr(controller, "choices", None)
    if not choices:
        return None
    counts: dict[str, int] = {}
    for choice in choices:
        depth = getattr(choice, "chosen_depth", 1)
        counts[str(depth)] = counts.get(str(depth), 0) + 1
    return counts


def _planner_stats(controller: Controller) -> dict[str, float] | None:
    """planner 진단값. 쿼터 사다리 해석에 필요하다 (프로토콜 D10).

    ``depth_cap_hit`` 이 0 이 아니면 계산 상한 때문에 쿼터를 다 쓰지 못한 step 이
    있다는 뜻이므로, 쿼터 사다리 비교가 훼손된다. 조용히 넘기면 게이트 C 결론이
    계산 예산의 부산물이 되므로 반드시 기록한다.

    ``quota_used_fraction`` 은 채택된 계획이 쿼터의 몇 %를 실제로 썼는지다.
    1.0 에 가까우면 쿼터가 실제로 구속하고 있다는 뜻이고, 아주 작으면 쿼터를
    늘려도 planner 가 쓰지 않는다는 뜻이다.
    """
    choices = getattr(controller, "choices", None)
    if not choices:
        return None
    quotas = [c.quota_ge for c in choices if math.isfinite(getattr(c, "quota_ge", float("nan")))]
    if not quotas:
        return None
    used = [
        c.plan_used_ge / c.quota_ge
        for c in choices
        if math.isfinite(getattr(c, "plan_used_ge", float("nan"))) and c.quota_ge > 0.0
    ]
    sims = [float(getattr(c, "n_simulations", 0)) for c in choices]
    n_cap = sum(1 for c in choices if getattr(c, "depth_cap_hit", False))
    stats: dict[str, float] = {
        "quota_ge": quotas[0],
        "depth_cap_hit": n_cap / len(choices),
        "quota_used_fraction": (sum(used) / len(used)) if used else float("nan"),
        "mean_simulations": sum(sims) / len(sims),
        "max_depth_seen": float(max(getattr(c, "chosen_depth", 1) for c in choices)),
    }
    # shrinking 전용. 재계획이 이전 계획을 실제로 바꿨는지 (프로토콜 D15).
    retention = getattr(controller, "suffix_retention_rate", None)
    if retention is not None:
        stats["suffix_retention_rate"] = float(retention)
        stats["n_replans"] = float(getattr(controller, "n_replans", 0))
    windows = getattr(controller, "windows", None)
    if windows is not None:
        stats["windows"] = float(windows)
    return stats


def run_controller(
    config: HeadroomConfig,
    factory: ControllerFactory,
    *,
    label: str,
    exp_id: str,
    difficulty: str | None = None,
    store: ResultStore | None = None,
    verbose: bool = False,
    seeds: Sequence[int] | None = None,
) -> list[RunSummary]:
    """모든 ``(spec, seed)`` 인스턴스에서 컨트롤러를 실행하고 집계한다.

    paired design 이므로 인스턴스는 ``make_task(spec, seed)`` 로 결정론적으로
    만들어진다. 컨트롤러가 난수를 얼마나 쓰든 문제는 동일하다.

    ``store`` 가 주어지면 **재개 가능**하다. 이미 완료된
    ``(controller, task_instance, seed, target)`` 조합은 건너뛰고 저장된 결과를
    쓴다. 각 run 이 끝나는 즉시 기록하므로 프로세스가 끊겨도 손실이 없다.

    예외는 삼키지 않고 ``failed`` 로 기록한 뒤 다음 run 으로 넘어간다. 하나의
    task 가 깨져도 전체 실험이 멈추지 않아야 하고, 실패 사실은 남아야 한다.

    Args:
        difficulty: target 난이도. ``None`` 이면 ``primary_difficulty``.
        store: 재개 가능한 결과 저장소.
        verbose: 건너뛴 run 수를 보고한다.
        seeds: seed 부분집합. ``None`` 이면 ``config.seeds`` 전체.
            **진단 전용 컨트롤러의 계산을 줄이는 데만 쓴다.** paired 비교에
            들어가는 컨트롤러는 반드시 같은 seed 집합을 써야 한다.
    """
    opt_config = config.optimizer_config()
    level = difficulty or config.primary_difficulty
    summaries: list[RunSummary] = []
    n_skipped = 0
    seed_set = config.seeds if seeds is None else seeds

    for spec in config.specs:
        target = config.target_for(spec, level)
        for seed in seed_set:
            task = make_task(spec, seed, device=config.device)
            if not _is_eligible(task):
                continue

            key = RunKey(
                experiment_id=exp_id,
                controller=label,
                task_instance_id=task.instance_id,
                seed=seed,
                target=target.label,
            )
            if store is not None and store.is_completed(key):
                cached = store.get(key)
                if cached is not None and cached.summary is not None:
                    summaries.append(cached.summary)
                    n_skipped += 1
                    continue

            controller = factory(task, target)
            started = time.perf_counter()
            try:
                optimizer = NewtonCGOptimizer(
                    task,
                    controller,
                    opt_config,
                    run_id=f"{label}|{task.instance_id}",
                    seed=seed,
                )
                trace = optimizer.run()
            except Exception as exc:  # noqa: BLE001 - 실패를 기록하고 계속한다
                elapsed = time.perf_counter() - started
                message = f"{type(exc).__name__}: {exc}"
                if store is not None:
                    store.record_failure(key, message, wall_clock_sec=elapsed)
                print(f"  실패 {key.as_str()}: {message}", flush=True)
                continue

            elapsed = time.perf_counter() - started
            trace.controller = label
            # 예산 초과 step 을 잘라 평가한다. 이것 없이는 큰 step 을 고르는
            # 컨트롤러가 최대 한 step 만큼 예산을 공짜로 더 쓴다 (프로토콜 D11).
            summary = summarize_run(trace, target, budget_ge=config.cost_budget_ge)
            summaries.append(summary)
            if store is not None:
                store.record_success(
                    summary,
                    exp_id,
                    wall_clock_sec=elapsed,
                    action_counts=_action_counts(trace),
                    chosen_depths=_chosen_depths(controller),
                    planner_stats=_planner_stats(controller),
                )

    if verbose and n_skipped:
        print(f"  {label}: 캐시에서 {n_skipped}개 재사용", flush=True)
    return summaries


def _relabel(group: GroupSummary, name: str) -> GroupSummary:
    """탐색 우승자의 라벨을 정규화한다. 쌍별 비교가 이름으로 조회하기 때문이다."""
    for run in group.runs:
        run.controller = name
    group.controller = name
    return group


def _rank_key_track_e(group: GroupSummary) -> float:
    """Track E 정렬 키. terminal loss 를 가장 많이 줄인 것이 최고다."""
    value = group.median_log_improvement
    return -value if math.isfinite(value) else math.inf


def search_best_static(
    config: HeadroomConfig,
    space: ActionSpace,
    *,
    n_tune: int,
    exp_id: str,
    store: ResultStore | None = None,
) -> tuple[ControllerAction, GroupSummary, SelectionManifest]:
    """행동 공간 조합을 고정으로 돌려 Track E 기준 최고를 고른다.

    탐색 횟수를 ``n_tune`` 으로 제한한다. 프로토콜 D5의 "모든 컨트롤러에
    동일한 탐색 예산" 원칙을 지키기 위해서다. 행동 공간이 ``n_tune`` 보다
    크면 균등 간격으로 부분집합을 뽑는다.

    ``SelectionManifest`` 를 함께 반환한다 (프로토콜 D16). 후보 점수와 tie-break
    를 남기지 않으면 선택 근거가 사라지고, 나중에 evaluation 결과를 보고
    역추정하게 되어 사후 선택이 된다.
    """
    total = len(space)
    if total <= n_tune:
        indices = list(range(total))
    else:
        stride = total / n_tune
        indices = sorted({int(i * stride) for i in range(n_tune)})

    best_group: GroupSummary | None = None
    best_flat = indices[0]
    best_label = f"static[{indices[0]}]"
    scores: dict[str, float] = {}
    labels: list[str] = []
    tuning_ge = 0.0
    n_instances = 0
    for flat in indices:
        action = space.action_from_flat(flat)
        label = f"static[{flat}]"
        runs = run_controller(
            config,
            lambda _t, _g, a=action: FixedController(a),
            label=label,
            exp_id=exp_id,
            store=store,
        )
        group = summarize_group(runs, controller=label)
        scores[label] = group.median_log_improvement
        labels.append(label)
        tuning_ge += sum(r.total_cost_ge for r in runs if math.isfinite(r.total_cost_ge))
        n_instances = max(n_instances, len(runs))
        if best_group is None or _rank_key_track_e(group) < _rank_key_track_e(best_group):
            best_group = group
            best_flat = flat
            best_label = label

    assert best_group is not None
    action = space.action_from_flat(best_flat)
    manifest = SelectionManifest(
        selection_id=selection_id(
            config.selection_payload(family="static", space=space, n_tune=n_tune)
        ),
        family="static",
        candidate_labels=labels,
        candidate_scores=scores,
        selected_label=best_label,
        selected_config={
            "flat_index": best_flat,
            "damping_multiplier": action.damping_multiplier,
            "cg_budget": action.cg_budget,
            "step_size": action.step_size,
            "damping_absolute": action.damping_absolute,
        },
        tuning_specs=[str(s) for s in config.specs],
        tuning_seeds=list(config.seeds),
        n_tune=n_tune,
        n_candidates=len(indices),
        n_tuning_instances=n_instances,
        tuning_object_ge=tuning_ge,
    )
    return action, best_group, manifest


def search_best_open_loop(
    config: HeadroomConfig,
    space: ActionSpace,
    *,
    n_tune: int,
    exp_id: str,
    store: ResultStore | None = None,
) -> tuple[GroupSummary, SelectionManifest]:
    """progress 만 보는 스케줄을 랜덤 서치한다 (프로토콜 D4).

    탐색 횟수는 ``best_static`` 과 **동일한** ``n_tune`` 이다. 파일럿에서 이
    예산이 12회였는데 우승자가 static 과 완전히 같은 결과를 냈다. 스케줄
    공간을 사실상 탐색하지 못한 것이므로 ``n_tune`` 을 충분히 크게 준다.

    ``SelectionManifest.is_constant_schedule`` 로 open-loop 이 static 으로
    퇴화했는지 확인할 수 있다 (프로토콜 D16).
    """
    rng = random.Random(config.tuning_seed)
    n_seg = config.n_schedule_segments

    best_group: GroupSummary | None = None
    best_label = ""
    best_flats: tuple[int, ...] = ()
    best_breaks: tuple[float, ...] = ()
    scores: dict[str, float] = {}
    labels: list[str] = []
    tuning_ge = 0.0
    n_instances = 0
    best_realized: OpenLoopController | None = None
    for trial in range(n_tune):
        flats = tuple(rng.randrange(len(space)) for _ in range(n_seg))
        cuts = sorted(rng.uniform(0.05, 0.95) for _ in range(n_seg - 1))
        breakpoints = (*cuts, 1.0)
        label = f"open_loop[{trial}]"
        made: list[OpenLoopController] = []

        def _factory(_t, _g, f=flats, b=breakpoints, sink=made):
            ctrl = make_open_loop_controller(space, f, b)
            sink.append(ctrl)
            return ctrl

        runs = run_controller(
            config, _factory, label=label, exp_id=exp_id, store=store
        )
        group = summarize_group(runs, controller=label)
        scores[label] = group.median_log_improvement
        labels.append(label)
        tuning_ge += sum(r.total_cost_ge for r in runs if math.isfinite(r.total_cost_ge))
        n_instances = max(n_instances, len(runs))
        if best_group is None or _rank_key_track_e(group) < _rank_key_track_e(best_group):
            best_group = group
            best_label = label
            best_flats = flats
            best_breaks = breakpoints
            # 스케줄이 실제로 몇 구간까지 실행됐는지 (D17). 캐시 재사용 시
            # 컨트롤러가 생성되지 않으므로 빈 dict 가 될 수 있다.
            best_realized = made[-1] if made else None

    assert best_group is not None
    # 선택된 스케줄 전체를 기록한다. 라벨만 바꾸면 어떤 스케줄이 왜 골라졌는지
    # 사라지고, 모든 구간이 같아 static 으로 퇴화했는지도 알 수 없다 (D16).
    schedule = [
        {
            "until": float(b),
            "flat_index": int(f),
            "damping_multiplier": space.action_from_flat(int(f)).damping_multiplier,
            "cg_budget": space.action_from_flat(int(f)).cg_budget,
            "step_size": space.action_from_flat(int(f)).step_size,
        }
        for f, b in zip(best_flats, best_breaks, strict=True)
    ]
    manifest = SelectionManifest(
        selection_id=selection_id(
            config.selection_payload(family="open_loop", space=space, n_tune=n_tune)
        ),
        family="open_loop",
        candidate_labels=labels,
        candidate_scores=scores,
        selected_label=best_label,
        selected_config={
            "schedule": [
                {k: v for k, v in seg.items() if k != "until"} for seg in schedule
            ],
            "breakpoints": [seg["until"] for seg in schedule],
            "n_segments": n_seg,
        },
        tuning_specs=[str(s) for s in config.specs],
        tuning_seeds=list(config.seeds),
        n_tune=n_tune,
        n_candidates=len(labels),
        n_tuning_instances=n_instances,
        tuning_object_ge=tuning_ge,
        progress_clock=OpenLoopController._CLOCK,
        realized_segment_counts=(
            {str(k): v for k, v in best_realized.realized_segment_counts.items()}
            if best_realized is not None
            else {}
        ),
        realized_ge_by_segment=(
            {str(k): v for k, v in best_realized.realized_ge_by_segment.items()}
            if best_realized is not None
            else {}
        ),
    )
    # constant 로 퇴화했으면 등가 static action 을 실제 값으로 기록한다.
    # open_loop 과 static 의 후보 grid 가 다를 수 있어 라벨 비교는 무의미하다.
    if manifest.is_constant_schedule:
        manifest.equivalent_static_action = dict(manifest.selected_config["schedule"][0])
    return best_group, manifest


@dataclass(slots=True)
class GateVerdict:
    """게이트 하나의 판정."""

    name: str
    track: str
    question: str
    statistic: float
    unit: str
    go_threshold: float
    pivot_threshold: float
    detail: str = ""
    nonsaturated: str = ""
    """비포화 쌍만으로 계산한 같은 통계 (프로토콜 D14).

    **민감도 분석 전용이다.** 주 판정은 floor-capped 전체 쌍으로 한다. 두 값의
    결론이 다르면 "쉬운 인스턴스의 포화 처리에 민감하다" 고 보고한다.
    """

    @property
    def verdict(self) -> str:
        if not math.isfinite(self.statistic):
            return "판정불가"
        if self.statistic >= self.go_threshold:
            return "GO"
        if self.statistic < self.pivot_threshold:
            return "재설계"
        return "조건부"

    def describe(self) -> str:
        return (
            f"[{self.name}] ({self.track}) {self.question}\n"
            f"    {self.statistic:+.3f} {self.unit}  "
            f"(GO >= {self.go_threshold:g}, 재설계 < {self.pivot_threshold:g})  "
            f"-> {self.verdict}"
            + (f"\n    비포화 민감도: {self.nonsaturated}" if self.nonsaturated else "")
            + (f"\n    {self.detail}" if self.detail else "")
        )


@dataclass(slots=True)
class HeadroomReport:
    """헤드룸 실험 전체 결과."""

    phase: Phase = "pilot"
    groups: dict[str, GroupSummary] = field(default_factory=dict)
    track_e_deltas: list[PairedDelta] = field(default_factory=list)
    track_t_ratios: dict[str, PairedComparison] = field(default_factory=dict)
    """target 난이도 → cost-to-target 비율 비교."""
    gates: list[GateVerdict] = field(default_factory=list)
    best_static_action: ControllerAction | None = None
    tuning_budget: int = 0
    n_instances: int = 0
    tuning_runs: dict[str, int] = field(default_factory=dict)
    """컨트롤러별 실제 사용한 탐색 run 수 (프로토콜 D5 회계)."""
    experiment_id: str = ""
    identity: dict[str, object] = field(default_factory=dict)
    """실험 정체성 payload. 재개 판단과 결과 추적의 기준이다."""
    sweep_id: str = ""
    """이번 실행이 요청한 run 집합의 정체성 (프로토콜 D13)."""
    aggregation_id: str = ""
    """집계 규칙 정체성. 바뀌면 재집계만 하고 재실행하지 않는다 (프로토콜 D13)."""
    provenance: dict[str, object] = field(default_factory=dict)
    """실행 흔적 (git commit, dirty, hostname, 시각). 어떤 ID 에도 안 들어간다."""
    selections: dict[str, SelectionManifest] = field(default_factory=dict)
    """baseline 튜닝 근거 (프로토콜 D16). ``static`` / ``open_loop``."""
    saturation_diagnostics: dict[str, dict[str, float]] = field(default_factory=dict)
    """primary 에서 분리한 포화 task 의 진단 지표 (프로토콜 D14).

    ``rosen_d2`` 를 버리지 않고 별도 표로 보고한다. 정확한 0 도달률, floor-hit
    비율, GE-to-zero, step 수.
    """

    def summary_table(self) -> str:
        header = (
            f"{'controller':<28} {'logΔ(nat)':>10} {'도달률':>7} "
            f"{'cost→τ':>9} {'총 GE':>8} {'탐색 GE':>10} {'거절':>6} {'CG수렴':>7}"
        )
        lines = [header, "-" * len(header)]
        for name, g in self.groups.items():
            cost = (
                f"{g.median_cost_to_target_ge:.1f}"
                if math.isfinite(g.median_cost_to_target_ge)
                else "미도달"
            )
            lines.append(
                f"{name:<28} {g.median_log_improvement:>10.3f} "
                f"{g.success_rate:>6.0%} {cost:>9} "
                f"{g.median_total_cost_ge:>8.1f} {g.median_search_cost_ge:>10.1f} "
                f"{g.mean_rejection_rate:>6.2f} {g.mean_cg_convergence_rate:>7.2f}"
            )
        return "\n".join(lines)


@dataclass(slots=True)
class BeamCalibration:
    """beam width 민감도 측정 결과 (프로토콜 F).

    beam 은 추측으로 정하지 않고 pilot subset 에서 측정해 고른다. beam search 는
    정확한 planner 가 아니므로 폭을 줄이면 계획 품질이 떨어질 수 있고, 그것이
    게이트 C의 결론을 바꿀 수 있다.

    Attributes:
        rows: ``(space, quota, beam)`` -> 측정값. ``quota`` 는 ``c_max`` 배수다.
        selected_beam: 선택 규칙을 적용한 결과.
        reference_beam: 비교 기준이 된 최대 beam.
        tolerance: 상대 허용 오차.
    """

    rows: dict[tuple[str, float, int], dict[str, float]] = field(default_factory=dict)
    selected_beam: int = 0
    reference_beam: int = 0
    tolerance: float = UTILITY_TOLERANCE
    utility_epsilon: float = UTILITY_EPSILON
    deep_fraction_tolerance: float = DEEP_FRACTION_TOLERANCE
    rationale: str = ""
    rejections: dict[int, str] = field(default_factory=dict)
    """beam 별 배제 사유. 사후 해석이 아니라 사전 규칙의 적용 결과다."""

    def table(self) -> str:
        header = (
            f"{'space':<8} {'Q/cmax':>7} {'beam':>5} {'logΔ(nat)':>11} "
            f"{'rel.diff':>9} {'depth>1':>8} {'Δdepth':>8} {'cap':>4} {'wall(s)':>9}"
        )
        lines = [header, "-" * len(header)]
        for (space, quota, beam), row in sorted(self.rows.items()):
            cap = row.get("depth_cap_fraction", float("nan"))
            lines.append(
                f"{space:<8} {quota:>7.1f} {beam:>5} {row['log_improvement']:>11.4f} "
                f"{row['relative_diff']:>9.4f} {row['deep_fraction']:>8.2f} "
                f"{row.get('deep_diff', float('nan')):>8.3f} "
                f"{cap:>4.2f} {row['wall_clock_sec']:>9.2f}"
            )
        return "\n".join(lines)


def _depth_stats(store: ResultStore | None, label: str) -> tuple[float, float]:
    """``(depth>1 비율, depth_cap 에 걸린 run 비율)``.

    ``depth_cap_fraction`` 이 0 이 아니면 쿼터를 다 쓰지 못한 계획이 있으므로
    쿼터 사다리 비교가 훼손된다. 게이트 C 보고에 포함해야 한다.
    """
    if store is None:
        return float("nan"), float("nan")
    depth_totals: dict[str, int] = {}
    n_runs = 0
    n_capped = 0
    for record in store:
        if record.key.controller != label:
            continue
        if record.chosen_depths:
            for depth, count in record.chosen_depths.items():
                depth_totals[depth] = depth_totals.get(depth, 0) + count
        if record.planner_stats is not None:
            n_runs += 1
            n_capped += record.planner_stats.get("depth_cap_hit", 0.0)
    total = sum(depth_totals.values())
    deep = 1.0 - depth_totals.get("1", 0) / total if total else float("nan")
    cap = n_capped / n_runs if n_runs else float("nan")
    return deep, cap


def calibrate_beam_width(
    config: HeadroomConfig,
    *,
    narrow: ActionSpace,
    wide: ActionSpace,
    beams: Sequence[int] = (1, 2, 4),
    quotas: Sequence[float] = (1.0, 4.0),
    tolerance: float = UTILITY_TOLERANCE,
    utility_epsilon: float = UTILITY_EPSILON,
    deep_fraction_tolerance: float = DEEP_FRACTION_TOLERANCE,
    store: ResultStore | None = None,
    code_dirty: bool = False,
    verbose: bool = True,
) -> BeamCalibration:
    """beam width 를 pilot subset 에서 측정해 고른다 (프로토콜 F).

    선택 규칙은 **사전 정의된 것**이며 결과를 본 뒤 바꾸지 않는다.

    ```text
    1. 최대 beam 을 기준으로 삼는다.
    2. 모든 (space, horizon) 조합에서 다음 둘을 모두 만족하는 beam 중
       가장 작은 것을 고른다.
         |J_b - J_ref| / max(|J_ref|, eps) < tolerance          (기본 0.02)
         |d_b - d_ref| <= deep_fraction_tolerance               (기본 0.05)
       여기서 d 는 chosen_depth > 1 비율이다.
    3. 동률이면 wall-clock 이 짧은 것, 그래도 같으면 beam 2.
    ```

    분모가 ``|J_ref| + eps`` 가 아니라 ``max(|J_ref|, eps)`` 인 이유는 ``J_ref``
    가 0 근처일 때 상대 오차가 폭발하기 때문이다. ``eps`` 값도 상수로 고정해
    사후 조정을 막는다.

    ``deep_fraction`` 을 함께 보는 이유: beam 을 줄여 깊은 계획이 사라지면
    효용이 비슷해도 게이트 C의 해석이 달라진다. "크게 다르면 제외"를 결과를
    본 뒤 판단하지 않기 위해 수치로 고정한다.

    Args:
        beams: 시험할 beam 폭.
        quotas: 시험할 미래 GE 쿼터 (``c_max`` 배수). 사다리의 양 끝을 쓴다.
        tolerance: 효용 상대 허용 오차.
        utility_epsilon: 상대 오차 분모의 하한.
        deep_fraction_tolerance: ``depth > 1`` 비율의 허용 차이.
        store: 재개 가능한 저장소.
        code_dirty: 커밋되지 않은 변경이 있는지. 실험 정체성에 포함된다.

    Returns:
        ``BeamCalibration``.
    """
    calibration = BeamCalibration(
        tolerance=tolerance,
        utility_epsilon=utility_epsilon,
        deep_fraction_tolerance=deep_fraction_tolerance,
    )
    reference = max(beams)
    calibration.reference_beam = reference
    spaces = {"narrow": narrow, "wide": wide}

    measured: dict[tuple[str, float, int], dict[str, float]] = {}
    for space_label, space in spaces.items():
        for quota in quotas:
            for beam in beams:
                label = f"cal_budgeted_Q{quota:g}_{space_label}_b{beam}"
                if verbose:
                    print(f"  {label}", flush=True)
                # beam 과 쿼터가 정체성에 들어가야 재개가 안전하다. 단
                # code_dirty 는 넣지 않는다 (프로토콜 D13). 실행 의미가 아니다.
                exp_id = run_semantics_id(
                    config.run_semantics_payload(
                        controller="budgeted_mpc",
                        space=space,
                        quota=float(quota),
                        extra={
                            "execution_mode": "shrinking",
                            "beam_width_override": beam,
                        },
                    )
                )
                started = time.perf_counter()
                runs = run_controller(
                    config,
                    # 주 컨트롤러가 shrinking 이므로 beam 도 그것으로 고른다
                    # (프로토콜 D12). fresh 는 진단 baseline 이라 beam 선택
                    # 근거로 쓰지 않는다.
                    lambda _t, _g, s=space, q=quota, b=beam: ShrinkingQuotaMPCController(
                        s,
                        quota_multiplier=q,
                        beam_width=b,
                        max_depth=config.max_plan_depth,
                        track="fixed_budget",
                    ),
                    label=label,
                    exp_id=exp_id,
                    store=store,
                    verbose=verbose,
                )
                elapsed = time.perf_counter() - started
                group = summarize_group(runs, controller=label)
                deep, cap = _depth_stats(store, label)
                measured[space_label, float(quota), beam] = {
                    "log_improvement": group.median_log_improvement,
                    "relative_diff": float("nan"),
                    "deep_fraction": deep,
                    "depth_cap_fraction": cap,
                    "wall_clock_sec": elapsed,
                }

    # 상대 차이와 depth 차이를 채운다.
    # 분모는 ``|J_ref| + eps`` 가 아니라 ``max(|J_ref|, eps)`` 다. J_ref 가 0
    # 근처일 때 상대 오차가 폭발하는 것을 막는다.
    for (space_label, quota, _beam), row in measured.items():
        ref_row = measured[space_label, quota, reference]
        ref = ref_row["log_improvement"]
        if math.isfinite(ref) and math.isfinite(row["log_improvement"]):
            row["relative_diff"] = abs(row["log_improvement"] - ref) / max(
                abs(ref), utility_epsilon
            )
        ref_deep = ref_row["deep_fraction"]
        if math.isfinite(ref_deep) and math.isfinite(row["deep_fraction"]):
            row["deep_diff"] = abs(row["deep_fraction"] - ref_deep)
        else:
            row["deep_diff"] = float("nan")
    calibration.rows = measured

    # 선택 규칙 (사전 정의. 결과를 본 뒤 바꾸지 않는다)
    candidates: list[int] = []
    rejections: dict[int, str] = {}
    for beam in sorted(beams):
        reasons: list[str] = []
        for space_label in spaces:
            for quota in quotas:
                row = measured[space_label, float(quota), beam]
                tag = f"{space_label}/Q{quota:g}"
                if not math.isfinite(row["relative_diff"]):
                    reasons.append(f"{tag}: 효용 비교 불가")
                elif row["relative_diff"] >= tolerance:
                    reasons.append(f"{tag}: 효용 상대차 {row['relative_diff']:.4f}")
                deep_diff = row.get("deep_diff", float("nan"))
                if math.isfinite(deep_diff) and deep_diff > deep_fraction_tolerance:
                    reasons.append(f"{tag}: depth>1 비율 차이 {deep_diff:.3f}")
        if reasons:
            rejections[beam] = "; ".join(reasons[:3])
        else:
            candidates.append(beam)

    calibration.rejections = rejections
    if candidates:
        # tie-break: 최소 beam -> wall-clock 짧은 것 -> beam 2
        selected = min(
            candidates,
            key=lambda b: (
                b,
                sum(measured[s, float(q), b]["wall_clock_sec"] for s in spaces for q in quotas),
                0 if b == 2 else 1,
            ),
        )
        calibration.selected_beam = selected
        calibration.rationale = (
            f"beam {selected}: 모든 (space, quota) 에서 기준 beam {reference} 대비 "
            f"효용 상대차 < {tolerance:g} 이고 depth>1 비율 차이 "
            f"<= {deep_fraction_tolerance:g}"
        )
    else:
        calibration.selected_beam = reference
        calibration.rationale = (
            f"어떤 축소 beam 도 기준을 만족하지 못했다 "
            f"(효용 {tolerance:g}, depth {deep_fraction_tolerance:g}). "
            f"기준 beam {reference} 를 유지한다. "
            + " | ".join(f"beam {b}: {why}" for b, why in rejections.items())
        )
    return calibration


def run_headroom(
    config: HeadroomConfig,
    *,
    narrow: ActionSpace,
    wide: ActionSpace,
    absolute: ActionSpace,
    store: ResultStore | None = None,
    git_commit: str = "",
    code_dirty: bool = False,
    verbose: bool = True,
) -> HeadroomReport:
    """게이트 A~D를 측정한다.

    Args:
        config: 실험 설정.
        narrow: 정책이 실제로 쓸 행동 공간.
        wide: damping 배수를 넓힌 공간.
        absolute: 도달성 제약 없는 분석용 공간. **로그 해상도가 narrow 와
            같아야** 게이트 B의 해석이 성립한다.
        git_commit: 실행 시점 커밋. **어떤 ID 에도 들어가지 않고**
            ``execution_provenance`` 로만 기록된다 (프로토콜 D13). 넘기지 않으면
            summary 의 provenance 가 빈 문자열이 되어 원고에서 결과를 커밋에
            연결할 수 없다.
        code_dirty: 실행 시점에 커밋되지 않은 변경이 있었는가.
        verbose: 진행 상황 출력.
    """
    report = HeadroomReport(phase=config.phase)
    report.n_instances = sum(
        1 for spec in config.specs for seed in config.seeds if _is_eligible(make_task(spec, seed))
    )
    n_tune = config.tuning_budget or len(narrow)
    report.tuning_budget = n_tune

    # 실험 정체성. 설정이 하나라도 다르면 재개 시 별개 run 으로 취급된다.
    identity = config.identity_payload(
        {"narrow": narrow, "wide": wide, "absolute": absolute},
        code_dirty=code_dirty,
        extra={"n_tune": n_tune, "mode": "headroom"},
    )
    exp_id = experiment_id(identity)
    report.experiment_id = exp_id
    report.identity = identity

    # --- 3계층 정체성 (프로토콜 D13) ---
    #
    # 컨트롤러마다 **그 컨트롤러가 실제 쓰는 설정만**으로 semantics id 를 만든다.
    # beam 을 바꿔도 best_static / heuristic / one-step 의 id 는 유지되고,
    # 집계 정책을 바꿔도 어떤 run 도 무효화되지 않는다.
    def sem(
        controller: str,
        *,
        space: ActionSpace | None = None,
        quota: float | None = None,
        uses_target: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> str:
        return run_semantics_id(
            config.run_semantics_payload(
                controller=controller,
                space=space,
                quota=quota,
                uses_target=uses_target,
                extra=extra,
            )
        )

    report.sweep_id = sweep_id(
        config.sweep_payload(
            controllers=[
                "best_static",
                "best_open_loop",
                "heuristic",
                "onestep",
                "shrinking",
                "committed",
                "fresh",
            ],
        )
    )
    report.aggregation_id = aggregation_id(config.aggregation_payload())
    # git commit 과 dirty 는 어떤 ID 에도 들어가지 않는다 (프로토콜 D13).
    # 다만 provenance 에는 반드시 남아야 한다. 없으면 원고에서 결과를 커밋에
    # 연결할 수 없다.
    report.provenance = execution_provenance(
        git_commit=git_commit, code_dirty=code_dirty
    )

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log(
        f"[{config.phase}] 인스턴스 {report.n_instances}개, "
        f"GE 예산 {config.cost_budget_ge:g}, N_tune {n_tune}"
    )

    # --- baseline ---
    log(f"best_static (탐색 {n_tune}회)")
    best_action, static_group, static_manifest = search_best_static(
        config,
        narrow,
        n_tune=n_tune,
        exp_id=sem("best_static", space=narrow),
        store=store,
    )
    report.selections["static"] = static_manifest
    log(f"  {static_manifest.describe()}")
    static_group = _relabel(static_group, "best_static")
    static_runs = static_group.runs
    report.best_static_action = best_action
    report.groups["best_static"] = static_group
    report.tuning_runs["best_static"] = n_tune

    log(f"best_open_loop (탐색 {n_tune}회)")
    open_raw, open_manifest = search_best_open_loop(
        config,
        narrow,
        n_tune=n_tune,
        exp_id=sem(
            "best_open_loop",
            space=narrow,
            extra={"n_schedule_segments": config.n_schedule_segments},
        ),
        store=store,
    )
    report.selections["open_loop"] = open_manifest
    log(f"  {open_manifest.describe()}")
    open_group = _relabel(open_raw, "best_open_loop")
    report.groups["best_open_loop"] = open_group
    report.tuning_runs["best_open_loop"] = n_tune
    if open_manifest.is_constant_schedule:
        log(
            "  주의: 선택된 open-loop 스케줄이 constant 다. static 으로 퇴화했으므로 "
            "P2 에서 best_static 과 독립적인 baseline 으로 세지 않는다 (D16)."
        )

    log("heuristic")
    heuristic_runs = run_controller(
        config,
        lambda _t, _g: HeuristicController(narrow),
        label="heuristic",
        exp_id=sem("heuristic", space=narrow),
        store=store,
    )
    report.groups["heuristic"] = summarize_group(heuristic_runs, controller="heuristic")
    report.tuning_runs["heuristic"] = 1

    # --- H=1 (one-step efficiency): 행동 공간 3종. 게이트 A1, B ---
    #
    # H=1 에서 planner 의 fixed_budget 효용은 gain/cost 이고, one-step 의
    # efficiency_score 와 같은 식이다. 그런데 one-step 은 HVP 그래프를
    # 후보 전체에 공유하므로 약 10배 싸다. absolute (34 damping x 4 budget =
    # 136 action, sweep 1292 HVP) 를 감당할 수 있는 유일한 경로다.
    onestep_runs: dict[str, list[RunSummary]] = {}
    for space_label, space in (
        ("narrow", narrow),
        ("wide", wide),
        ("absolute", absolute),
    ):
        label = f"onestep_{space_label}"
        log(f"{label} (sweep {space.hvp_per_sweep} HVP)")
        runs = run_controller(
            config,
            lambda _t, _g, s=space: OneStepEfficiencyController(s),
            label=label,
            # one-step 은 quota / beam / suffix 정책을 쓰지 않는다.
            exp_id=sem("onestep", space=space),
            store=store,
        )
        onestep_runs[label] = runs
        report.groups[label] = summarize_group(runs, controller=label)
        report.tuning_runs[label] = 0

    # --- Track E planner: narrow / wide 만. 게이트 A2, C ---
    #
    # absolute 는 여기서 제외한다. 현재 damping 과 무관하게 순간 이동하므로
    # damping ramp-up 과 temporal credit assignment 를 **제거해 버린다.**
    # 장기 계획의 필요성을 묻는 게이트 C에 넣을 이유가 없고, 비용도
    # 감당할 수 없다 (실제 step 당 약 1,200회 시뮬레이션).
    # 실행 방식 3종 (프로토콜 D12). 탐색은 완전히 동일하고 실행만 다르므로
    # 차이가 탐색 품질 차이와 섞이지 않는다.
    #
    #   shrinking  주 컨트롤러. 쓴 비용을 차감하고 horizon 을 연장하지 않는다
    #   committed  계획 상한 / open-loop oracle. 초기 상태에 조건화됨
    #   fresh      시간 불일치가 확인된 진단 baseline. 주 결과에 쓰지 않는다
    modes: dict[str, type[BudgetedMPCController]] = {
        "shrinking": ShrinkingQuotaMPCController,
        "committed": CommittedPlanController,
        "fresh": BudgetedMPCController,
    }
    planner_runs: dict[str, list[RunSummary]] = {}
    fresh_seeds = tuple(config.seeds[: config.fresh_diagnostic_seeds])
    spaces_to_run = [
        (name, sp)
        for name, sp in (("narrow", narrow), ("wide", wide))
        if name in config.planner_spaces
    ]
    for space_label, space in spaces_to_run:
        for quota in config.quotas:
            for mode, factory in modes.items():
                if mode not in config.execution_modes:
                    continue
                # fresh 는 진단 baseline 이므로 계산을 줄인다 (프로토콜 D12).
                # P1~P3 판정에 쓰지 않으므로 seed 집합이 달라도 문제가 없다.
                if mode == "fresh":
                    if not fresh_seeds:
                        continue
                    if space_label == "wide" and not config.run_fresh_wide:
                        continue
                seeds = fresh_seeds if mode == "fresh" else None
                label = f"{mode}_Q{quota:g}_{space_label}"
                note = f", seed {len(seeds)}개 진단" if seeds is not None else ""
                log(f"{label} ({len(space)} actions, beam {config.beam_width}{note})")
                runs = run_controller(
                    config,
                    lambda _t, _g, s=space, q=quota, f=factory: f(
                        s,
                        quota_multiplier=q,
                        beam_width=config.beam_width,
                        max_depth=config.max_plan_depth,
                        track="fixed_budget",
                    ),
                    label=label,
                    # planner 는 quota / beam / max_plan_depth / 실행 방식이
                    # 모두 결과를 바꾼다. Track E 이므로 target 은 넣지 않는다.
                    exp_id=sem(
                        "budgeted_mpc",
                        space=space,
                        quota=quota,
                        extra={"execution_mode": mode},
                    ),
                    store=store,
                    seeds=seeds,
                )
                planner_runs[label] = runs
                report.groups[label] = summarize_group(runs, controller=label)
                report.tuning_runs[label] = 0

    # --- Track E 쌍별 차이 ---
    def delta(base: Sequence[RunSummary], treat: Sequence[RunSummary]) -> PairedDelta:
        return compare_paired_delta(base, treat, metric="log_improvement")

    max_q = max(config.quotas)
    min_q = min(config.quotas)

    # 단계별 실행에서는 일부 arm 이 없다. 없는 라벨은 조용히 건너뛰고
    # 해당 게이트를 판정불가로 남긴다 (프로토콜 D13 sweep 커버리지).
    def pr(label: str) -> list[RunSummary] | None:
        return planner_runs.get(label)

    _candidate_pairs: list[tuple[Sequence[RunSummary] | None, Sequence[RunSummary] | None]] = [
        # 게이트 A1: 순간적 absolute headroom (도달성 제약 제거, H=1)
        (static_runs, onestep_runs["onestep_absolute"]),
        # 게이트 A2: 도달 가능한 sequential headroom (주 컨트롤러 shrinking)
        (static_runs, pr(f"shrinking_Q{max_q:g}_narrow")),
        (static_runs, pr(f"shrinking_Q{max_q:g}_wide")),
        # 게이트 B: action-space restriction (모두 H=1, 같은 조건)
        (onestep_runs["onestep_narrow"], onestep_runs["onestep_absolute"]),
        (onestep_runs["onestep_narrow"], onestep_runs["onestep_wide"]),
        # 게이트 C1: time-consistency. 쿼터 초기화가 성능을 떨어뜨리는가
        (pr(f"fresh_Q{max_q:g}_narrow"), pr(f"shrinking_Q{max_q:g}_narrow")),
        # 게이트 C2: sequential planning value. 주 판정 통계다
        (onestep_runs["onestep_narrow"], pr(f"shrinking_Q{max_q:g}_narrow")),
        # 게이트 C3: feedback value. committed 대비 추가 이득
        (pr(f"committed_Q{max_q:g}_narrow"), pr(f"shrinking_Q{max_q:g}_narrow")),
        # 참고: 최소 쿼터. quota scale 이지 planning depth 가 아니다 (D15)
        (onestep_runs["onestep_narrow"], pr(f"shrinking_Q{min_q:g}_narrow")),
        # 참고 baseline
        (static_runs, open_group.runs),
        (static_runs, heuristic_runs),
    ]
    e_pairs = [(b, t) for b, t in _candidate_pairs if b and t]
    report.track_e_deltas = [delta(b, t) for b, t in e_pairs]

    # --- Track T: target 난이도별 cost-to-target ---
    log("Track T: target 난이도별 재집계" if config.run_track_t else "Track T: 건너뜀")
    for level in config.difficulties() if config.run_track_t else ():
        static_t = run_controller(
            config,
            lambda _t, _g, a=best_action: FixedController(a),
            label=f"best_static@{level}",
            difficulty=level,
            # Track T 는 target 이 종료 조건이므로 정체성에 포함한다.
            exp_id=sem("best_static", space=narrow, uses_target=True),
            store=store,
        )
        # planner 는 task 별 **절대** target loss 를 받아야 한다. 상대 target
        # v 의 절대값은 v * L_0 이므로 인스턴스마다 다르다. 고정값을 넘기면
        # cost-to-go 추정이 무의미해진다.
        planner_t = run_controller(
            config,
            lambda task, target, s=narrow: ShrinkingQuotaMPCController(
                s,
                quota_multiplier=max_q,
                beam_width=config.beam_width,
                max_depth=config.max_plan_depth,
                track="cost_to_target",
                target_loss=absolute_target_loss(task, target),
            ),
            label=f"shrinking_Q{max_q:g}@{level}",
            difficulty=level,
            exp_id=sem(
                "budgeted_mpc",
                space=narrow,
                quota=max_q,
                uses_target=True,
                extra={"execution_mode": "shrinking", "track": "cost_to_target"},
            ),
            store=store,
        )
        report.groups[f"best_static@{level}"] = summarize_group(
            static_t, controller=f"best_static@{level}"
        )
        report.groups[f"shrinking_Q{max_q:g}@{level}"] = summarize_group(
            planner_t, controller=f"shrinking_Q{max_q:g}@{level}"
        )
        report.track_t_ratios[level] = compare_paired(
            static_t, planner_t, metric="cost_to_target_ge"
        )

    # --- 게이트 판정 ---
    by_e = {(d.baseline, d.treatment): d for d in report.track_e_deltas}

    def e_delta(base: str, treat: str) -> float:
        d = by_e.get((base, treat))
        return d.median_delta if d else float("nan")

    # 비포화 민감도 (프로토콜 D14). 주 통계는 floor-capped 전체 쌍이고 이것은
    # 병기용이다. 비포화만 primary 로 쓰면 최적점에 도달한 강한 run 을 제거하는
    # 편향이 된다.
    def e_delta_nonsat(base: Sequence[RunSummary], treat: Sequence[RunSummary]) -> str:
        b, t = drop_saturated_pairs(base, treat)
        if not b:
            return "비포화 쌍 없음"
        d = compare_paired_delta(b, t, metric="log_improvement")
        return f"{d.median_delta:+.3f} nat (n={d.n_valid})"

    # --- 3층 보고 (프로토콜 D19) ---
    #
    # ``rosen_d2 제외 = primary`` 라는 정의는 폐기했다. **포화는 task 이름이
    # 아니라 실제 ``floor_hit`` 으로 발생한다.** ``quad_spd`` 도 floor 아래로
    # 내려가 컨트롤러를 구분하지 못한다.
    #
    # ``drop_saturated_pairs`` 를 primary 게이트로 승격하지도 않는다. 비교 쌍마다
    # 표본이 달라지고, one-sided saturation 은 그 컨트롤러가 **더 잘했다는 증거**
    # 인데 쌍을 삭제하면 좋은 결과를 제거한다. 민감도 분석으로만 쓴다.
    def three_layer(base: Sequence[RunSummary], treat: Sequence[RunSummary]) -> str:
        if not base or not treat:
            return ""
        a = compare_paired_delta(base, treat, metric="log_improvement")
        base_by_key = {(r.task_instance_id, r.seed): r for r in base}
        pairs = [
            (t, base_by_key[(t.task_instance_id, t.seed)])
            for t in treat
            if (t.task_instance_id, t.seed) in base_by_key
        ]
        # delta = treat - base. 양수면 treat 가 좋다.
        deltas = [(t, b, t.log_improvement - b.log_improvement) for t, b in pairs]
        finite = [d for _t, _b, d in deltas if math.isfinite(d)]
        unsat = a.n_valid - a.n_saturated

        lines = [
            f"[1] all-task floor-capped  {a.median_delta:+.3f} nat  n={a.n_valid}  "
            f"CI {a.delta_ci[0]:+.3f}~{a.delta_ci[1]:+.3f}  "
            f"joint={a.n_joint_saturated} one-sided={a.n_one_sided_saturated} "
            f"unsat={unsat}  "
            f"(+{sum(1 for d in finite if d > 0)}/0×{sum(1 for d in finite if d == 0.0)}"
            f"/−{sum(1 for d in finite if d < 0)})"
        ]

        # [2] spec 별. 난이도에 따라 headroom 이 달라지는 것 자체가 결과다.
        by_spec: dict[str, list[float]] = {}
        for t, _b, d in deltas:
            spec = t.task_instance_id.rsplit("_seed", 1)[0]
            by_spec.setdefault(spec, []).append(d)
        for spec in sorted(by_spec):
            # `_median` 을 쓴다. `sorted(vals)[len(vals)//2]` 는 짝수 표본에서
            # 상위 중앙값을 골라 `compare_paired_delta` 의 all-task 통계와 규약이
            # 어긋난다. n=10 에서 spec 별 값이 0.01 nat 규모로 갈렸다.
            med = median_of(by_spec[spec])
            listed = ", ".join(f"{d:+.3f}" for d in by_spec[spec])
            lines.append(
                f"[2] {spec:<34} median {med:+.3f} n={len(by_spec[spec])} [{listed}]"
            )

        # [3] pairwise nonsaturated 민감도. **primary 가 아니다.**
        nb, nt = drop_saturated_pairs(base, treat)
        if nb:
            s = compare_paired_delta(nb, nt, metric="log_improvement")
            lines.append(
                f"[3] pairwise nonsaturated 민감도  {s.median_delta:+.3f} nat  "
                f"n={s.n_valid}  (비교마다 n 이 달라진다. primary 아님)"
            )
        else:
            lines.append("[3] pairwise nonsaturated 민감도  비포화 쌍 없음")
        return "\n      ".join(lines)

    gate_a1_nonsat = e_delta_nonsat(static_runs, onestep_runs["onestep_absolute"])
    gate_b_nonsat = e_delta_nonsat(
        onestep_runs["onestep_narrow"], onestep_runs["onestep_absolute"]
    )

    gate_a1 = e_delta("best_static", "onestep_absolute")
    report.gates.append(
        GateVerdict(
            name="A1",
            track="Track E",
            question=(
                "현재 상태에서 좋은 damping 이 존재하는가 "
                "(instantaneous absolute-action headroom, H=1)"
            ),
            statistic=gate_a1,
            nonsaturated=gate_a1_nonsat,
            unit="nat",
            go_threshold=1.0,
            pivot_threshold=0.3,
            detail=(
                (f"loss {math.exp(gate_a1):.2f}배 차이. " if math.isfinite(gate_a1) else "")
                + "도달성 제약을 완전히 없앤 **순간적** 이득이다. "
                "전역 상한이나 장기 헤드룸이 아니다."
            ),
        )
    )

    gate_a2_narrow = e_delta("best_static", f"shrinking_Q{max_q:g}_narrow")
    gate_a2_wide = e_delta("best_static", f"shrinking_Q{max_q:g}_wide")
    gate_a2 = (
        max(v for v in (gate_a2_narrow, gate_a2_wide) if math.isfinite(v))
        if any(math.isfinite(v) for v in (gate_a2_narrow, gate_a2_wide))
        else float("nan")
    )
    report.gates.append(
        GateVerdict(
            name="A2",
            track="Track E",
            question=(
                "현실적인 multiplier action 으로 그 이득에 접근할 수 있는가 "
                f"(shrinking Q={max_q:g}xc_max, narrow/wide vs best_static)"
            ),
            statistic=gate_a2,
            unit="nat",
            go_threshold=0.7,
            pivot_threshold=0.2,
            nonsaturated=three_layer(
                static_runs, planner_runs.get(f"shrinking_Q{max_q:g}_narrow") or []
            ),
            detail=(
                f"narrow {gate_a2_narrow:+.3f}, wide {gate_a2_wide:+.3f} nat. "
                "A1 대비 크게 낮으면 행동 공간 도달성이 병목이다."
            ),
        )
    )

    gate_b = e_delta("onestep_narrow", "onestep_absolute")
    wide_gain = e_delta("onestep_narrow", "onestep_wide")
    report.gates.append(
        GateVerdict(
            name="B",
            track="Track E",
            question="행동 공간이 병목인가 (absolute vs narrow, 모두 H=1, 해상도 정렬)",
            statistic=gate_b,
            nonsaturated=gate_b_nonsat,
            unit="nat",
            go_threshold=0.5,
            pivot_threshold=0.1,
            detail=f"참고: wide - narrow = {wide_gain:+.3f} nat",
        )
    )

    # --- 게이트 C1/C2/C3 (프로토콜 D12) ---
    #
    # 주 컨트롤러는 shrinking 이다. fresh 는 시간 불일치가 확인된 진단
    # baseline 이므로 판정에 쓰지 않는다.
    shrink = f"shrinking_Q{max_q:g}_narrow"
    commit = f"committed_Q{max_q:g}_narrow"
    fresh = f"fresh_Q{max_q:g}_narrow"

    curves = []
    depth_notes = []
    for mode in ("shrinking", "committed", "fresh"):
        points = " → ".join(
            f"Q{q:g}:{report.groups[f'{mode}_Q{q:g}_narrow'].median_log_improvement:.3f}"
            for q in config.quotas
            if f"{mode}_Q{q:g}_narrow" in report.groups
        )
        if points:
            curves.append(f"{mode} {points}")
    for q in config.quotas:
        if f"shrinking_Q{q:g}_narrow" not in report.groups:
            continue
        deep, cap = _depth_stats(store, f"shrinking_Q{q:g}_narrow")
        depth_notes.append(f"Q{q:g}:d>1={deep:.2f},cap={cap:.2f}")

    report.gates.append(
        GateVerdict(
            name="C1",
            track="Track E",
            question="쿼터를 매 step 초기화하는 것이 실제 성능을 떨어뜨리는가 (shrinking - fresh)",
            statistic=e_delta(fresh, shrink),
            unit="nat",
            go_threshold=0.3,
            pivot_threshold=0.05,
            detail=(
                f"쿼터 곡선(narrow): {' | '.join(curves)}. "
                f"**fresh 는 seed {config.fresh_diagnostic_seeds}개에서만 돌린 "
                "진단 baseline 이다.** paired 비교는 겹치는 seed 에서만 이루어지므로 "
                "이 통계의 표본이 다른 게이트보다 작다. "
                "**연구 결과이지만 PPO 착수 게이트가 아니다.** 양수면 receding "
                "horizon 이 준비 행동만 반복하며 payoff 를 뒤로 미룬다는 증거다."
            ),
        )
    )
    # 포화 task 진단 표 (프로토콜 D14). 분리한 task 를 버리지 않고 별도로 낸다.
    for label in ("best_static", "onestep_narrow", shrink, commit):
        group = report.groups.get(label)
        if group is None:
            continue
        _, sat_runs = split_by_task_family(
            group.runs, exclude_prefixes=config.saturated_task_prefixes
        )
        if sat_runs:
            report.saturation_diagnostics[label] = saturation_report(sat_runs)

    gate_c2 = e_delta("onestep_narrow", shrink)
    q1_gain = e_delta("onestep_narrow", f"shrinking_Q{min_q:g}_narrow")
    report.gates.append(
        GateVerdict(
            name="C2",
            track="Track E",
            question=(
                f"시간 일관적인 다단계 재계획이 one-step 효율 제어보다 나은가 "
                f"(shrinking Q={max_q:g} - C0)"
            ),
            statistic=gate_c2,
            unit="nat",
            go_threshold=0.3,
            pivot_threshold=0.05,
            nonsaturated=three_layer(
                onestep_runs["onestep_narrow"], planner_runs.get(shrink) or []
            ),
            detail=(
                f"깊이/상한(shrinking): {' '.join(depth_notes)}. "
                f"참고 Q={min_q:g}(depth 1만 가능) - C0 = {q1_gain:+.3f} nat. "
                "**GO 판정에는 두 조건이 모두 필요하다** (P3). 개선이 있고, "
                "동시에 depth>1 이 실제로 채택돼야 한다. 개선만 있고 depth 가 "
                "계속 1 이면 planning 이 아니라 탐색량이 기여한 것이다. "
                "``cap`` 이 0 이 아니면 계산 상한 때문에 쿼터를 다 쓰지 못한 "
                "step 이 있으므로 사다리 비교가 훼손된 것으로 보고한다. "
                "이 게이트만으로 PPO 를 시작하지 않는다. P1~P4 를 함께 본다."
            ),
        )
    )
    report.gates.append(
        GateVerdict(
            name="C3",
            track="Track E",
            question="상태를 관찰하며 재계획하는 것이 고정 실행보다 추가 이득인가 (shrinking - committed)",
            statistic=e_delta(commit, shrink),
            unit="nat",
            go_threshold=0.3,
            pivot_threshold=0.05,
            nonsaturated=three_layer(
                planner_runs.get(commit) or [], planner_runs.get(shrink) or []
            ),
            detail=(
                "약 0 이면 좋은 sequence 는 존재하지만 feedback 자체의 추가 "
                "가치는 작다. 음수면 approximate replanning 이 계획을 훼손한다. "
                "**committed 는 초기 상태에 조건화된 oracle 이므로 open_loop "
                "baseline 과 다르다.** committed 가 open_loop 보다 좋다고 해서 "
                "feedback 이 필요하다는 뜻이 아니다."
            ),
        )
    )

    primary = report.track_t_ratios.get(config.primary_difficulty)
    gate_d = primary.ratio_geometric_mean if primary else float("nan")
    all_levels = ", ".join(
        f"{level}:{c.ratio_geometric_mean:.3f}x(도달 {c.n_both_reached}/{c.n_pairs})"
        for level, c in report.track_t_ratios.items()
    )
    report.gates.append(
        GateVerdict(
            name="D",
            track="Track T",
            question=(
                f"cost-to-target 헤드룸이 있는가 "
                f"(planner vs best_static, {config.primary_difficulty})"
            ),
            statistic=gate_d,
            unit="배",
            go_threshold=1.20,
            pivot_threshold=1.05,
            detail=(
                f"난이도별 {all_levels}. "
                "게이트 A와 결론이 다르면 그 불일치 자체를 결과로 보고한다 "
                "(프로토콜 D9)."
            ),
        )
    )
    return report
