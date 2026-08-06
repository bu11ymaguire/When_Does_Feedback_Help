"""재개 가능한 결과 저장소.

왜 필요한가
-----------
Stage 2 실험은 컨트롤러 x 행동공간 x horizon x task x seed x target 조합이라
수백~수천 run 이 된다. 프로세스가 중간에 끊기면(셸 중단, timeout, OOM) 이미
계산한 결과까지 잃는 것은 받아들일 수 없다.

그래서 run 단위로 즉시 append-only JSONL 에 기록하고, 재실행 시 이미 완료된
``(controller, task_instance, seed, target)`` 조합을 건너뛴다. 단순한 백그라운드
실행보다 이것이 우선이다.

상태를 구분한다
---------------
```text
completed    정상 종료. 재실행 시 건너뛴다.
failed       예외 발생. 원인을 기록하고 재실행 시 다시 시도한다.
interrupted  기록 전에 프로세스가 끊긴 경우. 파일에 남지 않으므로
             자동으로 재시도 대상이 된다.
```

미완료를 조용히 성공으로 취급하지 않는다. 프로토콜 D6의 절단 규칙과 같은
원칙이다 (실패한 run 도 삭제하지 않고 원인을 기록한다).

집계와 raw 를 분리한다
----------------------
이 저장소는 raw run 기록만 담는다. 집계(``GroupSummary``, 게이트 판정)는
``metrics`` / ``oracle`` 이 raw 를 읽어 계산한다. 그래서 집계 로직을 바꿔도
실험을 다시 돌릴 필요가 없다.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rl_newton.benchmark.metrics import RunSummary
from rl_newton.utils.logging import sanitize_for_json
from rl_newton.utils.provenance import config_hash

__all__ = [
    "HOST_ID_FALLBACK",
    "RunKey",
    "RunRecord",
    "ResultStore",
    "RunStatus",
    "experiment_id",
    "run_semantics_id",
    "sweep_id",
    "aggregation_id",
    "OPTIMIZER_SEMANTICS_VERSION",
    "PLANNER_SEMANTICS_VERSION",
    "TASK_SEMANTICS_VERSION",
    "OPEN_LOOP_SEMANTICS_VERSION",
    "SELECTION_SEMANTICS_VERSION",
    "AGGREGATION_VERSION",
    "SelectionManifest",
    "selection_id",
    "execution_provenance",
    "environment_fingerprint",
]

RunStatus = Literal["completed", "failed"]


OPTIMIZER_SEMANTICS_VERSION = 1
"""optimizer 실행 의미 버전 (프로토콜 D13).

**step 하나의 결과를 바꾸는 변경이 있을 때만 올린다.** CG 종료 조건, damping
적용 규칙, fallback, 비용 회계, trust ratio 계산 등이다.

git commit 을 정체성에 넣으면 문서나 집계 코드만 바꿔도 모든 run 이 무효화된다.
실제로 그런 일이 있었다 (집계 코드 수정으로 423 run 재실행). 그래서 실행 의미를
명시적 버전으로 관리하고, git commit 과 dirty 상태는 provenance 로만 저장한다.
"""

PLANNER_SEMANTICS_VERSION = 1
"""planner 탐색·선택 의미 버전. planner 계열 컨트롤러만 영향받는다.

쿼터 허용 규칙, Pareto/bucket 가지치기, 선택 규칙, suffix incumbent 정책 등.
``best_static`` 이나 ``heuristic`` 은 이 값에 영향받지 않아야 한다.
"""

TASK_SEMANTICS_VERSION = 1
"""task 생성 의미 버전. 초기점, Hessian 구성, instance_id 규칙 등."""

OPEN_LOOP_SEMANTICS_VERSION = 2
"""open-loop 스케줄 의미 버전 (프로토콜 D17). **open-loop 만 영향받는다.**

2: ``progress`` 를 ``step / total_steps`` 에서 **소모 GE 비율**로 교체.
   초판은 GE 예산으로 종료하는데 breakpoint 가 step 비율이어서 스케줄의 첫
   구간만 실행됐다. static / heuristic / one-step / planner 는 ``progress`` 를
   쓰지 않으므로 이 버전을 정체성에 넣지 않는다.
"""

SELECTION_SEMANTICS_VERSION = 1
"""baseline 선택(튜닝) 의미 버전 (프로토콜 D16).

후보 집합 생성, 선택 지표, tie-break 규칙이 바뀌면 올린다. ``best_static`` 과
``best_open_loop`` 는 컨트롤러가 아니라 튜닝 결과이므로 선택 과정 자체가
재현 가능해야 한다.
"""

HOST_ID_FALLBACK = "host-unspecified"
"""``EXPERIMENT_HOST_ID`` 가 없을 때 쓰는 라벨.

**장치의 실제 이름을 대신 넣지 않는다.** 공개 산출물에 개인 식별 정보를 남길 이유가
없다. 여러 기계의 결과를 구별해야 하면 실행 시 별칭을 지정한다.

```bash
EXPERIMENT_HOST_ID=host-a python scripts/run_headroom.py ...
```
"""

AGGREGATION_VERSION = 4
"""집계 의미 버전 (프로토콜 D13/D14).

log floor 정책, 포화 분류, paired intersection 규칙, bootstrap 설정, 게이트
정의, report schema. **이것이 바뀌면 raw run 은 그대로 두고 재집계만 한다.**

2: D14 relative loss floor 와 포화 분류 도입.
3: primary(비포화 task) / all-task / saturation diagnostic 3층 보고 도입.
   bridge 분류에서 범위 밖(``LEGACY_ONLY_OUT_OF_SCOPE``)과 누락을 구별.
4: 3층 보고를 all-task / spec별 / pairwise-nonsaturated 로 재정의 (D19).
   ``rosen_d2 제외 = primary`` 정의 폐기. 포화는 task 이름이 아니라 ``floor_hit``
   으로 발생하고, ``drop_saturated_pairs`` 는 민감도 분석으로만 쓴다.
"""


def experiment_id(payload: Mapping[str, Any]) -> str:
    """정체성 해시. ``run_semantics_id`` / ``sweep_id`` / ``aggregation_id`` 공통.

    직렬화는 ``config_hash`` 가 정규화하므로 dict 순서에 의존하지 않는다.
    """
    return config_hash(dict(payload))


def run_semantics_id(payload: Mapping[str, Any]) -> str:
    """**개별 run 의 결과를 바꾸는 설정만**으로 만든 해시 (프로토콜 D13).

    같으면 저장된 run 을 안전하게 재사용할 수 있다. 컨트롤러가 실제로 쓰지 않는
    설정은 넣지 않는다. ``best_static`` run 이 ``beam`` 이나 ``quota`` 변경으로
    무효화될 이유가 없다.

    ```text
    포함:  task spec / seed / controller 종류 / 그 컨트롤러가 쓰는 action space
           GE budget / max steps / damping·CG 설정 / solver·fallback 규칙
           planner 계열이면 quota / beam / max plan depth / suffix 정책
           target (termination 에 쓰는 경우만)
           semantics version (optimizer / planner / task)
    제외:  sweep 커버리지 (어떤 run 을 도는가)
           집계 정책 (floor, CI, 게이트 정의)
           git commit / dirty (provenance 로만 저장)
    ```
    """
    return config_hash(dict(payload))


def sweep_id(payload: Mapping[str, Any]) -> str:
    """이번 명령이 **어떤 run 집합을 요청했는지** (프로토콜 D13).

    컨트롤러 목록, 전체 task·seed 목록, 진단 arm 부분집합, 단계(screening /
    confirmation), 출력 경로 등이다. **``sweep_id`` 가 바뀌어도 같은
    ``run_semantics_id`` 의 run 은 재사용한다.**
    """
    return config_hash(dict(payload))


def aggregation_id(payload: Mapping[str, Any]) -> str:
    """집계 규칙 정체성 (프로토콜 D13). 바뀌면 재집계만 하고 재실행하지 않는다."""
    return config_hash(dict(payload))


def selection_id(payload: Mapping[str, Any]) -> str:
    """baseline 선택(튜닝) 과정의 정체성 (프로토콜 D16)."""
    return config_hash(dict(payload))


def execution_provenance(*, git_commit: str = "", code_dirty: bool = False) -> dict[str, Any]:
    """실행 흔적. **어떤 ID 에도 들어가지 않는다** (프로토콜 D13).

    git commit 이나 code-dirty 를 ``sweep_id`` 에 넣으면, 같은 run 집합을
    요청했는데 문서만 수정해도 ID 가 달라진다. "어떤 집합을 요청했는가" 라는
    의미가 깨지므로 provenance 로 분리한다.

    **장치의 실제 이름을 기록하지 않는다.** 공개 저장소에 개인 식별 정보를 남길
    이유가 없다. 재현에 필요한 것은 OS, Python / PyTorch 버전, CPU 모델, 스레드 수,
    dtype, 관련 환경변수, 코드 commit 과 configuration hash 다. 장치 이름은 같은
    실행 환경을 묶는 라벨 역할만 하므로 별칭으로 충분하다.

    ``EXPERIMENT_HOST_ID`` 환경변수로 안정적인 별칭을 준다. 없으면
    ``HOST_ID_FALLBACK`` 을 쓴다. hostname 을 해시하는 방법은 쓰지 않는다. salt 가
    없으면 사전 대입으로 복원되고, salt 를 관리하는 복잡도가 이득보다 크다.
    """
    import os
    import platform
    from datetime import UTC, datetime

    return {
        "git_commit": git_commit,
        "code_dirty": code_dirty,
        "host_id": os.environ.get("EXPERIMENT_HOST_ID", HOST_ID_FALLBACK),
        "platform": f"{platform.system()} {platform.release()}",
        "recorded_at": datetime.now(UTC).isoformat(),
    }


@dataclass(slots=True)
class SelectionManifest:
    """baseline 튜닝이 **무엇을 근거로 무엇을 골랐는지** (프로토콜 D16).

    ``best_static`` 과 ``best_open_loop`` 는 컨트롤러가 아니라 튜닝 결과다. 라벨만
    ``static[7]`` → ``best_static`` 으로 바꾸면 어떤 설정이 왜 선택됐는지 사라진다.

    **evaluation 결과를 보고 선택을 역추정하면 사후 선택이다.** 당시 tuning 후보
    점수와 tie-break 만으로 결정론적으로 재현되어야 한다. 재현할 수 없으면
    ``legacy_unresolved`` 로 두고 해당 게이트를 미판정으로 남긴다.

    Attributes:
        selection_id: 선택 과정 정체성 해시.
        family: ``static`` 또는 ``open_loop``.
        candidate_labels: 후보 라벨 (``static[0]`` ...).
        candidate_scores: 라벨 → 선택 지표 값.
        selection_metric: 선택 지표 이름.
        tie_break_rule: 동률 처리 규칙.
        selected_label: 선택된 후보 라벨.
        selected_config: 선택된 실제 설정. static 은 action, open_loop 은 schedule.
        tuning_specs: 튜닝에 쓴 task spec.
        tuning_seeds: 튜닝에 쓴 seed.
        n_tune: 후보 수 (``N_tune``).
        semantics_version: 선택 규칙 버전.
        resolved: ``False`` 면 ``legacy_unresolved``. 게이트를 미판정으로 둔다.
    """

    selection_id: str
    family: str
    candidate_labels: list[str] = field(default_factory=list)
    candidate_scores: dict[str, float] = field(default_factory=dict)
    selection_metric: str = "median_log_improvement"
    tie_break_rule: str = "lowest_flat_index"
    selected_label: str = ""
    selected_config: dict[str, Any] = field(default_factory=dict)
    tuning_specs: list[str] = field(default_factory=list)
    tuning_seeds: list[int] = field(default_factory=list)
    n_tune: int = 0
    semantics_version: int = SELECTION_SEMANTICS_VERSION
    resolved: bool = True

    # --- 튜닝 비용. planner 의 decision-search GE 와 성격이 다르므로 분리한다 ---
    n_candidates: int = 0
    n_tuning_instances: int = 0
    tuning_object_ge: float = 0.0
    """baseline 선택에 투입한 **object-level** GE 총합.

    P2 에서 최종 성능만 보여주면 baseline 이 얼마나 튜닝됐는지 숨겨진다.
    planner 의 ``search_cost_ge`` 와는 별개 열로 보고한다 (프로토콜 D16).
    """
    tuning_wall_clock_sec: float = 0.0
    """운영 기록. CPU 공유 영향을 받으므로 성능 비교에 쓰지 않는다."""

    # --- open_loop 이 static 으로 퇴화한 경우 ---
    equivalent_static_action: dict[str, Any] | None = None
    """constant schedule 이면 그와 동일한 action. **label 이 아니라 실제 값**이다.

    open_loop 과 static 의 후보 grid 가 다를 수 있으므로 라벨 비교는 무의미하다.
    """
    progress_clock: str = ""
    """스케줄 시계 정의 (프로토콜 D17). ``object_ge_fraction`` 이어야 한다."""
    progress_evaluated_at: str = "before_step"
    realized_segment_counts: dict[str, int] = field(default_factory=dict)
    """구간별 실행 step 수. 스케줄이 실제로 얼마나 쓰였는지 보여준다 (D17).

    비싼 action 하나가 breakpoint 를 건너뛰면 특정 구간이 실행되지 않을 수 있다.
    오류는 아니지만 기록해야 한다.
    """
    realized_ge_by_segment: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return sanitize_for_json(asdict(self))

    def describe(self) -> str:
        if not self.resolved:
            return f"{self.family}: legacy_unresolved (선택 근거 복원 불가)"
        best = self.candidate_scores.get(self.selected_label, float("nan"))
        # 동점 후보를 명시한다. tie-break 가 실제로 작동했는지 보여야 한다.
        ties = [
            label
            for label, score in self.candidate_scores.items()
            if label != self.selected_label and score == best
        ]
        lines = [
            f"{self.family}: {self.selected_label} 선택 "
            f"({self.selection_metric}={best:.4f}, 후보 {self.n_candidates}개, "
            f"tie-break={self.tie_break_rule})",
            f"    설정 {self.selected_config}",
            f"    튜닝 비용 object-level {self.tuning_object_ge:.0f} GE "
            f"(인스턴스 {self.n_tuning_instances}개), selection_id={self.selection_id[:12]}",
        ]
        if ties:
            lines.append(f"    동점 후보 {ties} -> {self.tie_break_rule} 로 선택")
        if self.progress_clock:
            n_seg = len(self.selected_config.get("schedule", []) or [])
            if not self.realized_segment_counts:
                # 캐시 재사용 시 컨트롤러가 생성되지 않아 계측값이 없다.
                # "구간이 실행되지 않았다" 와 구별해야 한다.
                lines.append(
                    f"    시계={self.progress_clock}, 구간 {n_seg}개. "
                    "구간 사용량 미계측 (캐시 재사용)"
                )
            else:
                used = sum(self.realized_segment_counts.values())
                reached = len(self.realized_segment_counts)
                lines.append(
                    f"    시계={self.progress_clock}, 실행된 구간 {reached}/{n_seg} "
                    f"(총 {used} step) {self.realized_segment_counts}"
                )
                if n_seg and reached < n_seg:
                    lines.append(
                        "    주의: 일부 구간이 실행되지 않았다. 비싼 action 이 "
                        "breakpoint 를 건너뛴 것이며 오류는 아니다."
                    )
        if self.equivalent_static_action is not None:
            lines.append(
                f"    **constant schedule 이다. static 으로 퇴화했다.** "
                f"등가 action {self.equivalent_static_action}. "
                f"P2 에서 best_static 과 독립 baseline 으로 세지 않는다 (D16)."
            )
        return "\n".join(lines)

    @property
    def is_constant_schedule(self) -> bool:
        """open_loop 이 static 으로 퇴화했는가 (프로토콜 D16).

        모든 구간의 action 이 같으면 튜닝된 open-loop 가 비정적 스케줄의 이점을
        찾지 못하고 최적 static 설정으로 퇴화한 것이다. 버그가 아니라 결과다.
        단 **P2 에서 두 개의 독립적인 강한 baseline 으로 세면 안 된다.**
        """
        actions = self.selected_config.get("schedule")
        if not isinstance(actions, list) or not actions:
            return False
        return all(a == actions[0] for a in actions)


@dataclass(frozen=True, slots=True)
class RunKey:
    """run 하나를 유일하게 식별한다. 재개 판단의 기준이다.

    ``experiment_id`` 가 앞에 오는 것이 중요하다. 설정이 달라지면 같은
    ``(controller, task, seed, target)`` 이라도 다른 run 으로 취급된다.
    """

    experiment_id: str
    controller: str
    task_instance_id: str
    seed: int
    target: str

    def as_str(self) -> str:
        return (
            f"{self.experiment_id}|{self.controller}|{self.task_instance_id}"
            f"|{self.seed}|{self.target}"
        )

    @classmethod
    def from_summary(cls, summary: RunSummary, experiment_id: str) -> RunKey:
        return cls(
            experiment_id=experiment_id,
            controller=summary.controller,
            task_instance_id=summary.task_instance_id,
            seed=summary.seed,
            target=summary.target,
        )


@dataclass(slots=True)
class RunRecord:
    """저장되는 한 줄.

    Attributes:
        key: 식별자.
        status: ``completed`` 또는 ``failed``.
        summary: 정상 종료 시의 ``RunSummary``. 실패면 ``None``.
        wall_clock_sec: 실제 소요 시간. GE 와 별개로 기록한다 (프로토콜 D1).
        action_counts: 선택한 action 의 빈도. 정책 분석용 (README §8).
        chosen_depths: planner 가 채택한 계획 길이의 빈도. 게이트 C 해석용.
        planner_stats: planner 진단값. ``depth_cap_hit`` (계산 상한에 걸린
            step 비율), ``mean_quota_used`` (쿼터 소진율), ``mean_simulations``
            등이 들어간다. ``depth_cap_hit`` 가 0 이 아니면 쿼터 사다리 비교가
            훼손되므로 게이트 C 보고에 포함해야 한다 (프로토콜 D10).
        error: 실패 원인.
        recorded_at: 기록 시각.
        git_commit: 코드 버전.
        config_hash: 설정 해시.
    """

    key: RunKey
    status: RunStatus
    summary: RunSummary | None = None
    wall_clock_sec: float = float("nan")
    action_counts: dict[str, int] | None = None
    chosen_depths: dict[str, int] | None = None
    planner_stats: dict[str, float] | None = None
    error: str | None = None
    recorded_at: str = ""
    git_commit: str = ""
    config_hash: str = ""

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": asdict(self.key),
            "key_str": self.key.as_str(),
            "status": self.status,
            "wall_clock_sec": self.wall_clock_sec,
            "action_counts": self.action_counts,
            "chosen_depths": self.chosen_depths,
            "planner_stats": self.planner_stats,
            "error": self.error,
            "recorded_at": self.recorded_at or datetime.now(UTC).isoformat(),
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "summary": asdict(self.summary) if self.summary is not None else None,
        }
        return sanitize_for_json(payload)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RunRecord:
        summary_fields = {f.name for f in fields(RunSummary)}
        raw = payload.get("summary")
        summary = None
        if raw is not None:
            # JSON 에서 비유한값은 None 으로 저장되므로 float 필드를 복원한다.
            restored = {k: v for k, v in raw.items() if k in summary_fields}
            for name in summary_fields:
                if name not in restored:
                    restored[name] = None
            summary = _restore_summary(restored)
        return cls(
            key=RunKey(**payload["key"]),
            status=payload["status"],
            summary=summary,
            wall_clock_sec=_as_float(payload.get("wall_clock_sec")),
            action_counts=payload.get("action_counts"),
            chosen_depths=payload.get("chosen_depths"),
            planner_stats=payload.get("planner_stats"),
            error=payload.get("error"),
            recorded_at=payload.get("recorded_at", ""),
            git_commit=payload.get("git_commit", ""),
            config_hash=payload.get("config_hash", ""),
        )


def _as_float(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _restore_summary(raw: dict[str, Any]) -> RunSummary:
    """JSON 에서 ``RunSummary`` 를 복원한다.

    ``None`` 처리를 필드 타입에 따라 나눈다. 이것이 **절단 규칙의 핵심**이다.

    ```text
    float          NaN 이 "측정 불가" 를 뜻한다. JSON 의 null -> NaN
    float | None   None 이 "미도달" 을 뜻한다.   JSON 의 null -> None 유지
    ```

    ``cost_to_target_ge`` 를 NaN 으로 바꾸면 "목표에 도달하지 못했다"는 정보가
    "값을 모른다"로 바뀐다. 프로토콜 D6은 미도달 run 을 큰 값으로 대입하지도,
    삭제하지도 않는다고 규정하므로 이 구분을 보존해야 한다.
    """
    optional_fields = {
        name
        for name, annotation in ((f.name, str(f.type)) for f in fields(RunSummary))
        if "None" in annotation
    }
    kwargs: dict[str, Any] = {}
    for spec in fields(RunSummary):
        value = raw.get(spec.name)
        if value is None and str(spec.type) == "float":
            value = float("nan")
        elif value is None and spec.name in optional_fields:
            value = None
        kwargs[spec.name] = value
    # 필수 필드 타입 보정
    kwargs["reached"] = bool(kwargs.get("reached"))
    kwargs["seed"] = int(kwargs.get("seed") or 0)
    kwargs["n_steps"] = int(kwargs.get("n_steps") or 0)
    kwargs["total_hvp"] = int(kwargs.get("total_hvp") or 0)
    return RunSummary(**kwargs)


class ResultStore:
    """append-only JSONL 기반 재개 가능한 저장소.

    Args:
        path: JSONL 경로. 없으면 만든다.
        git_commit: 기록에 붙일 코드 버전.
        config_hash: 기록에 붙일 설정 해시.

    Example:
        >>> import tempfile, pathlib
        >>> with tempfile.TemporaryDirectory() as d:
        ...     store = ResultStore(pathlib.Path(d) / "runs.jsonl")
        ...     key = RunKey("fixed", "quad_seed0", 0, "relative_loss<=1e-06")
        ...     store.is_completed(key)
        False
    """

    def __init__(
        self,
        path: str | Path,
        *,
        git_commit: str = "",
        config_hash: str = "",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.git_commit = git_commit
        self.config_hash = config_hash
        self._records: dict[str, RunRecord] = {}
        self._load()

    # --- 로드 -------------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                record = RunRecord.from_json(payload)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                # 프로세스가 쓰는 중 끊기면 마지막 줄이 깨질 수 있다.
                # 그 줄만 버리고 나머지는 살린다.
                print(
                    f"경고: {self.path}:{line_number} 를 읽을 수 없어 건너뛴다 ({exc})",
                    flush=True,
                )
                continue
            # 같은 키가 여러 번 있으면 마지막 것이 유효하다 (재시도 결과).
            self._records[record.key.as_str()] = record

    # --- 조회 -------------------------------------------------------------

    def is_completed(self, key: RunKey) -> bool:
        """정상 완료된 run 인지. ``True`` 면 재실행에서 건너뛴다."""
        record = self._records.get(key.as_str())
        return record is not None and record.status == "completed"

    def get(self, key: RunKey) -> RunRecord | None:
        return self._records.get(key.as_str())

    def summaries(self, *, controller: str | None = None) -> list[RunSummary]:
        """완료된 run 의 ``RunSummary`` 목록. 집계 입력이다."""
        out: list[RunSummary] = []
        for record in self._records.values():
            if record.status != "completed" or record.summary is None:
                continue
            if controller is not None and record.key.controller != controller:
                continue
            out.append(record.summary)
        return out

    def failures(self) -> list[RunRecord]:
        """실패한 run 들. 삭제하지 않고 원인을 보존한다 (README §15)."""
        return [r for r in self._records.values() if r.status == "failed"]

    def controllers(self) -> list[str]:
        return sorted({r.key.controller for r in self._records.values()})

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self._records.values())

    # --- 기록 -------------------------------------------------------------

    def put(self, record: RunRecord) -> None:
        """한 줄 추가하고 즉시 flush 한다.

        프로세스가 다음 run 에서 끊겨도 이 결과는 보존된다.
        """
        if not record.git_commit:
            record.git_commit = self.git_commit
        if not record.config_hash:
            record.config_hash = self.config_hash
        if not record.recorded_at:
            record.recorded_at = datetime.now(UTC).isoformat()

        line = json.dumps(record.to_json(), ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
        self._records[record.key.as_str()] = record

    def record_success(
        self,
        summary: RunSummary,
        experiment_id: str,
        *,
        wall_clock_sec: float,
        action_counts: dict[str, int] | None = None,
        chosen_depths: dict[str, int] | None = None,
        planner_stats: dict[str, float] | None = None,
    ) -> None:
        self.put(
            RunRecord(
                key=RunKey.from_summary(summary, experiment_id),
                status="completed",
                summary=summary,
                wall_clock_sec=wall_clock_sec,
                action_counts=action_counts,
                chosen_depths=chosen_depths,
                planner_stats=planner_stats,
            )
        )

    def record_failure(
        self, key: RunKey, error: str, *, wall_clock_sec: float = float("nan")
    ) -> None:
        self.put(
            RunRecord(
                key=key,
                status="failed",
                error=error,
                wall_clock_sec=wall_clock_sec,
            )
        )

    def describe(self) -> str:
        completed = sum(1 for r in self._records.values() if r.status == "completed")
        failed = len(self._records) - completed
        return (
            f"ResultStore({self.path.name}): 완료 {completed}, 실패 {failed}, "
            f"컨트롤러 {len(self.controllers())}종"
        )


# ---------------------------------------------------------------------------
# 실행 환경 기록과 CPU 스레드 고정
# ---------------------------------------------------------------------------


def environment_fingerprint(*, pin_threads: int | None = 1) -> dict[str, Any]:
    """실행 환경을 기록하고 필요하면 CPU 스레드를 고정한다.

    GE 가 주 지표이므로 핵심 결론은 CPU 경쟁에 영향받지 않는다. 그러나 beam
    선택의 마지막 tie-break 로 wall-clock 을 쓰므로, 다른 실험과 CPU 를
    공유하면 그 tie-break 가 흔들린다.

    ``pin_threads`` 를 주면 ``torch.set_num_threads`` 와
    ``set_num_interop_threads`` 를 고정한다. 이미 병렬 영역이 시작된 뒤에는
    interop 설정이 실패할 수 있으므로 예외를 삼키고 실제 값을 기록한다.

    Args:
        pin_threads: 고정할 스레드 수. ``None`` 이면 건드리지 않는다.

    Returns:
        기록용 dict. 동시 실행 프로세스 수는 관측할 수 없으므로 호출자가
        알고 있다면 별도로 넣는다.
    """
    import contextlib
    import os
    import platform

    import torch

    if pin_threads is not None:
        torch.set_num_threads(pin_threads)
        # 이미 병렬 영역이 초기화된 뒤에는 interop 설정이 실패한다.
        # 실제 값을 아래에서 기록하므로 삼켜도 정보가 사라지지 않는다.
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(1)

    return {
        "platform": f"{platform.system()} {platform.release()}",
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "pinned": pin_threads,
    }
