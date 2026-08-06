"""Bridge 검증: D13·D16·D17 이후 planner 실행 궤적이 legacy pilot 과 같은가.

D13(3계층 정체성), D16(selection manifest), D17(open-loop resource clock) 은
**실행 의미를 바꾸지 않아야 한다.** identity / logging / baseline 시계만 고쳤다.
그러면 planner trajectory 는 bitwise 동일해야 한다.

동일 CPU, float64, 단일 스레드, 동일 seed 이므로 **기대값은 bitwise exact** 다.
tolerance 는 bitwise 가 깨졌을 때만 보조로 쓴다.

분류 (실행 전 고정)
-------------------
```text
EXACT                     bitwise 동일
CLOSE                     수치 tolerance 만 통과
MISMATCH                  tolerance 도 실패
ZERO_MISMATCH             한쪽만 정확히 0. saturation 상태가 바뀌므로 별도로 센다
LEGACY_FIELD_MISSING      legacy 에 새 계측 필드가 없음. 실패로 세지 않는다
LEGACY_MISSING            expected 키가 legacy 에 없음. 실패
NEW_MISSING               expected 키가 new 에 없음. 실패
LEGACY_ONLY_OUT_OF_SCOPE  legacy 에만 있고 expected 가 아님. **실패가 아니다**
DUPLICATE                 한 키에 2행 이상. 비교하지 않는다. 실패
```

**의도된 범위 축소와 진짜 누락을 구별한다.** bridge 범위를 좁히면 legacy 에만
있는 키가 당연히 생긴다. 그것을 실패로 세면 통과가 불가능하다.

통과 조건
---------
```text
expected 키 전부가 legacy 와 new 양쪽에 존재
모든 공통 이산 필드 exact match
loss 는 bitwise 또는 tolerance 충족
MISMATCH / ZERO_MISMATCH / DUPLICATE 가 0개
```

**중복을 평균하거나 최신 timestamp 로 임의 선택하지 않는다.** 같은 논리적 run 이
여러 ``experiment_id`` 에 존재하므로 legacy 기준을 명시적으로 고정한다.

사용법:
    python scripts/bridge_validate.py --legacy <exp_prefix> --new <exp_prefix> <raw.jsonl...>

sweep 커버리지가 달라지면 표시용 ``config_hash`` 가 바뀌어 raw 파일이 분리된다.
그래서 여러 파일을 함께 읽는다. ``run_semantics_id`` 가 정체성을 보장하므로
파일 경계는 비교에 영향을 주지 않는다.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

from rl_newton.benchmark.store import ResultStore

# 수치 tolerance. bitwise 가 깨졌을 때만 쓴다 (프로토콜 D18).
REL_TOL = 1.0e-12
ABS_TOL = 1.0e-14

# 실행 의미가 같으면 exact match 가 기대되는 이산/비용 필드.
EXACT_FIELDS = (
    "total_cost_ge",
    "n_steps",
    "stop_reason",
    "total_hvp",
    "search_cost_ge",
)
# 수치 비교 필드.
NUMERIC_FIELDS = ("initial_loss", "final_loss")
# legacy 에 없을 수 있는 새 계측값. 실패로 세지 않는다.
NEW_ONLY_STATS = ("suffix_retention_rate", "n_replans", "windows")

# planner_stats 중 실행 의미가 같으면 일치해야 하는 것.
# mean_simulations 가 달라지면 탐색 순서 / pruning / incumbent carry-over 가
# 바뀐 것이므로 로깅 차이로 넘기지 않는다.
STATS_EXACT = ("mean_simulations", "depth_cap_hit", "quota_ge", "max_depth_seen")

_LABEL = re.compile(r"^(?P<mode>shrinking|committed|fresh)_Q(?P<quota>[\d.]+)_(?P<space>\w+)$")


def parse_label(label: str) -> tuple[str, str, str] | None:
    m = _LABEL.match(label)
    if not m:
        return None
    return m["mode"], m["quota"], m["space"]


def classify_number(legacy: float, new: float) -> str:
    l_zero, n_zero = legacy == 0.0, new == 0.0
    if l_zero != n_zero:
        # 0 과 작은 양수를 tolerance 로 같다고 넘기면 saturation 이 바뀐다.
        return "ZERO_MISMATCH"
    if legacy == new:
        return "EXACT"
    if not (math.isfinite(legacy) and math.isfinite(new)):
        return "EXACT" if repr(legacy) == repr(new) else "MISMATCH"
    if math.isclose(legacy, new, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        return "CLOSE"
    return "MISMATCH"


def collect(records, prefixes: tuple[str, ...]) -> dict[tuple, list]:
    out: dict[tuple, list] = {}
    for rec in records:
        if rec.status != "completed" or rec.summary is None:
            continue
        if not any(rec.key.experiment_id.startswith(p) for p in prefixes):
            continue
        parsed = parse_label(rec.key.controller)
        if parsed is None:
            continue
        key = (*parsed, rec.key.task_instance_id, rec.key.seed)
        out.setdefault(key, []).append(rec)
    return out


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-file",
        required=True,
        type=Path,
        help="legacy raw jsonl. 기준 run 이 여기 있다",
    )
    parser.add_argument(
        "--legacy",
        required=True,
        help="legacy 기준 experiment_id 접두사. 여러 정체성이 섞인 파일에서 하나를 고정한다",
    )
    parser.add_argument("--new-file", required=True, type=Path, help="새 raw jsonl")
    # expected 키 집합을 명시한다. 이것이 없으면 범위 밖과 누락을 구별할 수 없다.
    parser.add_argument("--modes", nargs="+", default=["shrinking", "committed"])
    parser.add_argument("--quotas", nargs="+", default=["2", "4"])
    parser.add_argument("--spaces", nargs="+", default=["narrow", "wide"])
    args = parser.parse_args(argv[1:])

    for path in (args.legacy_file, args.new_file):
        if not path.exists():
            print(f"없음: {path}")
            return 1

    legacy_prefix = args.legacy
    # legacy 는 하나의 정체성으로 고정한다. new 는 컨트롤러마다 run_semantics_id
    # 가 다르므로 파일 전체를 받는다 (D13 이후 정상 동작이다).
    legacy = collect(ResultStore(args.legacy_file), (legacy_prefix,))
    new = collect(ResultStore(args.new_file), ("",))

    print(f"legacy={args.legacy_file.name}  new={args.new_file.name}")
    print(f"  legacy_reference_experiment_id = {legacy_prefix}")
    print("  new 는 컨트롤러별 run_semantics_id 전체를 사용한다")
    print(f"  legacy 키 {len(legacy)}개, new 키 {len(new)}개")
    print(f"  tolerance  rel={REL_TOL:g}  abs={ABS_TOL:g}  (bitwise 우선)")
    print()

    # expected 키 = 사전 등록된 bridge 범위. task/seed 는 양쪽에 나타난 것을 쓴다.
    instances = {
        (task, seed) for (_m, _q, _s, task, seed) in set(legacy) | set(new)
    }
    expected = {
        (mode, quota, space, task, seed)
        for mode in args.modes
        for quota in args.quotas
        for space in args.spaces
        for task, seed in instances
    }
    print(f"  expected 키 {len(expected)}개 "
          f"(modes={args.modes} quotas={args.quotas} spaces={args.spaces})")
    print()

    tally: dict[str, int] = {}
    first_mismatch: tuple | None = None
    details: list[str] = []

    # expected 가 아닌 legacy 키는 범위 밖이다. 실패가 아니다.
    out_of_scope = sorted(set(legacy) - expected)
    if out_of_scope:
        tally["LEGACY_ONLY_OUT_OF_SCOPE"] = len(out_of_scope)

    for key in sorted(expected):
        lrows, nrows = legacy.get(key, []), new.get(key, [])
        if not lrows:
            tally["LEGACY_MISSING"] = tally.get("LEGACY_MISSING", 0) + 1
            details.append(f"  LEGACY_MISSING {key}")
            continue
        if not nrows:
            tally["NEW_MISSING"] = tally.get("NEW_MISSING", 0) + 1
            details.append(f"  NEW_MISSING {key}")
            continue
        if len(lrows) > 1 or len(nrows) > 1:
            # 평균하거나 최신 것을 고르지 않는다. 비교 자체를 보류한다.
            tally["DUPLICATE"] = tally.get("DUPLICATE", 0) + 1
            details.append(
                f"  DUPLICATE {key}: legacy {len(lrows)}행, new {len(nrows)}행"
            )
            continue

        lr, nr = lrows[0], nrows[0]
        ls, ns = lr.summary, nr.summary
        problems: list[str] = []

        for field in EXACT_FIELDS:
            lv, nv = getattr(ls, field), getattr(ns, field)
            if isinstance(lv, float) and isinstance(nv, float):
                if not (lv == nv or (math.isnan(lv) and math.isnan(nv))):
                    problems.append(f"{field}: {lv!r} != {nv!r}")
            elif lv != nv:
                problems.append(f"{field}: {lv!r} != {nv!r}")

        worst = "EXACT"
        for field in NUMERIC_FIELDS:
            verdict = classify_number(getattr(ls, field), getattr(ns, field))
            if verdict == "ZERO_MISMATCH":
                problems.append(
                    f"{field}: 한쪽만 0 ({getattr(ls, field)!r} vs {getattr(ns, field)!r})"
                )
                worst = "ZERO_MISMATCH"
            elif verdict == "MISMATCH":
                problems.append(
                    f"{field}: {getattr(ls, field)!r} != {getattr(ns, field)!r}"
                )
                worst = "MISMATCH"
            elif verdict == "CLOSE" and worst == "EXACT":
                worst = "CLOSE"

        if lr.action_counts != nr.action_counts:
            problems.append(f"action_counts: {lr.action_counts} != {nr.action_counts}")
        if lr.chosen_depths != nr.chosen_depths:
            problems.append(f"chosen_depths: {lr.chosen_depths} != {nr.chosen_depths}")

        lstats, nstats = lr.planner_stats or {}, nr.planner_stats or {}
        missing_new_fields = False
        for field in STATS_EXACT:
            if field not in lstats:
                missing_new_fields = True
                continue
            if field not in nstats:
                problems.append(f"planner_stats.{field}: new 에 없음")
                continue
            if lstats[field] != nstats[field]:
                # 로깅 차이로 넘기지 않는다. 탐색 순서나 pruning 이 바뀐 것이다.
                problems.append(
                    f"planner_stats.{field}: {lstats[field]!r} != {nstats[field]!r}"
                )
        for field in NEW_ONLY_STATS:
            if field not in lstats and field in nstats:
                missing_new_fields = True

        if problems:
            verdict = worst if worst in ("ZERO_MISMATCH",) else "MISMATCH"
            if first_mismatch is None:
                first_mismatch = key
            details.append(f"  {verdict} {key}")
            for p in problems[:6]:
                details.append(f"      {p}")
        else:
            verdict = worst
            if missing_new_fields and verdict == "EXACT":
                verdict = "EXACT+LEGACY_FIELD_MISSING"
        tally[verdict] = tally.get(verdict, 0) + 1

    print("=== 분류 집계 ===")
    for name in sorted(tally):
        print(f"  {name:<28} {tally[name]}")
    if details:
        print()
        print("=== 상세 ===")
        for line in details[:60]:
            print(line)

    # LEGACY_ONLY_OUT_OF_SCOPE 는 실패가 아니다 (의도된 범위 축소).
    bad = sum(
        tally.get(k, 0)
        for k in (
            "MISMATCH",
            "ZERO_MISMATCH",
            "DUPLICATE",
            "LEGACY_MISSING",
            "NEW_MISSING",
        )
    )
    compared = sum(
        v for k, v in tally.items() if k.startswith(("EXACT", "CLOSE"))
    )
    print()
    print(f"  비교 완료 {compared}/{len(expected)} expected 키")
    if bad == 0 and compared == len(expected):
        print("  BRIDGE 통과: expected 전부 비교됨, 설명되지 않는 불일치 0개.")
        print("  D13/D16/D17 은 planner 실행 궤적을 바꾸지 않았다.")
    else:
        print(f"  BRIDGE 실패: 문제 {bad}개.")
        if first_mismatch is not None:
            print(f"  최초 불일치 키: {first_mismatch}")
            print("  최종 loss 부터 역추적하지 말고 첫 action / 초기 계획 /")
            print("  첫 planner 후보 점수 / remaining quota / 첫 step GE 순으로 본다.")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
