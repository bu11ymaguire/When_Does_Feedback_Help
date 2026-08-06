"""진단: paired comparison 에서 조용히 빠진 쌍의 원시값을 출력한다.

beam 4 screening 에서 ``onestep_absolute`` 와 ``heuristic`` 이 ``쌍 6/9`` 였다.
``compare_paired_delta`` 는 target 을 참조하지 않고 ``math.isfinite`` 만 보므로,
9쌍이 모두 성립했고 그중 3쌍의 ``log_improvement`` 가 비유한값이었다는 뜻이다.

``RunSummary.log_improvement`` 는 다음이면 NaN 을 반환한다.

```python
if self.initial_loss <= 0.0 or self.final_loss <= 0.0:
    return float("nan")
```

즉 ``final_loss`` 가 0 이하로 내려간 쌍이 빠졌다. **가장 잘 최적화된 run 이
집계에서 제거되는 편향이다.** ``onestep_absolute`` 는 Track E 에서 가장 강한
컨트롤러이므로 영향이 크다.

이 스크립트는 원인을 분류만 한다. 정책 결정은 프로토콜에서 한다.

```text
final_loss == 0                     underflow
-floor <= final_loss < 0            부동소수점 roundoff
final_loss < -floor                 numerical failure
NaN / inf                           numerical failure
```

사용법:
    python scripts/diagnose_excluded_pairs.py results/raw/<file>.jsonl
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

# float64 기준 상대 floor. 결과를 보고 고른 값이 아니라 dtype 특성에서 나온다.
# finfo.tiny (약 2.2e-308) 를 그대로 쓰면 최대 log improvement 가 708 nat 까지
# 커져서 underflow 여부가 통계를 지배한다. 초기 loss 에 상대적으로 잡는다.
_EPS = torch.finfo(torch.float64).eps
_TINY = torch.finfo(torch.float64).tiny
RELATIVE_FLOOR = 100.0 * _EPS


def loss_floor(initial_loss: float) -> float:
    """``max(L0 * 100eps, tiny)``. scale invariant 한 수치 하한."""
    return max(_TINY, abs(initial_loss) * RELATIVE_FLOOR)


def classify(initial_loss: float, final_loss: float) -> str:
    if not math.isfinite(final_loss):
        return "non-finite (NaN/inf)"
    floor = loss_floor(initial_loss)
    if final_loss > 0.0:
        return "정상" if final_loss > floor else f"floor 이하 (>0, floor={floor:.3e})"
    if final_loss == 0.0:
        return "정확히 0 (underflow)"
    if final_loss >= -floor:
        return f"작은 음수 roundoff (floor={floor:.3e})"
    return "허용 범위 초과 음수"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"없음: {path}")
        return 1

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"{path.name}   기록 {len(records)}개")
    print(f"상대 floor = 100 x eps = {RELATIVE_FLOOR:.3e}   tiny = {_TINY:.3e}")
    print()

    # 컨트롤러별로 log_improvement 가 비유한값인 run 을 모은다.
    problems: dict[str, list[dict]] = {}
    counts: dict[str, dict[str, int]] = {}
    for rec in records:
        summary = rec.get("summary")
        if rec.get("status") != "completed" or not summary:
            continue
        controller = rec["key"]["controller"]
        l0 = summary.get("initial_loss")
        lt = summary.get("final_loss")
        if l0 is None or lt is None:
            kind = "필드 없음 (JSON None)"
        elif l0 <= 0.0 or lt <= 0.0 or not math.isfinite(lt):
            kind = classify(l0, lt)
        else:
            counts.setdefault(controller, {})["정상"] = (
                counts.setdefault(controller, {}).get("정상", 0) + 1
            )
            continue
        counts.setdefault(controller, {})[kind] = (
            counts.setdefault(controller, {}).get(kind, 0) + 1
        )
        problems.setdefault(controller, []).append(rec)

    if not problems:
        print("비유한 log_improvement 없음.")
        return 0

    print("=== 컨트롤러별 분류 ===")
    for controller in sorted(counts):
        kinds = counts[controller]
        if len(kinds) == 1 and "정상" in kinds:
            continue
        parts = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        print(f"  {controller:<26} {parts}")

    print()
    print("=== 빠진 run 의 원시값 ===")
    for controller in sorted(problems):
        print(f"\n  [{controller}]")
        for rec in problems[controller]:
            s = rec["summary"]
            key = rec["key"]
            print(
                f"    {key['task_instance_id']:<34} seed={key['seed']}  "
                f"target={key['target']}"
            )
            print(
                f"      initial_loss={s.get('initial_loss')!r}  "
                f"final_loss={s.get('final_loss')!r}"
            )
            print(
                f"      n_steps={s.get('n_steps')}  object_GE={s.get('total_cost_ge')}  "
                f"stop={s.get('stop_reason')!r}  reached={s.get('reached')}  "
                f"cost_to_target={s.get('cost_to_target_ge')!r}"
            )
            print(
                f"      failure_rate={s.get('failure_rate')}  "
                f"rejection_rate={s.get('rejection_rate')}  "
                f"분류={classify(s.get('initial_loss') or 0.0, s.get('final_loss') or 0.0)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
