"""실행 중인 headroom run 의 진행 상황을 raw jsonl 에서 읽는다.

셸 리다이렉트 로그는 인코딩이 깨지는 경우가 있어 raw 기록을 직접 본다.

사용법:
    python scripts/show_progress.py results/raw/<file>.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"없음: {path}")
        return 1

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    by_controller: dict[str, list[dict]] = {}
    by_experiment: dict[str, int] = {}
    n_failed = 0
    for rec in records:
        exp = rec["key"].get("experiment_id", "?")
        by_experiment[exp] = by_experiment.get(exp, 0) + 1
        if rec.get("status") != "completed":
            n_failed += 1
            continue
        by_controller.setdefault(rec["key"]["controller"], []).append(rec)

    print(f"{path.name}")
    print(f"  기록 {len(records)}개, 실패 {n_failed}개, 컨트롤러 {len(by_controller)}종")
    # 한 파일에 여러 experiment_id 가 섞일 수 있다. RunKey 가 experiment_id 를
    # 포함하므로 재사용은 안전하지만, 몇 개 정체성이 섞였는지는 봐야 한다.
    print(f"  experiment_id {len(by_experiment)}종: ", end="")
    print(", ".join(f"{e[:12]}={n}" for e, n in sorted(by_experiment.items())))
    print()
    print(f"  {'controller':<28} {'run':>4} {'object GE':>10} {'search GE':>12} {'wall(s)':>8}")
    print(f"  {'-' * 28} {'-' * 4} {'-' * 10} {'-' * 12} {'-' * 8}")
    for name in sorted(by_controller):
        runs = by_controller[name]
        obj = [r["summary"]["total_cost_ge"] for r in runs if r.get("summary")]
        search = [r["summary"]["search_cost_ge"] for r in runs if r.get("summary")]
        wall = [r["wall_clock_sec"] for r in runs]
        print(
            f"  {name:<28} {len(runs):>4} "
            f"{(sum(obj) / len(obj) if obj else 0):>10.1f} "
            f"{(sum(search) / len(search) if search else 0):>12.0f} "
            f"{(sum(wall) / len(wall) if wall else 0):>8.1f}"
        )
    total_search = sum(r["summary"]["search_cost_ge"] for r in records if r.get("summary"))
    total_obj = sum(r["summary"]["total_cost_ge"] for r in records if r.get("summary"))
    total_wall = sum(r["wall_clock_sec"] for r in records if r.get("wall_clock_sec"))
    print()
    print(f"  object-level GE 합계   {total_obj:>14.0f}")
    print(f"  decision-search GE 합계 {total_search:>13.0f}")
    print(f"  누적 wall-clock         {total_wall / 60.0:>12.1f} 분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
