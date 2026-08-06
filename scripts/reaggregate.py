"""저장된 raw 결과를 **재실행 없이** 새 집계 정책으로 다시 계산한다.

왜 필요한가
-----------
``experiment_id`` 에 git commit 과 code-dirty 해시가 들어 있어서, 집계 코드만
바꿔도 모든 run 이 무효화되고 optimizer 가 다시 돈다. 그런데 floor 정책(D14)은
``RunSummary.initial_loss`` / ``final_loss`` 에서 파생되는 **property** 이고 그
두 값은 이미 저장되어 있다. 따라서 raw 기록만 다시 읽으면 된다.

프로토콜 D13 이 이 문제의 근본 수정이다 (run_semantics_id / sweep_id 분리).
이 스크립트는 그 전까지 쓰는 재집계 경로다.

사용법:
    python scripts/reaggregate.py results/raw/<file>.jsonl [experiment_id_prefix]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from rl_newton.benchmark.metrics import (
    RunSummary,
    compare_paired_delta,
    drop_saturated_pairs,
)
from rl_newton.benchmark.store import ResultStore

# 게이트 정의 (프로토콜 D12/D14). (이름, baseline, treatment, GO, pivot)
GATES = [
    ("A1", "best_static", "onestep_absolute", 1.0, 0.3),
    ("A2-narrow", "best_static", "shrinking_Q4_narrow", 0.7, 0.2),
    ("A2-wide", "best_static", "shrinking_Q4_wide", 0.7, 0.2),
    ("B", "onestep_narrow", "onestep_absolute", 0.5, 0.1),
    ("B-wide", "onestep_narrow", "onestep_wide", 0.5, 0.1),
    ("C1", "fresh_Q4_narrow", "shrinking_Q4_narrow", 0.3, 0.05),
    ("C2", "onestep_narrow", "shrinking_Q4_narrow", 0.3, 0.05),
    ("C3", "committed_Q4_narrow", "shrinking_Q4_narrow", 0.3, 0.05),
    ("C2-wide", "onestep_wide", "shrinking_Q4_wide", 0.3, 0.05),
    ("C3-wide", "committed_Q4_wide", "shrinking_Q4_wide", 0.3, 0.05),
    ("ref-Q1", "onestep_narrow", "shrinking_Q1_narrow", 0.3, 0.05),
    ("ref-openloop", "best_static", "best_open_loop", 0.3, 0.05),
    ("ref-heuristic", "best_static", "heuristic", 0.3, 0.05),
]


def verdict(stat: float, go: float, pivot: float) -> str:
    if not math.isfinite(stat):
        return "판정불가"
    if stat >= go:
        return "GO"
    if stat < pivot:
        return "재설계"
    return "조건부"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    prefix = argv[2] if len(argv) > 2 else None

    store = ResultStore(path)
    by_controller: dict[str, list[RunSummary]] = {}
    exp_ids: set[str] = set()
    for record in store:
        if record.status != "completed" or record.summary is None:
            continue
        if prefix and not record.key.experiment_id.startswith(prefix):
            continue
        exp_ids.add(record.key.experiment_id)
        by_controller.setdefault(record.key.controller, []).append(record.summary)

    print(f"{path.name}")
    print(f"  experiment_id {len(exp_ids)}종, 컨트롤러 {len(by_controller)}종")
    if len(exp_ids) > 1:
        print("  주의: 여러 정체성이 섞여 있다. prefix 인자로 하나만 골라라.")
        for e in sorted(exp_ids):
            print(f"    {e}")
    print()

    # best_static / best_open_loop 는 탐색 후 선택된 설정이므로 라벨이 다르다.
    # 저장된 라벨을 그대로 쓴다.
    print("=== 컨트롤러별 Track E 지표 (floor-capped, 프로토콜 D14) ===")
    print(
        f"  {'controller':<26} {'n':>3} {'median logΔ':>12} {'floor_hit':>10} "
        f"{'exact0':>7} {'obj GE':>8}"
    )
    for name in sorted(by_controller):
        runs = by_controller[name]
        vals = [r.log_improvement for r in runs if math.isfinite(r.log_improvement)]
        n_floor = sum(1 for r in runs if r.floor_hit)
        n_zero = sum(1 for r in runs if r.exact_zero)
        obj = sum(r.total_cost_ge for r in runs) / len(runs)
        med = sorted(vals)[len(vals) // 2] if vals else float("nan")
        print(
            f"  {name:<26} {len(runs):>3} {med:>12.4f} {n_floor:>10} "
            f"{n_zero:>7} {obj:>8.1f}"
        )

    print()
    print("=== 게이트 재판정 ===")
    print("  주 통계 = floor-capped 전체 쌍.  비포화 = 민감도 분석용 (primary 아님)")
    print()
    for label, base, treat, go, pivot in GATES:
        if base not in by_controller or treat not in by_controller:
            print(f"  [{label}] 라벨 없음 ({base} 또는 {treat})")
            continue
        b_runs, t_runs = by_controller[base], by_controller[treat]
        d = compare_paired_delta(b_runs, t_runs, metric="log_improvement")
        nb, nt = drop_saturated_pairs(b_runs, t_runs)
        ns = (
            compare_paired_delta(nb, nt, metric="log_improvement")
            if nb
            else None
        )
        ci = d.delta_ci
        p = f"{d.p_value:.4f}" if math.isfinite(d.p_value) else "n/a"
        print(
            f"  [{label:<13}] {d.median_delta:+8.4f} nat  "
            f"CI {ci[0]:+8.3f}~{ci[1]:+8.3f}  p={p:<7} "
            f"n={d.n_valid}/{d.n_pairs}  -> {verdict(d.median_delta, go, pivot)}"
        )
        sat = (
            f"joint={d.n_joint_saturated} one-sided={d.n_one_sided_saturated}"
            if d.n_saturated
            else "없음"
        )
        nonsat = (
            f"{ns.median_delta:+.4f} nat (n={ns.n_valid}) -> "
            f"{verdict(ns.median_delta, go, pivot)}"
            if ns and ns.n_valid
            else "비포화 쌍 없음"
        )
        print(f"                   포화 {sat}   비포화 민감도 {nonsat}")
        if d.excluded_pairs:
            for task, seed, why in d.excluded_pairs:
                print(f"                   제외 {task}/seed{seed}: {why}")
        if ns and ns.n_valid and verdict(ns.median_delta, go, pivot) != verdict(
            d.median_delta, go, pivot
        ):
            print("                   ** 결론이 포화 처리에 민감하다 **")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
