"""공개 결과 로더와 그 위의 통계를 검사한다.

이 파일은 두 가지를 지킨다.

```text
1  공개 CSV 를 RunSummary 로 되돌릴 때 정보가 새지 않는가
2  공개 데이터만으로 원고의 머릿수치가 다시 나오는가
```

`2` 는 회귀 방지가 목적이다. 공개 CSV 를 잘못 재생성하면 여기서 걸린다.
raw 와의 비트 단위 대조는 `scripts/verify_public_results.py` 가 한다.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from rl_newton.reporting import (
    PUBLIC_COLUMNS,
    load_public_results,
    paired,
    positive_count,
    public_roles,
)

PUBLIC_DIR = Path(__file__).resolve().parents[1] / "results" / "public"
HELD_OUT = PUBLIC_DIR / "heldout_quadratic.csv"
MICRO = PUBLIC_DIR / "micro_neural.csv"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PUBLIC_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in PUBLIC_COLUMNS})


def make_row(**over: object) -> dict[str, object]:
    row: dict[str, object] = {
        "raw_source": "synthetic.jsonl",
        "role": "test",
        "acceptance_rule": "control",
        "beam_width": 8,
        "budget_ge": 150.0,
        "task_spec": "quad_d4_k1e+02",
        "condition_number": "1e+02",
        "batch_size": "",
        "seed": 0,
        "controller": "ctrl_a",
        "controller_role": "",
        "initial_loss": repr(10.0),
        "final_loss": repr(1.0),
        "log_improvement": repr(math.log(10.0)),
        "floor_hit": 0,
        "total_cost_ge": repr(150.0),
        "search_cost_ge": repr(0.0),
        "n_steps": 5,
        "stop_reason": "budget",
        "rejection_rate": repr(0.0),
        "failure_rate": repr(0.0),
        "negative_curvature_rate": repr(0.0),
        "target": "medium",
        "reached": 0,
        "cost_to_target_ge": "",
        "steps_to_target": "",
    }
    row.update(over)
    return row


class TestLoader:
    def test_rejects_unexpected_columns(self, tmp_path: Path):
        path = tmp_path / "bad.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="열 구성"):
            load_public_results(path)

    def test_log_improvement_recomputes_from_losses(self, tmp_path: Path):
        """CSV 의 log_improvement 열과 재계산값이 같아야 한다.

        복원된 `RunSummary` 는 `initial_loss` / `final_loss` 에서 값을 다시
        계산한다. 두 값이 갈리면 CSV 가 정보를 잃었다는 뜻이다.
        """
        path = tmp_path / "one.csv"
        rows = [
            make_row(initial_loss=repr(24.2), final_loss=repr(3.930839434133)),
            make_row(seed=1, initial_loss=repr(1.0), final_loss=repr(1e-30)),
        ]
        write_csv(path, rows)
        runs = load_public_results(path)
        assert len(runs) == 2
        for row, run in zip(rows, runs, strict=True):
            assert run.log_improvement == pytest.approx(
                math.log(float(row["initial_loss"]))
                - math.log(max(float(row["final_loss"]), run.loss_floor)),
                rel=0,
                abs=0,
            )

    def test_pairing_key_survives_reconstruction(self, tmp_path: Path):
        """서로 다른 seed 는 서로 다른 인스턴스로 남아야 한다."""
        path = tmp_path / "pair.csv"
        write_csv(
            path,
            [
                make_row(seed=0, controller="ctrl_a"),
                make_row(seed=1, controller="ctrl_a"),
                make_row(seed=0, controller="ctrl_b", final_loss=repr(0.5)),
                make_row(seed=1, controller="ctrl_b", final_loss=repr(0.5)),
            ],
        )
        runs = load_public_results(path)
        assert len({(r.task_instance_id, r.seed) for r in runs}) == 2
        result = paired(runs, "ctrl_a", "ctrl_b")
        assert result.n_valid == 2
        assert result.median_delta > 0.0

    def test_acceptance_rule_filter(self, tmp_path: Path):
        path = tmp_path / "rules.csv"
        write_csv(
            path,
            [
                make_row(acceptance_rule="control"),
                make_row(acceptance_rule="fixed_eval", seed=1),
            ],
        )
        assert len(load_public_results(path)) == 2
        assert len(load_public_results(path, acceptance_rule="control")) == 1


class TestRoles:
    def test_maps_role_to_label(self, tmp_path: Path):
        path = tmp_path / "roles.csv"
        write_csv(
            path,
            [
                make_row(controller="static[2]", controller_role="best_static"),
                make_row(controller="shrinking_Q4_narrow", seed=1),
            ],
        )
        assert public_roles(path) == {"best_static": "static[2]"}

    def test_conflicting_role_is_an_error(self, tmp_path: Path):
        """수락 규칙마다 선택 결과가 다르면 분리하지 않고는 쓸 수 없다."""
        path = tmp_path / "conflict.csv"
        write_csv(
            path,
            [
                make_row(
                    acceptance_rule="control",
                    controller="static[4]",
                    controller_role="best_static",
                ),
                make_row(
                    acceptance_rule="fixed_eval",
                    seed=1,
                    controller="static[9]",
                    controller_role="best_static",
                ),
            ],
        )
        with pytest.raises(ValueError, match="두 라벨"):
            public_roles(path)
        assert public_roles(path, acceptance_rule="control") == {
            "best_static": "static[4]"
        }

    def test_role_name_resolves_in_paired(self, tmp_path: Path):
        path = tmp_path / "byrole.csv"
        write_csv(
            path,
            [
                make_row(controller="static[2]", controller_role="best_static"),
                make_row(
                    controller="static[2]",
                    controller_role="best_static",
                    seed=1,
                ),
                make_row(controller="shrinking_Q4_narrow", final_loss=repr(0.1)),
                make_row(
                    controller="shrinking_Q4_narrow", seed=1, final_loss=repr(0.1)
                ),
            ],
        )
        runs = load_public_results(path)
        roles = public_roles(path)
        by_role = paired(runs, "best_static", "shrinking_Q4_narrow", roles=roles)
        by_label = paired(runs, "static[2]", "shrinking_Q4_narrow")
        assert by_role.median_delta == by_label.median_delta
        with pytest.raises(KeyError, match="best_static"):
            paired(runs, "best_static", "shrinking_Q4_narrow")


@pytest.mark.skipif(not HELD_OUT.exists(), reason="공개 CSV 가 없다")
class TestManuscriptNumbers:
    """공개 데이터만으로 원고의 머릿수치가 나오는가.

    값은 `docs/results_stage2.md` 에서 왔다. 공개 CSV 를 잘못 재생성하면 여기서
    걸린다.
    """

    def setup_method(self):
        self.runs = load_public_results(HELD_OUT)
        self.roles = public_roles(HELD_OUT)

    def test_row_count(self):
        assert len(self.runs) == 1200

    def test_selected_baselines(self):
        assert self.roles == {
            "best_static": "static[2]",
            "best_open_loop": "open_loop[4]",
        }

    @pytest.mark.parametrize(
        ("gate", "base", "treat", "median", "n_pos", "n_pairs"),
        [
            ("A2", "best_static", "shrinking_Q4_narrow", 1.690, 40, 40),
            ("C2", "onestep_narrow", "shrinking_Q4_narrow", 0.456, 35, 40),
            ("C3", "committed_Q4_narrow", "shrinking_Q4_narrow", 0.010, 21, 40),
            ("open_loop", "best_static", "best_open_loop", 0.395, 40, 40),
            ("ladder", "best_static", "onestep_narrow", 1.155, 40, 40),
            ("ladder", "best_static", "committed_Q4_narrow", 2.090, 40, 40),
        ],
    )
    def test_paired_gate(
        self, gate: str, base: str, treat: str, median: float, n_pos: int, n_pairs: int
    ):
        result = paired(self.runs, base, treat, roles=self.roles)
        assert result.n_valid == n_pairs, gate
        assert round(result.median_delta, 3) == median, gate
        assert positive_count(self.runs, base, treat, roles=self.roles) == (
            n_pos,
            n_pairs,
        ), gate

    def test_c3_interval_includes_zero(self):
        """`C3` 의 CI 가 좁게 0 을 포함한다. 등가성 주장은 하지 않는다 (§7.4)."""
        result = paired(
            self.runs, "committed_Q4_narrow", "shrinking_Q4_narrow", roles=self.roles
        )
        low, high = result.delta_ci
        assert low < 0.0 < high
        assert round(low, 3) == -0.033
        assert round(high, 3) == 0.053

    def test_search_cost_ratio(self):
        """planner 의 탐색 비용이 배포 예산의 세 자릿수 배다 (C06)."""
        from rl_newton.benchmark.metrics import median_of

        planner = [
            r.search_cost_ge for r in self.runs if r.controller == "shrinking_Q4_narrow"
        ]
        assert round(median_of(planner) / 150.0) == 1294


@pytest.mark.skipif(not MICRO.exists(), reason="공개 CSV 가 없다")
class TestMicroNeural:
    def test_two_acceptance_rules_are_separable(self):
        control = load_public_results(MICRO, acceptance_rule="control")
        fixed = load_public_results(MICRO, acceptance_rule="fixed_eval")
        assert len(control) == 216
        assert len(fixed) == 216

    def test_c3_flips_sign_between_regimes(self):
        """full-batch 에서는 0 근처이고 minibatch 에서는 양수다 (§10.3).

        크기는 주장하지 않는다. regime 당 `n=3` 이다.
        """
        runs = load_public_results(MICRO, acceptance_rule="control")
        roles = public_roles(MICRO, acceptance_rule="control")
        full = paired(
            runs,
            "committed_Q4_narrow",
            "shrinking_Q4_narrow",
            spec="mlp_d32_h128_c5_n512_fb",
            roles=roles,
        )
        small = paired(
            runs,
            "committed_Q4_narrow",
            "shrinking_Q4_narrow",
            spec="mlp_d32_h128_c5_n512_cs128",
            roles=roles,
        )
        assert full.median_delta < 0.0
        assert small.median_delta > 1.0

    def test_committed_rejection_rate_rises_in_minibatch(self):
        """stale plan 이 거절된다 (§10.3, Figure 3)."""
        from rl_newton.benchmark.metrics import median_of

        runs = load_public_results(MICRO, acceptance_rule="control")
        committed = [r for r in runs if r.controller == "committed_Q4_narrow"]

        def rate(tag: str) -> float:
            sub = [r for r in committed if r.task_instance_id.endswith(f"{tag}_seed2")]
            sub += [r for r in committed if r.task_instance_id.endswith(f"{tag}_seed3")]
            sub += [r for r in committed if r.task_instance_id.endswith(f"{tag}_seed4")]
            return median_of([r.rejection_rate for r in sub])

        assert rate("_fb") == 0.0
        assert rate("_cs128") > 0.5
        assert rate("_cs64") > 0.5
