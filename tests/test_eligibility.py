"""Benchmark eligibility 검증 (프로토콜 D25).

D23 에서 두 결함이 드러났다.

```text
[1] rosen_d5 가 국소최소점에 갇혔는데 ceiling 공식이 전역최소점 도달을 가정했다
[2] randomize_start=False 라 seed 2/3/4 가 같은 인스턴스였다 (n=3 이 실제로 n=1)
```

calibration 이 이 둘을 **자동으로** 잡아야 한다.
"""

from __future__ import annotations

import math

import torch

from rl_newton.benchmark.eligibility import (
    REFERENCE_AGREEMENT_NAT,
    AchievableCeiling,
    ReferenceRun,
    achievable_ceiling,
    check_seed_variation,
    reference_panel,
)
from rl_newton.benchmark.metrics import RELATIVE_LOSS_FLOOR
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask
from rl_newton.tasks.rosenbrock import RosenbrockSpec, RosenbrockTask


def run(name: str, loss: float, grad: float = 1.0e-9) -> ReferenceRun:
    return ReferenceRun(name=name, final_loss=loss, grad_norm=grad, n_iters=10)


class TestAchievableCeiling:
    def test_critical_point_caps_the_ceiling(self):
        """국소최소점이 수치 하한보다 위면 그것이 상한을 정한다 (D23)."""
        ceiling = achievable_ceiling(24.2, [run("lbfgs", 3.930839434133)])
        assert ceiling.limited_by == "critical_point"
        assert ceiling.nats == math.log(24.2) - math.log(3.930839434133)
        assert ceiling.nats < 2.0

    def test_rosen_d5_would_fail_the_three_nat_rule(self):
        """관측된 baseline median 1.8175 nat 이 여유 3 nat 을 만족할 수 없다."""
        ceiling = achievable_ceiling(24.2, [run("lbfgs", 3.930839434133)])
        observed_baseline_median = 1.8175
        assert ceiling.nats - observed_baseline_median < 3.0

    def test_d20_formula_would_have_passed(self):
        """구 공식은 통과시켰다. 이 대비가 D23 의 요지다."""
        l0 = 24.2
        old_ceiling = math.log(l0) - math.log(l0 * RELATIVE_LOSS_FLOOR)
        assert old_ceiling - 1.8175 > 3.0
        new = achievable_ceiling(l0, [run("lbfgs", 3.930839434133)])
        assert new.nats - 1.8175 < 3.0

    def test_numerical_floor_used_when_reference_reaches_zero(self):
        ceiling = achievable_ceiling(100.0, [run("lbfgs", 0.0)])
        assert ceiling.limited_by == "numerical_floor"
        assert ceiling.effective_floor == 100.0 * RELATIVE_LOSS_FLOOR

    def test_minimum_over_panel_is_used(self):
        ceiling = achievable_ceiling(
            100.0, [run("lbfgs", 5.0), run("adam", 1.0), run("newton", 3.0)]
        )
        assert ceiling.reference_loss == 1.0

    def test_nonfinite_reference_is_skipped(self):
        ceiling = achievable_ceiling(
            100.0, [run("lbfgs", float("nan")), run("adam", 2.0)]
        )
        assert ceiling.reference_loss == 2.0

    def test_all_references_failed(self):
        ceiling = achievable_ceiling(100.0, [run("lbfgs", float("nan"))])
        assert ceiling.limited_by == "reference_failed"
        assert not math.isfinite(ceiling.nats)

    def test_spread_detects_disagreement(self):
        """참조 solver 가 크게 갈리면 상한 추정이 불안정하다."""
        agree = achievable_ceiling(100.0, [run("lbfgs", 1.0), run("adam", 1.2)])
        assert agree.references_agree

        disagree = achievable_ceiling(100.0, [run("lbfgs", 1.0), run("adam", 50.0)])
        assert disagree.reference_spread_nat > REFERENCE_AGREEMENT_NAT
        assert not disagree.references_agree

    def test_floor_limited_skips_agreement_check(self):
        """floor cap 으로 값이 갈리는 것은 solver 불일치가 아니다."""
        ceiling = achievable_ceiling(100.0, [run("lbfgs", 0.0), run("adam", 1.0e-300)])
        assert ceiling.limited_by == "numerical_floor"
        assert ceiling.references_agree

    def test_single_run_has_zero_spread(self):
        assert achievable_ceiling(100.0, [run("lbfgs", 1.0)]).reference_spread_nat == 0.0

    def test_describe_names_the_best_solver(self):
        ceiling = achievable_ceiling(100.0, [run("lbfgs", 5.0), run("newton", 0.5)])
        assert "newton" in ceiling.describe()

    def test_nonpositive_initial_loss_is_nan(self):
        assert not math.isfinite(achievable_ceiling(0.0, [run("lbfgs", 1.0)]).nats)


class TestSeedVariation:
    def test_rosen_default_is_seed_duplicated(self):
        """D23 원인 3 을 재현한다. 이 검사가 자동으로 걸러야 한다."""
        spec = RosenbrockSpec(dimension=5)
        report = check_seed_variation(
            lambda s: RosenbrockTask(spec, seed=s, dtype=torch.float64), [2, 3, 4]
        )
        assert report.n_distinct_initial_loss == 1
        assert report.n_distinct_start_point == 1
        assert not report.ok
        assert "복제" in report.describe()

    def test_rosen_randomized_start_is_distinct(self):
        spec = RosenbrockSpec(dimension=5, randomize_start=True)
        report = check_seed_variation(
            lambda s: RosenbrockTask(spec, seed=s, dtype=torch.float64), [2, 3, 4]
        )
        assert report.ok

    def test_challenge_quadratics_are_distinct(self):
        """D22 의 n=12 결과가 유효하다는 근거다."""
        for kappa in (1.0e3, 1.0e4, 1.0e5, 1.0e6):
            spec = QuadraticSpec(
                kind="ill_conditioned", dimension=100, condition_number=kappa
            )
            report = check_seed_variation(
                lambda s, _spec=spec: QuadraticTask(_spec, seed=s, dtype=torch.float64),
                [2, 3, 4],
            )
            assert report.ok, f"kappa={kappa:g} 에서 seed 복제"


class TestReferencePanel:
    def test_panel_finds_rosen_d5_local_minimum(self):
        """참조 panel 이 관측된 정체점을 찾아내야 한다."""
        spec = RosenbrockSpec(dimension=5)
        runs = reference_panel(
            lambda: RosenbrockTask(spec, seed=2, dtype=torch.float64),
            max_iter=2000,
        )
        ceiling = achievable_ceiling(24.2, runs)
        assert ceiling.limited_by == "critical_point"
        # 관측된 컨트롤러 정체 loss 는 3.9308388~3.9310777 였다.
        assert ceiling.reference_loss < 3.94

    def test_extra_init_diagnoses_but_does_not_raise_the_ceiling(self):
        """다른 초기화는 **진단**이다. 상한을 올리면 안 된다.

        컨트롤러는 task 시작점에서만 출발한다. 다른 basin 의 최적값을 상한으로
        쓰면 국소최소점에 갇힌 task 가 eligibility 를 통과한다.
        """
        spec = RosenbrockSpec(dimension=5)
        l0 = float(RosenbrockTask(spec, seed=2, dtype=torch.float64).initial_loss)
        runs = reference_panel(
            lambda: RosenbrockTask(spec, seed=2, dtype=torch.float64),
            max_iter=2000,
            extra_inits=(0.9,),
        )
        assert any("init" in r.name for r in runs)
        off_best = min(
            r.final_loss for r in runs if not r.from_task_start and math.isfinite(r.final_loss)
        )
        assert off_best < 1.0e-6, "0.9 에서 출발하면 전역최소점에 도달한다"

        ceiling = achievable_ceiling(l0, runs)
        # 상한은 여전히 시작점 basin 의 국소최소점이 정한다.
        assert ceiling.limited_by == "critical_point"
        assert ceiling.reference_loss > 3.9
        assert ceiling.nats < 3.0
        assert ceiling.start_basin_is_suboptimal
        assert "전역최적 아님" in ceiling.describe()

    def test_rosen_d5_fails_eligibility_even_with_extra_inits(self):
        """D25 소급 적용 결과. 다중 초기화가 있어도 탈락해야 한다."""
        spec = RosenbrockSpec(dimension=5, randomize_start=True)
        task = RosenbrockTask(spec, seed=0, dtype=torch.float64)
        l0 = float(task.initial_loss)
        runs = reference_panel(
            lambda: RosenbrockTask(spec, seed=0, dtype=torch.float64),
            max_iter=2000,
            extra_inits=(0.9,),
        )
        ceiling = achievable_ceiling(l0, runs)
        # 관측된 baseline median 은 약 2.58 nat 였다. 여유 3 nat 을 만족할 수 없다.
        assert ceiling.nats < 3.5, f"J_achievable={ceiling.nats}"

    def test_start_basin_optimal_when_no_extra_inits(self):
        spec = QuadraticSpec(kind="spd", dimension=8, condition_number=10.0)
        l0 = float(QuadraticTask(spec, seed=0, dtype=torch.float64).initial_loss)
        runs = reference_panel(
            lambda: QuadraticTask(spec, seed=0, dtype=torch.float64), max_iter=500
        )
        ceiling = achievable_ceiling(l0, runs)
        assert not ceiling.start_basin_is_suboptimal
        assert ceiling.off_start_runs == ()

    def test_panel_on_quadratic_reaches_floor(self):
        spec = QuadraticSpec(kind="spd", dimension=20, condition_number=1.0e2)
        runs = reference_panel(
            lambda: QuadraticTask(spec, seed=0, dtype=torch.float64), max_iter=2000
        )
        l0 = float(QuadraticTask(spec, seed=0, dtype=torch.float64).initial_loss)
        ceiling = achievable_ceiling(l0, runs)
        assert ceiling.nats > 10.0, "볼록 문제는 상한이 커야 한다"

    def test_panel_includes_multiple_solvers(self):
        spec = QuadraticSpec(kind="spd", dimension=8, condition_number=10.0)
        runs = reference_panel(
            lambda: QuadraticTask(spec, seed=0, dtype=torch.float64), max_iter=200
        )
        assert {"lbfgs", "adam", "sgd_momentum"} <= {r.name for r in runs}

    def test_newton_skipped_above_dim_limit(self):
        spec = QuadraticSpec(kind="spd", dimension=64, condition_number=10.0)
        runs = reference_panel(
            lambda: QuadraticTask(spec, seed=0, dtype=torch.float64),
            max_iter=100,
            newton_dim_limit=16,
        )
        assert "newton" not in {r.name for r in runs}


class TestReferenceRunFlags:
    def test_critical_point_flag(self):
        assert ReferenceRun("a", 1.0, 1.0e-9, 1).is_critical_point
        assert not ReferenceRun("a", 1.0, 1.0e-1, 1).is_critical_point
        assert not ReferenceRun("a", 1.0, float("nan"), 1).is_critical_point

    def test_ceiling_is_frozen(self):
        ceiling = AchievableCeiling(
            initial_loss=1.0, reference_loss=0.5, loss_floor=1.0e-14
        )
        try:
            ceiling.initial_loss = 2.0  # type: ignore[misc]
        except (AttributeError, TypeError):
            return
        raise AssertionError("AchievableCeiling 은 불변이어야 한다")
