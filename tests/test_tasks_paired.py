"""Stage 1: task 구성과 paired design 결정론성 (프로토콜 D7).

``seed`` 는 실험 조건 식별자다. ``seed=s`` 면 모든 optimizer가 동일한 문제를
본다는 전제가 여기서 깨지면 쌍별 비교 통계 전체가 무의미해진다.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from rl_newton.benchmark.paired import (
    PairedTaskFactory,
    make_task,
    quadratic_meta_test_specs,
    quadratic_meta_train_specs,
)
from rl_newton.tasks.quadratics import QuadraticSpec, QuadraticTask
from rl_newton.tasks.rosenbrock import RosenbrockSpec, RosenbrockTask


class TestQuadraticConstruction:
    @pytest.mark.parametrize("kappa", [1.0e1, 1.0e3, 1.0e6])
    def test_condition_number_matches_request(self, kappa: float):
        """조건수를 지정값과 정확히 일치시켜야 한다.

        Stage 2 헤드룸 측정은 조건수별 이득 차이를 보는 것이 핵심이므로,
        조건수가 대충 맞으면 결과 해석이 불가능하다.
        """
        task = QuadraticTask(
            QuadraticSpec(dimension=64, condition_number=kappa),
            seed=0,
            dtype=torch.float64,
        )
        assert task.condition_number == pytest.approx(kappa, rel=1e-6)

    def test_matrix_is_symmetric(self):
        """CG는 A의 대칭성을 가정한다."""
        task = QuadraticTask(
            QuadraticSpec(dimension=50, condition_number=1e4), seed=0, dtype=torch.float64
        )
        a = task.matrix
        assert float((a - a.T).abs().max()) < 1e-12

    def test_spd_is_positive_definite(self):
        task = QuadraticTask(
            QuadraticSpec(kind="spd", dimension=50, condition_number=1e4),
            seed=0,
            dtype=torch.float64,
        )
        assert task.min_eigenvalue > 0.0
        assert task.is_bounded_below
        assert task.optimal_loss == 0.0
        assert float(torch.linalg.eigvalsh(task.matrix).min()) > 0.0

    def test_indefinite_has_negative_eigenvalues_and_is_unbounded(self):
        """indefinite는 아래로 유계가 아니므로 target 집계에서 제외해야 한다."""
        task = QuadraticTask(
            QuadraticSpec(kind="indefinite", dimension=50, condition_number=1e2),
            seed=0,
            dtype=torch.float64,
        )
        eigvals = torch.linalg.eigvalsh(task.matrix)

        assert float(eigvals.min()) < 0.0
        assert not task.is_bounded_below
        assert task.optimal_loss == float("-inf")

    def test_indefinite_negative_fraction_is_respected(self):
        spec = QuadraticSpec(
            kind="indefinite", dimension=100, condition_number=1e2, negative_fraction=0.3
        )
        task = QuadraticTask(spec, seed=0, dtype=torch.float64)
        n_negative = int((task.eigenvalues < 0).sum())

        assert n_negative == 30

    def test_loss_is_zero_at_origin(self):
        """``L* = 0`` 이어야 log-loss 보상(D3)이 정의된다."""
        task = QuadraticTask(QuadraticSpec(dimension=20), seed=0)
        with torch.no_grad():
            task.params[0].zero_()

        assert float(task.loss().detach()) == 0.0

    def test_reset_restores_initial_point(self):
        task = QuadraticTask(QuadraticSpec(dimension=20), seed=0)
        snapshot = task.params[0].detach().clone()

        with torch.no_grad():
            task.params[0].add_(1.0)
        assert not torch.allclose(task.params[0], snapshot)

        task.reset()
        assert torch.allclose(task.params[0], snapshot)
        assert float(task.loss().detach()) == pytest.approx(task.initial_loss, rel=1e-6)

    def test_exact_newton_step_solves_the_system(self):
        """테스트 기준값 자체가 옳은지 확인한다. CG 검증의 신뢰성 근거."""
        task = QuadraticTask(
            QuadraticSpec(dimension=32, condition_number=1e3), seed=0, dtype=torch.float64
        )
        p = task.exact_newton_step()
        grad = task.matrix @ task.params[0].detach()

        residual = task.matrix @ p + grad
        assert float(residual.norm() / grad.norm()) < 1e-10

    def test_rejects_invalid_spec(self):
        with pytest.raises(ValueError, match="dimension"):
            QuadraticSpec(dimension=0)
        with pytest.raises(ValueError, match="condition_number"):
            QuadraticSpec(condition_number=0.5)
        with pytest.raises(ValueError, match="negative_fraction"):
            QuadraticSpec(kind="indefinite", negative_fraction=0.0)
        with pytest.raises(ValueError, match="initial_scale"):
            QuadraticSpec(initial_scale=0.0)

    def test_rejects_unsupported_dtype(self):
        with pytest.raises(ValueError, match="float32 or float64"):
            QuadraticTask(QuadraticSpec(dimension=8), seed=0, dtype=torch.float16)


class TestRosenbrockConstruction:
    def test_standard_start_loss_matches_analytic_value(self):
        """2D 표준 시작점 (-1.2, 1.0) 에서 L = 100(1-1.44)^2 + (1+1.2)^2 = 24.2."""
        task = RosenbrockTask(RosenbrockSpec(dimension=2), seed=0, dtype=torch.float64)
        assert task.initial_loss == pytest.approx(24.2, rel=1e-9)

    def test_loss_is_zero_at_minimizer(self):
        task = RosenbrockTask(RosenbrockSpec(dimension=5), seed=0, dtype=torch.float64)
        task.move_to(task.minimizer)

        assert float(task.loss().detach()) == pytest.approx(0.0, abs=1e-20)
        assert task.optimal_loss == 0.0

    def test_negative_curvature_point_is_indefinite(self):
        for d in (2, 5):
            task = RosenbrockTask(RosenbrockSpec(dimension=d), seed=0, dtype=torch.float64)
            task.move_to(task.negative_curvature_point(x0=0.3))
            assert float(torch.linalg.eigvalsh(task.hessian_matrix()).min()) < 0.0

    def test_randomized_start_differs_by_seed(self):
        spec = RosenbrockSpec(dimension=4, randomize_start=True)
        a = RosenbrockTask(spec, seed=0)
        b = RosenbrockTask(spec, seed=1)

        assert not torch.allclose(a.params[0], b.params[0])

    def test_standard_start_is_seed_independent(self):
        spec = RosenbrockSpec(dimension=4, randomize_start=False)
        a = RosenbrockTask(spec, seed=0)
        b = RosenbrockTask(spec, seed=99)

        assert torch.allclose(a.params[0], b.params[0])

    def test_rejects_dimension_below_two(self):
        with pytest.raises(ValueError, match="dimension >= 2"):
            RosenbrockSpec(dimension=1)

    def test_move_to_rejects_wrong_shape(self):
        task = RosenbrockTask(RosenbrockSpec(dimension=3), seed=0)
        with pytest.raises(ValueError, match="must have shape"):
            task.move_to(torch.zeros(4))


class TestPairedDeterminism:
    """프로토콜 D7의 핵심 보장."""

    def test_same_spec_and_seed_reproduce_identical_instance(self):
        spec = QuadraticSpec(kind="spd", dimension=64, condition_number=1e4)

        a = make_task(spec, 3)
        b = make_task(spec, 3)

        assert torch.equal(a.matrix, b.matrix)
        assert torch.equal(a.params[0].detach(), b.params[0].detach())
        assert a.initial_loss == b.initial_loss
        assert a.instance_id == b.instance_id

    def test_different_seeds_give_different_instances(self):
        spec = QuadraticSpec(dimension=64, condition_number=1e4)

        a = make_task(spec, 0)
        b = make_task(spec, 1)

        assert not torch.allclose(a.matrix, b.matrix)
        # 조건수는 spec이 정하므로 seed와 무관하게 같아야 한다
        assert a.condition_number == pytest.approx(b.condition_number, rel=1e-6)

    def test_instances_are_independent_objects(self):
        """한 optimizer가 파라미터를 바꿔도 다른 인스턴스에 영향이 없어야 한다."""
        spec = QuadraticSpec(dimension=16)
        a = make_task(spec, 0)
        b = make_task(spec, 0)

        with torch.no_grad():
            a.params[0].add_(100.0)

        assert not torch.allclose(a.params[0], b.params[0])
        assert float(b.loss().detach()) == pytest.approx(b.initial_loss, rel=1e-6)

    def test_optimizer_rng_consumption_does_not_affect_task(self):
        """optimizer마다 난수 소비량이 달라도 task 인스턴스는 동일해야 한다.

        RL 컨트롤러는 정책 샘플링으로 난수를 쓰고 fixed baseline은 쓰지 않는다.
        전역 스트림을 공유하면 그 차이가 문제 자체를 바꿔 버린다.
        """
        spec = QuadraticSpec(dimension=32, condition_number=1e3)

        torch.manual_seed(0)
        baseline = make_task(spec, 5)

        torch.manual_seed(0)
        _ = torch.randn(1000)  # 다른 optimizer가 난수를 많이 썼다고 가정
        after_consumption = make_task(spec, 5)

        assert torch.equal(baseline.matrix, after_consumption.matrix)
        assert torch.equal(baseline.params[0].detach(), after_consumption.params[0].detach())

    def test_determinism_across_processes(self):
        """별도 프로세스에서도 같은 인스턴스가 나와야 한다.

        벤치마크는 optimizer별로 따로 실행될 수 있다. 프로세스 경계를 넘어
        재현되지 않으면 paired 비교가 성립하지 않는다.
        """
        code = (
            "import torch;"
            "from rl_newton.benchmark.paired import make_task;"
            "from rl_newton.tasks.quadratics import QuadraticSpec;"
            "t = make_task(QuadraticSpec(dimension=16, condition_number=100.0), 7);"
            "print(float(t.matrix.sum()), float(t.params[0].detach().sum()))"
        )
        local = make_task(QuadraticSpec(dimension=16, condition_number=100.0), 7)
        expected = f"{float(local.matrix.sum())} {float(local.params[0].detach().sum())}"

        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        assert proc.stdout.strip() == expected

    def test_rosenbrock_is_deterministic(self):
        spec = RosenbrockSpec(dimension=6, randomize_start=True)
        a = make_task(spec, 2)
        b = make_task(spec, 2)

        assert torch.equal(a.params[0].detach(), b.params[0].detach())


class TestPairedTaskFactory:
    def test_iterates_all_spec_seed_pairs_deterministically(self):
        specs = [
            QuadraticSpec(dimension=16, condition_number=10.0),
            QuadraticSpec(dimension=32, condition_number=100.0),
        ]
        factory = PairedTaskFactory(specs=specs, seeds=[0, 1, 2])

        first = list(factory.iter_instances())
        second = list(factory.iter_instances())

        assert len(first) == factory.n_instances() == 6
        assert first == second

    def test_build_reproduces_instances(self):
        spec = QuadraticSpec(dimension=16)
        factory = PairedTaskFactory(specs=[spec], seeds=[0])

        assert torch.equal(factory.build(spec, 0).matrix, factory.build(spec, 0).matrix)

    def test_derived_seeds_are_stable_and_distinct(self):
        spec = QuadraticSpec(dimension=16)
        factory = PairedTaskFactory(specs=[spec], seeds=[0, 1])

        assert factory.batch_order_seed(spec, 0) == factory.batch_order_seed(spec, 0)
        assert factory.batch_order_seed(spec, 0) != factory.batch_order_seed(spec, 1)
        assert factory.batch_order_seed(spec, 0) != factory.model_init_seed(spec, 0)

    def test_derived_seeds_differ_across_specs(self):
        a = QuadraticSpec(dimension=16, condition_number=10.0)
        b = QuadraticSpec(dimension=16, condition_number=100.0)
        factory = PairedTaskFactory(specs=[a, b], seeds=[0])

        assert factory.batch_order_seed(a, 0) != factory.batch_order_seed(b, 0)

    def test_rejects_duplicate_seeds(self):
        with pytest.raises(ValueError, match="unique"):
            PairedTaskFactory(specs=[QuadraticSpec()], seeds=[0, 0])

    def test_rejects_empty_inputs(self):
        with pytest.raises(ValueError, match="specs"):
            PairedTaskFactory(specs=[], seeds=[0])
        with pytest.raises(ValueError, match="seeds"):
            PairedTaskFactory(specs=[QuadraticSpec()], seeds=[])

    def test_rejects_unknown_spec_type(self):
        with pytest.raises(TypeError, match="unsupported task spec"):
            make_task("not-a-spec", 0)  # type: ignore[arg-type]


class TestMetaDistributions:
    def test_meta_train_covers_documented_range(self):
        specs = quadratic_meta_train_specs()
        kappas = {s.condition_number for s in specs}
        dims = {s.dimension for s in specs}

        assert dims == {50, 100, 200}
        assert min(kappas) == pytest.approx(1e1)
        assert max(kappas) == pytest.approx(1e4)
        assert any(s.kind == "indefinite" for s in specs)

    def test_meta_test_is_outside_meta_train_range(self):
        """정책 암기가 아니라 일반화를 시험하려면 분포가 겹치지 않아야 한다."""
        train = quadratic_meta_train_specs()
        test = quadratic_meta_test_specs()

        train_kappas = {s.condition_number for s in train}
        train_dims = {s.dimension for s in train}
        extrapolating = [
            s
            for s in test
            if s.condition_number not in train_kappas or s.dimension not in train_dims
        ]

        assert extrapolating, "meta-test에 학습 분포 밖 인스턴스가 있어야 한다"

    def test_meta_train_excludes_indefinite_when_requested(self):
        specs = quadratic_meta_train_specs(include_indefinite=False)
        assert all(s.kind != "indefinite" for s in specs)

    def test_all_meta_specs_are_constructible(self):
        for spec in quadratic_meta_train_specs() + quadratic_meta_test_specs():
            task = make_task(spec, 0)
            assert task.initial_loss > 0.0
            assert torch.isfinite(task.matrix).all()
