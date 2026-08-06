"""Micro-neural task 계약 검증 (프로토콜 D24).

이 task 의 목적은 **feedback 의 가치가 있는 regime 을 만드는 것**이다. 따라서
검증의 핵심은 성능이 아니라 다음 세 가지다.

```text
paired design 이 유지되는가      같은 (spec, seed) 면 모든 컨트롤러가 같은 것을 본다
regime 이 실제로 다른가          full_batch 는 결정론적, controlled_stochastic 은 아님
평가 지표가 regime 간 비교 가능한가  loss() 는 항상 full dataset
```
"""

from __future__ import annotations

import math

import pytest
import torch

from rl_newton.tasks.micro_neural import MicroNeuralSpec, MicroNeuralTask

SMALL = MicroNeuralSpec(
    input_dim=8, hidden_dim=16, n_classes=3, n_samples=64, teacher_hidden_dim=32
)
STOCHASTIC = MicroNeuralSpec(
    input_dim=8,
    hidden_dim=16,
    n_classes=3,
    n_samples=64,
    teacher_hidden_dim=32,
    regime="controlled_stochastic",
    batch_size=16,
)


def build(spec: MicroNeuralSpec, seed: int = 0) -> MicroNeuralTask:
    return MicroNeuralTask(spec, seed=seed, dtype=torch.float64)


class TestSpecValidation:
    def test_parameter_count_matches_actual_tensors(self):
        task = build(SMALL)
        assert SMALL.n_parameters == sum(p.numel() for p in task.params)

    def test_parameter_count_is_in_target_range(self):
        """D24 는 수천~수만 개를 요구한다. 기본 spec 이 그 범위여야 한다."""
        default = MicroNeuralSpec()
        assert 1_000 <= default.n_parameters <= 100_000

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"input_dim": 0},
            {"hidden_dim": 0},
            {"n_classes": 1},
            {"n_samples": 0},
            {"regime": "sgd"},
            {"label_noise": 0.5},
            {"init_scale": 0.0},
        ],
    )
    def test_invalid_spec_rejected(self, kwargs):
        with pytest.raises(ValueError):
            MicroNeuralSpec(**kwargs)

    def test_batch_size_must_fit_dataset(self):
        with pytest.raises(ValueError):
            MicroNeuralSpec(n_samples=32, regime="controlled_stochastic", batch_size=64)

    def test_batch_size_ignored_in_full_batch(self):
        """full_batch 에서는 batch_size 를 검사하지 않는다. 의미가 없기 때문이다."""
        spec = MicroNeuralSpec(n_samples=32, regime="full_batch", batch_size=999)
        assert spec.regime == "full_batch"

    def test_instance_id_encodes_regime(self):
        assert SMALL.instance_id(3).endswith("_fb_seed3")
        assert "cs16" in STOCHASTIC.instance_id(3)


class TestPairedDesign:
    """같은 ``(spec, seed)`` 면 항상 같은 인스턴스여야 한다 (프로토콜 D7)."""

    def test_same_spec_seed_gives_identical_instance(self):
        a, b = build(SMALL, 5), build(SMALL, 5)
        assert a.initial_loss == b.initial_loss
        for pa, pb in zip(a.params, b.params, strict=True):
            assert torch.equal(pa, pb)

    def test_different_seeds_give_different_instances(self):
        """D23 원인 3. `randomize_start=False` 같은 seed 복제가 없어야 한다."""
        losses = {build(SMALL, s).initial_loss for s in (0, 1, 2, 3)}
        assert len(losses) == 4

    def test_batch_stream_is_seed_determined(self):
        a, b = build(STOCHASTIC, 7), build(STOCHASTIC, 7)
        for _ in range(5):
            assert float(a.curvature_loss()) == float(b.curvature_loss())
            a.advance_batch()
            b.advance_batch()

    def test_batch_stream_differs_across_seeds(self):
        a, b = build(STOCHASTIC, 7), build(STOCHASTIC, 8)
        # 파라미터를 같게 맞춘다. 0 으로 두면 logit 이 전부 0 이 되어 loss 가
        # 데이터와 무관해지므로 (log C) 비교가 성립하지 않는다.
        point = torch.linspace(-0.3, 0.3, STOCHASTIC.n_parameters, dtype=torch.float64)
        a.move_to(point)
        b.move_to(point)
        assert float(a.curvature_loss()) != float(b.curvature_loss())


class TestRegimeSemantics:
    def test_full_batch_curvature_equals_evaluation(self):
        task = build(SMALL)
        assert float(task.curvature_loss()) == float(task.loss())

    def test_full_batch_is_invariant_to_advance(self):
        task = build(SMALL)
        before = float(task.curvature_loss())
        for _ in range(10):
            task.advance_batch()
        assert float(task.curvature_loss()) == before

    def test_stochastic_curvature_differs_from_evaluation(self):
        task = build(STOCHASTIC)
        assert float(task.curvature_loss()) != float(task.loss())

    def test_stochastic_advances_batch(self):
        task = build(STOCHASTIC)
        seen = []
        for _ in range(6):
            seen.append(float(task.curvature_loss()))
            task.advance_batch()
        assert len(set(seen)) > 1, "batch 가 전진하지 않으면 R2 가 성립하지 않는다"

    def test_two_regimes_share_data_and_initialization(self):
        """D24: "모델과 데이터는 하나만 고정한다."

        `data_key` 가 regime 을 포함하면 두 regime 이 다른 데이터셋을 받아 `C3` 의
        regime 간 비교가 데이터 차이에 오염된다.
        """
        import dataclasses

        fb = build(SMALL)
        cs = build(dataclasses.replace(SMALL, regime="controlled_stochastic", batch_size=16))
        assert SMALL.data_key == cs.spec.data_key
        assert fb.initial_loss == cs.initial_loss
        for a, b in zip(fb.params, cs.params, strict=True):
            assert torch.equal(a, b), "초기 파라미터가 같아야 한다"
        assert torch.equal(fb._inputs, cs._inputs), "데이터가 같아야 한다"
        assert torch.equal(fb._labels, cs._labels), "라벨이 같아야 한다"

    def test_evaluation_loss_is_regime_independent(self):
        """`loss()` 가 항상 full dataset 이어야 Track E 가 regime 간 비교 가능하다."""
        import dataclasses

        fb = build(SMALL)
        cs = build(dataclasses.replace(SMALL, regime="controlled_stochastic", batch_size=16))
        flat = torch.linspace(-0.2, 0.2, SMALL.n_parameters, dtype=torch.float64)
        fb.move_to(flat)
        cs.move_to(flat)
        assert float(fb.loss()) == float(cs.loss())

    def test_data_key_excludes_regime_and_batch_size(self):
        import dataclasses

        base = SMALL
        variants = [
            dataclasses.replace(base, regime="controlled_stochastic", batch_size=16),
            dataclasses.replace(base, regime="controlled_stochastic", batch_size=8),
        ]
        assert all(v.data_key == base.data_key for v in variants)
        # instance_id 는 달라야 한다. 저장소에서 run 이 구별되어야 하기 때문이다.
        ids = {v.instance_id(0) for v in [base, *variants]}
        assert len(ids) == 3

    def test_data_key_includes_model_and_data_shape(self):
        import dataclasses

        base = SMALL
        for field, value in (
            ("input_dim", 9),
            ("hidden_dim", 17),
            ("n_classes", 4),
            ("n_samples", 65),
            ("teacher_hidden_dim", 33),
            ("label_noise", 0.1),
            ("init_scale", 2.0),
        ):
            assert dataclasses.replace(base, **{field: value}).data_key != base.data_key

    def test_advance_does_not_change_evaluation_loss(self):
        task = build(STOCHASTIC)
        before = float(task.loss())
        for _ in range(7):
            task.advance_batch()
        assert float(task.loss()) == before


class TestTaskInterface:
    def test_loss_is_scalar_with_graph(self):
        task = build(SMALL)
        for closure in (task.loss, task.curvature_loss):
            value = closure()
            assert value.dim() == 0
            assert value.grad_fn is not None

    def test_hvp_is_well_defined(self):
        """HVP 가 계산 가능해야 Newton-CG 를 돌릴 수 있다."""
        from rl_newton.curvature.hvp import HvpGraph

        task = build(SMALL)
        graph = HvpGraph(task.curvature_loss, task.params)
        v = torch.randn(SMALL.n_parameters, dtype=torch.float64)
        hv = graph.matvec(v)
        assert hv.shape == v.shape
        assert bool(torch.isfinite(hv).all())

    def test_hessian_is_symmetric(self):
        """``v^T H w == w^T H v``. HVP 구현 검증이다."""
        from rl_newton.curvature.hvp import HvpGraph

        task = build(SMALL)
        graph = HvpGraph(task.curvature_loss, task.params)
        v = torch.randn(SMALL.n_parameters, dtype=torch.float64)
        w = torch.randn(SMALL.n_parameters, dtype=torch.float64)
        left = float(torch.dot(w, graph.matvec(v)))
        right = float(torch.dot(v, graph.matvec(w)))
        assert left == pytest.approx(right, rel=1.0e-9, abs=1.0e-12)

    def test_bounded_below_and_optimal_loss(self):
        task = build(SMALL)
        assert task.is_bounded_below
        assert task.optimal_loss == 0.0

    def test_initial_loss_is_finite_and_positive(self):
        task = build(SMALL)
        assert math.isfinite(task.initial_loss)
        assert task.initial_loss > 0.0

    def test_reset_restores_params_and_batch_index(self):
        task = build(STOCHASTIC)
        original = [p.detach().clone() for p in task.params]
        task.move_to(torch.ones(STOCHASTIC.n_parameters, dtype=torch.float64))
        task.advance_batch()
        task.reset()
        assert task.step_index == 0
        for p, o in zip(task.params, original, strict=True):
            assert torch.equal(p, o)

    def test_move_to_rejects_wrong_size(self):
        task = build(SMALL)
        with pytest.raises(ValueError):
            task.move_to(torch.zeros(3, dtype=torch.float64))

    def test_move_to_roundtrips(self):
        task = build(SMALL)
        point = torch.linspace(-1.0, 1.0, SMALL.n_parameters, dtype=torch.float64)
        task.move_to(point)
        flat = torch.cat([p.detach().reshape(-1) for p in task.params])
        assert torch.allclose(flat, point)


class TestNoFloorSaturation:
    """teacher 가 학생보다 넓고 라벨 노이즈가 있어 loss 가 0 으로 붕괴하지 않아야 한다.

    D19 에서 `quad_spd` 와 `rosen_d2` 가 numerical floor 로 포화해 컨트롤러를
    구분하지 못했다. 같은 실패를 반복하면 안 된다.
    """

    def test_label_noise_makes_teacher_unreachable(self):
        import dataclasses

        noisy = dataclasses.replace(SMALL, label_noise=0.2)
        task = build(noisy)
        # 라벨이 뒤집혔으므로 teacher 를 완벽히 복원해도 오분류가 남는다.
        assert task.accuracy() < 1.0

    def test_accuracy_is_reported_but_not_the_objective(self):
        task = build(SMALL)
        acc = task.accuracy()
        assert 0.0 <= acc <= 1.0

    def test_teacher_is_wider_than_student_by_default(self):
        assert MicroNeuralSpec().teacher_hidden_dim > MicroNeuralSpec().hidden_dim
