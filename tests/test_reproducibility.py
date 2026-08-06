"""paired design의 기반이 되는 결정론성 테스트 (프로토콜 D7).

seed는 난수 시드가 아니라 실험 조건 식별자다. seed=s 이면 모든 optimizer가
동일한 task 인스턴스와 minibatch 순서를 본다. 그 전제가 깨지면 쌍별 비교
통계 전체가 무의미해지므로 여기서 고정한다.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from rl_newton.utils.provenance import config_hash, git_commit
from rl_newton.utils.seed import seed_everything, spawn_seed, torch_generator


class TestSpawnSeed:
    def test_is_deterministic_within_process(self):
        assert spawn_seed(0, "task") == spawn_seed(0, "task")
        assert spawn_seed(7, "batch_order", 3) == spawn_seed(7, "batch_order", 3)

    def test_different_namespaces_give_different_streams(self):
        base = 0
        derived = {
            spawn_seed(base, "task"),
            spawn_seed(base, "batch_order"),
            spawn_seed(base, "init"),
        }
        assert len(derived) == 3

    def test_different_base_seeds_give_different_streams(self):
        assert spawn_seed(0, "task") != spawn_seed(1, "task")

    def test_result_fits_in_uint32(self):
        for s in range(50):
            value = spawn_seed(s, "task", s * 3)
            assert 0 <= value < 2**32

    def test_is_stable_across_processes(self):
        """내장 hash()는 PYTHONHASHSEED로 salt되므로 프로세스마다 값이 달라진다.

        blake2b를 쓰는 이유가 바로 이것이다. 별도 프로세스를 띄워
        실행 간 안정성을 실제로 확인한다.
        """
        code = (
            "from rl_newton.utils.seed import spawn_seed; "
            "print(spawn_seed(42, 'task'), spawn_seed(42, 'batch_order'))"
        )
        expected = f"{spawn_seed(42, 'task')} {spawn_seed(42, 'batch_order')}"

        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        assert proc.stdout.strip() == expected


class TestTorchGenerator:
    def test_same_namespace_reproduces_same_samples(self):
        a = torch.randn(16, generator=torch_generator(0, "batch_order"))
        b = torch.randn(16, generator=torch_generator(0, "batch_order"))
        assert torch.equal(a, b)

    def test_different_namespace_gives_different_samples(self):
        a = torch.randn(16, generator=torch_generator(0, "batch_order"))
        b = torch.randn(16, generator=torch_generator(0, "init"))
        assert not torch.equal(a, b)

    def test_does_not_disturb_global_rng(self):
        """국소 generator는 전역 스트림을 오염시키지 않아야 한다."""
        seed_everything(123)
        expected = torch.randn(4)

        seed_everything(123)
        _ = torch.randn(8, generator=torch_generator(999, "unrelated"))
        actual = torch.randn(4)

        assert torch.equal(expected, actual)


class TestSeedEverything:
    def test_reproduces_torch_stream(self):
        seed_everything(2024)
        first = torch.randn(32)
        seed_everything(2024)
        second = torch.randn(32)
        assert torch.equal(first, second)

    def test_rejects_out_of_range_seed(self):
        with pytest.raises(ValueError, match="uint32"):
            seed_everything(2**32)


class TestProvenance:
    def test_config_hash_ignores_key_order(self):
        assert config_hash({"a": 1, "b": {"c": 2}}) == config_hash({"b": {"c": 2}, "a": 1})

    def test_config_hash_detects_value_change(self):
        assert config_hash({"lr": 0.001}) != config_hash({"lr": 0.002})

    def test_git_commit_returns_status_tuple(self):
        commit, dirty = git_commit(".")
        assert isinstance(commit, str)
        assert commit
        assert isinstance(dirty, bool)
