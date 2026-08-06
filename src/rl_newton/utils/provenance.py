"""run provenance 수집.

프로토콜 §3 원칙: 모든 run은 config 스냅샷과 git commit hash를 함께 저장한다.
working tree가 dirty하면 commit hash만으로는 코드를 복원할 수 없으므로
그 사실을 명시적으로 기록하고, 필요하면 diff까지 남긴다.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["RunProvenance", "collect_provenance", "config_hash", "git_commit", "git_diff"]

_GIT_TIMEOUT_SEC = 10


def _run_git(args: list[str], repo: Path) -> str | None:
    """git 명령을 실행하고 stdout을 반환한다. 실패하면 ``None``."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_commit(repo: str | Path = ".", *, short: bool = True) -> tuple[str, bool]:
    """현재 commit hash와 dirty 여부를 반환한다.

    Args:
        repo: 저장소 경로.
        short: ``True`` 이면 축약 hash.

    Returns:
        ``(commit, is_dirty)``. git 정보를 얻을 수 없으면 ``("unknown", True)``.
        커밋이 아예 없는 저장소는 ``("no-commit", is_dirty)``.

    Note:
        정보를 못 얻었을 때 ``is_dirty=True`` 로 두는 것은 의도적이다.
        재현 가능성을 낙관적으로 가정하지 않는다.
    """
    path = Path(repo).resolve()
    rev = _run_git(["rev-parse", "--short=8" if short else "HEAD", "HEAD"], path)
    if rev is None:
        # 커밋이 없는 저장소인지, git이 아예 없는지 구분한다.
        inside = _run_git(["rev-parse", "--is-inside-work-tree"], path)
        commit = "no-commit" if inside == "true" else "unknown"
    else:
        commit = rev

    status = _run_git(["status", "--porcelain"], path)
    is_dirty = True if status is None else bool(status)
    return commit, is_dirty


def git_diff(repo: str | Path = ".") -> str | None:
    """추적 중인 파일의 diff를 반환한다. dirty run의 코드 상태 보존용."""
    return _run_git(["diff", "HEAD"], Path(repo).resolve())


def config_hash(config: dict[str, Any]) -> str:
    """config dict의 정규화된 해시.

    키를 정렬해 직렬화하므로 dict 삽입 순서에 영향받지 않는다.
    같은 설정이면 언제 어디서 실행해도 같은 값이 나온다.

    Example:
        >>> config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
        True
    """
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, default=repr)
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).hexdigest()


@dataclass(slots=True)
class RunProvenance:
    """run 하나를 재현하기 위한 최소 정보."""

    git_commit: str
    git_dirty: bool
    config_hash: str
    python_version: str
    platform: str
    torch_version: str
    cuda_available: bool
    cuda_runtime: str | None = None
    gpu_name: str | None = None
    gpu_capability: str | None = None
    gpu_total_memory_mb: float | None = None
    config: dict[str, Any] = field(default_factory=dict)
    diff: str | None = None

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def collect_provenance(
    config: dict[str, Any] | None = None,
    *,
    repo: str | Path = ".",
    include_diff: bool = True,
) -> RunProvenance:
    """현재 실행 환경과 코드 상태를 수집한다.

    Args:
        config: 실험 설정. 해시와 스냅샷으로 함께 기록된다.
        repo: git 저장소 경로.
        include_diff: dirty일 때 diff를 포함할지. 기본 ``True``.
            dirty run의 코드를 사후에 복원할 수 있게 한다.

    Returns:
        ``RunProvenance``.
    """
    import torch  # 지연 import: git/config 해시만 쓸 때 torch 로딩을 피한다

    cfg = config or {}
    commit, dirty = git_commit(repo)

    cuda_available = torch.cuda.is_available()
    gpu_name: str | None = None
    gpu_capability: str | None = None
    gpu_total_mb: float | None = None
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        gpu_capability = f"{major}.{minor}"
        gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2

    return RunProvenance(
        git_commit=commit,
        git_dirty=dirty,
        config_hash=config_hash(cfg),
        python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()}",
        # torch.__version__ 은 str 서브클래스(TorchVersion)다. 직렬화 안전을 위해 정규화한다.
        torch_version=str(torch.__version__),
        cuda_available=cuda_available,
        cuda_runtime=str(torch.version.cuda) if torch.version.cuda else None,
        gpu_name=gpu_name,
        gpu_capability=gpu_capability,
        gpu_total_memory_mb=gpu_total_mb,
        config=cfg,
        diff=git_diff(repo) if (include_diff and dirty) else None,
    )
