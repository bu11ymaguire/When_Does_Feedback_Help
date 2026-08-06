"""결정론적 시드 설정.

프로토콜 D7(paired design)에서 ``seed`` 는 난수 시드가 아니라 **실험 조건
식별자**로 쓴다. ``seed=s`` 일 때 모든 optimizer가 동일한 task 인스턴스,
동일한 minibatch 순서, 동일한 train/val split을 보아야 한다.

그래서 파생 시드는 ``random`` 이나 내장 ``hash()`` 로 만들지 않는다.
파이썬의 ``hash()`` 는 문자열에 대해 프로세스마다 salt가 달라지므로
(``PYTHONHASHSEED``) 실행 간 재현성이 깨진다. 대신 blake2b 해시를 쓴다.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch

__all__ = ["seed_everything", "spawn_seed", "torch_generator"]

_UINT32_MAX = 0xFFFF_FFFF


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """python / numpy / torch / cuda 시드를 모두 설정한다.

    Args:
        seed: 기준 시드.
        deterministic_algorithms: ``True`` 이면 ``torch.use_deterministic_algorithms``
            를 켠다. 비트 단위 재현성을 얻지만 일부 연산이 느려지거나
            ``RuntimeError`` 를 던질 수 있다. 기본값은 ``False`` 이고,
            재현성 검증 테스트에서만 켠다.

    Note:
        cuDNN benchmark를 끄면 커널 선택이 고정되어 wall-clock 측정의
        분산이 줄어든다. 이 프로젝트는 wall-clock을 보조 지표로 쓰므로
        기본적으로 끈다.
    """
    if not 0 <= seed <= _UINT32_MAX:
        raise ValueError(f"seed must fit in uint32, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False

    if deterministic_algorithms:
        # 일부 CUDA 커널은 이 환경변수가 없으면 예외를 던진다.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True


def spawn_seed(base_seed: int, *namespace: str | int) -> int:
    """``base_seed`` 와 namespace로부터 결정론적 파생 시드를 만든다.

    같은 인자를 주면 프로세스와 실행 시점에 무관하게 항상 같은 값이 나온다.
    paired design에서 하나의 ``seed`` 로 여러 독립적인 난수 스트림
    (task 인스턴스 / 모델 초기화 / 배치 순서)을 만들 때 쓴다.

    Args:
        base_seed: 실험 조건 식별자.
        *namespace: 스트림 이름. 예: ``"task"``, ``"batch_order"``, ``"init"``.

    Returns:
        ``[0, 2**32)`` 범위의 정수.

    Example:
        >>> spawn_seed(0, "task") == spawn_seed(0, "task")
        True
        >>> spawn_seed(0, "task") != spawn_seed(0, "batch_order")
        True
    """
    key = "|".join([str(base_seed), *(str(n) for n in namespace)])
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big")


def torch_generator(base_seed: int, *namespace: str | int, device: str = "cpu") -> torch.Generator:
    """``spawn_seed`` 로 초기화된 ``torch.Generator`` 를 만든다.

    전역 시드를 오염시키지 않고 국소적인 난수 스트림을 쓰고 싶을 때 사용한다.
    task 인스턴스 생성이나 배치 순서 shuffle에 적합하다.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(spawn_seed(base_seed, *namespace))
    return gen
