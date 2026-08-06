"""공용 유틸리티.

이 서브패키지는 Stage 0에서 완전히 구현된다. 나머지 모든 모듈이 의존하므로
동작이 검증된 코드만 둔다.

``flatten``     파라미터 리스트 <-> 단일 1차원 벡터 변환
``seed``        결정론적 시드 설정
``logging``     step 단위 JSONL 로거
``provenance``  git commit / config hash / 환경 정보 수집
"""

from rl_newton.utils.flatten import (
    ParameterFlattener,
    flatten_tensors,
    unflatten_like,
)
from rl_newton.utils.logging import JsonlStepLogger
from rl_newton.utils.provenance import (
    RunProvenance,
    collect_provenance,
    config_hash,
    git_commit,
)
from rl_newton.utils.seed import seed_everything, spawn_seed

__all__ = [
    "ParameterFlattener",
    "flatten_tensors",
    "unflatten_like",
    "JsonlStepLogger",
    "RunProvenance",
    "collect_provenance",
    "config_hash",
    "git_commit",
    "seed_everything",
    "spawn_seed",
]
