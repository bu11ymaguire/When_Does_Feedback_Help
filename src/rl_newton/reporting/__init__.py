"""공개 결과 파일을 읽고 원고의 통계를 재계산하는 얇은 계층.

**실험 경로 코드가 아니다.** 이 패키지는 읽기 전용이며 optimizer, 컨트롤러, task,
비용 회계에 손대지 않는다. 따라서 이 모듈을 추가해도 기존 결과의 정체성
(`run_semantics_id`, `aggregation_payload`)은 바뀌지 않는다.

notebook 과 검증 스크립트가 통계 구현을 복사하지 않도록 여기서만 제공한다.
"""

from rl_newton.reporting.public import (
    PUBLIC_COLUMNS,
    load_public_grouped,
    load_public_results,
    paired,
    positive_count,
    public_roles,
    split_by,
)

__all__ = [
    "PUBLIC_COLUMNS",
    "load_public_grouped",
    "load_public_results",
    "paired",
    "positive_count",
    "public_roles",
    "split_by",
]
