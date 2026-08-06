"""벤치마크 — 비용 모델, 실행기, 지표, 그림.

주 지표는 wall-clock이 아니라 **grad-equivalent(GE)** 다 (프로토콜 D1).
MNIST MLP 규모(약 10만 파라미터)에서는 GPU 시간이 FLOP이 아니라 커널 런치
오버헤드에 지배되므로, wall-clock으로는 최적화 효율을 측정할 수 없다.

구현 완료 (Stage 0/1)
---------------------
``cost_model.py``
    대상 하드웨어에서 grad / HVP / forward 비용을 실측해 GE 환산계수를 산출한다.
    이론값(c_hvp ~ 2.5, c_fwd ~ 0.3)을 쓰지 않고 측정값을 쓴다.

구현 완료 (Stage 1)
-------------------
``paired.py``
    ``seed -> (task_instance, batch_order, init)`` 결정론적 매핑.
    모든 optimizer가 동일 조건을 보게 해 분산을 줄인다 (프로토콜 D7).
    task 생성 스트림을 optimizer 실행 스트림과 분리하므로, optimizer가
    난수를 얼마나 소비하든 문제 인스턴스는 동일하다.

구현 예정 (Stage 2)
-------------------
``oracle.py``
    헤드룸 측정. 이 프로젝트의 분기점이다.
      - ``best_static``   : 36개 action 조합 고정 실행 후 최고 선택
      - ``best_open_loop``: progress 만 보는 스케줄 랜덤 서치
      - ``greedy_oracle`` : 매 step 36개 action을 모두 시도해
                            (delta log L / cost_GE) 최대인 것 선택
    게이트: ``cost_to_target(best_static) / cost_to_target(greedy_oracle)``
    기하평균이 1.10 미만이면 RL 단계로 넘어가지 않는다.
    부수 산출물: greedy trajectory = behavior cloning 데이터셋.

구현 예정 (Stage 3)
-------------------
``runner.py``
    paired design 실행기. JSONL 로깅, 비용 회계, 예산 소진 판정.

``metrics.py``
    cost-to-target 집계. **절단 규칙 준수** (프로토콜 D6):
    미도달 run을 버리거나 최댓값으로 대입하지 않고, ``success_rate`` 와
    도달 run의 **중앙값** 을 함께 보고한다.
    통계: Wilcoxon signed-rank + 비율의 기하평균 + 부트스트랩 95% CI,
    주 가설 3개에 Holm 보정.

``plotting.py``
    주 그림은 loss vs **cumulative GE**. wall-clock 그림에는 오버헤드 지배
    여부를 표기한다. 핵심 그림은 greedy_oracle / best_static / rl 3자 비교.
"""

from rl_newton.benchmark.cost_model import CostModel, measure_cost_model
from rl_newton.benchmark.metrics import (
    GroupSummary,
    PairedComparison,
    PairedDelta,
    RunSummary,
    TargetSpec,
    compare_paired,
    compare_paired_delta,
    recovery_ratio,
    summarize_group,
    summarize_run,
)
from rl_newton.benchmark.paired import (
    PairedTaskFactory,
    make_task,
    quadratic_meta_test_specs,
    quadratic_meta_train_specs,
)

__all__ = [
    "CostModel",
    "measure_cost_model",
    "PairedTaskFactory",
    "make_task",
    "quadratic_meta_test_specs",
    "quadratic_meta_train_specs",
    "TargetSpec",
    "RunSummary",
    "GroupSummary",
    "PairedComparison",
    "PairedDelta",
    "summarize_run",
    "summarize_group",
    "compare_paired",
    "compare_paired_delta",
    "recovery_ratio",
]
