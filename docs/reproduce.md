# 재현 명령

Stage 2 의 모든 결과를 재생성하는 명령이다. 순서대로 실행하면 된다.

## 환경

```text
Python 3.12
torch (CPU. GPU 를 쓰지 않는다)
단일 스레드 고정          wall-clock tie-break 안정화 (프로토콜 §환경)
float64 는 진단 스크립트에서만. 실험 본문은 기본 dtype
```

```bash
python -m pip install -e .
python -m pytest tests/ -q
python -m ruff check .
```

## 왜 `run_semantics_id` 를 신뢰할 수 있는가

`ResultStore` 는 완료된 run 을 건너뛴다. 정체성이 3계층으로 분리되어 있어
(프로토콜 D13) 집계 코드나 문서를 고쳐도 optimizer 가 다시 돌지 않는다.

```text
run_semantics_id  이 컨트롤러가 실제 쓰는 optimizer 설정만
sweep_id          이번 실행이 요청한 run 집합
aggregation_id    집계 정책
```

`git_commit` 과 `code_dirty` 는 어떤 ID 에도 넣지 않고 `execution_provenance` 로
분리한다. 문서만 고쳐도 해시가 바뀌면 "어떤 집합을 요청했는가" 라는 의미가 깨진다.

## 1. Beam 폭 선정 (D8)

```bash
python scripts/run_headroom.py --mode calibrate-beam --beams 1 2 4 --cal-quotas 1 4 --budget 150
```

## 2. Dev pilot (원래 subset, beam 4)

```bash
python scripts/run_headroom.py --mode pilot --beam 4 --quotas 2 4 --budget 150 \
    --modes shrinking committed --skip-track-t
```

이 결과에서 2개 spec 이 포화됐다는 것이 확인됐다 (D19).

## 3. Bridge 검증 (D18)

D13 / D16 / D17 이 planner 실행 궤적을 바꾸지 않았는지 확인한다. 기대값은
**bitwise exact** 다.

```bash
python scripts/bridge_validate.py \
    --legacy-file results/raw/headroom_pilot_step_size_fixed_b4_8e9cdd02.jsonl \
    --legacy <legacy_experiment_id_prefix> \
    --new-file results/raw/headroom_pilot_step_size_fixed_b4_9d725689.jsonl \
    --modes shrinking committed --quotas 2 4 --spaces narrow wide
```

## 4. Challenge set calibration (D20 / D25)

**planner 결과를 열지 않는다.** 비적응 baseline 과 참조 solver panel 만 쓴다.

```bash
python scripts/calibrate_challenge.py --seeds 0 1 --reference-iters 3000
```

seed 역할이 분리되어 있다.

```text
CALIBRATION_SEEDS   0, 1          challenge spec 선정에만
SELECTION_SEEDS     2, 3, 4       설정 선택에만
HELD_OUT_SEEDS      100 ~ 109     최종 평가에만
```

## 5. Beam 8 challenge dev (설정 선택, D21 / D22)

```bash
python scripts/run_headroom.py --mode challenge --beam 8 --budget 150 --seeds 3 \
    --quotas 2 4 --modes shrinking committed --fresh-seeds 0 \
    --planner-spaces narrow wide --tuning-budget 6
python scripts/select_configuration.py results/raw/headroom_challenge_step_size_fixed_b8_fed9aebd.jsonl
```

선택 규칙은 실행 전에 고정했다 (D21). `shrinking` 자신의 median logΔ 를 최대화하고,
`0.05 nat` 이내 동률이면 `decision-search GE → 작은 Q → narrow` 순으로 tie-break 한다.

결과: `shrinking_Q4_narrow` 단독 선택.

## 6. Held-out confirmatory (D26)

**설정을 다시 고르지 않는다.**

```bash
python scripts/run_headroom.py --mode challenge-heldout --beam 8 --budget 150 --seeds 10 \
    --quotas 4 --modes shrinking committed --fresh-seeds 0 \
    --planner-spaces narrow --tuning-budget 6
python scripts/select_configuration.py results/raw/headroom_challenge-heldout_step_size_fixed_b8_9a18b6e9.jsonl
```

## 7. 비선형 진단 (D23. **사용 불가 판정**)

```bash
python scripts/run_headroom.py --mode nonlinear-diagnostic --beam 8 --budget 150 --seeds 3 \
    --quotas 4 --modes shrinking committed --fresh-seeds 0 --planner-spaces narrow --tuning-budget 6
python scripts/diagnose_rosen.py results/raw/headroom_nonlinear-diagnostic_step_size_fixed_b8_2a09bd45.jsonl --seed 2
python scripts/probe_rosen_basin.py
python scripts/probe_instance_variation.py
```

`rosen_d5` 는 국소최소점에 갇혀 있고 `randomize_start=False` 라 seed 가 복제된다.
두 결함 모두 D25 의 교정된 eligibility 조건으로 자동 탈락한다.

## 8. Micro-neural 두 regime (D24 / D27 / D28 / D29)

```bash
python scripts/probe_micro_neural.py
python scripts/run_headroom.py --mode micro-neural --beam 8 --budget 150 --seeds 3 \
    --quotas 4 --modes shrinking committed --fresh-seeds 0 --planner-spaces narrow \
    --tuning-budget 6 --acceptance-loss control
python scripts/run_headroom.py --mode micro-neural --beam 8 --budget 150 --seeds 3 \
    --quotas 4 --modes shrinking committed --fresh-seeds 0 --planner-spaces narrow \
    --tuning-budget 6 --acceptance-loss fixed_eval
python scripts/analyze_regimes.py results/raw/headroom_micro-neural_step_size_fixed_b8_<hash>.jsonl
```

`--acceptance-loss fixed_eval` 은 수락 판정만 고정 목적함수로 옮기는 ablation 이다
(D28). gradient 와 HVP 는 계속 minibatch 를 쓰고, 평가 forward 비용은 GE 회계에
포함된다.

`acceptance_loss` 기본값은 `run_semantics_id` 에 들어가지 않으므로 기존 결과가
보존된다. 결정론적 task 는 `acceptance_loss()` 를 제공하지 않아 자동으로 `control`
로 되돌아간다.

## 9. 결과표 생성

```bash
python scripts/make_report.py --out docs/results_stage2.md
```

## 재집계 (재실행 없이)

집계 정책만 바꿔 다시 계산한다.

```bash
python scripts/reaggregate.py results/raw/<file>.jsonl [experiment_id_prefix]
```

## 진단 도구

```text
scripts/probe_regime.py            paired delta 가 0 인 쌍의 원인
scripts/diagnose_excluded_pairs.py 조용한 dropna 방지
scripts/probe_quota.py             쿼터 사다리
scripts/probe_execution_modes.py   committed / fresh / shrinking 비교
scripts/probe_search_quality.py    탐색 품질
scripts/compare_controllers.py     컨트롤러 대조
scripts/show_progress.py           진행 상황
```

## 알려진 한계

```text
micro-neural 은 regime 당 n=3. CI 와 p-value 를 인용하지 않는다
batch_size 는 3 점(full / 128 / 64). 전이 방향만 탐색적으로 본다
게이트 C1 (fresh) 은 seed 1개 진단 baseline 이므로 판정에 쓰지 않는다
three_layer 가 A1/B 게이트에는 미적용 (단일 통계 사용)
```
