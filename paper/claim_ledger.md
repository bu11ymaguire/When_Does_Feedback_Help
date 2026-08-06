# Claim ledger

원고에 들어갈 **모든 핵심 주장**을 근거와 함께 관리한다. 초안의 문장은 여기 등록된
주장만 쓴다. 등록되지 않은 새 해석을 본문에 넣지 않는다.

## 상태 라벨

```text
SUPPORTED       사전 고정 설정으로 held-out 에서 측정됐고 부호와 CI 가 일관됨
LIMITED         측정됐으나 표본이 작거나 교란이 남아 크기를 주장하지 않음
EXPLORATORY     탐색적. primary 결론의 근거로 쓰지 않음
NOT SUPPORTED   데이터가 지지하지 않음. **원고에 쓰지 않는다**
```

## 결과의 역할 (사전 지정)

```text
beam 4 dev pilot          탐색적. 2개 spec 이 포화됐다 (D19)
beam 8 challenge dev      configuration selection 근거만 (D21/D22)
seed 100~109              held-out confirmation. 최종 효과 추정 (D26)
micro-neural              exploratory ablation (D27/D31)
acceptance ablation       exploratory ablation (D28/D31)
rosen_d2 / rosen_d5       diagnostic. 사용 불가 판정 (D19/D23)
```

## 통계 표기 규칙 (D26)

```text
p 를 0.0000 으로 쓰지 않는다. p<0.0001 로 쓴다
equivalence margin 을 사전 등록하지 않았으므로 "효과가 0 이다" 를 주장하지 않는다
CI 가 좁게 0 을 포함하면 "실용적으로 큰 이득이 관측되지 않았다" 로 쓴다
```

---

## SUPPORTED

### C1. 자원 시계 기반 open-loop 스케줄이 튜닝된 상수 설정보다 낫다

```text
Evidence     held-out n=40
             median paired delta +0.395 nat
             95% CI +0.350 ~ +0.476
             p<0.0001
source       docs/results_stage2.md  (held-out confirmatory 절)
raw          headroom_challenge-heldout_step_size_fixed_b8_9a18b6e9.jsonl
protocol     D17 (progress 를 GE 비율로), D26
status       SUPPORTED
```

**단서.** `best_open_loop` 는 컨트롤러가 아니라 **튜닝 결과**다 (D16). 인스턴스당
878 GE 의 튜닝 비용을 썼다. 원고에서 "스케줄이 공짜로 더 좋다" 로 읽히지 않게
튜닝 비용을 함께 적는다.

### C2. One-step 상태 의존 제어가 튜닝된 상수 설정보다 낫다

```text
Evidence     held-out n=40
             onestep_narrow − best_static  = +1.155 nat
             95% CI +1.092 ~ +1.811,  p<0.0001,  40/40 양수
             onestep_absolute − best_static = +1.423 nat  (게이트 A1)
source       docs/results_stage2.md  (ladder 행)
raw          headroom_challenge-heldout_...9a18b6e9.jsonl
protocol     게이트 A1, D26
status       SUPPORTED
```

**금지.** `A2 − C2` 로 이 값을 계산하지 않는다. 쌍별 차이의 median 은 선형이 아니다.
초판 초안이 `1.690 − 0.456 = 1.233` 으로 적었고 직접 측정값은 `+1.155` 다 (C26).

### C3. 다단계 planning 이 held-out 에서 one-step 제어를 이긴다

```text
Evidence     held-out n=40
             median paired delta +0.456 nat
             95% CI +0.254 ~ +0.720
             p<0.0001
             35 positive / 5 negative
             depth>1 채택률 0.84,  plan-depth 상한 도달 0.00
source       docs/results_stage2.md
raw          headroom_challenge-heldout_...9a18b6e9.jsonl
protocol     게이트 C2, P3 (두 조건 모두 필요), D26
status       SUPPORTED
```

**P3 의 두 조건이 모두 충족됐다.** 개선이 있고, 동시에 `depth>1` 이 실제로
채택됐다. 개선만 있고 depth 가 계속 1 이면 planning 이 아니라 탐색량이 기여한
것이다. `cap=0.00` 이므로 계산 상한 때문에 쿼터를 못 쓴 step 도 없다.

**dev 에서는 조건부였다** (`+0.251`, `p=0.0771`). held-out `n=40` 에서 GO 로
승격됐다. 이 변화를 숨기지 않고 보고한다.

### C4. 결정론적 quadratic 에서 실행 중 재계획의 큰 추가 이득은 관측되지 않았다

```text
Evidence     held-out n=40
             shrinking − committed = +0.010 nat
             95% CI −0.033 ~ +0.053
             p=0.97
             21 positive / 1 zero / 18 negative
             CI 폭 0.086 nat  (비교: A2 의 CI 폭 0.906 nat)
source       docs/results_stage2.md
raw          headroom_challenge-heldout_...9a18b6e9.jsonl
protocol     게이트 C3, D26
status       SUPPORTED  (단, 아래 표현만 허용)
```

**허용되는 문장.**

> held-out 결과에서 feedback 효과는 `+0.010 nat` 였으며 95% CI 가
> `[−0.033, +0.053]` 으로 좁게 0 을 포함했다. 따라서 실용적으로 큰 feedback 이득은
> 관측되지 않았다.

**금지되는 문장.** C11 참조.

### C5. 행동 공간의 도달성은 이 설정에서 병목이 아니다

```text
Evidence     held-out n=40
             onestep_absolute − onestep_narrow = +0.005 nat
             dev n=12: +0.015 nat,  95% CI −0.070 ~ +0.055,  p=0.91
             absolute 132 action (log10 범위 15.27) vs narrow 12 action (0.95)
             wide − narrow = −0.001 nat (held-out)
source       docs/results_stage2.md
raw          headroom_challenge-heldout_...9a18b6e9.jsonl
protocol     게이트 B, D26
status       SUPPORTED
```

damping 을 자유롭게 고를 수 있게 해도 1-step 성능이 오르지 않는다. **음의 결과지만
값싼 좁은 multiplier 공간을 정당화한다.**

### C6. Oracle planner 의 탐색 비용은 배포 예산의 세 자릿수 배다

```text
Evidence     held-out n=40, shrinking_Q4_narrow
             decision-search 194,095 GE / object budget 150 GE = 1,294배
             committed_Q4_narrow  약 462배 (dev)
             onestep_narrow       약 7.9배 (dev)
             best_open_loop 튜닝  인스턴스당 878 GE
source       docs/results_stage2.md (탐색 비용 표)
protocol     D26
status       SUPPORTED
```

**이 수치를 숨기지 않는다.** 보고된 헤드룸은 예산의 1,294배를 쓴 oracle 값이다.

### C7. 사다리별 측정값. **합으로 분해되지 않는다**

```text
Evidence     held-out n=40. 모두 튜닝 상수 기준 쌍별 median
             best_open_loop        +0.395   CI +0.350 ~ +0.476   40/40
             onestep_narrow        +1.155   CI +1.092 ~ +1.811   40/40
             committed_Q4_narrow   +2.090   CI +1.532 ~ +2.407   40/40
             shrinking_Q4_narrow   +1.690   CI +1.462 ~ +2.368   40/40   (A2)

             직접 측정한 증분
             shrinking − onestep   +0.456   CI +0.254 ~ +0.720   35/40   (C2)
             shrinking − committed +0.010   CI −0.033 ~ +0.053   21/40   (C3)
source       docs/results_stage2.md  (ladder 행 포함)
protocol     D26
status       SUPPORTED
```

**표의 값을 서로 빼지 않는다.** `committed` 의 상수 대비 값(`+2.090`)이 `shrinking`
의 값(`+1.690`)보다 크지만 직접 측정한 `shrinking − committed` 는 `+0.010` 이다.
spec 별로는 `+0.472 / −0.019 / +0.009 / +0.000` 로 두 planner 가 사실상 동률이고,
pooled median 이 서로 다른 인스턴스에 떨어져 marginal 값 차이가 생긴다.

정성적 결론만 주장한다.

```text
가장 큰 단일 기여   상수 -> 1-step 의 +1.155
다단계 증분        +0.456,  CI 하한 +0.254
재계획 증분        +0.010,  CI [−0.033, +0.053]
```

### C8. Benchmark eligibility 는 수치 하한이 아니라 **도달 가능한** 상한으로 판정해야 한다

```text
Evidence     rosen_d5 (표준 시작점)
             국소최소점 x* = (−0.96205102, 0.93573939, 0.88071360, 0.77787767, 0.60509367)
             loss 3.930839434133,  |grad| 1.06e-08,  Hessian eig min +0.595 (양정)
             수치 하한 기준 ceiling 31.44 nat  ->  통과
             도달 가능 상한         1.8175 nat  ->  네 baseline 전부 정확히 그 값
             교정 조건 적용 시 여유 −0.00 / 0.00  ->  탈락
source       docs/experiment_protocol.md D23, D25
raw          headroom_nonlinear-diagnostic_...2a09bd45.jsonl
tools        scripts/probe_rosen_basin.py, scripts/probe_instance_variation.py
status       SUPPORTED
```

방법론 기여로 보고할 수 있다. **결과를 본 뒤 만든 사후 임계값이 아니라 기전 기반
조건이다.**

### C9. 단일 참조 solver 로 도달 가능 상한을 추정하면 과소평가한다

```text
Evidence     quad_d100_k1e6, seed 0
             lbfgs   final 1.007485e-06,  |grad| 3.456e-02,  미수렴
             newton  final 3.059391e-32,  |grad| 2.650e-16,  수렴
             sgd     final nan
             lbfgs 만 썼다면 J_achievable 을 약 28.3 nat 로 과소평가
source       docs/experiment_protocol.md D25
tools        scripts/calibrate_challenge.py
status       SUPPORTED
```

### C10. 다른 초기화의 최적값은 도달 가능 상한이 아니다

```text
Evidence     rosen_d5 (randomize_start=True)
             task 시작점 basin 의 임계점  L_ref = 3.930839
             다른 초기화 (0.9,...,0.9)    4.810595e-21
             후자를 L_ref 에 넣으면 J_achievable 이 2.58 -> 31.44 로 부풀고
             국소최소점에 갇힌 spec 이 eligibility 를 통과한다
source       docs/experiment_protocol.md D25
tests        tests/test_eligibility.py::test_extra_init_diagnoses_but_does_not_raise_the_ceiling
status       SUPPORTED
```

컨트롤러는 항상 task 의 시작점에서 출발한다. 다른 basin 의 최적값은 **진단**이다.

---

## LIMITED

### C12. Stochastic regime 에서 stale plan 을 고수하면 큰 손해가 발생한다

```text
Evidence     micro-neural exploratory, regime 당 n=3
             control 수락 규칙
               committed_Q4_narrow  full_batch 20.491  ->  cs64 1.086,  cs128 −0.402
               거절률                full_batch 0.00    ->  cs64 0.66,  cs128 0.79
               committed − onestep   cs64 −2.322 (0/3),  cs128 −4.444 (0/3)
             fixed_eval 수락 규칙에서도 committed 가 가장 낮다
source       docs/results_stage2.md (micro-neural 절)
raw          headroom_micro-neural_..._0bec1125.jsonl, ..._9f3194be.jsonl
protocol     D27, D31
status       LIMITED
```

**크기를 주장하지 않는다.** `n=3` 이고 두 수락 규칙에서 값이 크게 달라진다. 방향만
보고한다.

### C13. 대체 수락 기준이 재계획의 겉보기 이점을 줄였다

```text
Evidence     C3 = shrinking − committed
                          control          fixed_eval
             full_batch   −0.095 (1/3)     −0.949 (0/3)
             cs128        +2.941 (3/3)     +1.110 (3/3)
             cs64         +1.666 (3/3)     +0.973 (3/3)
source       docs/results_stage2.md
protocol     D28, D31
status       LIMITED
```

**허용되는 문장.**

> We replaced minibatch-local monotonic acceptance with a fixed-evaluation-objective
> criterion. This alternative criterion reduced, but did not eliminate, the apparent
> advantage of replanning over a committed stale plan. The estimated `C3` magnitude
> decreased substantially under the alternative acceptance criterion, indicating that
> the original magnitude was partly acceptance-dependent.

**금지되는 문장.** C15 참조.

### C14. 대체 수락 기준은 완화가 아니라 엄격화였다

```text
Evidence     거절률           control   fixed_eval
             committed cs128    0.79       0.89
             committed cs64     0.66       0.92
             shrinking cs128    0.04       0.42
             shrinking cs64     0.00       0.36
             onestep   cs64     0.00       0.57
source       docs/results_stage2.md
protocol     D31
status       LIMITED
```

`control` 에서는 `loss_before` 와 `candidate_loss` 가 같은 minibatch 위에 있다. 방금
gradient 를 계산한 batch 의 loss 를 줄이는 것은 쉽다. `fixed_eval` 은 전체 데이터
loss 의 감소를 요구한다.

**이 ablation 은 단일 요인 변경이 아니다.** 평가 forward 가
`n_samples / batch_size` 배 비싸므로 같은 예산의 step 수도 줄어든다. 비용을 회계에서
빼면 숨기는 것이 되므로 넣었고, 그 결과 두 효과가 섞였다.

---

## EXPLORATORY

### C16. Planner 모델의 정확도가 planning 의 가치를 좌우하는 것으로 보인다

```text
Evidence     micro-neural, regime 당 n=3, control 수락 규칙
                                  full_batch    cs128     cs64
             A2 shrinking−static    +15.176     −0.900   −0.277
             C2 shrinking−onestep    +0.547     −1.111   −0.716
             C3 shrinking−committed  −0.095     +2.941   +1.666
source       docs/results_stage2.md
protocol     D27, D29
status       EXPLORATORY
```

**주의 세 가지.**

```text
regime 당 n=3. CI 와 p-value 를 인용하지 않는다
batch_size 3 점의 전이가 단조가 아니다. cs128 이 cs64 보다 나쁘다
regime 간 절대값 비교는 FLOP 정규화가 아니다 (C17)
```

`full_batch` 에서 `C2 = +0.547` 이고 `fixed_eval` 에서 `−0.080` 인데, held-out
quadratic 의 `C2 = +0.456` (`n=40`, `p<0.0001`) 이 강한 쪽이다. **불일치 자체를
결과로 보고한다** (D9).

### C17. GE 는 regime 내부에서만 compute-matched 다

```text
Evidence     D1 정의  1 GE = gradient batch 1회 forward + backward
             full_batch (n=512)  1 GE = 512 샘플 gradient
             cs64                1 GE =  64 샘플 gradient
             150 GE 에서 full_batch 는 cs64 의 약 8배 FLOP
source       docs/experiment_protocol.md D30
status       EXPLORATORY (한계 공개 항목)
```

**허용되는 문장.**

> GE normalizes gradient-equivalent oracle calls within a regime, not total
> floating-point operations across different batch sizes. Cross-regime absolute
> performance comparisons are therefore descriptive rather than compute-normalized.

### C18. `κ` 만으로 헤드룸을 설명하기 어렵다

```text
Evidence     held-out n=10 per spec,  A2 = shrinking − best_static
             κ=1e3  +6.983   CI +6.956 ~ +7.101   10/10 양수
             κ=1e4  +2.317   CI +1.529 ~ +2.507   10/10
             κ=1e5  +1.399   CI +0.953 ~ +1.517   10/10
             κ=1e6  +1.253   CI +0.966 ~ +1.693   10/10
             dev n=3 에서도 같은 비단조 패턴
source       docs/results_stage2.md
protocol     D24, D26
status       EXPLORATORY
```

**허용되는 문장.**

> Adaptive headroom did not increase monotonically with condition number; the largest
> effect was observed at `κ=1e3`. This suggests that condition number alone does not
> explain the headroom.

**금지.** 단조 증가도 **단조 감소도** 주장하지 않는다 (C19).

### C19. `full_batch` 열의 두 수락 규칙 차이는 의미 변화가 아니다

```text
Evidence     full_batch 에서 acceptance_loss() == curvature_loss()
             fixed_eval 이 바꾸는 것은 step 마다 forward 1회 추가 청구뿐
             best_static full_batch: control 5.2199 -> fixed_eval 5.1094
source       docs/experiment_protocol.md D31
status       EXPLORATORY (해석 주의 항목)
```

`full_batch` 두 열을 "수락 규칙 비교" 로 읽으면 안 된다. 예산 교란이다.

---

## NOT SUPPORTED — **원고에 쓰지 않는다**

### C11. "Feedback 은 쓸모없다" / "효과가 0 임을 증명했다"

```text
이유   equivalence margin 을 사전 등록하지 않았다. CI 가 0 을 포함한다는 것은
       "큰 효과가 관측되지 않았다" 이지 "효과가 없다" 가 아니다
대체   C4 의 허용 문장
```

### C15. "`C3` 의 정확히 절반이 artifact 였다"

```text
이유   직관적 요약으로는 가능하지만 "절반" 이라는 분해가 일반적으로 성립하지 않는다
       regime 마다 감소 비율이 다르고 (cs128 62%, cs64 42%) n=3 이다
대체   C13 의 허용 문장 ("decreased substantially ... partly acceptance-dependent")
```

### C20. "PPO / 강화학습은 optimizer 제어에 실패한다"

```text
이유   PPO 를 구현하지 않았다. 우리가 측정한 것은 oracle planner 의 헤드룸 분해이며
       학습 방법의 성능이 아니다
대체   "사전 등록한 게이트 기준에서 상태 의존 feedback 정책을 학습할 실험적 근거를
       확보하지 못했으므로 PPO 를 진행하지 않았다"
```

### C21. "결정론적 환경이 stochastic 환경보다 N배 좋다"

```text
이유   GE 가 batch 크기별 FLOP 을 맞추지 않는다 (C17)
대체   C17 의 허용 문장. 절대값 비교는 descriptive 로만
```

### C22. "adaptive headroom 은 condition number 에 따라 감소한다"

```text
이유   κ 당 seed 10개로 비단조를 관측했을 뿐이다. 반대 방향의 단조 관계도 주장할 수 없다
대체   C18 의 허용 문장
```

### C23. "비선형 문제에서 방법이 실패했다"

```text
이유   rosen_d2 는 수치 하한 포화, rosen_d5 는 국소최소점 cap 과 seed 복제로 둘 다
       사용 불가 판정을 받았다. 방법의 성능을 측정한 것이 아니라 benchmark 결함이다
대체   C8. "표준 시작점에서 모든 참조 solver 와 baseline 이 동일한 국소최소점에
       도달했으므로 이 spec 은 controller 구분력을 갖지 못한다"
```

### C24. "micro-neural 결과가 quadratic 결과를 뒤집는다"

```text
이유   micro-neural 은 regime 당 n=3 exploratory 이고 held-out quadratic 은 n=40 이다
대체   C16. 불일치 자체를 결과로 보고한다 (D9)
```

### C26. 쌍별 median 을 더하거나 빼서 만든 값

```text
이유   쌍별 차이의 median 은 선형이 아니다. 실측에서
         median(committed − constant)  = +2.090
         median(replanning − constant) = +1.690
         median(replanning − committed)= +0.010
       marginal 값 차이 −0.400 과 직접 측정한 +0.010 이 다르다
       초판 초안이 onestep 의 상수 대비 값을 A2 − C2 = 1.233 으로 적었다.
       직접 측정값은 +1.155 다
대체   비교하려는 두 컨트롤러를 직접 쌍별로 측정한다.
       make_report.py 가 ladder 행으로 모든 조합을 생성한다
```

### C25. "설정을 held-out 결과로 선택했다"

```text
이유   설정은 beam 8 challenge dev (seeds 2/3/4) 에서 D21 규칙으로 한 번 선택했고
       held-out (seeds 100~109) 에서는 다시 고르지 않았다
대체   Methods 에 seed 역할 분리를 명시한다
```

---

## 프로토콜 이탈 목록 (원고 Limitations 에 반드시 포함)

```text
[E1] D21 순서 이탈
     비용 측정용 1 인스턴스 dry run (quad_d100_k1e3, seed 2, 30 run) 의 게이트 표를
     본 뒤 설정 선택 규칙을 확정했다. Q x space 별 planner 순위는 열지 않았으나
     순서가 "규칙 확정 -> 실행" 이 아니었다
     -> beam 8 결과를 configuration selection 과 가설 정교화로만 쓴다 (D24)
     source: docs/experiment_protocol.md D21, D24

[E2] (quad_d100_k1e5, seed 2) 가 original dev audit 과 challenge selection 에 중복
     selection 12 인스턴스 중 1개. beam 8 조합은 당시 미실행
     source: docs/experiment_protocol.md D20

[E3] 비선형 진단과 일부 실행 시 git 이 dirty 였다
     D13 에 따라 run_semantics_id 는 영향받지 않고 code_dirty 가 provenance 에 남는다
     source: paper/evidence_map.md

[E4] held-out 실행 중 테스트와 lint 를 병행 실행했다
     wall_clock_sec 이 일부 run 에서 부풀 수 있다. 단일 스레드 float 연산은 부하와
     무관하게 결정론적이므로 수치 결과와 판정에는 영향이 없다.
     concurrent_processes=1 기록은 이 점에서 부정확하다
     source: docs/experiment_protocol.md 변경이력 2026-08-04

[E5] execution_provenance.git_commit 이 일부 summary 에서 빈 문자열이었다
     run_headroom() 이 git_commit 을 받지 않던 버그. 수정했다. 기존 summary 는
     raw 레코드의 per-run git_commit 으로 복원했다
     source: paper/evidence_map.md, scripts/make_manifest.py

[E6] fixed_eval ablation 이 단일 요인 변경이 아니다
     수락 기준과 유효 step 수가 함께 바뀐다 (C14)

[E7] full_batch 의 fixed_eval 열은 예산 교란이다 (C19)

[E8] 게이트 C1 (fresh) 은 판정불가
     seed 1개 진단 baseline 이므로 PPO 착수 게이트에 쓰지 않는다
     source: docs/experiment_protocol.md 게이트 C1

[E9] three_layer 보고가 A1/B 게이트에는 미적용
     단일 통계를 쓴다
     source: docs/reproduce.md 알려진 한계

[E10] spec 별 분해와 all-task 통계가 한동안 다른 median 규약을 썼다
     n=10 에서 0.01~0.09 nat 규모로 갈렸다. 단일 규약(metrics.median_of)으로 통일하고
     수치를 정정했다. 설정 선택 결과는 표본이 4의 배수여서 영향이 없었다
     원고 문장:
       An earlier diagnostic script used an upper-middle convention for even sample
       counts, whereas the main paired analysis used the conventional arithmetic
       median. All reporting code was unified before manuscript generation; the
       selected configuration and qualitative conclusions were unchanged.
     source: docs/experiment_protocol.md D22 (median 규약 절)

[E11] run_semantics_id 에 이 run 이 쓰지 않는 설정이 들어 있었다
     TARGETS 에 micro-neural 항목을 추가한 것만으로 quadratic held-out 의 Track T
     240 run 이 무효화됐다. 실제로 쓰는 spec 종류의 target 만 넣도록 고쳤다.
     Track E 는 영향이 없고 게이트 D 값도 변하지 않는다 (같은 run 재집계).
     표시용 config_hash 에 acceptance_loss 를 무조건 넣은 것도 같은 유형이었다.
     source: docs/experiment_protocol.md D32

[E12] 동결 태그를 동결 시점이 아니라 Stage 2 종료 후에 만들었다
     docs/experiment_protocol.md 의 `C2 protocol freeze` 절은 config 고정 시점에
     `git tag protocol-freeze-stage2-v1` 을 남기고 이후 변경을 금지하도록 규정했다.
     태그는 존재한다. 그러나 시점과 대상이 규정과 다르다.

       17:48  e58e5cd  D24/D25  shrinking_Q4_narrow 동결   <- 규정된 태깅 시점
       19:07  0e5a182  D26      held-out confirmatory n=40
       19:45  a140ed8  D27      micro-neural
       20:04  b444b89  D28/D29  acceptance ablation
       21:47  40c84c3  D30/D31
       21:53  tag 16681b3  protocol-freeze-stage2-v1 -> 40c84c3

     즉 태그는 동결 커밋보다 4시간 5분 뒤, **confirmatory 실행 이후**에 만들어졌고
     동결 커밋이 아니라 Stage 2 종료 커밋을 가리킨다. 따라서 "held-out 실행 전에
     config 가 고정됐다" 를 태그로 증명할 수 없다. 그 순서는 날짜가 붙은 결정 기록
     D24/D25 와 동결 커밋 e58e5cd 자체가 증명한다.
     태그는 원격에 올리지 않았다. 가리키는 커밋이 장치 이름 정리 이전이기 때문이다.
     **소급해서 e58e5cd 로 태그를 옮기지 않는다.** 그 시점에 태깅했다는 이력을
     사후에 만드는 셈이기 때문이다.

     이 항목의 이전 판은 "그 태그는 만들어지지 않았다. 로컬과 원격 모두에 없다" 고
     적었다. **그것이 틀렸다.** 원인은 그 기록을 clone 환경에서 작성했고 태그가
     push 되지 않았다는 점이다. clone 에서는 `git tag -l`, `git ls-remote`, reflog 가
     모두 빈 결과를 낸다. 덧붙여 `git tag` 는 HEAD reflog 를 쓰지 않으므로
     "reflog 에 흔적이 없다" 는 애초에 태그 존재를 판정할 수 있는 검사가 아니었다.

     원고 문장:
       The protocol specified that a tag be created at the configuration-freeze
       point. The tag exists in the private repository, but it was created after the
       confirmatory runs had completed and it points at the final Stage 2 commit
       rather than at the freeze commit. It therefore cannot attest that the
       configuration was fixed before the held-out runs; that ordering is attested by
       the dated decision record and by the freeze commit itself. We did not move the
       tag retroactively. The tag was not published, because the commit it marks
       precedes the removal of device names from committed artifacts.
     source: docs/experiment_protocol.md (C2 protocol freeze 절),
             git for-each-ref refs/tags/protocol-freeze-stage2-v1
```

## 인용

**해결됐다.** `paper/references.bib` 에 26 항목이 있고 `paper/draft.md` 의
`[CITATION NEEDED]` 는 남아 있지 않다. `scripts/check_claims.py` 의 `[6]` 검사가
키 존재와 서지정보 상태를 기계적으로 확인한다.

```text
Hessian-free / truncated-Newton     dembo1982inexact steihaug1983cg nash1984lanczos
                                    nash2000survey martens2010hessianfree martens2011rnn
Levenberg-Marquardt damping         levenberg1944 marquardt1963
trust-region                        conn2000trustregion steihaug1983cg
Conjugate gradient                  hestenes1952cg
Pearlmutter HVP                     pearlmutter1994hvp
부분표본 곡률                        byrd2011stochastic
Learning to optimize                andrychowicz2016l2l metz2019pathologies
                                    metz2020effective
PPO                                 schulman2017ppo
MPC / receding horizon              rawlings2017mpc
open-loop 대 closed-loop            bertsekas2017dp
amortized optimization              amos2023amortized bae2022apo
Rosenbrock d>=4 국소최소점            shang2006rosenbrock kok2009rosenbrock
Wilcoxon signed-rank                wilcoxon1945
bootstrap CI                        efron1979bootstrap
Equivalence testing / TOST          schuirmann1987tost lakens2017equivalence
```

`Paired experimental design 의 분산 감소` 항목은 **인용하지 않기로 했다.** 우리 문장
(`§4.1` 의 "컨트롤러가 난수를 얼마나 쓰든 인스턴스가 같다")은 문헌에서 가져온 주장이
아니라 우리 실행기의 설계 사실이며, `§4.3` 의 bitwise 재현 검사가 근거다.

**서지정보 확인과 내용 일치는 다른 작업이다.** 각 인용이 실제로 그 자리의 주장을
지지하는지, 어떤 오인용 위험이 있는지는 `paper/CITATIONS.md` 에 항목별로 기록했다.
