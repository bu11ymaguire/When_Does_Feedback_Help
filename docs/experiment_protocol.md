# 실험 프로토콜

> `README.md` 는 **무엇을 만드는가**(명세)를 정의한다.
> 이 문서는 **어떻게 실행하고 무엇으로 판정하는가**(실행 계획)를 정의한다.
>
> 판정 기준과 목표치는 실험 실행 **전에** 확정한다. 결과를 본 뒤 임계값을 조정하면
> 원하는 결론이 나오므로, 변경이 필요하면 §9 변경 이력에 이유와 함께 기록한다.

---

## 1. 기준 환경

| 항목 | 값 |
|---|---|
| OS | Windows 11, PowerShell |
| GPU | NVIDIA RTX 3060 Ti, 8 GB, compute capability 8.6 (Ampere) |
| 드라이버 | 610.88 |
| Python | 3.12.12 (uv managed) |
| PyTorch | 2.13.0+cu130 (CUDA runtime 13.0) |
| Gymnasium | 1.3.0 |
| Stable-Baselines3 | 2.9.0 |
| 의존성 고정 | `uv.lock` (해시 포함, 커밋 대상) |
| venv 위치 | 저장소 **밖**에 둔다 (`UV_PROJECT_ENVIRONMENT`). 파일 동기화 서비스가 있는 디렉터리 안에 두면 설치가 손상될 수 있다 |

사전 확인 완료 사항:

- `torch.autograd.grad(..., create_graph=True)` 기반 double-backward HVP가
  CUDA에서 정상 동작. SPD quadratic(d=64) 대조 상대오차 `5.8e-8`.

8 GB VRAM은 이 프로젝트의 상한을 규정한다. Phase 1·2 모델은 모두 수십만~수백만
파라미터 규모로 제한하고, 대형 모델 실험은 범위에 넣지 않는다.

---

## 2. 확정된 설계 결정

README가 열어둔 항목 중 결과 해석에 직접 영향을 주는 것들을 여기서 고정한다.

### D1. 주 비용 지표는 grad-equivalent(GE), wall-clock은 보조 지표

**문제.** MNIST MLP 784-128-10은 약 101,770 파라미터다. 이 규모에서 GPU 시간은
FLOP이 아니라 커널 런치와 파이썬 오버헤드가 지배한다. HVP 1회의 실제 연산이
수십 µs인데 런치 비용이 수백 µs라면, wall-clock으로 측정되는 것은 최적화 효율이
아니라 구현 오버헤드다. 이 상태에서 "RL이 wall-clock을 15% 줄였다"는 주장은
방어할 수 없다.

**결정.** 1 GE = gradient batch 1회 forward+backward. 모든 비용을 GE로 환산한다.

```text
gradient (create_graph=False)  = 1.0 GE          (정의)
gradient (create_graph=True)   ≈ c_grad_graph GE
HVP 1회 (그래프 재사용)         ≈ c_hvp  GE
step acceptance forward        ≈ c_fwd  GE
Newton-CG step (k iters)       = c_grad_graph + k·c_hvp + c_fwd  GE
```

계수는 이론값(`c_hvp ≈ 2.5`)을 쓰지 않고 **대상 하드웨어에서 실측**한다
(`scripts/measure_cost_model.py`). 산출값은 `configs/cost_model.<model>.yaml` 로
저장하고 config·commit hash와 함께 기록한다.

- 주 지표: **GE** — 하드웨어 독립적, 재현 가능. 단 GE를 무엇에 쓰는지는
  트랙에 따라 다르다 (D9 참조)
- 보조 지표: wall-clock — "실제로도 이득이 남는가" 확인용, 오버헤드 지배 구간임을 명시하여 보고
- Stage 5에서 최소 하나의 task는 FLOP 지배 규모(수백만 파라미터급 CNN)로 두어
  wall-clock 결론을 별도 검증한다

**주장의 범위를 넘지 않는다.** 아래 실측에서 도출할 수 있는 결론은

> 소규모 GPU workload에서는 kernel-launch overhead와 낮은 utilization 때문에
> wall-clock이 계산량을 제대로 반영하지 않을 수 있다.

까지다. "wall-clock 기반 optimizer 논문을 신뢰할 수 없다"로 일반화하려면 다른
GPU, CPU 실험, 여러 모델 크기와 batch size, GPU utilization 또는 profiler 근거가
필요하다. 현재 근거는 단일 GPU에서의 **calibration finding** 이며 독립적인
주요 기여로 선언하지 않는다.

#### 실측 결과 (2026-08-01, RTX 3060 Ti, torch 2.13.0+cu130)

| 구성 | 파라미터 | `t_grad` | `c_grad_graph` | `c_hvp` | `c_fwd` | GE(k=10) | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| MNIST MLP 784-128-10, B=512 | 101,770 | 0.68 ms | 0.93 | 1.59 | 0.26 | 17.0 | launch-bound |
| small CNN (CIFAR 규모), B=128 | 2,193,226 | 5.64 ms | 1.00 | 3.12 | 0.37 | 32.5 | flop-bound |

D1의 가정이 실측으로 확인되었다. MNIST MLP에서 **배치를 256에서 4096으로 16배
늘려도 gradient 시간이 0.657 ms → 0.706 ms로 거의 변하지 않는다.** 연산량과
무관하게 고정 오버헤드가 시간을 지배한다는 직접적인 증거다. 같은 구성에서
`c_hvp` 측정값도 반복 간 1.30 ~ 2.53으로 흔들려 이론값(약 2~3)에서 벗어난다.

small CNN에서는 배치 2배에 시간이 1.6~2.1배로 비례하고 `c_hvp` 가 3.12로
이론 기대치에 부합한다. 즉 **이 GPU에서 FLOP 지배 구간은 도달 가능하며,
Stage 5의 wall-clock 검증은 실현 가능하다.**

**이 결정에서 따라오는 냉정한 계산.** 실측 기준 MNIST MLP에서 `k = 10` 이면
Newton-CG 1 step ≈ **17 GE**. 200 step = 3,400 GE 로, `B=512` 기준 약 29 에폭에
해당한다. AdamW는 5에폭(약 600 GE) 안에 98%에 도달한다. 따라서 **동등 예산에서
경쟁하려면 Newton-CG는 약 35 step 안에 목표에 도달해야 한다.** 이 수치를 알고 시작한다.

### D2. `B_curv / B_grad` 비율은 1급 실험 변수

D1의 비용식에서 이 비율 하나가 step 비용을 4배 좌우한다. RL 컨트롤러가 목표로 하는
이득(10~30%)보다 영향이 크다. 따라서 grid에 포함한다.

```text
B_curv / B_grad ∈ {1/4, 1/2, 1}
```

**gradient batch와 curvature batch의 동일성.** `g`와 `H`를 다른 배치에서 뽑으면
선형계 `(H_A + λI)p = -g_B` 가 불일치해져 Newton 방향이 편향된다. 기본값은
**동일 배치**(`B_c = B_g`)로 두고, 분리 방식(Martens 2010 스타일)은 ablation으로 돌린다.
CG solve 1회 내부에서 curvature batch를 바꾸지 않는다는 README 원칙은 유지한다.

### D3. 보상은 트랙마다 다르다. per-step ratio 보상은 쓰지 않는다

초판은 단일 보상을 썼다.

```text
r_t = (log L_t − log L_{t+1}) − β · (cost_t / GE_ref) − γ · I_failure      [폐기]
```

**Stage 2 파일럿에서 이 설계의 결함이 드러났다.** 같은 형태의 목적
(`Δlog L / cost`)으로 매 step 최선을 고르는 컨트롤러가 고정 설정보다
cost-to-target에서 **나빴다** (비율 0.967x). 국소 효율 최대화가 총비용 최소화와
다른 문제이기 때문이다.

```text
행동 A:  3 GE 로 loss 10% 감소     → 순간 효율 높음
행동 B: 20 GE 로 loss 60% 감소     → 목표까지 총비용은 더 적을 수 있음
```

per-step ratio 보상을 쓰면 정책이 `k=3` 같은 싸고 작은 행동만 반복할 유인이
생긴다. 그래서 보상을 트랙별로 분리한다 (D9).

#### Track E 보상 (고정 GE 예산)

```text
r_t = (log L_t − log L_{t+1}) − γ · I_failure
에피소드 종료: 누적 GE ≥ B
```

리턴이 텔레스코핑되어 `log(L_0 / L_B) − γ·(총 실패)` 가 된다. **트랙 E의 목적과
정확히 일치한다.** 행동별 비용 차이는 보상을 나누는 대신 **남은 예산에서
차감**하는 방식으로 반영한다. 비용이 큰 행동은 예산을 더 많이 먹으므로 자연히
에피소드가 짧아진다.

#### Track T 보상 (목표 도달 총비용)

```text
r_t = −c_t / GE_ref − γ · I_failure
에피소드 종료: L ≤ τ (도달) 또는 누적 GE ≥ B (절단)
```

stochastic shortest-path 형태다. 리턴이 `−(총 GE)/GE_ref` 가 되어 트랙 T의
목적과 정확히 일치한다. 미도달 절단에 큰 임의 벌점을 주지 않고 D6의 절단 규칙을
유지한다. 학습이 불안정하면 potential-based shaping을 더한다.

```text
r'_t = r_t + γ_disc · Φ(s_{t+1}) − Φ(s_t),   Φ(s) = −max(0, log L − log τ)
```

potential-based shaping은 최적 정책을 바꾸지 않는다는 것이 알려져 있으므로
목적을 훼손하지 않는다.

#### 공통 사항

- `γ = 1.0` (실패 패널티), per-step reward clip `[-5, 5]`
- `log L` 이 정의되지 않는 경우를 막기 위해 quadratic 계열은 `L* = 0` 으로
  구성하고 `log(max(L, 1e-30))` 를 쓴다
- indefinite quadratic은 아래로 유계가 아니므로 두 트랙 모두에서 제외한다
- README의 상대 감소량 형태와 폐기된 ratio 형태는 ablation으로 보존한다

### D4. baseline에 open-loop schedule과 best-of-36 static을 추가

README의 4종(AdamW, SGD+momentum, Fixed, Heuristic)만으로는 결론을 낼 수 없다.

| baseline | 정의 | 왜 필요한가 |
|---|---|---|
| `best_static` | 36개 action 조합을 각각 고정해 전부 실행, meta-train 성능 최고를 선택 | "동일 탐색 예산" 원칙의 정직한 구현 |
| `open_loop` | `progress`(step/총 step)만 입력으로 받는 학습된 스케줄, 랜덤 서치 36회 | **RL이 "적응 제어"인지 "튜닝된 스케줄"인지 구분** |

`open_loop`이 없으면, RL이 Fixed를 이겼을 때 그것이 상태 기반 적응 때문인지
단순히 시간에 따른 스케줄 때문인지 알 수 없다. 이는 논문의 주장 자체가 달라지는
문제이므로 ablation이 아니라 독립 baseline으로 세운다.

전체 비교군:

```text
adamw · sgd_momentum · fixed_newton_cg · heuristic_newton_cg
best_static · open_loop · rl_newton_cg
```

### D5. 탐색 예산을 명시적으로 회계 처리

결과 표에 "optimizer를 얻는 데 든 비용" 열을 넣는다. RL은 meta-training 비용이
있고 Fixed는 없다. 이를 숨기면 learned-optimizer 문헌의 흔한 함정에 빠진다.

| Optimizer | 튜닝/학습 비용 (GE) | 튜닝 run 수 | Cost-to-target (GE) | Wall-clock (s) | Final Acc | Failure Rate |
|---|---:|---:|---:|---:|---:|---:|

**탐색 예산은 모든 컨트롤러에 동일해야 한다.** `best_static` 을 200개 설정에서
찾고 `open_loop` 은 50개만 평가하면 static 쪽에 유리하다. 반대도 마찬가지다.
파일럿에서 이 문제가 실제로 발생했다. static 12개 조합 전수 탐색 대 open_loop
랜덤 서치 12회였는데, open_loop 우승자가 static과 **완전히 동일**한 결과를 냈다
(비율 1.000x, CI 1.000–1.000). 12회로는 스케줄 공간을 사실상 탐색하지 못한다.

```text
탐색 예산 N_tune = 각 컨트롤러가 평가받는 설정 후보 수. 모두 같게 맞춘다.
  best_static   행동 공간 전수 (부족하면 N_tune 까지 반복 없이 확장)
  open_loop     스케줄 파라미터 랜덤 서치 N_tune 회
  heuristic     rho_low / rho_high / 배수 랜덤 서치 N_tune 회
  adamw / sgd   learning rate + weight decay 랜덤 서치 N_tune 회
  rl            PPO 하이퍼파라미터 탐색 횟수를 기록하고 meta-training GE 합산
```

행동 공간이 `N_tune` 보다 작으면 `N_tune` 을 행동 공간 크기로 내리거나, static에
초기 damping 축을 추가해 후보를 늘린다. **어느 쪽이든 실제 사용한 횟수를
결과 표에 기록한다.**

**선택은 dev task/seed 에서만** 하고, 선택된 설정을 held-out task/seed 에 그대로
적용한다 (D6의 pilot / confirmatory 구분과 동일한 분할을 쓴다).

planner의 분석 비용도 별도 열로 기록한다. one-step efficiency controller는 step당
행동 공간 전수 sweep, H-step MPC planner는 그 위에 beam 확장 비용이 든다. 이
비용은 배포 비용(deployment GE)과 합치지 않지만 반드시 보고한다.

### D6. 목표치 다단계 사전 등록, pilot / confirmatory 분리, 절단 규칙

#### target은 난이도별로 여러 개 둔다

target 하나만 잡으면 그 값 선정에 따라 결론이 흔들린다. Track T는 난이도
3단계로 본다.

```text
easy    L / L_0 ≤ 1e-2
medium  L / L_0 ≤ 1e-4
hard    L / L_0 ≤ 1e-6
```

Rosenbrock은 `L* = 0` 이므로 절대값으로 `{1e-1, 1e-2, 1e-4}` 를 쓴다.
신경망 task는 Stage 3에서 확정한다.

#### pilot과 confirmatory를 분리한다

예산과 target을 결과를 본 뒤에 고치면 사후적으로 유리한 프로토콜을 고른 것이
된다. 그래서 두 국면으로 나눈다.

| 국면 | task / seed | 용도 |
|---|---|---|
| **pilot** | dev seed `{0, 1, 2}`, 초기 condition number 집합 | GE 예산과 target 난이도 **선정**. 프로토콜 결정에만 사용 |
| **confirmatory** | held-out seed `{100..109}`, 새 condition number와 초기점 | 최종 결론. 선정된 예산/target을 그대로 적용 |

실행 순서는 세 국면으로 나눈다.

```text
C1  pilot            budget / target / timeout / beam / horizon / N_tune 결정
C2  protocol freeze  config 고정 + 태그. 이후 변경 금지
C3  confirmatory     held-out seed 에서 게이트 판정
```

**C1 pilot 절차:**

1. 여러 GE 예산을 시험한다
2. 방법 대부분이 너무 쉽게 성공하지도, 전부 실패하지도 않는 예산을 고른다
   (도달률이 20~80% 구간에 오도록)
3. easy / medium / hard target을 pilot 분포를 보고 확정한다
4. beam width를 위 calibration 규칙으로 확정한다
5. pilot 결과는 **최종 효과 크기 계산에 섞지 않는다**

**C2 protocol freeze:**

config를 고정하고 태그를 남긴다.

```text
git tag protocol-freeze-stage2-v1
```

함께 저장할 것: 최종 config 파일, pilot 결과 요약, 각 파라미터를 그 값으로
선택한 이유, 이후 변경 금지 항목, 예외적으로 변경 가능한 오류 조건.

**confirmatory 중 버그를 발견하면 조용히 고치고 계속하지 않는다.**

```text
중단 → 버그 범위 기록 → 영향받은 결과 폐기 → 버전 증가(v2) → 전체 재실행
```

**C3 confirmatory:** held-out seed `{100..109}` 에서 게이트 A1·A2·B·C·D를 평가한다.
Track E와 Track T를 분리해 보고한다.

| 트랙 | 보고 항목 |
|---|---|
| Track E | 고정 GE에서 terminal log-loss improvement, paired difference, CI, 행동공간별 헤드룸 |
| Track T | target별 도달률, cost-to-target, restricted mean, 절단 run 수, success-conditioned cost와 전체 성과를 구분 |

**2026-08-01 시점의 파일럿 결과는 예산 300 GE, seed {0,1}에서 도달률 67% 였다.
이 결과는 pilot으로만 분류하며 어떤 결론에도 쓰지 않는다.**

#### 절단(censoring) 규칙

예산 내 미도달 run은 삭제하거나 최댓값으로 대입하지 않는다.

- `success_rate` = 도달한 run 비율 (별도 보고)
- `cost_to_target` = **도달한 run만의 중앙값** (평균이 아님)
- `restricted_mean` = 미도달을 예산값으로 절단한 제한 평균 (보조 지표)
- 위 세 지표를 항상 함께 보고한다. 하나만 보면 왜곡된다.
- 실패 run도 `results/raw/` 에 보존하고 실패 원인 태그(`nan`, `budget_exhausted`,
  `divergence`, `cg_breakdown`, `oom`)를 기록한다.

파일럿에서 이 규칙이 실제로 작동했다. heuristic은 cost-to-target 중앙값이
best_static보다 68% 나빴지만 도달률은 더 높았다(75% vs 67%). 중앙값만 봤다면
"느리지만 더 자주 도달한다"는 다른 성격을 놓쳤을 것이다.

### D7. Paired design과 통계 프로토콜

**Paired design.** seed는 난수 시드가 아니라 **실험 조건 식별자**로 쓴다.
`seed=s` 이면 모든 optimizer가 동일한:

- task 인스턴스 (quadratic의 `A`, 모델 초기화)
- minibatch 순서
- train/val split

을 본다. 구현은 `benchmark/paired.py` 에서 `seed → (task_instance, batch_order)` 를
결정론적으로 매핑한다. 비용이 들지 않는 순수 분산 감소이므로 반드시 적용한다.

**통계.** `mean ± std` 는 보고하지 않는다(n=5에서 무의미하고 정규성 가정도 없다).

- 쌍별 비교: `(task, seed)` 쌍에 대한 **Wilcoxon signed-rank test**
- 효과 크기: `cost_to_target` **비율의 기하평균**과 부트스트랩 95% CI (`n_boot = 10000`)
- seed 수: 최소 5, 주장 근거가 되는 비교는 10
- 다중 비교: 주 가설(RL vs fixed, RL vs heuristic, RL vs open_loop) 3개에 대해
  Holm 보정

### D9. 실험을 두 트랙으로 분리한다

Stage 2 파일럿에서 드러난 것은 지표 불일치가 아니라 **서로 다른 두 최적화 문제를
한 실험에 섞고 있었다**는 사실이다. 분리한다.

#### Track E — 고정 예산에서 얼마나 개선하는가

> 동일한 GE 예산 `B` 를 받았을 때 어떤 컨트롤러가 loss를 가장 많이 낮추는가?

```text
목적:  max  log(L_0 / L_B)      s.t.  Σ c_t ≤ B
지표:  J_E = log L_0 − log L_B  (B가 모두 같으므로 사실상 최종 loss 비교)
```

#### Track T — 목표까지 얼마나 싸게 도달하는가

> 사전 지정한 target loss `τ` 에 도달하는 데 필요한 총 GE는 얼마인가?

```text
목적:  min  Σ_{t≤T_τ} c_t       s.t.  L ≤ τ
지표:  J_T = GE-to-target,  도달률,  제한 예산 내 restricted mean,  절단 run 수
```

#### 두 트랙은 같은 답을 주지 않는다

파일럿이 이미 보여줬다. 국소 효율 컨트롤러는 고정 예산에서는 쓸 만하지만
cost-to-target에서는 고정 설정보다 나빴다. 다음도 충분히 가능하다.

```text
고정 예산에서는 adaptive 가 좋다
하지만 특정 target 까지는 best_static 이 더 싸다
```

**이 불일치 자체가 연구 결과다.** 그래서 둘 다 보고한다.

#### 헤드룸도 트랙별로 정의한다

```text
H_E = J_E(planner) − J_E(best_static)              [nat, 클수록 여지 큼]
H_T = C_τ(best_static) / C_τ(planner)              [배수, 클수록 여지 큼]
```

하나의 "헤드룸"으로 묶으면 같은 혼동이 재발한다.

#### 상한이라고 부르지 않는다

이름과 해석을 정정한다.

| 초판 이름 | 정정된 이름 | 이유 |
|---|---|---|
| `greedy_oracle` | **one-step efficiency controller** | 전역 상한이 아니다. 매 step 즉시 효율이 가장 좋은 후보를 고르는 컨트롤러일 뿐이며, 실제로 고정 설정보다 나쁠 수 있음이 확인됐다 |
| `lookahead_oracle` | **H-step MPC planner** | 유한 horizon과 beam 폭에 제한된 근사다. 전역 최적해가 아니다 |

문서와 표에서 `oracle` 이라는 단어는 도달성 제약이 없는 `absolute` 행동 공간을
쓰는 planner에 한해서만, 그리고 "one-step" / "H-step" 을 함께 붙여서 쓴다.

### D8. Truncated horizon의 근시안 편향 대응

에피소드를 50 step에서 끊으면 정책이 "지금 loss를 최대한 줄이는" 행동을 학습한다.
그런데 우리가 원하는 것은 장기 수렴 효율이다. 목적 불일치가 발생한다.

대응:

- horizon 랜덤화: `H ~ Uniform{30, 50, 80, 120}`
- SB3에서 `TimeLimit.truncated` 시 value bootstrapping이 켜지도록 환경을 구성
  (termination과 truncation을 구분해 반환)
- `gamma = 0.995`, `gae_lambda = 0.95`
- 상태에 `progress` 를 포함하되, 이것이 open_loop baseline과 겹치므로
  `progress` 제거 ablation을 반드시 수행

### D10. Planner 목적함수를 비율에서 고정 GE 쿼터로 교체한다

Track E planner의 초기 목적함수는 다음이었다.

```text
U = (log L_start − log L_terminal) / cumulative_cost
```

dry run에서 이것이 **장기 계획을 구조적으로 검출하지 못한다**는 것이 드러났다.

#### 무엇이 관측됐는가

`quadratic`, seed 0, beam 3, 150 GE 예산:

```text
SPD κ=1e2    H=1/3/5 전부 logΔ=59.8636, depth 히스토그램 {1: 8}
ill κ=1e5    H=1 → 10.4998 {1:10} / H=3 → 10.5116 {1:9,2:1} / H=5 → 동일
wide κ=1e5   H=1 → 10.3315 {1:9}  / H=5 → 10.3564 {1:7,2:2}
```

- `H=3` 과 `H=5` 가 모든 조건에서 완전히 동일했다
- `depth ≥ 3` 은 `H=5` 에서도 채택 0회
- 게이트 C 효과 크기 0.025 nat (GO 0.3, 재설계 0.05)
- `H=5` 의 search 비용은 본문의 약 100배 (16,848 GE vs 166 GE)

#### 왜 그런가

`U` 는 step별 rate의 **비용 가중 평균**이다. mediant 부등식에 의해

```text
min(r₁, r₂) ≤ (g₁+g₂)/(c₁+c₂) ≤ max(r₁, r₂)
```

depth 1에서 이미 최대 rate `R*` 를 골랐으면, depth 2가 이기려면 `r₂ > R*`,
즉 두 번째 step이 **지금 당장 가능한 모든 행동보다** 효율적이어야 한다.
수익 체감이 일반적인 환경에서는 드물다. mediant 부등식 때문에 깊은 계획이
수학적으로 절대 불가능한 것은 아니고, 두 번째 상태에서 더 효율적인 행동이 열리면
이길 수 있다. 실제로 ill-conditioned 문제에서 depth 2가 간헐적으로 채택됐다.
그러나 **장기 투자 행동을 검출하는 목적함수로는 부적합**하다.

#### 결정

이 결과를 "lookahead가 불필요하다"는 근거로 **쓰지 않는다.** 증명되는 것은
다음뿐이다.

> 누적 평균 효율을 최대화하는 목적에서는 짧은 계획이 유리하다.

Track E의 실제 연구 질문은 고정 예산 문제다.

```text
max  log L_t − log L_{t+m}     s.t.  Σ_{i=t}^{t+m−1} c_i ≤ Q
```

여기에 비용으로 나누는 비율은 들어가지 않는다. 따라서 게이트 C의 주 컨트롤러를
`BudgetedMPCController` 로 교체한다. 후보마다 동일한 미래 GE 쿼터 `Q` 를 주고
그 안에서 도달한 terminal loss를 비교한다.

#### 기존 planner는 버리지 않고 이름을 바꿔 보존한다

```text
기존:  HorizonPlannerController   (게이트 C 주 컨트롤러로 오해될 이름)
수정:  AverageRateEfficiencyPlanner  (진단 baseline)
```

버그가 아니었다. 푸는 문제가 달랐을 뿐이다. 이 발견은 별도 결과로 보고한다.

> `Δlog L / GE` 의 누적 평균을 최적화하면 planner가 거의 항상 depth 1을
> 선택했으며, 이는 cost-to-target이나 fixed-budget terminal performance를
> 개선하지 못했다.

이는 **RL 보상을 ratio로 설계할 때 생기는 실제 함정**을 보여준다. D3의 보상
설계 근거를 강화한다.

#### Beam pruning도 비율을 쓰지 않는다

후보를 `Δlog L / c` 스칼라 하나로 정렬하면 같은 문제가 가지치기 안에서 재발한다.
비싼 장기 계획이 싼 단기 계획과 섞여 조기에 탈락한다. 대신 `(used_GE,
terminal_loss)` 의 **Pareto frontier** 를 유지한다.

```text
A 가 B 보다 GE 를 같거나 적게 쓰고 terminal loss 도 같거나 낮으면 B 를 제거
```

Pareto frontier는 크기 상한이 없으므로 계산량 제한을 위해 GE cost bucket을
함께 쓴다. 구간마다 terminal loss가 좋은 후보를 `beam_width` 개 남긴다.

Pareto는 **terminal loss 최소 후보를 절대 지우지 않으므로** incumbent
carry-over를 대체한다. depth 1 최선은 더 나은 계획에 의해서만 밀려난다.

#### Track T planner는 lexicographic

임의의 큰 실패 벌점이나 비율을 넣지 않는다.

```text
1. target 에 도달한 sequence 가 있으면 누적 GE 가 가장 작은 것
2. 아무도 도달하지 못하면 같은 쿼터에서 terminal loss 가 가장 낮은 것
3. 동률이면 더 적은 GE, 그다음 더 짧은 sequence
```

### D11. Track E는 예산을 넘지 않는 prefix에서 평가한다

optimizer 루프는 `spent >= budget` 에서 종료한다. 즉 **마지막 step이 예산을
초과한다.** 초과량은 컨트롤러가 고른 action 크기에 비례하므로, 고정 예산
비교에서 큰 step을 고르는 컨트롤러가 공짜로 이득을 본다.

```text
C0  (평균 k=17.9)   150 GE 예산에 실제 171 GE 소모     <- 큰 step 하나가 공짜
Q=4 (평균 k=3.3)    150 GE 예산에 실제 154 GE 소모
```

약 11% 예산 차이다. 그런데 게이트 C의 쿼터 사다리는 정확히 "쿼터를 키우면
planner가 싼 action을 고른다"는 현상을 다루므로, 이 편향이 결론과 **같은 방향**
으로 섞인다. 즉 편향을 제거하지 않으면 "planning이 나쁘다"는 결론의 일부가
회계 인공물이 된다.

따라서 집계 시 **누적비용이 예산을 넘지 않는 마지막 prefix** 에서 평가한다
(`budget_respecting_prefix`). optimizer의 동역학은 바꾸지 않고, 절단된 step은
raw trace에 남는다. 이렇게 하면 모든 컨트롤러의 `total_cost_ge ≤ budget` 이므로
planner의 쿼터 회계와 의미가 일치한다.

Track T 지표(`cost_to_target_ge`, `reached`)는 목표 도달 시점으로 정의되므로
영향받지 않는다. Track E를 공정하게 만드는 수정이 Track T의 정의를 바꾸면 안 된다.

동일한 버그가 진단 스크립트에도 있었다. `cost_budget_ge=Q` 로 돌린 one-step
참조가 Q=30에서 실제 49.9 GE를 썼고(1.66배), 그 상태로 beam search를 비교해
"탐색 손실"이라고 잘못 판정했다. 동일 비용으로 고치니 18개 조건 중 17개가
`B ≤ A` 였다. **고정 예산 비교에서는 "예산"과 "실제 소모량"을 항상 함께
확인한다.**

### D20. Challenge set을 측정 가능성 기준으로 사전 등록한다

D19 에서 dev subset 3 spec 중 2개가 컨트롤러를 구분하지 못한다는 것이 확인됐다.
**측정 가능한 regime 이 `quad_ill κ=1e5` 하나뿐이다.** 기존 9쌍만으로 beam 8 을
돌려 설정을 freeze 하면 seed 하나에 좌우된다.

#### 기존 dev set은 보존한다

쉬운 spec 을 몰래 교체하거나 제거하지 않는다. 결과를 본 뒤 benchmark 를 바꿨다는
문제가 생긴다.

```text
Original dev audit    rosen_d2 / quad_spd κ=1e2 / quad_ill κ=1e5   그대로 유지
New challenge set     별도로 사전 등록. 아래 규칙으로 선정
```

#### 선정 기준은 성능 우열이 아니라 **측정 가능성**이다

**planner 결과를 보지 않는다.** 비적응 baseline 패널만으로 판정한다.

```text
사용 baseline:  best_static / best resource-clock open_loop / heuristic / C0
비공개:         shrinking / committed / fresh / beam 결과
```

##### 채택 조건 (사전 고정)

```text
failure_rate = 0                             numerical failure 없음
joint floor-hit rate <= 1/3                  포화가 과도하지 않음
각 baseline median logΔ >= 1 nat             문제를 전혀 못 줄이는 조건 아님
median distance-to-ceiling >= 3 nat          floor 까지 e^3 ~ 20배 여유
```

여기서 ceiling 은 `log(L0 / loss_floor)` 이고 `RELATIVE_LOSS_FLOOR` 기준 약
31.4 nat 이다. `0.8 x ceiling` 같은 비율 기준은 현재 open-loop 가 25.456 이므로
지나치게 빡빡하다. **절대 여유 3 nat** 으로 정한다.

##### 최대 개수와 tie-break

```text
사전 정의한 측정 가능성 조건을 통과한 모든 spec 을 채택한다.
최대 4개를 초과하면 log10(κ) 간격을 가장 고르게 덮는 spec 을 선택한다.
planner 성능은 선택에 사용하지 않는다.
```

"3개를 골라야 한다" 로 미리 정하면 결과를 보고 유리한 셋을 고를 위험이 있다.

#### 후보군

conditioning 이 핵심 축이므로 quadratic 을 촘촘히 만든다. 기존 `κ=1e2` 와
`κ=1e5` 사이에 중간 구간이 없어 **임계점이 어디인지 모른다.**

```text
quad d=100 κ=1e3       중간 구간. 특히 중요
quad d=100 κ=1e4       중간 구간. 특히 중요
quad d=100 κ=1e5       기존 anchor
quad d=100 κ=1e6
Rosenbrock d=5         비선형. d=2 는 너무 쉬움
```

Rosenbrock 은 quadratic 과 수치 특성이 다르므로 calibration 에서 floor 도달 속도,
Newton-CG fallback 지배 여부, 발산/정체를 먼저 확인한다.

#### seed 역할을 분리한다

challenge spec 을 고르는 seed 와 beam 8 을 평가하는 seed 가 같으면 benchmark
tuning 이 된다.

```text
CALIBRATION_SEEDS   0, 1          challenge spec 선정에만
SELECTION_SEEDS     2, 3, 4       설정(Q, space, beam) 선택에만
HELD_OUT_SEEDS      100 ~ 109     선택된 단일 설정 평가에만
```

held-out 은 권고 범위가 5~14 였으나 **기존에 100~109 로 고정해 둔 것을 유지한다.**
분리 조건(calibration/selection 과 서로소)을 이미 만족하므로 바꿀 이유가 없고,
바꾸면 기존 confirmatory 정의가 흔들린다.

##### 남은 중복 하나를 명시한다

`quad_d100_k1e5` 는 **original dev audit 과 challenge set 에 모두 있다.** 따라서
`(quad_d100_k1e5, seed 2)` 는 두 국면에서 같은 인스턴스다.

```text
original dev audit   quad_ill k=1e5, seeds 0/1/2, beam 4     이미 실행. planner 결과 공개됨
challenge selection  quad_ill k=1e5, seeds 2/3/4, beam 8     예정
```

이것을 숨기지 않고 기록한다. 완전 분리를 원하면 `SELECTION_SEEDS` 를 `(3, 4, 5)`
로 바꿔야 하지만, 그러면 사전 등록한 권고 범위에서 벗어난다. 현재는 다음 근거로
`(2, 3, 4)` 를 유지한다.

```text
설정 선택은 beam 8 결과로 하고, 그 조합은 아직 실행되지 않았다
겹치는 것은 3개 인스턴스 중 1개, 12개 selection 인스턴스 중 1개다
spec 선정은 planner 를 보지 않는 calibration seed 0/1 로만 했다
```

#### 전체 구조

```text
Original dev audit        기존 3 specs x 3 seeds. 포화 현상 보고
Challenge calibration     baseline-only. spec 선정
Beam 8 dev                challenge specs 에서 Q / space / beam 선택
Held-out confirmation     선택된 단일 설정을 새 seed 에서 평가
```

기존 결과를 버리지 않으면서도 너무 쉬운 benchmark 만으로 beam 을 고르는 문제를
피한다.

#### Calibration 결과와 freeze (2026-08-03, seeds 0/1, 150 GE, N_tune 6)

**후보 5개가 사전 조건을 전부 통과했다.** floor hit 0, numerical failure 0.

```text
spec              L0       median logΔ (static/open_loop/heuristic/C0)   최소 ceiling 여유
quad_d100_k1e3    9.7e3    13.23 / 14.66 / 13.23 / 17.72                 13.71
quad_d100_k1e4    3.7e4     8.33 /  8.80 /  8.33 / 10.46                 20.98
quad_d100_k1e5    4.4e5     8.71 /  9.05 /  8.71 /  9.56                 21.88
quad_d100_k1e6    2.6e6     8.69 /  9.51 /  8.71 /  9.44                 21.93
rosen_d5          2.4e1      1.82 /  1.82 /  1.82 /  1.82                 29.62
```

5개 > `MAX_SPECS = 4` 이므로 사전 등록된 tie-break("`log10(κ)` 간격을 가장 고르게
덮는 spec")를 적용한다.

##### Challenge selection set (freeze)

```text
quad_d100_k1e3
quad_d100_k1e4
quad_d100_k1e5
quad_d100_k1e6
```

선정 근거: baseline-only 측정 가능성 조건 전부 통과, numerical failure 0,
floor hit 0, `log10(κ)` 축을 간격 1로 균등 포괄. **planner 결과는 보지 않았다.**

##### Nonlinear diagnostic (선택에 사용하지 않음)

```text
rosen_d5
```

`rosen_d5` 는 `κ` 축에 없어 tie-break 규칙상 탈락했다. 관측 사실을 함께 기록한다.

> 네 baseline 이 모두 정확히 `1.8175` nat 을 기록했다. baseline panel 로는
> controller 구분력을 확인하지 못했다.

**사후에 `baseline spread` 임계값을 추가하지 않는다.** 임계값 선택이 사후적이
되고, baseline 4종이 같다고 planner 도 같다는 보장이 없다. 두 해석이 모두 가능하다.

```text
문제가 controller 에 둔감해서 실제로 구분력이 없음
단순 baseline 은 같은 행동으로 수렴하지만 다단계 planner 는 다른 궤적을 찾을 수 있음
```

따라서 "무가치한 task" 가 아니라 **baseline panel 에서는 구분력이 관측되지 않은
비선형 진단 문제**로 보존한다. baseline spread 는 향후 benchmark 설계의 참고
지표로만 기록하고 이번 eligibility 에 소급 적용하지 않는다.

##### `rosen_d5` 사용 규칙

**가장 엄격한 순서를 쓴다.** quadratic 4개에서 설정 하나를 freeze 한 뒤 그 설정만
`rosen_d5` 에 적용한다. 그러면 Rosenbrock 결과를 보고 설정을 조정했다는 문제가
사라진다.

계산 여유로 모든 `Q × space` 조합을 돌리더라도 다음을 지킨다.

> Rosenbrock 결과는 configuration selection 에 사용하지 않고 비선형 행동 분석에만
> 사용한다.

### D28. `fixed_eval` 수락 규칙 ablation. R2 의 교란을 닫는다

D27 의 R2 결과에는 두 효과가 섞여 있다.

```text
[1] minibatch 가 바뀌어 committed plan 이 낡아지는 현상
[2] noisy loss 에서 엄격한 단조 감소를 요구해 step 이 거절되는 현상
```

`newton_cg.py` 의 `_accept` 주석이 이미 [2] 를 경고했다. 따라서 R2 의
`C3 = +1.666` 크기를 그대로 주장할 수 없다.

#### ablation 설계 (범위를 하나로 제한한다)

**optimizer 를 재설계하거나 기존 실험을 무효화하지 않는다.**

```text
gradient 와 HVP    계속 minibatch 에서 계산
accept / reject    step 마다 바뀌지 않는 목적함수로 판정
평가 forward 비용   object-level GE 회계에 포함
설정              이미 freeze 한 Q4 narrow 그대로
비교 대상          static / C0 / committed / shrinking
정체성            새 semantics 키 아래 별도 결과로 보존
```

고정 평가 목적함수는 **전체 데이터**를 쓴다. `batch_size` 크기의 고정 부분집합보다
참 목적함수에 가깝고, 이 ablation 의 목적이 표본 잡음 제거이므로 그쪽이 맞다.

#### 비용을 숨기지 않는다

전체 데이터 forward 는 minibatch forward 보다 `n_samples / batch_size` 배 비싸다.

```text
acceptance_forward_units = n_samples / batch_size     (full_batch 면 1.0)
```

이 배수가 GE 회계에 들어간다. `step_cost_ge(hvp, forward)` 의 `forward` 를 실수로
바꿨다. **`StepRecord.forward_count` 는 호출 횟수(정수)로 유지**하고 비용 단위와
분리했다. 기존 기록이 그대로 보존된다.

#### 기본값은 해시를 바꾸지 않는다

새 옵션을 무조건 `_core_payload` 에 넣으면 기존 run 전체의 `run_semantics_id` 가
바뀌어 재실행된다. 기본값은 이전과 같은 의미이므로 해시도 같아야 한다.

```text
acceptance_loss == "control"     키를 넣지 않는다.  해시 불변
acceptance_loss == "fixed_eval"  키를 넣는다.      해시 변경
```

`OPTIMIZER_SEMANTICS_VERSION` 을 올리지 않는다. 결정론적 task 는
`acceptance_loss()` 를 제공하지 않으므로 조용히 `control` 로 되돌아가고 결과가
bitwise 동일하다. `tests/test_control_vs_eval_loss.py::TestFixedEvalAcceptance` 가
검증한다.

#### 결과 해석 시나리오 (사전 등록)

```text
C3 가 여전히 크지만 shrinking 이 C0 보다 나쁨
  -> stale-plan 방지일 뿐이다. RL 근거 없음

C3 가 작아짐
  -> 기존 feedback 효과 대부분이 acceptance artifact 였다

shrinking 이 C0 까지 이김
  -> feedback 연구를 재검토할 근거가 생긴다
```

현재 증거상 마지막 경우의 가능성은 높지 않다. 어느 쪽이든 **결과를 본 뒤 해석을
만들지 않기 위해** 여기에 미리 적는다.

### D32. `run_semantics_id` 에 **이 run 이 쓰지 않는 설정**이 들어 있었다

원고 작업 중 발견했다. D13 이 막으려던 실패가 두 곳에 남아 있었다.

#### 증상

`TARGETS` 에 micro-neural 항목을 추가했더니 **quadratic held-out 의 Track T 240 run
전부가 무효화**되어 재실행됐다. Track E 는 캐시가 유지됐다.

```text
git_commit 9679fe33   960 run   원본
git_commit b713c1d9   145 run   best_static@easy/medium, shrinking_Q4@easy/medium
                                -> 같은 논리적 run 이 두 experiment_id 로 존재
```

#### 원인

`run_semantics_payload(uses_target=True)` 가 `self.targets` **전체**를 넣었다.

```python
payload["targets"] = {kind: {...} for kind, levels in self.targets.items()}
```

`micro_neural` 키를 추가한 것만으로 `ill_conditioned` quadratic run 의 해시가 바뀐다.
**이 run 은 그 target 을 쓰지 않는다.**

#### 수정

이 실행이 실제로 쓰는 spec 종류의 target 만 넣는다.

```python
kinds = {spec_kind_label(spec) for spec in self.specs}
payload["targets"] = {k: v for k, v in self.targets.items() if k in kinds}
```

개별 target 문자열은 `RunKey.target` 에 이미 들어 있다. 여기서 고정할 것은 "이 spec
종류에 어떤 난이도 사다리를 썼는가" 뿐이다.

#### 세 번째 사례: `aggregation_payload`

같은 문제가 집계 정체성에도 있었다. `aggregation_id` 가 `ec5019fa2cc67699` 에서
`1fb5800f9248290c` 로 바뀐 원인이다. run 을 무효화하지는 않지만 **보고 라벨이 이유
없이 달라진다.** 같은 필터를 적용했다.

#### 두 번째 사례: 표시용 `config_hash`

`meta["acceptance_loss"]` 를 무조건 넣었더니 기본 설정 실험의 **raw 파일 경로**가
바뀌어 held-out 960 run 이 캐시에서 빠졌다. `meta` 는 정체성이 아니라 파일 이름을
정하는 표시용 해시인데, 그것이 사실상 재실행 여부를 결정한다.

기본값이면 넣지 않도록 고쳤다. 정체성은 `run_semantics_id` 가 담당한다 (D28).

#### 처리

```text
재실행분 145 run 을 제거해 원본 960 집합으로 되돌렸다
수정된 정체성으로 Track T 240 run 을 한 번 재실행했다
Track E 는 영향이 없다 (uses_target=False 이므로 targets 키가 없다)
게이트 D 의 값은 같은 run 을 다시 집계한 것이므로 변하지 않는다
```

#### 교훈

D13 은 "무관한 설정 변경으로 baseline 이 무효화되지 않게 한다" 를 목표로 했다. 그
원칙을 **payload 에 키를 추가할 때마다 확인해야 한다.**

```text
이 run 이 실제로 그 설정을 쓰는가
쓰지 않으면 payload 에 넣지 않는다
표시용 해시(meta)도 파일 경로를 정하므로 같은 기준을 적용한다
```

### D31. ablation 결과. `C3` 의 **크기 절반쯤이 acceptance artifact** 였다. 결론은 유지

micro-neural 3 spec × seeds 2/3/4 를 두 수락 규칙에서 각각 216 run, 실패 0.

#### `C3 = shrinking − committed`

```text
              control            fixed_eval
full_batch    −0.095 (1/3)       −0.949 (0/3)
cs128         +2.941 (3/3)       +1.110 (3/3)
cs64          +1.666 (3/3)       +0.973 (3/3)
```

R2 에서 `C3` 가 **약 40~60% 줄었지만 부호는 3/3 로 유지**됐다.

#### `C2 = shrinking − onestep`

```text
              control            fixed_eval
full_batch    +0.547 (2/3)       −0.080 (1/3)
cs128         −1.111 (1/3)       −0.118 (1/3)
cs64          −0.716 (1/3)       +0.104 (2/3)
```

R2 에서 `C2` 가 명확한 음수에서 **약 0** 으로 올라왔다. `shrinking` 이 `onestep` 을
비기는 수준이고 **이기지는 않는다.**

#### `A2 = shrinking − best_static`

```text
              control            fixed_eval
full_batch   +15.176 (3/3)      +14.065 (3/3)
cs128         −0.900 (1/3)       −0.598 (0/3)
cs64          −0.277 (1/3)       −0.093 (0/3)
```

R2 에서 여전히 음수이고, `fixed_eval` 에서는 **0/3 으로 부호가 더 일관되게 음수**다.

#### 사전 등록한 시나리오 중 어느 것인가

D28 에 세 시나리오를 미리 적어 두었다. 결과는 **1번과 2번 사이**다.

```text
1번 C3 여전히 크지만 shrinking 이 C0 보다 나쁨  -> stale-plan 방지일 뿐. RL 근거 없음
2번 C3 작아짐                                 -> 상당 부분이 acceptance artifact
3번 shrinking 이 C0 까지 이김                  -> feedback 연구 재검토  <- 해당 없음
```

정확한 진술은 이것이다.

> `C3` 크기의 절반쯤은 엄격한 minibatch 수락 규칙이 만든 것이었다. 나머지는
> 부호가 일관된 실제 stale-plan 손해다. 그러나 두 수락 규칙 모두에서 `shrinking`
> 은 stochastic regime 에서 `best_static` 을 이기지 못하고 `onestep` 을 비기는 데
> 그친다. **3번 시나리오는 관측되지 않았다.**

따라서 D24 의 PPO 보류 결정을 유지한다.

#### 거절률은 **올라갔다.** 예상과 반대다

```text
                    control      fixed_eval
committed  cs128      0.79          0.89
committed  cs64       0.66          0.92
shrinking  cs128      0.04          0.42
shrinking  cs64       0.00          0.36
onestep    cs64       0.00          0.57
```

`control` 에서는 `loss_before` 와 `candidate_loss` 가 **같은 minibatch** 위에 있다.
방금 gradient 를 계산한 batch 의 loss 를 줄이는 것은 쉽다. `fixed_eval` 은 전체
데이터 loss 의 감소를 요구하므로 훨씬 어렵다.

즉 `fixed_eval` 은 완화가 아니라 **엄격화**다. "현재 batch 에 과적합하는 step" 을
막는다. 과학적으로 옳은 방향이지만 거절이 늘어난다.

#### ablation 이 단일 요인 변경이 아니라는 점을 명시한다

`fixed_eval` 은 두 가지를 동시에 바꾼다.

```text
[1] 수락 기준이 참 목적함수로 바뀐다
[2] 평가 forward 가 n_samples/batch_size 배 비싸므로 같은 예산에 들어가는 step 이 줄어든다
```

[2] 를 회계에서 빼면 비용을 숨기는 것이 되므로 넣었다. 그 대가로 **단일 요인
ablation 이 아니다.** R2 의 절대 logΔ 가 전반적으로 낮아진 것은 두 효과가 섞인
결과다.

```text
cs64        control   fixed_eval
best_static   3.029      1.120
onestep       3.468      0.575
shrinking     2.757      1.002
committed     1.086      0.073
```

**절대값 비교로 두 수락 규칙의 우열을 주장하지 않는다.** 같은 규칙 안의 paired
delta 만 해석한다.

#### `full_batch` 열은 의미 변화가 아니라 예산 교란이다

`full_batch` 에서는 `acceptance_loss()` 와 `curvature_loss()` 가 같은 값이다. 따라서
`fixed_eval` 이 바꾸는 것은 **step 마다 forward 1회를 더 청구하는 것뿐**이다
(`acceptance_forward_units = 1.0`).

```text
best_static  full_batch   control 5.2199  ->  fixed_eval 5.1094
```

이 차이는 의미 변화가 아니라 예산이 조금 줄어든 효과다. **`full_batch` 의 두 열을
"수락 규칙 비교" 로 읽으면 안 된다.**

같은 값을 돌려주는 forward 를 청구하는 것은 비효율이지만, optimizer 가 실제로 그
연산을 수행하므로 회계상 맞다. 특수 처리로 건너뛰는 것은 최적화이고 정확성 수정이
아니므로 하지 않았다. **알려진 비효율로 남긴다.**

#### `full_batch` 에서 `C3` 가 더 음수가 됐다

```text
C3 full_batch   control −0.095 (1/3)  ->  fixed_eval −0.949 (0/3)
```

엄격한 수락 규칙 아래에서는 결정론적 regime 에서도 재계획이 **적극적으로 해롭다.**
`committed` 가 3/3 로 더 낫다. D26 의 quadratic held-out (`C3 = +0.010`, `n=40`) 과
방향이 다르지만, 이쪽은 `n=3` 이고 예산 교란이 섞여 있으므로 **quadratic 결과를
뒤집는 근거로 쓰지 않는다.**

#### `C2` 의 micro-neural full_batch 값이 quadratic 과 다르다

```text
C2 quadratic held-out   +0.456  n=40  p<0.0001   GO
C2 micro full_batch     −0.080  n=3   fixed_eval
```

`n=40` held-out 결과가 강한 쪽이다. `n=3` micro-neural 이 이를 뒤집지 못한다.
**불일치 자체를 결과로 보고한다** (프로토콜 D9).

### D30. GE 는 **FLOP 이 아니라 gradient 호출 횟수**다. regime 간 절대값 비교의 한계

D1 의 정의는 이것이다.

```text
1 GE = gradient batch 1회 forward + backward
```

**task 자신의 gradient batch 를 단위로 쓴다.** 따라서 batch size 가 다르면 같은
`150 GE` 가 다른 FLOP 을 뜻한다.

```text
full_batch (n=512)   1 GE = 512 샘플 gradient
cs128                1 GE = 128 샘플 gradient
cs64                 1 GE =  64 샘플 gradient
```

`150 GE` 에서 `full_batch` 는 `cs64` 의 약 **8배 FLOP** 을 쓴다.

#### 무엇이 유효하고 무엇이 아닌가

```text
유효    regime 내부의 paired delta (A2, C2, C3)
        같은 batch size, 같은 예산이므로 컨트롤러끼리 compute-matched 다

주의    regime 간 절대 logΔ 비교
        "같은 gradient 호출 횟수" 기준이며 "같은 FLOP" 기준이 아니다
```

따라서 `full_batch 20.396` 과 `cs64 2.757` 을 나란히 놓고 "결정론적 환경이 7배 더
좋다" 고 쓰면 안 된다. 정확한 진술은 이것이다.

> 같은 수의 gradient 평가를 허용했을 때, 결정론적 full-batch 목적함수에서 planner 가
> 도달한 개선이 minibatch regime 보다 훨씬 컸다. 두 조건은 FLOP 이 아니라 gradient
> 호출 횟수로 정규화됐다.

#### 이 정의를 바꾸지 않는다

FLOP 정규화로 바꾸면 `full_batch` 예산이 `1/8` 로 줄어 결정론적 조건의 결과가 전부
무효가 된다. 그리고 D1 의 정의는 실측 `CostModel` 계수와 묶여 있다. **정의를
유지하고 한계를 명시하는 쪽을 택한다.**

### D29. `batch_size` 축은 **중간 한 점만** 추가한다

`batch_size ∈ {16, 32, 64, 128, ...}` 를 전면 스캔하면 프로젝트가 끝없이 늘어난다.
이미 두 끝점이 있다.

```text
full batch   planning 가치 큼, feedback 가치 없음
batch 64     planning 붕괴, myopic controller 우세
```

중간 한 점(`batch_size=128`, `n_samples=512` 이므로 epoch 당 4 batch)만 추가해 세
점의 전이 방향을 본다.

```text
full batch  ->  batch 128  ->  batch 64
```

관심 지표를 사전에 고정한다.

```text
planning − C0            (C2)
shrinking − committed    (C3)
거절률
suffix retention / 계획 변경률
```

**전이 방향만 탐색적으로 보고한다.** 세 점으로 함수 형태를 주장하지 않는다.

### D27. micro-neural 두 regime. **모델 정확도가 축이다.** `C3 > 0` 의 원인은 feedback 이 아니다

`shrinking_Q4_narrow` 를 micro-neural 두 regime × seeds 2/3/4 에 적용했다. 144 run,
실패 0, regime 당 `n=3`.

#### regime 별 절대 median logΔ

```text
controller              R2 controlled_stochastic   R1 full_batch
best_static                          3.029              5.220
best_open_loop                       2.984              3.299
heuristic                            0.986              5.105
onestep_narrow                       3.468             19.886
onestep_absolute                     3.732             19.460
committed_Q4_narrow                  1.086             20.491
shrinking_Q4_narrow                  2.757             20.396
```

#### regime 별 paired delta

```text
                          R1 full_batch              R2 controlled_stochastic
A2  shrinking − static    +15.176  (3/3 양수)          −0.277  (1/3)
C2  shrinking − onestep    +0.547  (2/3)               −0.716  (1/3)
C3  shrinking − committed  −0.095  (1/3)               +1.666  (3/3 양수)
ref committed − onestep    +0.470  (2/3)               −2.322  (0/3)
```

#### `C3 > 0` 은 `shrinking` 이 좋아진 것이 아니다

두 해석을 절대값으로 구별해야 한다.

```text
해석 A  shrinking 이 더 좋아졌다  ->  feedback 이 이득을 만든다
해석 B  committed 가 더 나빠졌다  ->  낡은 계획을 고수하면 손해다
```

**실측은 B 다.**

```text
committed_Q4_narrow   R1 20.491  ->  R2 1.086     붕괴
shrinking_Q4_narrow   R1 20.396  ->  R2 2.757     같이 나빠졌다
best_static           R1  5.220  ->  R2 3.029
onestep_narrow        R1 19.886  ->  R2 3.468
```

R2 에서 `committed` 는 `best_static`(3.029)보다도 나쁘고 `onestep`(3.468)보다 `2.322
nat` 뒤진다. `shrinking` 도 `best_static` 과 `onestep` 보다 나쁘다.

결정적 진단은 거절률이다.

```text
committed_Q4_narrow  거절률   R1 0.00   ->   R2 0.66
shrinking_Q4_narrow  거절률   R1 0.00   ->   R2 0.00
```

batch 0 에서 세운 계획의 **2/3 가 이후 batch 에서 거절된다.** 계획이 낡는다.

#### 따라서 R2 는 PPO 를 지지하지 않는다

```text
R2 에서 planner 는 tuned 상수보다 나쁘다        A2 = −0.277
R2 에서 planner 는 1-step greedy 보다 나쁘다     C2 = −0.716
R2 planner 탐색 비용 275,286 GE = 예산의 1,835배
```

`C3 > 0` 의 교훈은 "feedback 정책을 학습하라" 가 아니라 **"낡은 계획을 고수하지
말라"** 다. 그리고 R2 에서 가장 좋은 것은 값싼 1-step greedy 다.

상태 조건 제어 자체는 R2 에서도 도움이 된다 (`onestep 3.468` vs `static 3.029`,
`+0.44 nat`). 그러나 그것은 이미 `onestep` 이 하고 있고 탐색 비용이 `1,879 GE` 로
planner 의 `1/146` 이다.

#### 프로젝트 전체 결론: **모델 정확도가 축이다**

```text
                        planner 모델 정확  planner 모델 부정확
                        (결정론적)          (minibatch)
planning vs static        +15.18            −0.28
planning vs greedy         +0.55            −0.72
feedback (C3)              −0.10            +1.67  <- committed 붕괴 때문
committed 거절률            0.00              0.66
```

> 다단계 계획은 planner 의 내부 모델이 정확할 때 큰 가치가 있고 feedback 은 가치가
> 없다. 모델이 부정확해지면 다단계 계획의 가치가 사라지고, feedback 이 하는 일은
> 낡은 계획을 고수하는 재앙을 피하는 것뿐이다.

`D24` 의 두 갈래 중 **첫 번째**가 성립한다. PPO 보류 결정을 유지한다.

#### 반드시 함께 보고할 교란 요인

`newton_cg.py` 의 `_accept` 는 **단조 감소**를 요구한다. 코드 주석이 이미 경고했다.

> minibatch loss 에는 노이즈가 있으므로 신경망 task 에서 그대로 쓰면 정상 step 도
> 대량 거절된다. 고정 평가 배치로 판정하거나 trust ratio 기준으로 완화해야 한다.

`committed` 의 R2 거절률 `0.66` 은 **계획이 낡은 것과 수락 규칙이 엄격한 것이
섞인 값이다.** 수락 규칙을 완화하면 `C3` 의 크기가 달라질 수 있다.

방향은 바뀌지 않을 것으로 보인다. `committed` 가 `onestep` 보다 `2.32 nat` 뒤지는
것은 거절률만으로 설명되지 않는다 (`shrinking` 은 거절률 0 인데도 `onestep` 보다
나쁘다). 그러나 **크기를 주장하지 않는다.**

#### 그 밖의 한계

```text
regime 당 n=3. CI 와 p-value 를 인용하지 않는다
모델 1종, 데이터 1종, batch_size 1종
batch_size 를 바꾸면 "모델 부정확도" 의 정도가 달라진다. 축을 스캔하지 않았다
R1 에서 onestep 이 19.886 까지 가므로 floor 에 접근한다 (하한 4.8e-14)
```

### D26. held-out confirmatory 결과. `C2` 승격, `C3` 는 **CI 가 좁게 0 을 포함**

사전 고정한 `shrinking_Q4_narrow` 를 challenge 4 spec × held-out seed 100~109 에
적용했다. 960 run, 실패 0, floor hit 0, `n=40`.

**설정을 다시 고르지 않았다.** 후보가 하나뿐이라 D21 규칙이 자동으로 그것을
선택한다 (median logΔ 10.5551).

#### 게이트

```text
A1=GO  A2=GO  B=재설계  C1=판정불가  C2=GO  C3=재설계  D=GO
```

```text
A2   shrinking − best_static   +1.690  CI +1.462~+2.368  p<0.0001  40/40 양수
C2   shrinking − onestep       +0.456  CI +0.254~+0.720  p<0.0001  35 양수 / 5 음수
C3   shrinking − committed     +0.010  CI −0.033~+0.053  p=0.97    21 양 / 1 영 / 18 음
B    absolute − narrow (H=1)   +0.005
ref  open_loop − best_static   +0.395  CI +0.350~+0.476  p<0.0001
ref  heuristic − best_static   −0.000  p=0.78
D    cost-to-target (medium)   1.706배  절감 41.4%  p=0.0004  도달 16/40
```

**`p` 값을 `0.0000` 으로 쓰지 않는다.** 부트스트랩/순열 기반이므로 `p<0.0001` 이
정확하다.

#### dev 대비 변화

```text
             dev n=12                 held-out n=40
A2   +1.502  p=0.0005                 +1.690  p=0.0000
C2   +0.251  p=0.0771  조건부          +0.456  p=0.0000  GO      <- 승격
C3   −0.044  p=0.3804                 +0.010  p=0.9725          <- CI 가 좁아졌다
```

**`C2` 가 조건부에서 GO 로 승격됐다.** `n=40` 에서 GO 임계값 `0.3` 을 명확히 넘고
CI 하한이 `+0.254` 다. `depth>1` 채택률 `0.84`, `cap 0.00` 이므로 P3 의 두 조건이
모두 충족된다. **다단계 lookahead 는 one-step greedy 보다 실제로 낫다.**

#### `C3` 의 성격이 바뀌었다

dev 에서는 `p=0.38` 로 "검출하지 못했다" 였다. held-out 에서는 CI 가 좁다.

```text
C3 = +0.010 nat,  95% CI [−0.033, +0.053],  n=40,  21승 1무 18패
```

##### 주장할 수 있는 문장과 할 수 없는 문장

**equivalence margin 을 사전 등록하지 않았다.** 따라서 "효과가 0 이다" 나 "효과가
`0.053 nat` 보다 작다" 를 검정 결과로 주장할 수 없다. 정확한 표현은 이것이다.

> held-out 결과에서 feedback 효과는 `+0.010 nat` 였으며, 95% CI 가
> `[−0.033, +0.053]` 으로 좁게 0 을 포함했다. 따라서 **실용적으로 큰 feedback
> 이득은 관측되지 않았다.**

`p=0.0000` 이라고 쓰지 않는다. `p<0.0001` 로 쓴다.

부호가 21승 1무 18패로 균형인 것과 CI 폭이 `0.086 nat` 인 것을 함께 보고한다.
`A2` 의 CI 폭이 `0.906 nat` 인 것과 대비하면 정밀도의 차이가 드러난다.

#### 사다리별 측정값. **합으로 분해하지 않는다**

쌍별 차이의 median 은 선형이 아니다. 아래 값들은 각각 독립적으로 측정한 통계다.

```text
튜닝 상수 기준 쌍별 median (held-out n=40)
best_open_loop        +0.395   CI +0.350~+0.476   40/40
onestep_narrow        +1.155   CI +1.092~+1.811   40/40
committed_Q4_narrow   +2.090   CI +1.532~+2.407   40/40
shrinking_Q4_narrow   +1.690   CI +1.462~+2.368   40/40   (A2)

직접 측정한 증분
shrinking − onestep   +0.456   CI +0.254~+0.720   35/40   (C2)
shrinking − committed +0.010   CI −0.033~+0.053   21/40   (C3)
```

##### 표의 값을 서로 빼면 안 된다

`committed` 의 상수 대비 값(`+2.090`)이 `shrinking`(`+1.690`)보다 크지만 직접 측정한
`shrinking − committed` 는 `+0.010` 이다. **모순이 아니다.**

```text
spec 별 shrinking − committed
κ=1e3  +0.472    κ=1e4  −0.019    κ=1e5  +0.009    κ=1e6  +0.000
```

각 spec 안에서 두 planner 는 사실상 동률이고, pooled median 이 서로 다른 인스턴스에
떨어져 marginal 값 차이가 생긴다.

##### 초판 보고의 파생 오류를 정정한다

초판은 `onestep` 의 상수 대비 값을 `A2 − C2 = 1.690 − 0.456 = 1.233` 으로 계산했다.
**틀렸다.** 직접 측정값은 `+1.155` 다.

`scripts/make_report.py` 의 비교 목록에 `onestep − best_static` 과
`committed − best_static` 을 추가해 **손으로 빼지 않아도 되게** 했다. 원고 표와
`Figure 1(b)` 에 "막대를 서로 빼지 말라" 는 경고를 넣었다.

#### κ 의존성 (n=10 each)

```text
κ=1e3   +6.983   CI +6.956~+7.101   10/10 양수
κ=1e4   +2.317   CI +1.529~+2.507   10/10
κ=1e5   +1.399   CI +0.953~+1.517   10/10
κ=1e6   +1.253   CI +0.966~+1.693   10/10
```

dev 의 비단조 패턴이 `n=10` 에서 재현됐다. `κ=1e3` 최대, `κ=1e5` 까지 감소 후 평탄.
**단조 증가 가설은 기각된다.** 다만 여전히 단조 감소를 주장하지 않는다 (D24).

##### median 규약을 하나로 통일했다

초판 보고는 `+6.992 / +2.365 / +1.431 / +1.344` 였다. `three_layer` 의 spec 별 분해가
`sorted(vals)[len(vals)//2]` (상위 중앙값)를 쓴 반면 `compare_paired_delta` 의
all-task 통계는 `statistics.median` (짝수 표본에서 두 중앙값의 평균)을 썼다.

```text
n=10 에서 두 규약이 0.01~0.09 nat 규모로 갈렸다
프로토콜 본문과 자동 생성 표에 다른 숫자가 실렸다
```

`metrics.median_of` 를 **프로젝트 유일 규약**으로 만들고 `three_layer` 와 보고
스크립트 3종을 모두 그것으로 바꿨다. 위 표는 통일 후 값이다.

`n=12` 인 dev 결과와 D21 의 설정 선택은 표본이 4의 배수라 규약 차이가 없다.
**선택 결과 `shrinking_Q4_narrow` 와 순위는 변하지 않았다.**

#### 탐색 비용

```text
shrinking_Q4_narrow  decision-search 194,095 GE / object budget 150 GE = 1,294배
```

#### 이 결과가 PPO 판단을 바꾸지 않는다

`C2` 승격은 **planning 의 가치**를 강화하지만 `C3` 는 여전히 귀무다. PPO 가 학습하는
것은 `π(a|s)` 이고 그 추가 가치가 `0.053 nat` 미만이다. D24 의 보류 결정을 유지한다.

`C2` 가 GO 라는 것은 오히려 **amortized schedule selector** 방향을 지지한다. 좋은
다단계 시퀀스가 존재하고, 그것을 초기 상태에서 정할 수 있다.

### D25. eligibility 를 **달성 가능한** 상한으로 판정한다. 참조 solver panel

D23 에서 D20 의 `ceiling` 공식이 전역최소점 도달을 가정한다는 것이 드러났다.
교정안을 채택한다. 단 **단일 solver 의 수렴점을 절대 상한으로 쓰지 않는다.**

#### 참조 solver panel

planner 를 제외한 강한 방법들을 긴 예산으로 돌려 **최소값**을 쓴다.

```text
long-budget LBFGS
Adam
SGD + momentum
explicit damped Newton  (d <= 512)
필요하면 추가 초기화
```

```text
L_ref        = min over reference runs of L_final
J_achievable = log(L0) − log(max(L_ref, L_floor))
```

단일 solver 만 믿으면 그 solver 의 약점이 상한으로 굳는다. `rosen_d5` 에서
`extra_inits=(0.9,)` 를 넣으면 전역최소점을 찾아 `L_ref` 가 내려간다. panel 이
그 차이를 드러낸다.

**planner 를 panel 에 넣지 않는다.** 넣으면 spec 선정에 planner 결과가 새어 든다
(D20).

#### 이 값의 용도를 제한한다

`J_achievable` 은 **컨트롤러 평가 점수가 아니다.** calibration 지표로만 쓴다.

```text
baseline 이 이미 도달 가능한 최적점에 포화됐는가
여전히 비교할 수 있는 headroom 이 남았는가
```

#### 채택 조건 (D20 개정)

```text
failure_rate = 0
joint floor-hit rate <= 1/3
각 baseline median logΔ >= 1 nat
J_achievable − median logΔ >= 3 nat        <- ceiling 을 바꿨다
참조 solver 간 수렴점 산포 <= 0.5 nat        <- 신설
seed 마다 실제로 다른 인스턴스               <- 신설
```

`limited_by` 를 함께 보고한다.

```text
critical_point    국소최소점이 상한을 정했다. rosen_d5 가 이 경우다
numerical_floor   수치 하한이 상한을 정했다. 볼록 문제의 정상 상태다
reference_failed  참조 solver 가 전부 실패했다. spec 을 쓸 수 없다
```

`numerical_floor` 로 제한된 경우는 floor cap 때문에 값이 갈리므로 산포 검사를
적용하지 않는다.

#### seed 복제 검사

`initial_loss` 만 보면 우연히 같을 수 있으므로 **시작점 벡터도** 본다.
`RosenbrockSpec(dimension=5)` 의 `randomize_start=False` 를 자동으로 잡는다.

#### 다중 초기화는 **상한을 올리지 않는다**

초판 구현이 틀렸다. `extra_inits` 결과를 `L_ref` 후보에 넣었더니 `rosen_d5_rand` 가
통과했다.

```text
잘못된 구현  L_ref = min over 전체 panel  ->  lbfgs@init0.9 가 4.8e-21 을 찾음
             -> J_achievable = 31.44 (numerical_floor)  ->  여유 28.86  ->  채택
```

**컨트롤러는 항상 task 자신의 시작점에서 출발한다.** 다른 초기화에서 더 좋은 점을
찾았다는 사실은 그 시작점의 basin 이 전역최적이 아니라는 **진단**이지, 컨트롤러가
도달할 수 있는 상한이 아니다.

```text
L_ref               = min over runs with from_task_start=True
off_start_best      = min over 다른 초기화. 진단용
start_basin_is_suboptimal   두 값이 0.5 nat 이상 벌어지면 True
```

`start_basin_is_suboptimal` 이면 결과 해석에 **반드시 함께 보고한다.** "방법이
전역최적에 도달했다" 고 쓰면 틀린 주장이 된다.

#### 소급 적용 결과 (검증, seeds 0/1, reference 3000 iters)

```text
spec               판정   J_achievable  제한             최소여유   사유
quad_d100_k1e3     채택        31.44    numerical_floor    13.71
quad_d100_k1e4     채택        31.44    numerical_floor    20.98
quad_d100_k1e5     채택        31.44    numerical_floor    21.88
quad_d100_k1e6     채택        31.44    numerical_floor    21.93
rosen_d5           탈락         1.82    critical_point     −0.00   seed 복제 + 여유 0
rosen_d5_rand      탈락         2.58    critical_point      0.00   여유 0
mlp_full_batch     채택        31.44    numerical_floor    11.68
mlp_stochastic     채택        31.44    numerical_floor    28.75
```

`rosen_d5` 계열의 여유가 **정확히 0** 이다. 모든 baseline 이 도달 가능한 최적점에
이미 도달했다는 뜻이고, D23 의 진단과 정확히 일치한다.

두 Rosenbrock 변형 모두 `start_basin_is_suboptimal=True` 로 표시된다
(`off_start_best = 4.81e-21` vs `L_ref = 3.930839`).

교정된 조건으로도 **challenge selection set 4종은 그대로 통과한다.** D22 의 `n=12`
와 D26 의 `n=40` 결과는 유효하다.

##### `newton` 참조가 `quad_k1e6` 에서 `lbfgs` 를 크게 이겼다

```text
lbfgs   final=1.007485e-06  |grad|=3.456e-02  미수렴
newton  final=3.059391e-32  |grad|=2.650e-16  수렴
sgd     final=nan
```

단일 solver 를 상한으로 쓰면 안 되는 이유의 실측 예다. `lbfgs` 만 썼다면
`J_achievable` 을 `28.3 nat` 로 과소평가했을 것이다.

### D24. `shrinking_Q4_narrow` freeze. **PPO 보류.** 연구 질문을 둘로 분해한다

D22 결과를 리뷰한 결정이다. 여기서 정한 것은 되돌리지 않는다.

#### 확정된 결론

```text
비싼 다단계 탐색은 좋은 Newton-CG 제어 시퀀스를 찾는 데 가치가 있었지만,
결정론적 quadratic 에서는 실행 중 상태 피드백으로 계획을 수정하는 추가 가치는
관측되지 않았다.
```

현재 확인된 헤드룸은 **feedback control** 이 아니라 **sequence planning /
schedule selection** 에 있다.

성능이 좋아진 구간과 좋아지지 않은 구간이 갈린다.

```text
좋아짐    best static -> open-loop -> one-step -> committed planner
안 좋아짐  committed planner -> shrinking feedback planner
```

#### 원래의 넓은 질문이 둘로 분해됐다

```text
[Q1] 좋은 action sequence 가 존재하는가          -> 현재 결과 Yes
[Q2] 그 sequence 를 실행 중 feedback 으로 수정할 가치가 있는가
                                                  -> quadratic 에서는 No 또는 미확인
```

**이것은 실패가 아니다.** 초기의 "RL optimizer" 아이디어가 검증 가능한 두 명제로
쪼개진 것이다.

#### 게이트 C3 판정: **불충족**

```text
C3 = J_E(shrinking) − J_E(committed) = −0.044 nat
CI −0.590~+0.070 (0 포함),  p=0.3804,  6승 6패
```

현재 quadratic 결과만으로 가장 정직한 결론은 "feedback replanning 의 추가 가치가
없다" 다.

#### PPO 착수를 **보류**한다

PPO 가 학습하는 것은 `π(a_t | s_t)` 다. 상태가 변할 때 행동을 바꾸는 가치가 있어야
정당화된다. 현재는 초기 상태에서 한 번 계획한 `committed` 가 매 step 재계획하는
`shrinking` 과 같거나 낫다. 이 상태에서 PPO 를 학습하면 다음 중 하나가 될 가능성이
크다.

```text
고정 schedule 암기
초기 상태만 보고 schedule 선택
불필요한 재계획 노이즈 학습
planner 의 비싼 탐색 비용을 줄이지 못한 채 성능도 안 나옴
```

현재 결과에 더 잘 맞는 학습 대상은 PPO 가 아니라 **amortized schedule selector** 다.

```text
초기 문제 특징 (loss, gradient norm, 곡률 통계, CG residual 특성, condition 추정)
  -> 전체 Newton-CG schedule
```

**다만 지금 방향을 바꾸지 않는다.** 먼저 비선형 또는 micro-neural 환경에서 feedback
가치가 있는지 확인한다.

#### 최종 결론은 아직 두 갈래다

```text
micro-neural 에서도 C3 ~ 0
  -> 적응적 계산 배분의 이득은 주로 초기 상태 기반 sequence selection 에서 발생하며
     매 단계 feedback 기반 RL 은 정당화되지 않았다.
     PPO 를 구현하지 않는 것이 올바른 연구 결정이다.
     후속 방향은 committed planner 를 저비용으로 근사하는 schedule predictor 다.

micro-neural 에서 C3 > 0
  -> quadratic 에서는 모델이 정확해 feedback 이 불필요했지만, 미래 상태가 불확실한
     neural optimization 에서는 feedback control 이 가치가 있다.
     PPO 착수 근거가 생긴다.
```

#### 설정 freeze

```text
shrinking_Q4_narrow
```

**다른 `Q` / space 를 다시 들여다보고 바꾸지 않는다.** D21 규칙으로 한 번 선택했다.

#### beam 8 결과의 역할을 제한한다

D21 의 순서 이탈(비용 측정 dry run 의 게이트 표를 본 뒤 선택 규칙 확정) 때문에
현재 결과를 완전한 confirmatory 로 부를 수 없다.

```text
beam 8 challenge-dev  configuration selection 과 가설 정교화에만 사용
최종 효과 추정        사전 고정된 설정으로 새 held-out seed 에서 수행
```

#### held-out confirmatory 비교 집합 (사전 고정)

```text
best_static
best_open_loop          resource-clock 스케줄
onestep_narrow          C0
committed_Q4_narrow
shrinking_Q4_narrow
```

seed 는 이미 분리해 둔 `HELD_OUT_SEEDS = 100~109` 를 쓴다.

#### κ 결과를 단조 관계로 해석하지 않는다

```text
κ=1e3 +6.584   κ=1e4 +2.361   κ=1e5 +0.735   κ=1e6 +1.232
```

"condition number 가 커질수록 adaptive control 이 중요해진다" 는 가설은 지지되지
않는다. 그러나 κ 당 seed 가 3개뿐이므로 **반대 방향의 단조 관계를 주장해서도 안
된다.** 정확한 진술은 이것이다.

> Adaptive headroom 은 condition number 에 따라 단조 증가하지 않았으며, 가장 큰
> 효과는 `κ=1e3` 에서 관측됐다. 이는 condition number 만으로 headroom 을 설명하기
> 어렵다는 것을 시사한다.

damping grid, CG budget, 초기 gradient alignment, spectrum 분포가 함께 영향을 줄
수 있다.

#### Rosenbrock 을 계속 고치지 않는다

`rosen_d5` 에 start noise 를 더 주거나 차원을 바꾸며 쓸 만한 버전을 찾는 것은
**결과를 본 뒤 benchmark 를 조정하는 모양**이 되고, Rosenbrock 의 basin 구조가
연구 질문을 흐린다. 다음 용도로만 남긴다.

```text
rosen_d2   쉬운 문제의 floor saturation 진단
rosen_d5   특정 시작점에서 국소최소점으로 포화되는 비선형 진단
```

비선형 일반성은 원래 계획했던 **micro-neural (P4)** 에서 확인한다.

#### micro-neural 설계: 핵심 질문은 비선형성이 아니다

```text
틀린 질문   모델이 비선형인가
맞는 질문   초기 계획 시점에 미래 상태를 정확히 예측할 수 없는가
```

deterministic quadratic 에서는 planner 의 모델이 거의 정확했기 때문에 feedback 이
필요 없었을 수 있다. 따라서 같은 task 를 두 regime 으로 나눈다.

```text
[R1] deterministic full-batch
     동일 데이터 전체로 gradient 와 HVP 계산
     quadratic 처럼 local model 이 일관됨. committed 와 shrinking 차이가 작을 것으로 예상

[R2] controlled stochastic
     고정 seed 의 mini-batch 시퀀스. HVP/gradient 계산 batch 가 시간에 따라 변함
     실행 중 관측 상태가 초기 계획의 예상과 달라질 수 있다
     **이때만 feedback 의 진짜 가치가 생길 가능성이 있다**
```

모델과 데이터는 **하나만 고정한다.**

```text
2-layer MLP
파라미터 수 수천~수만
작은 분류 데이터셋 또는 고정 subset
HVP 가 CPU 에서 안정적으로 계산 가능
```

### D23. `rosen_d5` 는 **국소최소점에 갇힌** task 였다. D20 ceiling 공식이 틀렸다

선택된 설정 `shrinking_Q4_narrow` 를 `rosen_d5` 에 적용한 결과, 12개 컨트롤러의
모든 게이트가 `+0.000 nat` 이었다. 원인을 진단했다.

#### 표면 관측

```text
A1=재설계  A2=재설계  B=재설계  C1=판정불가  C2=재설계  C3=재설계  D=판정불가
모든 delta 가 +0.000 또는 −0.000
target 도달률 0/3  (easy = absolute_loss <= 1e-1 조차 미달)
```

그런데 planner 는 depth 8 까지 계획을 채택했다 (`chosen_depths {8:1, 7:1, ..., 1:19}`,
`depth>1` 비율 0.27, `depth_cap_hit 0.00`). 계획이 무력화된 것이 아니다.

#### 원인 1: 표준 시작점의 basin 에 **strict 국소최소점**이 있다

`final_loss` 는 실제로 14개 서로 다른 값이었다. 전부 `3.9308` 근방이다.

```text
onestep_narrow        3.9308388233
committed_Q4_narrow   3.9308393002
shrinking_Q4_narrow   3.9308393002
best_static           3.9308681488
open_loop[0]          3.9310777187
static[2]             4.6165843010   (나쁜 설정은 더 못 갔다)
```

LBFGS 로 정밀 수렴시켜 임계점을 특정했다.

```text
x*        (-0.96205102, 0.93573939, 0.88071360, 0.77787767, 0.60509367)
loss      3.930839434133
|grad|    1.06e-08
Hessian   eig min +5.95e-01, eig max +1.44e+03   -> 양정
```

표준 시작점 `(-1.2, 1, 1, 1, 1)` 에서 LBFGS 를 돌려도 **전역최소점이 아니라 이
점으로 수렴한다.** `(0.9, ..., 0.9)` 에서 출발하면 `loss=0` 에 도달한다. 즉 시작점의
basin 이 국소최소점의 basin 이다.

`rosenbrock.py` 모듈 docstring 이 이미 `d >= 4` 에서 국소최소값이 존재한다고
적어 두었다. **그 경고를 D20 calibration 이 반영하지 못했다.**

#### 원인 2: D20 의 `ceiling` 공식이 이 경우 무의미하다

D20 은 이렇게 계산했다.

```text
ceiling = log(L0) − log(L0 x RELATIVE_LOSS_FLOOR) = log(1/2.22e-14) = 31.44 nat
```

이것은 **전역최소점이 0 이고 도달 가능하다**고 가정한다. `rosen_d5` 의 실제
달성 가능 상한은 국소최소점이 정한다.

```text
D20 이 쓴 ceiling         31.44 nat   -> "여유 29.62 nat" 으로 통과
실제 달성 가능 상한        log(24.2 / 3.930839) = 1.8175 nat
관측된 baseline median     1.8175 nat  -> 여유 0.0000 nat
```

**모든 baseline 이 정확히 1.8175 였던 이유가 이것이다.** 전부 국소최소점에
도달했다. `κ` 축 tie-break 로 탈락시킨 것은 결과적으로 옳았지만, 근거가
`log10(κ)` 균등 포괄이었을 뿐 이 결함을 잡아낸 것이 아니다.

##### 교정된 eligibility 조건 (제안)

사후 임계값이 아니라 **기전 기반**이다. 컨트롤러 비교를 열지 않고 계산할 수 있다.

```text
achievable_ceiling = log(L0) − log(L_ref)
  L_ref = 시작점에서 강한 참조 solver 로 정밀 수렴시킨 임계점의 loss
채택 조건에서 ceiling 을 numerical floor 대신 이 값으로 바꾼다
추가 조건: achievable_ceiling >= (요구 median logΔ + 여유)
```

`rosen_d5` 는 `achievable_ceiling = 1.8175` 이므로 `median logΔ >= 1` 과
`여유 >= 3` 을 **동시에 만족할 수 없다.** 즉 교정된 조건으로는 자동 탈락한다.

#### 원인 3: `randomize_start=False` 라 `n=3` 이 실제로는 `n=1` 이었다

```text
RosenbrockSpec(dimension=5)  ->  randomize_start=False
seed 2 / 3 / 4  전부 x0 = (-1.2, 1, 1, 1, 1),  L0 = 24.200000
```

seed 2 와 seed 4 의 24개 run 이 **모든 컬럼에서 bitwise 동일**했다. 따라서 이
진단의 `n=3`, CI, p-value 는 전부 무의미하다. **표본은 1개다.**

`confirmatory_specs()` 는 `RosenbrockSpec(dimension=10, randomize_start=True)` 를
쓰므로 플래그는 존재했다. D20 이 `dimension=5` 만 지정하고 기본값을 확인하지
않은 것이 원인이다.

##### `randomize_start=True` 로 바꿔도 cap 은 남는다

```text
seed2  L0=64.88 -> 3.9308394341  국소최소점  달성가능 ceiling 2.804 nat
seed3  L0=70.51 -> 3.9308394341  국소최소점  달성가능 ceiling 2.887 nat
seed4  L0=19.00 -> 3.9308394341  국소최소점  달성가능 ceiling 1.575 nat
```

세 시작점 모두 **같은 국소최소점**으로 수렴한다. `start_noise=0.1` 은 basin 을
벗어나기에 너무 작다. 따라서 `randomize_start=True` 는 원인 3만 고치고 원인 1을
고치지 못한다.

#### quadratic challenge set 은 영향이 없다

```text
spec           seed 별 L0                                  eig min
d100_k1e+03    6532.47 / 8105.36 / 5409.81                 +1.0
d100_k1e+04   38955.02 / 50803.88 / 60202.39               +1.0
d100_k1e+05  347484.25 / 470283.05 / 243511.90             +1.0
d100_k1e+06  4681346.48 / 6335280.50 / 3146287.74          +1.0
```

seed 마다 다른 인스턴스이고 전부 SPD 다. 최소점이 유일하므로 국소최소점 cap 이
없다. **D22 의 `n=12` 결과는 유효하다.**

#### 따라서 비선형 일반성은 아직 미검증이다

D22 의 `C3 ≈ 0` 이 quadratic 의 결정론성 때문인지 일반적 성질인지 구별하려면
비선형 task 가 필요하다. 그런데 현재 비선형 후보는 다음 상태다.

```text
rosen_d2   numerical floor 로 포화 (D19)
rosen_d5   국소최소점으로 cap, 게다가 n=1 (D23)
```

**둘 다 사용 불가다.** 새 비선형 task 설계는 프로토콜 변경이므로 리뷰 대상이다.
`start_noise` 를 키워 basin 을 넘게 하거나, `d=10` 처럼 국소최소점이 없는 차원을
쓰거나, 다른 함수족을 도입하는 선택지가 있다. **이 결정을 코딩 에이전트가 하지
않는다.**

### D22. beam 8 challenge 결과. 헤드룸은 **feedback 이 아니라 sequence** 에 있다

challenge selection set (quad d=100, κ∈{1e3,1e4,1e5,1e6}) × seeds 2/3/4 = 12
인스턴스, 150 GE, beam 8, 360 run, 실패 0, floor hit 0.

#### D21 규칙 적용 결과

```text
configuration          n   median logΔ    search GE
shrinking_Q4_narrow   12       10.5306     193893.6   <- 선택
shrinking_Q2_wide     12       10.3691     127609.1
shrinking_Q2_narrow   12       10.3208      55801.5
shrinking_Q4_wide     12       10.2508     441612.3
```

최대값이 2위와 `0.16 nat` 차이로 `TIE_TOLERANCE=0.05` 를 넘으므로 **단독 선택**
이다. tie-break 사다리를 쓰지 않았다.

`Q4_wide` 가 `Q4_narrow` 보다 낮다. 행동 집합이 커지면 탐색 가능 집합은 포함관계로
커지지만 **실현 성능은 비감소가 아니다** (§게이트 C 에 이미 명시). beam 8 이
넓어진 공간을 다 감당하지 못한다.

#### 사다리: 총 헤드룸 `+1.502 nat` 의 분해

```text
controller             logΔ     vs best_static   비고
best_static            8.895      —              상수 action, 튜닝됨
heuristic              8.886      −0.000          static 과 사실상 동일 (p=0.79)
best_open_loop         9.278      +0.413          4구간 고정 스케줄, 상태 미관측
onestep_absolute       9.989      +1.263          A1. 1-step, 132 action
onestep_narrow        10.188      +1.29           1-step, 12 action
shrinking_Q2_narrow   10.321      +1.43
shrinking_Q4_narrow   10.531      +1.502          A2. 선택된 설정
committed_Q4_narrow   10.566      +1.67           초기 상태에서 한 번 계획, 맹목 실행
committed_Q4_wide     10.574      +1.68
```

paired median 으로 다시 쓰면 이렇다.

```text
A2   shrinking − best_static     +1.502  CI +1.160~+4.402  p=0.0005  12/12 양수
C2   shrinking − onestep         +0.251  CI −0.036~+0.487  p=0.0771  8 양수 / 4 음수
C3   shrinking − committed       −0.044  CI −0.590~+0.070  p=0.3804  6 양수 / 6 음수
B    absolute − narrow (H=1)     +0.015  CI −0.070~+0.055  p=0.9097
ref  open_loop − best_static     +0.413  CI +0.317~+0.754  p=0.0005
```

게이트: `A1=GO  A2=GO  B=재설계  C1=판정불가  C2=조건부  C3=재설계  D=GO`.

#### 가장 중요한 관측: `C3 ≈ 0`

`committed` 는 **초기 상태에서 한 번 계획하고 그대로 실행하는** oracle 이다. 그것이
매 step 재계획하는 `shrinking` 과 같거나 약간 낫다.

> 이득은 좋은 **행동 시퀀스**를 찾는 데서 나오고, 상태를 보고 **적응**하는 데서
> 나오지 않는다.

이것이 PPO 착수 판단에 직접 영향을 준다. PPO 가 학습하는 것은 `π(a|s)` 즉 상태
조건 feedback 정책이다. feedback 의 추가 가치가 0 이면 PPO 의 상한은 committed
수준이고, 그 수준은 per-instance 탐색으로 이미 도달 가능하다.

##### 다만 이것을 일반화하면 안 된다

quadratic 은 **결정론적이고 예측 가능하다.** 초기 상태와 행동 시퀀스가 주어지면
궤적이 완전히 결정된다. 따라서 초기 상태에서 세운 계획이 이미 최적 예측이고
재계획이 새 정보를 얻을 수 없다.

```text
가능한 해석 1  feedback 은 원래 가치가 없다
가능한 해석 2  이 task 족이 결정론적이라 planner 의 모델이 정확했다
```

**두 해석을 이 실험으로 구별할 수 없다.** 비선형 진단(`rosen_d5`, D20)이 정확히
이 지점을 시험한다.

#### κ 의존성은 가설과 반대였다

```text
A2 (shrinking − best_static)  spec 별 median
κ=1e3   +6.584   [+6.329, +7.152, +6.584]
κ=1e4   +2.361   [+2.476, +1.523, +2.361]
κ=1e5   +0.735   [+1.464, +0.735, +0.605]
κ=1e6   +1.232   [+1.480, +1.232, +1.087]
```

"adaptive-control headroom 이 condition number 에 따라 증가하는가" 라는 질문의
답은 **아니오** 다. `κ=1e3` 에서 가장 크고 `κ=1e5` 까지 감소한 뒤 평평해진다.

pilot 의 dev subset 에서는 `quad_ill κ=1e5` 만 측정 가능해 "ill-conditioned 에서만
헤드룸" 처럼 보였다. **그것은 비교 대상이 포화됐기 때문이었고, κ 축을 채우니
방향이 뒤집혔다.** D19/D20 의 교정이 이 관측을 가능하게 했다.

#### 행동 공간은 병목이 아니다 (B = 재설계)

```text
absolute  132 action, log10 범위 15.27   vs   narrow  12 action, log10 범위 0.95
차이 +0.015 nat, p=0.9097
wide − narrow = +0.044 nat, p=0.3804
```

damping 을 자유롭게 고를 수 있게 해도 1-step 성능이 오르지 않는다. **좁은 multiplier
공간으로 충분하다.** 이것은 음의 결과지만 유용하다. 값싼 행동 공간을 정당화한다.

#### planning 이 실제로 일어났다 (P3 두 번째 조건)

```text
Q4  depth>1 채택률 0.85   plan-depth 상한에 걸린 비율 0.00
Q2  depth>1 채택률 0.68   상한 0.00
```

개선이 탐색량만으로 생긴 것이 아니다. 다만 `C2 = +0.251, p=0.0771` 은 GO 임계값
`0.3` 에 못 미치므로 **조건부**다.

#### 탐색 비용을 숨기지 않는다

```text
shrinking_Q4_narrow  decision-search 193,894 GE   /   object budget 150 GE  =  1,293배
committed_Q4_narrow   69,336 GE  =  462배
onestep_narrow         1,186 GE  =  7.9배
best_open_loop 튜닝   10,531 GE (12 인스턴스 전체) = 인스턴스당 878 GE
```

`+1.502 nat` 은 인스턴스당 예산의 **1,293배**를 쓴 oracle 값이다. amortize 질문
(P4)이 남는다.

### D21. 설정 선택 통계를 실행 전에 하나로 못박는다

§게이트 C 의 "설정(Q, action space) 선택은 beam 8 dev 결과의 median 으로 한 번만
한다" 는 **무엇의 median 인지 미지정이었다.** D19 에서 all-task median 과 spec별
median 이 반대 결론을 냈으므로 이 자유도를 남겨두면 사후 선택이 된다.

```text
all-task median  A2/C2/C3 모두 +0.000  ->  재설계
quad_ill κ=1e5   +0.494 / +0.542 / +0.370  ->  GO
```

#### 선택 대상은 `shrinking` 의 (Q, space) 하나다

`committed` 는 배포 가능한 컨트롤러가 아니라 **초기 상태에 조건화된 비교 oracle**
(C3)이다. `fresh` 는 진단 baseline 이다. 따라서 freeze 대상은 `shrinking` 뿐이고,
`committed` 는 선택된 같은 `(Q, space)` 에서 비교용으로 함께 돌린다.

#### 선택 통계: planner **자신의** median logΔ 를 최대화한다

baseline 과의 delta 로 고르지 않는다. **strongest baseline 순위가 현재 표본에서
안정적이지 않기 때문이다** (P2 각주: `onestep_absolute` 와 `heuristic` 이 둘 다
31.438, CI −5.982~+24.346, p=0.906). 불안정한 기준점으로 나눗셈을 하면 설정
선택이 baseline 잡음을 따라간다.

```text
선택 통계  median over 12 challenge 인스턴스 (4 spec x 3 seed) of
           shrinking 의 Track E log improvement (절대값, delta 아님)
목표       최대화
```

challenge set 은 D20 calibration 에서 floor hit 0 을 확인했으므로 이 median 이
floor cap 에 눌리지 않는다. **dev subset 과 달리 all-task median 을 쓸 수 있다.**

#### tie-break 사다리 (사전 고정)

median 이 `0.05 nat` 이내로 같으면 순서대로 적용한다.

```text
1  decision-search GE 가 적은 쪽      같은 성능이면 싼 것
2  Q 가 작은 쪽                       계획 지평이 짧아 단순하다
3  narrow 가 wide 보다 우선           행동 공간이 작아 단순하다
```

#### 함께 보고하되 선택에 쓰지 않는 것

```text
spec 별 median 4개          conditioning 의존성 분석
개별 12 delta               표본이 작아 CI 가 거칠다
shrinking − best_static     헤드룸 주장 (A2)
shrinking − C0              (C2)
shrinking − committed       (C3)
rosen_d5                    선택된 설정만 사후 적용 (D20)
```

#### 사전 공개: 1 인스턴스 dry run 을 먼저 실행했다

비용 측정을 위해 `--max-tasks 1 --seeds 1` (즉 `quad_d100_k1e3`, seed 2) 로 beam 8
을 한 번 돌렸고 그 게이트 표를 봤다. 30 run, 약 8분.

이 규칙은 그 결과와 무관하게 정할 수 있는 것만 담았다. dry run 에서 `Q × space`
별 planner logΔ 순위를 열지 않았고, 선택 통계를 "planner 자신의 median 최대화"
로 정한 근거는 baseline 순위 불안정성(P2, 기존 관측)이다. 그럼에도 순서가
`규칙 확정 → 전체 실행` 이 아니라 `dry run → 규칙 확정 → 전체 실행` 이었다는
사실을 기록한다.

dry run 산출물은 삭제하지 않고 `config_hash` 가 다른 별도 파일로 남긴다. 전체
실행과 섞이지 않는다.

### D19. 포화는 task 이름이 아니라 `floor_hit`으로 판정하고, spec별로 보고한다

D14 초판은 `rosen_d2 제외 = primary` 로 정의했다. **틀렸다.** 포화는 task 이름이
아니라 실제 `floor_hit` 으로 발생한다.

#### 실측: `quad_spd`도 포화 task였다

```text
quad_spd_d64_k1e+02  shrinking vs C0
  seed0  loss 1.69e-20 vs 1.14e-18   GE 139.1 vs 134.3   steps 7 vs 11
  seed1  loss 1.03e-19 vs 2.62e-21   GE 139.1 vs 147.6   steps 7 vs 12
  seed2  loss 5.41e-20 vs 8.86e-19   GE 139.1 vs 135.6   steps 7 vs 12
  -> trajectory 가 전부 다른데 delta 가 정확히 0
```

`L0` 이 약 `1e2` 규모이므로 floor 는 `1e-12` 근처다. 관측된 loss 는 `1e-18~1e-21`
로 floor 훨씬 아래다. 양쪽이 cap 되어 같은 값이 된다.

이 spec 의 결론은 "적응 제어가 실패했다" 가 아니다.

> 150 GE 에서는 모든 강한 방법이 요구 정밀도를 넘어 수렴해 차이를 식별할 수 없다.

**benchmark calibration 결과다.**

#### joint saturation과 one-sided saturation을 구별한다

```text
joint saturation      양쪽 모두 floor. 구분력이 없다
one-sided saturation  한쪽만 floor. **그 컨트롤러가 명확히 우수하다는 증거다**
```

`rosen_d2` 에서 `shrinking − best_static = −5.422` (3 seed 전부)가 나왔다. 이것은
`best_static` 만 floor 에 도달한 one-sided saturation 이므로 **명확한 성능 차이**
다. "구분 불가" 가 아니다.

#### `drop_saturated_pairs`를 primary 게이트로 승격하지 않는다

비교 쌍마다 표본이 달라지고, one-sided saturation 쌍을 삭제하면 **좋은 결과를
제거한다.** 실측에서 그 위험이 확인됐다.

```text
A2 all-task            +0.000 nat  n=9
A2 pairwise nonsat      −2.675 nat  n=6   부호가 반대다
```

`rosen_d2` 의 `−5.422` 가 삭제되지 않고 median 에 남아 방향을 뒤집었다. 민감도
분석으로만 쓴다.

#### spec별 보고가 핵심이다

전체 median 하나로 요약하면 난이도별 차이가 사라진다.

```text
beam 4 pilot, shrinking Q4 narrow
                                  vs best_static   vs C0      vs committed
quad_ill_conditioned d100 k1e+05     +0.494        +0.542       +0.370
quad_spd d64 k1e+02                  +0.000        +0.000       +0.000
rosen_d2_s100_std                    −5.422        +0.000       +0.000
```

세 spec 의 역할이 다르다.

| Spec | 관측된 역할 |
|---|---|
| `quad_spd κ=1e2` | joint saturation 으로 구분 불가 |
| `rosen_d2` | `best_static` 이 planner 보다 명확히 우수한 easy regime |
| `quad_ill κ=1e5` | adaptive planning 의 양의 headroom 후보 |

정확한 진술은 이것이다.

> Well-conditioned quadratic 은 모든 강한 방법이 numerical floor 에 도달해 구분력이
> 없었고, Rosenbrock (d=2) 에서는 best static 이 planner 보다 우수했으며,
> ill-conditioned quadratic 에서만 adaptive planning 의 일관된 양의 headroom 이
> 관측됐다.

**측정 가능한 regime 이 현재 하나뿐이다.** "실질 표본이 3쌍" 이라기보다 이렇게
말하는 것이 정확하다.

### D18. Bridge 검증 규칙을 실행 전에 수치로 고정한다

D13(3계층 정체성), D16(selection manifest), D17(open-loop resource clock) 은
**실행 의미를 바꾸지 않아야 한다.** identity / logging / baseline 시계만 고쳤다.
그러면 planner trajectory 는 legacy pilot 과 동일해야 한다. 이것을 확인하는 것이
bridge 다.

#### tolerance

동일 CPU, float64, 단일 스레드, 동일 seed 이므로 **기대값은 bitwise exact** 다.
tolerance 는 bitwise 가 깨졌을 때만 보조로 쓴다.

```python
REL_TOL = 1.0e-12
ABS_TOL = 1.0e-14
bitwise_equal or math.isclose(legacy, new, rel_tol=REL_TOL, abs_tol=ABS_TOL)
```

**정확히 0인 loss 는 양쪽 모두 `0.0` 이어야 한다.** `0` 과 작은 양수를 단순히
tolerance 로 같다고 넘기면 D14 의 saturation 상태가 바뀐다.

#### 분류 (실행 전 고정)

```text
EXACT                 bitwise 동일
CLOSE                 수치 tolerance 만 통과
MISMATCH              tolerance 도 실패
ZERO_MISMATCH         한쪽만 정확히 0. 별도로 센다
LEGACY_FIELD_MISSING  legacy 에 새 계측 필드가 없음. 실패로 세지 않는다
MISSING / DUPLICATE   키에 행이 0개 또는 2개 이상. 비교하지 않는다
```

#### legacy row 선택은 결정론적으로

같은 논리적 run 이 여러 `experiment_id` 에 존재하므로 기준을 명시한다.

```text
legacy_reference_experiment_id = 0a63f5e6de3d
```

각 `(mode, quota, space, task, seed)` 키에 legacy 1행, new 1행이 정확히 있어야
한다. **중복을 평균하거나 최신 timestamp 로 임의 선택하지 않는다.** 0개나 2개
이상이면 `MISSING` / `DUPLICATE` 로 분류하고 비교를 보류한다.

#### exact match가 기대되는 항목

```text
object GE / step 수 / termination reason / total HVP / search GE
action counts / chosen-depth counts
raw final loss / initial loss
planner_stats: mean_simulations, depth_cap_hit, quota_ge, max_depth_seen
```

**`mean_simulations` 가 달라지면 로깅 차이로 넘기지 않는다.** 탐색 순서, pruning,
incumbent carry-over, 컨트롤러 구현이 달라졌을 가능성이 있으므로 trajectory 가
같아도 원인을 규명한다.

legacy 에 없을 수 있는 새 계측값(`suffix_retention_rate`, `n_replans`,
`windows`)은 `LEGACY_FIELD_MISSING` 으로 처리한다.

#### 통과 조건

```text
모든 공통 이산 필드 exact match
loss 는 bitwise 또는 tolerance 충족
설명되지 않는 MISMATCH / ZERO_MISMATCH / DUPLICATE / MISSING 이 0개
```

통과하면 beam 4 전체를 다시 돌리지 않고 사전 등록된 beam 8 로 넘어간다.

#### 실패 시 진단 순서

aggregate median 을 먼저 보지 않는다. **최초로 갈라지는 지점**을 찾는다.

```text
첫 action -> 초기 계획 sequence -> 첫 planner 후보 점수
-> remaining quota -> 실제 첫 step GE -> 다음 상태의 loss / grad norm
```

최종 loss 부터 역추적하면 원인을 좁히기 어렵다.

#### calibration은 별도로 취급한다

`calibrate-beam` 이 쓰는 컨트롤러를 `fresh` → `shrinking` 으로 고쳤다 (D13
커밋). 이것은 **실제 의미 변경**이므로 일반 planner bridge 와 섞어 비교하지
않는다. calibration run 은 bridge 대상에서 제외한다.

### D17. open-loop 스케줄의 시계를 GE 예산으로 바꼾다

#### 기존 결함

```text
step-indexed progress = step / total_steps
```

가 실제 종료 조건인 **GE 예산과 불일치**했다 (D1). `total_steps = 200`, 예산
150 GE, `cg_budget = 20` 이면 약 7 step 만에 끝나므로 `progress` 가 0.035 를
넘지 못한다. 따라서 **스케줄의 첫 구간만 실행됐고, 뒤쪽 구간은 도달 불가였다.**

실측 (beam 4 pilot, dev 9 인스턴스):

```text
best_open_loop = open_loop[4]   4구간 스케줄을 선택
결과가 best_static 과 9쌍 전부 bitwise 동일
paired delta +0.000 nat,  CI +0.000~+0.000
```

이것은 "비정적 스케줄이 static 으로 퇴화했다" 는 과학적 결과가 **아니다.**
baseline 구현 결함이다. D4 의 목적이 적응 정책과 사전 스케줄을 비교하는 것이므로
이 상태를 유지하면 P2 에서 "스케줄 baseline 을 이겼다" 는 주장을 검증할 수 없다.

#### 수정된 정의

```text
progress_t = min(1, spent_object_ge_t / cost_budget_ge)
progress_clock = object_ge_fraction
progress_evaluated_at = before_step
OPEN_LOOP_SEMANTICS_VERSION = 2
```

이것을 **budget-indexed open-loop schedule** (resource-clock schedule) 이라
부른다. 실제 GE 소비는 CG 조기 종료 등으로 변하지만, 컨트롤러가 loss / gradient /
Hessian 상태를 **관찰하지 않고** 계산 예산 시계만 쓰므로 적응 제어와 구별되는
강한 단순 baseline 이다.

#### 수정 후 검증

```text
open_loop[7] 선택   median logΔ 25.456  (수정 전 9.337)
schedule 4구간 전부 실행
  구간 0 [0.000~0.107] x3.0,   K=5    4 step
  구간 1 [0.107~0.683] x1.0,   K=20  20 step
  구간 2 [0.683~0.707] x1.0,   K=5    1 step
  구간 3 [0.707~1.000] x0.333, K=20  11 step
realized_segment_counts = {0:4, 1:20, 2:1, 3:11}
is_constant_schedule = False
```

정체성 격리도 확인됐다. **open-loop 108 run (12 후보 × 9 인스턴스) 만 재실행**되고
`static` / `heuristic` / `one-step` 의 `run_semantics_id` 와 `selection_id` 는
유지됐다.

```text
open_loop selection_id  3ed50a72067a -> 1b6de46d089f   변경
static    selection_id  649ef8f9878f -> 649ef8f9878f   유지
```

`realized_segment_counts` 를 기록하는 이유는 비싼 action 하나가 breakpoint 를
건너뛰면 특정 구간이 실행되지 않을 수 있기 때문이다. 오류는 아니지만 스케줄이
실제로 얼마나 쓰였는지 보여야 한다.

#### 기존 step-clock 결과의 위치

**confirmatory baseline 으로 사용하지 않는다.** 구현 결함 진단 기록으로만
보존한다.

> Step-indexed progress 가 실제 GE-budget horizon 과 불일치하여 첫 구간만
> 실행되는 구조적 결함을 확인했고, confirmatory 분석 전에 resource-clock 방식으로
> 수정했다.

#### P2 해석에 대한 주의

`best_open_loop` 의 median logΔ 가 9.337 → 25.456 으로 올랐지만 **곧바로 static
보다 우수하다는 뜻이 아니다.** paired delta 는 −0.137 nat, CI −5.982~+24.346,
`p = 0.906`, 포화 `one-sided = 6/9` 이다.

> Open-loop 가 강한 baseline 후보가 되었지만, 현재 dev 표본은 baseline 간 순위를
> 안정적으로 구분하지 못한다.

`onestep_absolute` 와 `heuristic` 이 둘 다 31.438 인 것도 floor ceiling 의 영향을
받은 값이다. **P2 의 strongest baseline 을 단순 median 최대값 하나로 정하지 않고**,
confirmatory 에서 paired 비교와 saturation-aware 보고를 함께 쓴다.

### D16. baseline 선택 과정을 manifest로 남긴다

`best_static` 과 `best_open_loop` 는 컨트롤러가 아니라 **튜닝을 통해 선택된
결과**다. 그런데 raw 기록에는 `static[3]`, `open_loop[4]` 같은 후보 라벨만 남고
어느 것이 선택됐는지가 없어서, 재집계에서 A1·A2를 계산할 수 없었다.

**라벨만 `static[7]` → `best_static` 으로 바꾸면 안 된다.** 어떤 설정이 왜
선택됐는지가 사라지고, 나중에 evaluation 결과를 보고 역추정하게 되어 사후 선택이
된다.

#### SelectionManifest

```text
selection_id                 선택 과정 정체성 해시
family                       static / open_loop
candidate_labels             후보 라벨
candidate_scores             라벨 -> 선택 지표 값
selection_metric             median_log_improvement
tie_break_rule               lowest_flat_index
selected_label               선택된 후보
selected_config              선택된 **실제 설정**
tuning_specs / tuning_seeds  튜닝 범위
n_tune                       후보 수
selection_semantics_version  선택 규칙 버전
resolved                     False 면 legacy_unresolved
```

`static` 은 action 을, `open_loop` 은 **schedule 전체**를 기록한다.

#### 정체성 규칙

evaluation run 의 `run_semantics_id` 에 `"best_static"` 만 들어가면 안 된다.
선택 결과가 달라졌는데 generic label 때문에 같은 ID 가 나오면 낡은 결과를
재사용한다. 실제 선택된 설정이 semantics 에 들어가야 한다.

#### 복원은 당시 tuning 결과만으로 한다

```text
당시 tuning 후보 점수 + selection metric + tie-break 가 남아 있음
    -> 선택 결과를 결정론적으로 재현. selection_id 를 붙인다

evaluation 결과만 있음
    -> best_static = legacy_unresolved
       best_open_loop = legacy_unresolved
       A1·A2 는 미판정으로 유지하고 새 프로토콜로 다시 실행한다
```

"`static[7]` 이 좋아 보이니 그게 best_static 이었을 것" 이라고 정하면 사후 선택이다.

#### open-loop이 static으로 퇴화하는 경우

최종 성능이 같다는 것만으로는 부족하다. 실제 schedule 을 출력해 모든 구간의
action 이 같은지 본다 (`SelectionManifest.is_constant_schedule`). 같으면

> 튜닝된 open-loop baseline 은 비정적 스케줄의 이점을 발견하지 못하고 최적 static
> configuration 으로 퇴화했다.

이는 버그가 아니라 결과다. 단 **P2 에서 `best_static` 과 `best_open_loop` 를 서로
다른 두 개의 강한 증거처럼 세면 안 된다.** 사실상 같은 baseline 이다.

추가로 같은 task·seed 에서 action sequence, damping trajectory, CG budget
trajectory, object GE, terminal loss 를 bitwise 비교해 확인한다.

### D14. log improvement의 수치 하한을 사전 고정한다

초판 `RunSummary.log_improvement` 는 `final_loss <= 0` 이면 NaN 을 반환했고,
paired 비교는 비유한값을 **조용히 버렸다.** 그러면 최적점에 정확히 도달한 run 이
집계에서 제거된다.

beam 4 pilot 실측: `rosen_d2_s100_std` 의 `onestep_absolute` 와 `heuristic` 이
`final_loss = 0.0` (정확히 0)이라 각각 3쌍이 아무 기록 없이 빠졌다. `쌍 6/9` 로만
표시됐고 게이트 A1 과 B 가 낮게 잡혔다.

#### 상대 floor

```python
RELATIVE_LOSS_FLOOR = 100.0 * float64_eps        # 2.220e-14
loss_floor = max(tiny, abs(initial_loss) * RELATIVE_LOSS_FLOOR)
log_improvement = log(initial_loss) - log(max(final_loss, loss_floor))
```

**`finfo.tiny` 를 floor 로 쓰면 안 된다.** 최대 log improvement 가 708 nat 까지
커져서 실제 최적화 차이보다 underflow 여부가 통계를 지배한다. 초기 loss 에
상대적으로 잡으면 scale invariant 하다. `d=2` Rosenbrock (`L_0 = 24.2`) 에서
floor 는 `5.4e-13`, 최대 logΔ 는 `31.4` nat 다.

`100 x eps` 는 누적 반올림 오차가 machine epsilon 의 수십 배까지 커지는 것을
감안한 값이며 **결과를 보고 고른 것이 아니다.**

#### 음수는 두 종류로 나눈다

```text
-floor <= final_loss < 0     부동소수점 roundoff. 0 으로 clamp, negative_roundoff=True
final_loss < -floor          numerical failure. NaN, excluded_pairs 에 사유 기록
```

모든 음수를 clamp 하면 실제 계산 오류를 숨기고, 모든 음수를 실패로 처리하면
최적점 근처 run 을 부당하게 제거한다.

#### 포화를 세 종류로 기록한다

```text
exact_zero              raw final_loss == 0
joint_saturation        비교하는 두 컨트롤러 모두 floor_hit -> terminal objective 상 실제 동률
one_sided_saturation    한쪽만 floor_hit -> 엄격히 우수하지만 차이 크기는 floor 의존 하한
```

paired 결과에 `n_valid`, `n_joint_saturated`, `n_one_sided_saturated`,
`excluded_pairs` 를 함께 출력한다. **조용한 `dropna` 를 금지한다.** `n_valid` 가
`n_pairs` 보다 작으면 반드시 task·seed·사유가 남는다.

#### 게이트는 두 버전을 병기한다

주 통계는 floor-capped 전체 쌍이고, 비포화 쌍만의 결과를 **민감도 분석으로만**
병기한다. 비포화만 primary 로 쓰면 최적점에 도달한 강한 run 을 다시 제거하는
편향이 된다. 두 결론이 다르면 "쉬운 인스턴스의 포화 처리에 민감하다" 고 보고한다.

beam 4 pilot 재집계 실측: `B-wide` 는 주 `+0.308`(조건부) 대 비포화 `+0.962`(GO),
`C1` 은 주 `+0.000`(재설계) 대 비포화 `+0.518`(GO), `C3` 은 주 `+0.000` 대 비포화
`+0.115`(조건부)로 **세 게이트에서 결론이 뒤집혔다.** 9쌍 중 3쌍이 `rosen_d2`
포화이고 그 쌍의 delta 가 0 이어서 중앙값을 지배한다.

#### Rosenbrock d=2를 primary에서 분리하고 세 층으로 보고한다

`d=2` Rosenbrock 은 150 GE 에서 여러 컨트롤러가 정확한 최적점(`loss = 0.0`)에
도달한다. 더 이상 adaptive controller 의 성능 차이를 재는 benchmark 가 아니라
sanity check 다.

**결과를 보고 불리한 task 를 제거하는 것이 아니다.** beam 8 실행 전에 다음을
확인하고 정했다.

```text
여러 컨트롤러가 정확히 loss = 0 에 도달
9쌍 중 동일한 3쌍이 joint saturation
floor 처리 때문에 paired delta 가 기계적으로 0 으로 고정
  A2 = +0.000, C2 = +0.000, C3 = +0.000  (전부 joint=3)
```

##### 세 층을 **동시에** 보고한다 (D19에서 재정의)

```text
[1] all-task floor-capped        n=9. joint / one-sided / unsat 개수를 함께
[2] spec 별                      난이도 regime 별 median 과 개별 delta
[3] pairwise nonsaturated 민감도  비교마다 n 이 달라진다. **primary 아님**
```

**`rosen_d2 제외 = primary` 정의는 폐기했다** (D19). 포화는 task 이름이 아니라
실제 `floor_hit` 으로 발생한다. `quad_spd` 도 floor 아래로 내려가 컨트롤러를
구분하지 못한다.

`GO` / `재설계` 이진 라벨보다 **개별 paired delta 를 함께 출력**한다. 표본이
작아 CI 와 p-value 가 거칠기 때문이다.

**전체 GE budget 을 낮추지 않는다.** 낮추면 어려운 quadratic 에서 필요한 헤드룸까지
제거된다.

### D15. 재계획이 계획을 실제로 바꿨는지 행동 내용으로 계측한다

`shrinking` 과 `committed` 가 `Q=1` 에서 bitwise 같은 `final_loss` 를 냈다
(`4.554803848266602`). alias 가 아니라 **실제 동률**이다. 근거는 두 컨트롤러의
`chosen_depths` 와 `planner_stats` 가 달랐다는 것이다.

```text
committed  chosen_depths {4: 9}                      계획 9회
shrinking  chosen_depths {4: 9, 3: 9, 2: 9, 1: 8}    계획 35회
           mean_simulations 56.4 (committed 60.0)
```

즉 `shrinking` 은 매 step 재계획했고 남은 suffix 를 4→3→2→1 로 유지했다.

그런데 **`chosen_depth` 히스토그램만으로는 부족하다.** 깊이만 같고 행동 내용이
다를 수 있다. 그래서 행동 내용을 직접 비교해 계측한다.

```text
replanned_actions == previous_plan[1:]
suffix_retention_rate = 유지 횟수 / 재계획 횟수
```

`1.0` 이면 재계획이 계획을 한 번도 바꾸지 않았다는 뜻이고 committed 와 같은 경로를
간다. `planner_stats["suffix_retention_rate"]` 로 기록한다.

따라서 `C3 = +0.000` 은 구현 실패가 아니라 유효한 pilot 결과다.

> 작은 quota scale 에서는 피드백을 받아 재계획하더라도 기존 계획을 수정할 만한
> 추가 헤드룸이 없다.

`Q2`·`Q4` 에서는 결과가 갈리므로(9.670 vs 9.681, 9.601 vs 9.831) 진짜 sequential
control 헤드룸은 그쪽에서 검증한다.

RL 이나 adaptive controller 의 가치가 있으려면 세 조건이 모두 필요하다.

```text
재계획을 한다                  <- Q1 에서 충족
상태 변화로 계획이 실제로 바뀐다  <- Q1 에서 미충족
그 수정이 terminal objective 를 개선한다
```

객체·mutable 상태 비공유는 회귀 테스트로 봉인했다. 두 컨트롤러가 서로 다른 객체와
타입이고, `choices`/`trajectory` 를 공유하지 않으며, 실행 순서를 바꿔도 결과가
같다는 것을 검증한다.

### D13. 실험 정체성을 run semantics와 sweep coverage로 분리한다

D8에서 `experiment_id = hash(canonicalized_full_config)` 로 정했다. 낡은 결과
재사용을 막는 데는 맞았지만 **하나의 해시가 두 역할을 겸하고 있었다.**

실측 사례: `fresh_diagnostic_seeds` 와 `run_fresh_wide` 를 추가하자
`experiment_id` 가 `eab1697716b5 → 0a63f5e6de3d` 로 갈리면서 `best_static`,
`open_loop`, `heuristic`, `shrinking`, `committed` 320 run이 전부 무효화됐다.
그런데 그 두 값은 **어떤 run을 도는가**만 정하고 **각 run이 어떻게 동작하는가**는
바꾸지 않는다. 재사용을 막을 이유가 없었다.

#### 세 개의 ID로 나눈다

집계 코드만 바꿨는데 423 run 이 다시 돈 사건이 실행 정체성과 집계 정체성도
분리해야 함을 보여줬다.

```text
run_semantics_id = hash(effective_controller_config)
    개별 run 의 출력값을 바꾸는 설정만. 컨트롤러별로 다르다.
      task spec / seed / controller 종류 / 그 컨트롤러가 쓰는 action space
      GE budget / max steps / damping·CG 설정 / solver·fallback 규칙
      planner 계열이면 quota / beam / max plan depth / 실행 방식
      target (종료 조건으로 쓰는 경우만)
      semantics version (optimizer / planner / task)

sweep_id = hash(run_selection_config)
    이번 실행에서 어떤 run 들을 모으는가.
      controller 목록 / 전체 task·seed 목록
      fresh_diagnostic_seeds / run_fresh_wide
      screening 인지 confirmation 인지 / 출력 경로 / code_dirty

aggregation_id = hash(aggregation_rules)
    표·paired delta·게이트 판정을 만드는 규칙.
      log floor 정책 / 포화 분류 / paired intersection 규칙
      bootstrap·CI 설정 / 게이트 정의 / report schema version
```

```text
run_semantics_id  ->  raw trajectory / RunSummary
aggregation_id    ->  표·paired delta·게이트 판정
sweep_id          ->  어떤 run 과 어떤 보고서를 묶었는지
```

```text
RunKey = run_semantics_id | controller | task_instance | seed | target
```

**`sweep_id` 나 `aggregation_id` 가 바뀌어도 semantics 가 같으면 재사용한다.**
floor 정책이나 게이트 정의를 바꾸면 `aggregation_id` 만 달라져야 하며 optimizer
trajectory 를 다시 돌면 안 된다.

#### 코드 버전을 통째로 해시에 넣지 않는다

git commit 을 `run_semantics_id` 에 넣으면 문서나 집계 코드만 바꿔도 모든 run 이
무효화된다. 완전히 제외하면 optimizer 구현 변경을 놓친다. 그래서 실행 의미를
명시적 버전으로 관리한다.

```python
OPTIMIZER_SEMANTICS_VERSION   # step 하나의 결과를 바꾸는 변경
PLANNER_SEMANTICS_VERSION     # planner 탐색·선택 규칙. planner 계열만 영향
TASK_SEMANTICS_VERSION        # 초기점, Hessian 구성, instance_id 규칙
AGGREGATION_VERSION           # 집계 규칙
```

git commit 과 `code_dirty` 는 **provenance 로만 저장**한다. `code_dirty` 는
`sweep_payload` 에만 들어간다.

#### `effective_controller_config` 는 컨트롤러가 실제 쓰는 설정만 넣는다

공통 config를 통째로 넣으면 안 된다. `best_static` run은 `beam` 이나 `quota` 가
바뀌어도 무효화될 이유가 없다. 컨트롤러별로 관련 키만 골라 해시한다.

#### 파일 구조와 출력 이름

`results/raw/{sweep_id}.jsonl` 로 sweep 단위 파일을 만들고, **각 행에
`run_semantics_id` 를 기록한다.** 한 screening 결과를 한 파일에서 보면서 재사용
근거도 남는다.

출력에서 `config_hash` 라는 모호한 이름을 쓰지 않고 다음을 명시한다.

```text
sweep_id
run_semantics_id
protocol_version
git_commit
code_dirty
```

#### 회귀 테스트 (D13 완료 조건)

1. `fresh_diagnostic_seeds` 만 바꾸면 baseline 과 planner 의 `run_semantics_id` 가 유지된다
2. beam 을 바꾸면 static·heuristic·C0 의 ID 는 유지되고 planner 계열만 변경된다
3. floor·CI·게이트 규칙을 바꾸면 raw run ID 는 유지되고 `aggregation_id` 만 변경된다
4. optimizer 또는 planner 실행 규칙을 바꾸면 관련 컨트롤러의 ID 가 변경된다
5. 같은 semantics run 이 다른 sweep 에 포함돼도 한 번만 실행되고 양쪽에서 참조된다
6. dict 순서나 기본값 생략 여부가 달라도 ID 가 동일하다
7. 실제 effective config 가 다른 두 run 은 절대 같은 ID 를 만들지 않는다
8. Track E 는 target 을 정체성에서 제외하고 Track T 는 포함한다
9. `code_dirty` 는 `run_semantics_id` 에 없고 `sweep_id` 에만 있다

#### 기존 raw 결과 마이그레이션

기존 결과를 버리거나 다시 실행하지 않는다. `final_loss = 0` 은 실행 오류가 아니라
올바른 결과다.

```text
기존 raw row -> 당시 저장된 config 와 controller label 복원
             -> effective config 생성 -> run_semantics_id 부여
             -> migration_version 기록
```

설정 정보가 부족해 의미를 확실히 복원할 수 없는 row 만 `legacy_unresolved` 로
분리한다. **추측해서 새 ID 를 붙이지 않는다.**

### D12. 계획의 가치와 실행 방식을 분리한다

D10 쿼터 사다리에서 "쿼터를 키우면 성능이 나빠진다"가 관측됐다. 그런데 planner가
찾은 **계획 자체**는 동일 비용의 greedy 궤적보다 좋았다. 따라서 문제는 목적함수도
탐색도 아니라 **실행 방식**이었다.

세 방식을 동일 planner, 동일 쿼터, 동일 GE 예산에서 비교한다. 실행 방식만 다르므로
차이가 탐색 품질 차이와 섞이지 않는다.

```text
committed        계획을 끝까지 실행. 재계획 없음. 소진되면 새 window
fresh-quota      매 step 미래 예산 Q 를 새로 지급 (D10 초판 방식)
shrinking-quota  쓴 비용을 차감. horizon 을 새로 연장하지 않음
```

#### 먼저 확인할 불변조건

```text
J_predicted_plan  ≈  J_committed_execution
```

synthetic task는 결정적이므로 planner가 예측한 terminal loss와 그 계획을 끝까지
실행한 결과가 같아야 한다. **이게 맞지 않으면 이후 비교는 의미가 없다.** 12개
조건 전부에서 상대오차 `< 1e-9` 로 일치했다.

#### shrinking은 이전 계획의 suffix를 보장 후보로 포함한다

결정적 환경에서 재계획이 더 나은 것을 못 찾아도 이전 suffix는 유지할 수 있어야
한다. 그렇지 않으면 beam 근사 때문에 재계획 자체가 성능을 떨어뜨리고, 그것이
"피드백이 해롭다"로 오해된다.

단, **탐색에서 살아남는 것은 보장되지만 채택이 보장되는 것은 아니다.** 목적함수가
남은 쿼터 안에서 더 낮은 terminal loss를 찾으면 계획을 버린다. 그 이탈이
국소적으로는 개선이어도 episode 전체로는 손해일 수 있다.

#### 쿼터 차감은 직전 step의 실제 비용으로 한다

예측 비용을 쓰면 CG가 조기 수렴한 만큼 쿼터가 과도하게 줄어들어 window가 일찍
닫힌다. 실측에서 이 차이가 컸다.

```text
예측 비용 차감:  SPD beam8 Q=4  shrinking 48.90   (셋 중 최악)
실제 비용 차감:  SPD beam8 Q=4  shrinking 56.73   (committed 와 동일)
```

`context.previous.cost_ge` 를 쓴다.

#### 실측 결과 (quadratic, seed 0, narrow, 150 GE, max_depth 24)

```text
SPD κ=1e2   C0 = 52.137 nat
  beam 8, Q=4×c_max   committed +4.59   shrinking +4.59   fresh +3.67
  beam 4, Q=4×c_max   committed −3.38   shrinking −4.21   fresh −13.26
  beam 4, Q=2×c_max   committed −2.03   shrinking −1.91   fresh −4.82

ill κ=1e5   C0 = 9.558 nat
  beam 8, Q=4×c_max   committed +0.91   shrinking +0.89   fresh −0.23
  beam 4, Q=4×c_max   committed +0.35   shrinking +0.87   fresh −0.16
  beam 4, Q=2×c_max   committed +0.57   shrinking +0.65   fresh +0.33
```

해석: **`committed > C0`, `shrinking ≈ committed`, `fresh < committed`.**
원인은 쿼터 초기화에 의한 시간 불일치다. fresh-quota는 매 step 미래 예산을 새로
지급하므로 "나중에 이득을 얻을 준비 행동"을 계속 고르면서 payoff를 뒤로 미룬다.
horizon을 연장하지 않으면(shrinking) 피드백 재계획은 committed 대비 손실이 없다.

`beam 4 → 8` 에서 `Q=4` 결과가 SPD에서 8 nat 이상 움직인다. **`Q=4` 조건은 아직
탐색 한계에 걸려 있으므로, 그 수치를 planning 가치의 하한으로만 읽는다.**

#### 두 종류의 비용을 항상 구분한다

탐색 비용이 실제 최적화 비용의 500~2,600배다 (committed 80,311 GE vs fresh
389,245 GE, 본문 150 GE). **planner를 실용 optimizer로 제시하면 안 된다.**

```text
object-level cost     실제 모델을 최적화하는 데 쓴 GE          (total_cost_ge)
decision-search cost  oracle/planner 가 action 을 고르려고
                      쓴 분석 GE                              (search_cost_ge)
```

이 구분이 연구 서술의 핵심이다. planner는 **헤드룸 측정 장치**이고, PPO 단계의
질문은 다음이다.

> 더 좋은 optimizer를 무에서 발명하는 것이 아니라, 비싼 shrinking planner가
> 발견한 계산 배분 전략을 저비용 정책으로 amortize 할 수 있는가?

주장 범위도 여기에 맞춘다. "우리 optimizer가 더 빠르다"가 아니라 "시간 일관적인
계산 배분에 헤드룸이 있고, 그것을 정책으로 근사할 수 있는지 측정했다"다.

---

## 3. 로깅과 provenance

step 단위 JSONL(README §13 스키마)에 다음을 추가한다.

```json
{
  "cost_ge": 26.4,
  "cost_model_id": "rtx3060ti_cu130_b512",
  "git_commit": "a1b2c3d",
  "config_hash": "9f8e7d6c",
  "task_instance_id": "spd_d100_kappa1e3_seed0",
  "failure_tag": null
}
```

원칙:

- 모든 run은 config 스냅샷 + git commit hash를 함께 저장한다. dirty working tree면
  경고를 남기고 diff도 저장한다.
- wall-clock 측정은 warm-up 후 `torch.cuda.synchronize()` 를 앞뒤로 호출한다.
- peak VRAM은 `torch.cuda.max_memory_allocated()` 로 기록한다.
- 실패한 run을 삭제하지 않는다.

---

## 4. 실행 계획

각 Stage 끝에 **게이트**가 있다. 게이트를 통과하지 못하면 다음 Stage로 가지 않고
설계를 수정한다.

### Stage 0 — 환경과 scaffold  (완료)

- uv + venv + pyproject + `uv.lock`
- 패키지 구조, 인터페이스 정의(`types.py`), 동작하는 유틸(flatten/seed/logging/provenance)
- `benchmark/cost_model.py` + `scripts/measure_cost_model.py` 및 실측 산출물
- 이 문서
- 첫 커밋

게이트: `uv run pytest -q` 통과, `torch.cuda.is_available() == True`,
double-backward HVP 상대오차 < `1e-5`. **통과** (71 tests, ruff clean,
GPU HVP 상대오차 `5.8e-8`)

### Stage 1 — 수치 커널  (완료)

비용 모델(`benchmark/cost_model.py`)과 실측은 Stage 0에서 완료했다.
`configs/cost_model.mnist_mlp.yaml`, `configs/cost_model.small_cnn.yaml` 참조.

1. `curvature/hvp.py` — `HvpGraph`. 그래프를 한 번만 만들고 k회 재사용한다.
   그 결과 "한 CG solve 안에서 동일한 curvature batch"(README §15)가
   규율이 아니라 **구조로** 보장된다. 다른 배치를 쓰려면 새 그래프가 필요하다.
2. `curvature/operators.py` — `DampedHessianOperator`, preconditioner 2종.
   damping은 그래프 재사용 중에도 바꿀 수 있다 (step 거절 후 재풀이 시 절약).
3. `solvers/conjugate_gradient.py` — truncated PCG, `CGResult` 반환
4. `tasks/quadratics.py`, `tasks/rosenbrock.py`
5. `benchmark/paired.py` — 결정론적 `seed → task instance` 매핑
6. `scripts/verify_numerics.py` — 게이트를 수치로 보고

게이트: **전체 통과** (CPU / CUDA 양쪽, 183 tests, ruff clean)

| 항목 | 임계값 | 실측 (FP32) |
|---|---|---|
| HVP vs explicit Hessian (κ ≤ 1e3) | < 1e-5 | 4.1e-8 ~ 9.4e-8 |
| HVP, ill-conditioned κ=1e5 | < 1e-4 | 6.1e-8 |
| Newton-CG 방향 vs explicit solve, κ=1e1 | < 1e-3 | 3.7e-7 (27 iters) |
| Newton-CG 방향, κ=1e4 | < 1e-3 | 1.3e-4 (307 iters) |
| damping 증가 → CG 수렴률 | 단조 비감소 | 단조, 최대 damping에서 1.00 |
| indefinite negative curvature 탐지 | 탐지 + damping으로 복구 | 양쪽 확인 |

#### Stage 1에서 발견한 것

세 가지가 초기 가정과 달랐고, 모두 이후 단계에 영향이 있다.

**1. negative curvature 판정은 상대 기준이어야 한다.** `p^T A p <= eps` 처럼
절대 임계값을 쓰면 `p^T A p ∝ ||p||²` 이므로 수렴이 진행되어 `p` 가 작아질 때
양정 행렬에서도 조건이 성립해 **오탐**이 난다. 곡률이 아니라 스케일을 재는 셈이다.
`p^T A p <= eps * ||p||²` 로 바꿨다. RL 상태 특징에 `negative_curvature` 가
들어가므로, 이 오탐은 정책 학습을 직접 오염시킬 수 있었다.

**2. Rosenbrock 표준 시작점은 Hessian이 양정이다.** `det H = 8s²(x² − y) + 4s`
이므로 negative curvature는 `y > x² + 1/(2s)`, 즉 골짜기 **위쪽**에서만 발생한다.
표준 시작점 `(-1.2, 1.0)` 은 `y = 1.0 < x² = 1.44` 로 아래쪽이고 고유값이
23.6, 1506이다. "비볼록 문제이니 시작부터 음의 곡률"이라는 가정은 틀렸다.

**3. `tasks/quadratics` 에서는 Jacobi preconditioner가 원리적으로 무력하다.**
`A = Q diag(λ) Qᵀ` 를 랜덤 직교기저로 만들면 `A` 의 대각이 거의 상수가 된다
(실측 분산 < 10배). Stage 5에서 diagonal preconditioner의 이득이 없다고 나오면
구현 결함이 아니라 문제 구조 때문이다. 대각이 퍼진 계에서 별도로 평가해야 한다.

**4. `κ=1e6` 문제는 damping을 `1e6` 수준까지 올려야 예산 20회 안에 풀린다.**
`1e2` 정도로는 damped 조건수가 여전히 ~1e4다. Stage 2 헤드룸 측정에서
action space의 damping 배수 `{0.3, 1.0, 3.0}` 만으로는 극단적 ill-conditioned
구간에 도달하는 데 여러 step이 걸린다는 뜻이다. 이 점이 헤드룸의 크기에
영향을 줄 수 있으므로, 초기 damping 설정과 배수 범위를 Stage 2에서 함께 본다.

### Stage 2 — 헤드룸 측정  ← 이 프로젝트의 분기점

**목적.** RL 스택을 만들기 전에 적응 제어의 여지가 얼마나 있는지 측정한다.
README 순서대로 가면 RL이 돌아가기까지 2~3주가 걸리고 그때서야 "애초에 이득이
있었나"를 알게 된다.

**초판 설계는 파일럿에서 실패했다.** 단일 `greedy_oracle` 을 상한으로 쓰려 했으나,
그것의 목적(`Δlog L / cost`)과 평가 지표(cost-to-target)가 다른 문제여서 오라클이
고정 설정보다 나쁜 결과를 냈다. D9에 따라 두 트랙으로 분리하고 게이트를 재정의한다.

#### 비교군

```text
best_static           행동 공간 전수 고정 → 최고 선택            (N_tune 회)
best_open_loop        progress 만 보는 스케줄, 랜덤 서치         (N_tune 회)
heuristic             trust ratio 규칙                          (N_tune 회)
one_step_efficiency   매 step 전수 sweep, 즉시 효율 최대 선택     (게이트 C의 C0)
budgeted_Q1/Q2/Q4     동일 미래 GE 쿼터 안의 계획 비교            (게이트 C의 C1~C3)
avgrate_H3/H5         누적 평균 효율 planner                     (진단 baseline, D10)
lagrangian_b*         `Δlog L − β·Σc` planner                    (보조 민감도, D10)
```

`one_step_efficiency` 는 **상한이 아니다** (D9). `mpc_*` 도 유한 horizon 근사다.
행동 공간은 `narrow` / `wide` / `absolute` 세 가지를 쓰며, 세 공간의 **로그
해상도를 맞춘다**. `absolute` 가 범위만 넓고 해상도가 거칠면 게이트 B가 도달성
손실과 해상도 손실을 섞는다 (파일럿에서 실제로 발생).

기본 조건은 **step_size 고정 1.0** 이다. damping이 큰 구간에서 update가
`-(α/λ)g` 로 근사되어 `(λ, α)` 와 `(10λ, 10α)` 가 aliasing되므로, 먼저
`damping × CG budget` 만 분리해 본다. 헤드룸이 확인되면 step_size 축을 추가한다.

#### 계산 자원

**Stage 2는 GPU를 쓰지 않는다.** 대상이 quadratic(d=32~100)과 Rosenbrock(d=2~10)
뿐이므로 CPU가 더 빠르다. Stage 0 실측에서 10만 파라미터 MNIST MLP조차 GPU
런치 오버헤드 지배(0.68 ms/gradient)였으므로 d=100 matvec을 GPU로 보내면 순손실이다.

| 단계 | 디바이스 | VRAM |
|---|---|---|
| Stage 1~2 | CPU (실측) | 0 |
| Stage 2.5 micro-neural | CPU 가능 | ~0 |
| Stage 3 MNIST MLP (102k, B=512) | GPU 권장 | 계획값 < 100 MB |
| Stage 4 PPO (DummyVecEnv) | GPU 또는 CPU | 계획값 수백 MB |
| Stage 5 small CNN (2.19M, B=128) | GPU | 계획값 ~1 GB |

**Stage 3 이후의 VRAM 숫자는 확정값이 아니라 계획값이다.** HVP 메모리는
`create_graph` 유지 기간, 후보를 병렬 평가하는지, mixed precision 여부,
curvature batch 크기, activation 크기에 따라 달라진다. 각 Stage 진입 시
`torch.cuda.max_memory_allocated()` 와 `max_memory_reserved()` 를 다시 실측한다.
PPO controller 자체는 작아서 GPU가 반드시 필요한 것도 아니다.

#### 실행은 재개 가능해야 한다

Stage 2는 컨트롤러 × 행동공간 × horizon × task × seed × target 조합이라 수백~수천
run이 된다. 프로세스가 끊겨도(셸 중단, timeout) 계산한 결과를 잃지 않아야 한다.

- run 하나가 끝나는 즉시 `results/raw/headroom_<tag>.jsonl` 에 append
- 재실행 시 완료된 `(controller, task_instance, seed, target)` 조합은 건너뜀
- 상태를 `completed` / `failed` 로 구분. 미완료는 파일에 없으므로 자동 재시도
- 각 run에 GE, HVP, wall-clock, action 빈도, `chosen_depth` 분포, git commit,
  config hash를 함께 기록
- **raw와 집계를 분리한다.** 집계 로직을 바꿔도 실험을 다시 돌리지 않는다

#### Beam width와 쿼터는 추측이 아니라 측정으로 정한다

beam search는 정확한 planner가 아니므로 폭을 줄이면 계획 품질이 떨어질 수 있고,
그것이 게이트 C의 결론을 바꿀 수 있다. 따라서 **pilot calibration parameter**로
취급한다. D10 이후에는 planning 쿼터와 beam을 **함께** calibration한다.

측정 범위:

```text
beam   ∈ {1, 2, 4}
quota  ∈ {1, 4} × c_max        사다리의 양 끝
space  ∈ {narrow, wide}
seed   dev seed 만 (held-out 100~109 사용 금지)
```

**선택 규칙 (사전 정의. 결과를 본 뒤 바꾸지 않는다):**

1. 최대 beam(4)을 기준으로 삼는다
2. 모든 `(space, quota)` 조합에서 다음 **둘을 모두** 만족하는 beam 중
   **가장 작은 것**을 고른다

   ```text
   |J_b − J_ref| / max(|J_ref|, ε) < 0.02          ε = 1e-6
   |d_b − d_ref| ≤ 0.05                            d = chosen_depth > 1 비율
   ```

3. tie-break: 가장 작은 beam → wall-clock이 짧은 것 → 그래도 같으면 beam 2

수치는 코드 상수로 고정되어 있다 (`UTILITY_TOLERANCE = 0.02`,
`UTILITY_EPSILON = 1e-6`, `DEEP_FRACTION_TOLERANCE = 0.05`). "크게 다르면 제외"를
결과를 본 뒤 판단하면 사후 선택이 되므로 수치로 못박는다.

분모가 `|J_ref| + ε` 이 아니라 `max(|J_ref|, ε)` 인 이유는 `J_ref` 가 0 근처일 때
상대 오차가 폭발하기 때문이다.

`chosen_depth` 를 함께 보는 이유는 효용이 비슷해도 깊은 계획을 쓰는 빈도가
달라지면 게이트 C의 해석이 바뀌기 때문이다. 큰 쿼터를 줬는데도 거의 항상 1이면
장기 planning의 실질적 가치가 낮다는 직접적 증거이고, 이는 PPO 착수 판단에
직결된다.

`depth_cap_hit` 도 함께 기록한다. 0이 아니면 계산 상한 때문에 쿼터를 다 쓰지
못한 step이 있다는 뜻이므로, 그 calibration 행은 쿼터 사다리 비교가 훼손된
것으로 표시한다.

#### wall-clock tie-break를 쓰므로 CPU를 고정한다

GE가 주 지표이므로 핵심 결론은 CPU 경쟁에 영향받지 않는다. 그러나 tie-break에
wall-clock을 쓰므로 스레드를 고정하고 환경을 기록한다.

```python
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

기록 항목: CPU 모델, 코어 수, torch/interop 스레드 수, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, 동시 실행 프로세스 수. **실행 환경이 달라지면 calibration을
재사용하지 않는다.**

#### 재개 판단은 전체 config 해시로 한다

`(controller, task, seed, target)` 만으로 완료를 판단하면 위험하다. beam,
horizon, GE 예산, action space, CG budget, damping 격자 중 어느 하나가 바뀌어도
같은 조합으로 보고 **낡은 결과를 새 결과로 착각**한다. 재개 기능이 오히려 실험을
오염시킨다.

```text
experiment_id = hash(canonicalized_full_config)
run_key       = experiment_id | controller | task_instance | seed | target
```

`experiment_id` payload에 포함되는 것: `protocol_version`, phase, device,
GE 예산, `max_steps`, damping 초기값·경계, CG tolerance, `pap_eps`,
`max_loss_increase_ratio`, `safe_fallback`, horizons, beam, `tuning_budget`,
스케줄 구간 수, tuning seed, target 정의, task spec 목록, seed 목록,
**행동 공간 정의 전체**(damping 값, CG budget, step size), 그리고 `code_dirty`.

git commit은 provenance로 기록하지만 정체성으로는 불충분하다. 커밋되지 않은
변경이 있을 수 있으므로 `code_dirty` 를 함께 넣는다.

`chosen_depth`를 반드시 본다. `H=5`인데 거의 항상 1이면 장기 planning의 실질적
가치가 낮다는 직접적 증거이고, 이는 PPO 착수 판단에 직결된다.

#### 게이트

**Gate A1 — Instantaneous absolute-action headroom (Track E)**

> 현재 상태에서 좋은 damping이 존재하는가?

도달성 제약을 완전히 없앤 `absolute` 행동 공간에서 **H=1**로 측정한다.

```text
H_E(A1) = J_E(absolute, H=1) − J_E(best_static)      [nat]
```

| H_E | 판단 |
|---|---|
| ≥ 1.0 nat (약 2.7배 loss) | GO |
| 0.3 ~ 1.0 nat | 조건부. 주 주장을 강건성으로 이동 |
| < 0.3 nat | 적응 제어 연구를 중단하거나 음성 결과로 정리 |

**"순간적" 헤드룸이다.** 전역 상한이나 장기 헤드룸이라고 부르지 않는다.
`absolute`는 H=3, 5 planning에 쓰지 않는다. 현재 damping과 무관하게 순간
이동하므로 damping ramp-up과 temporal credit assignment 자체를 제거하고,
따라서 장기 계획의 필요성을 묻는 게이트 C의 질문과 무관하다. 비용도 감당할
수 없다 (33 damping × 4 budget = 132 action이면 H=3 beam=4에서 실제 step당
약 1,200회 시뮬레이션).

**Gate A2 — Reachable sequential headroom (Track E)**

> 현실적인 multiplier action으로 그 이득에 접근할 수 있는가?

```text
H_E(A2) = max over {narrow, wide} of J_E(space, H=5) − J_E(best_static)
```

A1 대비 크게 낮으면 행동 공간 도달성이 병목이다. GO 기준 0.7 nat, 재설계 0.2 nat.

**Gate B — Action-space restriction**

```text
absolute  vs  wide multiplier  vs  narrow multiplier      (모두 H=1)
```

세 공간이 **같은 `3^e` 격자** 위에 있어야 이 비교가 성립한다. 해상도가 다르면
도달성 손실과 해상도 손실이 섞인다. 실제로 초기 구성(2 decade 간격)에서
`absolute`가 범위가 4배 넓은데도 `narrow`보다 나쁜 결과를 냈다.

```text
NARROW    {3^-1 .. 3^1}      3점
WIDE      {3^-3 .. 3^3}      7점
ABSOLUTE  {3^-16 .. 3^16}   33점
```

`ABSOLUTE` 범위 `[2.3e-8, 4.3e7]`은 optimizer 경계 `[1e-8, 1e8]` 안에 있다.
경계에서 클립되면 서로 다른 action이 같은 damping으로 붕괴한다.

**CG budget `{3, 5, 10, 20}`은 축소하지 않는다.** 부차적 하이퍼파라미터가 아니라
이 프로젝트의 핵심인 inexactness control 축이다. 특히 파일럿 발견이 "국소 효율
목적이 `k=3`을 과도하게 선호한다"는 것이므로, `k=3`을 삭제하면 문제를 해결하는
게 아니라 관찰된 현상을 action space 밖으로 숨기는 것이 된다.

계산량이 문제라면 순서는 이렇다.

```text
1. absolute 를 H=1 로 제한                    (적용됨)
2. planner 는 narrow / wide 만                 (적용됨)
3. beam width 민감도 측정 후 축소               (calibration)
4. 마지막 수단으로 CG budget 축 축소            (하지 않음)
```

**Gate C — 세 부분으로 분리 (D12에 따라 재정의)**

`H = 1, 3, 5` 비교는 D10에서 폐기했고, D10의 `C3 − C1` 단일 통계도 **중심
게이트에서 내렸다.** 그 통계는 fresh-quota MPC라는 **결함이 확인된 실행 방식**을
포함하고 있어서, 무엇을 재는지가 불분명하다.

주 컨트롤러는 `shrinking-quota MPC` 다. 세 컨트롤러의 역할이 다르다.

```text
fresh       시간 불일치가 확인된 진단 baseline. 주 결과에 쓰지 않는다
committed   계획 상한 / open-loop oracle. 초기 상태에 조건화된 oracle 이다
shrinking   실제 adaptive planning 후보. 주 컨트롤러
```

**Gate C1 — Time-consistency diagnostic**

```text
J_E(shrinking) − J_E(fresh)      같은 Q, beam, action space
```

> 쿼터를 매 step 초기화하는 것이 실제 성능을 떨어뜨리는가?

**연구 결과이지만 PPO 착수 게이트는 아니다.** 실측에서 크게 양수였다 (D12).

**Gate C2 — Sequential planning value**

```text
J_E(shrinking, Q>1) − J_E(C0)
```

> 시간 일관적인 다단계 재계획이 one-step 효율 제어보다 나은가?

GO 0.3 nat, 재설계 0.05 nat. **primary controller 는 `shrinking` 이다.**

**Gate C3 — Feedback value**

```text
J_E(shrinking) − J_E(committed)
```

> 계획을 고정 실행하는 것보다 상태를 관찰하며 재계획하는 것이 추가 이득인가?

| 결과 | 의미 |
|---|---|
| `shrinking ≈ committed` | 좋은 sequence는 존재하지만 feedback 자체의 추가 가치는 작다 |
| `shrinking > committed` | 재계획에 실질적 가치가 있다 |
| `shrinking < committed` | approximate replanning이 계획을 훼손한다 |

D12 실측은 대체로 첫 번째와 두 번째 사이였다.

**committed와 `open_loop`를 혼동하지 않는다.**

```text
committed planner   각 task의 현재 상태를 보고 비싼 탐색으로 sequence 생성
open_loop           상태 피드백 없이 미리 정한 schedule 실행
```

committed가 `open_loop`보다 좋다고 해서 feedback이 필요하다는 뜻이 **아니다.**
committed는 초기 상태에 조건화된 oracle에 가깝다. PPO 필요성은
`shrinking` vs `dev에서 튜닝한 open_loop` vs `현재 상태를 쓰는 heuristic` 으로
보아야 하고, 튜닝 비용(`N_tune`)을 동일하게 맞춰야 한다.

---

이하는 D10 쿼터 사다리 설명이다. 사다리 자체는 유지한다.

`c_max` 를 단일 action 최대 비용이라 할 때, planner마다 동일한 미래 GE 쿼터
`Q` 를 준다. `narrow` / `wide` 만 쓰고 `absolute` 는 제외한다 (A1 참조).

```text
C0  one_step_efficiency        비율 baseline
C1  budgeted MPC  Q = 1 × c_max
C2  budgeted MPC  Q = 2 × c_max
C3  budgeted MPC  Q = 4 × c_max
```

각 planner는 쿼터 안에서 여러 action을 선택할 수 있다. `Q = 1 × c_max` 는
"비싼 action 한 번"과 "싼 action 여러 번"을 같은 예산에서 겨루게 한다.

쿼터는 `shrinking` / `committed` / `fresh` 세 실행 방식에 **동일하게** 부여한다.
판정 통계는 위 C1 / C2 / C3 이며, `J_E(C3) − J_E(C1)` 형태의 단일 사다리 통계는
쓰지 않는다 (D12).

**개선이 있어도 `chosen_depth` 가 계속 1이면 GO 판정을 내리지 않는다.** 기여한
것이 planning이 아니라 늘어난 탐색량이기 때문이다.

함께 보고할 것:

```text
실제 episode 의 고정 GE terminal loss
쿼터별 개선량 (C1 → C2 → C3)
chosen_depth 분포와 최대 채택 depth
quota_used_fraction   쿼터를 실제로 얼마나 썼는가
depth_cap_hit         계산 상한에 걸린 step 비율
narrow 와 wide 의 차이
planner 분석 비용 (search_cost_ge)
action sequence 분포
```

`depth_cap_hit` 이 0이 아니면 계산 상한 때문에 쿼터를 다 쓰지 못한 계획이
있으므로 **사다리 비교가 훼손된 것으로 보고한다.** 조용히 넘기면 게이트 C
결론이 계산 예산의 부산물이 된다.

`C0 → C1` 차이도 별도로 보고한다. 이것은 "목적함수를 비율에서 고정 예산으로
바꾼 효과"이고 "예산을 늘린 효과"와 다르다. 섞으면 어느 쪽이 기여했는지
알 수 없다.

**beam 8을 planner 성능의 하한으로 명시한다.** D12 실측에서 SPD `Q=4` 결과가
beam 4↔8 사이에서 8 nat 이상 움직였다. 따라서 `Q=4` 수치는 안정된 효과 크기가
아니라 하한이다. beam을 더 키우는 것은 게이트 판정용이 아니라 **일회성 민감도
진단**으로만 하고, 범위를 `Q=4`, seed 1개, task 2개, `shrinking`/`committed`,
narrow로 제한한다. "좋은 결과가 나올 때까지 탐색폭을 늘렸다"는 인상을 피하고,
분석 성능보다 deployable controller 비교에 자원을 쓴다.

#### beam 8 승격 대상은 사전에 고정한다

**beam 4 결과를 보고 Q를 고르지 않는다.** beam 4에서 SPD `Q=4`가 −4.21 nat,
beam 8에서 같은 조건이 +4.59 nat였다. beam 4로 유망성을 판정하면 실제로 좋은
`Q=4`를 탈락시킨다. 따라서 승격 규칙을 실행 전에 못박는다.

```text
Q=1   beam 8 불필요. depth 1 만 가능하므로 beam 폭이 무의미하다
Q=2   narrow, wide 모두 beam 8 재평가
Q=4   narrow, wide 모두 beam 8 재평가
```

`fresh` 는 beam 8 단계에서 **전체 조합에서 제외한다** (`--fresh-seeds 0`).
진단 baseline이고 P1~P3 판정에 쓰지 않으며 탐색 비용이 가장 크다.

설정(Q, action space) 선택은 **beam 8 dev 결과의 median** 으로 한 번만 하고,
그 시점에 protocol freeze 한다. **어떤 median 인지는 D21 에서 못박았다**
(`shrinking` 자신의 Track E logΔ 를 challenge 12 인스턴스에서 median, 최대화).

#### `fresh` 에 계산을 과도하게 쓰지 않는다

`fresh` 의 시간 불일치는 이미 여러 조건에서 명확히 관측됐다 (D12). 전체 dev에
반복할 과학적 가치가 낮은 반면 탐색 비용은 가장 크다 (`Q=4` wide 에서
인스턴스당 1.4M GE). 필요한 것은 다음뿐이다.

```text
task 종류별 일부 seed 에서 fresh vs shrinking
C1 의 방향이 반복되는지 확인
fresh 가 왜 나쁜지 action / CG budget 분포 확인
```

`HeadroomConfig.fresh_diagnostic_seeds` (기본 1) 과 `run_fresh_wide`
(기본 `False`) 로 제한한다. `fresh` 의 seed 집합이 다르므로 **C1의 표본은 다른
게이트보다 작다.** paired 비교는 겹치는 seed에서만 이루어진다. P1~P3에는
`fresh` 가 들어가지 않으므로 이 비대칭이 판정을 오염시키지 않는다.

**실현 성능의 단조성을 가정하지 않는다.** MPC는 매 step 재계획하므로 큰 쿼터가
항상 좋다는 보장이 없다. 쿼터가 커지면 탐색 가능 집합이 포함관계로 커지므로
**최대 채택 depth** 는 감소할 수 없고, 그 성질만 테스트로 검증한다.

```text
탐색 가능 집합의 포함관계   ≠   실제 episode 성능의 비감소
```

**보조 분석: Lagrangian planner.** `U_β = Δlog L − β·Σc` 를 최대화하는 변종을
사전 고정 β 격자 `{0, 0.01, 0.03, 0.1, 0.3, 1.0}` 전체에서 돌린다. 특정 β 하나를
골라 주 결과로 쓰지 않는다. discrete action에서 Lagrangian 완화는 고정 예산
문제와 정확히 같지 않고(duality gap이 0이라는 보장이 없음), β 선택에 따라 결론이
뒤집힐 수 있다. 탐색과 가지치기는 `BudgetedMPCController` 와 동일하고 최종 선택
규칙만 다르므로, 목적함수 차이가 탐색 품질 차이와 섞이지 않는다.

**Gate D — Cost-to-target headroom (Track T)**

target 난이도별로 `best_static` 과 MPC planner의 GE-to-target, 도달률,
restricted mean을 비교한다.

```text
H_T(τ) = C_τ(best_static) / C_τ(mpc)      [배수]
```

Gate A와 결론이 다를 수 있다. **그 불일치 자체를 결과로 보고한다.**

**Gate E — Micro-neural transfer (Stage 2.5)**

가장 큰 미지 위험은 synthetic → 신경망 전이다. PPO 전에 확인한다.

- 아주 작은 2-layer MLP, MNIST 일부 샘플
- 짧은 horizon, 고정 배치와 확률적 배치 각각
- absolute / multiplier planner 모두

| 결과 | 판단 |
|---|---|
| synthetic 헤드룸 큼, neural 거의 없음 | PPO 중단. 연구 질문을 synthetic 수치해석으로 축소 |
| absolute 헤드룸 큼, multiplier 만 낮음 | 행동 공간 재설계 |
| H1 이 이미 충분히 좋음 | contextual bandit 우선 |
| look-ahead 만 좋음 | sequential RL 진행 근거 확보 |

**Gate F — Learnability (Stage 4 이후)**

```text
Recovery = (J_learned − J_static) / (J_reachable_planner − J_static)
```

분모는 `absolute` 가 아니라 **정책과 같은 행동 공간을 쓰는 reachable planner**다.
absolute를 분모에 두면 정책이 구조적으로 도달할 수 없는 부분까지 요구하게 된다.

#### 부수 산출물

MPC planner의 trajectory가 behavior cloning 데이터셋이 된다.
README 위험 2번(PPO 불안정)의 대응책으로 적힌 warm start를 여기서 얻는다.
`results/raw/planner/` 에 `(state, action)` 쌍으로 저장한다.

#### 재현성 확인 항목

파일럿에서 나온 "높은 damping은 CG를 쉽게 만들지만 최적화를 망친다"는 결과는
단일 quadratic 조건에서 관측됐다. 기여로 올리기 전에 다음에서 재현되는지 본다.

- 여러 condition number
- 여러 eigenvalue 분포 (log-spaced 외에 clustered, two-cluster)
- Rosenbrock
- 작은 MLP
- step_size 고정과 line search 각각

### Stage 3 — baseline 정면 비교  (3~4일)

Stage 2를 통과하면 README Milestone 2·3에 해당하는 작업을 수행한다.

- `optimizers/fixed_newton_cg.py` — step acceptance, safe fallback
- `optimizers/heuristic_newton_cg.py` — trust ratio 기반 damping 제어
- `benchmark/runner.py` — paired design, JSONL 로깅, 비용 회계
- 대상: `adamw`, `sgd_momentum`, `fixed`, `heuristic`, `best_static`, `open_loop`
- task: SPD/ill-conditioned quadratic, Rosenbrock, MNIST MLP
- seed 5개 이상, `B_c/B_g ∈ {1/4, 1/2, 1}`

게이트:

- MNIST에서 NaN 없이 100 step 완주, loss 지속 감소
- step별 HVP / GE / wall-clock / peak VRAM 기록
- heuristic이 최소 하나의 ill-conditioned task에서 fixed 대비
  failure rate 또는 cost-to-target 개선
- D5 결과 표 자동 생성

이 단계에서 RL 없이도 발표 가능한 결과가 나온다. RL 결과가 놓일 좌표계다.

### Stage 4 — RL 컨트롤러  (1~2주)

- `rl/state_features.py` — README §5.1 특징, running mean/std 정규화, NaN 가드
- `rl/rewards.py` — D3 보상
- `rl/environment.py` — `MultiDiscrete([3, 4, 3])`, termination/truncation 구분
- `rl/train_policy.py` — behavior cloning warm start → PPO

구현 주의사항:

- **병렬화**: Windows + `SubprocVecEnv` + CUDA는 프로세스별 CUDA 컨텍스트로
  8 GB VRAM을 소진한다. 모델이 작으므로 `DummyVecEnv` 로 8~16 env를 한 GPU에 올린다.
  quadratic 단계는 CPU가 더 빠를 수 있으므로 측정해서 결정한다.
- **보상 해킹 감시**: step size를 최소로 깔고 버티기, 의도적 step rejection으로
  실패 패널티 회피. action 히스토그램과 entropy를 매 학습 구간 로깅한다.
- 커리큘럼: random SPD quadratic → ill-conditioned/indefinite → Rosenbrock → MNIST MLP
- 체크포인트와 observation normalization 통계를 함께 저장한다 (둘 중 하나만 저장하면 평가 불가)

게이트:

- `gymnasium.utils.env_checker.check_env` 통과
- random policy로 여러 에피소드 실행 시 crash 없음, observation에 NaN/Inf 없음
- random policy 대비 평균 리턴 향상
- unseen quadratic에서 `fixed` 와 비교 가능한 결과 산출
- deterministic 평가 스크립트로 동일 결과 재현

### Stage 5 — 일반화와 분석  (1주)

meta-train과 meta-test 분포를 수치로 분리한다.

```text
meta-train
  quadratic : κ ~ LogUniform(1e1, 1e4),  d ∈ {50, 100, 200}
  MLP       : width ∈ {64, 128, 256},  init scale ~ Uniform(0.5, 2.0)
  batch     : B_g ∈ {256, 512}

meta-test
  quadratic : κ = 1e5,  d = 500              (조건수·차원 외삽)
  Fashion-MNIST MLP                          (데이터셋 전이)
  더 깊은 MLP (784-256-256-128-10)            (깊이 외삽)
  CIFAR-10 small CNN, 수백만 파라미터급        (구조 전이 + wall-clock 검증)
  B_g = 1024                                 (batch 외삽)
```

CIFAR-10 CNN을 넣는 이유는 정확도가 아니라 **D1의 wall-clock 단서 확보**다.
FLOP 지배 구간에서 결론이 유지되는지 확인해야 한다.

ablation (README §9): 최소 1~8번 전부 수행. 우선순위는
`progress 제거` > `Hessian feature 제거` > `HVP penalty 제거` > `단일 축만 제어`.

---

## 5. 컴퓨트 예산

추정치이며 Stage 1의 실측으로 교정한다.

| 단계 | 추정 소요 (RTX 3060 Ti) |
|---|---|
| Stage 1 마이크로벤치 + 단위 테스트 | 분 단위 |
| Stage 2 헤드룸 (36 config × 20 task × 50 step) | 수십 분 |
| Stage 3 baseline (7 optimizer × 5 seed × 4 task × 3 batch ratio) | 1~3시간 |
| Stage 4 PPO meta-training 1회 (100k env step, MNIST) | 1.5~3시간 |
| Stage 5 meta-test 전체 + ablation | 4~8시간 |

지배 비용은 Stage 4의 **재실행 횟수**다. 3060 Ti에서 meta-training을 무한 반복할 수
없으므로, quadratic에서 하이퍼파라미터를 굳힌 뒤 신경망 task로 올라가는 순서를 지킨다.
Stage 4 재실행이 5회를 넘어가면 contextual bandit 또는 supervised policy imitation으로
축소하는 것을 검토한다 (README 위험 1).

---

## 6. 가설을 반증하는 조건

무엇이 나오면 가설이 틀린 것으로 볼지 미리 적어둔다.

1. Stage 2 헤드룸 < 1.10 → 상태 기반 적응 제어의 여지가 없다
2. RL이 `open_loop` 를 이기지 못한다 → 학습된 것은 적응 제어가 아니라 스케줄이다
3. RL이 `best_static` 를 이기지 못한다 → 동일 탐색 예산에서 이득이 없다
4. iteration 기준으로는 이기지만 GE 기준으로는 진다 → 계산 비용이 이득을 잠식한다
5. meta-test에서 이득이 사라진다 → 정책이 task를 암기했다
6. ablation에서 Hessian feature 제거 후에도 성능이 유지된다 → curvature 신호를 쓰지 않았다

4·5·6은 실패가 아니라 보고 가치가 있는 결과다 (README §17 기준 4). 1·2·3은 설계
변경을 요구하는 실패다.

---

## 7. 최소 성공 기준 (README §17 구체화)

다음 중 하나를 만족하면 의미 있는 결과로 본다. 모두 D7 통계 프로토콜로 검정한다.

1. RL이 `fixed` 대비 동일 target까지 GE를 **10% 이상** 감소
   (기하평균 비율 ≤ 0.90, Wilcoxon p < 0.05, Holm 보정 후)
2. RL이 `heuristic` 대비 meta-test에서 failure rate를 유의미하게 감소
3. RL이 동일 GE 예산에서 더 낮은 loss 또는 더 높은 accuracy 달성
4. 우위가 없더라도, 어떤 조건(조건수 / batch regime / target 수준)에서
   `fixed`·`heuristic`·`AdamW` 중 무엇이 우세한지 재현 가능한 분석 제공

"AdamW를 항상 이긴다"는 성공 기준이 아니다.

---

## 8. 보고 산출물

권장 figure (README §18):

- loss vs optimizer step
- loss vs **cumulative GE** ← 주 그림
- loss vs wall-clock (오버헤드 지배 여부 표기)
- damping over time / CG budget over time
- trust ratio histogram
- task별 cost-to-target (도달률 병기)
- policy action heatmap (state feature 축 기준)
- `greedy_oracle` vs `best_static` vs `rl` 3자 비교 (헤드룸 대비 달성률)

마지막 그림이 이 프로젝트의 핵심 주장을 한 장에 담는다.

---

## 9. 변경 이력

| 날짜 | 변경 | 이유 |
|---|---|---|
| 2026-08-01 | 초판. D1~D8 확정, Stage 0~5 정의, target 사전 등록 | — |
| 2026-08-01 | D1 예산 계산을 실측값으로 교정 (26 GE → 17 GE, 24 step → 35 step) | 이론 계수 대신 RTX 3060 Ti 실측 사용. MNIST MLP 오버헤드 지배 확인 |
| 2026-08-01 | 비용 모델 실측을 Stage 1 → Stage 0 으로 이동 | Stage 0에서 이미 완료했고, 이후 모든 지표가 여기에 의존 |
| 2026-08-01 | Stage 1 완료. negative curvature 판정을 상대 기준으로 변경 | 절대 임계값은 수렴 구간에서 오탐. RL 상태 특징을 오염시킬 수 있었다 |
| 2026-08-01 | indefinite quadratic을 진단 전용으로 명시 | 아래로 유계가 아니므로 cost-to-target 과 log 보상이 정의되지 않는다 |
| 2026-08-01 | `max_damping` 1e3 → 1e8, damping을 로그공간 지속 상태로 | Stage 1에서 κ=1e6이 damping ~1e6을 요구함이 확인됨. 이전 값은 그 자체로 병목 |
| 2026-08-01 | damping 배수를 정확한 역수쌍으로 (`0.3` → `1/3`) | `3 × 0.3 = 0.9` 라 배수를 번갈아 고르면 damping이 step당 10% 아래로 표류. `1/3` 이면 `3 × (1/3) = 1` 로 표류 없음 |
| 2026-08-01 | **D9 신설: 실험을 Track E / Track T로 분리** | 파일럿에서 `Δlog L / cost` 목적의 컨트롤러가 cost-to-target에서 best_static보다 나빴다(0.967x). 국소 효율 최대화와 총비용 최소화는 다른 문제다 |
| 2026-08-01 | **D10 신설: planner 목적함수를 비율 → 고정 GE 쿼터로 교체. 게이트 C를 `H=1/3/5` → 쿼터 사다리 `C0~C3` 로 재정의** | dry run에서 `H=3`과 `H=5`가 완전히 동일하고 `depth≥3`이 채택 0회였다. 누적 비율 효용은 step별 rate의 가중 평균이므로(mediant 부등식) depth 1 incumbent가 지나치게 강해져 "지금 손해, 나중에 이득"을 표현할 수 없다. 이 목적함수 아래의 음성 게이트 C는 증거가 아니라 항진명제에 가깝다 |
| 2026-08-01 | `HorizonPlannerController` → `AverageRateEfficiencyPlanner` 로 개명, 진단 baseline으로 보존 | 버그가 아니라 푸는 문제가 달랐다. "RL 보상을 ratio로 설계하면 생기는 함정"의 증거로 별도 보고 |
| 2026-08-01 | Beam pruning을 비율 스칼라 → `(used_GE, terminal_loss)` Pareto + GE cost bucket으로 교체 | 비율로 정렬하면 mediant 문제가 가지치기 안에서 재발한다. 비싼 장기 계획이 싼 단기 계획과 섞여 조기 탈락한다 |
| 2026-08-01 | Track T planner 선택 규칙을 cost-to-go 추정 → lexicographic(도달 여부 → 누적 GE)으로 교체 | 임의의 실패 벌점이나 비율 없이 "도달이 우선, 비용이 그다음"을 순서로 표현한다 |
| 2026-08-02 | **D11 신설: Track E를 예산 초과 step 절단 후 평가** | `spent >= budget` 종료 규칙 때문에 마지막 step이 예산을 넘고, 초과량이 action 크기에 비례한다. 150 GE 예산에서 C0는 171 GE, Q=4는 154 GE를 썼다. 큰 step을 고르는 컨트롤러가 공짜로 11% 예산을 더 쓰는 편향이 게이트 C 결론과 같은 방향으로 섞여 있었다 |
| 2026-08-02 | **D12 신설: 계획의 가치와 실행 방식을 분리. committed / fresh-quota / shrinking-quota 3종 비교** | D10 쿼터 사다리의 음성 결과가 목적함수나 탐색 때문이 아니라 **fresh-quota 실행 방식의 시간 불일치** 때문임이 확인됐다. `committed > C0`, `shrinking ≈ committed`, `fresh < committed`. 최선 조건에서 planning이 C0를 SPD +4.59 nat, ill +0.91 nat 앞선다 |
| 2026-08-02 | shrinking 쿼터 차감을 예측 비용 → `context.previous.cost_ge` 실제 비용으로 교체 | 예측 비용은 CG 조기 수렴을 반영하지 못해 window가 일찍 닫힌다. SPD beam8 Q=4에서 shrinking이 48.90 → 56.73으로 바뀌었다 |
| 2026-08-02 | **게이트 C를 C1(time-consistency) / C2(sequential planning value) / C3(feedback value)로 분리. 주 컨트롤러를 `shrinking-quota MPC`로 지정** | 기존 `C3 − C1` 단일 통계는 결함이 확인된 fresh-quota 실행 방식을 포함하므로 무엇을 재는지 불분명하다. protocol freeze 전 pilot 단계이므로 사후 조작이 아니다. D12 결과는 pilot diagnostic으로 보존 |
| 2026-08-02 | **PPO 착수 조건을 Gate C에서 분리해 P1~P4로 신설** | Gate C는 planner 오라클 간 비교이고, PPO는 그 오라클을 저비용 정책으로 amortize 할 수 있는지를 묻는 별도 질문이다. P2(baseline superiority)와 P4(micro-neural)가 없으면 open-loop schedule로 설명되는 이득을 RL 성과로 오인할 수 있다 |
| 2026-08-02 | object-level cost와 decision-search cost를 용어로 분리 | planner 탐색비가 본문의 500~2,600배다. planner를 실용 optimizer로 제시할 수 없고, 헤드룸 측정 장치로 위치를 명시해야 한다 |
| 2026-08-02 | beam 16 전체 실행 보류. beam 8을 planner 성능의 **하한**으로 명시 | SPD `Q=4` 가 beam 4↔8 사이에서 8 nat 이상 움직였다. 그러나 탐색폭을 계속 늘리면 deployable controller 비교가 늦어지고 "좋은 결과가 나올 때까지 늘렸다"는 인상을 준다. 민감도 진단은 `Q=4`·seed 1개·task 2개로 제한 |
| 2026-08-03 | **beam 8 승격 대상을 사전 고정: `Q ∈ {2, 4}` × {narrow, wide} 전부** | beam 4 결과로 Q를 선별하면 사후 선택이 된다. beam 4에서 −4.21 nat였던 SPD `Q=4`가 beam 8에서 +4.59 nat였으므로, beam 4 판정은 좋은 설정을 탈락시킬 수 있다 |
| 2026-08-03 | `fresh` 를 seed 부분집합 + narrow 로 제한 (`fresh_diagnostic_seeds=1`, `run_fresh_wide=False`). beam 8 단계에서는 전체 제외 | 진단 baseline이고 P1~P3 판정에 쓰지 않는데 탐색 비용이 가장 크다 (`Q=4` wide 인스턴스당 1.4M GE). 시간 불일치는 이미 여러 조건에서 확인됐다. C1의 표본이 작아지는 것은 판정에 영향이 없다 |
| 2026-08-03 | **D13 신설: 실험 정체성을 `run_semantics_id` 와 `sweep_id` 로 분리** | D8의 단일 해시가 두 역할을 겸하고 있었다. diagnostic arm의 커버리지만 바꿨는데 baseline 320 run이 무효화됐다. sweep 설정이 바뀌어도 run semantics가 같으면 재사용해야 한다. beam 8 실행 전에 적용 |
| 2026-08-03 | 게이트 C1과 C2·C3의 표본 크기를 분리 보고하도록 명시 | `fresh` 가 seed 부분집합에만 있으므로 C1의 paired n이 작다. 한 표에 넣으면 같은 신뢰구간처럼 보인다 |
| 2026-08-03 | **D14 신설: log improvement에 상대 수치 하한. `final_loss <= 0` 을 조용히 제외하지 않는다** | `rosen_d2` 에서 `onestep_absolute` / `heuristic` 이 `final_loss = 0.0` 이라 각 3쌍이 기록 없이 빠졌고 게이트 A1·B가 낮게 잡혔다. `finfo.tiny` 를 floor 로 쓰면 최대 logΔ 가 708 nat 이 되어 underflow 여부가 통계를 지배하므로 `100 x eps` 상대 floor 를 쓴다. 재집계에서 세 게이트의 결론이 포화 처리에 따라 뒤집혔다 |
| 2026-08-03 | **D15 신설: 재계획이 계획을 실제로 바꿨는지 행동 내용으로 계측** | `Q1` 에서 `shrinking` 과 `committed` 가 bitwise 같은 결과를 냈다. alias 가 아니라 실제 동률이며, `suffix_retention_rate` 로 확인한다. `chosen_depth` 히스토그램만으로는 깊이만 같고 내용이 다를 수 있다 |
| 2026-08-03 | **D16 신설: baseline 선택 과정을 `SelectionManifest` 로 기록** | `best_static` / `best_open_loop` 는 컨트롤러가 아니라 튜닝 결과인데 raw 에 후보 라벨만 남아 A1·A2를 재집계할 수 없었다. 라벨만 바꾸면 선택 근거가 사라지고 evaluation 결과로 역추정하게 되어 사후 선택이 된다 |
| 2026-08-03 | `git_commit` 과 `code_dirty` 를 `sweep_id` 에서 제거해 `execution_provenance` 로 분리 | `sweep_id` 가 "어떤 run 집합을 요청했는가" 를 뜻한다면 문서 수정으로 ID 가 달라지는 것은 의미가 어긋난다. 어떤 ID 에도 넣지 않는다 |
| 2026-08-03 | **D17 신설: open-loop `progress` 를 `step/total_steps` → 소모 GE 비율로 교체** | GE 예산으로 종료하는데 breakpoint 가 step 비율이라 4구간 스케줄의 첫 구간만 실행됐다. `best_open_loop` 이 `best_static` 과 9쌍 전부 bitwise 동일(CI `+0.000~+0.000`)했던 것은 퇴화가 아니라 구현 결함이다. 수정 후 4/4 구간 실행, median logΔ 9.337 → 25.456 |
| 2026-08-03 | `OPEN_LOOP_SEMANTICS_VERSION` 을 open-loop payload 에만 넣어 격리 | open-loop 108 run 만 재실행되고 static / heuristic / one-step 의 `run_semantics_id` 와 `selection_id` 는 유지됐다. D13 3계층 분리가 의도대로 작동한 사례 |
| 2026-08-03 | P2 strongest baseline 을 단순 median 최대값으로 정하지 않도록 명시 | `onestep_absolute` 와 `heuristic` 이 둘 다 31.438 인 것은 floor ceiling 영향이다. baseline 간 순위를 현재 dev 표본으로는 안정적으로 구분할 수 없다 (CI −5.982~+24.346, p=0.906) |
| 2026-08-03 | **D18 신설: bridge 검증 tolerance(`1e-12`/`1e-14`)와 분류를 실행 전 고정. bridge 통과** | expected 72쌍 전부 bitwise EXACT, `MISMATCH` 0. D13/D16/D17 이 planner 실행 궤적을 바꾸지 않았음을 확인. 범위 밖(`LEGACY_ONLY_OUT_OF_SCOPE`)과 누락을 구별하지 않으면 범위를 좁힐 때 통과가 불가능해진다 |
| 2026-08-03 | **D19 신설: 포화를 task 이름 대신 `floor_hit` 으로 판정. `rosen_d2 제외 = primary` 정의 폐기** | `quad_spd` 도 floor 아래로 내려가 delta 가 0이었다(trajectory 는 전부 다름). joint saturation(구분 불가)과 one-sided saturation(명확한 차이)은 다르다. `drop_saturated_pairs` 를 primary 로 쓰면 one-sided 쌍의 좋은 결과를 삭제한다 |
| 2026-08-03 | spec 별 보고가 all-task median 과 반대 결론을 냈다 | all-task 는 A2/C2/C3 모두 `+0.000`(재설계)인데, `quad_ill κ=1e5` 만 보면 `+0.494`/`+0.542`/`+0.370` 으로 GO 기준을 넘는다. 측정 가능한 regime 이 하나뿐이라는 것이 핵심 발견이다 |
| 2026-08-03 | **D20 신설: challenge set 을 측정 가능성 기준으로 사전 등록. quadratic κ∈{1e3,1e4,1e5,1e6} freeze** | dev subset 3개 중 2개가 포화되어 beam 8 을 기존 9쌍만으로 돌리면 seed 하나에 좌우된다. 선정에 planner 결과를 쓰지 않고 baseline-only 로 판정했다. `rosen_d5` 는 tie-break 규칙상 탈락, 비선형 진단 층으로 보존 |
| 2026-08-03 | `rosen_d5` 에 사후 `baseline spread` 조건을 추가하지 않음 | 임계값 선택이 사후적이 되고, baseline 4종이 동일하다고 planner 도 동일하다는 보장이 없다. 기존 tie-break 규칙만으로 같은 결론에 도달한다 |
| 2026-08-03 | challenge set 을 `--mode challenge`, 진단 층을 `--mode nonlinear-diagnostic` 으로 코드에 등록. `CALIBRATION_SEEDS`/`SELECTION_SEEDS` 신설 | 목록을 코드 밖에 두면 사전 등록이 무의미해진다. `TestChallengeSetFreeze` 가 spec 4개, `log10(κ)` 간격, seed 서로소, `phase` 가 `run_semantics_id` 를 바꾸지 않음을 검증한다 |
| 2026-08-03 | `HELD_OUT_SEEDS` 를 권고 범위 5~14 로 바꾸지 않고 100~109 유지 | 이미 calibration(0,1) 및 selection(2,3,4) 과 서로소다. 바꾸면 기존 confirmatory 정의가 흔들린다 |
| 2026-08-03 | `(quad_d100_k1e5, seed 2)` 가 original dev audit 과 challenge selection 에 모두 포함됨을 명시 | 숨기지 않고 기록한다. 완전 분리에는 `SELECTION_SEEDS=(3,4,5)` 가 필요하지만 사전 등록 범위를 벗어난다. selection 12 인스턴스 중 1개이고 beam 8 조합은 아직 미실행이다 |
| 2026-08-03 | **D21 신설: 설정 선택 통계를 `shrinking` 자신의 median logΔ 최대화로 확정** | 기존 규칙("beam 8 dev median")이 무엇의 median 인지 미지정이었다. D19 에서 all-task 와 spec별 median 이 반대 결론을 냈으므로 남겨두면 사후 선택이 된다. baseline delta 로 고르지 않는 이유는 strongest baseline 순위가 불안정하기 때문이다 (p=0.906) |
| 2026-08-03 | D21 tie-break 사다리 고정: `decision-search GE` → `작은 Q` → `narrow` | median 이 `0.05 nat` 이내면 적용한다. 결정론적이어야 재현 가능하다 |
| 2026-08-03 | beam 8 전체 실행 전에 1 인스턴스 dry run 을 먼저 돌렸음을 공개 | 비용 측정 목적(`quad_d100_k1e3`, seed 2, 30 run, 약 8분). `Q × space` 별 planner 순위는 열지 않았으나 순서가 `dry run → 규칙 확정 → 전체 실행` 이었다는 사실을 기록한다 |
| 2026-08-04 | **D22 신설: beam 8 challenge 결과. `shrinking_Q4_narrow` 선택 (median logΔ 10.5306, 단독)** | 360 run, 실패 0, floor hit 0. `A2=+1.502 nat, p=0.0005, 12/12 양수`. `Q4_wide` 가 `Q4_narrow` 보다 낮다 (탐색 가능 집합 포함관계 ≠ 실현 성능 비감소) |
| 2026-08-04 | **`C3 = −0.044 nat (p=0.38)`. 헤드룸은 feedback 이 아니라 sequence 에 있다** | 초기 상태에서 한 번 계획하고 맹목 실행하는 `committed` 가 매 step 재계획과 같거나 낫다. PPO 가 학습하는 `π(a|s)` 의 추가 가치가 이 task 족에서는 0 이다. 다만 quadratic 이 결정론적이라 planner 모델이 정확했던 것과 구별할 수 없다 |
| 2026-08-04 | κ 의존성이 가설과 반대. `κ=1e3` 에서 헤드룸 최대(+6.584), `κ=1e5` 까지 감소(+0.735) | pilot 의 "ill-conditioned 에서만 헤드룸" 은 비교 대상이 포화됐기 때문이었다. κ 축을 채우니 방향이 뒤집혔다. D19/D20 교정이 이 관측을 가능하게 했다 |
| 2026-08-04 | `B = 재설계 (+0.015 nat, p=0.91)`. 행동 공간은 병목이 아니다 | `absolute` 132 action(log10 범위 15.27)이 `narrow` 12 action(범위 0.95) 대비 이득이 없다. 값싼 좁은 multiplier 공간을 정당화하는 음의 결과다 |
| 2026-08-04 | **D23 신설: `rosen_d5` 는 국소최소점에 갇힌 task. D20 ceiling 공식이 틀렸다** | 표준 시작점 basin 에 strict 국소최소점(`loss=3.930839434`, `\|grad\|=1e-8`, PSD)이 있다. D20 이 쓴 `ceiling=log(L0/numerical_floor)=31.44` 는 전역최소점 도달을 가정한다. 실제 달성 가능 상한은 `1.8175 nat` 이고 모든 baseline 이 정확히 거기 도달했다 |
| 2026-08-04 | `rosen_d5` 의 `n=3` 이 실제로는 `n=1` 이었다 | `RosenbrockSpec(dimension=5)` 의 `randomize_start` 기본값이 `False` 다. seed 2/3/4 의 24개 run 이 모든 컬럼에서 bitwise 동일했다. 이 진단의 CI 와 p-value 는 무의미하다 |
| 2026-08-04 | `randomize_start=True` 로 바꿔도 cap 은 남는다 | 세 randomized 시작점 모두 같은 국소최소점으로 수렴한다 (`start_noise=0.1` 이 basin 을 벗어나기에 작다). 달성 가능 ceiling 이 1.575~2.887 nat 에 불과하다 |
| 2026-08-04 | 교정된 eligibility 조건을 **제안만** 하고 적용하지 않음 | `achievable_ceiling = log(L0) − log(L_ref)`, `L_ref` 는 강한 참조 solver 의 수렴점. 기전 기반이고 컨트롤러 비교를 열지 않고 계산 가능하다. 다만 새 비선형 task 설계는 프로토콜 변경이므로 리뷰 대상이다 |
| 2026-08-04 | quadratic challenge set 은 D23 의 영향을 받지 않음 | seed 마다 다른 인스턴스이고 전부 SPD(`eig min = +1.0`)로 최소점이 유일하다. D22 의 `n=12` 결과는 유효하다 |
| 2026-08-04 | **D24 신설: `shrinking_Q4_narrow` freeze, PPO 보류, 연구 질문을 Q1/Q2 로 분해** | 헤드룸이 feedback 이 아니라 sequence 에 있다. `C3` 불충족. PPO 는 `π(a\|s)` 를 학습하는데 상태 조건의 추가 가치가 0 이면 고정 schedule 암기가 된다. micro-neural 에서 한 번 더 검증한 뒤 결정한다 |
| 2026-08-04 | beam 8 결과의 역할을 configuration selection 과 가설 정교화로 제한 | D21 의 순서 이탈(dry run 게이트 표를 본 뒤 규칙 확정) 때문에 완전한 confirmatory 로 부를 수 없다. 최종 효과 추정은 사전 고정 설정으로 held-out seed 에서 한다 |
| 2026-08-04 | Rosenbrock 을 계속 고치지 않는다 | start noise 를 키우거나 차원을 바꾸며 쓸 만한 버전을 찾는 것은 결과를 본 뒤 benchmark 를 조정하는 모양이 된다. `rosen_d2` 는 floor 진단, `rosen_d5` 는 국소최소점 진단으로만 남긴다 |
| 2026-08-04 | **D25 신설: eligibility 를 참조 solver panel 의 `J_achievable` 로 판정** | 단일 solver 의 수렴점을 절대 상한으로 쓰면 그 solver 의 약점이 상한으로 굳는다. `L_ref = min over panel`. planner 는 panel 에 넣지 않는다 (D20). `J_achievable` 은 calibration 지표이며 컨트롤러 점수가 아니다 |
| 2026-08-04 | D25 에 참조 산포 조건과 seed 복제 검사 신설 | 참조 solver 가 `0.5 nat` 이상 갈리면 상한 추정이 불안정하다. seed 복제는 `initial_loss` 와 시작점 벡터를 함께 봐 자동으로 잡는다. 소급 적용하니 `rosen_d5` 는 두 사유로 탈락하고 challenge quadratic 4종은 그대로 통과한다 |
| 2026-08-04 | **micro-neural task 신설. 두 regime 으로 `C3` 를 다시 잰다** | 핵심 질문은 "비선형인가" 가 아니라 "초기 계획 시점에 미래 상태를 예측할 수 없는가" 다 (D24). `full_batch` 와 `controlled_stochastic` 이 **같은 모델·데이터**를 쓰고 optimizer 표본만 다르다 |
| 2026-08-04 | optimizer 에 control loss / evaluation loss 분리와 `advance_batch` 훅 추가 | `curvature_loss()` 가 gradient·HVP·수락 판정을, `loss()` 가 Track E 점수를 담당한다. batch 는 **실제 step 뒤에만** 전진한다. planner 가 미래 batch 를 보면 데이터 oracle 이 되어 feedback 검증이 무의미해진다 |
| 2026-08-04 | `OPTIMIZER_SEMANTICS_VERSION` 을 올리지 않음 | 결정론적 task 는 `loss() == curvature_loss()` 이므로 bitwise 동일하다. `tests/test_control_vs_eval_loss.py` 가 `trace.final_loss == records[-1].train_loss_after` 를 컨트롤러 4종 x task 2종에서 검증한다. 마지막 기록이 비유한값이면 기존 동작을 유지해 실패가 숨지 않게 한다 |
| 2026-08-04 | `spec_kind_label` 을 duck-typing 에서 명시 타입 분기로 변경 | `getattr(spec, "kind", None)` 이 없으면 `"rosenbrock"` 으로 떨어뜨렸다. 새 spec 을 추가하면 조용히 Rosenbrock target 을 쓰게 된다. `MicroNeuralSpec` 이 실제로 그 경로에 걸렸다 |
| 2026-08-04 | micro-neural 의 `data_key` 를 `instance_id` 와 분리 | 초판은 데이터 생성에 `instance_id` 를 썼고 거기에 regime 이 들어가 **두 regime 이 다른 데이터셋**을 받았다. `C3` 의 regime 간 비교가 데이터 차이에 오염된다. 테스트가 잡았다 |
| 2026-08-04 | held-out 실행 중 테스트와 lint 를 병행 실행했음을 기록 | `wall_clock_sec` 이 일부 run 에서 부풀 수 있다. 단일 스레드 float 연산은 부하와 무관하게 결정론적이므로 수치 결과와 어떤 판정에도 영향이 없다. `concurrent_processes=1` 기록은 이 점에서 부정확하다 |
| 2026-08-04 | **D26 신설: held-out confirmatory `n=40`. `C2` 가 조건부 → GO 로 승격** | `+0.251 (p=0.077)` → `+0.456 (p=0.0000)`, CI 하한 `+0.254`. `depth>1` 채택률 `0.84`, `cap 0.00` 이므로 P3 두 조건 충족. 다단계 lookahead 는 one-step greedy 보다 실제로 낫다 |
| 2026-08-04 | **`C3` 의 CI 가 좁아졌다** | `+0.010 nat`, CI `[−0.033, +0.053]`, `n=40`, 21승 1무 18패. **equivalence margin 을 사전 등록하지 않았으므로 "효과가 0.053 nat 보다 작다" 를 검정 결과로 주장하지 않는다.** "CI 가 좁게 0 을 포함했으므로 실용적으로 큰 이득이 관측되지 않았다" 로 쓴다 |
| 2026-08-04 | 사다리별 측정값을 **합으로 분해하지 않는다** | 상수 대비 `+0.395 / +1.155 / +2.090 / +1.690`, 직접 측정 증분 `+0.456 / +0.010`. 쌍별 median 은 선형이 아니다. 초판이 `A2 − C2 = 1.233` 으로 파생한 것을 직접 측정값 `+1.155` 로 정정했다 |
| 2026-08-04 | `C2` 승격이 PPO 판단을 바꾸지 않음 | PPO 가 학습하는 것은 `π(a\|s)` 인데 그 추가 가치에서 실용적으로 큰 이득이 관측되지 않았다. D24 의 보류 결정을 유지한다 |
| 2026-08-04 | **D25 초판 구현 오류 수정: 다중 초기화가 `L_ref` 에 들어가 상한을 올렸다** | `rosen_d5_rand` 가 `lbfgs@init0.9` 의 `4.8e-21` 덕에 통과했다. 컨트롤러는 task 시작점에서만 출발하므로 다른 basin 의 최적값은 도달 가능한 상한이 아니다. `from_task_start` 로 분리하고 `start_basin_is_suboptimal` 진단을 추가했다 |
| 2026-08-04 | D25 소급 적용 확정 | `rosen_d5`(여유 −0.00) 와 `rosen_d5_rand`(여유 0.00) 탈락, quadratic 4종과 micro-neural 2종 통과. Rosenbrock 계열의 여유가 정확히 0 인 것이 D23 진단과 일치한다 |
| 2026-08-04 | `reference_spread_nat` 을 수렴한 run 만으로 계산 | micro-neural 에서 Adam/SGD 가 예산 안에 수렴하지 못해 산포가 32 nat 로 나왔으나 LBFGS 가 하한에 도달했으므로 상한 추정은 모호하지 않았다. "느린 solver" 와 "다른 임계점에 갇힌 solver" 를 구별해야 한다 |
| 2026-08-04 | `quad_k1e6` 에서 `newton` 이 `lbfgs` 를 크게 이겼다 | `lbfgs 1.0e-06 (미수렴)` vs `newton 3.1e-32 (수렴)`. `lbfgs` 만 썼다면 `J_achievable` 을 `28.3 nat` 로 과소평가했을 것이다. 단일 solver panel 을 금지하는 실측 근거다 |
| 2026-08-04 | `label_noise` 가 floor saturation 을 막는다는 초판 주장을 철회 | 서로 다른 `x` 에 붙은 뒤집힌 라벨은 과매개화된 신경망이 암기할 수 있다. 실측에서 `onestep` 이 정확도 1.000 에 도달했다. 측정 가능성은 참조 solver panel 로만 판정한다 |
| 2026-08-04 | **D27 신설: micro-neural 두 regime. 모델 정확도가 축이다** | R1(결정론적): `A2=+15.18`, `C3=−0.10`. R2(minibatch): `A2=−0.28`, `C2=−0.72`, `C3=+1.67`. 다단계 계획은 planner 모델이 정확할 때 큰 가치가 있고 부정확해지면 가치가 사라진다 |
| 2026-08-04 | **R2 의 `C3 > 0` 은 feedback 이 아니라 `committed` 붕괴 때문** | `committed` 가 `20.491 → 1.086` 으로 떨어져 `best_static`(3.029)보다도 나쁘다. 거절률이 `0.00 → 0.66` 이다. 교훈은 "feedback 정책을 학습하라" 가 아니라 "낡은 계획을 고수하지 말라" 다 |
| 2026-08-04 | R2 는 PPO 를 지지하지 않음. D24 보류 결정 유지 | R2 에서 planner 는 tuned 상수보다(−0.28), 1-step greedy 보다(−0.72) 나쁘다. 탐색 비용은 예산의 1,835배다. R2 최선은 값싼 `onestep`(탐색 1,879 GE, planner 의 1/146) 이다 |
| 2026-08-04 | 상태 조건 제어 자체는 R2 에서도 유효함을 구별해 기록 | `onestep 3.468` vs `best_static 3.029` = `+0.44 nat`. 그러나 그것은 이미 1-step greedy 가 하고 있고 다단계 계획이나 학습 정책을 필요로 하지 않는다 |
| 2026-08-04 | `_accept` 의 단조 감소 규칙이 R2 결과의 교란 요인임을 명시 | 코드 주석이 이미 경고했다. `committed` 의 거절률 `0.66` 은 계획 노후화와 엄격한 수락 규칙이 섞인 값이다. 방향은 바뀌지 않을 것으로 보이나(`shrinking` 은 거절률 0 인데도 `onestep` 보다 나쁘다) **크기를 주장하지 않는다** |
| 2026-08-04 | micro-neural 한계를 명시 | regime 당 `n=3` 이므로 CI 와 p-value 를 인용하지 않는다. 모델·데이터·`batch_size` 각 1종이며 "모델 부정확도" 축을 스캔하지 않았다 |
| 2026-08-04 | 비선형 진단 실행 시 git 이 dirty 였음을 기록 | 프로토콜 문서를 수정한 상태에서 돌렸다. D13 에 따라 `run_semantics_id` 는 영향받지 않고 `code_dirty` 는 `execution_provenance` 에 남는다 |
| 2026-08-01 | **D3 보상을 트랙별로 재정의. per-step ratio 보상 폐기** | ratio 보상은 정책이 `k=3` 같은 싸고 작은 행동만 반복하게 만든다. Track E는 additive log 감소, Track T는 `-cost` + target 종료 |
| 2026-08-01 | `greedy_oracle` → one-step efficiency controller, `lookahead_oracle` → H-step MPC planner | 전역 상한이 아니다. 실제로 고정 설정보다 나쁠 수 있음이 확인됐다 |
| 2026-08-01 | D6에 target 난이도 3단계와 pilot/confirmatory 분리 추가 | target 하나면 그 값 선정이 결론을 좌우한다. 결과를 본 뒤 예산을 고치면 사후 선택이 된다 |
| 2026-08-01 | D5에 탐색 예산 동일화 규칙 강화 | 파일럿에서 open_loop 랜덤 서치 12회가 static 전수 12개와 동일한 결과를 냈다. 스케줄 공간을 사실상 탐색하지 못했다 |
| 2026-08-01 | Stage 2 게이트를 A~F로 재정의, Stage 2.5(micro-neural)를 Gate E로 편입 | 단일 헤드룸 게이트로는 도달성·해상도·시간축 가치가 구분되지 않는다 |
| 2026-08-01 | D1의 wall-clock 주장 범위를 명시적으로 축소 | 단일 GPU 관측을 "wall-clock 논문 불신"으로 일반화할 수 없다. calibration finding으로 위치를 낮춤 |

### PPO 착수 조건 (명시)

**PPO 착수는 Gate C 하나로 결정하지 않는다** (D12). Gate C는 planner 오라클
사이의 비교이고, PPO는 "그 오라클을 저비용 정책으로 amortize 할 수 있는가"를
묻는 별도 단계다. 아래 P1~P4를 **모두** 통과할 때만 Stage 4를 시작한다.

주 컨트롤러는 `shrinking-quota MPC` 다. `fresh-quota` 는 진단 baseline이므로
어떤 P 게이트에도 쓰지 않는다.

**P1. Sequential headroom**

dev subset에서 `shrinking` 이 `C0` 보다 개선한다.

```text
median[ J_E(shrinking) − J_E(C0) ] >= 0.3 nat
```

그리고 paired bootstrap CI가 심하게 음수 영역을 포함하지 않아야 한다.

**P2. Baseline superiority**

```text
J_E(shrinking) > max( J_E(best_static), J_E(open_loop),
                      J_E(heuristic),   J_E(C0) )
```

전부를 크게 이길 필요는 없지만, **적어도 단순 open-loop schedule로 같은 이득이
설명되지 않아야 한다.** `open_loop` 와 `heuristic` 의 튜닝 예산(`N_tune`)을
`shrinking` 과 동일하게 맞춘다.

**P3. Multi-step usage**

```text
chosen_depth > 1 이 의미 있는 비율로 발생
Q=1 보다 Q>1 에서 개선
채택된 action sequence 가 단순 반복 패턴만은 아님
```

`Q=1` 은 depth 1만 가능하므로 여기서 이득이 없어야 "다단계"가 원인이라고 말할 수
있다.

**P4. Micro-neural transfer**

작은 MLP에서도 measurable headroom이 있어야 한다. **synthetic quadratic에서만
통과하면 PPO 구현을 연구의 핵심으로 밀지 않는다.**

---

행동 공간 전제로 Gate B를 함께 확인한다. 병목이면 고친 공간을 먼저 확정한다.

하나라도 실패하면 contextual bandit 또는 supervised policy imitation으로 축소하고,
그 판단 근거를 결과로 보고한다. 특히 `committed` 만 좋고 `shrinking` 도 실패하면,
연구 질문은 RL보다 다음에 가깝다.

> 좋은 계산 배분 schedule이 존재하지만, 상태 피드백을 이용한 재계획은 이를
> 안정적으로 실행하지 못한다.

#### 효과 크기의 안정성을 먼저 확인한다

D12 진단은 task 2개, seed 1개다. 그 결과는 **"P1·P2 충족 가능성을 관측했다"**
이며 "충족했다"가 아니다. dev subset 전체에서 paired 통계로 확인한 뒤에만
P 게이트 판정을 기록한다.
