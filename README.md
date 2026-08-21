# When Does Feedback Help?

### Hessian-free Newton 제어에서의 planning 과 model mismatch

**한국어** · [English](README.en.md)

Hessian-free Newton--CG 안에서 컨트롤러가 damping 과 conjugate-gradient 예산을 배분할 때,
그 이득이 **planning** 에서 오는지 **실행 중 feedback** 에서 오는지를 분해한 연구다. 코드,
행 단위 결과, 원고 소스가 모두 들어 있다.

> **문서의 지위.** 학부생이 쓴 **미심사 technical report** 이며, 그 자체로 완결된 산출물로
> 공개한다. **peer review 를 받지 않았고, 출판되지 않았고, 어떤 preprint 서버에도 없다.**
> 검증된 학술적 기여가 아니라, 주장을 직접 확인할 수 있게 만들어 둔 공학·측정 연습으로
> 읽어 주기를 바란다. 원고 집필을 포함해 전 과정에서 AI 를 사용했다. 무엇을 맡기고 무엇을
> 맡기지 않았는지는 [AI 사용 공개](#ai-사용-공개)에 있다.

주목할 부분은 효과 크기가 아니다. **여기 있는 모든 수치를 공개된 행 단위 데이터에서 몇 분
안에 다시 계산할 수 있고**, 모든 주장이 근거와 함께 등록돼 있고, 이 연구가 못 미친 지점을
덮지 않고 적어 뒀다는 점이다. 프로토콜 이탈 12건이 기록돼 있고 그중 하나는 이탈 기록
자체의 이전 판을 정정한 것이다.

---

## 질문

Truncated-Newton 의 성능은 두 개의 계산 자원 결정에 크게 좌우된다. damping 수준과 step 당
CG 반복 예산이다. 이것을 최적화 도중에 조절하는 것은 강화학습 문제처럼 보인다. 그러나
정책을 학습하기 전에 물어야 할 것이 있다. **그 이득이 어디서 나오는가.**

|  | 질문 |
|---|---|
| **Q1** | 좋은 행동 *sequence* 가 존재하는가? |
| **Q2** | 실행 중 feedback 으로 그 sequence 를 *수정*할 가치가 있는가? |

두 질문이 요구하는 산출물이 다르다. **Q1** 만 성립하면 필요한 것은 초기 상태에서 스케줄을
확정하는 예측기다. **Q2** 가 성립할 때만 step 단위 feedback 정책이 정당화된다.

둘을 분리하기 위해 동일한 예산 150 gradient-equivalent unit (GE) 에서 컨트롤러 사다리를
비교한다.

```
튜닝 상수  →  open-loop 스케줄  →  1-step greedy  →  committed plan  →  재계획
```

`committed` 는 초기 상태에서 계획 하나를 세우고 그대로 실행한다. `shrinking` 은 매 step
재계획한다. 둘은 같은 인스턴스와 같은 예산을 보므로, **그 사이의 차이가 feedback 의 가치를
분리해 낸다.**

## 세 가지 결과

`d = 100`, `κ ∈ {1e3, 1e4, 1e5, 1e6}`, regime 당 held-out seed 10개인 ill-conditioned SPD
quadratic 40 인스턴스에서 측정했다.

1. **이득의 대부분은 상태 의존적 제어에서 온다.** 튜닝 상수에서 1-step greedy 제어로 옮기면
   `+1.155 nat` (95% CI `+1.092` ~ `+1.811`, `p < 0.0001`, 40/40 인스턴스).
2. **다단계 planning 은 실재하지만 더 작은 증분을 더한다.** 1-step 제어 대비 `+0.456 nat`
   (95% CI `+0.254` ~ `+0.720`, `p < 0.0001`, 35/40).
3. **실행 중 재계획은 여기서 거의 더하지 않는다.** committed 계획 대비 `+0.010 nat`
   (95% CI `−0.033` ~ `+0.053`, `p = 0.97`, 21/40). 구간이 좁고 0 을 포함한다. **실용적으로
   큰 feedback 이득은 관측되지 않았다.**

탐색적 확장에서는 모델과 데이터를 고정하고 optimizer 가 관측하는 표본만 바꿨다. minibatch
노이즈 아래에서 committed 계획은 취약해졌다. step 기각률이 `0.00` 에서 `0.66`~`0.79` 로
올랐고 최종 개선량이 튜닝 상수보다 낮아졌다. 재계획이 그 손실을 줄였지만, **값싼 1-step
제어를 일관되게 이기지는 못했다.** 이 실행은 regime 당 `n = 3` 이므로 부호와 방향만
보고하고 크기는 인용하지 않는다.

정책 학습(PPO)은 사전 선언된 조건부 다음 단계였다. 이 증거로는 진행하지 않았다.
**PPO 를 실행하지 않았고 따라서 실패하지도 않았다.** [범위와 한계](#범위와-한계) 참조.

![컨트롤러 사다리](paper/figures/figure2_planning_vs_feedback.png)

> 각 막대는 동일한 40 인스턴스에 대한 쌍별 median 이다. **서로 다른 통계량이므로 더하거나
> 빼면 안 된다.** 쌍별 차이의 median 은 선형이 아니다.

## planner 는 진단 도구이고 방법이 아니다

> 전체 planner 는 oracle 진단 도구이며 배포 예산보다 훨씬 많은 탐색 계산을 쓴다.

| 컨트롤러 | 결정 탐색 GE | 150 GE 예산 대비 |
|---|---|---|
| `onestep_narrow` | 1,186 | 7.9배 |
| `committed_Q4_narrow` | 69,401 | 462.7배 |
| `shrinking_Q4_narrow` | 194,095 | 1,294.0배 |

다단계 planning 의 `+0.456 nat` 증분은 통계적으로 견고하고, 이 비용 격차가 그 대가다.
planner 는 **헤드룸이 얼마나 있는지 재는 계측기**로 읽어야 하며, 배포할 optimizer 로 읽으면
안 된다.

탐색 GE 는 시뮬레이션된 oracle 작업량이며 wall-clock 계산량이 아니다.

---

## 설치

Python 3.12 가 필요하다. lock 파일이 모든 의존성을 고정한다.

```bash
git clone https://github.com/bu11ymaguire/When_Does_Feedback_Help.git
cd When_Does_Feedback_Help

# uv 사용 (권장. uv.lock 에서 설치한다)
uv sync

# 또는 pip
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

아래 모든 것은 CPU 에서 돈다. **이 저장소는 어디에서도 GPU 를 쓰지 않는다.**

## 빠른 재현 (몇 분)

이 경로는 보고된 모든 통계를 다시 계산하고 모든 표와 그림을 **공개된 행 단위 결과에서**
재생성한다. optimizer 를 다시 돌리지 않는다.

```bash
# 1. results/public/*.csv 에서 쌍별 통계를 다시 계산한다
python scripts/verify_public_results.py

# 2. 원고 표를 재생성한다
python scripts/make_tables.py --public-dir results/public --out-dir paper/tables

# 3. Figure 1-4 를 재생성한다
python scripts/make_figures.py --public-dir results/public --out-dir paper/figures
```

`uv.lock` 의 고정된 환경에서 2단계와 3단계는 바이트 단위로 재현된다. `results/public/*.csv`
에서 생성한 출력이 private raw 기록에서 생성한 것과 동일하다. 쌍별 분석은 수치 스택이 아니라
**Python 표준 라이브러리**로 고정 시드에서 부트스트랩 10,000회를 뽑으므로, 보고된 median 과
신뢰구간과 `p` 값을 공개된 행에서 다시 계산할 수 있다.

**라이브러리 버전이나 플랫폼이 다를 때의 비트 수준 일치는 주장하지 않는다.** 리샘플링 자체는
NumPy 나 SciPy 에 의존하지 않지만, 그림의 바이트는 matplotlib 판본에 직접 영향을 받고
부동소수점 누적 순서는 플랫폼 간에 보장되지 않는다.

설치와 공개 수치를 함께 검사하려면 테스트를 돌린다.

```bash
pytest tests/ -q
```

`tests/test_public_reporting.py` 가 헤드라인 수치를 공개 CSV 에 고정하므로, 재생성 과정에서
실수가 있으면 테스트가 실패한다.

## 노트북

```bash
jupyter lab notebooks/overview_and_reproduction.ipynb
```

노트북은 패키지 위의 얇은 인터페이스다. 통계를 다시 구현하지 않고 스크립트가 쓰는 것과 같은
함수를 import 한다. 컨트롤러 사다리를 훑고, 공개 데이터에서 게이트 비교를 다시 계산하고,
작은 end-to-end 실험 하나를 돌리고, 그림을 재생성한다. **전체 planner 스위트는 돌리지
않는다.**

## 전체 재현 (오래 걸린다)

실험을 처음부터 다시 돌리는 것은 선택이며 비용의 대부분을 차지한다. 명령은
[`docs/reproduce.md`](docs/reproduce.md) 에 있다. held-out 확인만으로도 optimizer 실행
960회이고, planner 의 결정 탐색이 비용을 지배한다. `shrinking_Q4_narrow` 는 150 GE 배포
예산에 대해 인스턴스당 약 194,095 GE 를 쓴다.

**wall-clock 추정치는 의도적으로 공개하지 않는다.** 이 연구에서 wall-clock 은 어떤 판정에도
쓰이지 않았고, held-out 실행 중에 다른 작업이 함께 돌았으므로 기록된 시간은 깨끗한 측정이
아니다 (이탈 E4). 비용은 전부 GE 로 보고한다.

논문의 어떤 주장을 확인하기 위해서도 전체 재현이 필요하지 않다. **보고된 모든 통계는
`results/public/` 에서 몇 분 안에 다시 계산된다.**

---

## 무엇이 들어 있는가

```text
src/rl_newton/            optimizer, 컨트롤러, task, 비용 계산
  benchmark/metrics.py      쌍별 통계, 부트스트랩 CI, Wilcoxon
  reporting/                공개 결과 위의 읽기 전용 계층
scripts/
  make_public_results.py    private raw  ->  results/public/*.csv
  verify_public_results.py  공개 CSV 를 논문 통계와 대조한다
  make_tables.py            공개 CSV 또는 private raw 에서 LaTeX 표 생성
  make_figures.py           공개 CSV 또는 private raw 에서 Figure 1-4 생성
  run_headroom.py           실험 드라이버 (전체 재현)
results/public/           행 단위 결과와 checksum 이 담긴 manifest
paper/                    원고 소스, 표, 그림, 서지정보
docs/reproduce.md         전체 재현 명령
notebooks/                개요와 재현 노트북
tests/                    CPU 기준 490개
```

490개 중 458개가 구현을 검사하고, 32개가 이 공개 재현 패키지 자체를 검사한다. 행 단위 CSV
가 원고의 헤드라인 수치를 재생산하는지, 그리고 노트북이 실행 가능하며 출력이 저장되지 않은
상태인지를 본다.

**개수는 장비에 따라 다르다.** 값이 틀렸다고 판단하기 전에 알아 둘 필요가 있다.
`tests/test_cg.py` 와 `tests/test_hvp.py` 의 수치 정확도 테스트 7개가 사용 가능한 device 로
파라미터화돼 있어, CUDA 가 있는 장비는 497개를 수집하고 없는 장비는 490개를 수집한다. 이
저장소의 모든 것은 CPU 에서 도므로 **490 이 보여야 하는 값이다.**

### 공개된 결과

`results/public/` 에는 optimizer 실행 하나가 한 행씩 들어 있다. 초기 손실과 최종 손실, 파생된
log 개선량, GE 비용, step 기각률, 그리고 컨트롤러 간에 실행을 쌍으로 묶는 데 필요한 식별자다.

| 파일 | 행 수 | 역할 |
|---|---|---|
| `heldout_quadratic.csv` | 1,200 | held-out 확인. **주 결과다** |
| `configuration_selection.csv` | 390 | development seed 에서의 설정 선택 |
| `micro_neural.csv` | 432 | model mismatch 연구. 두 acceptance 규칙 모두 |
| `nonlinear_diagnostic.csv` | 72 | 사용 불가로 판정된 benchmark |
| `dev_pilot.csv` | 324 | 초기 pilot |
| `manifest.json` | — | 소스 커밋, raw checksum, 집계 규약 |

### 언어에 대해

원고와 모든 코드 docstring 은 영문이다. 이 README 는 한국어(이 파일)와
영문([`README.en.md`](README.en.md)) 둘 다 있고, 한국어를 기본으로 둔다. 아래 연구 기록들은
한국어만 있다. 출판용이 아니라 연구 중의 작업 문서로 쓰였기 때문이다.

```text
paper/claim_ledger.md          주장 원장과 프로토콜 이탈
paper/CITATIONS.md             인용별 내용 검증 체크리스트
paper/evidence_map.md          raw checksum
docs/reproduce.md              전체 재현 명령
docs/results_stage2.md         자동 생성된 결과표
docs/experiment_protocol.md    결정 D1-D32
```

**원고가 이것들을 artifact 로 인용하기 때문에 포함했다.** 없는 파일을 인용하는 것이 다른
언어로 된 파일을 인용하는 것보다 나쁘다. 안에 있는 수치와 식별자는 언어와 무관하다. 위의
재현 경로에서 한국어를 읽어야 하는 부분은 없다.

### 공개 결과의 provenance

`manifest.json` 은 private raw 파일별 SHA-256, 공개 CSV 별 SHA-256, 그리고 정확한 집계 규약
(median 규칙, 부트스트랩 횟수와 시드, 사용한 검정)을 기록한다. step 단위 궤적과 실행 환경
기록은 공개하지 않는다. **논문의 모든 통계를 다시 계산하는 데는 실행 단위 행으로 충분하다.**

이 파일을 읽을 때 `controller_role` 이 중요하다. `best_static` 과 `best_open_loop` 는
컨트롤러가 아니라 *튜닝 결과*이며, 선택된 후보가 실험마다 다르다. 그 열이 어느 라벨이 어디서
선택됐는지를 알려 준다.

---

## 범위와 한계

**확인적 결과는 좁다. 논문도 그렇게 적었다.** 이보다 넓게 읽는 것은 데이터가 지지하지 않는다.

- 주 결과는 합성 ill-conditioned SPD quadratic 에 한정된다. `d = 100` 고정, condition number
  4개, held-out seed 10개.
- model mismatch 와 acceptance 기준 결과는 regime 당 `n = 3` 의 탐색적 결과다. 신뢰구간과
  `p` 값을 인용하지 않는다.
- GE 는 regime **내부에서** oracle 호출을 맞추며, batch 크기가 다른 경우의 부동소수점 연산량을
  맞추지 않는다. regime 간 절대 비교는 서술적이다.
- equivalence margin 을 사전 등록하지 않았으므로, 작은 feedback 효과는 "효과가 없다" 가 아니라
  **"실용적으로 큰 이득이 관측되지 않았다"** 로 보고한다.
- 정책을 학습하지 않았으므로 이 연구는 학습된 정책의 성능에 대해 아무것도 말하지 않는다.
- 결과는 Hessian-free Newton 계열 전체나 일반적인 신경망 최적화로 확장되지 않는다.

시도한 Rosenbrock 변종 두 개는 모두 benchmark 로 사용 불가 판정을 받았다. 표준 시작점이
strict 국소최소점의 basin 안에 있어서 모든 baseline 이 동일한 값을 반환했고, 그 문제는
컨트롤러를 구별할 수 없었다. **이것은 benchmark 결함이며 비선형 문제에 대한 발견이 아니다.**

원고는 Limitations 절에 프로토콜 이탈 12건 (E1–E12) 을 기록한다. 그중 하나는 우리 자신의
이탈 기록의 이전 판이 틀렸던 경우다.

## 재현성 규약

코드를 읽기 전에 알아 둘 규약이 몇 개 있다.

**seed 는 난수 시드가 아니라 실험 조건의 이름이다.** 주어진 seed 에서 모든 컨트롤러가 동일한
인스턴스, 동일한 초기점, 동일한 minibatch 순서를 본다. task 생성 난수 스트림은 optimizer 실행
스트림과 완전히 분리돼 있다.

**seed 역할을 분리했다.** calibration seed 가 benchmark spec 을 골랐고, selection seed 가 설정을
골랐고, held-out seed `100~109` 는 최종 효과 추정에만 썼다. 설정은 development seed 에서 한 번
고정했고 다시 선택하지 않았다.

**결과 정체성을 3계층으로 분리했다.** 집계 코드나 문서를 고쳐도 optimizer 가 절대 재실행되지
않는다. 커밋 해시는 provenance 로 기록하되 **의도적으로 어떤 식별자에도 넣지 않는다.**

**숫자를 손으로 타이핑하지 않는다.** 원고의 모든 표와 그림은 결과 파일에서 스크립트가
생성한다.

---

## AI 사용 공개

**원고 집필을 포함해 이 연구 전 과정에서 AI 를 사용했다.** Kiro 를 통해 접근한 Claude Opus 가
구현, 리팩터링, 테스트 생성, 실험 오케스트레이션, 리포트 생성을 도왔다. ChatGPT 를 통해
접근한 GPT-5.6 Sol 이 방법론 비판, 교란 요인 분석, 주장 강도 조정, 원고 검토를 도왔다.

사람 저자가 연구 질문을 세우고, 실험 프로토콜을 정하고 동결했고, 모든 프로토콜 변경을
승인했고, 보고된 결과를 검증했고, 해석을 결정했고, **오류를 포함해 이 작업에 대한 모든 책임을
진다.**

그 역할 분담을 주장으로 남기지 않고 기록으로 남긴다. **"사람이 감독했다" 는 검사할 수 없는
주장이지만, 이것은 검사할 수 있다.**

```text
docs/experiment_protocol.md   결정 D1-D32 를 순서대로, 각각 고정된 날짜와 함께.
                              게이트 임계값이 확인적 실행보다 앞섰다는 주장을
                              커밋 이력으로 대조할 수 있다
paper/claim_ledger.md         모든 주장과 그 근거, 그리고 프로토콜 이탈 12건 E1-E12.
                              E12 는 이탈 기록 자체의 이전 판을 정정한 것이다
scripts/check_claims.py       원장이 미지지로 표시한 것을 주장하는 원고 문장을
                              기계적으로 거부하고, 근거 출처가 없는 수치 주장을 거부한다
scripts/check_latex.py        원고가 원격에 존재하지 않는 저장소나 릴리스 태그를
                              인용하면 통과시키지 않는다
```

**도구가 닿지 못하는 부분이 남는다.** 기계적 검사는 원고가 기록된 증거를 과장하지 않았음을
보인다. 연구 질문이 물을 가치가 있었는지, 실험 설계가 도메인 전문가가 택할 설계인지는 보이지
못한다. **이 작업은 독립 연구자의 검토를 받지 않았다.** 결함을 발견하면 issue 를 열어 주면
고맙겠다.

### 질문은 어디서 시작했는가

위의 공개는 누가 무엇을 했는지를 다룬다. **질문이 어디서 시작했는지는 말하지 않는다.**
AI 생성물로 보인다는 지적을 받은 프로젝트라면 그것을 기록으로 남기는 편이 맞다고 생각한다.
코드가 하나도 없던 시점에 내가 처음 물은 것이다.

> 딥러닝 역전파에서 GD말고 뉴턴 메소드를 쓰고 싶은데, Hessian 역행렬 연산이 병목이잖아.
> 행렬의 역행렬 연산 최적화 기법을 탐구해줘. 아니면 Optimizer 문제를 강화학습의 에이전트가
> 최적화를 찾아가는 방식으로 접근한 문제는 없어? 애초에 에이전트를 사전학습 시키려면
> Optimizer가 필요한가?ㅋㅋ

잘 벼려진 연구 질문이 아니다. 미적분학 수업에서 뉴턴법을 막 배운 학부생이 Hessian 이
장애물임을 알아채고, 그 장애물을 학습된 컨트롤러에 넘길 수 있는지 궁금해하고, 곧바로 자기
아이디어의 순환성을 발견해 웃은 기록이다.

**살아남은 것은 그 마지막 한 줄이다.** 정책을 학습하려면 optimizer 가 필요하고, 그 계산을
쓰기 전에 정당화가 있어야 한다. 이 프로젝트가 정책을 학습하지 않고 oracle planner 를 만들어
정책이 최대로 가져갈 수 있는 헤드룸을 측정한 이유가 그것이다. **이득이 애초에 있는가**를 먼저
물어야 했다. 이 연구의 전부인 `committed` 대 `shrinking` 비교는 그 첫 메시지의 농담에서
직접 내려온 것이다.

그 질문에서 이 저장소까지의 경로는 ChatGPT 와의 긴 대화, 그리고 구현에 관해서는 Kiro 를 통한
Claude Opus 와의 대화를 지난다. **대화 기록은 공개하지 않는다.** 그 안에는 제3자의 사적 서신이
들어 있고, 그것이 바로 다음 절의 주제다. 링크라는 형태를 취한다고 해서 공개가 더 허용되는
것은 아니다.

## 한 연구자가 해 준 말과, 그 뒤에 한 일

위의 "독립 연구자의 검토를 받지 않았다" 는 검토를 구하지 않았기 때문이 아니다. 2026년 8월,
이 분야의 한 선임 연구자에게 원고의 arXiv endorsement 를 요청했다. 그는 거절했고, endorsement
보다 값진 이유를 적어 보냈다. **논문이 AI 생성물처럼 읽힌다는 것,** 그런 논문이 이미 충분히
많아 시스템에 잡음을 만든다는 것, 그리고 학부생이 지금 해야 할 일은 자기 학교에서 이 작업을
함께 발전시키고 학계로 가져가 줄 사람을 찾는 것이라는 조언이었다.

**그의 이름을 밝히지 않고 메시지를 그대로 옮기지도 않는다.** 무시할 수도 있었던 낯선 사람의
요청에 사적으로 답해 준 사람이고, 나는 그것을 공개할 허락을 구하지 않았다. 공개할 수 있는
것은 내가 보낸 답장이다.

> Thank you very much for taking the time, despite your busy schedule, to read my
> manuscript and share your honest feedback.
>
> This project began with a simple curiosity: what would happen if I applied the idea
> behind Newton's method, which I first encountered in an undergraduate calculus course,
> to optimization instead of relying only on gradient descent? As an undergraduate,
> developing that initial question into an implementation and a series of experiments
> with the help of generative AI was sometimes confusing, but also a genuinely new and
> exciting experience.
>
> At the same time, your comments made me reflect on the important distinction between
> implementing an idea and developing it into mature research through discussion with an
> academic community and in one's own scholarly voice. Moving forward, I will seek more
> feedback from faculty members and fellow students at my university, and work to ensure
> that I can independently explain, defend, and take responsibility for every part of
> the manuscript.
>
> Thank you again for your time and thoughtful advice. I believe this exchange will
> remain a valuable experience as I continue pursuing research and a career in machine
> learning.

그 결과로 이 저장소에서 세 가지가 바뀌었다. arXiv 관련 서술을 조용히 지우는 대신 이 절을
남긴 이유이기도 하다.

```text
표지        "Technical report. Unrefereed. peer review 를 받지 않았고 어떤 preprint
            서버에도 없다" 를 넣었다. PDF 는 혼자 돌아다니고, 그 줄이 없으면
            심사를 통과한 논문으로 읽힐 수 있다
릴리스 태그  arxiv-submission-v1 을 삭제했다. 일어나지 않은 제출을 가리키는 이름이었고,
            그것은 \PLACEHOLDER 장치와 check_latex.py 의 원격 검사가 막으려는 오류와
            정확히 같은 종류다. 지금은 stage2-report-v1 이다
공개 서술    위의 AI 사용 공개가 더 이상 사람의 감독을 단정하지 않는다. 날짜가 붙은
            결정 기록, 주장 원장, 그리고 그 단정을 반증 가능하게 만드는 검사기를
            가리킨다
```

그가 그은 구분은, 이 프로젝트에서 하나만 남길 수 있다면 남기고 싶은 것이다. **아이디어를
구현하는 것과 그것을 연구로 발전시키는 것은 다른 행위이며, 후자에는 다른 사람이 필요하다.**
이 저장소는 전자에 대한 정직한 기록이고, 그 경계를 표시해 두었다.

## 인용

[`CITATION.cff`](CITATION.cff) 를 참조한다. 이 코드나 공개된 행 단위 결과를 사용하면 이
저장소를 인용해 주기를 바란다. 원고는 미심사 technical report 이며 DOI 나 preprint 식별자를
갖지 않는다.

## 라이선스

MIT. [`LICENSE`](LICENSE) 참조.
