# 인용 검증 체크리스트

**이 문서가 인용 검증의 유일한 기록이다.** `paper/references.bib` 에는 출판에 필요한
정상 서지정보만 둔다. 검증 상태와 오인용 위험을 `.bib` 의 필드로 넣지 않는다.
`plainnat` 이 `note` 필드를 조판 결과에 인쇄하므로, 거기에 내부 메모를 적으면 그 메모가
논문 참고문헌 목록에 그대로 나온다.

이 문서는 두 질문을 나눠 다룬다.

```text
서지정보가 맞는가            -> §9 서지정보 검증 출처, §7 남은 항목
그 논문이 실제로 우리가      -> §1~§6 판정과 주의
그 자리에서 하는 말을
지지하는가
```

두 작업은 분리해야 한다. 서지정보가 정확해도 인용이 틀릴 수 있다. `scripts/check_claims.py`
의 `[6]` 검사는 키가 존재하는지와 항목에 `TODO` 가 남아 있지 않은지만 본다. **내용 일치는
기계로 확인할 수 없으므로 이 문서로 관리한다.**

## 판정 기호

```text
DIRECT     원문이 우리 문장을 직접 진술한다
BACKGROUND 방법·개념의 표준 출처로 인용한다. 우리 결과를 지지하는 인용이 아니다
PARTIAL    원문이 인접한 것을 말한다. 본문 표현을 원문 범위에 맞춰 좁혔다
RISK       오인용 위험이 있다. 아래 주의 항목을 지키지 않으면 틀린다
```

---

## 1. Truncated / inexact Newton, Hessian-free

### `dembo1982inexact`, `steihaug1983cg`, `nash1984lanczos`, `nash2000survey`

```text
사용 위치  §1 도입, §2.1 Newton-CG step, §1 CG 예산 설명
우리 문장  Newton 방정식을 반복법으로 근사해 풀고, CG 반복 예산 절단이 핵심 설계
           변수다
판정       BACKGROUND
```

원문은 inexact Newton 계열의 수렴 조건과 절단 규칙을 다룬다. 우리는 **방법의 출처**로만
인용하고, "우리 실험 결과를 이 논문들이 예측했다" 고 쓰지 않는다.

주의. `steihaug1983cg` 는 trust-region 안에서의 CG 절단을 다룬다. 우리는 trust-region 을
쓰지 않고 damping 을 쓴다. 그래서 §1 에서 "대응 관계가 있다" 로만 쓰고 "같다" 로 쓰지
않았다.

### `martens2010hessianfree`, `martens2011rnn`

```text
사용 위치  Abstract, §1, §2.1
우리 문장  심층망 규모에서 Hessian-free Newton-CG 가 적용됐고 damping 조절 규칙이
           성능을 좌우한다는 보고가 있다
판정       BACKGROUND
```

주의. `martens2010hessianfree` 는 우리 damping 격자나 CG 예산 격자를 정당화하지 않는다.
우리 action space (`§2.1`) 는 우리가 고른 것이다. 또한 원문은 우리와 달리 학습 문제를
대상으로 하므로, `§7` 의 quadratic 결과를 원문 설정으로의 전이로 읽으면 안 된다.
`§14 [L9]` 가 그 경계를 명시한다.

### `pearlmutter1994hvp`

```text
사용 위치  §1, §2.1
우리 문장  곡률 접근을 HVP 만으로 하며 정확 HVP 계산법이 있다
판정       DIRECT
```

원문이 Hessian-vector product 를 정확히, gradient 비용의 상수배로 계산하는 방법을
제시한다. 우리 GE 회계(`cost_GE(k) = c_grad_graph + k·c_hvp + c_fwd`) 가 "HVP 1회 =
상수 비용" 을 전제하는데, 그 전제의 출처가 이 논문이다.

### `byrd2011stochastic`

```text
사용 위치  §10.1
우리 문장  부분표본 곡률을 Newton-CG 에 넣는 설정 자체가 어렵다는 것은 알려져 있다
판정       PARTIAL
```

원문은 sub-sampled Hessian 정보를 쓰는 방법과 그 표본 크기 선택을 다룬다. 우리 `R2`
regime 과 설정이 가깝다. 다만 원문은 **해법**을 제시하는 논문이고 우리는 "어렵다" 는
배경으로만 쓴다. 우리 `R2` 결과의 부호나 크기를 원문으로 설명하지 않는다.

---

## 2. Conjugate gradient, damping

### `hestenes1952cg`

```text
사용 위치  §2.1
우리 문장  선형계를 CG 로 푼다
판정       BACKGROUND
```

### `levenberg1944`, `marquardt1963`

```text
사용 위치  §1
우리 문장  damping 은 Levenberg-Marquardt 정칙화 계열이다
판정       BACKGROUND
```

주의. 두 원문은 비선형 최소제곱 문제를 다룬다. 우리 목적함수는 최소제곱이 아니다. 그래서
"Levenberg-Marquardt 알고리즘을 쓴다" 가 아니라 "**LM 유형** 정칙화" 로 쓴다. 우리가 쓰는
것은 `(H + λI)p = −g` 이고 `λ` 를 로그 공간에서 배수로 조절한다.

### `conn2000trustregion`

```text
사용 위치  §1
우리 문장  damping 과 trust-region 반경 사이에 대응 관계가 있다
판정       BACKGROUND
```

주의. **우리는 trust-region 을 구현하지 않았다.** 수락 규칙은 단조 감소 검사이고 신뢰
반경 갱신이 아니다. 이 인용은 개념 위치를 잡는 용도이며, 우리 방법이 trust-region 수렴
이론의 보증을 받는다는 뜻이 아니다.

---

## 3. Learned optimizers, amortized control

### `andrychowicz2016l2l`, `metz2019pathologies`, `metz2020effective`

```text
사용 위치  §1, §12.4
우리 문장  optimizer 하이퍼파라미터를 실행 중 조절하려는 시도는 learned optimizer
           문헌과 맞닿아 있다
판정       BACKGROUND
```

주의. 이 논문들은 **update rule 자체를 학습**한다. 우리는 update rule 을 Newton-CG 로
고정하고 그 위의 자원 배분만 제어한다. 두 문제를 같다고 쓰면 안 된다. `metz2019pathologies`
와 `metz2020effective` 는 학습된 optimizer 의 불안정성을 보고하므로, 우리가 PPO 로
진행하지 않은 결정의 **일반적 배경**으로 쓸 수 있으나 그것이 우리 게이트 판정의 근거는
아니다. 근거는 우리 held-out 결과다 (`§12.2`).

### `bae2022apo`

```text
사용 위치  §1, §12.4
우리 문장  learning rate 나 preconditioner 를 온라인으로 메타학습하는 접근이 있다
판정       DIRECT (배경으로서)
```

원문이 learning rate 와 구조화된 preconditioner 를 proximal 목적으로 메타학습한다. 우리
`§12.4` 의 "amortized 방향" 언급과 가장 가까운 기존 연구다.

주의. **우리는 이 방향을 구현하지 않았다.** `§12.4` 는 후속 방향 서술이며, 원문 성능과
우리 planner 성능을 비교하지 않는다.

### `amos2023amortized`

```text
사용 위치  §12.4
우리 문장  유사한 문제 인스턴스의 공유 구조를 학습해 해를 예측하는 접근을 amortized
           optimization 이라 한다
판정       DIRECT (용어 정의)
```

주의. 서지정보가 부분 미확인이다(`§7 [C2]`, `§9`). FnT 권/호/페이지를 확정하지 못했으므로
`.bib` 항목의 `note` 에 `arXiv:2202.00665` 를 두어 참고문헌에 arXiv ID 가 인쇄되게 했다.
이 `note` 는 내부 메모가 아니라 독자용 서지정보다. 최종본에서 권/호/페이지를 확정하면
지운다.

---

## 4. Planning, feedback, MPC

### `rawlings2017mpc`

```text
사용 위치  §3
우리 문장  shrinking 컨트롤러가 MPC 의 receding-horizon 구조를 따르되 남은 예산이
           줄어들어 horizon 이 축소된다
판정       BACKGROUND
```

주의. 우리는 안정성 보증이나 제약 처리를 하지 않는다. MPC 의 **구조**만 차용한다. 원문의
수렴·안정성 정리를 우리 컨트롤러가 만족한다고 쓰면 안 된다.

### `bertsekas2017dp`

```text
사용 위치  §1, §3
우리 문장  계획을 미리 정하는 것과 실행 중 상태에 반응하는 것의 구분은 open-loop 대
           closed-loop 제어의 고전적 구분이다
판정       BACKGROUND
```

주의(RISK). **절 번호를 지목하지 않는다.** 서지 확인 과정에서 판·권은 확정했지만 해당 절
번호는 확인하지 못했다. `§6.x` 같은 구체적 지시를 넣으려면 원문을 직접 확인해야 한다.

또한 우리 `committed` 는 교과서적 open-loop 제어와 완전히 같지 않다. 우리 `committed` 는
**그 인스턴스의 초기 상태에 조건화된** 계획이고, `best_open_loop` 가 인스턴스 집합에서
튜닝된 스케줄이다. 이 구분은 `§3.1` 에 우리 정의로 서술돼 있고 문헌 인용으로 대체할 수
없다.

### `schulman2017ppo`

```text
사용 위치  §1
우리 문장  강화학습으로 조절 정책을 학습하려면 (PPO 등)
판정       BACKGROUND
```

주의(RISK). **PPO 를 실행하지 않았다.** 이 인용이 "우리가 PPO 를 썼다" 로 읽히면 안 된다.
`§12.2` 가 "PPO training was a conditional next stage" 라고 명시하고 `§14 [L8]` 이 다시
경계를 긋는다. 원고 어디에서도 PPO 결과를 보고하지 않는다.

---

## 5. Benchmark 구조 (Rosenbrock)

### `shang2006rosenbrock`, `kok2009rosenbrock`

```text
사용 위치  §5.1
우리 문장  확장 Rosenbrock 은 좌표 결합 변종에서 d >= 4 일 때 추가 정류점을 갖는다
판정       DIRECT + RISK
```

`shang2006rosenbrock` 은 `n = 4~30` 에서 최소점이 2개라고 명시한다. `kok2009rosenbrock` 은
**두 변종을 구별하고**, 짝수 차원에서만 정의되는 비결합 2D 합은 전역최소점만 갖는 반면
좌표가 결합된 변종은 고차원에서 다수의 정류점을 갖는다고 보고한다.

주의(RISK). 변종을 밝히지 않으면 이 인용은 틀린다. 우리 구현은 **결합 변종**이다.

```text
L(x) = sum_{i=1}^{d-1} [ 100 (x_{i+1} − x_i²)² + (1 − x_i)² ]
출처   src/rl_newton/tasks/rosenbrock.py  RosenbrockTask.loss
```

`§5.1` 이 이 식을 본문에 명시한다. 그 문장을 지우면 인용이 무효가 된다.

교차 확인. 우리가 수치로 찾은 국소최소점의 첫 좌표는 `x_1 = −0.96205102` 이고 문헌이
보고하는 국소최소점 위치(`x_1 ~ −1`) 와 부합한다. 다만 **우리 표의 좌표·loss·고유값은
우리 측정값**이며 문헌 표를 옮긴 것이 아니다. `§5.1` 이 그 사실도 명시한다.

---

## 6. 통계

### `wilcoxon1945`

```text
사용 위치  §4.1
우리 문장  쌍별 차이에 Wilcoxon signed-rank 검정을 쓴다
판정       BACKGROUND
```

주의. 원문이 제시한 것은 rank sum 과 signed rank 두 절차다. 우리는 **signed-rank** 를
쓴다. 표기에서 둘을 섞지 않는다.

### `efron1979bootstrap`

```text
사용 위치  §4.1
우리 문장  중앙값의 부트스트랩 신뢰구간을 쓴다
판정       BACKGROUND
```

주의. 원문은 부트스트랩의 원 논문이다. 우리가 쓰는 구체적 구간 방법(percentile 계열)의
정당화는 원문이 아니라 우리 구현 문서에 있다. 원문으로 "우리 CI 가 정확하다" 를 주장하지
않는다. `n=3` regime 에서 CI 를 인용하지 않는다는 규칙(`§10.4`, `§14 [L2]`)이 이 인용보다
우선한다.

### `schuirmann1987tost`, `lakens2017equivalence`

```text
사용 위치  §7.4, §12.1
우리 문장  equivalence margin 을 사전 등록하지 않았으므로 등가성을 주장할 수 없다
판정       DIRECT
```

`schuirmann1987tost` 가 TOST 절차의 출처이고, 그 절차는 상·하 경계를 **미리** 정할 것을
요구한다. `lakens2017equivalence` 는 등가 경계의 사전 지정이 권고 사항이라고 진술한다.
둘 다 우리 문장을 직접 지지한다.

주의(RISK). 이 인용은 **우리가 등가성 검정을 수행했다** 는 뜻이 아니다. 수행하지 않았다.
`C3` 의 CI 가 0 을 포함한다는 것과 "효과가 0 이다" 는 다른 진술이며, `check_claims.py` 의
금지어 검사가 `equivalent` / `equivalence` 를 막는다. 예외로 허용된 표현은
`equivalence margin` / `equivalence testing` 이며 둘 다 "우리는 하지 않았다" 는 문맥에서만
쓴다.

---

## 7. 남은 항목

```text
[C1] bertsekas2017dp 절 번호 미확인. 특정 절을 지목하지 않는 상태로 유지한다
[C2] amos2023amortized FnT 권/호/페이지 미확인. arXiv ID 로 인용한다
[C3] lakens2017equivalence 권/호/페이지 미확인. doi 로 인용한다
[C4] metz2020effective 제목이 arXiv 판과 workshop 판에서 다르다. 최종본에서 하나로 고정
[C5] rawlings2017mpc 판(edition)과 연도를 최종본에서 확정한다
```

이 다섯 항목은 **본문 주장에 영향을 주지 않는다.** 서지 표기 정밀도 문제이며 제출 직전에
정리한다.

## 8. 우리 주장 중 문헌 인용으로 지지되지 않는 것

아래는 **우리 측정만이 근거**다. 인용을 붙이면 오히려 틀린다.

```text
C01~C07  사다리 각 구간의 효과 크기. paper/evidence_map.md 의 SHA-256 이 근거다
C08~C10  benchmark eligibility 절차. 우리 프로젝트의 경험적 절차이며 범용 framework 가
         아니다 (§13.3)
C12~C14  micro-neural regime 과 수락 기준 ablation. n=3 exploratory
C17~C19  범위와 교란 서술
seed 복제 검출, 정체성 3층 분리, bitwise 재현 검사 (§4.3, §5.3)
```

`§4.1` 의 "컨트롤러가 난수를 얼마나 쓰든 인스턴스가 같다" 도 여기 속한다. 초안에는
`[CITATION NEEDED]` 가 붙어 있었으나, 이는 문헌에서 가져온 주장이 아니라 우리 실행기의
설계 사실이므로 인용을 붙이지 않고 `§4.3` 의 bitwise 검사로 근거를 댔다.

---

## 9. 서지정보 검증 출처

`references.bib` 의 `note` 필드에 있던 검증 기록을 여기로 옮겼다 (2026-08-05). `.bib` 에
두면 `plainnat` 이 참고문헌 목록에 인쇄하므로 내부 메모를 둘 자리가 아니다.

**판정 기준.** `VERIFIED` 는 저자·제목·연도·게재지와 권/호/페이지(또는 arXiv ID)를 원문
또는 출판사 페이지에서 확인했다는 뜻이다. 확인하지 못한 필드가 있으면 아래에 적고 `§7` 에
남긴다. 2026-08-05 기준 **미확인 상태로 인용하는 항목은 없다.**

### Truncated / inexact Newton, Hessian-free

```text
dembo1982inexact        VERIFIED  SIAM, JSTOR stable/2156954 에서 권/호/페이지 확인
steihaug1983cg          VERIFIED  SIAM doi 확인. 권/호/페이지는 SIAM 페이지 기준
nash1984lanczos         VERIFIED  Springer 인용 목록, JSTOR stable/2157008
nash2000survey          VERIFIED  ADS 2000JCoAM.124...45N 및 doi
martens2010hessianfree  VERIFIED  icml.cc/Conferences/2010/papers/458.pdf 원문
martens2011rnn          VERIFIED  icml.cc/2011/papers/532_icmlpaper.pdf 원문
pearlmutter1994hvp      VERIFIED  MIT Press direct.mit.edu/neco 원문 PDF
byrd2011stochastic      VERIFIED  SIAM epubs 원문, Nocedal CV 에서 권/페이지
```

### Conjugate gradient, damping, trust region

```text
hestenes1952cg      VERIFIED  AMS Mathematics of Computation 인용 목록, SIAM, Springer,
                              PETSc 문서가 모두 vol. 49, pp. 409--436 (1952) 로 일치.
                              호 번호 6 은 Springer JOTA 인용에서 확인
levenberg1944       VERIFIED  JSTOR Quarterly of Applied Mathematics vol. 2 no. 2
                              (July 1944) 목차, SIAM 및 Springer 인용 목록에서
                              pp. 164--168 확인
marquardt1963       VERIFIED  SIAM doi 10.1137/0111030, Garfield Citation Classics
                              (J. Soc. Indust. Appl. Math. 11:431-41, 1963)
conn2000trustregion VERIFIED  SIAM epubs doi 10.1137/1.9780898719857 에서 출판사/연도/
                              장 구조 확인
```

### Learned optimizers, amortized, RL, MPC

```text
andrychowicz2016l2l VERIFIED  proceedings.neurips.cc 2016 원문. arXiv:1606.04474
metz2019pathologies VERIFIED  proceedings.mlr.press/v97/metz19a 원문
metz2020effective   VERIFIED  arXiv:2009.11243 초록 확인. arXiv 판과 NeurIPS workshop
                              판의 제목 표기가 다르다 -> §7 [C4]. 정식 게재지가 없어
                              @article journal = {arXiv preprint arXiv:2009.11243} 로
                              둔다 (schulman2017ppo 와 같은 표기)
schulman2017ppo     VERIFIED  arXiv:1707.06347 초록
bae2022apo          VERIFIED  proceedings.neurips.cc 초록 페이지
                              (doi 10.52202/068431-0653), arXiv:2203.00089
amos2023amortized   VERIFIED(arXiv)  arXiv:2202.00665 초록 페이지에서 현재 제목과
                              comment 필드 "Foundations and Trends in Machine Learning"
                              확인. **FnT 권/호/페이지 미확인** -> §7 [C2].
                              v1--v2 제목은 "...for learning to optimize over continuous
                              domains" 였으므로 최종본에서 하나로 고정한다
rawlings2017mpc     VERIFIED  저자 구성을 sites.engineering.ucsb.edu/~jbraw/mpc/ 에서
                              확인. **판(edition)과 연도 미확정** -> §7 [C5]
bertsekas2017dp     VERIFIED(서지)  Athena Scientific 주문 페이지에서 Vol. I 4th ed.,
                              ISBN-13 978-1-886529-43-4, 576 pages 확인.
                              **절 번호 미확인** -> §7 [C1], §4 주의(RISK)
```

### Benchmark 구조

```text
shang2006rosenbrock VERIFIED  MIT Press 원문 초록. "the n-dimensional (n = 4--30)
                              Rosenbrock function has 2 minima" 라고 명시한다.
                              우리 d=5 국소최소점 관측을 직접 지지한다
kok2009rosenbrock   VERIFIED  MIT Press cognet 초록과 UP 저장소 원문(hdl:2263/13845).
                              두 변종 구별은 §5 에 있다. 우리 구현은 결합 변종이며
                              우리가 찾은 x_1 = -0.962 가 문헌의 x_1 ~ -1 과 부합한다
```

### 통계

```text
wilcoxon1945          VERIFIED  JSTOR stable/3001968, Biometrics Bulletin vol. 1 no. 6
                                (Dec. 1945) 목차에서 pp. 80--83 확인
efron1979bootstrap    VERIFIED  다수 독립 인용원(Springer, tandfonline, R CRAN refman)이
                                Ann. Statist. 7(1):1--26 로 일치
schuirmann1987tost    VERIFIED  PubMed 3450848, Springer doi
lakens2017equivalence VERIFIED  SAGE doi 10.1177/1948550617697177 원문.
                                **권/호/페이지 미확인** -> §7 [C3]. doi 로 인용한다
```
