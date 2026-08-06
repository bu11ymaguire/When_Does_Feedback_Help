"""강화학습 컨트롤러 — 환경, 상태, 보상, 정책 학습.

에이전트는 optimizer가 아니다. Newton-CG solver의 **컨트롤러**다.
수백만 차원의 업데이트 방향을 출력하지 않고, 3개 축의 이산 선택만 한다.

```text
MultiDiscrete([3, 4, 3])
  damping multiplier : [0.3, 1.0, 3.0]
  cg budget          : [3, 5, 10, 20]
  step size          : [0.25, 0.5, 1.0]
```

구현 예정 (Stage 4)
-------------------
``state_features.py``
    README §5.1 특징 벡터. running mean/std 정규화, log 변환, clipping.
    NaN/Inf 가드 필수 — observation에 비유한값이 들어가면 PPO가 조용히 망가진다.
    ``progress`` 는 포함하되 제거 ablation을 반드시 수행한다 (프로토콜 D8).

``rewards.py``
    프로토콜 D3 보상:
        ``r_t = (log L_t - log L_{t+1}) - beta * cost_ge / GE_ref - gamma * I_failure``
    리턴이 ``log(L_0/L_T) - beta*총비용 - gamma*총실패`` 로 텔레스코핑된다.
    README의 상대 감소량 형태는 ablation으로 보존한다.

``environment.py``
    Gymnasium 환경. **termination과 truncation을 구분해서 반환한다.**
    horizon 절단 시 value bootstrapping이 켜져야 정책이 근시안적으로
    학습되지 않는다 (프로토콜 D8). horizon 랜덤화: {30, 50, 80, 120}.
    ``gymnasium.utils.env_checker.check_env`` 를 통과해야 한다.

``train_policy.py``
    Stage 2 ``greedy_oracle`` trajectory로 behavior cloning warm start -> PPO.
    체크포인트와 observation normalization 통계를 **함께** 저장한다.
    Windows에서는 ``DummyVecEnv`` 를 쓴다 (``SubprocVecEnv`` 는 프로세스별
    CUDA 컨텍스트로 8GB VRAM을 소진한다).
    action 히스토그램과 entropy를 로깅해 보상 해킹을 감시한다.
"""
