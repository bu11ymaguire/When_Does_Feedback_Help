# Evidence map

`scripts/make_manifest.py` 가 생성한다. **손으로 수정하지 않는다.**

원고의 모든 수치는 `docs/results_stage2.md` 에서 오고, 그 표는 아래 raw 결과에서
생성된다. SHA-256 을 고정해 원고 수정 중 숫자가 수기로 변질되는 것을 막는다.

## 결과별 역할

리뷰어가 사전에 지정한 역할이다. **exploratory 결과로 primary 주장을 만들지 않는다.**

| raw | 역할 | 완료 | 실패 | beam | 예산 GE | 수락 규칙 |
|---|---|---|---|---|---|---|
| `headroom_challenge-heldout_step_size_fixed_b8_9a18b6e9.jsonl` | held-out confirmation | 1200 | 0 | 8 | 150 | `control` |
| `headroom_challenge_step_size_fixed_b8_fc78c2ad.jsonl` | configuration selection | 30 | 0 | 8 | 150 | `control` |
| `headroom_challenge_step_size_fixed_b8_fed9aebd.jsonl` | configuration selection | 360 | 0 | 8 | 150 | `control` |
| `headroom_micro-neural_step_size_fixed_b8_0bec1125.jsonl` | exploratory ablation | 216 | 0 | 8 | 150 | `control` |
| `headroom_micro-neural_step_size_fixed_b8_7aac1b26.jsonl` | exploratory ablation | 144 | 0 | 8 | 150 | `control` |
| `headroom_micro-neural_step_size_fixed_b8_9f3194be.jsonl` | exploratory ablation | 216 | 0 | 8 | 150 | `fixed_eval` |
| `headroom_nonlinear-diagnostic_step_size_fixed_b8_2a09bd45.jsonl` | diagnostic, unusable | 72 | 0 | 8 | 150 | `control` |
| `headroom_pilot_step_size_fixed_b4_44e3242e.jsonl` | dev pilot | 46 | 0 | 4 | 150 | `control` |
| `headroom_pilot_step_size_fixed_b4_8e9cdd02.jsonl` | dev pilot | 1526 | 0 | 4 | 150 | `control` |
| `headroom_pilot_step_size_fixed_b4_9d725689.jsonl` | dev pilot | 324 | 0 | 4 | 150 | `control` |

## Checksum 과 정체성

### `headroom_challenge-heldout_step_size_fixed_b8_9a18b6e9.jsonl`

최종 효과 추정 (D26)

```text
sha256          95b0d3c2191b05cdeb2faff89fa9ca617c0acf06b798d1823d307b3f5cad5f7a
완료 / 실패     1200 / 0
sweep_id        d540541777d40c93
aggregation_id  1fb5800f9248290c
git_commit      b713c1d9 (dirty)
seeds           [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
experiment_id   12종
                0566cacd125cdcbb
                210db6664bffcbf2
                21e70d55aa18d500
                339ac7dda0d4ca07
                4924921c77395980
                5f31db4e5f5a8756
                635c65850b755cd2
                af464282a6567662
                b9b6e7b5e2c79268
                bd0ec1988f4efbb5
                f18cc7378a679b2d
                fee0a575bb699322
selection       open_loop -> open_loop[4]
selection       static -> static[2]
```

### `headroom_challenge_step_size_fixed_b8_fc78c2ad.jsonl`

설정 선택 근거 (D21/D22)

```text
sha256          30ce464d73c4ecf8c63106e1ce5d7d2e5952a01f1e2f3a4328754085509d5329
완료 / 실패     30 / 0
sweep_id        5038356e37c57982
aggregation_id  ec5019fa2cc67699
git_commit      1b0a8eea
seeds           [2]
experiment_id   16종
                0117e8fff77cd4d7
                0566cacd125cdcbb
                210db6664bffcbf2
                2782bd060a64a149
                2bf7c89ab40005b2
                339ac7dda0d4ca07
                4924921c77395980
                5f31db4e5f5a8756
                635c65850b755cd2
                6e6885e1804e705f
                76bd6b4aee35161d
                b9b6e7b5e2c79268
                ... 4개 더
selection       open_loop -> open_loop[4]
selection       static -> static[2]
```

### `headroom_challenge_step_size_fixed_b8_fed9aebd.jsonl`

설정 선택 근거 (D21/D22)

```text
sha256          d4dffaf1790a82184965f92b00862142ebc11173312634adb5ef6a14f56b78b4
완료 / 실패     360 / 0
sweep_id        fd43dbfb3ffacc61
aggregation_id  ec5019fa2cc67699
git_commit      8b4ec7e0
seeds           [2, 3, 4]
experiment_id   16종
                0117e8fff77cd4d7
                0566cacd125cdcbb
                210db6664bffcbf2
                2782bd060a64a149
                2bf7c89ab40005b2
                339ac7dda0d4ca07
                4924921c77395980
                5f31db4e5f5a8756
                635c65850b755cd2
                6e6885e1804e705f
                76bd6b4aee35161d
                b9b6e7b5e2c79268
                ... 4개 더
selection       open_loop -> open_loop[4]
selection       static -> static[6]
```

### `headroom_micro-neural_step_size_fixed_b8_0bec1125.jsonl`

regime 및 수락 규칙 (D27/D28/D31)

```text
sha256          be11089349712219a2b2d82a23cec3d022684313b8d2f7827a09826a4d44b543
완료 / 실패     216 / 0
sweep_id        da27f6362e5b149b
aggregation_id  1fb5800f9248290c
git_commit      b444b890
seeds           [2, 3, 4]
experiment_id   10종
                0566cacd125cdcbb
                339ac7dda0d4ca07
                4851ff2201dd34ed
                5f31db4e5f5a8756
                635c65850b755cd2
                b9b6e7b5e2c79268
                bd0ec1988f4efbb5
                c03df36418be505d
                f18cc7378a679b2d
                fee0a575bb699322
selection       open_loop -> open_loop[0]
selection       static -> static[4]
```

### `headroom_micro-neural_step_size_fixed_b8_7aac1b26.jsonl`

regime 및 수락 규칙 (D27/D28/D31)

```text
sha256          0c1aa77aef8aaf7b9e78ee93b23b9073a3fb7163c51bb6e8a1da7321f7091047
완료 / 실패     144 / 0
sweep_id        bd14fe43f7d6bbf1
aggregation_id  1fb5800f9248290c
git_commit      0e5a182f
seeds           [2, 3, 4]
experiment_id   10종
                0566cacd125cdcbb
                339ac7dda0d4ca07
                4851ff2201dd34ed
                5f31db4e5f5a8756
                635c65850b755cd2
                b9b6e7b5e2c79268
                bd0ec1988f4efbb5
                c03df36418be505d
                f18cc7378a679b2d
                fee0a575bb699322
selection       open_loop -> open_loop[0]
selection       static -> static[4]
```

### `headroom_micro-neural_step_size_fixed_b8_9f3194be.jsonl`

regime 및 수락 규칙 (D27/D28/D31)

```text
sha256          f10573dd4b0601d96774d79cfc99ec9ee0ef2f0ccdc31f9f312612b2919896cd
완료 / 실패     216 / 0
sweep_id        da27f6362e5b149b
aggregation_id  1fb5800f9248290c
git_commit      b444b890 (dirty)
seeds           [2, 3, 4]
experiment_id   10종
                14f7385478d473c5
                1557a59a7ca01673
                1d1b9947fd93c1ec
                2a29bd38d563bd15
                52e582c8044a23e0
                91e07aa0d07211a7
                aa6f13617094e943
                bb12a494bfef8d97
                d840cf452ffadcfc
                eb2f57ffbad2b90e
selection       open_loop -> open_loop[1]
selection       static -> static[4]
```

### `headroom_nonlinear-diagnostic_step_size_fixed_b8_2a09bd45.jsonl`

국소최소점 cap (D23)

```text
sha256          19390b4e9fcef0021f48dc00661c179f75297d1626e14bb6632e0bcdca442115
완료 / 실패     72 / 0
sweep_id        7106190ee84b4f4c
aggregation_id  ec5019fa2cc67699
git_commit      aac429e2 (dirty)
seeds           [2, 3, 4]
experiment_id   10종
                0566cacd125cdcbb
                210db6664bffcbf2
                339ac7dda0d4ca07
                4924921c77395980
                5f31db4e5f5a8756
                635c65850b755cd2
                b9b6e7b5e2c79268
                bd0ec1988f4efbb5
                f18cc7378a679b2d
                fee0a575bb699322
selection       open_loop -> open_loop[1]
selection       static -> static[6]
```

### `headroom_pilot_step_size_fixed_b4_44e3242e.jsonl`

탐색적. 2개 spec 이 포화됐다 (D19)

```text
sha256          ec49d9ee9b9c6a6ecb189f8adcf3aa5d39960bfa5b319d90b1ba8db9910cc961
완료 / 실패     46 / 0
sweep_id        
aggregation_id  
git_commit      b3f5a728
seeds           [0]
experiment_id   1종
                caab3a2f3787aa22
```

### `headroom_pilot_step_size_fixed_b4_8e9cdd02.jsonl`

탐색적. 2개 spec 이 포화됐다 (D19)

```text
sha256          15c508b6a94a7e4cd316cd7e1cb14f1a3d7b322e529fbebf52171f74314e21ac
완료 / 실패     1526 / 0
sweep_id        f44a46d969f9d414
aggregation_id  04cacc21f2256580
git_commit      31e634f7, 94dc2d72, 99d433a1, e78cf8df (dirty)
seeds           [0, 1, 2]
experiment_id   10종
                0566cacd125cdcbb
                0a63f5e6de3dd85a
                2ac639d38eb52956
                339ac7dda0d4ca07
                5f31db4e5f5a8756
                b9b6e7b5e2c79268
                bd0ec1988f4efbb5
                eab1697716b5da31
                ec9100265ffdc56c
                fee0a575bb699322
selection       open_loop -> open_loop[7]
selection       static -> static[7]
```

### `headroom_pilot_step_size_fixed_b4_9d725689.jsonl`

탐색적. 2개 spec 이 포화됐다 (D19)

```text
sha256          7c7a280d34bfe9b108335cc70a323338462808e413cafb6acac9e5ac24fd3eb0
완료 / 실패     324 / 0
sweep_id        6a1e925383730305
aggregation_id  ec5019fa2cc67699
git_commit      ed1781af (dirty)
seeds           [0, 1, 2]
experiment_id   14종
                0566cacd125cdcbb
                2826233797eda089
                339ac7dda0d4ca07
                3616296f28badfac
                59269fc9f1eb092f
                5f31db4e5f5a8756
                63616b9c749d09f1
                97fc68f631beada5
                b7d5307abcafe21d
                b999ff6f11767f44
                b9b6e7b5e2c79268
                bd0ec1988f4efbb5
                ... 2개 더
selection       open_loop -> open_loop[7]
selection       static -> static[7]
```

## 왜 `experiment_id` 가 여러 개인가

정체성이 3계층으로 분리되어 있다 (프로토콜 D13). 컨트롤러마다 실제 쓰는
optimizer 설정이 다르므로 `run_semantics_id` 도 다르다. **정상 동작이다.**

```text
run_semantics_id  이 컨트롤러가 실제 쓰는 optimizer 설정만
sweep_id          이번 실행이 요청한 run 집합
aggregation_id    집계 정책
```

`git_commit` 과 `code_dirty` 는 어떤 ID 에도 넣지 않고
`execution_provenance` 로 분리한다. 문서만 고쳐도 해시가 바뀌면
"어떤 집합을 요청했는가" 라는 의미가 깨진다.
