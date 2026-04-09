# 경로 배정 및 선택 패턴 분석 결과

**날짜**: 2026-04-08  
**기반 모형**: Spec A (Total IVT only) unconstrained  
**데이터**: 444,295 ODs, 1,217,980 대안 (OTP 캐시 전체)  
**라벨 데이터**: 319,660 ODs, 892,557 rows (학습 parquet inner join)

---

## 1. 최종 모형 선정

| 항목           | 값                                                 |
| -------------- | -------------------------------------------------- |
| Specification  | A_total_uncon (Total IVT + transfers + fare + ASC) |
| ρ²             | 0.346                                              |
| Top-1 Accuracy | 67.0%                                              |

**Beta (추정값)**:

| 변수                   | β       | 해석                                  |
| ---------------------- | ------- | ------------------------------------- |
| total_ivt_min          | -0.0195 | 차내시간 1분당 효용 감소              |
| transfer_walk_time_min | -0.198  | 환승도보 1분당 효용 감소 (IVT의 10배) |
| num_transfers          | -3.048  | 환승 1회당 효용 감소 (매우 큰 패널티) |
| fare_1000won           | -0.882  | 1000원당 효용 감소                    |
| ASC_bus+train          | +2.324  | bus_only 대비 선호                    |
| ASC_train+gtx          | +5.497  | GTX 포함 경로 강한 선호               |
| ASC_train_only         | +5.626  | 철도 전용 경로 강한 선호              |

**선정 사유**: Spec B (IVT split)는 수단별 IVT와 num_transfers 간 구조적 다중공선성으로 불안정. Total IVT가 안정적이고 해석 가능.

---

## 2. 배정 방법론

1. `mnl_4spec.json`에서 beta 벡터 로드
2. OTP 캐시 DB 스트리밍 (배치 5,000)
3. 각 OD: 피처 변환 → V = X·β → softmax → 확률
4. 확률 합 = 1.0 검증 완료

**출력**: `assignment_A_total_uncon.parquet` (1,217,980 rows)

---

## 3. 선택 기준 분석

### Table 1: 단일 기준 일치율

| 기준                      | 실제 선택 일치% | 모델 Top-1% |
| ------------------------- | :-------------: | :---------: |
| min_transfers             |    **97.2%**    |    99.8%    |
| min_fare                  |    **93.1%**    |    94.5%    |
| min_walk                  |    **86.7%**    |    91.5%    |
| min_gc (generalized cost) |      52.6%      |    73.1%    |
| min_duration              |      46.8%      |    64.8%    |

**해석**:

- 승객은 **환승 최소화**(97%)와 **요금 최소화**(93%)를 가장 우선시
- Raptor의 generalized cost 최소 경로가 실제 선택과 일치하는 비율은 53%에 불과
- 모델은 GC 일치율을 73%로 과대추정 → 모델이 GC에 과도하게 의존

### Table 2: 교차표 (실제 선택)

실제 선택된 경로가 각 기준을 동시에 만족하는 비율:

|               | min_gc | min_transfers | min_walk | min_duration | min_fare |
| ------------- | :----: | :-----------: | :------: | :----------: | :------: |
| min_gc        |  100   |     98.0      |   92.4   |     85.8     |   96.3   |
| min_transfers |  53.1  |      100      |   88.0   |     46.8     |   93.4   |
| min_walk      |  56.1  |     98.6      |   100    |     49.9     |   94.5   |
| min_duration  |  96.6  |     97.3      |   92.5   |     100      |   96.8   |
| min_fare      |  54.5  |     97.5      |   88.0   |     48.6     |   100    |

**해석**: min_duration인 경로가 min_gc일 확률은 96.6%지만, 그 역은 85.8%. GC는 duration보다 넓은 개념.

### Table 3: Choice Set 크기별

| CS 크기 | min_gc | min_transfers | min_walk | min_duration | min_fare |
| :-----: | :----: | :-----------: | :------: | :----------: | :------: |
|    2    | 63.1%  |     98.3%     |  90.2%   |    57.1%     |  94.4%   |
|    3    | 49.5%  |     96.7%     |  85.5%   |    43.0%     |  92.7%   |
|    4    | 37.5%  |     95.6%     |  82.3%   |    32.1%     |  91.1%   |
|    5    | 23.4%  |     94.9%     |  76.1%   |    21.5%     |  89.6%   |

대안이 많을수록 모든 기준의 일치율 감소 → 다기준 trade-off가 복잡해짐

### Table 4: 수송 카테고리별

| 카테고리      | min_gc | min_transfers | min_duration | min_fare  | min_walk  |
| ------------- | :----: | :-----------: | :----------: | :-------: | :-------: |
| bus_only      | 59.2%  |   **99.1%**   |    52.2%     | **96.8%** | **96.5%** |
| bus+train     | 39.4%  |     91.3%     |    37.1%     |   89.5%   |   60.0%   |
| train_only    | 35.4%  |     96.4%     |    29.8%     |   76.6%   |   75.1%   |
| train+gtx     | 56.4%  |     50.8%     |  **56.8%**   |   22.5%   |   49.2%   |
| bus+gtx       | 43.1%  |     96.1%     |    41.2%     |   56.9%   |   45.1%   |
| bus+train+gtx | 35.8%  |     27.2%     |    33.3%     |   29.6%   |   28.4%   |

**해석**: bus_only는 단순해서 높은 일치율, 복합수단(bus+train+gtx)은 모든 기준 30% 이하.

---

## 4. Raptor GC 파라미터 보정

### 현재 Raptor GC 공식

```
GC = boardCost + transferCost × n_transfers
     + waitTime × waitReluctance
     + transitTime × transitReluctance
     + walkTime × walkReluctance
```

**현재 설정** (`router-config.json`): board=60s, transfer=120s, 모든 reluctance=1.0

### 보정 결과

Grid search (1,800+ 조합) → 최적 파라미터:

| 파라미터          | 현재값 |  최적값  | 변화      |
| ----------------- | :----: | :------: | --------- |
| boardCost         |  60s   |   60s    | -         |
| transferCost      |  120s  | **600s** | ×5 증가   |
| waitReluctance    |  1.0   |   1.0    | -         |
| transitReluctance |  1.0   | **0.3**  | ×0.3 감소 |
| walkReluctance    |  1.0   | **2.0**  | ×2 증가   |

| 지표                   | 현재  |   최적    |    개선     |
| ---------------------- | :---: | :-------: | :---------: |
| GC-min = Chosen 일치율 | 48.4% | **66.0%** | **+17.6%p** |

### 해석

1. **환승 패널티 600s** (10분): MNL β_transfers=-3.048이 암시하는 것과 일치. 승객은 환승을 극도로 기피
2. **Transit reluctance 0.3**: 차내시간은 실제 시간의 30%로 인식 → "타고 있는 건 괜찮다"
3. **Walk reluctance 2.0**: 도보는 실제 시간의 2배로 인식 → 도보 부담 크게 느낌

### MNL Beta → 시간가치(VOT) 비율 비교

β_transfers/β_ivt = 3.048/0.0195 = **156분** (환승 1회 = 2.6시간 차내시간)  
β_walk/β_ivt = 0.198/0.0195 = **10.2배** (도보 1분 = 차내 10분)

Raptor 보정 결과와 정성적으로 일치:

- 환승 패널티가 매우 큼
- 도보 가중치가 차내시간보다 훨씬 높음

### router-config.json 제안

```json
{
  "walkReluctance": 2.0,
  "transitCost": {
    "firstBoardCostSeconds": 60,
    "transferCostSeconds": 600,
    "waitReluctance": 1.0,
    "transitReluctanceForMode": {
      "SUBWAY": 0.3,
      "BUS": 0.3,
      "RAIL": 0.3,
      "GTX": 0.3
    }
  }
}
```

---

## 5. Raptor 실제 비교 검증

보정된 파라미터로 Rust Raptor를 실제 실행하여 SC 선택 경로와 비교 (2,000 ODs 샘플).

| 지표                        | Original (120s, 1.0) | Calibrated (600s, 0.3, 2.0) |    변화     |
| --------------------------- | :------------------: | :-------------------------: | :---------: |
| 평균 경로 수/OD             |         4.6          |             4.8             |    +0.2     |
| 환승 횟수 일치율            |        73.9%         |          **90.0%**          | **+16.1%p** |
| SC 경로가 Raptor set에 존재 |        98.4%         |            98.4%            |    동일     |

**해석**:

1. 보정 후 Raptor GC-최적 경로의 환승 횟수가 SC 실제 선택과 90% 일치
2. SC 경로 존재율 98.4%는 파라미터와 무관 → Raptor가 경로를 못 찾는 게 아니라 **랭킹 문제**
3. 나머지 10%는 GC 단일 지표의 한계 → MNL 확률 배정 필요성 재확인

---

## 6. 논문 서사 연결

1. **Raptor 현재 파라미터가 비현실적**: GC-min 일치율 48% → 반 이상의 OD에서 잘못된 경로를 "최적"으로 제시
2. **보정 후 66%**: 상당한 개선이지만 여전히 34%는 GC 외 요인 (습관, 정보 부족 등)
3. **MNL이 필요한 이유**: 단일 GC 최적화로는 설명 불가 → 다변량 확률 모형의 필요성 입증
4. **GTX 영향평가 시**: 보정된 파라미터로 Raptor 재실행 → MNL 배정 → 더 현실적인 수요 예측

---

## 6. 생성 파일

| 파일                                                     | 설명                |
| -------------------------------------------------------- | ------------------- |
| `data/training_set_new/assignment_A_total_uncon.parquet` | MNL 확률 배정 결과  |
| `data/training_set_new/choice_criterion_analysis.json`   | 선택 기준 분석 결과 |
| `data/training_set_new/raptor_calibration_results.json`  | Raptor 보정 결과    |
| `code/similarity/module/feature_transform.py`            | 공용 피처 변환 모듈 |
| `code/similarity/mnl/assign_mnl.py`                      | MNL 배정 스크립트   |
| `code/similarity/analysis/choice_criterion.py`           | 선택 기준 분석 모듈 |
| `code/similarity/analysis/run_choice_criterion.py`       | 선택 기준 분석 실행 |
| `code/similarity/analysis/raptor_calibration_prep.py`    | Raptor 보정 모듈    |
| `code/similarity/analysis/run_raptor_calibration.py`     | Raptor 보정 실행    |
