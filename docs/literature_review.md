# 선행연구 정리: 스마트카드 기반 경로선택모형 및 공간 유사도 매칭

## 1. 연구 배경

본 연구는 OTP K-best 대안경로와 스마트카드(SC) 실제 통행을 **복합 유사도 매칭**으로 연결하여 경로선택모형 학습용 데이터셋을 구축하고, 신규 수단(GTX) 도입 시 경로별 선택 확률을 예측하는 것을 목적으로 한다.

---

## 2. 선행연구 분류

### 2.1 소요시간 기반 확률적 매칭 (Travel Time Matching)

SC의 tap-in/tap-out 시간차를 OD별 가능 경로의 예상 소요시간과 비교하여 확률적으로 경로를 배정하는 접근법.

| 논문 | 저널 | 방법론 | 한계 |
|------|------|--------|------|
| Zhao et al. (2016) | IEEE TITS | 심천 지하철 AFC, 확률 모델로 OD별 승객 흐름을 경로/열차에 배분 | 단일 수단(지하철), 소요시간 단일 지표 |
| Hanseler et al. (2022) | European Transport Research Review | 열차 시간표 없이 AFC만으로 경로 추정, 실제 vs 예상 소요시간 비교 + walking time uncertainty 고려 | 단일 수단(철도) |

### 2.2 Choice Set 생성 + 경로선택모형 추정

K-shortest paths 등으로 대안경로를 생성한 후, SC 관측 통행을 매칭하여 MNL/Mixed Logit 모형을 추정하는 접근법.

| 논문 | 저널 | 방법론 | 특징 |
|------|------|--------|------|
| Nassir et al. (2014) | Transportation Planning & Technology | trip-chaining으로 SC 경로 추론 → choice set 생성 → logit 추정 | SC 기반 경로선택모형 초기 연구 |
| Sanchez-Martinez (2019) | Transportation | SC 전수 데이터로 OD별 관측 경로 비율 산출 → 모형 캘리브레이션 | 관측 확률 → 파라미터 추정 |
| Luo et al. (2023) | Transportation | 다수단 네트워크 경로선택모형 + SC 검증, First Preference Recovery 71.5% | multimodal MNL |
| Sfeir et al. (2025) | arXiv preprint | Generated vs Empirical choice set 비교, access/egress 가정 완화 | 우리 프레임과 가장 유사 |

### 2.3 학습 기반 / 행태 모형

시간에 따른 학습 효과, 경로 선택 이질성 등 행태적 요인을 반영하는 접근법.

| 논문 | 저널 | 방법론 | 특징 |
|------|------|--------|------|
| Yap et al. (2024) | Transportation | 신규 노선 개통 시 승객의 학습 과정 모형화 (IBL) | GTX 도입 시나리오와 관련 |
| Li et al. (2021) | Transportation Research Part C | Stickiness Index로 습관적 vs 다양한 경로 선택 패턴 구분 | 경로 선택 이질성 분석 |

### 2.4 공간/궤적 유사도 비교 방법론

경로 간 공간적 형태를 비교하는 유사도 측도(similarity measure)에 대한 연구.

| 논문 | 저널 | 방법론 | 적용 분야 |
|------|------|--------|----------|
| Magdy et al. (2015) | GIScience & Remote Sensing | DTW, LCSS, EDR, ERP, Frechet, Hausdorff 등 궤적 유사도 측도 비교 분석 | 궤적 클러스터링/분류 |
| Wang et al. (2021) | ACM Computing Surveys | 차량 궤적 유사도 분류 체계 (shape-based, spatiotemporal, semantic) | 차량 궤적 분석 |
| Lehmann et al. (2019) | ResearchGate | SMSM: 정류장(stops)과 이동(moves)을 모두 고려하는 시맨틱 궤적 유사도 | GPS 궤적 |

**주요 공간 유사도 측도:**

| 측도 | 개념 | 특징 |
|------|------|------|
| **Frechet Distance** | "개 산책" 비유: 두 경로를 동시에 걸을 때 필요한 최소 끈 길이 | 경로의 순서(방향성)를 고려, 가장 엄격 |
| **Hausdorff Distance** | 한 경로의 모든 점에서 다른 경로까지의 최대 최소 거리 | 순서 무시, 최악 케이스 측정 |
| **DTW (Dynamic Time Warping)** | 두 시계열의 최적 정렬, many-to-one 매칭 허용 | 길이가 다른 경로 비교에 적합 |
| **LCSS (Longest Common Subsequence)** | 공간 임계값 내 공통 부분 시퀀스 길이 | 노이즈에 강건, 정류장 시퀀스에 적합 |
| **EDR (Edit Distance on Real sequence)** | 두 궤적을 같게 만드는 최소 편집 연산 수 | 삽입/삭제/대체 비용 동일 |

---

## 3. 우리 연구와의 연결

### 3.1 전체 프레임워크 위치

```
                     ┌─────────────────────────────────────┐
                     │     Choice Set 생성 + 매칭 접근법    │
                     │  (Nassir 2014, Sfeir 2025 계열)     │
                     └──────────────┬──────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
Choice Set 생성              유사도 매칭                    모형 추정/예측
OTP K-best 경로              SC ↔ OTP 비교                  choice_prob → MNL
(Sfeir et al.)               (본 연구의 핵심)               (Sanchez-Martinez)
```

### 3.2 논문별 연결

#### Sfeir et al. (2025) — Choice Set 생성

| 구분 | Sfeir et al. | 본 연구 |
|------|-------------|---------|
| Choice Set | 네트워크 알고리즘 (K-shortest) | OTP K-best (GraphQL API) |
| 관측 데이터 | SC 기반 empirical set | SC(TCN) 실제 통행 |
| 비교 내용 | Generated vs Empirical 커버리지 비교 | OTP 대안 vs SC 통행 유사도 매칭 |
| 핵심 기여 | access/egress 가정 완화 → 커버리지 향상 | 정류장명 fuzzy 매칭 → 매칭률 향상 |

**연결**: 동일한 **Generated choice set + SC 관측 데이터** 구조. Sfeir et al.이 제기한 "choice set이 관측 경로를 충분히 포함하는가" 문제를 본 연구에서는 유사도 매칭률(93.4%)로 검증.

#### Luo et al. (2023) — 다수단 경로선택모형

| 구분 | Luo et al. | 본 연구 |
|------|-----------|---------|
| 네트워크 | 다수단 (버스+전철) | 다수단 (버스+전철+GTX) |
| 모형 | MNL 경로선택모형 | MNL/Mixed Logit/NN (예정) |
| 검증 | First Preference Recovery 71.5% | 유사도 매칭률 93.4% |
| SC 활용 | 모형 **검증**용 | 모형 **학습 데이터 생성**용 |
| 피처 | 소요시간, 환승, 요금 | 21개 경로피처 + 5개 컨텍스트 |

**연결**: 동일한 **multimodal 맥락**. Luo et al.은 SC를 모형 검증에 사용한 반면, 본 연구는 SC를 **학습 데이터 생성에 직접 활용**하여 choice_prob을 산출하는 점이 차별화.

#### Sanchez-Martinez (2019) — SC 전수 데이터 → 선택 확률

| 구분 | Sanchez-Martinez | 본 연구 |
|------|-----------------|---------|
| 데이터 규모 | SC 전수 (population) | SC 7일 ~77M건 |
| 확률 산출 | OD별 관측 경로 비율 → 확률 | OD별 매칭 횟수 비율 → choice_prob |
| 모형 추정 | 관측 확률로 파라미터 캘리브레이션 | choice_prob을 y값으로 모형 학습 |
| 경로 식별 | SC에서 직접 경로 추론 | SC↔OTP 유사도 매칭으로 경로 식별 |

**연결**: 가장 유사한 접근. **SC에서 OD별 경로 선택 비율을 산출하여 모형을 추정**하는 것이 본 연구의 choice_prob 방식과 동일한 철학. 다만 본 연구는 SC에서 경로를 직접 추론하기 어려운 multimodal 환경이므로 **유사도 매칭이라는 중간 단계**가 추가됨.

#### Hanseler et al. (2022) — 소요시간 기반 경로 추정

| 구분 | Hanseler et al. | 본 연구 |
|------|----------------|---------|
| 매칭 기준 | 실제 vs 예상 소요시간 비교 (1개 지표) | 6개 복합 유사도 지표 |
| 불확실성 | walking time uncertainty 고려 | 정류장명 fuzzy 매칭으로 불확실성 대응 |
| 대상 | 철도 단일 수단 | 버스+전철+GTX 다수단 |

**연결**: Hanseler et al.의 소요시간 비교는 본 연구의 `sim_time` 지표에 해당. 본 연구는 다수단 환경의 복잡성 때문에 소요시간 단독으로는 부족하여 **수단조합, 노선, 정류장 시퀀스, 공간 경로 등 6개 지표를 복합 가중**하는 방식으로 확장.

#### Yap et al. (2024) — 신규 수단 도입 학습 효과

| 구분 | Yap et al. | 본 연구 |
|------|-----------|---------|
| 신규 수단 | 신규 지하철 노선 개통 | GTX 도입 |
| 접근 | 승객 학습 과정 모형화 (IBL) | 경로 속성 기반 확률 예측 |
| 목적 | 시간에 따른 선택 변화 분석 | 신규 수단 포함 경로의 선택 확률 배정 |

**연결**: 본 연구의 최종 목적인 **"GTX 도입 시 경로 선택 확률 예측"**과 직접 연결. Yap et al.은 신규 노선에 대한 승객의 학습 과정을 다루고, 본 연구는 경로 속성(시간, 요금, 환승)으로 확률을 예측하는 **보완적 접근**.

---

## 4. 공간 유사도 비교

### 4.1 본 연구의 공간 유사도 (`sim_spatial`)

본 연구에서는 두 가지 공간 데이터 소스를 활용하여 OTP 경로와 SC 경로의 **공간적 경로 형태**를 비교한다.

**1단계: OTP legGeometry polyline (기본, GTFS 불필요)**
```
OTP 대안경로 → legGeometry.points (Google Encoded Polyline) 디코딩
  → 실제 도로/선로를 따른 상세 경로 좌표 [(lat, lon), ...]
SC 실제 통행 → OD 좌표 + 환승 정류장 좌표 (알려진 좌표)
  → SC 좌표들이 OTP polyline에 얼마나 가까운지 측정 (points-on-path)
```

**2단계: GTFS shapes 확장 (선택, 양방향 Hausdorff)**
```
OTP 대안경로 → GTFS polyline (정류장 시퀀스 → shape 좌표)
SC 실제 통행 → GTFS polyline (노선+정류장 → shape 좌표)
  → 두 polyline 간 양방향 Hausdorff 거리 기반 유사도
```

**가중치**:
- GTFS 사용 시: 전체 유사도의 50% (양방향 Hausdorff)
- GTFS 미사용 시: 전체 유사도의 30% (legGeometry points-on-path)

**변별력**: 같은 OD라도 다른 경로를 이용하면 환승 정류장이 OTP polyline에서 멀어지므로, GTFS 없이도 경로 간 공간적 차이를 포착할 수 있다.

### 4.2 선행연구와의 공간 유사도 비교

| 구분 | 선행연구 주류 접근 | 본 연구 |
|------|-------------------|---------|
| 데이터 소스 | GPS 궤적, AVL 데이터 | OTP legGeometry polyline + GTFS shapes |
| 비교 대상 | 차량 GPS 궤적 vs 궤적 | OTP polyline vs SC 좌표/polyline |
| 사용 측도 | Frechet, Hausdorff, DTW, LCSS | points-on-path (기본) + Hausdorff (GTFS) |
| GTFS 의존성 | 없음 (GPS 직접 사용) | 1단계 불필요, 2단계 선택적 |
| 정류장 매칭 | 좌표 기반 자동 매칭 | 이름 기반 fuzzy 매칭 + 좌표 보완 |

### 4.3 공간 유사도 관련 선행연구 현황

**SC 데이터 + 공간 유사도 직접 비교**를 다룬 논문은 희소하다. 대부분의 공간 유사도 연구는 GPS 궤적(차량, 택시, 보행자) 간 비교에 집중되어 있으며, SC 데이터는 좌표가 아닌 정류장/노선 수준의 이산적(discrete) 정보만 포함하기 때문이다.

**관련 연구 흐름:**

1. **궤적 유사도 측도 비교 연구** (Magdy et al. 2015, Wang et al. 2021)
   - Frechet, Hausdorff, DTW, LCSS, EDR 등 다양한 측도의 특성 분석
   - 주로 연속적 GPS 궤적에 적용
   - 본 연구는 이산적 정류장 시퀀스를 GTFS로 연속 좌표화한 후 비교하는 **하이브리드 접근**

2. **대중교통 궤적 복원 연구** (transit route reconstruction)
   - GTFS shapes를 이용해 노선의 물리적 경로를 polyline으로 복원
   - OTP의 legGeometry는 라우팅 엔진이 직접 생성한 상세 경로 좌표로, GTFS shapes보다 정밀
   - 본 연구의 `gtfs_lookup.py`(GTFS 확장)와 `legGeometry`(OTP 직접) 두 경로를 모두 활용

3. **시맨틱 궤적 유사도** (Lehmann et al. 2019 — SMSM)
   - 정류장(stops)과 이동(moves)을 모두 고려
   - 본 연구의 다중 유사도 접근(수단+시퀀스+시간+노선+공간)과 개념적 유사

### 4.4 본 연구의 차별점

기존 공간 유사도 연구가 GPS 궤적에 집중하는 반면, 본 연구는:

1. **OTP legGeometry 직접 활용**: GTFS 없이도 OTP 라우팅 엔진의 상세 경로 좌표(polyline)를 활용하여 공간 비교 가능
2. **SC 이산 데이터 → 연속 좌표 비교**: SC의 OD/환승 정류장 좌표를 OTP polyline과 직접 비교하는 points-on-path 접근
3. **2단계 fallback 구조**: GTFS 가용 시 양방향 Hausdorff, 미가용 시 legGeometry points-on-path로 항상 공간 유사도 산출
4. **공간 유사도 + 비공간 유사도 복합**: 공간 정보만으로는 불충분한 다수단 환경에서 6개 지표를 가중 결합

---

## 5. 종합: 본 연구의 위치와 기여

### 5.1 선행연구 한계와 본 연구의 대응

| 선행연구 한계 | 본 연구의 대응 |
|-------------|----------------|
| 단일 수단(철도/지하철) 위주 | 버스+전철+GTX **다수단** |
| 소요시간 단일 지표 매칭 | **6개 복합 유사도 지표** (수단, 시퀀스, 시간, 노선, 공간, 환승) |
| SC에서 경로 직접 추론 가정 | **OTP K-best + 유사도 매칭** 2단계 접근 |
| Choice set 커버리지 미검증 | 매칭률 **93.4%**로 커버리지 검증 |
| 기존 수단만 분석 | **신규 수단(GTX) 도입 예측** 목적 |
| GPS 궤적 기반 공간 비교 | **OTP legGeometry + GTFS polyline 기반 공간 유사도** (GTFS 없이도 가능) |

### 5.2 방법론적 기여

1. **복합 유사도 매칭 프레임워크**: 소요시간뿐 아니라 수단조합, 노선, 정류장 시퀀스, 공간 경로를 복합적으로 고려하는 다차원 매칭
2. **OTP legGeometry + GTFS 기반 공간 유사도**: OTP 라우팅 엔진의 상세 경로 좌표를 직접 활용하고, GTFS 가용 시 양방향 Hausdorff로 정밀 비교하는 2단계 공간 유사도
3. **OD별 선택 확률 산출**: 개별 binary choice가 아닌 OD별 aggregate choice_prob으로 모형 학습 데이터 생성

---

## References

1. Zhao, J., Zhang, F., Tu, L., Xu, C., Shen, D., Tian, C., Li, X.Y., & Li, Z. (2016). Estimation of Passenger Route Choice Pattern Using Smart Card Data for Complex Metro Systems. *IEEE Transactions on Intelligent Transportation Systems*, 18(4), 790-801. https://ieeexplore.ieee.org/document/7534790/

2. Hanseler, F. S., et al. (2022). Route choice estimation in rail transit systems using smart card data: handling vehicle schedule and walking time uncertainties. *European Transport Research Review*, 14, 30. https://link.springer.com/article/10.1186/s12544-022-00558-x

3. Nassir, N., Hickman, M., & Ma, Z.L. (2014). Estimation of a route choice model for urban public transport using smart card data. *Transportation Planning and Technology*, 37(7), 638-648. https://www.tandfonline.com/doi/abs/10.1080/03081060.2014.935570

4. Sanchez-Martinez, G. E. (2019). Calibration of a transit route choice model using revealed population data of smartcard in a multimodal transit network. *Transportation*, 47, 2479-2504. https://link.springer.com/article/10.1007/s11116-019-10008-8

5. Luo, D., et al. (2023). Validation of a multi-modal transit route choice model using smartcard data. *Transportation*, 51, 1933-1962. https://link.springer.com/article/10.1007/s11116-023-10387-z

6. Sfeir, G., Rodrigues, F., Seshadri, R., & Azevedo, C. L. (2025). Choice Sets and Smart Card Data in Public Transport Route Choice Models: Generated vs. Empirical Sets. *arXiv preprint*, arXiv:2503.17370. https://arxiv.org/abs/2503.17370

7. Yap, M., et al. (2024). An experiential learning-based transit route choice model using large-scale smart-card data. *Transportation*. https://link.springer.com/article/10.1007/s11116-024-10465-w

8. Li, W., et al. (2021). Unveiling route choice strategy heterogeneity from smart card data in a large-scale public transport network. *Transportation Research Part C*, 134, 103469. https://www.sciencedirect.com/science/article/abs/pii/S0968090X2100454X

9. Magdy, N., Sakr, M. A., & El-Bahnasy, K. (2015). A comparative analysis of trajectory similarity measures. *GIScience & Remote Sensing*, 58(5), 643-669. https://www.tandfonline.com/doi/full/10.1080/15481603.2021.1908927

10. Wang, S., Bao, Z., Culpepper, J. S., & Cong, G. (2021). Vehicle Trajectory Similarity: Models, Methods, and Applications. *ACM Computing Surveys*, 53(4), 1-37. https://dl.acm.org/doi/fullHtml/10.1145/3406096
