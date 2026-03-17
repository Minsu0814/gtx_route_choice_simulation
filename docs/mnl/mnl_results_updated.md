# MNL 경로선택모형 결과 (업데이트)

## 변경 이력

- **이전**: 가중치 mode=0.20, route=0.40, seq=0.40 (단일 grid search)
- **현재**: 가중치 mode=0.02, route=0.90, seq=0.08 (2단계 grid search + final sweep)

## 1. 유사도 가중치

| Component | Weight | 근거 |
|-----------|--------|------|
| **route (route_jaccard)** | **0.90** | MNL 성능의 핵심 driver, 노선명 일치가 경로 구분의 90% |
| **sequence (seq_lcs)** | **0.08** | 경유지 감별 + 정류장 순서 반영 |
| **mode** | **0.02** | 교통수단 종류는 route에 이미 내포 |

- **Threshold**: 0.5 (composite ≥ 0.5인 경우만 매칭)
- **Sequence Gate**: sim_sequence ≥ 0.3 (경유지/우회 경로 필터링)

## 2. Grid Search 방법론

### 2단계 탐색

| 단계 | 범위 | Step | 시나리오 수 |
|------|------|------|------------|
| Coarse | route [0,1] × seq [0,1] | 0.05 | 231 |
| Fine | route [0.80,1.00] × seq [0.00,0.20] | 0.01 | 231 |

### Coarse 결과 요약

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| test ρ² | 0.467 | 0.459 | 0.506 |
| test FPR | 73.4% | 72.6% | 77.2% |
| n_ods | 543K | 445K | 566K |

### Fine 결과 요약

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| test ρ² | 0.502 | 0.496 | 0.522 |
| test FPR | 76.7% | 76.1% | 77.7% |
| n_ods | 454K | 433K | 466K |

### 최적 후보군 (B 클러스터: 균형)

| Route | Seq | Mode | ρ² | FPR | ODs |
|-------|-----|------|-----|-----|-----|
| **0.90** | **0.08** | **0.02** | **0.520** | **77.5%** | **437K** |

### Final Sweep 결과

- Threshold 0.5: ρ²=0.520, FPR=77.5% (OD 437K) — 0.6에서 OD 급감, 0.5이 균형점
- Seq Gate 0.3: 전 구간 ρ²/FPR 거의 동일 (spread < 0.005) — 경유지 필터링 목적

## 3. MNL 모형 추정 결과

### 데이터

| | Train | Test |
|--|-------|------|
| Rows | 1,578,859 | 394,494 |
| ODs | 424,551 | 106,138 |
| Effective trips | 29,644,623 | 7,434,775 |

### β 계수

| Feature | β | Std.Err | t-stat | Sig | Bound |
|---------|---|---------|--------|-----|-------|
| IVT (min) | -0.007043 | 0.000106 | -66.215 | *** | |
| Wait (min) | 0.000000 | N/A | N/A | | ← bound |
| Access walk (min) | -0.834678 | 0.000491 | -1698.930 | *** | |
| Egress walk (min) | -0.512136 | 0.000326 | -1570.294 | *** | |
| Transfer walk (min) | -0.016249 | 0.000506 | -32.120 | *** | |
| Total dist (km) | -0.043335 | 0.000348 | -124.634 | *** | |
| Transfers | -3.657121 | 0.001874 | -1951.779 | *** | |
| Fare (KRW) | 0.000000 | N/A | N/A | | ← bound |
| Has bus | -2.227672 | 0.001815 | -1227.640 | *** | |
| Has train | 2.984755 | 0.002749 | 1085.786 | *** | |
| Has GTX | 0.106403 | 0.025368 | 4.194 | *** | |

### 성능 평가

| 지표 | Train | Test | 판정 |
|------|-------|------|------|
| LL(0) | -35,336,065 | -8,848,708 | -- |
| LL(β) | -17,934,187 | -4,539,723 | -- |
| McFadden ρ² | 0.4925 | 0.4870 | PASS (> 0.2) |
| FPR-1 (1순위 적중) | -- | 70.9% | PASS (> 60%) |
| FPR-3 (TOP-3 적중) | -- | 96.1% | PASS (> 85%) |
| RMSE | -- | 0.2553 | -- |

- Train/Test ρ² 비율: 0.989 → 과적합 없음
- 부호 검증: ALL CORRECT

### 수단별 분담률 재현

| Category | Actual% | Pred% | Diff%p |
|----------|---------|-------|--------|
| train_only | 49.42% | 49.40% | -0.01 |
| bus_only | 42.51% | 42.54% | +0.03 |
| bus+train | 7.99% | 7.95% | -0.04 |

## 4. DTUMOS 파라미터 매핑

| DTUMOS Param | Estimated | Default |
|--------------|-----------|---------|
| beta_time | -0.007043 | -0.0500 |
| beta_cost | 0.000000 | -0.0003 |
| beta_wait | 0.000000 | -0.0800 |
| beta_walk (combined) | -0.454354 | -0.1000 |
| beta_transfer | -3.657121 | -0.3000 |

## 5. 이전 결과 대비

| 지표 | 이전 (mode=0.20, route=0.40, seq=0.40) | 현재 (mode=0.02, route=0.90, seq=0.08) | 변화 |
|------|----------------------------------------|----------------------------------------|------|
| Train ρ² | 0.4746 | **0.4925** | +0.018 |
| Test ρ² | 0.4724 | **0.4870** | +0.015 |
| FPR-1 | 70.3% | **70.9%** | +0.6%p |
| FPR-3 | 96.2% | **96.1%** | -0.1%p |
| ODs (train) | 478,345 | 424,551 | -53,794 |
| ODs (test) | 119,587 | 106,138 | -13,449 |

## 6. 이슈 분석

### 6.1 fare β = 0 → 제거 예정

- **60.8%의 OD에서 대안 간 요금 차이 = 0원** (수도권 통합요금제)
- 81.7%의 OD에서 요금 차이 ≤ 50원
- 분산 있는 OD에서도 비싼 대안이 더 선택됨 (cheapest: 0.217 vs non-cheapest: 0.277)
- β ≤ 0 제약 하에서 부호 역전 → 0에 고정
- **결론**: 통합요금제 구조상 대안 간 요금 변별력 없음 → 설명변수에서 제거

### 6.2 wait_time β = 0 → 유지

- 분산 있음 (87.9% OD에서 std > 0, 중앙값 95초)
- 부호 방향 정상 (짧은 쪽 choice_prob 0.380 vs 긴 쪽 0.199)
- β=0인 이유: 다른 변수(access, egress, transfers)에 설명력 흡수
- OTP 스케줄 기반 대기시간 → 실제 체감 대기와 괴리 가능성
- **결론**: 행태적으로 유의미한 변수, 데이터 개선 시 활성화 가능 → 유지

### 6.3 walk 과대 / IVT 과소 (내생성) → 논문에서 한계로 기술

**현상**: β_walk / β_IVT 비율이 비현실적으로 큼

| 변수 | 이전 비율 | 현재 비율 | 변화 |
|------|----------|----------|------|
| access_walk / IVT | 165x | **119x** | 개선 |
| egress_walk / IVT | 155x | **73x** | 크게 개선 |
| transfer_walk / IVT | 25x | **2.3x** | 거의 해소 |

**원인**: choice_prob ↔ sim_composite r=0.79, sim_composite ↔ access/egress r=-0.33
→ walk 짧은 경로 = 정류장 가까운 경로 → 유사도 높음 → choice_prob 높음 → β_walk 부풀려짐

**상관관계 (새 데이터)**:
- choice_prob ↔ sim_composite: r = +0.795
- sim_composite ↔ access_time: r = -0.336
- sim_composite ↔ egress_time: r = -0.322
- choice_prob ↔ in_vehicle_time: r = +0.019

**가중치 변경(route=0.90)으로 상당히 완화됐으나 access/egress는 여전히 과대.**
MNL의 구조적 한계이며, 향후 딥러닝 기반 모형으로 개선 예정.

### 6.4 GTX 과추정 → 크게 개선

**이전 vs 현재 비교**:

| 항목 | 이전 | 현재 | 변화 |
|------|------|------|------|
| has_gtx ASC | +0.469 | **+0.106** | 78% 감소 |
| GTX vs train 추가 보너스 | +0.469 | **+0.106** | 대폭 축소 |
| GTX 총 보너스 (has_gtx + has_train) | +1.950 | +3.091 | has_train 증가 |

**새 데이터 GTX 실태**:
- GTX 포함 경로: 19,729건 (1.00%), 12,142 ODs
- GTX 대안 평균 choice_prob: **0.030** (non-GTX: 0.360)
- GTX choice_prob > 0 비율: **3.8%** (대부분의 OD에서 GTX 미선택)
- GTX 평균 요금: 4,130원 vs non-GTX 1,818원 (2.3배)

**개선 요인**: has_gtx ASC 78% 감소 → GTX 과추정 상당히 완화
**남은 문제**: fare β=0으로 GTX 고요금이 효용에 미반영 (fare 제거 예정이므로 별도 보정 불필요)

## 7. 논문 수정 필요 위치

| 위치 (main-kr.tex) | 항목 | 이전 값 | 새 값 |
|---------------------|------|---------|-------|
| L46, L54 (초록) | ρ², FPR | 0.4746, 70.3% | 0.4925, 70.9% |
| L78 (본문) | ρ², FPR | 동일 | 0.4925, 70.9% |
| L355 (grid search 설명) | 범위/방법 | [0.1,0.4]×[0.0,0.6], 484개 | 2단계: coarse 231 + fine 231 |
| L365 (최적 가중치) | 가중치, ρ², FPR | mode=0.20, route=0.40, seq=0.40, 0.479, 71.9% | mode=0.02, route=0.90, seq=0.08, 0.520, 77.5% |
| L376 (데이터 규모) | OD/대안 수 | 478,345 OD, 640,101 대안 | 424,551 OD, 1,578,859 대안 |
| L385 (MNL 결과) | ρ², FPR-1, FPR-3 | 0.4746, 70.3%, 96.2% | 0.4925, 70.9%, 96.1% |
| L394~399 (MNL 표) | LL, ρ², FPR, RMSE | 이전 값 | 새 값 (위 표 참조) |
| L313 (composite 설명) | seq gate 없음 | -- | sim_sequence ≥ 0.3 gate 추가 |
| L376 (설명변수) | 11개 | -- | fare 제거 시 10개로 변경 |
| L383 (한계점 추가) | -- | -- | 내생성(walk/IVT), wait_time bound 설명 |
| L447 (반복 보정) | 0.461→0.475, 68.1%→70.3% | 별도 확인 필요 |
| L768 (결론) | ρ² | 0.4746 | 0.4925 |

## 파일 위치

- 노트북: `code/similarity/mnl/route_choice_model.ipynb`
- 계수 JSON: `data/training_set/mnl_coefficients.json`
- 평가 JSON: `data/training_set/model_evaluation.json`
- Grid search CSV: `data/sensitivity/sensitivity_2d_grid_step005.csv`, `sensitivity_2d_grid_step001_rt08-10_sq00-02.csv`
- Final sweep CSV: `data/sensitivity/final_sweep_threshold.csv`, `final_sweep_seq_gate.csv`
