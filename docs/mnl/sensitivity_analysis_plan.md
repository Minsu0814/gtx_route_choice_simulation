# Sensitivity Analysis 계획

## 목적

저널 리뷰어 대응 — 3가지 파라미터의 임의성(arbitrariness)에 대한 robustness 증명

## 대상 파라미터

### #3. 유사도 가중치

- 현재: mode=0.10, seq=0.15, time=0.10, route=0.15, spatial=0.50
- 문제: "왜 이 가중치인가?" 이론적 근거 없음
- 대응: 8개 가중치 조합으로 모형 재추정, 결과 비교

### #4. 매칭 임계값

- 현재: composite ≥ 0.6이면 매칭 수락
- 문제: "왜 0.6인가?" 근거 없음
- 대응: 0.4~0.8 범위에서 5개 값 테스트

### #10. 공간 정규화 거리

- 현재: sim_spatial = max(0, 1 - hausdorff / 5000m)
- 문제: "왜 5km인가?" 근거 없음
- 대응: 2km~10km 범위에서 5개 값 테스트

## 방법: 3개 독립 1D Sweep

파라미터 하나만 변화시키고 나머지는 baseline 고정 (표준 sensitivity analysis)

### Sweep A: 가중치 (8개 시나리오)

| 이름          | mode | seq  | time | route | spatial | 의도                  |
| ------------- | ---- | ---- | ---- | ----- | ------- | --------------------- |
| baseline      | 0.10 | 0.15 | 0.10 | 0.15  | 0.50    | similarity.py 기본값  |
| build_actual  | 0.14 | 0.21 | 0.14 | 0.21  | 0.30    | 실제 학습에 사용된 값 |
| equal         | 0.20 | 0.20 | 0.20 | 0.20  | 0.20    | 사전 가정 없음        |
| no_spatial    | 0.20 | 0.30 | 0.15 | 0.35  | 0.00    | spatial 제거          |
| spatial_heavy | 0.05 | 0.10 | 0.05 | 0.10  | 0.70    | spatial 극대화        |
| route_heavy   | 0.10 | 0.15 | 0.10 | 0.40  | 0.25    | 노선 중심             |
| seq_heavy     | 0.10 | 0.40 | 0.10 | 0.15  | 0.25    | 정류장 순서 중심      |
| time_heavy    | 0.10 | 0.15 | 0.40 | 0.10  | 0.25    | 시간 중심             |

### Sweep B: 임계값 (5개 값)

0.4, 0.5, **0.6**(baseline), 0.7, 0.8

### Sweep C: 정규화 거리 (5개 값)

2km, 3km, **5km**(baseline), 7km, 10km

## 핵심 아이디어: Full pipeline 재실행 불필요

`route_choice_training.parquet`에 component scores가 이미 저장됨:

- `sim_mode`, `sim_sequence`, `sim_time`, `sim_route`, `sim_spatial`

따라서:

1. **가중치 변경**: `new_composite = Σ w_i × sim_i` 로 재계산
2. **임계값 변경**: new_composite ≥ threshold로 필터링
3. **정규화 거리 변경**: `raw_hausdorff = (1 - sim_spatial) × 5000`로 역산 후 `new_spatial = max(0, 1 - raw_hausdorff / new_norm_dist)`로 재계산

→ 각 시나리오별 ~1분 (MNL 추정 포함), 총 18개 시나리오 약 15~20분

## 시나리오별 파이프라인

```
1. 데이터 로드 (2.4M rows)
2. raw_hausdorff 역산 (sim_spatial → 원래 거리)
3. sim_spatial 재계산 (norm_dist 변경 시)
4. composite 재계산 (가중치 적용)
5. 필터링 (threshold 적용, OD당 ≥2 대안 유지)
6. Train/Test split (GroupShuffleSplit, random_state=42)
7. MNL 추정 (L-BFGS-B, 11 피처, 부호 제약)
8. 평가 (ρ², FPR, RMSE, key β)
```

## 기록할 지표

| 지표           | 설명                      |
| -------------- | ------------------------- |
| n_ods          | 필터링 후 남은 OD 수      |
| train_rho_sq   | 학습 McFadden ρ²          |
| test_rho_sq    | 테스트 McFadden ρ²        |
| test_fpr       | First Preference Recovery |
| test_rmse      | RMSE                      |
| beta_IVT       | 차내시간 계수             |
| beta_access    | 접근보행 계수             |
| beta_transfers | 환승 계수                 |
| beta_bus       | 버스 더미 계수            |
| beta_train     | 철도 더미 계수            |

## 결과 해석 기준

- **β의 CV(변동계수) < 20%** → robust (계수 안정적)
- **ρ² 변동폭 < 0.05** → robust (적합도 안정적)
- **FPR 변동폭 < 5%p** → robust (예측 정확도 안정적)
- 모든 시나리오에서 ρ² > 0.2, FPR > 60% → 기준 충족

## 출력물

### CSV (논문 Table용)

- `data/sensitivity/sensitivity_weights.csv`
- `data/sensitivity/sensitivity_threshold.csv`
- `data/sensitivity/sensitivity_normdist.csv`

### Figure (논문 Figure용)

- `fig_sensitivity_weights.png` — 가중치별 ρ², FPR, 샘플수
- `fig_sensitivity_threshold.png` — 임계값별 ρ², FPR, 샘플수
- `fig_sensitivity_normdist.png` — 정규화거리별 ρ², FPR, 샘플수
- `fig_beta_stability.png` — 전 시나리오 β 안정성

### JSON (전체 데이터)

- `data/sensitivity/sensitivity_all.json`

## Limitation

**Option A (pragmatic approach)** 사용:

- 가중치/임계값 변경이 OD-level 필터링만 변경
- trip-level 재매칭(어떤 대안이 best match인지)은 변경하지 않음
- 155M행 개별 재매칭은 시간상 불가 (~8시간/시나리오)
- choice_prob(y)는 기존 값 유지

이 한계는 스크립트 docstring과 논문에 명시.
가중치 변경이 필터링에만 영향을 미치므로, "매칭 품질 기준이 달라져도 모형 결과가 안정적인가?"를 검증하는 것으로 해석.
