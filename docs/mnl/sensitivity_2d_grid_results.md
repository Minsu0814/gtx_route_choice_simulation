# 2D Grid Sensitivity Analysis Results

## 개요

similarity composite 가중치 (mode, route, sequence)에 대한 2D grid sensitivity 분석.
MNL 모델 성능이 가중치 조합에 얼마나 민감한지 검증.

- **Grid**: route × sequence (mode = 1 - route - seq)
- **Step**: 0.05 coarse (231 scenarios) → 0.01 fine-grained (231 scenarios)
- **Threshold**: composite ≥ 0.5
- **Sequence Gate**: sim_sequence ≥ 0.3 (경유지/우회 경로 필터)
- **Train/Test Split**: 80/20 by OD hash (seed=42)
- **Data**: route_choice_training.parquet (2,232,801 rows, ~598K ODs)

## 최종 확정 가중치

| Component | Weight | 근거 |
|-----------|--------|------|
| **route (route_jaccard)** | **0.89** | MNL 성능의 핵심 driver |
| **sequence (seq_lcs)** | **0.11** | 경유지 감별 + 정류장 순서 반영 |
| **mode** | **0.00** | 교통수단 종류는 route에 이미 내포 |

- **ρ² = 0.4919**, **FPR = 90.3%**, RMSE = 0.187, ODs = 329K
- Sequence gate (sim_sequence ≥ 0.3) 별도 적용

### 가중치 선정 논거

1. **Route similarity가 지배적**: route weight가 높을수록 ρ²와 FPR 모두 단조 증가
2. **Sequence의 역할**: 가중치 자체보다 경유지/우회 경로 필터링(gate)에서 핵심적
   - sim_sequence ≥ 0.3 gate로 경유지 1,064건 100% 제거, 정상 경로 손실 0%
3. **Mode 불필요**: mode=0으로 해도 성능 저하 없음. route_jaccard가 노선명 집합을 비교하므로 교통수단 정보가 이미 내포됨
4. **과적합 없음**: 전 시나리오에서 test ρ² > train ρ² (gap ≈ -0.004). 파라미터 11개 vs OD 33만개 (ratio 29,537:1)

### 두 개의 성능 클러스터

| Cluster | Route | Seq | ρ² | FPR | 특징 |
|---------|-------|-----|-----|-----|------|
| A (ρ² 최대) | 0.96~0.99 | 0.01~0.04 | 0.495~0.497 | 90.4~90.5% | seq 가중치 너무 낮아 방어 어려움 |
| **B (균형)** | **0.84~0.91** | **0.08~0.16** | **0.489~0.494** | **90.2~90.4%** | **seq 10%+ 로 설명 가능** |

→ **B 클러스터의 route=0.89, seq=0.11 선택** (ρ² 0.005 손해로 seq 가중치의 논리적 방어 확보)

## Coarse Grid (step=0.05) 결과

### 전체 성능 요약

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| test ρ² | 0.4732 | 0.0083 | 0.4409 | 0.4950 |
| test FPR | 75.4% | 3.5%p | 73.6% | 90.4% |
| test RMSE | 0.219 | 0.014 | 0.186 | 0.229 |
| n_ods | 530K | 96K | 326K | 596K |

### Robustness Checks

| Check | Result |
|-------|--------|
| All ρ² > 0.2 | PASS |
| All FPR > 60% | PASS |
| ρ² spread < 0.05 | FAIL (0.054) — route weight 차이에 의한 구조적 차이 |
| FPR spread < 5%p | FAIL (16.8%p) — 동일 원인 |
| Converged | 231/231 (100%) |

## Fine-grained Grid (step=0.01, route 0.80~1.00, seq 0.00~0.20) 결과

### 전체 성능 요약

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| test ρ² | 0.4878 | 0.0035 | 0.4819 | 0.4971 |
| test FPR | 85.6% | 5.7%p | 78.1% | 90.5% |
| test RMSE | 0.190 | 0.002 | 0.185 | 0.193 |

### Beta Stability (fine-grained 영역)

| Beta | Mean | CV(%) | 판정 |
|------|------|-------|------|
| beta_access | -0.820 | 0.9% | ROBUST |
| beta_transfers | -3.174 | 0.8% | ROBUST |
| beta_bus | -2.781 | 3.0% | ROBUST |
| beta_train | 0.695 | 11.1% | ROBUST |
| beta_IVT | -0.000 | 164% | VARIABLE (값 ≈ 0) |

### Robustness (route ≥ 0.90 영역)

| Check | Result |
|-------|--------|
| ρ² spread | 0.009 — **PASS** |
| FPR spread | 0.8%p — **PASS** |

## Sequence Gate (sim_sequence ≥ 0.3)

### 배경

경유지/우회 경로 문제: SC 데이터에서 직행이 아니라 중간에 어디 들렀다 오는 경로가
높은 sim_route 점수를 받아 잘못 매칭될 수 있음.

### 지표별 경유지 감별력

| 지표 | 정상 vs 경유지 gap | 감별력 |
|------|-------------------|--------|
| **sim_sequence** | **0.88** | 압도적 |
| sim_spatial | 0.06 | 같은 노선 위 좌표라 감별 불가 |
| sim_time | ≈0 | 없음 |

### 경유지 유형별 필터링

| 경유지 유형 | 감별 방법 |
|------------|----------|
| 같은 노선 다른 구간 (역방향 등) | sim_route 높지만 sim_sequence < 0.3 → gate로 제거 |
| 경유지에서 다른 버스 추가 탑승 | route_jaccard 자체가 낮아져서 composite에서 탈락 |

### Gate 효과

| | 정상 유지율 | 경유지 제거율 |
|--|-----------|-------------|
| sim_sequence ≥ 0.3 | **100%** | **100%** (1,064건 전부 제거) |

- 전체 OD 손실: 0.2% (577건)
- 모델 성능: 미세 개선 (ρ² +0.0005, FPR +0.1%p)

## Heatmap 패턴

- **ρ²**: 좌상단(높은 route, 낮은 seq)이 최적. route가 높을수록 단조 증가.
- **FPR**: 동일 패턴. route 0.85 이상에서 90%대 FPR.
- **n_ods**: route가 높을수록 OD 수 감소 (strict matching), 약 33만.
- **β coefficients**: access, transfers, bus, train은 전 영역에서 안정적.

## 파일 위치

- Coarse CSV: `data/sensitivity/sensitivity_2d_grid_step005.csv`
- Fine CSV: `data/sensitivity/sensitivity_2d_grid_step001_rt08-10_sq00-02.csv`
- Heatmaps: `data/sensitivity/fig_heatmap_*_step001_*.png`
- Checkpoint: `data/sensitivity/checkpoints/`
- Code: `code/similarity/mnl/sensitivity_2d_grid.py`
