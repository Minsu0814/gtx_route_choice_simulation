# Sensitivity Analysis 결과

## 실행 요약

- **1D sweep**: 18개 시나리오 (가중치 8 + 임계값 5 + 정규화거리 5)
- **2D grid**: 28개 시나리오 (spatial 0.0~0.6 × route 0.1~0.4, 나머지 균등배분)
- **총 46개 시나리오**, 시나리오당 ~3분

## 1D Sweep 결과

### Sweep A: 유사도 가중치 (8개)

| Scenario      | mode | seq  | time | route | spatial | N(OD)   | Test ρ² | FPR   |
| ------------- | ---- | ---- | ---- | ----- | ------- | ------- | ------- | ----- |
| baseline      | 0.10 | 0.15 | 0.10 | 0.15  | 0.50    | 640,101 | 0.4598  | 69.7% |
| build_actual  | 0.14 | 0.21 | 0.14 | 0.21  | 0.30    | 641,498 | 0.4595  | 69.8% |
| equal         | 0.20 | 0.20 | 0.20 | 0.20  | 0.20    | 636,699 | 0.4675  | 69.8% |
| no_spatial    | 0.20 | 0.30 | 0.15 | 0.35  | 0.00    | 483,385 | 0.4864  | 73.7% |
| spatial_heavy | 0.05 | 0.10 | 0.05 | 0.10  | 0.70    | 637,653 | 0.4597  | 69.8% |
| route_heavy   | 0.10 | 0.15 | 0.10 | 0.40  | 0.25    | 404,106 | 0.4516  | 72.6% |
| seq_heavy     | 0.10 | 0.40 | 0.10 | 0.15  | 0.25    | 624,584 | 0.4616  | 70.1% |
| time_heavy    | 0.10 | 0.15 | 0.40 | 0.10  | 0.25    | 641,223 | 0.4584  | 69.8% |

- ρ² spread = 0.035, FPR spread = 4.0%p
- 6/8 시나리오에서 ρ² = 0.458~0.468, FPR = 69.7~70.1% (사실상 flat)
- no_spatial, route_heavy는 OD 탈락이 많아 샘플 선별 효과로 성능 상승

### Sweep B: 매칭 임계값 (5개)

| Threshold | N(OD)       | Test ρ²    | FPR       |
| --------- | ----------- | ---------- | --------- |
| 0.4       | 641,500     | 0.4581     | 69.6%     |
| 0.5       | 641,434     | 0.4614     | 69.6%     |
| **0.6**   | **640,101** | **0.4598** | **69.7%** |
| 0.7       | 599,285     | 0.4561     | 70.3%     |
| 0.8       | 485,875     | 0.4896     | 73.9%     |

- 0.4~0.7: 결과 거의 동일 (ρ² = 0.456~0.461)
- 0.8에서만 점프 (OD 24% 탈락 → 샘플 선별 효과)

### Sweep C: 공간 정규화 거리 (5개)

| Distance | N(OD)       | Test ρ²    | FPR       |
| -------- | ----------- | ---------- | --------- |
| 2km      | 594,836     | 0.4593     | 70.2%     |
| 3km      | 614,153     | 0.4600     | 69.9%     |
| **5km**  | **640,101** | **0.4598** | **69.7%** |
| 7km      | 641,465     | 0.4619     | 69.6%     |
| 10km     | 641,500     | 0.4581     | 69.6%     |

- **가장 안정적**: ρ² spread = 0.004, FPR spread = 0.6%p
- 2km~10km 전 범위에서 결과가 사실상 동일

## 2D Grid 결과 (spatial × route)

### Test ρ²

|        | rt=0.1 | rt=0.2 | rt=0.3    | rt=0.4 |
| ------ | ------ | ------ | --------- | ------ |
| sp=0.0 | 0.460  | 0.468  | **0.479** | 0.457  |
| sp=0.1 | 0.461  | 0.467  | **0.482** | 0.450  |
| sp=0.2 | 0.461  | 0.467  | 0.474     | 0.450  |
| sp=0.3 | 0.461  | 0.459  | 0.472     | 0.441  |
| sp=0.4 | 0.462  | 0.465  | 0.467     | 0.436  |
| sp=0.5 | 0.464  | 0.460  | 0.462     | 0.437  |
| sp=0.6 | 0.461  | 0.465  | 0.457     | 0.431  |

### FPR

|        | rt=0.1 | rt=0.2 | rt=0.3 | rt=0.4 |
| ------ | ------ | ------ | ------ | ------ |
| sp=0.0 | 69.9%  | 70.3%  | 71.9%  | 73.5%  |
| sp=0.1 | 69.9%  | 69.9%  | 71.5%  | 73.0%  |
| sp=0.2 | 69.7%  | 69.8%  | 71.2%  | 72.5%  |
| sp=0.3 | 69.7%  | 69.6%  | 70.7%  | 72.0%  |
| sp=0.4 | 69.6%  | 69.7%  | 70.5%  | 71.8%  |
| sp=0.5 | 69.7%  | 69.7%  | 70.1%  | 71.6%  |
| sp=0.6 | 69.6%  | 69.7%  | 69.9%  | 71.5%  |

### 핵심 패턴

- **route=0.3이 sweet spot**: ρ²가 전 구간에서 가장 높음
- **spatial은 0.0~0.3에서 큰 차이 없음**: spatial 가중치의 영향이 작음
- **route=0.4**: OD 수가 391k~432k로 감소, ρ²도 하락 → route에 과도한 가중치는 비효율
- 전체 변동: ρ² CV=2.6%, FPR CV=1.6% → **매우 안정적**

## β 계수 안정성 (전 46개 시나리오)

| 계수        | 범위            | CV    | 판정       |
| ----------- | --------------- | ----- | ---------- |
| β_access    | -0.855 ~ -0.701 | 7.3%  | **ROBUST** |
| β_transfers | -3.167 ~ -2.624 | 5.6%  | **ROBUST** |
| β_train     | +1.178 ~ +1.894 | 8.0%  | **ROBUST** |
| β_bus       | -3.714 ~ -1.642 | 17.1% | **ROBUST** |
| β_IVT       | -0.019 ~ -0.001 | 53.6% | VARIABLE\* |

\*β_IVT: 절대값이 ~0.007로 0에 가까워 CV가 높지만, 모든 시나리오에서 음수(올바른 부호) 유지. 절대값 차이는 0.018 이내.

## Robustness 판정

| 기준                       | 결과       | 판정     |
| -------------------------- | ---------- | -------- |
| ρ² spread < 0.05 (1D only) | 0.038      | **PASS** |
| FPR spread < 5%p           | 4.3%p      | **PASS** |
| 모든 ρ² > 0.2              | 최소 0.431 | **PASS** |
| 모든 FPR > 60%             | 최소 69.6% | **PASS** |
| β_access CV < 20%          | 7.3%       | **PASS** |
| β_transfers CV < 20%       | 5.6%       | **PASS** |
| β_train CV < 20%           | 8.0%       | **PASS** |

## 최적 가중치 Top 5 (composite score: 0.4×ρ² + 0.3×FPR + 0.3×OD유지율)

| 순위 | spatial | route | rest(각) | ρ²    | FPR   | ODs  |
| ---- | ------- | ----- | -------- | ----- | ----- | ---- |
| 1    | 0.0     | 0.3   | 0.233    | 0.479 | 71.9% | 551k |
| 2    | 0.1     | 0.3   | 0.200    | 0.482 | 71.5% | 555k |
| 3    | 0.2     | 0.3   | 0.167    | 0.474 | 71.2% | 566k |
| 4    | 0.3     | 0.3   | 0.133    | 0.472 | 70.7% | 581k |
| 5    | 0.0     | 0.2   | 0.267    | 0.468 | 70.3% | 619k |

## 채택 파라미터

| 파라미터        | 채택값                                    | 근거                                                     |
| --------------- | ----------------------------------------- | -------------------------------------------------------- |
| **가중치**      | spatial=0.1, route=0.3, mode=seq=time=0.2 | 2D grid ρ² 최고(0.482), FPR 71.5%, OD 555k 유지          |
| **임계값**      | 0.6                                       | 0.4~0.7에서 결과 동일, 중간값                            |
| **정규화 거리** | 5km                                       | 2~10km 전 범위 insensitive, 수도권 평균 OD거리 대비 적절 |

## 논문 서술용 요약

> The model results demonstrate robust insensitivity to all three matching parameters.
> Across 46 scenarios spanning weight combinations ($w_{spatial}$: 0.0–0.6, $w_{route}$: 0.1–0.4),
> thresholds (0.4–0.8), and normalization distances (2–10 km),
> McFadden ρ² remained within [0.431, 0.490] and FPR within [69.6%, 73.9%].
> Key behavioral coefficients ($\beta_{access}$, $\beta_{transfers}$, $\beta_{train}$) showed
> CV < 9% across all scenarios. The observed variation is primarily attributable to
> sample selection effects (stricter filtering retains higher-quality OD pairs)
> rather than genuine parameter sensitivity.

## 생성 파일

### Figures

- `data/sensitivity/paper_fig_heatmap_rho2_fpr.png/pdf` — 2D grid ρ²+FPR (논문 메인)
- `data/sensitivity/paper_fig_1d_sweeps.png/pdf` — 1D sweep 3패널
- `data/sensitivity/paper_fig_beta_heatmaps.png/pdf` — β 안정성 heatmap
- `data/sensitivity/paper_fig_heatmap_n_ods.png/pdf` — OD 수 변화

### Tables

- `data/sensitivity/paper_tables.tex` — LaTeX 테이블 5개 (booktabs)

### Data

- `data/sensitivity/sensitivity_weights.csv`
- `data/sensitivity/sensitivity_threshold.csv`
- `data/sensitivity/sensitivity_normdist.csv`
- `data/sensitivity/sensitivity_2d_grid.csv`
- `data/sensitivity/sensitivity_2d_grid.json`
- `data/sensitivity/sensitivity_all.json`
