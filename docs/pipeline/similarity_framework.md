# OTP-스마트카드 유사도 비교 및 Route Choice Simulation

## 1. 프로젝트 개요

OTP(OpenTripPlanner) 경로 탐색 결과와 스마트카드(TCD) 실제 통행 데이터를 비교하여:
1. **유사도 기반 통행 배정** - OTP K-best 경로에 스마트카드 통행을 매칭
2. **Train Set 생성** - Route Choice Model 학습용 데이터 구축
3. **미래 예측** - GTX-B, GTX-C 등 신규 교통수단의 통행 배정 예측

---

## 2. 프로젝트 구조

```
route_choice_simulation/
├── code/
│   ├── module/
│   │   ├── similarity.py          # OTP-SC 유사도 비교 (6-Level Metrics)
│   │   ├── tcd_to_tcn_route.py    # TCD→TCN 변환 (ROUTESTTN 기반)
│   │   └── tcn_to_otp_od.py       # TCN→OTP OD pair 생성
│   ├── tcd_to_parquet.ipynb       # TCD 원본 → Parquet 변환
│   ├── tcd2tcn.ipynb              # TCN 생성 파이프라인 실행
│   ├── tcn_to_otp_od.ipynb        # OTP 입력 OD 데이터 생성
│   ├── similarity_test_10od.ipynb # 10개 OD 유사도 테스트
│   └── gtx_route_choice_simulation.ipynb  # 메인 시뮬레이션
├── data/
│   ├── gtfs/a1/                   # GTFS 데이터 (stops, routes, trips, stop_times)
│   ├── otp/input/                 # OTP 입력 OD pair (CSV)
│   ├── tcn/                       # TCN Parquet (날짜별)
│   ├── shp/                       # Shapefile (시도, 시군구, 읍면동)
│   └── tfcmn/                     # 교통수단코드 요금 정보
└── docs/
    └── similarity_framework.md    # 본 문서
```

---

## 3. 데이터 파이프라인

```
TCD (스마트카드 원본)
    │
    ▼  tcd_to_tcn_route.py
TCN (Trip Chain Network)
    │
    ├──▶ tcn_to_otp_od.py ──▶ OTP OD Input (CSV)
    │                              │
    │                              ▼  OTP GraphQL API
    │                         OTP K-best 경로
    │                              │
    └──────────────────────────────┘
                    │
                    ▼  similarity.py
              유사도 계산 + 매칭
                    │
                    ▼
              통행 배정 결과
```

### 3.1 TCD → TCN 변환 (`tcd_to_tcn_route.py`)

**입력**: TCD (개별 환승 레코드) + ROUT (노선) + ROUTESTTN (노선-정류장)
**출력**: TCN (한 통행 = 한 행)

주요 처리:
- ROUTESTTN 기반 정류장 매칭 (노선ID + 정류장ID → 좌표, 명칭, 순서)
- 숨겨진 지하철 환승 탐지 (SubwayTransferGraph, BFS)
- 왕복 통행 분리
- 교통수단 카테고리 분류 (7개)

**TCN 주요 컬럼**:
| 컬럼 | 설명 |
|------|------|
| `정류장명칭시퀀스` | 경유 정류장 명칭 리스트 (숨겨진 환승역 포함) |
| `정류장시퀀스` | 경유 정류장 ID 리스트 |
| `노선명` | 이용 노선명 리스트 |
| `환승횟수` | 명시적 + 숨겨진 환승 합계 |
| `숨겨진환승역` | BFS로 탐지된 환승역 명칭 리스트 |
| `transport_category` | 7개 카테고리 (bus_only, train_only, ...) |
| `od_pair` | 승차정류장ID_하차정류장ID |
| `총탑승시간` | 전체 탑승시간 (초) |

### 3.2 숨겨진 환승 탐지 (SubwayTransferGraph)

스마트카드에서 하나의 TCD 레코드가 여러 지하철 노선을 거치는 경우를 탐지.

```
예: 고색 → 역삼 (수인분당선 → 2호선)
    TCD에는 1건으로 기록되지만 실제로는 선릉에서 환승
    BFS로 숨겨진 환승역 '선릉' 탐지 → 환승횟수 +1
```

- 노선명(short) 기반 그래프 구축 (같은 역에서 다른 노선 탑승 가능 = 간선)
- 수인분당선은 세그먼트(106/112) 구분 없이 하나의 노선으로 처리

### 3.3 TCN → OTP OD (`tcn_to_otp_od.py`)

TCN을 OD pair로 그룹화하여 OTP 경로탐색 입력 생성.
여러 날짜 TCN을 통합, 통행 건수 집계.

### 3.4 OTP 경로 탐색

```graphql
plan(
    from: {lat, lon}
    to: {lat, lon}
    numItineraries: 10
    searchWindow: 7200
    transportModes: [TRANSIT]
)
```

OTP 응답에서 사용하는 필드:
- `from.name` / `to.name`: 정류장 명칭 (SC 정류장명칭시퀀스와 직접 비교)
- `route.shortName`: 노선 명칭 (SC 노선명과 직접 비교)
- `duration`: 소요시간

---

## 4. 유사도 비교 (similarity.py)

### 4.1 명칭 기반 비교 체계

OTP와 스마트카드는 서로 다른 ID 체계를 사용하므로, **명칭(name)** 기반으로 비교:

| 항목 | OTP | SC (TCN) |
|------|-----|----------|
| 정류장 | `from.name` (예: 운서역) | `정류장명칭시퀀스` (예: 운서) |
| 노선 | `route.shortName` (예: 공항철도) | `노선명` (예: 공항철도) |

노선명 정규화: OTP `서울2호선` → `2호선` (SC와 동일 체계)

### 4.2 6-Level Similarity Metrics

#### Level 1: 모드 조합 (가중치 0.15)

| 지표 | 설명 | 서브 가중치 |
|------|------|------------|
| `mode_exact` | 모드 집합 완전 일치 | 0.3 |
| `mode_jaccard` | Jaccard 유사도 | 0.5 |
| `mode_contains` | SC 모드가 OTP에 포함 | 0.2 |

#### Level 2: 환승 (가중치 0.15)

| 지표 | 설명 | 서브 가중치 |
|------|------|------------|
| `transfer_exact` | 환승 횟수 정확 일치 | 0.3 |
| `transfer_diff` | 차이 기반 유사도 | 0.5 |
| `transfer_cat` | 범주 일치 (0/1/2+) | 0.2 |

#### Level 3: 정류장 시퀀스 (가중치 0.30) - 핵심 지표

| 지표 | 설명 | 서브 가중치 |
|------|------|------------|
| `seq_jaccard` | 정류장 집합 Jaccard | 0.2 |
| `seq_lcs` | LCS 비율 (순서 보존) | 0.5 |
| `seq_levenshtein` | 편집거리 유사도 | 0.2 |
| `seq_prefix` | 시작부 일치율 | 0.05 |
| `seq_suffix` | 도착부 일치율 | 0.05 |

#### Level 4: 소요시간 (가중치 0.15)

| 지표 | 설명 | 서브 가중치 |
|------|------|------------|
| `time_ratio` | min/max 비율 | 0.4 |
| `time_band` | +-5분 이내 일치 | 0.3 |
| `time_score` | 정규화 점수 (30분 기준) | 0.3 |

#### Level 5: 노선 (가중치 0.15)

| 지표 | 설명 | 서브 가중치 |
|------|------|------------|
| `route_exact` | 노선 집합 완전 일치 | 0.2 |
| `route_jaccard` | Jaccard 유사도 | 0.5 |
| `route_main` | 주 노선 일치 | 0.3 |

주 노선: 가장 긴 leg(차내시간 기준)의 노선

#### Level 6: 공간 (가중치 0.10)

| 지표 | 설명 | 서브 가중치 |
|------|------|------------|
| `od_distance_sim` | OD 직선거리 유사도 | 0.5 |
| `transfer_loc_match` | 환승 정류장 위치 일치율 | 0.5 |

### 4.3 복합 점수 및 등급

```
composite = 0.15*mode + 0.15*transfer + 0.30*sequence + 0.15*time + 0.15*route + 0.10*spatial
```

| 등급 | 점수 범위 | 판정 |
|------|----------|------|
| 우수 | 0.8 ~ 1.0 | 확실한 매칭 |
| 양호 | 0.6 ~ 0.8 | 매칭 성공 |
| 보통 | 0.4 ~ 0.6 | 부분 매칭 |
| 불량 | 0.0 ~ 0.4 | 매칭 실패 |

매칭 임계값: **0.6** (양호 이상)

### 4.4 경로 중복 제거

OTP가 동일 구조(모드+노선+정류장)에 출발시간만 다른 경로를 반환하는 경우,
`deduplicate_itineraries()`로 합치고 소요시간은 평균.

### 4.5 교통수단 카테고리

| 카테고리 | 포함 모드 |
|----------|----------|
| `bus_only` | 버스만 |
| `train_only` | 지하철/철도만 |
| `gtx_only` | GTX만 |
| `bus+train` | 버스 + 지하철 |
| `bus+gtx` | 버스 + GTX |
| `train+gtx` | 지하철 + GTX |
| `bus+train+gtx` | 버스 + 지하철 + GTX |

---

## 5. 10개 OD 테스트 결과

`similarity_test_10od.ipynb`에서 상위 10개 OD (통행 건수 기준) 테스트.

테스트 대상: 공항철도 OD (4211↔4213), 2호선 OD (220↔222, 222↔228 등)

주요 결과:
- 매칭 성공률: ~99.5% (threshold=0.6)
- 평균 복합점수: ~0.97
- sequence_score: ~1.0 (명칭 기반 비교로 정확 매칭)
- route_score: ~1.0 (노선명 정규화 적용)

---

## 6. 향후 계획

1. **Route Choice Model 학습**
   - Feature 추출: 경로 속성 (시간, 환승, 거리) + 시공간 속성 (시간대, 지역)
   - MNL (Multinomial Logit) / Mixed Logit / Neural Network
2. **GTX-B, GTX-C 통행 배정 예측**
   - 새로운 GTFS 적용 → OTP 경로탐색 → 학습된 모델로 선택 확률 예측
3. **전체 OD 대상 확장**
   - 10개 OD → 336,498개 OD 전체 실행
