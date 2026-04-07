# 경로선택모형 설계 (Model Specification)

> **작성일**: 2026.04.03 (최종 수정: 2026.04.05)

---

## 1. 설계 원칙

### 1.1 SC 데이터의 한계와 모형 범위

스마트카드(SC)는 **승차정류장 → 하차정류장**만 기록한다. 실제 출발지(집)와 최종 도착지(회사)를 모른다.

```
SC가 아는 것:    [승차정류장] ──── transit ──── [하차정류장]
SC가 모르는 것:  [집] ── walk ── [승차정류장]    [하차정류장] ── walk ── [회사]
                  ↑ first mile                              ↑ last mile
```

이 한계를 인정하고 모형의 범위를 정직하게 설정:

| 구간                     | 담당       | 근거                                          |
| ------------------------ | ---------- | --------------------------------------------- |
| first/last mile (도보)   | **Raptor** | SC에 정보 없음 → 학습 불가 → Raptor GC가 처리 |
| transit (차내+환승+대기) | **모형**   | SC에 정보 있음 → 학습 가능 → β 추정           |

### 1.2 도보 계수 양수 문제

RP 데이터에서 β_walk이 양수로 추정됨. 원인: 자기선택 + 구조적 역상관.

```
버스 정류장(가까움): access 2분 → transit 25분, 환승 1회
지하철역(멀음):      access 8분 → transit 12분, 환승 0회
                              ↑
              지하철 선택자가 다수 → 긴 도보 = 선택됨 → β_walk > 0
```

SP와 달리 RP에서는 도보를 다른 속성과 독립적으로 변이시킬 수 없어 순수 기피 효과 분리 불가. centroid 기반 access_time의 측정 오차도 attenuation bias 유발.

**결정**: access_time, egress_time, walk_time, walk_distance를 **모형에서 제거**. 단, transfer_walk_time은 유지 (OTP 직접 계산값, measurement error 없음).

### 1.3 Raptor-모형 역할 분담

```
[Raptor]                        [모형]                      [배분]
좌표 → 대안 K개 생성        →   피처 → 확률 계산        →   flow × P(j)
- 도보 포함 경로 탐색            - transit 속성만 평가         - 경로별 통행량
- GC 기반 도보 필터링            - P(route_j) 출력            - GC 산출 (logsum)
- choice set 생성                - 도보는 모름
```

적용 시 Raptor가 도보 긴 경로에 높은 GC를 부여하여, 도보 짧은 경로를 우선 반환. 도보 선호는 Raptor 수준에서 암묵적 반영.

---

## 2. 모형 구조

### 2.1 모형: Mixed Logit

MNL이 아닌 **Mixed Logit을 기본 모형으로 사용**. MNL은 Mixed Logit의 특수한 경우(분산=0). 해석은 β의 평균과 표준편차로 동일하게 가능.

### 2.2 수단별 차내시간 분리 (mode-specific IVT)

`in_vehicle_time` 통합 시 "버스 10분 + 지하철 20분"과 "택시 30분"이 동일한 값이 되어 체감 비효용 차이를 반영 못 함. 수단별로 분리.

```
ivt_bus, ivt_train, ivt_gtx, ivt_taxi, ivt_uam, ivt_bike, ivt_kick
```

수단별 VOT 산출 가능: `VOT_train = E[β_ivt_train] / β_fare`

### 2.3 ASC 설계: transport_category (수단 조합 더미)

#### 검토한 방식들

| 방식                    | 장점                       | 단점                                                    | 판정               |
| ----------------------- | -------------------------- | ------------------------------------------------------- | ------------------ |
| transport_category 더미 | 수단 조합의 고유 효과 캡처 | 수단 많으면 조합 폭발                                   | **Phase 1-2 채택** |
| has\_수단 (additive)    | 확장성 좋음                | bus+train ≠ ASC_bus+ASC_train (가법성 비현실적)         | Phase 3+ 전환      |
| 주수단 ASC              | 파라미터 적음              | 멀티모달에서 주수단 판별 불가 (bus 16분+train 15분 → ?) | 기각               |

#### 채택: Phase별 전환

**Phase 1 (bus, train, gtx)**: transport_category 더미

| 카테고리      | 예시                     | 비고                         |
| ------------- | ------------------------ | ---------------------------- |
| bus_only      | 버스 직통/환승           | base (ASC=0)                 |
| train_only    | 지하철 직통/환승         |                              |
| bus+train     | 버스→지하철, 지하철→버스 |                              |
| train+gtx     | 지하철→GTX               |                              |
| gtx_only      | GTX 직통                 | 관측 적으면 train+gtx에 합침 |
| bus+train+gtx | 버스→지하철→GTX          | 관측 적으면 train+gtx에 합침 |
| bus+gtx       | 버스→GTX (지하철 없이)   | 거의 없음 → 기타 처리        |

실질 ASC 5~6개. MNL/Mixed Logit에서 문제없는 규모.

**Phase 2 (+taxi)**: 이론 15개 → 관측되는 조합만 유지 + 희소 조합 그룹핑 → 실질 8~10개

**Phase 3+ (+uam, bike, kick)**: 조합 폭발 → `has_수단 + 주요 교호항`으로 전환 또는 embedding

### 2.4 효용함수 (Phase 1)

```
V_j = β_ivt_bus   · ivt_bus_j            ← random (lognormal)
    + β_ivt_train · ivt_train_j           ← random (lognormal)
    + β_ivt_gtx   · ivt_gtx_j            ← random (lognormal)
    + β_wait      · wait_time_j           ← random (lognormal)
    + β_xwalk     · transfer_walk_time_j  ← random (lognormal)
    + β_xfer      · num_transfers_j       ← random (normal)
    + β_fare      · fare_j               ← fixed
    + ASC_category(j)                     ← fixed
```

### 2.5 Random vs Fixed 파라미터

| 파라미터 | 분포                   | 근거                                          |
| -------- | ---------------------- | --------------------------------------------- |
| β*ivt*\* | **random (lognormal)** | 시간가치 개인차. `-exp(μ + σ·z)`로 음수 강제  |
| β_wait   | **random (lognormal)** | 대기 민감도 개인차                            |
| β_xwalk  | **random (lognormal)** | 환승도보 민감도 개인차                        |
| β_xfer   | **random (normal)**    | 환승 민감도 개인차                            |
| β_fare   | **fixed**              | random 시 양수 개인 발생 → VOT 분포 발산 위험 |
| ASC\_\*  | **fixed**              | 수단 조합 고유효용                            |

### 2.6 파라미터 수

random 6개 × 2(μ,σ) + fixed (β_fare 1개 + ASC 5개) = **18개**
추정 방법: MSL (Maximum Simulated Likelihood), Halton draws 500+

### 2.7 기대 부호

| 파라미터 | 기대 부호             | 해석                     |
| -------- | --------------------- | ------------------------ |
| β*ivt*\* | 음수 (lognormal 강제) | 차내시간 길면 기피       |
| β_wait   | 음수 (lognormal 강제) | 대기시간 길면 기피       |
| β_xwalk  | 음수 (lognormal 강제) | 환승도보 길면 기피       |
| β_xfer   | 음수 (기대)           | 환승 많으면 기피         |
| β_fare   | 음수 (고정)           | 비싸면 기피              |
| ASC\_\*  | 부호 자유             | 수단 조합 고유 선호/기피 |

β_ivt 크기 관계 (예상): |β_ivt_bus| > |β_ivt_train| > |β_ivt_gtx| (쾌적할수록 시간 비효용 작음)

### 2.8 해석 예시

```
β_ivt_train ~ -exp(N(μ=-5.8, σ=0.5))

→ 평균: E[β] = -exp(μ + σ²/2) ≈ -0.0037
→ 해석: "지하철 1분 증가 시 평균 효용 -0.0037"
→ σ > 0이면 시간가치에 개인차 존재 (이질성 확인)
→ VOT_train = E[β_ivt_train] / β_fare
```

---

## 3. 학습 데이터 구축

### 3.1 초이스셋: 같은 출발정류장 대안만 유지

도보를 모형에서 제거했으므로, 초이스셋에도 도보 차이가 개입하면 안 됨.

```
SC: "강남역에서 2호선 탑승"

OTP 결과 필터링:
  ✓ Alt 1: 강남역 → 2호선 직통                    (같은 출발정류장)
  ✓ Alt 2: 강남역 → 2호선 → 환승 → 3호선          (같은 출발정류장)
  ✓ Alt 3: 강남역 → 2호선 → 환승 → 신분당선        (같은 출발정류장)
  ✗ Alt 4: 걸어서 신논현역 → 9호선                  (다른 정류장 → 제거)
  ✗ Alt 5: 걸어서 버스정류장 → 버스 140              (다른 정류장 → 제거)
```

**효과**:

- 대안 간 access walk = 전부 0 → 도보가 비교에 개입하지 않음
- 순수 transit 속성(차내시간, 환승, 요금)의 효과만 학습
- β 오염 방지: "도보 때문에 더 좋은 대안을 안 골랐다"는 왜곡 신호 제거

**주의**: 필터링 후 choice set size < 2인 OD는 제외. 사전에 분포 확인 필요.

### 3.2 정답(chosen) 매핑

SC 승차정류장 + 노선 시퀀스로 exact match. 같은 출발정류장 대안만 남겼으므로 모호함 없음.

```
SC: 강남역 → 2호선 탑승
→ Alt 1 (강남역 → 2호선 직통)과 정확히 매칭 → chosen = 1
→ Alt 2, Alt 3 → chosen = 0
```

매칭 실패 시 해당 SC 통행 제외.

### 3.3 피처 추출

| 변수               | 설명                 | 출처                        | 분포        |
| ------------------ | -------------------- | --------------------------- | ----------- |
| ivt_bus            | 버스 차내시간 (초)   | OTP leg                     | random      |
| ivt_train          | 지하철 차내시간 (초) | OTP leg                     | random      |
| ivt_gtx            | GTX 차내시간 (초)    | OTP leg                     | random      |
| wait_time          | 대기시간 합계 (초)   | OTP itinerary               | random      |
| transfer_walk_time | 환승 도보시간 (초)   | OTP leg (transit 사이 walk) | random      |
| num_transfers      | 환승 횟수            | OTP itinerary               | random      |
| fare               | 요금 (원)            | 모드+거리 기반 계산         | fixed       |
| transport_category | 수단 조합 카테고리   | mode set 판별               | fixed (ASC) |

**제외 변수**: access_time, egress_time, walk_time, walk_distance, total_duration

### 3.4 파이프라인

```
[1] 기존 OTP 결과 로드 (526K stop-to-stop OD × 2-5개 대안)
    │   ← HuggingFace에 업로드된 기존 데이터 재사용
    │
[2] 같은 출발정류장 필터링
    │   OTP 대안의 첫 transit leg 출발정류장 == SC 승차정류장인 것만 유지
    │   choice set size < 2인 OD 제외
    │
[3] 피처 추출 (route_features.py 수정 버전)
    │
    │   for each itinerary:
    │       legs = itinerary['legs']
    │       classified = [_classify_leg(leg) for leg in legs]
    │
    │       # 수단별 IVT
    │       ivt_bus   = sum(c['duration'] for c in classified if c['internal_mode'] == 'bus')
    │       ivt_train = sum(c['duration'] for c in classified if c['internal_mode'] == 'train')
    │       ivt_gtx   = sum(c['duration'] for c in classified if c['internal_mode'] == 'gtx')
    │
    │       # 공통 변수
    │       wait_time = max(0, total_duration - ivt_total - walk_total)
    │       transfer_walk_time = sum(walk between transit legs)
    │       num_transfers = len(transit_legs) - 1
    │       fare = calc_fare_from_itinerary(itinerary)
    │
    │       # 수단 조합 카테고리
    │       mode_set = {c['internal_mode'] for c in classified if c['is_transit']}
    │       transport_category = _mode_set_to_category(mode_set)
    │
[4] SC 매칭 → chosen 라벨 생성
    │   출발정류장 + 노선 시퀀스 exact match
    │   매칭 실패 → 해당 SC 통행 제외
    │
[5] 학습 데이터 조립 → Parquet 저장
```

### 3.5 학습 데이터 형태

```
obs_id | alt_id | chosen | ivt_bus | ivt_train | ivt_gtx | wait  | xwalk | xfer | fare | category
-------|--------|--------|---------|-----------|---------|-------|-------|------|------|----------
  1    |  r1    |   1    |   0     |   1500    |   0     |  180  |   0   |  0   | 1550 | train_only
  1    |  r2    |   0    |   0     |   1800    |   0     |  120  |  180  |  1   | 1550 | train_only
  1    |  r3    |   0    |   600   |   900     |   0     |  240  |  120  |  1   | 1500 | bus+train
  2    |  r1    |   0    |   480   |   0       |   0     |  300  |   0   |  0   | 1500 | bus_only
  2    |  r2    |   1    |   0     |   1200    |   0     |  200  |   0   |  0   | 1550 | train_only
  ...  |  ...   |  ...   |   ...   |   ...     |  ...    |  ...  |  ...  |  ... | ...  | ...
```

예상 규모: ~200만 SC 통행 × 2-4개 대안 = 400만-800만 행

---

## 4. 구현 가이드 (route_features.py 수정)

### 4.1 수단별 IVT 분리

현재 `_classify_leg()`에서 `internal_mode`를 이미 반환하므로, `in_vehicle_time` 합산 부분을 수단별로 분리.

```python
# === 현재 코드 (extract_itinerary_features 내부) ===
in_vehicle_time = sum(c['duration'] for c in classified if c['is_transit'])

# === 수정 ===
ivt_bus   = sum(c['duration'] for c in classified if c['internal_mode'] == 'bus')
ivt_train = sum(c['duration'] for c in classified if c['internal_mode'] == 'train')
ivt_gtx   = sum(c['duration'] for c in classified if c['internal_mode'] == 'gtx')
# Phase 2+ 수단 추가 시 한 줄씩 추가
```

### 4.2 transport_category 판별

기존 `_mode_set_to_category()` 함수 그대로 사용. 현재 7개 카테고리 반환:
`bus_only, train_only, gtx_only, bus+train, bus+gtx, train+gtx, bus+train+gtx`

희소 카테고리 그룹핑은 학습 데이터 조립 단계에서 처리:

```python
# 희소 카테고리 그룹핑 (학습 데이터 조립 시)
CATEGORY_MAP = {
    'bus_only': 'bus_only',
    'train_only': 'train_only',
    'bus+train': 'bus+train',
    'train+gtx': 'train+gtx',
    'gtx_only': 'train+gtx',         # 관측 적으면 합침
    'bus+train+gtx': 'train+gtx',    # 관측 적으면 합침
    'bus+gtx': 'train+gtx',          # 관측 적으면 합침
}
# ※ 실제 그룹핑은 관측 수 확인 후 결정
```

### 4.3 같은 출발정류장 필터링

```python
def filter_same_origin_stop(otp_alternatives, sc_boarding_stop):
    """SC 승차정류장과 동일한 출발정류장의 대안만 유지"""
    filtered = []
    for alt in otp_alternatives:
        legs = alt.get('legs', [])
        # 첫 번째 transit leg의 출발정류장
        first_transit = next((l for l in legs if l.get('mode') != 'WALK'), None)
        if first_transit is None:
            continue
        from_stop = first_transit.get('from', {}).get('name', '')
        # 정류장명 정규화 후 비교
        if normalize_stop_name(from_stop) == normalize_stop_name(sc_boarding_stop):
            filtered.append(alt)
    return filtered
```

### 4.4 extract_itinerary_features 수정 반환값

```python
def extract_itinerary_features(itinerary):
    """수정: 수단별 IVT + transport_category 반환"""
    # ... (기존 코드에서 legs 분류까지 동일)

    return {
        # 수단별 IVT (신규)
        'ivt_bus': ivt_bus,
        'ivt_train': ivt_train,
        'ivt_gtx': ivt_gtx,

        # 공통 변수 (기존 유지)
        'wait_time': wait_time,
        'transfer_walk_time': transfer_walk_time,
        'num_transfers': num_transfers,
        'fare': fare,

        # 수단 조합 카테고리 (기존 유지)
        'transport_category': transport_category,

        # === 아래는 모형에 사용하지 않으나 참고용 유지 ===
        'total_duration': total_duration,
        'in_vehicle_time': ivt_bus + ivt_train + ivt_gtx,
        'walk_time': walk_time,
        'access_time': access_time,
        'egress_time': egress_time,
        'total_distance': round(total_distance, 1),
        'has_bus': int('bus' in mode_set),
        'has_train': int('train' in mode_set),
        'has_gtx': int('gtx' in mode_set),
        'main_route': main_route,
    }
```

### 4.5 새 수단 추가 절차

1. `_classify_leg()`에 수단 판별 조건 추가 → `internal_mode` 반환값 추가
2. `extract_itinerary_features()`에 `ivt_수단` 항목 추가
3. `_mode_set_to_category()`에 새 조합 카테고리 추가
4. 학습 데이터의 `CATEGORY_MAP`에서 희소 조합 그룹핑 결정

---

## 5. 적용

### 5.1 적용 파이프라인

```
[1] OD 좌표 (H3 centroid 또는 raw lat/lon)
    │
[2] Raptor → K개 대안 생성 (도보 포함 전체 결과)
    │
[3] 피처 추출 (학습과 동일한 extract_itinerary_features)
    │
[4] 모형 적용
    │   V_j = β · x_j
    │   P(route_j) = exp(V_j) / Σ_k exp(V_k)
    │
[5] GC 산출 (선택적)
    │   GC_od = -1/β_fare · ln(Σ_j exp(V_j))
    │
[6] 통행 배분
        flow_j = demand_od × P(route_j)
```

### 5.2 모형의 공간 무관성

모형은 경로 피처만 평가. OD 좌표, H3 셀, 공간 단위를 모름. 따라서:

- **대규모 (넓은 지역)**: H3로 OD 집계 → centroid에서 Raptor → 효율적
- **소규모 (정밀 분석)**: raw 위경도에서 Raptor → 정확
- **GTX 평가**: GTX GTFS 포함 Raptor → 신설역 경유 대안 자동 생성

모형 β는 동일. 바뀌는 건 Raptor 입력(OD 좌표)뿐.

### 5.3 학습-적용 차이점

|          | 학습                   | 적용                  |
| -------- | ---------------------- | --------------------- |
| OD       | SC 승/하차 정류장      | 임의 좌표             |
| 초이스셋 | 같은 출발정류장 대안만 | Raptor 전체 결과      |
| 도보     | 제거 (β 오염 방지)     | Raptor GC로 암묵 반영 |
| chosen   | SC 매칭 (1/0)          | 없음 (확률 예측)      |

### 5.4 도보 보정 (선택적)

적용 결과에서 도보 영향이 부족하다고 판단되면, **문헌값 β_walk**을 적용 시에만 추가. 학습 구조는 건드리지 않음.

```python
# 적용 시 도보 보정 (선택적, 결과 검증 후 필요 시)
WALK_MULTIPLIER = 2.0  # 문헌: 도보 1분 ≈ 차내시간 2분 (통상 1.5~2.5배)
beta_walk_lit = WALK_MULTIPLIER * mean(beta_ivt_train)

for alt in alternatives:
    alt.V += beta_walk_lit * alt.walk_time  # OTP가 제공하는 도보시간
```

---

## 6. 단계적 확장

### Phase 1: 대중교통 (현재)

```
V_j = β_ivt_bus · ivt_bus + β_ivt_train · ivt_train + β_ivt_gtx · ivt_gtx
    + β_wait · wait_time + β_xwalk · transfer_walk_time
    + β_xfer · num_transfers + β_fare · fare
    + ASC_category(j)
```

- 데이터: SC + OTP (526K OD, 현재 보유)
- ASC: transport_category 더미 5~6개
- 추정 파라미터: ~18개

### Phase 2: + 택시

택시 API 경로 데이터 + 통신사 RP 확보 후 `ivt_taxi` 추가.
ASC: transport_category에 택시 포함 조합 추가 (관측되는 것만, 희소 조합 그룹핑)

### Phase 3: + 마이크로모빌리티, UAM

조합 폭발 → ASC를 `has_수단 + 주요 교호항`으로 전환.

```
ASC = ASC_bus · has_bus + ASC_train · has_train + ASC_gtx · has_gtx
    + ASC_taxi · has_taxi + ASC_uam · has_uam + ...
    + γ_bus_train · has_bus · has_train + ...   ← 주요 교호항만
```

또는 DTUMOS SP 데이터 활용하여 신규 수단 선호도 반영.

### 장기: Attention-based Leg Embedding

수단 조합을 leg sequence로 인코딩하여 interaction 자동 학습. 수단 추가 시 embedding만 추가.

```
[택시, 5분] → [GTX, 25분] → [버스, 10분]
    ↓              ↓              ↓
  embed_taxi    embed_gtx     embed_bus
    ↓              ↓              ↓
         Attention / Pooling
                 ↓
           route vector → V_j
```

---

## 7. 변수 요약

### 모형 변수

| 변수               | 설명            | 단위 | 분포               |
| ------------------ | --------------- | ---- | ------------------ |
| ivt_bus            | 버스 차내시간   | 초   | random (lognormal) |
| ivt_train          | 지하철 차내시간 | 초   | random (lognormal) |
| ivt_gtx            | GTX 차내시간    | 초   | random (lognormal) |
| wait_time          | 대기시간 합계   | 초   | random (lognormal) |
| transfer_walk_time | 환승 도보시간   | 초   | random (lognormal) |
| num_transfers      | 환승 횟수       | 회   | random (normal)    |
| fare               | 요금            | 원   | fixed              |
| transport_category | 수단 조합       | 범주 | fixed (ASC)        |

### 제외 변수 (참고용 저장만)

| 변수           | 제외 사유                                       |
| -------------- | ----------------------------------------------- |
| access_time    | RP 자기선택 → β 양수                            |
| egress_time    | RP 자기선택 → β 양수                            |
| walk_time      | access + egress + transfer_walk 합계, 위와 동일 |
| walk_distance  | walk_time과 동일 문제                           |
| total_duration | ivt + wait + walk 합계, 도보 효과 재유입        |

---

## 8. 한계 및 주의사항

### 8.1 도보 공백

모형이 도보를 평가하지 못하므로, 적용 시 transit 속성이 좋지만 도보가 긴 경로를 과대 평가할 수 있음. GTX 신설역처럼 도보 접근이 먼 경우 수요 과대추정 위험.

**완화**: Raptor walk budget (기본 15분)이 1차 필터. 필요 시 문헌값 β_walk 적용 (Section 5.4).

### 8.2 초이스셋 크기

같은 출발정류장 필터링 후 choice set size < 2인 OD 발생 가능. 소규모 버스 정류장에서 대안이 1개뿐인 경우. **학습 전 분포 확인 필수**.

### 8.3 SC 매칭 실패

SC 통행이 OTP 대안과 매칭되지 않는 경우 (비정상 경로, OTP searchWindow 밖 출발 등) 해당 관측 제외. 매칭률 모니터링 필요.

### 8.4 모형 범위

본 모형은 **"출발 정류장이 주어진 상태에서의 transit 경로선택"**을 예측. 정류장 접근(first/last mile)은 Raptor가 처리. 수단선택(대중교통 vs 자가용)은 미포함.
