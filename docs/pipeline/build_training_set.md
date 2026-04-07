# Route Choice Model Training Dataset 구축

## 1. 개요

OTP K-best 경로와 스마트카드 실제 통행을 매칭한 결과로, Route Choice Model(MNL/Mixed Logit/NN) 학습용 데이터셋을 구축한다.

**산출물:**

- `code/module/route_features.py` — 피처 추출 모듈
- `code/build_training_set.ipynb` — 데이터셋 구축 노트북
- `data/training_set/route_choice_training.parquet` — 최종 데이터셋

**출력 포맷:** 각 행 = 1개 선택상황(trip) × 1개 대안경로(alt)

---

## 2. 파이프라인

```
TCN (스마트카드)  ──┐
                    ├──→ 유사도 매칭 → chosen 결정 → Choice Set 생성
OTP (K-best 경로) ──┘
```

### 처리 흐름

1. **OTP 결과 로드** — ijson 스트리밍 (23GB JSON), CSV id→od_pair 매핑
2. **거리 복원** — transit leg distance=0 → legGeometry polyline에서 복원
3. **중복 제거** — 출발시간만 다른 동일 경로를 합침, ≥2개 대안 필터
4. **TCN 로드 + OD 그룹화** — 전체 날짜 TCN을 OTP OD 기준으로 필터링
5. **OD별 루프:**
   - OTP 대안 피처 사전 추출 (OD별 1회, 효율화)
   - SC trip별 유사도 매칭 → chosen 경로 결정
   - 매칭 실패 (score < 0.6) → skip
   - 모든 대안에 대해 row 생성 (chosen=1/0)
6. **검증** — chosen 정합성, choice set 크기 확인
7. **저장** — parquet + csv

---

## 3. 피처 정의

### 3.1 대안경로 피처 (21개)

`extract_itinerary_features(itinerary)` 함수로 추출.

| #   | 피처                 | 설명                                | 단위   |
| --- | -------------------- | ----------------------------------- | ------ |
| 1   | `total_duration`     | 총 소요시간                         | 초     |
| 2   | `in_vehicle_time`    | 차내시간 (transit leg duration 합)  | 초     |
| 3   | `walk_time`          | 도보시간 (WALK leg duration 합)     | 초     |
| 4   | `wait_time`          | 대기시간 (= total - vehicle - walk) | 초     |
| 5   | `access_time`        | 접근시간 (첫 transit 전 WALK)       | 초     |
| 6   | `egress_time`        | 이탈시간 (마지막 transit 후 WALK)   | 초     |
| 7   | `transfer_walk_time` | 환승도보시간 (중간 WALK legs)       | 초     |
| 8   | `walk_distance`      | 도보거리                            | m      |
| 9   | `total_distance`     | 총 이동거리 (모든 leg)              | m      |
| 10  | `bus_distance`       | 버스 이동거리                       | m      |
| 11  | `subway_distance`    | 지하철 이동거리                     | m      |
| 12  | `gtx_distance`       | GTX 이동거리                        | m      |
| 13  | `num_transfers`      | 환승 횟수 (transit leg 수 - 1)      | 회     |
| 14  | `num_legs`           | transit leg 수                      | 개     |
| 15  | `fare`               | 요금 (모드+거리 기반 계산)          | 원     |
| 16  | `generalized_cost`   | OTP 일반화비용                      | -      |
| 17  | `transport_category` | 수단조합 (7종)                      | 범주형 |
| 18  | `has_bus`            | 버스 포함 여부                      | 0/1    |
| 19  | `has_train`          | 지하철 포함 여부                    | 0/1    |
| 20  | `has_gtx`            | GTX 포함 여부                       | 0/1    |
| 21  | `main_route`         | 주 노선명 (최장 leg의 route)        | 문자열 |

### 3.2 컨텍스트 피처 (5개)

`extract_trip_context(sc_row)` 함수로 추출. SC 통행/OD별 공유.

| #   | 피처                    | 설명             | 소스                   |
| --- | ----------------------- | ---------------- | ---------------------- |
| 1   | `od_distance`           | OD 직선거리 (m)  | 좌표 haversine         |
| 2   | `departure_hour`        | 출발시각 (시)    | 승차일시.hour          |
| 3   | `departure_dow`         | 요일 (0=월~6=일) | 승차일시.dayofweek     |
| 4   | `is_peak`               | 첨두시간 여부    | 7-9시, 17-19시         |
| 5   | `sc_transport_category` | 실제 이용수단    | TCN.transport_category |

### 3.3 라벨 + 검증용

| 피처              | 설명                                   |
| ----------------- | -------------------------------------- |
| `trip_id`         | 선택상황 식별자 (od_pair_date_counter) |
| `od_pair`         | OD pair 키                             |
| `alt_idx`         | 대안 인덱스                            |
| `choice_set_size` | 대안 수                                |
| `chosen`          | 라벨 (0 또는 1)                        |
| `sim_composite`   | 복합 유사도 점수                       |
| `sim_mode`        | 모드 유사도                            |
| `sim_sequence`    | 시퀀스 유사도                          |
| `sim_time`        | 시간 유사도                            |
| `sim_route`       | 노선 유사도                            |
| `sim_spatial`     | 공간 유사도                            |

---

## 4. 요금 계산 로직

`calc_fare_from_itinerary(itinerary)` — 모드별 거리를 합산하여 요금 규칙 적용.

### GTX 판별

- OTP leg의 `route.shortName`에 `GTX` 또는 `290` 키워드 포함 여부

### 요금 규칙 (성인 기본)

| 조건                | 기본요금      | 추가요금                                                 |
| ------------------- | ------------- | -------------------------------------------------------- |
| **버스 단독**       | 1,500원       | 정액                                                     |
| **전철 단독**       | 1,550원       | 10km 무료 → 50km까지 5km당 100원 → 50km 초과 8km당 100원 |
| **GTX 단독**        | 3,200원       | 10km 무료 → 5km당 250원                                  |
| **통합 (2종 이상)** | max(기본요금) | 10km 초과 5km당: GTX 포함 시 250원, 미포함 시 100원      |

### 검증 예시

```
GTX 25km   → 3,200 + (25-10)/5 × 250 = 3,950원
Subway 15km → 1,550 + (15-10)/5 × 100 = 1,650원
Bus+GTX 35km → max(1500,3200) + (35-10)/5 × 250 = 4,450원
```

---

## 5. 시간 분해 로직

총 소요시간을 3가지 구성요소로 분해:

```
total_duration = in_vehicle_time + walk_time + wait_time
```

도보시간 세분화:

```
walk_time = access_time + transfer_walk_time + egress_time
```

- **in_vehicle_time**: transit leg(BUS/SUBWAY/RAIL) duration 합
- **walk_time**: WALK leg duration 합
- **wait_time**: total_duration - in_vehicle_time - walk_time (잔차로 계산)
- **access_time**: 첫 번째 transit leg 이전 WALK
- **egress_time**: 마지막 transit leg 이후 WALK
- **transfer_walk_time**: 중간 WALK legs

---

## 6. 데이터 정규화

### 6.1 노선명 정규화 (`_normalize_route_name`)

OTP와 SC의 노선명 표기 차이를 통일:

| 원본                             | 정규화  | 규칙                  |
| -------------------------------- | ------- | --------------------- |
| `서울2호선`                      | `2호선` | 서울 접두사 제거      |
| ` 1호선`                         | `1호선` | 앞뒤 공백 제거        |
| `5531번(군포동행정복지센터방면)` | `5531`  | `번(...)` 접미사 제거 |
| `5531번`                         | `5531`  | `번` 접미사 제거      |

적용 위치:

- OTP: `parse_otp_itinerary()` — `route.shortName` 파싱 시
- SC: `parse_smartcard_trip()` — `노선명` 파싱 시

### 6.2 정류장명 정규화 (`_stop_name_match`)

OTP와 SC의 정류장명 표기 차이를 fuzzy 매칭:

| OTP                          | SC             | 매칭 결과                               |
| ---------------------------- | -------------- | --------------------------------------- |
| `군포1동행정복지센터.군포역` | `군포역`       | `.`으로 분리된 부분 중 포함 관계 → 일치 |
| `산본중학교앞.G마크프라자`   | `산본중학교앞` | `.`으로 분리된 부분 중 포함 관계 → 일치 |

`compute_sequence_metrics()` 호출 시 `_normalize_stops_for_comparison()`으로 OTP 정류장명을 SC 기준으로 통일 후 비교.

### 6.3 거리 복원 (`fix_missing_distances`)

OTP transit leg의 distance=0 문제 → `legGeometry` polyline에서 좌표 복원 후 haversine 합산.

```python
fix_missing_distances(itinerary)  # in-place 수정
```

---

## 7. 유사도 가중치

### GTFS 사용 시 (기본)

| 레벨     | 가중치 |
| -------- | ------ |
| mode     | 0.10   |
| sequence | 0.15   |
| time     | 0.10   |
| route    | 0.15   |
| spatial  | 0.50   |

### GTFS 미사용 시 (USE_GTFS=False)

spatial 가중치를 나머지에 비례 재분배:

| 레벨     | 가중치 |
| -------- | ------ |
| mode     | 0.20   |
| sequence | 0.30   |
| time     | 0.20   |
| route    | 0.30   |
| spatial  | 0.00   |

---

## 8. 핵심 설계 결정

| 결정                              | 근거                                                 |
| --------------------------------- | ---------------------------------------------------- |
| 매칭 실패 시 skip (threshold=0.6) | 관측 불가능한 경로에 피처 부여 불가                  |
| MIN_CHOICE_SET_SIZE = 2           | 대안 1개뿐인 OD는 모델 학습에 무의미                 |
| OTP 피처는 OD별 사전계산          | 같은 OD의 SC trip들은 같은 대안 피처 공유 → 효율성   |
| OTP 파싱도 OD별 캐시              | `parse_otp_itinerary`를 매번 호출하지 않고 사전 파싱 |
| access/egress 분리                | GTX 연결성 분석(IR)에 활용                           |
| wait_time = 잔차 계산             | timestamp 불필요, 더 견고함                          |
| GTFS 룩업 optional                | 속도/정확도 트레이드오프, USE_GTFS 플래그로 제어     |
| ijson 스트리밍                    | 23GB JSON 파일을 메모리에 모두 올리지 않음           |
| polyline 거리 복원                | OTP transit leg distance=0 → legGeometry에서 복원    |
| 정류장명 fuzzy 매칭               | OTP `.` 구분 복합명 ↔ SC 단일명 불일치 해결          |

---

## 9. 모듈 의존성

```
route_features.py
  └── similarity.py (OTP_MODE_MAP, GTX_ROUTE_KEYWORDS,
                     _normalize_route_name, _mode_set_to_category, _haversine)

build_training_set.ipynb
  ├── route_features.py (extract_itinerary_features, extract_trip_context,
  │                      fix_missing_distances)
  ├── similarity.py (parse_otp_itinerary, parse_smartcard_trip,
  │                  deduplicate_itineraries, compute_all_metrics,
  │                  compute_composite_similarity)
  └── [optional] gtfs_lookup.py (GTFSRouteLookup)
```

---

## 10. 실행 방법

```bash
# 1. OTP 서버 실행 (이미 배치 결과가 있으면 불필요)
# 2. Jupyter에서 build_training_set.ipynb 실행
# 3. 설정 조정 (Cell 2):
#    - SIMILARITY_THRESHOLD: 매칭 기준 (기본 0.6)
#    - MIN_CHOICE_SET_SIZE: 최소 대안 수 (기본 2)
#    - USE_GTFS: GTFS 폴리라인 매칭 (기본 False)
#    - MAX_ODS: None=전체, int=테스트용 OD 수 제한
```

체크포인트 기능으로 중간에 중단해도 이어서 실행 가능 (2000 OD 단위).

---

## 11. 테스트 결과 (100 ODs 샘플)

| 항목                   | 값              |
| ---------------------- | --------------- |
| OTP ODs (중복 제거 후) | 89              |
| TCN 매칭 대상          | 4,233건         |
| 매칭 성공              | 3,954건 (93.4%) |
| 매칭 실패 skip         | 279건 (6.6%)    |
| 출력 행 수             | 12,467          |
| 선택상황(trips)        | 3,954           |
| 검증 (chosen=1 정합성) | PASS            |

### Chosen vs Not-chosen 비교

| 피처             | Chosen  | Not-chosen | 차이(%) |
| ---------------- | ------- | ---------- | ------- |
| total_duration   | 737.8s  | 823.4s     | -10.4%  |
| in_vehicle_time  | 652.3s  | 538.1s     | +21.2%  |
| wait_time        | 54.8s   | 113.9s     | -51.9%  |
| walk_time        | 30.7s   | 171.4s     | -82.1%  |
| fare             | 1,501원 | 1,508원    | -0.5%   |
| num_transfers    | 0.1     | 0.4        | -81.2%  |
| generalized_cost | 79,572  | 91,352     | -12.9%  |
