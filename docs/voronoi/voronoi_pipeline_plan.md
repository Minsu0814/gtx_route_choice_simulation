# Voronoi 대표점 기반 OTP 재쿼리 파이프라인

## 목적

기존 정류장 OD에서 "선택된 경로 walk ≈ 0" 문제를 해결하기 위해,
출발/도착 좌표를 정류장이 아닌 **Voronoi 접근권역 centroid**로 교체하여
OTP 경로를 재생성한다.

## 기존 문제

```
기존:  정류장A 좌표 → OTP → 선택된 경로 walk ≈ 0
       → VoWT 과대추정 (access β = -3.19, VoWT = 22~68× IVT)
       → 실제 도보 민감도가 아닌 "정류장 전환 페널티"를 학습
```

## 해결 방법

```
변경:  Voronoi centroid → OTP → 선택된 경로도 walk > 0
       → 모든 경로에 현실적 walk 값
       → VoWT가 문헌 범위 (1~3×)에 가까워질 것으로 기대
```

## 전체 파이프라인

```
[Step 1] Voronoi Diagram + 대표점 계산
    입력: 34,454개 정류장 좌표
    처리: Voronoi → 각 셀 centroid (= 접근권역 중심)
    출력: voronoi_centroids.csv

[Step 2] Raptor 입력 OD 생성
    입력: 기존 526,968 OD + voronoi_centroids.csv
    처리: 출발/도착 좌표를 Voronoi centroid로 교체
    출력: raptor_input_voronoi.csv → Raptor 프로젝트로 복사

[Step 3] Raptor 경로 생성 ← 민수가 직접 실행
    입력: raptor_input_voronoi.csv
    처리: MC-Raptor (numItineraries=10, searchWindow=7200)
    출력: similarity_voronoi.json

[Step 4] SC 매칭 + 학습 데이터 구축
    입력: similarity_voronoi.json + TCN (SC) + GTFS
    처리: 기존 build_training_set.py 파이프라인 재활용
          (transit leg 기준 매칭 → choice_prob → 피처 추출)
    출력: route_choice_training_voronoi.parquet

[Step 5] MNL 추정 + 비교
    입력: route_choice_training_voronoi.parquet
    처리: K3 사양 MNL 추정
    비교: 기존 모형 (정류장 OD) vs Voronoi 모형
          - ρ², Top-1, Top-3
          - β 비교 (특히 access/egress)
          - VoWT 비교 (과대추정 해소 여부)

[Step 6] 검증
    - 수단 분담률 비교
    - VoWT 일관성 (환승 도보 vs 접근 도보)
    - SC 매칭률 비교 (정류장 OD vs Voronoi)
```

## SC 매칭이 유지되는 이유

Voronoi centroid는 해당 정류장의 접근권역 안에 있으므로,
OTP가 해당 정류장 경유 경로를 높은 확률로 반환한다.

```
Voronoi centroid (정류장A에서 250m) → Raptor 쿼리
  경로1: 도보 250m → 정류장A → 버스100    ← SC: 정류장A 탑승 → 매칭 ✓
  경로2: 도보 400m → 지하철역B → 2호선     ← 비선택 대안
  경로3: 도보 150m → 정류장C → 버스200     ← 비선택 대안
```

핵심: transit leg(정류장, 노선, 수단)가 일치하면 기존 매칭 로직 그대로 사용 가능.

## 기대 효과

| 항목 | 기존 (정류장 OD) | Voronoi |
|---|---|---|
| 선택된 경로 walk | ≈ 0 | **> 0 (현실적)** |
| VoWT | 22~68× (과대) | **1~3× (기대)** |
| SC 매칭 | 정확 | 유사 (transit leg 기준) |
| β_access | -3.19 (과대?) | **적절한 크기 기대** |

## 현재 진행 상태

- [x] Step 1: voronoi_centroids.csv 생성 코드
- [x] Step 2: raptor_input_voronoi.csv 생성 코드
- [ ] Step 3: Raptor 실행 (대기 중)
- [ ] Step 4: SC 매칭 + 학습 데이터
- [ ] Step 5: MNL 추정 + 비교
- [ ] Step 6: 검증
