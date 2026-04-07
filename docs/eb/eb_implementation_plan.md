# EB 정류장 배분 + 경로선택모형 구현 계획

---

## 1. 전체 구조

```
기존 파이프라인 (건드리지 않음)
├── data/otp/input/          → 정류장 OD 목록
├── data/otp/output/         → OTP Raptor 경로 (similarity.json)
├── data/training_set/       → 학습 데이터 (route_choice_training.parquet)
├── code/similarity/module/  → 유사도, 피처 추출
├── code/similarity/mnl/     → MNL 추정 (run_k3.py 등)
└── 모형 결과: mnl_coefficients.json, model_evaluation.json

새로 만들 것 (EB 모듈)
├── code/similarity/eb/
│   ├── __init__.py
│   ├── build_h3_mapping.py      ← Phase 1: 정류장→H3 매핑
│   ├── estimate_beta.py         ← Phase 2: β 추정 (MLE)
│   ├── eb_allocation.py         ← Phase 2: EB prior/posterior 계산
│   ├── aggregate_h3_choice.py   ← Phase 3: H3 단위 choice_prob 재집계
│   └── apply_eb_model.py        ← Phase 5: GTX 적용 파이프라인
└── data/eb/
    ├── stop_h3_mapping.csv      ← 정류장-H3 매핑
    ├── h3_stop_distance.csv     ← centroid-정류장 거리
    ├── eb_beta.json             ← β 추정 결과
    ├── eb_prior.csv             ← prior 확률
    ├── eb_posterior.csv         ← posterior 확률
    └── h3_choice_prob.parquet   ← H3 재집계 결과
```

---

## 2. 처리 흐름

```
[Phase 1] build_h3_mapping.py
  입력: data/gtfs/a1/stops.txt + data/otp/input/otp_od_input_over13.csv
  처리: 정류장 좌표 → H3 셀 매핑 (res 8)
        H3 centroid → 각 정류장 거리 계산
  출력: stop_h3_mapping.csv, h3_stop_distance.csv

[Phase 2] estimate_beta.py + eb_allocation.py
  입력: h3_stop_distance.csv + SC 데이터 (data/tcn/)
  처리: 거리-이용빈도 관계에서 β 추정 (MLE)
        prior = exp(-β × distance)
        likelihood = SC 관측 빈도
        posterior = prior × likelihood
  출력: eb_beta.json, eb_prior.csv, eb_posterior.csv

[Phase 3] aggregate_h3_choice.py
  입력: route_choice_training.parquet + stop_h3_mapping.csv
  처리: 기존 정류장 OD 결과에 H3 키 부여
        H3 OD 단위로 choice_prob 재집계
  출력: h3_choice_prob.parquet

[Phase 5] apply_eb_model.py
  입력: H3→H3 수요 + eb_posterior + 경로선택모형
  처리: EB로 정류장 배분 → 각 정류장 OD에 경로선택모형 적용
  출력: 경로별 최종 수요
```

---

## 3. 적용 시 전체 흐름

```
H3 셀 → H3 셀: 1000명 통행

[1단계] EB 정류장 배분:
  H3 셀 내 정류장 목록 추출
  각 정류장에서 목적지까지 OTP 경로 존재 여부 확인 (경로 없는 정류장 제외)
  EB posterior로 정류장별 인원 배분:
    GTX역:      100명
    지하철역:    700명
    버스정류장:  200명

[2단계] 각 정류장에서 경로선택모형 적용:
  GTX역 100명:
    → 각 경로의 피처 (소요시간, 요금, 환승 등)로 선택확률 예측
    → GTX직통 80명, GTX+환승 20명

  지하철역 700명:
    → 2호선+1호선 490명, 2호선+버스 210명

  버스정류장 200명:
    → 버스100 120명, 버스200 80명

[결과] 경로별 최종 수요
```

---

## 4. 경로선택모형 walk 피처 처리

EB가 1단계에서 도보 거리 기반으로 정류장 배분을 처리하므로, 2단계 경로선택모형에서는 access_time/egress_time 피처를 제거하여 walk 영향의 이중 반영을 방지한다.

- 기존 모형 (walk 포함): β_access = -3.190, β_egress = -2.083
- EB 연동 모형 (walk 제거): 재추정 필요 → walk 제거 전후 성능 비교

---

## 5. GTX 신설역 처리

GTX역이 기존 지하철역에 병설되는 경우 (강남역 등):

- 물리적으로 같은 위치 → EB prior가 거의 동일
- 하나의 접근점으로 묶고, GTX vs 지하철 선택은 경로선택모형에서 처리

GTX-B/C/D 신설 시:

- SC 데이터 없음 → posterior = prior (도보 거리만으로 배분)
- GTX-A의 SC 패턴을 참고하여 prior 보정 가능성 검토

---

## 6. 구현 순서

| 순서 | 작업                                               | 파일                   |
| ---- | -------------------------------------------------- | ---------------------- |
| 1    | 정류장 좌표 → H3 매핑, centroid-정류장 거리 계산   | build_h3_mapping.py    |
| 2    | SC 이용빈도 ~ 도보거리 관계에서 β 추정 (MLE)       | estimate_beta.py       |
| 3    | EB prior/likelihood/posterior 계산                 | eb_allocation.py       |
| 4    | 기존 학습 데이터에 H3 키 부여 + choice_prob 재집계 | aggregate_h3_choice.py |
| 5    | walk 피처 제거한 경로선택모형 재추정 + 성능 비교   | run_k3.py 수정         |
| 6    | GTX 시나리오 적용 파이프라인                       | apply_eb_model.py      |
