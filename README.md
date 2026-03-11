# Route Choice Simulation Pipeline

스마트카드 데이터(TCD) 기반 경로 선택 모델 구축 파이프라인.
OTP 대안경로와 스마트카드 실적경로를 매칭하여 학습 데이터를 생성하고, MNL/딥러닝 경로선택모델을 추정한다.

---

## 실행 순서

### Phase 1. 데이터 전처리

| 순서 | 파일 | 설명 |
|:---:|------|------|
| 1 | `code/tcd_to_parquet.ipynb` | TCD DAT 파일 → Parquet 변환 |
| 2 | `code/tcd2tcn.ipynb` | TCD → TCN(Trip Chain Network) 변환. O-D 좌표 정제, 단거리 필터링 |
| 3 | `code/tcn_to_otp_od.ipynb` | TCN → OTP 입력용 OD pair 생성 |

### Phase 2. 유사도 매칭 & 학습 데이터 구축

| 순서 | 파일 | 설명 |
|:---:|------|------|
| 4 | `code/similarity_test_10od.ipynb` | OTP-스마트카드 6-Level 유사도 지표 검증 (10개 OD 샘플) |
| 5 | `code/sensitivity_2d_grid.ipynb` | 유사도 가중치 2D grid 탐색 (spatial × route). **`.py` 스크립트로 실행 권장** — 조합 수가 많아 노트북보다 `python sensitivity_2d_grid.py`가 훨씬 빠름 |
| 6 | `code/sensitivity_final_sweep.ipynb` | 임계값·정규화 거리 최종 파라미터 결정 |
| 7 | `code/build_training_set.ipynb` | 유사도 매칭 기반 choice set 구축 → `route_choice_training.parquet`. **먼저 `python build_training_set.py`로 데이터 생성 후 노트북에서 결과 확인** |

### Phase 3. 경로 선택 모델 추정

| 순서 | 파일 | 설명 |
|:---:|------|------|
| 8 | `code/route_choice_model.ipynb` | MNL / Conditional Logit 모델 추정 |
| 9 | `code/route_choice_deep_learning.ipynb` | DNN / TasteNet / ResLogit 모델 추정 |

---

## 모듈 구조

```
code/module/
├── preprocessing/              # 데이터 전처리
│   ├── tcd_to_tcn_route.py     # TCD → TCN 변환 (경로 기반)
│   └── tcn_to_otp_od.py        # TCN → OTP 입력 OD pair 생성
├── similarity/                 # 유사도 매칭
│   ├── similarity.py           # 6-Level 유사도 지표 + 복합 점수
│   ├── gtfs_lookup.py          # GTFS route → stop sequence 조회
│   └── route_features.py       # 경로 피처 추출 (21개 itinerary + 5개 context)
└── deep_learning/              # 딥러닝 모델
    ├── dl_data.py              # DataLoader, 피처 정의
    ├── dl_models.py            # DNN, TasteNet, ResLogit 모델 정의
    └── dl_train.py             # 학습·평가 루프
```

## 데이터 흐름

```
TCD (DAT)
  → Parquet
    → TCN (Trip Chain Network)
      → OTP OD pairs
        → OTP K-best 대안경로 + SC 실적경로 유사도 매칭
          → route_choice_training.parquet
            → MNL / DNN / TasteNet / ResLogit
```
