# Codebase & Data Cleanup (2026-04-13)

## 작업 내용

### 1. 불필요한 코드 삭제 (11 files)

구 OTP 기반 파이프라인 코드 전부 삭제:

- `build_training_set.py`, `build_filtered_training.py`, `build_filtered_training_set.py` — 구 학습데이터 빌더
- `attach_trip_counts.py` — 구 fingerprint 매칭 (현재는 composite similarity 사용)
- `choice_criterion.py`, `run_choice_criterion.py` — 구 분석
- `run_raptor_comparison.py`, `run_raptor_param_compare.py` — 중간 비교 코드
- `run_mnl_4spec.py`, `run_ml_4spec.py` — 구 binary MNL/ML (현재는 weighted MNL)
- `run_tastenet_4spec.py` — 구 TasteNet (참고용 module만 유지)

### 2. 불필요한 데이터 삭제 (9 files)

- `route_choice_training.csv` (339M) — 구 학습데이터
- `mnl_4spec.json`, `ml_4spec.json`, `tastenet_4spec.json` — 구 모형 결과
- `choice_criterion_analysis.json` — 구 분석 결과
- `raptor_comparison_*.parquet`, `raptor_coverage_*.parquet` — 중간 비교
- `route_choice_filtered_training.parquet` — binary chosen만 있던 중간 파일

### 3. 데이터 경로 재구성

**Before** (산발적):

```
data/otp/           — OD 입력 + similarity.json + 캐시
data/cache/         — otp/raptor 캐시
data/training_set/  — 구 학습데이터
data/training_set_new/ — 신 학습데이터
data/raptor_choice_set/ — Raptor choice set
```

**After** (체계적):

```
data/routing/
├── input/                 — OD 입력 CSV
└── output/
    ├── java_default/      — Java RAPTOR (기본 파라미터) 캐시
    └── rust_calibrated/   — Rust RAPTOR (보정 파라미터) 캐시
data/choice_set/
├── java_default/          — Java 학습데이터
└── rust_calibrated/       — Rust choice set
data/results/              — 모형 결과, 배분, 비교 분석
```

### 4. 네이밍 수정: OTP/Raptor → Java/Rust

- 둘 다 RAPTOR 알고리즘을 사용하므로 "OTP vs Raptor"는 부정확
- Java (OTP 서버, 기본 파라미터) vs Rust (dtumos-raptor, 보정 파라미터)로 구분

## 의사결정

- `deep_learning/module/` — TasteNet 모듈은 참고용으로 유지 (코드 자체는 깔끔)
- `analysis/raptor_calibration_*.py` — 보정 코드 유지 (재현 가능성)
- `spec_config.py` — 공통 설정으로 유지

## 최종 파이프라인 (5단계)

```
1. run_raptor_full.py              — Raptor 라우팅 (Rust, 보정 파라미터)
2. build_raptor_choice_set.py      — Choice set 구축 (중복제거 + 필터)
3. attach_raptor_trip_counts.py    — SC 매칭 (composite similarity)
4. mnl/run_mnl_weighted.py         — Weighted MNL 추정
5. mnl/assign_mnl.py               — 선택확률 배분
```
