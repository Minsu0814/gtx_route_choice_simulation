# Assignment & Raptor Calibration (2026-04-08~09)

## 작업 내용

### 1. 경로 선택 확률 분포 분석

- `code/similarity/analysis/choice_criterion.py` — 기준별 매칭률 분석
- 결과: min_transfers 97.2%, min_fare 93.1%, min_walk 86.7%, min_gc 52.6%, min_duration 46.8%
- 모델 Top-1: min_transfers 99.8%, min_gc 73.1%
- 사람들은 환승 최소화를 가장 중시, generalized cost 최소화는 절반만 일치

### 2. Raptor 파라미터 보정 (calibration)

#### v1: 단일 transit_reluctance

- 1,800 조합 grid search (gc_accuracy 단일 목적함수)
- 최적: transfer=600s, transit_rel=0.3, walk_rel=2.0
- GC accuracy: 48.4% → 66.0%

#### v2: MNL β 기반 + 수단별 reluctance + 다중 목적함수 (2026-04-09)

- MNL β 비율로 초기값 산출: walk_rel≈10.1, transfer≈9387s
- bus_rel / train_rel 분리 (ivt_bus, ivt_train+ivt_gtx)
- 4가지 메트릭 복합 점수: gc_accuracy(0.4) + category_match(0.3) + duration_180s(0.2) + transfer_match(0.1)
- 6,750 조합 2-phase grid search (Phase1: fast reduceat → Phase2: top-50 full scoring)
- **최적 파라미터**: transfer=4800s, bus_rel=1.5, train_rel=0.3, walk_rel=10.0
- **결과**:
  | 메트릭 | Default | v2 최적 | 변화 |
  |--------|---------|---------|------|
  | GC accuracy | 48.3% | 66.1% | +17.8pp |
  | Category match | 95.5% | 99.2% | +3.7pp |
  | Duration ±180s | 72.9% | 89.6% | +16.7pp |
  | Transfer match | 78.0% | 97.2% | +19.2pp |
- walk_rel=10은 MNL β ratio(10.1)와 거의 정확히 일치 — MNL 근거 검증됨
- train_rel(0.3) < bus_rel(1.5): 사람들이 철도를 선호하는 행태 반영

### 3. Raptor bucketing 최적화 (Rust)

- `rust/dtumos-raptor/` — H3 셀 + 시간 버킷 기반 배치 처리
- preset: precise(1x) / balanced / fast(40x) / turbo(155x)
- H3 res8 + 120s bucket → 서울 28K 정류장 기준 40배 속도 향상

### 4. MNL Assignment

- `code/similarity/mnl/assign_mnl.py` — 학습된 β로 선택 확률 배정
- 출력: `assignment_A_total_uncon.parquet`

## 의사결정

- IVT는 Total IVT로 확정 (수단 분리 시 collinearity로 양수 β 발생)
- EB 피처는 경로선택 모형에서 작동 안 함 → 적용 단계 배분용으로만 사용

## 다음 작업

- GTX 평가 방법론 문서 업데이트
- Raptor PR (교수님 Rust 코드에 금액 계산 수정)
- 논문 작성 본격 시작
