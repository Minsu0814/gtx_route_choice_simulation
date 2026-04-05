# 대중교통 경로선택모형 프로젝트

## 프로젝트 개요

SC 전수 데이터 + Raptor 기반 대중교통 경로선택모형 연구.
GTX 영향평가를 위한 경로선택확률 예측 모형 구축.

## 현재 상태

- **기존 모형**: MNL K3 사양 (9피처, ρ²=0.53, Top-1=72.5%) — 잘 작동
- **TasteNet**: ρ²=0.54, Top-1=74.1% — MNL보다 약간 개선
- **EB 실험 완료**: EB 피처는 MNL/TasteNet 모두 β=0 → 적용 단계 배분용으로만 사용
- **Voronoi**: Raptor 입력 생성 완료, 실행 대기 중

## 다음 작업 (교수님 지시)

EB 빼고, 기존 OTP 결과에 first/last mile만 매핑 + H3 셀 단위 합침:
1. ✅ H3 매핑 + OSRM 거리 (완료)
2. ✅ H3 OD choice set 통합 (완료)
3. ✅ 외부 access/egress 계산 (완료)
4. ❌ **개인 SC 통행 단위 학습 데이터** (route_choice_individual.parquet → chosen=0/1)
5. ❌ **MNL 추정** (외부 access/egress + H3 OD)

## 코드 구조

```
code/similarity/
├── module/          — 유사도, 피처 추출 (공통)
├── mnl/             — MNL 추정 (run_k3.py 등)
├── deep_learning/   — TasteNet, DNN 등
├── eb/              — EB 정류장 배분 모듈
├── voronoi/         — Voronoi 대표점 파이프라인
└── build_training_set.py — 학습 데이터 구축
```

## 데이터

```
data/
├── training_set/    — 학습 데이터, MNL 계수, OTP 캐시
├── tcn/             — SC 7일치 (2025.02.17~23)
├── gtfs/a1/         — GTFS (정류장, 노선, 시간표)
├── otp/             — OTP 입력/출력
├── eb/              — EB 결과 (H3 매핑, β, posterior 등)
└── voronoi/         — Voronoi centroid, Raptor 입력
```

## Git

- 브랜치: Minsu-eb
- camuslab: https://github.com/camuslab/gtx-route-choice-simulation
- origin: https://github.com/Minsu0814/gtx_route_choice_simulation
- 커밋 시 양쪽 모두 푸시

## 핵심 수치 (외우기)

- MNL K3: ρ²=0.5304, Top-1=72.5%, β_access=-3.19, β_transfers=-3.40
- TasteNet: ρ²=0.5422, Top-1=74.1%
- H3 OD: ρ²=0.5249, Top-1=67.2%
- EB β: distance≈0, n_routes=+0.243
- LOO: within-cell corr=0.30 (M2)
- 정류장 34,454개, H3 셀 4,801개, OD 526,968개

## OSRM

Docker 컨테이너 `osrm-foot` (localhost:5000, 한국 foot routing)
시작: `docker start osrm-foot`

## Raptor

별도 레포: C:/Research/multi-modal-routing-algorithm/
Voronoi 입력이 data/raptor_input_voronoi.csv에 준비됨

## 원칙
- 한국어 해요체, 코드는 영어
- 3단계 이상 작업 → Plan Mode 필수
- 일이 꼬이면 즉시 멈추고 re-plan
- Git: Conventional commits (feat:, fix:, docs:, refactor:)

## Notes 디렉토리
- 작업/기능마다 `notes/` 디렉토리를 유지할 것
- 작업 중 발견한 내용, 의사결정 이유, 삽질 로그 등을 기록
- PR 올린 후에도 학습 내용이 남도록 `lessons.md` 파일 업데이트 해줘
- 형식: `notes/{작업명}.md`

## 코드 규칙
- 함수 30줄 이내, 파일 300줄 이내
- API 엔드포인트마다 에러 핸들링

## 검증
- 수정 후 관련 테스트 실행
- "시니어 엔지니어가 승인할까?" 기준

## 교훈 (틀린 것 다시 안 틀리기 위한 카닝)
<!-- 실수할 때마다 여기에 추가. 절대 삭제하지 말 것. -->
