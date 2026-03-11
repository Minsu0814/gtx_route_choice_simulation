# MNL 추정 결과 이슈 분석

## 1. fare β = 0 → 데이터 구조 문제 (모형 탓 아님)

- **OD의 59.8%에서 대안경로 간 요금 차이 = 0원** (중앙값도 0 KRW)
- 수도권 통합요금제 구조상 같은 OD면 거의 같은 요금
- 분산이 없는 변수는 추정 자체가 불가능 → 공선성/매칭 편향 아닌 구조적 현실

## 2. wait_time β = 0 → 분산은 있으나 설명력 부족

- zero_var OD: 12.3%, 중앙값 std: 95초 → 분산 자체는 존재
- 그러나 **choice_prob과의 상관 r = -0.056** (거의 무관)
- OTP wait_time이 스케줄 기반 → 실제 체감 대기시간과 괴리 가능성

## 3. walk 과대 / IVT 과소 → 내생성(endogeneity) 문제

### 핵심 상관관계

| 관계 | r |
|------|---|
| choice_prob ↔ sim_composite | **+0.76** |
| sim_composite ↔ access_time | **-0.37** |
| sim_composite ↔ egress_time | **-0.37** |
| choice_prob ↔ access_time | **-0.27** |
| choice_prob ↔ egress_time | **-0.27** |
| choice_prob ↔ in_vehicle_time | **+0.003** |

### 인과 경로

```
walk 작음 → spatial 유사도 높음 → 매칭 많이 됨 → choice_prob 높음 → β_walk 부풀려짐
```

- y(choice_prob)가 sim_composite에 강하게 의존 (r=0.76)
- sim_composite가 walk(access/egress)과 상관 (r=-0.37)
- **종속변수 y가 이미 walk 관련 정보를 포함** → β_walk 과대추정, β_IVT 과소추정

### Choice Set 내 변수별 분산 (OD별 std 중앙값)

| 변수 | median std | zero_var OD % |
|------|-----------|---------------|
| in_vehicle_time | 153.4 sec | 2.5% |
| wait_time | 95.3 sec | 12.3% |
| access_time | 39.2 sec | 38.6% |
| egress_time | 49.2 sec | 36.9% |
| fare | 0.0 KRW | **59.8%** |
| total_distance | 487.0 m | 8.2% |
| num_transfers | 0.5 | 29.4% |
| transfer_walk_time | 41.7 sec | 32.3% |

## 4. 해결 방안

| 방안 | 설명 | 난이도 |
|------|------|--------|
| **y 정의 변경** | sim_composite 기반 매칭 대신 mode + sequence만으로 매칭 (spatial 제거) | 중 |
| **2단계 추정** | 1단계: spatial 없이 매칭 → y 생성, 2단계: MNL 추정 | 중 |
| **binary choice** | choice_prob 대신 chosen=1/0 (1순위만 선택) → 내생성 완화 | 하 |
| **피처 스케일링** | 변수 단위 표준화 → 계수 크기 비교 가능하게 | 하 |
| **wait_time 재계산** | OTP 스케줄 기반 → headway 기반 평균대기로 대체 | 상 |

### 권장: spatial 유사도를 매칭에서 제거하고 y 재구성

- 가장 빠르고 효과적
- ρ²가 조금 내려갈 수 있지만, 계수의 행태적 합리성이 올라갈 것
