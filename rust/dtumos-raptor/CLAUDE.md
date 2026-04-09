# dtumos-raptor — Rust RAPTOR Transit Routing Engine

PyO3 기반 Rust 네이티브 RAPTOR 엔진. Python에서 `from dtumos_raptor import DtumosRaptor`로 사용.

## 빌드

```bash
cd rust/dtumos-raptor
maturin develop --release
```

## 구조

```
src/
├── lib.rs           # PyO3 진입점 (DtumosRaptor class)
├── batch.rs         # 배치 라우팅 + OD 그룹핑 + attrs 모드
├── types.rs         # TransitData, TimeTable, Transfer 등 핵심 타입
├── raptor/
│   ├── engine.rs    # Standard RAPTOR (이중기준: time+GC)
│   ├── mc_raptor.rs # Multi-criteria RAPTOR (Pareto bag)
│   ├── label.rs     # LabelArena + ParetoBag
│   ├── cost.rs      # CostConfig, TaxiCostConfig (reluctance, slack)
│   ├── fare.rs      # T-money 요금 계산 (shape polyline 기반)
│   └── range.rs     # Range-RAPTOR (시간 윈도우 탐색)
├── gtfs/
│   ├── builder.rs   # GTFS → TransitData 빌드 + 바이너리 캐시
│   └── loader.rs    # CSV 파서
├── access/
│   ├── finder.rs    # 도보/택시 access stop 탐색 (R-tree)
│   ├── street_graph.rs  # 도로 네트워크 Dijkstra
│   └── walk.rs      # 도보 거리 추정
└── output/
    ├── itinerary.rs # OTP JSON 변환 + attributes 계산
    └── polyline.rs  # Google polyline 인코딩
```

## 속도 최적화 옵션

대규모 배치(30K+) 시뮬레이션에서 속도가 필요할 때 사용할 수 있는 옵션.

### Speed Presets

Python에서 `bridge.set_speed_preset(name)`으로 한 번에 설정:

| Preset     | H3 해상도  | 시간 버켓 | 모드     | 시각화 | 용도           | 오차 |
| ---------- | ---------- | --------- | -------- | ------ | -------------- | ---- |
| `precise`  | 0 (없음)   | 0         | McRAPTOR | 100%   | <3K, 정밀 분석 | 0%   |
| `balanced` | 0 (없음)   | 0         | Standard | 10%    | 3K-30K         | 0%   |
| `fast`     | 8 (~460m)  | 120s      | Standard | 10%    | 30K-100K       | ~10% |
| `turbo`    | 7 (~1.2km) | 300s      | Standard | 0%     | 100K+          | ~15% |

### 개별 파라미터

- `h3_resolution`: H3 해상도 (0=정밀, 7-9=버켓팅). 같은 H3 셀의 OD 쿼리가 RAPTOR compute를 공유.
- `time_bucket_secs`: 시간 버켓 (0=정밀, 120=2분). 같은 버켓의 출발시각이 하나로 묶임.
- `route_batch_attrs()`: JSON/polyline 스킵, numpy 배열 직반환 (시뮬레이션 전용).
- `viz_ratio`: 시각화 trip 비율 (0.0-1.0). transit_simulator.save_results()에서 사용.

### 벤치마크 (서울 28K stops, 30K queries, 10orig x 5dest)

| 설정           | per-query | 스피드업 | 1M 추정 |
| -------------- | --------- | -------- | ------- |
| 디폴트         | 1.55ms    | 1x       | ~26분   |
| H3 res9 + 120s | 0.087ms   | 18x      | ~1.5분  |
| H3 res8 + 120s | 0.039ms   | 40x      | ~39초   |
| H3 res7 + 300s | 0.010ms   | 155x     | ~10초   |

### 사용 예시

```python
# 시뮬레이션 config에서 지정
configs["speed_preset"] = "fast"

# 또는 코드에서 직접
bridge.set_speed_preset("fast")
attrs = bridge.batch_route_attrs_from_df(df)  # numpy 배열 반환

# 개별 파라미터
attrs = bridge.batch_route_attrs_from_df(df, h3_resolution=8, time_bucket_secs=120)
```

## 주요 알고리즘 특성

- **earliest_trip**: 빌드 시 per-stop 정렬 검증 → 추월 trip이 있으면 linear scan fallback
- **Standard RAPTOR**: 이중기준 (arrival_time + generalized_cost) — GC 최적 경로도 탐색
- **McRAPTOR**: 3차원 Pareto (time, transfers, GC) — 카테고리별 최적 경로 반환
- **Grouped batch**: 같은 origin의 쿼리는 RAPTOR compute 1회 공유 → extract만 per-destination
- **OD grouped attrs**: origin+destination 둘 다 H3 셀로 버켓 → compute+extract 모두 공유
- **orjson**: 프로젝트 전체 JSON I/O에 orjson 적용 (write 28-50x faster)
