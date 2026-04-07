학습 데이터를 구축하거나 재구축합니다.

단계를 지정할 수 있습니다:

- /build-data h3-mapping — Step 1: 정류장→H3 매핑
- /build-data h3-aggregate — Step 2-3: H3 OD 합침 + access/egress 계산
- /build-data individual — Step 4: 개인 SC 통행 단위 학습 데이터
- /build-data all — 전체 재실행

기본은 현재 진행 상태를 확인하고 다음 단계를 실행합니다.

관련 코드:

- code/similarity/eb/build_h3_mapping.py
- code/similarity/eb/aggregate_h3_choice.py
- code/similarity/build_training_set.py
