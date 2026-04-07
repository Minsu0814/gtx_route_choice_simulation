MNL 경로선택모형을 지정한 사양으로 추정하고 결과를 테이블로 출력합니다.

사양을 지정하지 않으면 기본 K3 (9피처)로 실행합니다.
피처 목록을 수정하려면 인자로 전달합니다.

실행 후 반드시 다음을 출력:

1. β, t-stat 테이블
2. Train/Test ρ², Top-1, Top-3
3. 부호 검증 결과
4. 기존 K3와 비교

코드: code/similarity/mnl/run_k3.py 또는 code/similarity/eb/run_mnl_no_walk.py
