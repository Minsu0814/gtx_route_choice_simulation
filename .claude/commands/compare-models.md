5가지 사양으로 MNL 또는 TasteNet을 비교 실행합니다.

A. Stop OD + walk (9피처) — 기준
B. Stop OD - walk (6피처)
C. H3 OD + OTP walk (9피처)
D. H3 OD + 전부 (13피처)
E. H3 OD - OTP + EB (10피처)

모형 종류는 인자로 지정 (기본: MNL):

- /compare-models mnl
- /compare-models tastenet

결과를 비교 테이블로 출력 (ρ², Top-1, Top-3, 피처수)

MNL: code/similarity/eb/run_mnl_h3_comparison.py
TasteNet: code/similarity/eb/run_tastenet_comparison.py
