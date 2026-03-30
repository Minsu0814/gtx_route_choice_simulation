# -*- coding: utf-8 -*-
"""
VoWT (Value of Walk Time) 일관성 검증

3가지 walk 파라미터의 implied VoWT를 비교:
1. MNL β_access / β_ivt → OTP 경로 내 접근 도보의 IVT 대비 가치
2. MNL β_transfer_walk / β_ivt → 환승 도보의 IVT 대비 가치
3. EB β_distance → 정류장 접근 도보의 감쇠율

세 가지가 일관된 범위이면 → walk 파라미터 전체의 신빙성 확보
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'data')
EB_DIR = os.path.join(DATA_DIR, 'eb')
TRAINING_DIR = os.path.join(DATA_DIR, 'training_set')


def main():
    print("=" * 60)
    print("VoWT (Value of Walk Time) 일관성 검증")
    print("=" * 60)

    # 1. MNL 계수 로드 (K3 사양 — run_mnl_no_walk.py 결과 또는 직접 지정)
    # K3 추정 결과 (run_k3.py / run_mnl_no_walk.py에서 확인된 값)
    beta_ivt = -0.023409
    beta_access_log = -3.190052
    beta_egress_log = -2.083088
    beta_transfer_walk = -0.067160
    beta_fare = -0.412315

    # K1도 참고용으로 로드
    mnl_path = os.path.join(TRAINING_DIR, 'mnl_coefficients.json')
    with open(mnl_path, encoding='utf-8') as f:
        mnl_k1 = json.load(f)
    beta_ivt_k1 = mnl_k1['beta'].get('bus_ivt_min', 0)  # K1은 bus_ivt만 유의

    print(f"\n[MNL 계수]")
    print(f"  β_ivt            = {beta_ivt:.6f} /min")
    print(f"  β_access (log)   = {beta_access_log:.6f}")
    print(f"  β_egress (log)   = {beta_egress_log:.6f}")
    print(f"  β_transfer_walk  = {beta_transfer_walk:.6f} /min")
    print(f"  β_fare           = {beta_fare:.6f} /1000won")

    # 2. EB β 로드
    eb_path = os.path.join(EB_DIR, 'eb_beta.json')
    with open(eb_path) as f:
        eb = json.load(f)
    beta_eb_dist = eb['beta']

    print(f"\n[EB 계수]")
    print(f"  β_distance       = {beta_eb_dist:.6f} /m")

    # 3. VoWT 계산
    print(f"\n{'=' * 60}")
    print("VoWT 비교")
    print(f"{'=' * 60}")

    # --- (a) Transfer walk VoWT ---
    # β_transfer_walk / β_ivt = 환승 도보 1분의 IVT 환산 (분)
    if beta_ivt != 0:
        vowt_transfer = beta_transfer_walk / beta_ivt
        print(f"\n(a) 환승 도보 VoWT")
        print(f"    β_transfer_walk / β_ivt = {beta_transfer_walk:.6f} / {beta_ivt:.6f}")
        print(f"    = {vowt_transfer:.2f}")
        print(f"    → 환승 도보 1분 = IVT {vowt_transfer:.2f}분과 동일한 비효용")
    else:
        vowt_transfer = None
        print(f"\n(a) 환승 도보 VoWT: β_ivt=0이라 계산 불가")

    # --- (b) Access walk VoWT (log 변환이라 marginal effect로 계산) ---
    # ln(1+x) 변환: marginal effect = β / (1+x)
    # 중앙값 access_time_min ≈ 0.12분 (7초/60)에서의 marginal
    # 5분 접근에서의 marginal
    print(f"\n(b) 접근 도보 VoWT (log 변환, marginal effect)")
    for walk_min, label in [(0.12, '중앙값 0.12분'), (1.0, '1분'), (3.0, '3분'), (5.0, '5분')]:
        marginal_access = beta_access_log / (1 + walk_min)
        if beta_ivt != 0:
            vowt_access = marginal_access / beta_ivt
            print(f"    walk={label}: marginal={marginal_access:.4f}, VoWT={vowt_access:.2f}× IVT")
        else:
            print(f"    walk={label}: marginal={marginal_access:.4f}, β_ivt=0이라 VoWT 계산 불가")

    # --- (c) EB distance VoWT ---
    # β_distance (per meter) → per minute (도보 80m/min)
    beta_eb_per_min = beta_eb_dist * 80  # 80m/min 도보속도
    print(f"\n(c) EB 정류장 접근 VoWT")
    print(f"    β_distance = {beta_eb_dist:.6f} /m")
    print(f"    β_distance × 80m/min = {beta_eb_per_min:.6f} /min (도보시간 환산)")
    if beta_ivt != 0:
        vowt_eb = beta_eb_per_min / beta_ivt
        print(f"    VoWT = {vowt_eb:.2f}× IVT")
    else:
        vowt_eb = None
        print(f"    β_ivt=0이라 VoWT 계산 불가")

    # --- (d) 요금 환산 ---
    print(f"\n(d) 요금 환산 (참고)")
    if beta_fare != 0:
        # IVT 1분의 금전적 가치
        vot = beta_ivt / beta_fare * 1000  # won/min
        print(f"    VoT (IVT) = β_ivt / β_fare × 1000 = {vot:.0f} 원/분")
        if vowt_transfer is not None:
            print(f"    환승 도보 가치 = VoT × VoWT = {vot * vowt_transfer:.0f} 원/분")
        print(f"    EB 도보 가치 = β_distance / β_fare × 1000 = {beta_eb_dist / beta_fare * 1000 * 1000:.1f} 원/km")

    # 4. 일관성 요약
    print(f"\n{'=' * 60}")
    print("일관성 요약")
    print(f"{'=' * 60}")

    print(f"\n{'Source':<30} {'Walk 1분의 IVT 환산':>20}")
    print('-' * 52)
    if vowt_transfer is not None:
        print(f"{'환승 도보 (MNL)':<30} {vowt_transfer:>20.2f}× IVT")
    if beta_ivt != 0:
        for walk_min in [1.0, 3.0, 5.0]:
            mg = beta_access_log / (1 + walk_min)
            vw = mg / beta_ivt
            print(f"{'접근 도보 @' + str(walk_min) + '분 (MNL)':<30} {vw:>20.2f}× IVT")
    if vowt_eb is not None:
        print(f"{'EB 정류장 접근 (SC 행태)':<30} {vowt_eb:>20.2f}× IVT")

    print(f"\n해석:")
    print(f"  - VoWT > 1: 도보 1분이 차내시간 1분보다 더 싫음 (일반적)")
    print(f"  - VoWT = 1~3: 합리적 범위 (문헌)")
    print(f"  - VoWT > 5: 과대 추정 가능성")

    # 저장
    summary = {
        'mnl_coefficients': {
            'beta_ivt': float(beta_ivt),
            'beta_access_log': float(beta_access_log),
            'beta_egress_log': float(beta_egress_log),
            'beta_transfer_walk': float(beta_transfer_walk),
            'beta_fare': float(beta_fare),
        },
        'eb_beta_distance': float(beta_eb_dist),
        'vowt_transfer': float(vowt_transfer) if vowt_transfer else None,
        'vowt_eb': float(vowt_eb) if vowt_eb else None,
    }
    out_path = os.path.join(EB_DIR, 'vowt_validation.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n저장: {out_path}")


if __name__ == '__main__':
    main()
