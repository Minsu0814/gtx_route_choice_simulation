# -*- coding: utf-8 -*-
"""
Step 2: Voronoi 대표점 기반 OTP/Raptor 입력 OD 생성

기존 정류장 OD의 출발/도착 좌표를 Voronoi centroid로 교체하여
Raptor 입력 CSV를 생성한다.

Usage:
    python step2_generate_od_input.py
"""

import sys; sys.stdout.reconfigure(encoding='utf-8')
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'data')
OTP_INPUT = os.path.join(DATA_DIR, 'otp', 'input', 'otp_od_input_over13.csv')
VOR_DIR = os.path.join(DATA_DIR, 'voronoi')
RAPTOR_DIR = 'C:/Research/multi-modal-routing-algorithm/data'


def main():
    print("=" * 60)
    print("Step 2: Voronoi 대표점 기반 Raptor 입력 OD 생성")
    print("=" * 60)

    # 1. 기존 OD 로드
    print("\n[1/3] 기존 OD 로드...")
    od = pd.read_csv(OTP_INPUT)
    print(f"  기존 OD: {len(od):,}개")

    # 2. Voronoi 대표점 로드
    print("\n[2/3] Voronoi 대표점 로드...")
    vor = pd.read_csv(os.path.join(VOR_DIR, 'voronoi_centroids.csv'))
    vor['stop_id'] = vor['stop_id'].astype(str)

    # stop_id → (vor_centroid_lat, vor_centroid_lon) 딕셔너리
    vor_lookup = dict(zip(
        vor['stop_id'],
        zip(vor['vor_centroid_lat'], vor['vor_centroid_lon'])
    ))
    print(f"  Voronoi 대표점: {len(vor_lookup)}개")

    # 3. 좌표 교체
    print("\n[3/3] 출발/도착 좌표를 Voronoi centroid로 교체...")
    od['o_stop_id'] = od['o_stop_id'].astype(str)
    od['d_stop_id'] = od['d_stop_id'].astype(str)

    # 원래 좌표 보존
    od['orig_o_lat'] = od['o_lat']
    od['orig_o_lon'] = od['o_lon']
    od['orig_d_lat'] = od['d_lat']
    od['orig_d_lon'] = od['d_lon']

    # Voronoi centroid로 교체
    matched_o = 0
    matched_d = 0
    for idx, row in od.iterrows():
        if row['o_stop_id'] in vor_lookup:
            vlat, vlon = vor_lookup[row['o_stop_id']]
            od.at[idx, 'o_lat'] = vlat
            od.at[idx, 'o_lon'] = vlon
            matched_o += 1
        if row['d_stop_id'] in vor_lookup:
            vlat, vlon = vor_lookup[row['d_stop_id']]
            od.at[idx, 'd_lat'] = vlat
            od.at[idx, 'd_lon'] = vlon
            matched_d += 1

    print(f"  출발 좌표 교체: {matched_o:,} / {len(od):,} ({matched_o/len(od)*100:.1f}%)")
    print(f"  도착 좌표 교체: {matched_d:,} / {len(od):,} ({matched_d/len(od)*100:.1f}%)")

    # Raptor 입력 형식으로 변환
    # converted_od 형식: from_lat, from_lon, to_lat, to_lon, departure_time
    raptor_input = pd.DataFrame({
        'from_lat': od['o_lat'],
        'from_lon': od['o_lon'],
        'to_lat': od['d_lat'],
        'to_lon': od['d_lon'],
        'departure_time': od['departure_time'].astype(str).str[8:10] + ':' + od['departure_time'].astype(str).str[10:12],
    })

    # 저장: Voronoi OD (전체 정보 포함)
    vor_od_path = os.path.join(VOR_DIR, 'voronoi_od_input.csv')
    od.to_csv(vor_od_path, index=False)
    print(f"\n  저장 (전체): {vor_od_path}")

    # 저장: Raptor 입력 형식
    raptor_path = os.path.join(VOR_DIR, 'raptor_input_voronoi.csv')
    raptor_input.to_csv(raptor_path, index=False)
    print(f"  저장 (Raptor): {raptor_path}")

    # Raptor 프로젝트에도 복사
    raptor_copy = os.path.join(RAPTOR_DIR, 'raptor_input_voronoi.csv')
    raptor_input.to_csv(raptor_copy, index=False)
    print(f"  복사 (Raptor): {raptor_copy}")

    # 통계
    print(f"\n=== 요약 ===")
    print(f"  총 OD: {len(od):,}")
    print(f"  출발 좌표 변경량 (평균): ", end="")
    import math
    def hav(lat1, lon1, lat2, lon2):
        R = 6371000
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
        a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return R*2*math.atan2(math.sqrt(a), math.sqrt(1-a))

    sample = od.head(10000)
    o_offsets = sample.apply(lambda r: hav(r['orig_o_lat'], r['orig_o_lon'], r['o_lat'], r['o_lon']), axis=1)
    d_offsets = sample.apply(lambda r: hav(r['orig_d_lat'], r['orig_d_lon'], r['d_lat'], r['d_lon']), axis=1)
    print(f"{o_offsets.mean():.0f}m (출발), {d_offsets.mean():.0f}m (도착)")

    print("\nStep 2 완료.")
    print(f"\n다음 단계: Raptor로 경로 생성")
    print(f"  입력 파일: {raptor_copy}")
    print(f"  Raptor 프로젝트: C:/Research/multi-modal-routing-algorithm/")


if __name__ == '__main__':
    main()
