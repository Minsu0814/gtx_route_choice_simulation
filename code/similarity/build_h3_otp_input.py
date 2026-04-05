# -*- coding: utf-8 -*-
"""
H3 OD 단위 OTP 입력 데이터 생성

261,359개 H3 OD에 대해 H3 centroid 좌표로 OTP 쿼리용 입력 생성.
departure_time은 해당 H3 OD에서 가장 통행이 많은 stop OD의 시간 사용.

Usage:
    python build_h3_otp_input.py
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / 'data'
EB_DIR = DATA_DIR / 'eb'
OTP_DIR = DATA_DIR / 'otp' / 'input'

def main():
    print("H3 OD OTP 입력 데이터 생성")
    print("=" * 60)

    # 1. H3 centroid 좌표
    print("[1/4] H3 centroid 좌표 로드...")
    mapping = pd.read_csv(EB_DIR / 'stop_h3_mapping.csv')
    h3_centroids = mapping.groupby('h3_res8').agg(
        lat=('centroid_lat_res8', 'first'),
        lon=('centroid_lon_res8', 'first'),
    ).to_dict('index')
    print(f"  H3 셀: {len(h3_centroids):,}")

    # 2. H3 OD 목록 + 대표 stop OD
    print("[2/4] H3 OD 목록 로드...")
    h3 = pd.read_parquet(EB_DIR / 'h3_choice_prob.parquet',
                          columns=['h3_od', 'od_pair', 'n_matched'])

    # H3 OD별 대표 stop OD (n_matched 최대)
    idx_max = h3.groupby('h3_od')['n_matched'].idxmax()
    h3_rep = h3.loc[idx_max, ['h3_od', 'od_pair', 'n_matched']].reset_index(drop=True)
    print(f"  H3 OD: {len(h3_rep):,}")

    # H3 OD별 총 통행수
    h3_total = h3.groupby('h3_od')['n_matched'].sum().rename('total_trips')
    h3_rep = h3_rep.merge(h3_total, on='h3_od')

    # 3. departure_time 매핑 (기존 OTP 입력에서)
    print("[3/4] departure_time 매핑...")
    otp_input = pd.read_csv(OTP_DIR / 'otp_od_input.csv',
                             usecols=['od_pair', 'departure_time', 'trip_count'])
    od_to_time = dict(zip(otp_input['od_pair'], otp_input['departure_time']))

    h3_rep['departure_time'] = h3_rep['od_pair'].map(od_to_time)

    # 매핑 안 된 건 기본값 (평일 오전 8시)
    default_time = 20250219080000
    n_missing = h3_rep['departure_time'].isna().sum()
    h3_rep['departure_time'] = h3_rep['departure_time'].fillna(default_time).astype(int)
    print(f"  departure_time 매핑: {len(h3_rep) - n_missing:,} 성공, {n_missing:,} 기본값")

    # 4. OTP 입력 생성
    print("[4/4] OTP 입력 생성...")
    records = []
    n_skip = 0
    for _, row in h3_rep.iterrows():
        h3_od = row['h3_od']
        parts = h3_od.split('_')
        o_h3 = parts[0]
        d_h3 = '_'.join(parts[1:])  # H3 셀 ID에 _가 없으므로 사실상 parts[1]

        o_info = h3_centroids.get(o_h3)
        d_info = h3_centroids.get(d_h3)
        if o_info is None or d_info is None:
            n_skip += 1
            continue

        records.append({
            'h3_od': h3_od,
            'o_h3': o_h3,
            'd_h3': d_h3,
            'o_lat': o_info['lat'],
            'o_lon': o_info['lon'],
            'd_lat': d_info['lat'],
            'd_lon': d_info['lon'],
            'departure_time': int(row['departure_time']),
            'trip_count': int(row['total_trips']),
            'rep_stop_od': row['od_pair'],
        })

    out = pd.DataFrame(records)

    # 저장
    out_path = OTP_DIR / 'otp_h3_od_input.csv'
    out.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"완료")
    print(f"{'=' * 60}")
    print(f"  출력: {out_path}")
    print(f"  H3 OD: {len(out):,}")
    print(f"  스킵: {n_skip}")
    print(f"  trip_count: mean={out['trip_count'].mean():.0f}, median={out['trip_count'].median():.0f}")
    print(f"  좌표 범위:")
    print(f"    O lat: {out['o_lat'].min():.4f} ~ {out['o_lat'].max():.4f}")
    print(f"    O lon: {out['o_lon'].min():.4f} ~ {out['o_lon'].max():.4f}")
    print(f"    D lat: {out['d_lat'].min():.4f} ~ {out['d_lat'].max():.4f}")
    print(f"    D lon: {out['d_lon'].min():.4f} ~ {out['d_lon'].max():.4f}")

    # 샘플 출력
    print(f"\n  샘플 5행:")
    print(out.head().to_string(index=False))


if __name__ == '__main__':
    main()
