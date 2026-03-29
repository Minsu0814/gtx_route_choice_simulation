# -*- coding: utf-8 -*-
"""
Phase 1: 정류장 → H3 매핑 + centroid-정류장 거리 계산

Usage:
    python build_h3_mapping.py                # 기본 (res 8)
    python build_h3_mapping.py --res 7 8 9    # 여러 resolution
"""

import argparse
import math
import os
import sys
import time

import h3
import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'data')
OTP_INPUT_CSV = os.path.join(DATA_DIR, 'otp', 'input', 'otp_od_input_over13.csv')
GTFS_STOPS = os.path.join(DATA_DIR, 'gtfs', 'a1', 'stops.txt')
OUTPUT_DIR = os.path.join(DATA_DIR, 'eb')


def haversine(lat1, lon1, lat2, lon2):
    """두 좌표 간 거리 (미터)"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


OSRM_URL = 'http://localhost:5000'


def osrm_walk_distance(lat1, lon1, lat2, lon2):
    """OSRM foot profile로 네트워크 도보거리 (미터) 조회.
    실패 시 haversine × 1.4 fallback."""
    try:
        url = f"{OSRM_URL}/route/v1/foot/{lon1},{lat1};{lon2},{lat2}?overview=false"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get('code') == 'Ok':
            return data['routes'][0]['distance']
    except Exception:
        pass
    return haversine(lat1, lon1, lat2, lon2) * 1.4


def osrm_walk_distances_batch(origins, destinations, batch_size=50):
    """여러 OD에 대해 OSRM 거리를 배치로 조회.

    Args:
        origins: list of (lat, lon)
        destinations: list of (lat, lon)

    Returns:
        list of distances (meters)
    """
    results = []
    for i in range(0, len(origins), batch_size):
        batch_o = origins[i:i+batch_size]
        batch_d = destinations[i:i+batch_size]
        for (olat, olon), (dlat, dlon) in zip(batch_o, batch_d):
            dist = osrm_walk_distance(olat, olon, dlat, dlon)
            results.append(dist)
    return results


def load_stop_coords():
    """OTP 입력 CSV에서 정류장 좌표 추출 (승차/하차 모두)"""
    df = pd.read_csv(OTP_INPUT_CSV)

    o_stops = df[['o_stop_id', 'o_lat', 'o_lon']].rename(
        columns={'o_stop_id': 'stop_id', 'o_lat': 'lat', 'o_lon': 'lon'}
    )
    d_stops = df[['d_stop_id', 'd_lat', 'd_lon']].rename(
        columns={'d_stop_id': 'stop_id', 'd_lat': 'lat', 'd_lon': 'lon'}
    )

    stops = pd.concat([o_stops, d_stops]).drop_duplicates('stop_id').reset_index(drop=True)
    print(f"고유 정류장 수: {len(stops)}")
    return stops


def build_h3_mapping(stops, resolution):
    """정류장 좌표 → H3 셀 매핑"""
    stops = stops.copy()
    res_col = f'h3_res{resolution}'
    stops[res_col] = stops.apply(
        lambda r: h3.latlng_to_cell(r['lat'], r['lon'], resolution), axis=1
    )

    # H3 centroid 좌표 계산
    centroids = {}
    for cell in stops[res_col].unique():
        clat, clon = h3.cell_to_latlng(cell)
        centroids[cell] = (clat, clon)

    stops[f'centroid_lat_res{resolution}'] = stops[res_col].map(lambda c: centroids[c][0])
    stops[f'centroid_lon_res{resolution}'] = stops[res_col].map(lambda c: centroids[c][1])

    # centroid → 정류장 거리 (미터)
    stops[f'dist_to_centroid_res{resolution}'] = stops.apply(
        lambda r: haversine(
            r['lat'], r['lon'],
            r[f'centroid_lat_res{resolution}'], r[f'centroid_lon_res{resolution}']
        ), axis=1
    )

    return stops


def compute_h3_stop_distances(stops, resolution, use_osrm=True):
    """H3 셀별 centroid → 셀 내 모든 정류장 네트워크 도보거리 테이블"""
    res_col = f'h3_res{resolution}'

    # 먼저 모든 (centroid, stop) 쌍 수집
    pairs = []
    for cell, group in stops.groupby(res_col):
        clat, clon = h3.cell_to_latlng(cell)
        for _, row in group.iterrows():
            pairs.append({
                'h3_cell': cell,
                'h3_resolution': resolution,
                'stop_id': row['stop_id'],
                'stop_lat': row['lat'],
                'stop_lon': row['lon'],
                'centroid_lat': clat,
                'centroid_lon': clon,
            })

    df = pd.DataFrame(pairs)

    if use_osrm:
        # OSRM 서버 확인
        try:
            r = requests.get(f"{OSRM_URL}/route/v1/foot/126.97,37.56;126.98,37.57?overview=false", timeout=3)
            osrm_ok = r.json().get('code') == 'Ok'
        except Exception:
            osrm_ok = False

        if osrm_ok:
            print(f"  OSRM 서버 사용 ({OSRM_URL})")
            origins = list(zip(df['centroid_lat'], df['centroid_lon']))
            destinations = list(zip(df['stop_lat'], df['stop_lon']))

            distances = []
            total = len(origins)
            t0 = time.time()
            for i, ((olat, olon), (dlat, dlon)) in enumerate(zip(origins, destinations)):
                dist = osrm_walk_distance(olat, olon, dlat, dlon)
                distances.append(round(dist, 1))
                if (i + 1) % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (total - i - 1) / rate
                    print(f"    {i+1}/{total} ({rate:.0f}/s, ETA {eta:.0f}s)")

            df['distance_m'] = distances
            elapsed = time.time() - t0
            print(f"  OSRM 조회 완료: {total}건, {elapsed:.0f}초")
        else:
            print(f"  OSRM 서버 미응답, haversine × 1.4 사용")
            df['distance_m'] = df.apply(
                lambda r: round(haversine(r['centroid_lat'], r['centroid_lon'],
                                          r['stop_lat'], r['stop_lon']) * 1.4, 1),
                axis=1
            )
    else:
        df['distance_m'] = df.apply(
            lambda r: round(haversine(r['centroid_lat'], r['centroid_lon'],
                                      r['stop_lat'], r['stop_lon']), 1),
            axis=1
        )

    print(f"  Resolution {resolution}: {df['h3_cell'].nunique()} 셀, "
          f"셀당 평균 {df.groupby('h3_cell').size().mean():.1f}개 정류장, "
          f"평균 거리 {df['distance_m'].mean():.0f}m")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, nargs='+', default=[8],
                        help='H3 resolution (default: 8)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 정류장 좌표 로드
    print("[1/3] 정류장 좌표 로드...")
    stops = load_stop_coords()

    # 2. H3 매핑
    print("[2/3] H3 매핑...")
    for res in args.res:
        stops = build_h3_mapping(stops, res)

    # 저장: 정류장-H3 매핑
    mapping_path = os.path.join(OUTPUT_DIR, 'stop_h3_mapping.csv')
    stops.to_csv(mapping_path, index=False)
    print(f"  저장: {mapping_path}")

    # 3. H3 셀-정류장 거리 테이블
    print("[3/3] H3 셀-정류장 거리 계산...")
    all_distances = []
    for res in args.res:
        dist_df = compute_h3_stop_distances(stops, res)
        all_distances.append(dist_df)

    distances = pd.concat(all_distances, ignore_index=True)
    dist_path = os.path.join(OUTPUT_DIR, 'h3_stop_distance.csv')
    distances.to_csv(dist_path, index=False)
    print(f"  저장: {dist_path}")

    print("\nPhase 1 완료.")


if __name__ == '__main__':
    main()
