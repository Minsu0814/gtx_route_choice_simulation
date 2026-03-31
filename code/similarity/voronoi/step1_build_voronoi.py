# -*- coding: utf-8 -*-
"""
Step 1: Voronoi Diagram 생성 + 대표점 계산

정류장 좌표로 Voronoi를 만들고, 각 정류장의 접근권역 centroid(대표점)를 구한다.
대표점 = "이 정류장 이용자들의 평균 출발 위치" 근사.

Usage:
    python step1_build_voronoi.py
"""

import sys; sys.stdout.reconfigure(encoding='utf-8')
import os
import math
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.ops import unary_union

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'data')
EB_DIR = os.path.join(DATA_DIR, 'eb')
OTP_INPUT = os.path.join(DATA_DIR, 'otp', 'input', 'otp_od_input_over13.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'voronoi')


def load_stops():
    """OTP 입력에서 정류장 좌표 추출"""
    df = pd.read_csv(OTP_INPUT)
    o = df[['o_stop_id', 'o_lat', 'o_lon']].rename(columns={'o_stop_id': 'stop_id', 'o_lat': 'lat', 'o_lon': 'lon'})
    d = df[['d_stop_id', 'd_lat', 'd_lon']].rename(columns={'d_stop_id': 'stop_id', 'd_lat': 'lat', 'd_lon': 'lon'})
    stops = pd.concat([o, d]).drop_duplicates('stop_id').reset_index(drop=True)
    stops['stop_id'] = stops['stop_id'].astype(str)
    print(f"정류장: {len(stops)}개")
    return stops


def build_voronoi_centroids(stops):
    """Voronoi diagram 생성 + 각 셀의 centroid 계산

    무한 영역 처리: bounding box로 클리핑
    """
    points = stops[['lon', 'lat']].values  # (lon, lat) 순서
    n = len(points)

    # Bounding box (여유 포함)
    margin = 0.05  # ~5km
    min_lon, max_lon = points[:, 0].min() - margin, points[:, 0].max() + margin
    min_lat, max_lat = points[:, 1].min() - margin, points[:, 1].max() + margin
    bbox = Polygon([
        (min_lon, min_lat), (max_lon, min_lat),
        (max_lon, max_lat), (min_lon, max_lat)
    ])

    # 미러 포인트 추가 (경계 Voronoi 셀 처리)
    mirrored = np.vstack([
        points,
        np.column_stack([2 * min_lon - points[:, 0], points[:, 1]]),
        np.column_stack([2 * max_lon - points[:, 0], points[:, 1]]),
        np.column_stack([points[:, 0], 2 * min_lat - points[:, 1]]),
        np.column_stack([points[:, 0], 2 * max_lat - points[:, 1]]),
    ])

    vor = Voronoi(mirrored)

    centroids = []
    valid = 0
    invalid = 0

    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if not region or -1 in region:
            # 무한 영역 → 정류장 좌표 자체를 대표점으로
            centroids.append({
                'stop_id': stops.iloc[i]['stop_id'],
                'stop_lat': stops.iloc[i]['lat'],
                'stop_lon': stops.iloc[i]['lon'],
                'vor_centroid_lat': stops.iloc[i]['lat'],
                'vor_centroid_lon': stops.iloc[i]['lon'],
                'vor_area_m2': 0,
                'valid': False,
            })
            invalid += 1
            continue

        try:
            vertices = [vor.vertices[v] for v in region]
            poly = Polygon(vertices)

            # bbox로 클리핑
            clipped = poly.intersection(bbox)
            if clipped.is_empty or clipped.area == 0:
                raise ValueError("empty")

            c = clipped.centroid

            # 면적을 m²로 근사 (위도 37도 기준)
            area_deg2 = clipped.area
            m_per_deg_lon = 111320 * math.cos(math.radians(37))
            m_per_deg_lat = 110540
            area_m2 = area_deg2 * m_per_deg_lon * m_per_deg_lat

            centroids.append({
                'stop_id': stops.iloc[i]['stop_id'],
                'stop_lat': stops.iloc[i]['lat'],
                'stop_lon': stops.iloc[i]['lon'],
                'vor_centroid_lat': c.y,
                'vor_centroid_lon': c.x,
                'vor_area_m2': round(area_m2, 0),
                'valid': True,
            })
            valid += 1

        except Exception:
            centroids.append({
                'stop_id': stops.iloc[i]['stop_id'],
                'stop_lat': stops.iloc[i]['lat'],
                'stop_lon': stops.iloc[i]['lon'],
                'vor_centroid_lat': stops.iloc[i]['lat'],
                'vor_centroid_lon': stops.iloc[i]['lon'],
                'vor_area_m2': 0,
                'valid': False,
            })
            invalid += 1

    print(f"Voronoi 셀: valid={valid}, invalid={invalid}")
    return pd.DataFrame(centroids)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def main():
    print("=" * 60)
    print("Step 1: Voronoi Diagram + 대표점 계산")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 정류장 로드
    print("\n[1/3] 정류장 로드...")
    stops = load_stops()

    # 2. Voronoi 생성
    print("\n[2/3] Voronoi diagram 생성...")
    vor_df = build_voronoi_centroids(stops)

    # 3. 대표점-정류장 거리 계산
    print("\n[3/3] 대표점-정류장 거리 계산...")
    vor_df['offset_m'] = vor_df.apply(
        lambda r: round(haversine(r['stop_lat'], r['stop_lon'],
                                   r['vor_centroid_lat'], r['vor_centroid_lon']), 1),
        axis=1
    )

    # 통계
    valid = vor_df[vor_df['valid']]
    print(f"\n  유효 Voronoi 셀: {len(valid)}")
    print(f"  대표점-정류장 offset:")
    print(f"    mean: {valid['offset_m'].mean():.0f}m")
    print(f"    median: {valid['offset_m'].median():.0f}m")
    print(f"    25%ile: {valid['offset_m'].quantile(0.25):.0f}m")
    print(f"    75%ile: {valid['offset_m'].quantile(0.75):.0f}m")
    print(f"  Voronoi 셀 면적:")
    print(f"    mean: {valid['vor_area_m2'].mean():,.0f}m²")
    print(f"    median: {valid['vor_area_m2'].median():,.0f}m²")

    # 저장
    out_path = os.path.join(OUTPUT_DIR, 'voronoi_centroids.csv')
    vor_df.to_csv(out_path, index=False)
    print(f"\n  저장: {out_path}")

    print("\nStep 1 완료.")


if __name__ == '__main__':
    main()
