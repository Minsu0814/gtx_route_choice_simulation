# -*- coding: utf-8 -*-
"""
물리적으로 같은 위치의 정류장을 하나의 접근점으로 그룹화

GTX역과 기존 지하철역이 같은 건물에 있는 경우 등,
일정 거리(기본 100m) 이내의 정류장을 하나의 접근점으로 묶는다.

EB 배분 시 접근점 단위로 배분하고,
접근점 내 경로선택은 경로선택모형이 처리한다.

Usage:
    python group_access_points.py                # 기본 (100m)
    python group_access_points.py --threshold 50 # 50m 기준
"""

import argparse
import math
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'data')
EB_DIR = os.path.join(DATA_DIR, 'eb')


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def group_stops_by_proximity(stops_df, threshold_m=100):
    """근접 정류장을 Union-Find로 그룹화

    Args:
        stops_df: DataFrame with stop_id, stop_lat, stop_lon
        threshold_m: 그룹화 기준 거리 (미터)

    Returns:
        dict: {stop_id: access_point_id}
    """
    stops = stops_df[['stop_id', 'stop_lat', 'stop_lon']].drop_duplicates('stop_id')
    stop_ids = stops['stop_id'].astype(str).values
    lats = stops['stop_lat'].values
    lons = stops['stop_lon'].values
    n = len(stop_ids)

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # 같은 H3 셀 내에서만 비교 (전체 비교는 O(n²)이라 느림)
    # H3 매핑 로드
    mapping_path = os.path.join(EB_DIR, 'stop_h3_mapping.csv')
    if os.path.exists(mapping_path):
        mapping = pd.read_csv(mapping_path)
        stop_to_h3 = dict(zip(mapping['stop_id'].astype(str), mapping['h3_res8']))
    else:
        stop_to_h3 = {}

    # stop_id → index
    sid_to_idx = {sid: i for i, sid in enumerate(stop_ids)}

    # H3 셀별로 그룹화하여 비교
    h3_groups = {}
    for sid in stop_ids:
        h3 = stop_to_h3.get(sid, 'unknown')
        h3_groups.setdefault(h3, []).append(sid)

    n_unions = 0
    for h3_cell, sids in h3_groups.items():
        if len(sids) < 2:
            continue
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                idx_i = sid_to_idx[sids[i]]
                idx_j = sid_to_idx[sids[j]]
                d = haversine(lats[idx_i], lons[idx_i], lats[idx_j], lons[idx_j])
                if d <= threshold_m:
                    union(idx_i, idx_j)
                    n_unions += 1

    # 그룹 ID 생성: 그룹의 대표 stop_id를 access_point_id로 사용
    stop_to_ap = {}
    groups = {}
    for i, sid in enumerate(stop_ids):
        root = find(i)
        root_sid = stop_ids[root]
        stop_to_ap[sid] = root_sid
        groups.setdefault(root_sid, []).append(sid)

    n_groups = len(groups)
    n_merged = sum(1 for g in groups.values() if len(g) > 1)
    n_stops_merged = sum(len(g) for g in groups.values() if len(g) > 1)

    print(f"  그룹화 기준: {threshold_m}m")
    print(f"  총 정류장: {n}")
    print(f"  접근점(그룹) 수: {n_groups} (단독: {n_groups - n_merged}, 병합: {n_merged})")
    print(f"  병합된 정류장 수: {n_stops_merged}")

    return stop_to_ap, groups


def compute_access_point_distances(groups, dist_df):
    """접근점별 centroid 거리 = 그룹 내 정류장 거리의 최소값

    같은 접근점 내 여러 정류장 중 centroid에서 가장 가까운 거리를 사용.
    사람이 접근점으로 갈 때 가장 가까운 입구를 이용한다는 가정.
    """
    rows = []
    for ap_id, stop_ids in groups.items():
        # 이 그룹에 속한 정류장들의 H3 셀별 거리
        group_dist = dist_df[dist_df['stop_id'].astype(str).isin(stop_ids)]

        for cell, cell_group in group_dist.groupby('h3_cell'):
            min_dist = cell_group['distance_m'].min()
            # 대표 좌표 = 가장 가까운 정류장의 좌표
            closest = cell_group.loc[cell_group['distance_m'].idxmin()]
            rows.append({
                'h3_cell': cell,
                'access_point_id': ap_id,
                'stop_ids': ','.join(stop_ids),
                'n_stops': len(stop_ids),
                'distance_m': min_dist,
                'ap_lat': closest['stop_lat'],
                'ap_lon': closest['stop_lon'],
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=100,
                        help='그룹화 기준 거리 (미터, default: 100)')
    args = parser.parse_args()

    # 1. 정류장 거리 데이터 로드
    print("[1/3] 데이터 로드...")
    dist_path = os.path.join(EB_DIR, 'h3_stop_distance.csv')
    dist_df = pd.read_csv(dist_path)
    dist_df = dist_df[dist_df['h3_resolution'] == 8]

    stops_df = dist_df[['stop_id', 'stop_lat', 'stop_lon']].drop_duplicates('stop_id')
    print(f"  정류장: {len(stops_df)}개")

    # 2. 근접 정류장 그룹화
    print("[2/3] 근접 정류장 그룹화...")
    stop_to_ap, groups = group_stops_by_proximity(stops_df, args.threshold)

    # 예시 출력
    merged_groups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"\n  병합 그룹 예시 (상위 5개):")
    for i, (ap_id, sids) in enumerate(sorted(merged_groups.items(), key=lambda x: -len(x[1]))[:5]):
        print(f"    접근점 {ap_id}: {sids}")

    # 3. 접근점별 거리 재계산
    print("\n[3/3] 접근점별 centroid 거리 계산...")
    ap_dist = compute_access_point_distances(groups, dist_df)

    print(f"  접근점-셀 쌍: {len(ap_dist):,}")
    print(f"  접근점 수: {ap_dist['access_point_id'].nunique():,}")
    print(f"  평균 거리: {ap_dist['distance_m'].mean():.0f}m")

    # 저장
    mapping_path = os.path.join(EB_DIR, 'stop_to_access_point.csv')
    stop_ap_df = pd.DataFrame([
        {'stop_id': sid, 'access_point_id': ap_id}
        for sid, ap_id in stop_to_ap.items()
    ])
    stop_ap_df.to_csv(mapping_path, index=False)
    print(f"\n  저장: {mapping_path}")

    ap_dist_path = os.path.join(EB_DIR, 'access_point_distance.csv')
    ap_dist.to_csv(ap_dist_path, index=False)
    print(f"  저장: {ap_dist_path}")

    # 기존 posterior를 접근점 단위로 재계산
    posterior_path = os.path.join(EB_DIR, 'eb_posterior.csv')
    if os.path.exists(posterior_path):
        print("\n  접근점 단위 posterior 재집계...")
        post_df = pd.read_csv(posterior_path)
        post_df['stop_id'] = post_df['stop_id'].astype(str)
        post_df['access_point_id'] = post_df['stop_id'].map(stop_to_ap)

        # 접근점 단위로 posterior 합산
        ap_post = post_df.groupby(['h3_cell', 'access_point_id']).agg({
            'sc_count': 'sum',
            'prior': 'sum',
            'posterior': 'sum',
            'distance_m': 'min',  # 가장 가까운 정류장 거리
        }).reset_index()

        # 정규화
        for cell, group in ap_post.groupby('h3_cell'):
            total = group['posterior'].sum()
            if total > 0:
                ap_post.loc[group.index, 'posterior'] = group['posterior'] / total

        ap_post_path = os.path.join(EB_DIR, 'eb_posterior_access_point.csv')
        ap_post.to_csv(ap_post_path, index=False)
        print(f"  저장: {ap_post_path}")
        print(f"  접근점 수: {ap_post['access_point_id'].nunique():,} (정류장 {post_df['stop_id'].nunique():,}에서 축소)")

    print("\n완료.")


if __name__ == '__main__':
    main()
