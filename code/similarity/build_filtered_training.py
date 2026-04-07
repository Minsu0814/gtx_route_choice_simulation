# -*- coding: utf-8 -*-
"""
Filtered Training Dataset 구축 (Model Specification Phase 1)

기존 OTP 캐시에서:
1. 같은 출발정류장 대안만 필터링 (SC 승차정류장 기준)
2. 수단별 IVT 분리 (ivt_bus, ivt_train, ivt_gtx)
3. choice_set_size < 2 OD 제거
4. 새 SQLite 캐시 저장

Usage:
    python build_filtered_training.py
    python build_filtered_training.py --max-ods 1000   # 테스트
"""

import argparse
import csv
import math
import os
import pickle
import sqlite3
import sys
from collections import Counter, defaultdict

from tqdm import tqdm

# ============================================================
# 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')
INPUT_CACHE_DB = os.path.join(DATA_DIR, 'training_set', 'otp_cache.db')
OTP_INPUT_CSV = os.path.join(DATA_DIR, 'otp', 'input', 'otp_od_input_over13.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'training_set_new')
OUTPUT_CACHE_DB = os.path.join(OUTPUT_DIR, 'otp_cache_filtered.db')

# 같은 정류장 판정 거리 임계값 (미터)
SAME_STOP_DISTANCE_M = 30

# 최소 choice set 크기
MIN_CHOICE_SET_SIZE = 2

# OTP mode -> internal mode 매핑 (similarity.py의 OTP_MODE_MAP 재현)
OTP_MODE_MAP = {
    'BUS': 'bus',
    'SUBWAY': 'train',
    'RAIL': 'train',
    'TRAM': 'train',
}

GTX_ROUTE_KEYWORDS = ['GTX', 'gtx']


def _haversine(lat1, lon1, lat2, lon2):
    """두 좌표 간 거리 (미터)"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _get_mapped_mode(transit_leg):
    """transit_leg dict -> internal mode (bus/train/gtx)"""
    mode = transit_leg.get('mode', '')
    route_name = transit_leg.get('route_name', '')
    if any(kw in route_name for kw in GTX_ROUTE_KEYWORDS):
        return 'gtx'
    return OTP_MODE_MAP.get(mode, 'train')


def _extract_mode_specific_ivt(otp_parsed):
    """
    otp_parsed (single itinerary) -> (ivt_bus, ivt_train, ivt_gtx) in seconds.
    Uses the transit_legs field which has mode, duration for each transit leg.
    """
    ivt = {'bus': 0.0, 'train': 0.0, 'gtx': 0.0}
    for leg in otp_parsed.get('transit_legs', []):
        mapped = leg.get('mapped_mode') or _get_mapped_mode(leg)
        duration = float(leg.get('duration', 0) or 0)
        if mapped in ivt:
            ivt[mapped] += duration
    return ivt['bus'], ivt['train'], ivt['gtx']


def load_od_coords():
    """OTP input CSV -> {od_pair: (o_lat, o_lon, d_lat, d_lon)}"""
    od_coords = {}
    with open(OTP_INPUT_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            od_coords[row['od_pair']] = (
                float(row['o_lat']),
                float(row['o_lon']),
                float(row['d_lat']),
                float(row['d_lon']),
            )
    return od_coords


def _get_first_leg_mode(otp_parsed):
    """대안의 첫 번째 transit leg 수단 (bus/train/gtx) 반환."""
    for leg in otp_parsed.get('transit_legs', []):
        return _get_mapped_mode(leg)
    return None


def _filter_by_stop(otp_parsed_list, indices, sc_lat, sc_lon, stop_idx):
    """
    대안의 stop_coords[stop_idx] 기준으로 같은 정류장(30m) 필터링.
    stop_idx=0: 출발정류장, stop_idx=-1: 도착정류장.
    Returns: filtered indices.
    """
    alt_info = []
    for i in indices:
        parsed = otp_parsed_list[i]
        stop_coords = parsed.get('stop_coords', [])
        if stop_coords:
            coord = stop_coords[stop_idx]
            if coord and coord[0] is not None:
                dist = _haversine(sc_lat, sc_lon, coord[0], coord[1])
                alt_info.append((i, coord, dist))
                continue
        alt_info.append((i, None, float('inf')))

    valid = [(idx, coord, dist)
             for idx, coord, dist in alt_info
             if coord is not None]
    if not valid:
        return []

    closest = min(valid, key=lambda v: v[2])
    ref_coord = closest[1]

    return [idx for idx, coord, _ in valid
            if _haversine(ref_coord[0], ref_coord[1],
                          coord[0], coord[1]) <= SAME_STOP_DISTANCE_M]


def filter_same_od_stops(otp_parsed_list, alt_features_list,
                         o_lat, o_lon, d_lat, d_lon):
    """
    같은 출발정류장 AND 같은 도착정류장 대안만 유지.

    1) 출발: stop_coords[0] 기준 30m 이내
    2) 도착: stop_coords[-1] 기준 30m 이내
    """
    if not otp_parsed_list:
        return [], []

    all_indices = list(range(len(otp_parsed_list)))

    # 출발정류장 필터링
    origin_ok = _filter_by_stop(
        otp_parsed_list, all_indices, o_lat, o_lon, 0)

    # 도착정류장 필터링 (출발 통과한 것만)
    both_ok = _filter_by_stop(
        otp_parsed_list, origin_ok, d_lat, d_lon, -1)

    filtered_parsed = [otp_parsed_list[i] for i in both_ok]
    filtered_features = [alt_features_list[i] for i in both_ok]
    return filtered_parsed, filtered_features


def add_mode_specific_ivt(alt_features, otp_parsed):
    """
    alt_features dict에 ivt_bus, ivt_train, ivt_gtx 추가.
    기존 in_vehicle_time은 유지 (참고용).
    """
    ivt_bus, ivt_train, ivt_gtx = _extract_mode_specific_ivt(otp_parsed)
    alt_features = dict(alt_features)  # copy
    alt_features['ivt_bus'] = ivt_bus
    alt_features['ivt_train'] = ivt_train
    alt_features['ivt_gtx'] = ivt_gtx
    return alt_features


def main():
    parser = argparse.ArgumentParser(description='Build filtered training set')
    parser.add_argument('--max-ods', type=int, default=None, help='Max ODs to process (for testing)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. OTP input 좌표 로드
    print('Loading OTP input coordinates...')
    od_coords = load_od_coords()
    print(f'  {len(od_coords):,} OD pairs with coordinates')

    # 2. 입력 캐시 열기
    conn_in = sqlite3.connect(INPUT_CACHE_DB)
    total_ods = conn_in.execute('SELECT COUNT(*) FROM otp_cache').fetchone()[0]
    print(f'Input cache: {total_ods:,} ODs')

    if args.max_ods:
        total_ods = min(total_ods, args.max_ods)
        print(f'  (limited to {total_ods:,} ODs)')

    # 3. 출력 DB 초기화
    if os.path.exists(OUTPUT_CACHE_DB):
        os.remove(OUTPUT_CACHE_DB)
    conn_out = sqlite3.connect(OUTPUT_CACHE_DB)
    conn_out.execute('PRAGMA journal_mode=WAL')
    conn_out.execute('''CREATE TABLE otp_cache (
        od_pair TEXT PRIMARY KEY,
        n_alts  INTEGER,
        data    BLOB
    )''')

    # 4. 필터링 + IVT 분리
    query = 'SELECT od_pair, n_alts, data FROM otp_cache'
    if args.max_ods:
        query += f' LIMIT {args.max_ods}'

    cursor = conn_in.execute(query)
    BATCH_SIZE = 2000
    batch = []

    # 통계
    stats = {
        'total_ods_in': 0,
        'total_alts_in': 0,
        'ods_no_coords': 0,
        'ods_filtered_out': 0,     # choice_set < 2 after filtering
        'ods_kept': 0,
        'alts_kept': 0,
        'alts_removed': 0,
        'cs_size_before': Counter(),  # choice set size distribution before
        'cs_size_after': Counter(),   # choice set size distribution after
        'category_dist': Counter(),   # transport_category distribution
        'first_mode_dist': Counter(),  # 첫 수단 분포 (bus/train/gtx)
    }

    for row in tqdm(cursor, desc='Filtering', total=total_ods):
        od_pair, n_alts_in, blob = row
        stats['total_ods_in'] += 1
        stats['total_alts_in'] += n_alts_in
        stats['cs_size_before'][n_alts_in] += 1

        data = pickle.loads(blob)
        otp_parsed_list = data['otp_parsed']
        alt_features_list = data['alt_features']

        # 좌표 확인
        coords = od_coords.get(od_pair)
        if coords is None:
            stats['ods_no_coords'] += 1
            # 좌표 없으면 otp_parsed의 od_coords 시도
            if otp_parsed_list and otp_parsed_list[0].get('od_coords'):
                oc = otp_parsed_list[0]['od_coords']
                coords = (oc.get('o_lat', 0), oc.get('o_lon', 0),
                          oc.get('d_lat', 0), oc.get('d_lon', 0))
            else:
                stats['ods_filtered_out'] += 1
                continue

        o_lat, o_lon, d_lat, d_lon = coords

        # 같은 출발+도착 정류장 필터링
        filtered_parsed, filtered_features = filter_same_od_stops(
            otp_parsed_list, alt_features_list, o_lat, o_lon, d_lat, d_lon
        )

        n_alts_out = len(filtered_parsed)
        stats['alts_removed'] += (n_alts_in - n_alts_out)

        # choice set size < 2 → 제외
        if n_alts_out < MIN_CHOICE_SET_SIZE:
            stats['ods_filtered_out'] += 1
            continue

        # 첫 수단 통계
        first_mode = _get_first_leg_mode(filtered_parsed[0]) if filtered_parsed else 'unknown'
        stats['first_mode_dist'][first_mode] += 1

        # IVT 분리 추가
        updated_features = []
        for feat, parsed in zip(filtered_features, filtered_parsed):
            updated_features.append(add_mode_specific_ivt(feat, parsed))
            cat = feat.get('transport_category', 'unknown')
            stats['category_dist'][cat] += 1

        # 캐시 엔트리 구축
        cache_entry = {
            'alt_features': updated_features,
            'otp_parsed': filtered_parsed,
        }
        batch.append((
            od_pair,
            n_alts_out,
            pickle.dumps(cache_entry, protocol=pickle.HIGHEST_PROTOCOL),
        ))

        stats['ods_kept'] += 1
        stats['alts_kept'] += n_alts_out
        stats['cs_size_after'][n_alts_out] += 1

        if len(batch) >= BATCH_SIZE:
            conn_out.executemany('INSERT INTO otp_cache VALUES (?,?,?)', batch)
            conn_out.commit()
            batch = []

    if batch:
        conn_out.executemany('INSERT INTO otp_cache VALUES (?,?,?)', batch)
        conn_out.commit()

    conn_in.close()
    conn_out.close()

    # 5. 통계 출력
    print('\n' + '=' * 60)
    print('FILTERING RESULTS')
    print('=' * 60)
    print(f'Input:  {stats["total_ods_in"]:>10,} ODs, {stats["total_alts_in"]:>10,} alternatives')
    print(f'Output: {stats["ods_kept"]:>10,} ODs, {stats["alts_kept"]:>10,} alternatives')
    print(f'Removed ODs:  {stats["ods_filtered_out"]:>10,} (choice_set < {MIN_CHOICE_SET_SIZE} or no coords)')
    print(f'Removed alts: {stats["alts_removed"]:>10,} (different origin stop)')
    if stats['ods_no_coords'] > 0:
        print(f'No coords:    {stats["ods_no_coords"]:>10,}')

    print(f'\nRetention rate: {stats["ods_kept"] / max(1, stats["total_ods_in"]) * 100:.1f}% ODs, '
          f'{stats["alts_kept"] / max(1, stats["total_alts_in"]) * 100:.1f}% alternatives')

    print('\n--- Choice Set Size Distribution (BEFORE filtering) ---')
    for size in sorted(stats['cs_size_before']):
        cnt = stats['cs_size_before'][size]
        print(f'  size={size}: {cnt:>10,} ({cnt / stats["total_ods_in"] * 100:.1f}%)')

    print('\n--- Choice Set Size Distribution (AFTER filtering) ---')
    for size in sorted(stats['cs_size_after']):
        cnt = stats['cs_size_after'][size]
        print(f'  size={size}: {cnt:>10,} ({cnt / stats["ods_kept"] * 100:.1f}%)')

    print('\n--- First Boarding Mode Distribution (ODs) ---')
    for mode, cnt in stats['first_mode_dist'].most_common():
        print(f'  {mode:<10s}: {cnt:>10,} ({cnt / max(1, stats["ods_kept"]) * 100:.1f}%)')

    print('\n--- Transport Category Distribution (filtered alternatives) ---')
    for cat, cnt in stats['category_dist'].most_common():
        print(f'  {cat:<20s}: {cnt:>10,} ({cnt / max(1, stats["alts_kept"]) * 100:.1f}%)')

    # 6. IVT 분포 확인 (샘플)
    print('\n--- Mode-specific IVT sample check ---')
    conn_check = sqlite3.connect(OUTPUT_CACHE_DB)
    sample_rows = conn_check.execute('SELECT od_pair, data FROM otp_cache LIMIT 5').fetchall()
    for od_pair, blob in sample_rows:
        data = pickle.loads(blob)
        for i, feat in enumerate(data['alt_features']):
            ivt_b = feat.get('ivt_bus', 0)
            ivt_t = feat.get('ivt_train', 0)
            ivt_g = feat.get('ivt_gtx', 0)
            cat = feat.get('transport_category', '?')
            print(f'  {od_pair} alt[{i}]: ivt_bus={ivt_b:.0f}s ivt_train={ivt_t:.0f}s ivt_gtx={ivt_g:.0f}s ({cat})')
    conn_check.close()

    print(f'\nOutput saved: {OUTPUT_CACHE_DB}')
    print('Done.')


if __name__ == '__main__':
    main()
