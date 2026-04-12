"""
Post-process Raptor cache into filtered choice set.

Pipeline:
1. Load raptor_cache.db (774K+ ODs, raw itineraries)
2. Deduplicate by route topology (module/similarity.py)
3. Extract features (module/route_features.py)
4. Parse stop coords for OD stop filtering
5. Filter same origin/dest stops (30m haversine)
6. Add mode-specific IVT (ivt_bus, ivt_train, ivt_gtx)
7. Remove choice_set_size < 2
8. Save filtered cache + parquet

Usage: python build_raptor_choice_set.py [--max-ods N]
"""
import argparse
import csv
import math
import os
import pickle
import sqlite3
import sys
from collections import Counter

import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')
sys.path.insert(0, BASE_DIR)

INPUT_DB = os.path.join(DATA_DIR, 'cache', 'raptor', 'raptor_cache.db')
OTP_INPUT_CSV = os.path.join(DATA_DIR, 'otp', 'input', 'otp_od_input_over13.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'cache', 'raptor')
OUTPUT_DB = os.path.join(OUTPUT_DIR, 'raptor_cache_filtered.db')
OUTPUT_PARQUET = os.path.join(DATA_DIR, 'choice_set', 'raptor', 'raptor_choice_set.parquet')

SAME_STOP_DISTANCE_M = 500  # 500m (Raptor explores diverse access stops)
MIN_CHOICE_SET_SIZE = 2

OTP_MODE_MAP = {'BUS': 'bus', 'SUBWAY': 'train', 'RAIL': 'train', 'TRAM': 'train'}
GTX_ROUTE_KEYWORDS = ['GTX', 'gtx']


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_od_coords():
    od_coords = {}
    with open(OTP_INPUT_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            od_coords[row['od_pair']] = (
                float(row['o_lat']), float(row['o_lon']),
                float(row['d_lat']), float(row['d_lon']),
            )
    return od_coords


def itinerary_to_parsed(itin):
    """
    Convert raw Raptor itinerary to lightweight otp_parsed format.
    Only extracts stop_coords and transit_legs (needed for filtering + IVT).
    """
    legs = itin.get('legs', [])
    stop_coords = []
    transit_legs = []

    for leg in legs:
        mode = leg.get('mode', '')
        if mode == 'WALK':
            continue

        from_obj = leg.get('from', {})
        to_obj = leg.get('to', {})

        if from_obj.get('stop'):
            stop_coords.append((from_obj.get('lat'), from_obj.get('lon')))
        if to_obj.get('stop'):
            stop_coords.append((to_obj.get('lat'), to_obj.get('lon')))

        route_info = leg.get('route') or {}
        route_name = route_info.get('shortName', '') or ''
        is_gtx = any(kw in route_name for kw in GTX_ROUTE_KEYWORDS)
        mapped = 'gtx' if is_gtx else OTP_MODE_MAP.get(mode, 'train')

        transit_legs.append({
            'mode': mode,
            'mapped_mode': mapped,
            'route_name': route_name,
            'duration': leg.get('duration', 0),
            'distance': leg.get('distance', 0),
        })

    # Deduplicate consecutive identical coords
    unique = []
    for c in stop_coords:
        if not unique or c != unique[-1]:
            unique.append(c)

    return {'stop_coords': unique, 'transit_legs': transit_legs}


def _filter_by_stop(parsed_list, indices, sc_lat, sc_lon, stop_idx):
    alt_info = []
    for i in indices:
        coords = parsed_list[i].get('stop_coords', [])
        if coords:
            coord = coords[stop_idx]
            if coord and coord[0] is not None:
                dist = _haversine(sc_lat, sc_lon, coord[0], coord[1])
                alt_info.append((i, coord, dist))
                continue
        alt_info.append((i, None, float('inf')))

    valid = [(idx, coord, dist)
             for idx, coord, dist in alt_info if coord is not None]
    if not valid:
        return []

    ref_coord = min(valid, key=lambda v: v[2])[1]
    return [idx for idx, coord, _ in valid
            if _haversine(ref_coord[0], ref_coord[1],
                          coord[0], coord[1]) <= SAME_STOP_DISTANCE_M]


def filter_same_od_stops(parsed_list, features_list, o_lat, o_lon, d_lat, d_lon):
    if not parsed_list:
        return [], [], []
    all_idx = list(range(len(parsed_list)))
    origin_ok = _filter_by_stop(parsed_list, all_idx, o_lat, o_lon, 0)
    both_ok = _filter_by_stop(parsed_list, origin_ok, d_lat, d_lon, -1)
    return (
        [parsed_list[i] for i in both_ok],
        [features_list[i] for i in both_ok],
        both_ok,
    )


def extract_mode_ivt(parsed):
    ivt = {'bus': 0.0, 'train': 0.0, 'gtx': 0.0}
    for leg in parsed.get('transit_legs', []):
        mapped = leg.get('mapped_mode', 'train')
        dur = float(leg.get('duration', 0) or 0)
        if mapped in ivt:
            ivt[mapped] += dur
    return ivt['bus'], ivt['train'], ivt['gtx']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-ods', type=int, default=None)
    args = parser.parse_args()

    from module.similarity import deduplicate_itineraries
    from module.route_features import extract_itinerary_features

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load OD coords
    print('Loading OD coordinates...')
    od_coords = load_od_coords()
    print(f'  {len(od_coords):,} ODs')

    # 2. Open input DB
    conn_in = sqlite3.connect(INPUT_DB)
    total = conn_in.execute('SELECT COUNT(*) FROM raptor_cache').fetchone()[0]
    print(f'Input: {total:,} ODs in raptor_cache.db')

    if args.max_ods:
        total = min(total, args.max_ods)
        print(f'  (limited to {total:,})')

    # 3. Init output DB
    if os.path.exists(OUTPUT_DB):
        os.remove(OUTPUT_DB)
    conn_out = sqlite3.connect(OUTPUT_DB)
    conn_out.execute('PRAGMA journal_mode=WAL')
    conn_out.execute('''CREATE TABLE raptor_cache (
        od_pair TEXT PRIMARY KEY,
        n_alts INTEGER,
        data BLOB
    )''')

    # 4. Process
    query = 'SELECT od_pair, n_alts, data FROM raptor_cache'
    if args.max_ods:
        query += f' LIMIT {args.max_ods}'
    cursor = conn_in.execute(query)

    BATCH_SIZE = 2000
    db_batch = []
    rows = []

    stats = {
        'ods_in': 0, 'alts_in': 0,
        'ods_no_coords': 0, 'ods_filtered_out': 0,
        'ods_kept': 0, 'alts_kept': 0, 'alts_removed': 0,
        'cs_before': Counter(), 'cs_after': Counter(),
        'category': Counter(),
    }

    for od_pair, n_alts_in, blob in tqdm(cursor, total=total, desc='Processing'):
        stats['ods_in'] += 1
        stats['alts_in'] += n_alts_in
        stats['cs_before'][min(n_alts_in, 15)] += 1

        data = pickle.loads(blob)
        itineraries = data.get('itineraries', [])
        if not itineraries:
            stats['ods_filtered_out'] += 1
            continue

        # Deduplicate
        deduped = deduplicate_itineraries(itineraries)
        if not deduped:
            stats['ods_filtered_out'] += 1
            continue

        # Extract features + parse stop coords
        feat_list = []
        parsed_list = []
        for itin in deduped:
            feat_list.append(extract_itinerary_features(itin))
            parsed_list.append(itinerary_to_parsed(itin))

        # OD coords
        coords = od_coords.get(od_pair)
        if coords is None:
            stats['ods_no_coords'] += 1
            stats['ods_filtered_out'] += 1
            continue

        o_lat, o_lon, d_lat, d_lon = coords

        # Filter same origin/dest stops
        f_parsed, f_feats, f_idx = filter_same_od_stops(
            parsed_list, feat_list, o_lat, o_lon, d_lat, d_lon)

        # Add mode-specific IVT
        updated_feats = []
        for parsed, feat in zip(f_parsed, f_feats):
            ivt_b, ivt_t, ivt_g = extract_mode_ivt(parsed)
            feat = dict(feat)
            feat['ivt_bus'] = ivt_b
            feat['ivt_train'] = ivt_t
            feat['ivt_gtx'] = ivt_g
            updated_feats.append(feat)

        n_alts_out = len(updated_feats)
        stats['alts_removed'] += (len(deduped) - n_alts_out)

        if n_alts_out < MIN_CHOICE_SET_SIZE:
            stats['ods_filtered_out'] += 1
            continue

        stats['ods_kept'] += 1
        stats['alts_kept'] += n_alts_out
        stats['cs_after'][min(n_alts_out, 15)] += 1

        # Save to DB cache
        filtered_itins = [deduped[i] for i in f_idx]
        cache_entry = {
            'alt_features': updated_feats,
            'otp_parsed': f_parsed,
            'itineraries': filtered_itins,
        }
        db_batch.append((
            od_pair, n_alts_out,
            pickle.dumps(cache_entry, protocol=pickle.HIGHEST_PROTOCOL),
        ))

        if len(db_batch) >= BATCH_SIZE:
            conn_out.executemany(
                'INSERT INTO raptor_cache VALUES (?,?,?)', db_batch)
            conn_out.commit()
            db_batch = []

        # Parquet rows
        for alt_idx, feat in enumerate(updated_feats):
            stats['category'][feat.get('transport_category', '?')] += 1
            rows.append({
                'od_pair': od_pair,
                'alt_idx': alt_idx,
                'choice_set_size': n_alts_out,
                'ivt_bus': feat.get('ivt_bus', 0.0),
                'ivt_train': feat.get('ivt_train', 0.0),
                'ivt_gtx': feat.get('ivt_gtx', 0.0),
                'waiting_time': feat.get('wait_time', 0.0),
                'transfer_walk_time': feat.get('transfer_walk_time', 0.0),
                'num_transfers': feat.get('num_transfers', 0),
                'fare': feat.get('fare', 0),
                'transport_category': feat.get('transport_category', ''),
                'total_duration': feat.get('total_duration', 0.0),
                'in_vehicle_time': feat.get('in_vehicle_time', 0.0),
                'walk_time': feat.get('walk_time', 0.0),
                'access_time': feat.get('access_time', 0.0),
                'egress_time': feat.get('egress_time', 0.0),
                'walk_distance': feat.get('walk_distance', 0.0),
                'total_distance': feat.get('total_distance', 0.0),
                'bus_distance': feat.get('bus_distance', 0.0),
                'subway_distance': feat.get('subway_distance', 0.0),
                'gtx_distance': feat.get('gtx_distance', 0.0),
                'has_bus': feat.get('has_bus', 0),
                'has_train': feat.get('has_train', 0),
                'has_gtx': feat.get('has_gtx', 0),
            })

    # Final DB commit
    if db_batch:
        conn_out.executemany(
            'INSERT INTO raptor_cache VALUES (?,?,?)', db_batch)
        conn_out.commit()

    conn_in.close()
    conn_out.close()

    # 5. Save parquet
    print('\nBuilding parquet...')
    df = pd.DataFrame(rows)
    del rows
    df.to_parquet(OUTPUT_PARQUET, index=False)

    # 6. Stats
    print('\n' + '=' * 60)
    print('RAPTOR CHOICE SET RESULTS')
    print('=' * 60)
    print(f'Input:  {stats["ods_in"]:>10,} ODs, {stats["alts_in"]:>10,} alts')
    print(f'Output: {stats["ods_kept"]:>10,} ODs, {stats["alts_kept"]:>10,} alts')
    print(f'Removed ODs:  {stats["ods_filtered_out"]:>10,}')
    print(f'Removed alts: {stats["alts_removed"]:>10,}')
    if stats['ods_no_coords']:
        print(f'No coords:    {stats["ods_no_coords"]:>10,}')

    keep_rate = stats['ods_kept'] / max(1, stats['ods_in']) * 100
    print(f'OD retention: {keep_rate:.1f}%')
    print(f'Avg alts/OD:  {stats["alts_kept"] / max(1, stats["ods_kept"]):.1f}')

    print('\n--- Choice Set Size (before dedup+filter → after) ---')
    for size in sorted(set(list(stats['cs_before']) + list(stats['cs_after']))):
        b = stats['cs_before'].get(size, 0)
        a = stats['cs_after'].get(size, 0)
        print(f'  size={size:>2d}: {b:>10,} → {a:>10,}')

    print('\n--- Transport Category ---')
    for cat, cnt in stats['category'].most_common():
        print(f'  {cat:<20s}: {cnt:>10,} '
              f'({cnt / max(1, stats["alts_kept"]) * 100:.1f}%)')

    print(f'\nParquet: {OUTPUT_PARQUET}')
    print(f'  {len(df):,} rows, {df["od_pair"].nunique():,} ODs')
    print(f'Cache:   {OUTPUT_DB}')
    db_size = os.path.getsize(OUTPUT_DB) / 1024 / 1024
    print(f'  {db_size:.0f} MB')


if __name__ == '__main__':
    main()
