# -*- coding: utf-8 -*-
"""
Filtered Training Dataset 조립 (Phase 2)

otp_cache_filtered.db + 기존 route_choice_training.parquet →
개인 SC 통행 단위 학습 데이터 (chosen=0/1)

Strategy:
- 기존 aggregated data에서 OD별 chosen alt 정보 재사용
- Feature fingerprint (ivt, transfers, fare)로 old↔new alt 매칭
- 필터링으로 제거된 alt가 chosen이었으면 해당 OD 제외

Output: data/training_set_new/route_choice_filtered_training.parquet
"""

import os
import pickle
import sqlite3

import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')

FILTERED_CACHE_DB = os.path.join(DATA_DIR, 'training_set_new', 'otp_cache_filtered.db')
OLD_AGG_PARQUET = os.path.join(DATA_DIR, 'training_set', 'route_choice_training.parquet')
OUTPUT_PATH = os.path.join(DATA_DIR, 'training_set_new', 'route_choice_filtered_training.parquet')

# Feature columns for the output
OUTPUT_COLS = [
    'od_pair', 'alt_idx', 'chosen',
    'ivt_bus', 'ivt_train', 'ivt_gtx',
    'waiting_time', 'transfer_walk_time', 'num_transfers', 'fare',
    'transport_category',
    'choice_set_size',
    # extra useful columns
    'total_duration', 'in_vehicle_time', 'walk_time',
    'access_time', 'egress_time', 'walk_distance',
    'total_distance', 'bus_distance', 'subway_distance', 'gtx_distance',
    'has_bus', 'has_train', 'has_gtx',
]


def _fingerprint(feat):
    """Create a matching key from alt features."""
    return (
        round(feat.get('in_vehicle_time', 0), 1),
        int(feat.get('num_transfers', 0)),
        int(feat.get('fare', 0)),
    )


def main():
    # 1. Load old aggregated data (chosen info)
    print('Loading aggregated training data...')
    agg_df = pd.read_parquet(OLD_AGG_PARQUET,
        columns=['od_pair', 'alt_idx', 'in_vehicle_time',
                 'num_transfers', 'fare', 'choice_prob', 'n_total'])

    # Build lookup: od_pair -> {fingerprint: (old_alt_idx, choice_prob, n_total)}
    print('Building chosen lookup...')
    chosen_lookup = {}
    for _, row in tqdm(agg_df.iterrows(), total=len(agg_df), desc='Indexing'):
        od = row['od_pair']
        fp = (round(row['in_vehicle_time'], 1),
              int(row['num_transfers']),
              int(row['fare']))
        if od not in chosen_lookup:
            chosen_lookup[od] = {}
        chosen_lookup[od][fp] = {
            'old_alt_idx': row['alt_idx'],
            'choice_prob': row['choice_prob'],
            'n_total': row['n_total'],
        }
    del agg_df
    print(f'  {len(chosen_lookup):,} ODs indexed')

    # 2. Stream filtered cache and build training rows
    print('\nBuilding training data from filtered cache...')
    conn = sqlite3.connect(FILTERED_CACHE_DB)
    total = conn.execute('SELECT COUNT(*) FROM otp_cache').fetchone()[0]
    cursor = conn.execute('SELECT od_pair, n_alts, data FROM otp_cache')

    rows = []
    stats = {
        'ods_processed': 0,
        'ods_matched': 0,
        'ods_no_agg': 0,
        'ods_chosen_lost': 0,
        'ods_no_chosen': 0,
        'alts_total': 0,
    }

    for od_pair, n_alts, blob in tqdm(cursor, total=total, desc='Building'):
        stats['ods_processed'] += 1
        data = pickle.loads(blob)
        alt_features_list = data['alt_features']

        # Find chosen alt via fingerprint matching
        od_info = chosen_lookup.get(od_pair)
        if od_info is None:
            stats['ods_no_agg'] += 1
            continue

        # Match each new alt to old alt
        chosen_new_idx = None
        max_prob = 0.0
        n_total = 0

        for new_idx, feat in enumerate(alt_features_list):
            fp = _fingerprint(feat)
            match = od_info.get(fp)
            if match and match['choice_prob'] > max_prob:
                max_prob = match['choice_prob']
                chosen_new_idx = new_idx
                n_total = match['n_total']

        if chosen_new_idx is None or max_prob == 0:
            # The chosen alt was filtered out, or no match found
            stats['ods_chosen_lost'] += 1
            continue

        stats['ods_matched'] += 1
        stats['alts_total'] += n_alts

        for new_idx, feat in enumerate(alt_features_list):
            rows.append({
                'od_pair': od_pair,
                'alt_idx': new_idx,
                'chosen': 1 if new_idx == chosen_new_idx else 0,
                'ivt_bus': feat.get('ivt_bus', 0.0),
                'ivt_train': feat.get('ivt_train', 0.0),
                'ivt_gtx': feat.get('ivt_gtx', 0.0),
                'waiting_time': feat.get('wait_time', 0.0),
                'transfer_walk_time': feat.get('transfer_walk_time', 0.0),
                'num_transfers': feat.get('num_transfers', 0),
                'fare': feat.get('fare', 0),
                'transport_category': feat.get('transport_category', ''),
                'choice_set_size': n_alts,
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

    conn.close()

    # 3. Create DataFrame and save
    print('\nCreating DataFrame...')
    df = pd.DataFrame(rows)
    del rows

    # Verify: each OD has exactly one chosen=1
    chosen_per_od = df.groupby('od_pair')['chosen'].sum()
    bad_ods = chosen_per_od[chosen_per_od != 1]
    if len(bad_ods) > 0:
        print(f'  WARNING: {len(bad_ods)} ODs without exactly 1 chosen alt')
        df = df[~df['od_pair'].isin(bad_ods.index)].reset_index(drop=True)

    df.to_parquet(OUTPUT_PATH, index=False)

    # 4. Stats
    print('\n' + '=' * 60)
    print('TRAINING DATA BUILD RESULTS')
    print('=' * 60)
    print(f'Filtered cache ODs:     {stats["ods_processed"]:>10,}')
    print(f'Matched ODs (output):   {stats["ods_matched"]:>10,}')
    print(f'No aggregated data:     {stats["ods_no_agg"]:>10,}')
    print(f'Chosen alt lost:        {stats["ods_chosen_lost"]:>10,}')
    print(f'Total alternatives:     {stats["alts_total"]:>10,}')
    print(f'Output rows:            {len(df):>10,}')
    print(f'Output ODs:             {df["od_pair"].nunique():>10,}')
    print(f'Avg choice set size:    {len(df) / df["od_pair"].nunique():.2f}')
    print(f'\nChoice set size distribution:')
    cs_dist = df.groupby('od_pair')['alt_idx'].count().value_counts().sort_index()
    for size, cnt in cs_dist.items():
        print(f'  size={size}: {cnt:>10,}')

    print(f'\nTransport category distribution:')
    cat_dist = df['transport_category'].value_counts()
    for cat, cnt in cat_dist.items():
        print(f'  {cat:<20s}: {cnt:>10,} ({cnt/len(df)*100:.1f}%)')

    print(f'\nOutput: {OUTPUT_PATH}')
    print('Done.')


if __name__ == '__main__':
    main()
