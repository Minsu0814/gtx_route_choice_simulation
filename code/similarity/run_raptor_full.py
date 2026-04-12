"""
Run calibrated Raptor on all ODs and save results to SQLite cache.

Input:  data/otp/input/otp_od_input_over13.csv (774,895 ODs)
Output: data/cache/raptor/raptor_cache.db (SQLite, same schema as otp_cache.db)

Params: t600, w10, b0.7, tr1.5, wt1.5

Usage: python run_raptor_full.py
"""
import os
import pickle
import sqlite3
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')

GTFS_DIR = os.path.join(DATA_DIR, 'gtfs', 'a1')
OD_INPUT = os.path.join(DATA_DIR, 'otp', 'input', 'otp_od_input_over13.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'cache', 'raptor')
OUTPUT_DB = os.path.join(OUTPUT_DIR, 'raptor_cache.db')

RAPTOR_CONFIG = {
    'transfer_cost_secs': 600,
    'walk_reluctance': 10.0,
    'wait_reluctance': 1.5,
    'transit_reluctance': [0.7, 1.5, 1.5, 1.5, 1.5, 1.5],
    'max_results': 15,
}

BATCH_SIZE = 5000  # ODs per batch for progress tracking
COMMIT_EVERY = 10000  # commit to DB every N ODs


def parse_dep_secs(dep_time_series):
    """Parse departure_time (yyyyMMddHHmmss) to seconds since midnight."""
    dt_str = dep_time_series.astype(str).str.slice(8, 14)
    return (dt_str.str[:2].astype(int) * 3600
            + dt_str.str[2:4].astype(int) * 60
            + dt_str.str[4:6].astype(int))


def main():
    print('=== Raptor Full Routing (Calibrated Params) ===\n')
    print(f'Config: {RAPTOR_CONFIG}\n')

    # 1. Load OD input
    print('Loading OD input...')
    df = pd.read_csv(OD_INPUT)
    df['dep_secs'] = parse_dep_secs(df['departure_time'])
    print(f'  {len(df):,} ODs loaded')

    # 2. Initialize Raptor
    print('\nLoading Raptor...')
    from dtumos_raptor import DtumosRaptor
    raptor = DtumosRaptor(num_threads=8)
    raptor.load_gtfs(GTFS_DIR)
    raptor.update_config(RAPTOR_CONFIG)
    print('  Raptor ready')

    # 3. Initialize output DB
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_DB):
        # Resume: check existing ODs
        conn = sqlite3.connect(OUTPUT_DB)
        existing = set(
            r[0] for r in conn.execute(
                'SELECT od_pair FROM raptor_cache').fetchall()
        )
        print(f'\n  Resuming: {len(existing):,} ODs already cached')
        df = df[~df['od_pair'].isin(existing)].reset_index(drop=True)
        print(f'  Remaining: {len(df):,} ODs')
    else:
        conn = sqlite3.connect(OUTPUT_DB)
        conn.execute('''CREATE TABLE raptor_cache (
            od_pair TEXT PRIMARY KEY,
            n_alts INTEGER,
            data BLOB
        )''')
        conn.commit()

    if len(df) == 0:
        print('\nAll ODs already cached. Done!')
        conn.close()
        return

    # 4. Route all ODs
    print(f'\nRouting {len(df):,} ODs...')
    t0 = time.time()
    n_success = 0
    n_empty = 0
    batch = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Routing'):
        od_pair = row['od_pair']
        try:
            result = raptor.route(
                row['o_lat'], row['o_lon'],
                row['d_lat'], row['d_lon'],
                int(row['dep_secs']), mode='mc',
            )
            itineraries = (result.get('data', {}).get('plan', {})
                           .get('itineraries', []))
        except Exception:
            itineraries = []

        if not itineraries:
            n_empty += 1
            continue

        # Store as pickle blob (same format as otp_cache.db)
        data = {'itineraries': itineraries}
        blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        batch.append((od_pair, len(itineraries), blob))
        n_success += 1

        if len(batch) >= COMMIT_EVERY:
            conn.executemany(
                'INSERT OR REPLACE INTO raptor_cache VALUES (?,?,?)',
                batch)
            conn.commit()
            batch = []

    # Final commit
    if batch:
        conn.executemany(
            'INSERT OR REPLACE INTO raptor_cache VALUES (?,?,?)',
            batch)
        conn.commit()

    elapsed = time.time() - t0
    conn.close()

    # 5. Summary
    print(f'\n=== Done in {elapsed:.0f}s ({elapsed/60:.1f}min) ===')
    print(f'  Success (with itineraries): {n_success:,}')
    print(f'  Empty (no route found):     {n_empty:,}')
    print(f'  Speed: {(n_success + n_empty) / elapsed:.0f} ODs/sec')
    print(f'  Output: {OUTPUT_DB}')
    db_size = os.path.getsize(OUTPUT_DB) / 1024 / 1024
    print(f'  DB size: {db_size:.0f} MB')


if __name__ == '__main__':
    main()
