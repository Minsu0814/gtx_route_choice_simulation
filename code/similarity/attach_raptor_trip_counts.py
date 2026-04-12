"""
Attach SC trip counts to Raptor choice set using composite similarity matching.

Same matching logic as build_training_set.py:
  1. Parse SC trip → parse_smartcard_trip()
  2. Compare against each Raptor alt's otp_parsed → compute_all_metrics()
  3. Best composite score → chosen alt (threshold=0.5)
  4. Count matches per alt → choice_prob

Input:  data/routing/output/rust_calibrated/raptor_cache_filtered.db
        data/tcn/*/TCN_*.parquet (7 days)
Output: data/choice_set/rust_calibrated/raptor_choice_set_weighted.parquet

Usage: python attach_raptor_trip_counts.py
"""
import gc
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')
sys.path.insert(0, BASE_DIR)

from module.similarity import (
    parse_otp_itinerary,
    parse_smartcard_trip,
    compute_all_metrics,
    compute_composite_similarity,
)

RAPTOR_DB = os.path.join(DATA_DIR, 'routing', 'output', 'rust_calibrated',
                         'raptor_cache_filtered.db')
RAPTOR_PARQUET = os.path.join(DATA_DIR, 'choice_set', 'rust_calibrated',
                              'raptor_choice_set.parquet')
TCN_DIR = os.path.join(DATA_DIR, 'tcn')
OUTPUT_PATH = os.path.join(DATA_DIR, 'choice_set', 'rust_calibrated',
                           'raptor_choice_set_weighted.parquet')

# Same weights as build_training_set.py
SIM_WEIGHTS = {'mode': 0.39, 'route': 0.40, 'sequence': 0.21}
SIMILARITY_THRESHOLD = 0.5

TCN_COLUMNS = [
    'od_pair', 'transport_category',
    '환승횟수', '정류장명칭시퀀스', '정류장시퀀스',
    '총탑승시간', '노선명', '노선ID',
    '승차정류장 X 좌표', '승차정류장 Y 좌표',
    '하차정류장 X 좌표', '하차정류장 Y 좌표',
    '정류장lat시퀀스', '정류장lon시퀀스', '승차일시',
]


def load_raptor_od_set(db_path):
    """Get set of OD pairs in Raptor cache."""
    conn = sqlite3.connect(db_path)
    od_set = set(
        r[0] for r in conn.execute('SELECT od_pair FROM raptor_cache')
    )
    conn.close()
    return od_set


def match_od_trips(sc_records, otp_parsed_list, n_alts):
    """Match SC trips for one OD against Raptor alternatives.

    Returns: (n_matched_per_alt, n_total, n_above_threshold)
    """
    n_matched = np.zeros(n_alts, dtype=np.int64)
    n_total = len(sc_records)
    n_above = 0

    # Cache by (route_name, stop_seq) — same trip pattern gets same score
    score_cache = {}

    for sc_dict in sc_records:
        cache_key = (str(sc_dict.get('노선명', '')),
                     str(sc_dict.get('정류장명칭시퀀스', '')))

        if cache_key not in score_cache:
            sc_row = pd.Series(sc_dict)
            sc_parsed = parse_smartcard_trip(sc_row, lightweight=True)

            best_idx = -1
            best_score = -1.0
            for idx, otp_p in enumerate(otp_parsed_list):
                metrics = compute_all_metrics(
                    otp_p, sc_parsed, skip_diagnostics=True)
                result = compute_composite_similarity(
                    metrics, weights=SIM_WEIGHTS)
                if result['composite'] > best_score:
                    best_score = result['composite']
                    best_idx = idx
            score_cache[cache_key] = (best_idx, best_score)

        best_idx, best_score = score_cache[cache_key]
        if best_score >= SIMILARITY_THRESHOLD:
            n_matched[best_idx] += 1
            n_above += 1

    return n_matched, n_total, n_above


def main():
    print('=== Attach Trip Counts (Composite Similarity) ===\n')

    # 1. Load Raptor OD set
    print('Loading Raptor cache OD set...')
    raptor_od_set = load_raptor_od_set(RAPTOR_DB)
    print(f'  {len(raptor_od_set):,} ODs in Raptor cache')

    # 2. Load TCN dates
    tcn_dates = sorted([d for d in os.listdir(TCN_DIR)
                        if os.path.isdir(os.path.join(TCN_DIR, d))])
    print(f'TCN dates: {tcn_dates} ({len(tcn_dates)} days)')

    # 3. Process day by day — accumulate per-OD match counts
    # od_match_counts[od_pair] = {alt_idx: count, ..., '_total': N}
    od_match_counts = {}

    conn = sqlite3.connect(RAPTOR_DB)
    stats = {'total_sc': 0, 'matched_sc': 0, 'ods_processed': 0}

    for date in tcn_dates:
        print(f'\n--- {date} ---')
        import glob as glob_mod
        tcn_files = glob_mod.glob(
            os.path.join(TCN_DIR, date, '*.parquet'))
        if not tcn_files:
            print('  No files, skip')
            continue

        tcn_day = pd.read_parquet(tcn_files[0], columns=TCN_COLUMNS)
        n_total = len(tcn_day)
        tcn_day = tcn_day[tcn_day['od_pair'].isin(raptor_od_set)]
        n_matched = len(tcn_day)
        print(f'  {n_total:,} trips → {n_matched:,} in Raptor OD set')

        if n_matched == 0:
            del tcn_day
            continue

        # Batch-fetch Raptor cache for this day's ODs
        day_od_list = tcn_day['od_pair'].unique().tolist()
        raptor_batch = {}
        for chunk_start in range(0, len(day_od_list), 999):
            chunk = day_od_list[chunk_start:chunk_start + 999]
            ph = ','.join(['?'] * len(chunk))
            for row in conn.execute(
                    f'SELECT od_pair, n_alts, data FROM raptor_cache '
                    f'WHERE od_pair IN ({ph})', chunk):
                raptor_batch[row[0]] = (row[1], row[2])

        day_matched_sc = 0
        for od_pair, sc_trips in tqdm(
                tcn_day.groupby('od_pair'),
                desc=f'  Matching {date}',
                total=tcn_day['od_pair'].nunique()):

            cached = raptor_batch.get(od_pair)
            if cached is None:
                continue

            n_alts, blob = cached
            cache_data = pickle.loads(blob)
            # Use raw itineraries + parse_otp_itinerary (full format)
            # otp_parsed in cache is lightweight (stop_coords/transit_legs only)
            itineraries = cache_data.get('itineraries', [])
            otp_parsed_list = [
                parse_otp_itinerary(itin) for itin in itineraries
            ]

            sc_records = sc_trips.to_dict('records')
            n_matched_arr, n_total_trips, n_above = match_od_trips(
                sc_records, otp_parsed_list, n_alts)

            # Accumulate
            if od_pair not in od_match_counts:
                od_match_counts[od_pair] = {
                    'n_alts': n_alts,
                    'matched': np.zeros(n_alts, dtype=np.int64),
                    'total': 0,
                }
            entry = od_match_counts[od_pair]
            entry['matched'] += n_matched_arr
            entry['total'] += n_total_trips

            stats['total_sc'] += n_total_trips
            stats['matched_sc'] += n_above
            day_matched_sc += n_above
            stats['ods_processed'] += 1

        del raptor_batch, tcn_day
        gc.collect()
        print(f'  Day matched SC trips: {day_matched_sc:,}')

    conn.close()

    # 4. Summary of matching
    print(f'\n=== Matching Summary ===')
    print(f'  Total SC trips processed: {stats["total_sc"]:,}')
    print(f'  SC trips above threshold: {stats["matched_sc"]:,} '
          f'({stats["matched_sc"]/max(1,stats["total_sc"])*100:.1f}%)')
    print(f'  ODs with SC data: {len(od_match_counts):,}')

    ods_with_match = sum(
        1 for v in od_match_counts.values() if v['matched'].sum() > 0)
    print(f'  ODs with ≥1 matched alt: {ods_with_match:,} '
          f'({ods_with_match/max(1,len(od_match_counts))*100:.1f}%)')

    # 5. Apply to Raptor parquet
    print('\nLoading Raptor choice set parquet...')
    df = pd.read_parquet(RAPTOR_PARQUET)
    print(f'  {len(df):,} rows, {df["od_pair"].nunique():,} ODs')

    n_matched_col = np.zeros(len(df), dtype=np.int64)
    n_total_col = np.zeros(len(df), dtype=np.int64)

    assign_stats = {'matched': 0, 'unmatched': 0, 'no_sc': 0}

    for i, row in tqdm(df.iterrows(), total=len(df), desc='Assigning'):
        od = row['od_pair']
        alt_idx = int(row['alt_idx'])
        entry = od_match_counts.get(od)

        if entry is not None:
            n_total_col[i] = entry['total']
            if alt_idx < len(entry['matched']):
                n_matched_col[i] = entry['matched'][alt_idx]
            if n_matched_col[i] > 0:
                assign_stats['matched'] += 1
            else:
                assign_stats['unmatched'] += 1
        else:
            n_total_col[i] = 1  # fallback
            assign_stats['no_sc'] += 1

    df['n_matched'] = n_matched_col
    df['n_total'] = n_total_col

    print(f'\n  Alts with match:    {assign_stats["matched"]:>10,}')
    print(f'  Alts without match: {assign_stats["unmatched"]:>10,}')
    print(f'  ODs without SC:     {assign_stats["no_sc"]:>10,}')

    # 6. Compute choice_prob + renormalize
    print('\nComputing choice_prob...')
    df['choice_prob'] = df['n_matched'] / df['n_total'].clip(lower=1)

    prob_sums = df.groupby('od_pair')['choice_prob'].transform('sum')
    mask_pos = prob_sums > 0
    df.loc[mask_pos, 'choice_prob'] = (
        df.loc[mask_pos, 'choice_prob'] / prob_sums[mask_pos]
    )

    # Zero-sum ODs: uniform
    mask_zero = prob_sums == 0
    if mask_zero.any():
        n_zero = df.loc[mask_zero, 'od_pair'].nunique()
        print(f'  {n_zero:,} ODs with zero prob sum -> uniform')
        cs_sizes = (df[mask_zero].groupby('od_pair')['alt_idx']
                    .transform('count'))
        df.loc[mask_zero, 'choice_prob'] = 1.0 / cs_sizes

    # 7. Verify
    print('\n=== Verification ===')
    prob_sums_final = df.groupby('od_pair')['choice_prob'].sum()
    not_one = prob_sums_final[(prob_sums_final - 1.0).abs() > 0.01]
    print(f'  Total rows:        {len(df):,}')
    print(f'  Total ODs:         {df["od_pair"].nunique():,}')
    print(f'  choice_prob sum=1: '
          f'{"PASS" if len(not_one) == 0 else f"FAIL ({len(not_one)} ODs)"}')

    ods_matched = df[df['n_matched'] > 0]['od_pair'].nunique()
    total_ods = df['od_pair'].nunique()
    print(f'  ODs with matched alts: {ods_matched:,} / '
          f'{total_ods:,} ({ods_matched/total_ods*100:.1f}%)')
    print(f'  choice_prob > 0: '
          f'{(df["choice_prob"] > 0).sum():,} / {len(df):,}')

    od_ntotal = df.groupby('od_pair')['n_total'].first()
    print(f'  n_total: mean={od_ntotal.mean():.1f}, '
          f'median={od_ntotal.median():.0f}, max={od_ntotal.max():,}')

    # 8. Save
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f'\nSaved: {OUTPUT_PATH}')
    print(f'  {len(df):,} rows, {len(df.columns)} columns')


if __name__ == '__main__':
    main()
