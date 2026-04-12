"""
OTP vs Raptor choice set coverage comparison.

Same SC trips matched against both choice sets using compute_composite_similarity.
Sample 10K ODs, 1 day TCN.
"""
import gc
import glob as glob_mod
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code' / 'similarity'))

from module.similarity import (
    parse_otp_itinerary, parse_smartcard_trip,
    compute_all_metrics, compute_composite_similarity,
)

OTP_DB = ROOT / 'data' / 'routing' / 'output' / 'java_default' / 'otp_cache.db'
RAPTOR_DB = ROOT / 'data' / 'routing' / 'output' / 'rust_calibrated' / 'raptor_cache_filtered.db'
TCN_DIR = ROOT / 'data' / 'tcn'

SIM_WEIGHTS = {'mode': 0.39, 'route': 0.40, 'sequence': 0.21}
THRESHOLD = 0.5

TCN_COLUMNS = [
    'od_pair', 'transport_category',
    '환승횟수', '정류장명칭시퀀스', '정류장시퀀스',
    '총탑승시간', '노선명', '노선ID',
    '승차정류장 X 좌표', '승차정류장 Y 좌표',
    '하차정류장 X 좌표', '하차정류장 Y 좌표',
    '정류장lat시퀀스', '정류장lon시퀀스', '승차일시',
]

N_SAMPLE = 10000


def get_best_score(otp_parsed_list, sc_parsed):
    best = -1.0
    for otp_p in otp_parsed_list:
        metrics = compute_all_metrics(otp_p, sc_parsed, skip_diagnostics=True)
        result = compute_composite_similarity(metrics, weights=SIM_WEIGHTS)
        if result['composite'] > best:
            best = result['composite']
    return best


def main():
    print('=== OTP vs Raptor Choice Set Coverage ===\n')

    otp_conn = sqlite3.connect(str(OTP_DB))
    raptor_conn = sqlite3.connect(str(RAPTOR_DB))

    otp_ods = set(r[0] for r in otp_conn.execute(
        'SELECT od_pair FROM otp_cache'))
    raptor_ods = set(r[0] for r in raptor_conn.execute(
        'SELECT od_pair FROM raptor_cache'))
    common = otp_ods & raptor_ods
    print(f'OTP ODs: {len(otp_ods):,}')
    print(f'Raptor ODs: {len(raptor_ods):,}')
    print(f'Common ODs: {len(common):,}')

    np.random.seed(42)
    sample_ods = set(np.random.choice(
        list(common), min(N_SAMPLE, len(common)), replace=False))
    print(f'Sample: {len(sample_ods):,} ODs')

    # Load 1 day TCN
    tcn_dates = sorted([d for d in TCN_DIR.iterdir() if d.is_dir()])
    tcn_file = list(tcn_dates[0].glob('*.parquet'))[0]
    print(f'\nLoading TCN: {tcn_file.name}')
    tcn = pd.read_parquet(tcn_file, columns=TCN_COLUMNS)
    tcn = tcn[tcn['od_pair'].isin(sample_ods)]
    print(f'  SC trips in sample: {len(tcn):,}')

    # Pre-fetch caches
    print('\nFetching caches...')
    otp_cache = {}
    raptor_cache = {}
    sample_list = list(sample_ods)

    for cs in range(0, len(sample_list), 999):
        chunk = sample_list[cs:cs + 999]
        ph = ','.join(['?'] * len(chunk))
        for row in otp_conn.execute(
                f'SELECT od_pair, data FROM otp_cache WHERE od_pair IN ({ph})',
                chunk):
            otp_cache[row[0]] = pickle.loads(row[1])
        for row in raptor_conn.execute(
                f'SELECT od_pair, data FROM raptor_cache WHERE od_pair IN ({ph})',
                chunk):
            raptor_cache[row[0]] = pickle.loads(row[1])

    otp_conn.close()
    raptor_conn.close()
    print(f'  OTP: {len(otp_cache):,}, Raptor: {len(raptor_cache):,}')

    # Compare
    results = []
    score_cache = {}

    for od_pair, sc_trips in tqdm(tcn.groupby('od_pair'), desc='Comparing'):
        otp_data = otp_cache.get(od_pair)
        raptor_data = raptor_cache.get(od_pair)
        if otp_data is None or raptor_data is None:
            continue

        otp_parsed_list = otp_data['otp_parsed']
        raptor_parsed_list = [
            parse_otp_itinerary(it) for it in raptor_data['itineraries']
        ]

        for sc_dict in sc_trips.to_dict('records'):
            ck = (od_pair,
                  str(sc_dict.get('노선명', '')),
                  str(sc_dict.get('정류장명칭시퀀스', '')))

            if ck not in score_cache:
                sc_row = pd.Series(sc_dict)
                sc_parsed = parse_smartcard_trip(sc_row, lightweight=True)
                score_cache[ck] = (
                    get_best_score(otp_parsed_list, sc_parsed),
                    get_best_score(raptor_parsed_list, sc_parsed),
                )

            otp_best, raptor_best = score_cache[ck]
            results.append({
                'od_pair': od_pair,
                'sc_category': sc_dict.get('transport_category', ''),
                'otp_best': otp_best,
                'raptor_best': raptor_best,
                'n_otp_alts': len(otp_parsed_list),
                'n_raptor_alts': len(raptor_parsed_list),
            })

    df = pd.DataFrame(results)
    print(f'\nSC trips compared: {len(df):,}')

    # === Report ===
    print('\n' + '=' * 70)
    print('  OTP vs Raptor - SC Trip Coverage Comparison')
    print('=' * 70)

    for label, col in [('OTP (기존)', 'otp_best'),
                       ('Raptor (보정)', 'raptor_best')]:
        above = (df[col] >= 0.5).mean() * 100
        p60 = (df[col] >= 0.6).mean() * 100
        p80 = (df[col] >= 0.8).mean() * 100
        mean_s = df[col].mean()
        print(f'\n  [{label}]')
        print(f'    >= 0.5 (매칭):  {above:.1f}%')
        print(f'    >= 0.6 (양호):  {p60:.1f}%')
        print(f'    >= 0.8 (우수):  {p80:.1f}%')
        print(f'    Mean score:     {mean_s:.3f}')

    print(f'\n  [직접 비교]')
    rw = (df['raptor_best'] > df['otp_best']).mean() * 100
    ow = (df['otp_best'] > df['raptor_best']).mean() * 100
    tie = (df['otp_best'] == df['raptor_best']).mean() * 100
    diff = (df['raptor_best'] - df['otp_best']).mean()
    print(f'    Raptor > OTP:  {rw:.1f}%')
    print(f'    OTP > Raptor:  {ow:.1f}%')
    print(f'    Tie:           {tie:.1f}%')
    print(f'    Mean diff:     {diff:+.4f}')

    print(f'\n  [카테고리별 매칭률 >= 0.5]')
    print(f'    {"카테고리":20s}  {"OTP":>7s}  {"Raptor":>7s}  {"diff":>7s}  {"n":>8s}')
    for cat, grp in sorted(df.groupby('sc_category'),
                           key=lambda x: -len(x[1])):
        o = (grp['otp_best'] >= 0.5).mean() * 100
        r = (grp['raptor_best'] >= 0.5).mean() * 100
        print(f'    {cat:20s}  {o:6.1f}%  {r:6.1f}%  {r-o:+6.1f}%  {len(grp):8,}')

    print(f'\n  [평균 대안 수/OD]')
    print(f'    OTP:    {df["n_otp_alts"].mean():.1f}')
    print(f'    Raptor: {df["n_raptor_alts"].mean():.1f}')

    od_otp = df.groupby('od_pair')['otp_best'].max()
    od_raptor = df.groupby('od_pair')['raptor_best'].max()
    print(f'\n  [OD level (best trip >= 0.5)]')
    print(f'    OTP:    {(od_otp >= 0.5).mean() * 100:.1f}%')
    print(f'    Raptor: {(od_raptor >= 0.5).mean() * 100:.1f}%')

    out = ROOT / 'data' / 'results' / 'otp_vs_raptor_coverage.parquet'
    df.to_parquet(out, index=False)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
