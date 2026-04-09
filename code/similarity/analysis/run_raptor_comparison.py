"""
Raptor Choice Set vs SC 실제 선택 커버리지 분석.

각 OD에 대해 Raptor가 생성한 경로 세트가
SC 실제 선택 경로를 얼마나 잘 커버하는지 평가.

Usage:
    python run_raptor_comparison.py
    python run_raptor_comparison.py --n-sample 5000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]

OD_INPUT = ROOT / 'data' / 'otp' / 'input' / 'otp_od_input_over13.csv'
TRAIN_PATH = ROOT / 'data' / 'training_set_new' / 'route_choice_filtered_training.parquet'
GTFS_DIR = str(ROOT / 'data' / 'gtfs' / 'a1')

# Calibrated params from grid search
CALIBRATED = {
    'transfer_cost_secs': 600,
    'walk_reluctance': 2.0,
    'transit_reluctance': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
}


def load_od_coords(od_input_path: Path, od_pairs: set) -> dict:
    """Load OD coordinates + departure time from otp_od_input CSV."""
    df = pd.read_csv(od_input_path)
    df = df[df['od_pair'].isin(od_pairs)]
    dt_str = df['departure_time'].astype(str).str.slice(8, 14)
    df['dep_secs'] = (dt_str.str[:2].astype(int) * 3600
                      + dt_str.str[2:4].astype(int) * 60
                      + dt_str.str[4:6].astype(int))
    coords = {}
    for _, r in df.iterrows():
        coords[r['od_pair']] = {
            'o_lat': r['o_lat'], 'o_lon': r['o_lon'],
            'd_lat': r['d_lat'], 'd_lon': r['d_lon'],
            'dep_secs': int(r['dep_secs']),
        }
    return coords


def get_chosen_features(train_df: pd.DataFrame) -> dict:
    """Get chosen route features per OD for comparison."""
    chosen = train_df[train_df['chosen'] == 1].copy()
    return chosen.set_index('od_pair').to_dict('index')


def _get_category(itin: dict) -> str:
    """Derive transport_category from itinerary legs."""
    modes = set()
    for leg in itin.get('legs', []):
        m = leg.get('mode', '')
        if m == 'BUS':
            modes.add('bus')
        elif m in ('SUBWAY', 'RAIL'):
            modes.add('train')
        elif m == 'GTX':
            modes.add('gtx')
    parts = sorted(modes)
    if not parts:
        return 'unknown'
    return '+'.join(parts) + ('_only' if len(parts) == 1 else '')


def _extract_itin_features(itin: dict) -> dict:
    """Extract comparable features from a Raptor itinerary."""
    attrs = itin.get('attributes', {})
    fare_info = itin.get('fare', {})
    return {
        'transfers': attrs.get('num_transit_legs', 1) - 1,
        'category': _get_category(itin),
        'duration': itin.get('duration', 0),
        'fare': fare_info.get('total_krw', 0),
        'ivt': attrs.get('in_vehicle_time_sec', 0),
        'walk_dist': itin.get('walkDistance', 0),
        'gc': itin.get('generalizedCost', 0),
    }


def evaluate_coverage(raptor_itin: list, sc: dict) -> dict:
    """Evaluate how well Raptor choice set covers SC chosen route.

    For each OD, check:
    1. Does the set contain a route matching SC transfers?
    2. Does the set contain a route matching SC category?
    3. Does the set contain a route matching SC fare?
    4. Does the set contain min_transfers / min_fare / min_duration route?
    5. Best match: closest route to SC across all features
    """
    if not raptor_itin:
        return {'has_routes': False}

    sc_transfers = int(sc.get('num_transfers', 0) or 0)
    sc_category = sc.get('transport_category', '')
    sc_fare = int(sc.get('fare', 0) or 0)
    sc_duration = sc.get('total_duration', 0) or 0
    sc_ivt = sc.get('in_vehicle_time', 0) or 0

    routes = [_extract_itin_features(it) for it in raptor_itin]

    # --- Coverage: does any route in the set match SC on each criterion? ---
    has_transfer_match = any(r['transfers'] == sc_transfers for r in routes)
    has_category_match = any(r['category'] == sc_category for r in routes)
    has_fare_match = any(r['fare'] == sc_fare for r in routes)

    # Duration within tolerance
    has_dur_60s = any(abs(r['duration'] - sc_duration) <= 60 for r in routes)
    has_dur_180s = any(abs(r['duration'] - sc_duration) <= 180 for r in routes)

    # --- Does the set contain diverse route types? ---
    has_min_transfer = True  # MC Raptor always includes min-transfer
    has_min_duration = True  # MC Raptor always includes min-time
    unique_transfers = len(set(r['transfers'] for r in routes))
    unique_categories = len(set(r['category'] for r in routes))

    # --- Best match: route closest to SC on multiple criteria ---
    best_score = -1
    best_route = None
    for r in routes:
        score = 0
        if r['transfers'] == sc_transfers:
            score += 1
        if r['category'] == sc_category:
            score += 1
        if r['fare'] == sc_fare:
            score += 1
        if abs(r['duration'] - sc_duration) <= 180:
            score += 1
        if score > best_score:
            best_score = score
            best_route = r

    # How close is the best match?
    all_match = (best_score == 4)
    best_dur_diff = best_route['duration'] - sc_duration if best_route else 0
    best_ivt_diff = best_route['ivt'] - sc_ivt if best_route else 0

    return {
        'has_routes': True,
        'n_routes': len(routes),
        'sc_category': sc_category,
        # Coverage rates (binary per OD)
        'cover_transfers': has_transfer_match,
        'cover_category': has_category_match,
        'cover_fare': has_fare_match,
        'cover_dur_60s': has_dur_60s,
        'cover_dur_180s': has_dur_180s,
        # Diversity
        'unique_transfers': unique_transfers,
        'unique_categories': unique_categories,
        # Best match quality
        'best_match_score': best_score,  # out of 4
        'all_match': all_match,
        'best_dur_diff': best_dur_diff,
        'best_ivt_diff': best_ivt_diff,
    }


def run_coverage(raptor, od_coords, chosen_dict, label=''):
    """Run Raptor on ODs and evaluate choice set coverage."""
    results = []
    for od_pair, coords in tqdm(od_coords.items(), desc=f'Routing ({label})'):
        if od_pair not in chosen_dict:
            continue
        try:
            result = raptor.route(
                coords['o_lat'], coords['o_lon'],
                coords['d_lat'], coords['d_lon'],
                coords['dep_secs'], mode='mc',
            )
            itin = result.get('data', {}).get('plan', {}).get('itineraries', [])
        except Exception:
            itin = []

        m = evaluate_coverage(itin, chosen_dict[od_pair])
        m['od_pair'] = od_pair
        results.append(m)
    return pd.DataFrame(results)


def print_report(df):
    """Print choice set coverage report."""
    matched = df[df['has_routes']]
    n = len(matched)
    no_route = len(df) - n

    print('\n' + '=' * 70)
    print('Raptor Choice Set → SC 커버리지 분석')
    print(f'  총 {len(df)} ODs, 경로 있음 {n}, 경로 없음 {no_route}')
    print(f'  평균 경로 수/OD: {matched["n_routes"].mean():.1f}')
    print('=' * 70)

    print('\n  [1] SC 선택 경로를 Raptor set이 커버하는 비율')
    print(f'      환승 횟수 일치 경로 존재:    {matched["cover_transfers"].mean()*100:.1f}%')
    print(f'      수송 카테고리 일치 경로 존재: {matched["cover_category"].mean()*100:.1f}%')
    print(f'      요금 일치 경로 존재:         {matched["cover_fare"].mean()*100:.1f}%')
    print(f'      소요시간 ±60s 경로 존재:     {matched["cover_dur_60s"].mean()*100:.1f}%')
    print(f'      소요시간 ±180s 경로 존재:    {matched["cover_dur_180s"].mean()*100:.1f}%')

    print('\n  [2] Best Match (환승+카테고리+요금+소요시간 동시 일치)')
    scores = matched['best_match_score']
    for s in [4, 3, 2, 1, 0]:
        pct = (scores == s).mean() * 100
        label = {4: '4/4 완전 일치', 3: '3/4', 2: '2/4', 1: '1/4', 0: '0/4'}[s]
        print(f'      {label}:  {pct:.1f}%')

    print(f'\n      완전 일치율 (4/4):           {matched["all_match"].mean()*100:.1f}%')

    print('\n  [3] Choice Set 다양성')
    print(f'      평균 환승 옵션 수:           {matched["unique_transfers"].mean():.1f}')
    print(f'      평균 카테고리 옵션 수:       {matched["unique_categories"].mean():.1f}')

    print('\n  [4] 수송 카테고리별 커버리지')
    cols = ['cover_transfers', 'cover_category', 'cover_fare', 'cover_dur_180s']
    labels = ['환승', '카테고리', '요금', '시간±180s']
    print(f'      {"카테고리":20s}  {"환승":>6s}  {"카테고리":>8s}  {"요금":>6s}  {"시간±3m":>8s}  {"n":>5s}')
    cats = matched.groupby('sc_category')
    for cat, grp in sorted(cats, key=lambda x: -len(x[1])):
        vals = [f'{grp[c].mean()*100:5.1f}%' for c in cols]
        print(f'      {cat:20s}  {"  ".join(vals)}  {len(grp):5d}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-sample', type=int, default=2000,
                        help='Number of ODs to sample (0=all)')
    args = parser.parse_args()

    from dtumos_raptor import DtumosRaptor

    # Load training data
    print('Loading training data...')
    train = pd.read_parquet(TRAIN_PATH)
    chosen_dict = get_chosen_features(train)
    od_set = set(chosen_dict.keys())
    print(f'  {len(od_set):,} ODs with chosen routes')

    # Sample if needed
    if args.n_sample > 0 and args.n_sample < len(od_set):
        np.random.seed(42)
        od_set = set(np.random.choice(list(od_set), args.n_sample, replace=False))
        print(f'  Sampled {len(od_set):,} ODs')

    # Load OD coordinates
    print('Loading OD coordinates...')
    t0 = time.time()
    od_coords = load_od_coords(OD_INPUT, od_set)
    print(f'  {len(od_coords):,} ODs with coords ({time.time()-t0:.1f}s)')

    # Init Raptor with calibrated params
    print('Loading Raptor (calibrated params)...')
    raptor = DtumosRaptor(num_threads=8)
    raptor.load_gtfs(GTFS_DIR)
    raptor.update_config(CALIBRATED)

    # Run coverage analysis
    df = run_coverage(raptor, od_coords, chosen_dict, 'Calibrated')
    print_report(df)

    # Save
    out_dir = ROOT / 'data' / 'training_set_new'
    df.to_parquet(out_dir / 'raptor_coverage_calibrated.parquet', index=False)
    print(f'\nSaved to {out_dir / "raptor_coverage_calibrated.parquet"}')


if __name__ == '__main__':
    main()
