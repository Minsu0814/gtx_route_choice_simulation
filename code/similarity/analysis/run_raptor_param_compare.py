"""
Original vs Calibrated Raptor → SC 일치율 비교.

두 파라미터 세트로 Raptor를 돌려서
각각 SC 실제 선택과 얼마나 일치하는지 비교.

Usage:
    python run_raptor_param_compare.py
    python run_raptor_param_compare.py --n-sample 5000
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]

OD_INPUT = ROOT / 'data' / 'otp' / 'input' / 'otp_od_input_over13.csv'
TRAIN_PATH = ROOT / 'data' / 'training_set_new' / 'route_choice_filtered_training.parquet'
GTFS_DIR = str(ROOT / 'data' / 'gtfs' / 'a1')

CALIBRATED = {
    'transfer_cost_secs': 600,
    'walk_reluctance': 2.0,
    'transit_reluctance': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
}

ORIGINAL = {
    'transfer_cost_secs': 120,
    'walk_reluctance': 1.0,
    'transit_reluctance': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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


def match_route(raptor_itin: list, sc: dict) -> dict:
    """Compare Raptor GC-optimal with SC chosen route."""
    if not raptor_itin:
        return {'matched': False, 'reason': 'no_itinerary'}

    sc_transfers = int(sc.get('num_transfers', 0) or 0)
    sc_category = sc.get('transport_category', '')
    sc_duration = sc.get('total_duration', 0) or 0
    sc_fare = int(sc.get('fare', 0) or 0)
    sc_ivt = sc.get('in_vehicle_time', 0) or 0

    best = min(raptor_itin, key=lambda x: x.get('generalizedCost', float('inf')))
    attrs = best.get('attributes', {})
    raptor_transfers = attrs.get('num_transit_legs', 1) - 1
    raptor_category = _get_category(best)
    raptor_duration = best.get('duration', 0)
    raptor_fare = best.get('fare', {}).get('total_krw', 0)
    raptor_ivt = attrs.get('in_vehicle_time_sec', 0)

    return {
        'matched': True,
        'n_raptor_routes': len(raptor_itin),
        'transfer_match': raptor_transfers == sc_transfers,
        'category_match': raptor_category == sc_category,
        'fare_match': raptor_fare == sc_fare,
        'duration_diff': raptor_duration - sc_duration,
        'ivt_diff': raptor_ivt - sc_ivt,
        'sc_category': sc_category,
        'sc_route_in_raptor': any(
            (it['attributes'].get('num_transit_legs', 1) - 1) == sc_transfers
            for it in raptor_itin
        ),
    }


def run_comparison(raptor, od_coords, chosen_dict, label=''):
    """Run Raptor on ODs and compare with SC choices."""
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
        m = match_route(itin, chosen_dict[od_pair])
        m['od_pair'] = od_pair
        results.append(m)
    return pd.DataFrame(results)


def print_comparison(df_orig, df_calib):
    """Print comparison report."""
    print('\n' + '=' * 70)
    print('RAPTOR vs SC(스마트카드) 비교')
    print('  기준(정답) = SC 실제 선택 경로')
    print('  비교 대상 = Raptor GC-최적 경로 (파라미터별)')
    print('=' * 70)

    for label, df in [('Original', df_orig), ('Calibrated', df_calib)]:
        matched = df[df['matched']]
        n = len(matched)
        if n == 0:
            print(f'\n{label}: No matched routes')
            continue

        print(f'\n  [{label} Raptor vs SC] {n} ODs routed')
        print(f'    Avg routes per OD:          {matched["n_raptor_routes"].mean():.1f}')
        print(f'    --- Match rates ---')
        print(f'    Transfer count match:       {matched["transfer_match"].mean()*100:.1f}%')
        print(f'    Transport category match:   {matched["category_match"].mean()*100:.1f}%')
        print(f'    Fare match (exact):         {matched["fare_match"].mean()*100:.1f}%')
        print(f'    SC route in Raptor set:     {matched["sc_route_in_raptor"].mean()*100:.1f}%')
        print(f'    --- Duration (sec) ---')
        dur = matched['duration_diff']
        print(f'    Mean diff (Raptor - SC):    {dur.mean():.0f}s')
        print(f'    MAE:                        {dur.abs().mean():.0f}s')
        print(f'    Within ±60s:                {(dur.abs() <= 60).mean()*100:.1f}%')
        print(f'    Within ±180s:               {(dur.abs() <= 180).mean()*100:.1f}%')
        print(f'    --- IVT (sec) ---')
        ivt = matched['ivt_diff']
        print(f'    Mean diff:                  {ivt.mean():.0f}s')
        print(f'    MAE:                        {ivt.abs().mean():.0f}s')
        print(f'    --- Category breakdown ---')
        cats = matched.groupby('sc_category')['category_match'].agg(['mean', 'count'])
        for cat, row in cats.iterrows():
            print(f'    {cat:20s}  {row["mean"]*100:5.1f}%  (n={int(row["count"])})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-sample', type=int, default=2000,
                        help='Number of ODs to sample (0=all)')
    args = parser.parse_args()

    from dtumos_raptor import DtumosRaptor

    print('Loading training data...')
    train = pd.read_parquet(TRAIN_PATH)
    chosen_dict = get_chosen_features(train)
    od_set = set(chosen_dict.keys())
    print(f'  {len(od_set):,} ODs with chosen routes')

    if args.n_sample > 0 and args.n_sample < len(od_set):
        np.random.seed(42)
        od_set = set(np.random.choice(list(od_set), args.n_sample, replace=False))
        print(f'  Sampled {len(od_set):,} ODs')

    print('Loading OD coordinates...')
    t0 = time.time()
    od_coords = load_od_coords(OD_INPUT, od_set)
    print(f'  {len(od_coords):,} ODs with coords ({time.time()-t0:.1f}s)')

    print('Loading Raptor...')
    raptor = DtumosRaptor(num_threads=8)
    raptor.load_gtfs(GTFS_DIR)

    print('\n--- Original Raptor vs SC (transfer=120, walk=1.0, transit=1.0) ---')
    raptor.update_config(ORIGINAL)
    df_orig = run_comparison(raptor, od_coords, chosen_dict, 'Original→SC')

    print('\n--- Calibrated Raptor vs SC (transfer=600, walk=2.0, transit=0.3) ---')
    raptor.update_config(CALIBRATED)
    df_calib = run_comparison(raptor, od_coords, chosen_dict, 'Calibrated→SC')

    print_comparison(df_orig, df_calib)

    out_dir = ROOT / 'data' / 'training_set_new'
    df_orig.to_parquet(out_dir / 'raptor_comparison_original.parquet', index=False)
    df_calib.to_parquet(out_dir / 'raptor_comparison_calibrated.parquet', index=False)
    print(f'\nSaved to {out_dir}')


if __name__ == '__main__':
    main()
