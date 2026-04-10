"""
Raptor Calibration AutoResearch - Data + Routing + Evaluation (FIXED LAYER).
Do NOT modify this file.
"""
import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
GTFS_DIR = str(ROOT / 'data' / 'gtfs' / 'a1')
OD_INPUT = ROOT / 'data' / 'otp' / 'input' / 'otp_od_input_over13.csv'
TRAIN_PATH = (ROOT / 'data' / 'training_set_new'
              / 'route_choice_filtered_training.parquet')

RESULTS_TSV = Path(__file__).resolve().parent / 'results.tsv'
BASELINE_JSON = Path(__file__).resolve().parent / 'baseline.json'


def load_od_coords(od_pairs: set) -> dict:
    """Load OD coordinates + departure time."""
    df = pd.read_csv(OD_INPUT)
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


def get_chosen_features() -> dict:
    """Get chosen route features per OD from training set."""
    train = pd.read_parquet(TRAIN_PATH)
    chosen = train[train['chosen'] == 1].copy()
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


def run_raptor(params: dict, n_sample: int = 2000,
               seed: int = 42) -> dict:
    """Run Raptor with given params and evaluate against SC.

    Args:
        params: dict with keys transfer_cost_secs, walk_reluctance,
                bus_reluctance, train_reluctance, wait_reluctance
        n_sample: number of ODs to sample
        seed: random seed for sampling

    Returns:
        dict with metrics and details
    """
    from dtumos_raptor import DtumosRaptor

    # Build config
    bus_r = params.get('bus_reluctance', 1.0)
    train_r = params.get('train_reluctance', 1.0)
    config = {
        'transfer_cost_secs': params['transfer_cost_secs'],
        'walk_reluctance': params['walk_reluctance'],
        'wait_reluctance': params.get('wait_reluctance', 1.5),
        'transit_reluctance': [bus_r, train_r, train_r,
                               train_r, train_r, train_r],
    }

    print('Loading SC chosen routes...', flush=True)
    chosen_dict = get_chosen_features()
    od_set = set(chosen_dict.keys())

    if n_sample > 0 and n_sample < len(od_set):
        np.random.seed(seed)
        od_set = set(np.random.choice(list(od_set), n_sample,
                                      replace=False))
    print(f'  {len(od_set):,} ODs sampled', flush=True)

    print('Loading OD coordinates...', flush=True)
    od_coords = load_od_coords(od_set)
    print(f'  {len(od_coords):,} ODs with coords', flush=True)

    print('Loading Raptor...', flush=True)
    raptor = DtumosRaptor(num_threads=8)
    raptor.load_gtfs(GTFS_DIR)
    raptor.update_config(config)

    print(f'Routing {len(od_coords):,} ODs...', flush=True)
    t0 = time.time()
    results = []
    for od_pair, coords in tqdm(od_coords.items(), desc='Routing'):
        if od_pair not in chosen_dict:
            continue
        try:
            result = raptor.route(
                coords['o_lat'], coords['o_lon'],
                coords['d_lat'], coords['d_lon'],
                coords['dep_secs'], mode='mc',
            )
            itin = (result.get('data', {}).get('plan', {})
                    .get('itineraries', []))
        except Exception:
            itin = []

        sc = chosen_dict[od_pair]
        if not itin:
            results.append({'matched': False})
            continue

        best = min(itin,
                   key=lambda x: x.get('generalizedCost', float('inf')))
        attrs = best.get('attributes', {})
        r_xfer = attrs.get('num_transit_legs', 1) - 1
        r_cat = _get_category(best)
        r_walk = attrs.get('walking_time_sec', 0)

        sc_xfer = int(sc.get('num_transfers', 0) or 0)
        sc_cat = sc.get('transport_category', '')
        sc_walk = (float(sc.get('access_time', 0) or 0)
                   + float(sc.get('egress_time', 0) or 0)
                   + float(sc.get('transfer_walk_time', 0) or 0))

        results.append({
            'matched': True,
            'transfer_match': r_xfer == sc_xfer,
            'category_match': r_cat == sc_cat,
            'walk_match': r_walk <= sc_walk,
            'sc_category': sc_cat,
        })

    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.0f}s', flush=True)

    df = pd.DataFrame(results)
    matched = df[df['matched']]
    n = len(matched)

    if n == 0:
        return {'transfer_match': 0, 'category_match': 0,
                'walk_match': 0, 'n_matched': 0, 'n_total': len(df)}

    metrics = {
        'transfer_match': round(float(matched['transfer_match'].mean()), 4),
        'category_match': round(float(matched['category_match'].mean()), 4),
        'walk_match': round(float(matched['walk_match'].mean()), 4),
        'n_matched': int(n),
        'n_total': int(len(df)),
        'routing_time_s': round(elapsed, 1),
    }

    # Category breakdown
    cats = (matched.groupby('sc_category')[['category_match',
                                             'transfer_match',
                                             'walk_match']]
            .mean())
    counts = matched.groupby('sc_category').size()
    print('\n  Category breakdown:', flush=True)
    print(f'    {"category":20s} {"cat%":>6} {"xfer%":>6} '
          f'{"walk%":>6} {"n":>6}', flush=True)
    for cat in cats.index:
        row = cats.loc[cat]
        print(f'    {cat:20s} {row["category_match"]*100:5.1f}% '
              f'{row["transfer_match"]*100:5.1f}% '
              f'{row["walk_match"]*100:5.1f}% '
              f'{int(counts[cat]):>5}', flush=True)

    return metrics


def log_result(exp_name: str, params: dict, metrics: dict,
               notes: str = ''):
    """Append result to results.tsv."""
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, 'w') as f:
            f.write('timestamp\texperiment\ttransfer\twalk\tbus\t'
                    'train\twait\tcat_match\txfer_match\t'
                    'walk_match\tnotes\n')
    with open(RESULTS_TSV, 'a') as f:
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{ts}\t{exp_name}\t'
                f'{params["transfer_cost_secs"]}\t'
                f'{params["walk_reluctance"]}\t'
                f'{params.get("bus_reluctance", 1.0)}\t'
                f'{params.get("train_reluctance", 1.0)}\t'
                f'{params.get("wait_reluctance", 1.5)}\t'
                f'{metrics["category_match"]:.4f}\t'
                f'{metrics["transfer_match"]:.4f}\t'
                f'{metrics["walk_match"]:.4f}\t'
                f'{notes}\n')


def load_baseline():
    """Load current baseline."""
    if BASELINE_JSON.exists():
        with open(BASELINE_JSON, 'r') as f:
            return json.load(f)
    return None


def save_baseline(exp_name: str, params: dict, metrics: dict):
    """Save new baseline."""
    data = {'experiment': exp_name, 'params': params, **metrics}
    with open(BASELINE_JSON, 'w') as f:
        json.dump(data, f, indent=2)


def print_comparison(default_metrics, calibrated_metrics):
    """Print Default vs Calibrated comparison."""
    print(f'\n  {"Metric":<16} {"Default":>10} {"Calibrated":>10} '
          f'{"Delta":>10}', flush=True)
    print(f'  {"-" * 48}', flush=True)
    for key in ['category_match', 'transfer_match', 'walk_match']:
        old = default_metrics.get(key, 0)
        new = calibrated_metrics.get(key, 0)
        delta = new - old
        print(f'  {key:<16} {old*100:>9.1f}% {new*100:>9.1f}% '
              f'{delta*100:>+9.1f}pp', flush=True)


if __name__ == '__main__':
    print('=== Raptor Calibration AutoResearch: Data Check ===',
          flush=True)
    chosen = get_chosen_features()
    print(f'SC chosen routes: {len(chosen):,} ODs', flush=True)
    b = load_baseline()
    if b:
        print(f'Baseline: {b["experiment"]} (score={b["score"]:.4f})',
              flush=True)
    else:
        print('No baseline yet. Run calibrate.py to create one.',
              flush=True)
    print('Data check OK!', flush=True)
