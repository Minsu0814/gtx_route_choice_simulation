"""
MNL Route Choice Assignment.
Loads trained betas → computes choice probabilities for all ODs in OTP cache.

Usage:
    python assign_mnl.py                        # default: A_total_uncon
    python assign_mnl.py --spec B_total_walk_uncon
"""
import argparse
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code' / 'similarity'))

from spec_config import MNL_SPECS, CATEGORY_MAP, ASC_REF
from module.feature_transform import transform_raw_features, build_feature_vector

CACHE_DB = ROOT / 'data' / 'cache' / 'otp' / 'otp_cache_filtered.db'
MNL_JSON = ROOT / 'data' / 'results' / 'mnl_weighted.json'
OUT_DIR = ROOT / 'data' / 'results'

# Raw feature columns to keep in output
RAW_COLS = [
    'generalized_cost', 'num_transfers', 'total_duration',
    'walk_distance', 'fare', 'transport_category',
    'access_time', 'egress_time', 'in_vehicle_time',
    'waiting_time', 'transfer_walk_time',
]


def load_model(json_path: Path, spec_name: str):
    """Load beta vector and feature names from mnl_4spec.json."""
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    for r in results:
        if r['spec'] == spec_name:
            return np.array(r['beta']), r['names']
    raise ValueError(f'Spec {spec_name} not found in {json_path}')


def get_asc_cats(names: list) -> list:
    """Extract ASC category names from full feature names."""
    return [n.replace('ASC_', '') for n in names if n.startswith('ASC_')]


def softmax_probs(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute MNL probabilities for one choice set."""
    V = X @ beta
    V = V - V.max()  # numerical stability
    exp_V = np.exp(V)
    return exp_V / exp_V.sum()


def process_batch(rows, beta, feature_names, asc_cats, spec_features):
    """Process a batch of OD rows from cache. Returns list of row dicts."""
    output_rows = []
    for od_pair, n_alts, blob in rows:
        data = pickle.loads(blob)
        alt_features_list = data['alt_features']

        # Build feature matrix
        X = np.zeros((len(alt_features_list), len(beta)))
        transformed = []
        for i, feat in enumerate(alt_features_list):
            t = transform_raw_features(feat)
            transformed.append(t)
            X[i] = build_feature_vector(t, spec_features, asc_cats)

        probs = softmax_probs(X, beta)

        for i, feat in enumerate(alt_features_list):
            row = {'od_pair': od_pair, 'alt_idx': i, 'prob': probs[i]}
            for col in RAW_COLS:
                row[col] = feat.get(col)
            output_rows.append(row)

    return output_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', default='A_total_uncon')
    parser.add_argument('--batch-size', type=int, default=5000)
    args = parser.parse_args()

    print(f'Loading model: {args.spec}')
    beta, all_names = load_model(MNL_JSON, args.spec)
    asc_cats = get_asc_cats(all_names)
    spec_features = [n for n in all_names if not n.startswith('ASC_')]
    print(f'  Features: {spec_features}')
    print(f'  ASC cats: {asc_cats}')
    print(f'  Beta: {dict(zip(all_names, beta))}')

    conn = sqlite3.connect(str(CACHE_DB))
    total = conn.execute('SELECT COUNT(*) FROM otp_cache').fetchone()[0]
    print(f'\nCache: {total:,} ODs')

    cursor = conn.execute('SELECT od_pair, n_alts, data FROM otp_cache')
    all_rows = []
    batch = []

    t0 = time.time()
    for row in tqdm(cursor, total=total, desc='Assigning'):
        batch.append(row)
        if len(batch) >= args.batch_size:
            all_rows.extend(
                process_batch(batch, beta, all_names, asc_cats, spec_features))
            batch = []

    if batch:
        all_rows.extend(
            process_batch(batch, beta, all_names, asc_cats, spec_features))

    conn.close()
    elapsed = time.time() - t0

    df = pd.DataFrame(all_rows)
    out_path = OUT_DIR / f'assignment_{args.spec}.parquet'
    df.to_parquet(out_path, index=False)

    # Summary
    n_ods = df['od_pair'].nunique()
    print(f'\nDone in {elapsed:.1f}s')
    print(f'Output: {out_path}')
    print(f'  {n_ods:,} ODs, {len(df):,} alternatives')
    print(f'  Prob sum check (sample): {df.groupby("od_pair")["prob"].sum().describe()}')


if __name__ == '__main__':
    main()
