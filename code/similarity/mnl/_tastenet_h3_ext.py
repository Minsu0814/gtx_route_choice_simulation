# -*- coding: utf-8 -*-
"""TasteNet: H3 OD + ext access + n_routes 비교 (3사양)"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import os, json
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / 'data' / 'training_set'
EB_DIR = ROOT / 'data' / 'eb'

sys.path.insert(0, str(ROOT / 'code' / 'similarity' / 'deep_learning'))
from module.models import TasteNetModel
from module.train import set_seed, train_model, evaluate_model
from module import data as data_module
from module.data import CONTEXT_FEATURES, RouteChoiceDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
LR = 1e-3
BATCH = 512
SEED = 42


def load_h3_data():
    """H3 데이터 + 피처 생성"""
    df = pd.read_parquet(EB_DIR / 'h3_choice_prob.parquet')

    for col in ['access_time', 'egress_time', 'transfer_walk_time']:
        df[col + '_min'] = df[col] / 60
    df['in_vehicle_time_min'] = df['in_vehicle_time'] / 60
    df['total_ivt_min'] = df['in_vehicle_time_min']
    df['ln_access'] = np.log1p(df['access_time_min'])
    df['ln_egress'] = np.log1p(df['egress_time_min'])
    df['fare_1000won'] = df['fare'] / 1000
    df['od_distance_km'] = df.get('od_distance', 0) / 1000 if 'od_distance' in df.columns else 0

    # ext access (rename for clarity)
    df['ln_ext_access'] = df['ln_eb_access']
    df['ln_ext_egress'] = df['ln_eb_egress']

    # n_routes
    gtfs = pd.read_csv(EB_DIR / 'otp_gtfs_stop_matching.csv')
    s2r = dict(zip(gtfs['otp_stop_id'].astype(str), gtfs['n_routes']))
    df['ln_o_n_routes'] = np.log1p(df['o_stop'].map(s2r).fillna(1))
    df['ln_d_n_routes'] = np.log1p(df['d_stop'].map(s2r).fillna(1))

    if 'choice_set_size' not in df.columns:
        df['choice_set_size'] = df.groupby('h3_od')['h3_od'].transform('size')

    # context features 확인
    for cf in CONTEXT_FEATURES:
        if cf not in df.columns:
            df[cf] = 0

    return df


def split_data(df, od_col='h3_od'):
    """H3 OD 단위 split"""
    ods = np.array(df[od_col].unique())
    tr, te = train_test_split(ods, test_size=0.2, random_state=42)
    return df[df[od_col].isin(set(tr))].copy(), df[df[od_col].isin(set(te))].copy()


def create_datasets(train_df, test_df, features, od_col='h3_od', max_alts=10):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # rename h3_od → od_pair for RouteChoiceDataset
    if od_col != 'od_pair':
        if 'od_pair' in train_df.columns:
            train_df = train_df.drop(columns='od_pair')
            test_df = test_df.drop(columns='od_pair')
        train_df = train_df.rename(columns={od_col: 'od_pair'})
        test_df = test_df.rename(columns={od_col: 'od_pair'})

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Trim to max_alts
    for label, frame in [('train', train_df), ('test', test_df)]:
        rank = frame.groupby('od_pair')['choice_prob'].rank(method='first', ascending=False)
        keep = rank <= max_alts
        if label == 'train':
            train_df = frame[keep].copy()
        else:
            test_df = frame[keep].copy()

    for frame in [train_df, test_df]:
        sums = frame.groupby('od_pair')['choice_prob'].transform('sum')
        frame['choice_prob'] = frame['choice_prob'] / sums.replace(0, 1)

    orig = data_module.MAX_ALTS
    data_module.MAX_ALTS = max_alts

    train_ds = RouteChoiceDataset(train_df, fit_scaler=True, features=features)
    test_ds = RouteChoiceDataset(test_df, scaler=train_ds.scaler,
                                  context_scaler=train_ds.context_scaler, features=features)

    data_module.MAX_ALTS = orig
    return train_ds, test_ds


def run_spec(name, train_ds, test_ds, n_features):
    set_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
    n_context = len(CONTEXT_FEATURES)
    model = TasteNetModel(n_features, n_context).to(DEVICE)
    history = train_model(model, train_loader, test_loader, train_ds, test_ds,
                          lr=LR, epochs=EPOCHS, patience=15, device=DEVICE, verbose=False)
    result = evaluate_model(model, test_loader, test_ds, device=DEVICE)
    metrics = result[0]

    # Mean beta
    model.eval()
    all_betas = []
    with torch.no_grad():
        for X, z, y, mask, w in test_loader:
            z = z.to(DEVICE)
            beta = model.taste_net(z)
            all_betas.append(beta.cpu().numpy())
    mean_beta = np.concatenate(all_betas, axis=0).mean(axis=0)

    return metrics, mean_beta


def main():
    print(f"TasteNet H3 + ext + n_routes (device: {DEVICE})")
    print("=" * 70)

    df = load_h3_data()
    train_df, test_df = split_data(df)
    print(f"Train: {train_df['h3_od'].nunique():,} H3 ODs")
    print(f"Test:  {test_df['h3_od'].nunique():,} H3 ODs")

    SPECS = {
        'A. H3 + OTP walk (기존 9)': [
            'total_ivt_min', 'ln_access', 'ln_egress', 'transfer_walk_time_min',
            'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx',
        ],
        'B. H3 + OTP + ext + routes (13)': [
            'total_ivt_min', 'ln_access', 'ln_egress',
            'ln_ext_access', 'ln_ext_egress', 'ln_o_n_routes', 'ln_d_n_routes',
            'transfer_walk_time_min', 'num_transfers', 'fare_1000won',
            'has_bus', 'has_train', 'has_gtx',
        ],
        'C. H3 + ext + routes (OTP walk 제거, 10)': [
            'total_ivt_min',
            'ln_ext_access', 'ln_ext_egress', 'ln_o_n_routes', 'ln_d_n_routes',
            'transfer_walk_time_min', 'num_transfers', 'fare_1000won',
            'has_bus', 'has_train', 'has_gtx',
        ],
    }

    results = []
    for name, features in SPECS.items():
        print(f"\n{'=' * 70}")
        print(f"  {name}")
        print(f"{'=' * 70}")

        train_ds, test_ds = create_datasets(train_df, test_df, features)
        metrics, mean_beta = run_spec(name, train_ds, test_ds, len(features))

        print(f"\n  Test Results:")
        print(f"    rho2:  {metrics['rho_sq']:.4f}")
        print(f"    Top-1: {metrics['fpr_top1']:.4f}")

        print(f"\n  Mean β (TasteNet):")
        for f, b in zip(features, mean_beta):
            print(f"    {f:<25}: {b:>+.4f}")

        results.append({
            'name': name, 'k': len(features),
            'rho2': metrics['rho_sq'],
            'top1': metrics['fpr_top1'],
            'mean_beta': dict(zip(features, mean_beta.tolist())),
        })

    # Summary
    print(f"\n\n{'=' * 70}")
    print("  요약")
    print(f"{'=' * 70}")
    print(f"{'Spec':<45} {'K':>3} {'ρ²':>8} {'Top-1':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<45} {r['k']:>3} {r['rho2']:>8.4f} {r['top1']:>8.4f}")


if __name__ == '__main__':
    main()
