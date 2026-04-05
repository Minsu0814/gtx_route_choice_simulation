# -*- coding: utf-8 -*-
"""Threshold 0.5/0.7/0.8/0.9 비교 MNL"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import glob, os, gc
import numpy as np, pandas as pd
from collections import defaultdict
from scipy.optimize import minimize, approx_fprime
from sklearn.model_selection import train_test_split

ckpt_dir = 'data/training_set_h3/checkpoints'
files = sorted(glob.glob(os.path.join(ckpt_dir, 'day_*.parquet')))
print(f'체크포인트: {len(files)}개')

ROUTE_FEATURES = [
    'total_duration', 'in_vehicle_time', 'walk_time', 'wait_time',
    'access_time', 'egress_time', 'transfer_walk_time',
    'walk_distance', 'total_distance', 'bus_distance', 'subway_distance', 'gtx_distance',
    'num_transfers', 'num_legs', 'fare', 'generalized_cost',
    'transport_category', 'has_bus', 'has_train', 'has_gtx', 'main_route', 'bus_subtype',
]
FEATURES = ['ivt_min', 'ln_access', 'ln_egress', 'transfer_walk_time_min',
            'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx']
LABELS = ['IVT(min)', 'ln(access)', 'ln(egress)', 'Xfer walk(min)',
          'Transfers', 'Fare(1000won)', 'Has bus', 'Has train', 'Has GTX']
SIGN_NEG = {'ivt_min', 'ln_access', 'ln_egress', 'transfer_walk_time_min',
            'num_transfers', 'fare_1000won'}


def prepare_flat(data, features, od_col='h3_od'):
    X_list, y_list, w_list, gid_list = [], [], [], []
    gid = 0
    for _, grp in data.groupby(od_col):
        y = grp['choice_prob'].values.astype(np.float64)
        if abs(y.sum() - 1.0) > 0.01 or y.sum() == 0:
            continue
        X_list.append(grp[features].values.astype(np.float64))
        y_list.append(y)
        w = float(grp['n_total'].iloc[0])
        w_list.append(np.full(len(grp), w))
        gid_list.append(np.full(len(grp), gid, dtype=np.int64))
        gid += 1
    return {
        'X': np.vstack(X_list), 'y': np.concatenate(y_list),
        'w': np.concatenate(w_list), 'gid': np.concatenate(gid_list),
        'n_groups': gid,
    }


def neg_ll(beta, flat):
    X, y, w, gid, ng = flat['X'], flat['y'], flat['w'], flat['gid'], flat['n_groups']
    V = X @ beta
    V_max = np.full(ng, -np.inf)
    np.maximum.at(V_max, gid, V)
    exp_V = np.exp(V - V_max[gid])
    sum_exp = np.bincount(gid, weights=exp_V, minlength=ng)
    return -np.sum(w * y * (V - V_max[gid] - np.log(sum_exp[gid])))


def grad_fn(beta, flat):
    X, y, w, gid, ng = flat['X'], flat['y'], flat['w'], flat['gid'], flat['n_groups']
    V = X @ beta
    V_max = np.full(ng, -np.inf)
    np.maximum.at(V_max, gid, V)
    exp_V = np.exp(V - V_max[gid])
    sum_exp = np.bincount(gid, weights=exp_V, minlength=ng)
    return X.T @ (w * (exp_V / sum_exp[gid] - y))


def evaluate(beta, data, features, od_col='h3_od'):
    c1 = c3 = tot = 0
    for _, grp in data.groupby(od_col):
        y = grp['choice_prob'].values
        if y.sum() == 0:
            continue
        V = grp[features].values.astype(np.float64) @ beta
        pred = np.argsort(-V)
        actual = np.argmax(y)
        if pred[0] == actual:
            c1 += 1
        if actual in pred[:3]:
            c3 += 1
        tot += 1
    return c1 / tot, c3 / tot, tot


summary = []

for threshold in [0.5, 0.7, 0.8, 0.9]:
    print(f'\n{"=" * 80}')
    print(f'  Threshold = {threshold}')
    print(f'{"=" * 80}')

    agg = defaultdict(lambda: {'n_matched': 0, 'n_total': 0})
    features_cache = {}
    total_trips = 0

    for fi, f in enumerate(files):
        df = pd.read_parquet(f)

        # threshold: trip별 best sim >= threshold
        best_per_trip = df.groupby('trip_id')['sim_composite'].max()
        valid_trips = best_per_trip[best_per_trip >= threshold].index
        df = df[df['trip_id'].isin(valid_trips)]

        if len(df) == 0:
            del df
            continue

        total_trips += df['trip_id'].nunique()

        trip_chosen = df.groupby(['h3_od', 'alt_idx', 'trip_id'])['chosen'].max()
        od_alt_chosen = trip_chosen.groupby(level=[0, 1]).sum()
        od_alt_total = trip_chosen.groupby(level=[0, 1]).count()

        for (h3_od, alt_idx), n_m in od_alt_chosen.items():
            key = (h3_od, alt_idx)
            agg[key]['n_matched'] += n_m
            agg[key]['n_total'] += od_alt_total[(h3_od, alt_idx)]

        for (h3_od, alt_idx), grp in df.groupby(['h3_od', 'alt_idx']):
            key = (h3_od, alt_idx)
            if key not in features_cache:
                row = grp.iloc[0]
                feat = {col: row[col] for col in ROUTE_FEATURES if col in df.columns}
                feat['choice_set_size'] = row['choice_set_size']
                feat['od_distance'] = row.get('od_distance', 0)
                features_cache[key] = feat

        del df
        gc.collect()

    # DataFrame
    records = []
    for (h3_od, alt_idx), stats in agg.items():
        feat = features_cache.get((h3_od, alt_idx), {})
        records.append({
            'h3_od': h3_od, 'alt_idx': alt_idx, **feat,
            'n_matched': stats['n_matched'], 'n_total': stats['n_total'],
        })

    result = pd.DataFrame(records)
    od_totals = result.groupby('h3_od')['n_matched'].sum().rename('od_total')
    result = result.merge(od_totals, on='h3_od')
    result['choice_prob'] = result['n_matched'] / result['od_total'].replace(0, 1)
    result['n_total'] = result.groupby('h3_od')['n_total'].transform('max')

    n_ods = result['h3_od'].nunique()
    avg_alts = result.groupby('h3_od').size().mean()

    # access 변이
    var_acc = result.groupby('h3_od')['access_time'].std().fillna(0)
    pct_vary = (var_acc > 0).mean() * 100
    chosen_rows = result[result['choice_prob'] > 0]
    unchosen_rows = result[result['choice_prob'] == 0]
    acc_diff = chosen_rows['access_time'].mean() - unchosen_rows['access_time'].mean()

    print(f'  통행: {total_trips:,}, H3 ODs: {n_ods:,}, 평균 대안: {avg_alts:.1f}')
    print(f'  access 변이: {pct_vary:.1f}%, 선택-비선택: {acc_diff:.0f}s')

    # 피처
    for c in ['access_time', 'egress_time', 'transfer_walk_time']:
        result[c + '_min'] = result[c] / 60
    result['ln_access'] = np.log1p(result['access_time_min'])
    result['ln_egress'] = np.log1p(result['egress_time_min'])
    result['fare_1000won'] = result['fare'] / 1000
    result['ivt_min'] = result['in_vehicle_time'] / 60

    # Split
    h3_ods_arr = np.array(result['h3_od'].unique())
    tr, te = train_test_split(h3_ods_arr, test_size=0.2, random_state=42)
    train_df = result[result['h3_od'].isin(set(tr))]
    test_df = result[result['h3_od'].isin(set(te))]

    flat = prepare_flat(train_df, FEATURES)
    print(f'  유효 그룹: {flat["n_groups"]:,}')

    # Constrained
    bounds = [(None, 0) if f in SIGN_NEG else (None, None) for f in FEATURES]
    res = minimize(neg_ll, np.zeros(len(FEATURES)), args=(flat,), jac=grad_fn,
                   method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})

    # Unconstrained
    res_free = minimize(neg_ll, np.zeros(len(FEATURES)), args=(flat,), jac=grad_fn,
                        method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})

    # rho²
    gid_arr, w_arr, ng = flat['gid'], flat['w'], flat['n_groups']
    gs = np.bincount(gid_arr, minlength=ng)
    wg = np.bincount(gid_arr, weights=w_arr, minlength=ng) / gs
    ll0 = -np.sum(wg * np.log(gs))
    rho = 1 + res.fun / ll0

    # t-stat
    beta_hat = res.x
    k = len(FEATURES)
    hess = np.zeros((k, k))
    for i in range(k):
        def gi(b, _i=i):
            return grad_fn(b, flat)[_i]
        hess[i, :] = approx_fprime(beta_hat, gi, 1e-5)
    hess = (hess + hess.T) / 2
    try:
        se = np.sqrt(np.abs(np.diag(np.linalg.inv(hess))))
    except np.linalg.LinAlgError:
        se = np.sqrt(np.abs(np.diag(np.linalg.pinv(hess))))
    t_stat = beta_hat / np.where(se > 0, se, 1)

    # Evaluate
    t1_tr, t3_tr, _ = evaluate(res.x, train_df, FEATURES)
    t1_te, t3_te, n_te = evaluate(res.x, test_df, FEATURES)

    print(f'\n  {"Feature":<20} {"b(free)":>10} {"b(con)":>10} {"t-stat":>10}')
    print(f'  {"-" * 52}')
    for l, bf, bc, t in zip(LABELS, res_free.x, res.x, t_stat):
        print(f'  {l:<20} {bf:>+10.4f} {bc:>+10.4f} {t:>10.1f}')
    print(f'\n  rho2={rho:.4f}  Top-1={t1_te:.4f}  Top-3={t3_te:.4f}  (N={n_te:,})')

    summary.append({
        'threshold': threshold,
        'trips': total_trips,
        'h3_ods': n_ods,
        'avg_alts': avg_alts,
        'rho2': rho,
        'top1': t1_te,
        'top3': t3_te,
        'b_access': res.x[1],
        'b_egress': res.x[2],
        'b_ivt': res.x[0],
        'b_transfers': res.x[4],
        'acc_diff': acc_diff,
    })

    del agg, features_cache, result, records
    gc.collect()

# 요약
print(f'\n\n{"=" * 90}')
print(f'  요약 비교')
print(f'{"=" * 90}')
print(f'{"Thr":>5} {"Trips":>12} {"H3 ODs":>10} {"Alts":>5} {"rho2":>8} {"Top-1":>8} {"Top-3":>8} {"b_access":>10} {"b_egress":>10} {"acc_diff":>10}')
print(f'{"-" * 90}')
for s in summary:
    print(f'{s["threshold"]:>5.1f} {s["trips"]:>12,} {s["h3_ods"]:>10,} {s["avg_alts"]:>5.1f} '
          f'{s["rho2"]:>8.4f} {s["top1"]:>8.4f} {s["top3"]:>8.4f} '
          f'{s["b_access"]:>+10.4f} {s["b_egress"]:>+10.4f} {s["acc_diff"]:>10.0f}s')
