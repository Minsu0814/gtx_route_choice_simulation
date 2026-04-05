# -*- coding: utf-8 -*-
"""4사양 β 비교 (비제약 vs 제약)"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pickle, sqlite3
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize, approx_fprime
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / 'data' / 'training_set'
EB_DIR = ROOT / 'data' / 'eb'

# === Load H3 data ===
df = pd.read_parquet(EB_DIR / 'h3_choice_prob.parquet')

training_ods = set(df['od_pair'].unique())
conn = sqlite3.connect(str(DATA_DIR / 'otp_cache.db'))
ivt_records = []
for od_pair, blob in conn.execute('SELECT od_pair, data FROM otp_cache'):
    if od_pair not in training_ods:
        continue
    data = pickle.loads(blob)
    for i, p in enumerate(data['otp_parsed']):
        bus_ivt = train_ivt = gtx_ivt = 0.0
        for leg in p.get('transit_legs', []):
            mode = leg.get('mode', '')
            dur = float(leg.get('duration', 0))
            if mode == 'BUS':
                bus_ivt += dur
            elif mode == 'GTX':
                gtx_ivt += dur
            else:
                train_ivt += dur
        ivt_records.append((od_pair, i, bus_ivt / 60, train_ivt / 60, gtx_ivt / 60))
conn.close()

ivt_df = pd.DataFrame(ivt_records, columns=['od_pair', 'alt_idx', 'bus_ivt_min', 'train_ivt_min', 'gtx_ivt_min'])
df = df.merge(ivt_df, on=['od_pair', 'alt_idx'], how='left').fillna(0)

for c in ['access_time', 'egress_time', 'transfer_walk_time']:
    df[c + '_min'] = df[c] / 60
df['ln_access'] = np.log1p(df['access_time_min'])
df['ln_egress'] = np.log1p(df['egress_time_min'])
df['fare_1000won'] = df['fare'] / 1000
df['total_ivt_min'] = df['bus_ivt_min'] + df['train_ivt_min'] + df['gtx_ivt_min']
df['ln_access_ext'] = df['ln_eb_access']
df['ln_egress_ext'] = df['ln_eb_egress']

# H3 split
h3_ods = np.array(df['h3_od'].unique())
h3_train, h3_test = train_test_split(h3_ods, test_size=0.2, random_state=42)
train_df = df[df['h3_od'].isin(set(h3_train))].copy()
h3_n = df.groupby('h3_od')['n_matched'].sum().rename('h3_n_total')
train_df = train_df.merge(h3_n, on='h3_od', how='left')

# === Load Stop OD data ===
stop_df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')
stop_df = stop_df.merge(ivt_df, on=['od_pair', 'alt_idx'], how='left').fillna(0)
for c in ['access_time', 'egress_time', 'transfer_walk_time']:
    stop_df[c + '_min'] = stop_df[c] / 60
stop_df['ln_access'] = np.log1p(stop_df['access_time_min'])
stop_df['ln_egress'] = np.log1p(stop_df['egress_time_min'])
stop_df['fare_1000won'] = stop_df['fare'] / 1000
stop_df['total_ivt_min'] = stop_df['bus_ivt_min'] + stop_df['train_ivt_min'] + stop_df['gtx_ivt_min']
stop_ods = np.array(stop_df['od_pair'].unique())
s_train, s_test = train_test_split(stop_ods, test_size=0.2, random_state=42)
stop_train = stop_df[stop_df['od_pair'].isin(set(s_train))].copy()


# === MNL functions ===
def prepare_flat(data, features, od_col, weight_col):
    X_list, y_list, w_list, gid_list = [], [], [], []
    gid = 0
    for _, grp in data.groupby(od_col):
        y = grp['choice_prob'].values.astype(np.float64)
        if abs(y.sum() - 1.0) > 0.01 or y.sum() == 0:
            continue
        X_list.append(grp[features].values.astype(np.float64))
        y_list.append(y)
        w = float(grp[weight_col].iloc[0]) if weight_col in grp.columns else 1.0
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


def fit_and_report(flat, features, labels, sign_neg_set):
    """Fit unconstrained + constrained, return betas and t-stats."""
    k = len(features)

    # Unconstrained
    res_free = minimize(neg_ll, np.zeros(k), args=(flat,), jac=grad_fn,
                        method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})

    # Constrained
    bounds = [(None, 0) if f in sign_neg_set else (None, None) for f in features]
    res_con = minimize(neg_ll, np.zeros(k), args=(flat,), jac=grad_fn,
                       method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})

    # t-stat for constrained
    beta_hat = res_con.x
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

    # rho²
    gid_arr, w_arr, ng = flat['gid'], flat['w'], flat['n_groups']
    gs = np.bincount(gid_arr, minlength=ng)
    wg = np.bincount(gid_arr, weights=w_arr, minlength=ng) / gs
    ll0 = -np.sum(wg * np.log(gs))
    rho_free = 1 + res_free.fun / ll0
    rho_con = 1 + res_con.fun / ll0

    return res_free.x, res_con.x, se, t_stat, rho_free, rho_con


# === Specs ===
SIGN_NEG_BASE = {'total_ivt_min', 'ln_access', 'ln_egress',
                 'transfer_walk_time_min', 'num_transfers', 'fare_1000won'}

specs = [
    {
        'name': '1. K3 기존 (Stop OD)',
        'features': ['total_ivt_min', 'ln_access', 'ln_egress', 'transfer_walk_time_min',
                      'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx'],
        'labels': ['Total IVT(min)', 'ln(OTP access)', 'ln(OTP egress)', 'Transfer walk(min)',
                   'Transfers', 'Fare(1000won)', 'Has bus', 'Has train', 'Has GTX'],
        'od_col': 'od_pair', 'weight_col': 'n_total', 'data': 'stop',
    },
    {
        'name': '2. H3 + OTP walk',
        'features': ['total_ivt_min', 'ln_access', 'ln_egress', 'transfer_walk_time_min',
                      'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx'],
        'labels': ['Total IVT(min)', 'ln(OTP access)', 'ln(OTP egress)', 'Transfer walk(min)',
                   'Transfers', 'Fare(1000won)', 'Has bus', 'Has train', 'Has GTX'],
        'od_col': 'h3_od', 'weight_col': 'h3_n_total', 'data': 'h3',
    },
    {
        'name': '3. H3 + ext access/egress',
        'features': ['total_ivt_min', 'ln_access_ext', 'ln_egress_ext', 'transfer_walk_time_min',
                      'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx'],
        'labels': ['Total IVT(min)', 'ln(ext access)', 'ln(ext egress)', 'Transfer walk(min)',
                   'Transfers', 'Fare(1000won)', 'Has bus', 'Has train', 'Has GTX'],
        'od_col': 'h3_od', 'weight_col': 'h3_n_total', 'data': 'h3',
    },
    {
        'name': '4. H3 + OTP + ext',
        'features': ['total_ivt_min', 'ln_access', 'ln_egress', 'ln_access_ext', 'ln_egress_ext',
                      'transfer_walk_time_min', 'num_transfers', 'fare_1000won',
                      'has_bus', 'has_train', 'has_gtx'],
        'labels': ['Total IVT(min)', 'ln(OTP access)', 'ln(OTP egress)',
                   'ln(ext access)', 'ln(ext egress)',
                   'Transfer walk(min)', 'Transfers', 'Fare(1000won)',
                   'Has bus', 'Has train', 'Has GTX'],
        'od_col': 'h3_od', 'weight_col': 'h3_n_total', 'data': 'h3',
    },
]

for spec in specs:
    print(f'\n{"=" * 95}')
    print(f'  {spec["name"]}')
    print(f'{"=" * 95}')

    if spec['data'] == 'stop':
        flat = prepare_flat(stop_train, spec['features'], spec['od_col'], spec['weight_col'])
    else:
        flat = prepare_flat(train_df, spec['features'], spec['od_col'], spec['weight_col'])

    print(f'  Groups: {flat["n_groups"]:,}')

    b_free, b_con, se, t_stat, rho_free, rho_con = fit_and_report(
        flat, spec['features'], spec['labels'], SIGN_NEG_BASE
    )

    print(f'\n  {"Feature":<22} {"β(free)":>12} {"β(제약)":>12} {"SE":>12} {"t-stat":>10}')
    print(f'  {"-" * 70}')
    for i, (f, l) in enumerate(zip(spec['features'], spec['labels'])):
        on_bound = f in SIGN_NEG_BASE and abs(b_con[i]) < 1e-10
        se_str = f'{se[i]:>12.6f}' if not on_bound else f'{"(bound)":>12}'
        t_str = f'{t_stat[i]:>10.2f}' if not on_bound else f'{"":>10}'
        print(f'  {l:<22} {b_free[i]:>+12.6f} {b_con[i]:>+12.6f} {se_str} {t_str}')
    print(f'  {"-" * 70}')
    print(f'  ρ²(free): {rho_free:.4f}  |  ρ²(제약): {rho_con:.4f}')


if __name__ == '__main__':
    pass
