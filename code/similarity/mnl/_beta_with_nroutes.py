# -*- coding: utf-8 -*-
"""H3 MNL: ext access + n_routes 비교"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pickle, sqlite3
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize, approx_fprime
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / 'data' / 'training_set'
EB_DIR = ROOT / 'data' / 'eb'

# Load H3 data
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

# n_routes
gtfs = pd.read_csv(EB_DIR / 'otp_gtfs_stop_matching.csv')
s2r = dict(zip(gtfs['otp_stop_id'].astype(str), gtfs['n_routes']))
df['ln_o_routes'] = np.log1p(df['o_stop'].map(s2r).fillna(1))
df['ln_d_routes'] = np.log1p(df['d_stop'].map(s2r).fillna(1))

print(f"ln_o_routes: mean={df['ln_o_routes'].mean():.2f}, std={df['ln_o_routes'].std():.2f}")
var_routes = df.groupby('h3_od')['ln_o_routes'].std().fillna(0)
print(f"H3 OD 내 ln_o_routes 변이 있는 비율: {(var_routes > 0).mean() * 100:.1f}%")

# Split
h3_ods = np.array(df['h3_od'].unique())
h3_train, _ = train_test_split(h3_ods, test_size=0.2, random_state=42)
train_df = df[df['h3_od'].isin(set(h3_train))].copy()
h3_n = df.groupby('h3_od')['n_matched'].sum().rename('h3_n_total')
train_df = train_df.merge(h3_n, on='h3_od', how='left')


def prepare_flat(data, features):
    X_list, y_list, w_list, gid_list = [], [], [], []
    gid = 0
    for _, grp in data.groupby('h3_od'):
        y = grp['choice_prob'].values.astype(np.float64)
        if abs(y.sum() - 1.0) > 0.01 or y.sum() == 0:
            continue
        X_list.append(grp[features].values.astype(np.float64))
        y_list.append(y)
        w = float(grp['h3_n_total'].iloc[0])
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


specs = [
    ('A. H3 + OTP + ext (기존)',
     ['total_ivt_min', 'ln_access', 'ln_egress', 'ln_access_ext', 'ln_egress_ext',
      'transfer_walk_time_min', 'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx'],
     ['Total IVT', 'ln(OTP access)', 'ln(OTP egress)', 'ln(ext access)', 'ln(ext egress)',
      'Xfer walk', 'Transfers', 'Fare', 'Has bus', 'Has train', 'Has GTX']),

    ('B. H3 + OTP + ext + n_routes',
     ['total_ivt_min', 'ln_access', 'ln_egress', 'ln_access_ext', 'ln_egress_ext',
      'ln_o_routes', 'ln_d_routes',
      'transfer_walk_time_min', 'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx'],
     ['Total IVT', 'ln(OTP access)', 'ln(OTP egress)', 'ln(ext access)', 'ln(ext egress)',
      'ln(O n_routes)', 'ln(D n_routes)',
      'Xfer walk', 'Transfers', 'Fare', 'Has bus', 'Has train', 'Has GTX']),

    ('C. H3 + OTP + n_routes (ext 제거)',
     ['total_ivt_min', 'ln_access', 'ln_egress',
      'ln_o_routes', 'ln_d_routes',
      'transfer_walk_time_min', 'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx'],
     ['Total IVT', 'ln(OTP access)', 'ln(OTP egress)',
      'ln(O n_routes)', 'ln(D n_routes)',
      'Xfer walk', 'Transfers', 'Fare', 'Has bus', 'Has train', 'Has GTX']),

    ('D. H3 + ext + n_routes (OTP walk 제거)',
     ['total_ivt_min', 'ln_access_ext', 'ln_egress_ext',
      'ln_o_routes', 'ln_d_routes',
      'transfer_walk_time_min', 'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx'],
     ['Total IVT', 'ln(ext access)', 'ln(ext egress)',
      'ln(O n_routes)', 'ln(D n_routes)',
      'Xfer walk', 'Transfers', 'Fare', 'Has bus', 'Has train', 'Has GTX']),
]

for name, features, labels in specs:
    flat = prepare_flat(train_df, features)

    res = minimize(neg_ll, np.zeros(len(features)), args=(flat,), jac=grad_fn,
                   method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})

    beta_hat = res.x
    k = len(features)
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

    gid_arr = flat['gid']
    w_arr = flat['w']
    ng = flat['n_groups']
    gs = np.bincount(gid_arr, minlength=ng)
    wg = np.bincount(gid_arr, weights=w_arr, minlength=ng) / gs
    ll0 = -np.sum(wg * np.log(gs))
    rho = 1 + res.fun / ll0

    print(f'\n{"=" * 85}')
    print(f'  {name}   [rho2={rho:.4f}]')
    print(f'{"=" * 85}')
    print(f'  {"Feature":<22} {"beta":>12} {"SE":>12} {"t-stat":>10} {"sig":>5}')
    print(f'  {"-" * 63}')
    for l, b, s, t in zip(labels, beta_hat, se, t_stat):
        sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.645 else ''
        print(f'  {l:<22} {b:>+12.6f} {s:>12.6f} {t:>10.2f} {sig}')
