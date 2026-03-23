"""Walk/IVT 비율 실험 Part 2: 비선형 walk + 모드별 IVT.

발견사항:
- access 중앙값 = 0.12분(7초) → 대부분 정류장 바로 옆
- walk는 사실상 이산적 근접성 지표로 작동
- 비선형(log, piecewise, dummy) 사양 시도
"""
import sqlite3
import pickle
import pandas as pd
import numpy as np
from scipy.optimize import minimize, approx_fprime
from sklearn.model_selection import train_test_split
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / 'data' / 'training_set'

# ============================================================
# 1. 데이터 로드 + 모드별 IVT + 비선형 walk 피처
# ============================================================
print('Loading data...')
t0 = time.time()
df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')

# 모드별 IVT 추가 — training data에 있는 OD만 처리
if 'alt_idx' not in df.columns:
    df['alt_idx'] = df.groupby('od_pair').cumcount()
training_ods = set(df['od_pair'].unique())

conn = sqlite3.connect(str(DATA_DIR / 'otp_cache.db'))
cur = conn.execute('SELECT od_pair, data FROM otp_cache')
ivt_records = []
for od_pair, blob in cur:
    if od_pair not in training_ods:
        continue
    data = pickle.loads(blob)
    for i, p in enumerate(data['otp_parsed']):
        bus_ivt = train_ivt = 0.0
        for leg in p.get('transit_legs', []):
            if leg.get('mode', '') == 'BUS':
                bus_ivt += float(leg.get('duration', 0))
            else:
                train_ivt += float(leg.get('duration', 0))
        ivt_records.append((od_pair, i, bus_ivt / 60, train_ivt / 60))
conn.close()

ivt_df = pd.DataFrame(ivt_records, columns=['od_pair', 'alt_idx', 'bus_ivt_min', 'train_ivt_min'])
df = df.merge(ivt_df, on=['od_pair', 'alt_idx'], how='left')
df['bus_ivt_min'] = df['bus_ivt_min'].fillna(0)
df['train_ivt_min'] = df['train_ivt_min'].fillna(0)
del ivt_records, ivt_df

# 기본 전처리
for col in ['in_vehicle_time', 'access_time', 'egress_time', 'transfer_walk_time']:
    df[col + '_min'] = df[col] / 60
df['fare_1000won'] = df['fare'] / 1000

# 비선형 walk 피처
df['ln_access'] = np.log1p(df['access_time_min'])
df['ln_egress'] = np.log1p(df['egress_time_min'])
df['ln_trwalk'] = np.log1p(df['transfer_walk_time_min'])
df['wait_time_min'] = df['wait_time'] / 60
df['ln_wait'] = np.log1p(df['wait_time_min'])

# piecewise (breakpoint at 2 min)
df['access_over2'] = np.maximum(0, df['access_time_min'] - 2)
df['egress_over2'] = np.maximum(0, df['egress_time_min'] - 2)

# dummy (far walk > 2 min)
df['access_far'] = (df['access_time_min'] > 2).astype(float)
df['egress_far'] = (df['egress_time_min'] > 2).astype(float)

print(f'Data loaded ({time.time()-t0:.1f}s)')

# Split
od_dom = df.loc[df.groupby('od_pair')['choice_prob'].idxmax(),
                ['od_pair', 'transport_category']].set_index('od_pair')['transport_category']
od_strat = od_dom.map(lambda c: 'gtx_related' if 'gtx' in c else c)
tr_ods, te_ods = train_test_split(
    od_strat.index.to_numpy(), test_size=0.2, random_state=42, stratify=od_strat.values)
tr_set, te_set = set(tr_ods), set(te_ods)
for od in ['9007_9008', '9008_9007']:
    tr_set.discard(od); te_set.add(od)
train = df[df['od_pair'].isin(tr_set)].copy()
test = df[df['od_pair'].isin(te_set)].copy()
print(f'Train: {len(train):,}, Test: {len(test):,}')

# ============================================================
# 2. MNL engine
# ============================================================
def prepare_flat(data, features):
    X_l, y_l, w_l, g_l = [], [], [], []
    gid = 0
    for _, grp in data.groupby('od_pair'):
        y = grp['choice_prob'].values.astype(np.float64)
        if abs(y.sum() - 1.0) > 0.01:
            continue
        X_l.append(grp[features].values.astype(np.float64))
        y_l.append(y)
        n = len(grp)
        w_l.append(np.full(n, float(grp['n_total'].iloc[0])))
        g_l.append(np.full(n, gid, dtype=np.int64))
        gid += 1
    return dict(X=np.vstack(X_l), y=np.concatenate(y_l), w=np.concatenate(w_l),
                gid=np.concatenate(g_l), n_groups=gid)

def neg_ll(beta, fl):
    X, y, w, gid, ng = fl['X'], fl['y'], fl['w'], fl['gid'], fl['n_groups']
    V = X @ beta
    Vm = np.full(ng, -np.inf); np.maximum.at(Vm, gid, V)
    Vs = V - Vm[gid]; eV = np.exp(Vs)
    se = np.bincount(gid, weights=eV, minlength=ng)
    return -np.sum(w * y * (Vs - np.log(se[gid])))

def grad_fn(beta, fl):
    X, y, w, gid, ng = fl['X'], fl['y'], fl['w'], fl['gid'], fl['n_groups']
    V = X @ beta
    Vm = np.full(ng, -np.inf); np.maximum.at(Vm, gid, V)
    eV = np.exp(V - Vm[gid])
    se = np.bincount(gid, weights=eV, minlength=ng)
    return X.T @ (w * (eV / se[gid] - y))

def run_model(name, features, sign_neg):
    print(f'\n--- {name} ---')
    bounds = [(None, 0) if f in sign_neg else (None, None) for f in features]
    tr_fl = prepare_flat(train, features)
    te_fl = prepare_flat(test, features)

    t = time.time()
    res = minimize(neg_ll, np.zeros(len(features)), args=(tr_fl,), jac=grad_fn,
                   method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
    beta = res.x

    gid, ng = tr_fl['gid'], tr_fl['n_groups']
    gs = np.bincount(gid, minlength=ng)
    wpg = np.bincount(gid, weights=tr_fl['w'], minlength=ng) / gs
    ll0 = -np.sum(wpg * np.log(gs))
    ll_tr = -res.fun
    rho_tr = 1 - ll_tr / ll0

    ll_te = -neg_ll(beta, te_fl)
    gid_t, ng_t = te_fl['gid'], te_fl['n_groups']
    gs_t = np.bincount(gid_t, minlength=ng_t)
    wpg_t = np.bincount(gid_t, weights=te_fl['w'], minlength=ng_t) / gs_t
    ll0_te = -np.sum(wpg_t * np.log(gs_t))
    rho_te = 1 - ll_te / ll0_te

    V = te_fl['X'] @ beta
    Vm = np.full(ng_t, -np.inf); np.maximum.at(Vm, gid_t, V)
    eV = np.exp(V - Vm[gid_t])
    se_v = np.bincount(gid_t, weights=eV, minlength=ng_t)
    pred = eV / se_v[gid_t]
    actual = te_fl['y']
    top1 = sum(np.argmax(pred[gid_t == g]) == np.argmax(actual[gid_t == g])
               for g in range(ng_t)) / ng_t
    rmse = np.sqrt(np.mean((pred - actual) ** 2))

    n_p = len(beta)
    hess = np.zeros((n_p, n_p))
    for i in range(n_p):
        def _gi(b, _i=i):
            return grad_fn(b, tr_fl)[_i]
        hess[i, :] = approx_fprime(beta, _gi, 1e-5)
    hess = (hess + hess.T) / 2
    try:
        se_arr = np.sqrt(np.abs(np.diag(np.linalg.inv(hess))))
    except np.linalg.LinAlgError:
        se_arr = np.sqrt(np.abs(np.diag(np.linalg.pinv(hess))))
    t_stats = beta / np.where(se_arr > 0, se_arr, 1)

    on_bound = [f for f, b in zip(features, beta) if f in sign_neg and abs(b) < 1e-10]

    print(f'  rho_tr={rho_tr:.6f}  rho_te={rho_te:.6f}  top1={top1 * 100:.2f}%  rmse={rmse:.6f}')
    print(f'  on_bound: {on_bound}')
    for f, b, t in zip(features, beta, t_stats):
        star = ' *' if f in sign_neg and abs(b) < 1e-10 else ''
        print(f'    {f:<25} {b:>10.6f} (t={t:>7.1f}){star}')

    return dict(name=name, rho_tr=rho_tr, rho_te=rho_te, top1=top1, rmse=rmse,
                ll_tr=ll_tr, on_bound=on_bound)


# ============================================================
# 3. 모델 비교
# ============================================================
print('\n' + '=' * 70)
print('Model estimation')
print('=' * 70)

SN = lambda *fs: set(fs)

# H: Baseline (mode IVT, no dist, no wait) - from part 1
H = run_model('H: Mode IVT linear walk',
    ['bus_ivt_min', 'train_ivt_min', 'access_time_min', 'egress_time_min',
     'transfer_walk_time_min', 'num_transfers', 'fare_1000won',
     'has_bus', 'has_train', 'has_gtx'],
    SN('bus_ivt_min', 'train_ivt_min', 'access_time_min', 'egress_time_min',
       'transfer_walk_time_min', 'num_transfers', 'fare_1000won'))

# J1: log(walk) — all 3 walk components
J1 = run_model('J1: Mode IVT + log(walk)',
    ['bus_ivt_min', 'train_ivt_min', 'ln_access', 'ln_egress', 'ln_trwalk',
     'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx'],
    SN('bus_ivt_min', 'train_ivt_min', 'ln_access', 'ln_egress', 'ln_trwalk',
       'num_transfers', 'fare_1000won'))

# K1: log(acc+egr) + linear transfer_walk (ln_trwalk was bounded in J1)
K1 = run_model('K1: log(acc+egr) + lin trwalk',
    ['bus_ivt_min', 'train_ivt_min', 'ln_access', 'ln_egress',
     'transfer_walk_time_min', 'num_transfers', 'fare_1000won',
     'has_bus', 'has_train', 'has_gtx'],
    SN('bus_ivt_min', 'train_ivt_min', 'ln_access', 'ln_egress',
       'transfer_walk_time_min', 'num_transfers', 'fare_1000won'))

# K2: K1 + log(wait)
K2 = run_model('K2: K1 + log(wait)',
    ['bus_ivt_min', 'train_ivt_min', 'ln_access', 'ln_egress',
     'transfer_walk_time_min', 'ln_wait', 'num_transfers', 'fare_1000won',
     'has_bus', 'has_train', 'has_gtx'],
    SN('bus_ivt_min', 'train_ivt_min', 'ln_access', 'ln_egress',
       'transfer_walk_time_min', 'ln_wait', 'num_transfers', 'fare_1000won'))

# K3: all log(walk+wait)
K3 = run_model('K3: all log(walk+wait)',
    ['bus_ivt_min', 'train_ivt_min', 'ln_access', 'ln_egress',
     'ln_trwalk', 'ln_wait', 'num_transfers', 'fare_1000won',
     'has_bus', 'has_train', 'has_gtx'],
    SN('bus_ivt_min', 'train_ivt_min', 'ln_access', 'ln_egress',
       'ln_trwalk', 'ln_wait', 'num_transfers', 'fare_1000won'))

# ============================================================
# 4. 요약
# ============================================================
models = [H, J1, K1, K2, K3]
print('\n' + '=' * 95)
print('Summary')
print('=' * 95)
print(f'{"Model":<40} {"rho_tr":>10} {"rho_te":>10} {"top1":>10} {"rmse":>10} {"bound":>20}')
print('-' * 95)
for m in models:
    bnd = ','.join(m['on_bound']) if m['on_bound'] else 'none'
    print(f'{m["name"]:<40} {m["rho_tr"]:>10.6f} {m["rho_te"]:>10.6f} '
          f'{m["top1"]*100:>9.2f}% {m["rmse"]:>10.6f} {bnd:>20}')
