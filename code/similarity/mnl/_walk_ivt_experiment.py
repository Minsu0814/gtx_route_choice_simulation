"""Walk/IVT 비율 문제 해결 실험.

현재: access/IVT = 136x, egress/IVT = 83x (기대: 2-5x)
원인: IVT와 walk의 within-OD 음의 상관(-0.43) → IVT β 억제
해결: 다양한 시간 사양(specification) 비교
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize, approx_fprime
from sklearn.model_selection import train_test_split
from pathlib import Path
import time

DATA_DIR = Path(__file__).resolve().parents[3] / 'data' / 'training_set'

# ============================================================
# 1. 데이터 로드 + 전처리
# ============================================================
print('Loading data...')
df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')

for col in ['in_vehicle_time', 'wait_time', 'access_time', 'egress_time', 'transfer_walk_time']:
    df[col + '_min'] = df[col] / 60
df['total_distance_km'] = df['total_distance'] / 1000
df['fare_1000won'] = df['fare'] / 1000

# 새 피처
df['total_time_min'] = (df['in_vehicle_time'] + df['wait_time'] + df['access_time'] +
                         df['egress_time'] + df['transfer_walk_time']) / 60
df['walk_time_min'] = df['access_time_min'] + df['egress_time_min'] + df['transfer_walk_time_min']
df['ovt_min'] = df['walk_time_min'] + df['wait_time_min']
df['walk_share'] = df['walk_time_min'] / df['total_time_min'].clip(lower=0.1)
# 문헌 기반 일반화 시간: IVT + 2.5*walk + 1.5*wait
df['gen_time_lit'] = (df['in_vehicle_time_min'] + 2.5 * df['walk_time_min'] +
                       1.5 * df['wait_time_min'])

# Split
od_dom = df.loc[df.groupby('od_pair')['choice_prob'].idxmax(),
                ['od_pair', 'transport_category']].set_index('od_pair')['transport_category']
coarsen = lambda c: 'gtx_related' if 'gtx' in c else c
od_strat = od_dom.map(coarsen)
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

    t0 = time.time()
    res = minimize(neg_ll, np.zeros(len(features)), args=(tr_fl,), jac=grad_fn,
                   method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
    beta = res.x

    gid, ng = tr_fl['gid'], tr_fl['n_groups']
    gs = np.bincount(gid, minlength=ng)
    wpg = np.bincount(gid, weights=tr_fl['w'], minlength=ng) / gs
    ll0_tr = -np.sum(wpg * np.log(gs))
    ll_tr = -res.fun
    rho_tr = 1 - ll_tr / ll0_tr

    ll_te = -neg_ll(beta, te_fl)
    gid_t, ng_t = te_fl['gid'], te_fl['n_groups']
    gs_t = np.bincount(gid_t, minlength=ng_t)
    wpg_t = np.bincount(gid_t, weights=te_fl['w'], minlength=ng_t) / gs_t
    ll0_te = -np.sum(wpg_t * np.log(gs_t))
    rho_te = 1 - ll_te / ll0_te

    V = te_fl['X'] @ beta
    Vm = np.full(ng_t, -np.inf); np.maximum.at(Vm, gid_t, V)
    eV = np.exp(V - Vm[gid_t])
    se = np.bincount(gid_t, weights=eV, minlength=ng_t)
    pred = eV / se[gid_t]
    actual = te_fl['y']
    top1 = sum(np.argmax(pred[gid_t == g]) == np.argmax(actual[gid_t == g])
               for g in range(ng_t)) / ng_t
    rmse = np.sqrt(np.mean((pred - actual) ** 2))

    # SE & t-stat
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

    # Walk/IVT ratio if applicable
    if 'in_vehicle_time_min' in features:
        ivt_i = features.index('in_vehicle_time_min')
        ivt_b = beta[ivt_i]
        for wf in ['access_time_min', 'egress_time_min', 'transfer_walk_time_min',
                    'walk_time_min', 'ovt_min']:
            if wf in features:
                wi = features.index(wf)
                wb = beta[wi]
                ratio = abs(wb / ivt_b) if abs(ivt_b) > 1e-10 else float('inf')
                print(f'    {wf}/IVT ratio: {ratio:.1f}x')

    return dict(name=name, rho_tr=rho_tr, rho_te=rho_te, top1=top1, rmse=rmse,
                beta=beta, feats=features, t_stats=t_stats, on_bound=on_bound, ll_tr=ll_tr)


# ============================================================
# 3. 모델 비교
# ============================================================
print('\n' + '=' * 70)
print('Model estimation')
print('=' * 70)

SN = lambda *fs: set(fs)

# A: Baseline (현재)
A = run_model('A: Baseline (decomposed times)',
    ['in_vehicle_time_min', 'wait_time_min', 'access_time_min', 'egress_time_min',
     'transfer_walk_time_min', 'total_distance_km', 'num_transfers', 'fare_1000won',
     'has_bus', 'has_train', 'has_gtx'],
    SN('in_vehicle_time_min', 'wait_time_min', 'access_time_min', 'egress_time_min',
       'transfer_walk_time_min', 'total_distance_km', 'num_transfers', 'fare_1000won'))

# B: IVT + OVT (out-of-vehicle time combined)
B = run_model('B: IVT + OVT',
    ['in_vehicle_time_min', 'ovt_min', 'total_distance_km', 'num_transfers', 'fare_1000won',
     'has_bus', 'has_train', 'has_gtx'],
    SN('in_vehicle_time_min', 'ovt_min', 'total_distance_km', 'num_transfers', 'fare_1000won'))

# C: IVT + walk_time (access+egress+transfer walk combined)
C = run_model('C: IVT + WalkTime',
    ['in_vehicle_time_min', 'walk_time_min', 'wait_time_min', 'total_distance_km',
     'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx'],
    SN('in_vehicle_time_min', 'walk_time_min', 'wait_time_min', 'total_distance_km',
       'num_transfers', 'fare_1000won'))

# D: total_time + walk_share
D = run_model('D: TotalTime + WalkShare',
    ['total_time_min', 'walk_share', 'total_distance_km', 'num_transfers', 'fare_1000won',
     'has_bus', 'has_train', 'has_gtx'],
    SN('total_time_min', 'total_distance_km', 'num_transfers', 'fare_1000won'))

# E: total_time + walk_time
E = run_model('E: TotalTime + WalkTime',
    ['total_time_min', 'walk_time_min', 'total_distance_km', 'num_transfers', 'fare_1000won',
     'has_bus', 'has_train', 'has_gtx'],
    SN('total_time_min', 'walk_time_min', 'total_distance_km', 'num_transfers', 'fare_1000won'))

# F: GenTime (문헌 기반 고정비율 IVT+2.5walk+1.5wait)
F = run_model('F: GenTime(IVT+2.5w+1.5wt)',
    ['gen_time_lit', 'total_distance_km', 'num_transfers', 'fare_1000won',
     'has_bus', 'has_train', 'has_gtx'],
    SN('gen_time_lit', 'total_distance_km', 'num_transfers', 'fare_1000won'))

# ============================================================
# 4. 비교 요약
# ============================================================
models = [A, B, C, D, E, F]
print('\n' + '=' * 90)
print('Summary Comparison')
print('=' * 90)
print(f'{"Model":<35} {"rho_tr":>10} {"rho_te":>10} {"top1":>10} {"rmse":>10} {"bound":>15}')
print('-' * 90)
for m in models:
    bnd = ','.join(m['on_bound']) if m['on_bound'] else 'none'
    print(f'{m["name"]:<35} {m["rho_tr"]:>10.6f} {m["rho_te"]:>10.6f} '
          f'{m["top1"]*100:>9.2f}% {m["rmse"]:>10.6f} {bnd:>15}')
