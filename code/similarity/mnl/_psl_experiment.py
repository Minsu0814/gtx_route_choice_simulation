"""PSL (Path Size Logit) 보정항 추가 실험.

두 가지 PS 정의를 비교한다:
  1) stop-based  — full_stops 정류장 수준 중첩
  2) leg-based   — 노선(route) 수준 중첩 (Hoogendoorn-Lanser 2005)
"""
import sqlite3
import pickle
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / 'data' / 'training_set'

sys.path.insert(0, str(ROOT / 'code' / 'similarity' / 'module'))
from gtfs_lookup import GTFSRouteLookup

# ============================================================
# 1. GTFS lookup 로드 + full_stops 복구 + PS term 계산
# ============================================================
print('=' * 70)
print('Step 1: GTFS lookup + full_stops 복구 + PS term 계산')
print('=' * 70)

t0 = time.time()

gtfs = GTFSRouteLookup(str(ROOT / 'data' / 'gtfs' / 'a1'))

conn = sqlite3.connect(str(DATA_DIR / 'otp_cache.db'))
cur = conn.execute('SELECT od_pair, n_alts, data FROM otp_cache')

ps_stop_records = []   # stop-based
ps_leg_records = []    # leg-based
n_ods = 0
n_total_alts = 0

for od_pair, n_alts, blob in cur:
    data = pickle.loads(blob)
    parsed_list = data['otp_parsed']
    n_ods += 1

    # 각 대안의 정류장 집합 & 노선 리스트
    all_stops_per_alt = []
    all_legs_per_alt = []   # [(route_name, from_stop, to_stop), ...]
    for p in parsed_list:
        n_total_alts += 1
        # stop-based
        stops = p['full_stops'] if p['full_stops'] else p['stops']
        all_stops_per_alt.append(set(stops) if stops else set())

        # leg-based: 노선명 리스트
        legs = p.get('transit_legs', [])
        leg_routes = []
        for leg in legs:
            rn = leg['route_name'].strip()
            if rn:
                leg_routes.append(rn)
        all_legs_per_alt.append(leg_routes)

    n = len(all_stops_per_alt)

    # --- Stop-based PS ---
    for i in range(n):
        L_i = len(all_stops_per_alt[i])
        if L_i == 0:
            ps_stop_records.append((od_pair, i, 0.0, 0))
            continue
        ps = 0.0
        for stop in all_stops_per_alt[i]:
            n_shared = sum(1 for j in range(n) if stop in all_stops_per_alt[j])
            ps += (1.0 / L_i) * (1.0 / n_shared)
        ps_stop_records.append((od_pair, i, np.log(max(ps, 1e-10)), L_i))

    # --- Leg-based PS (Hoogendoorn-Lanser 2005) ---
    for i in range(n):
        L_i = len(all_legs_per_alt[i])
        if L_i == 0:
            ps_leg_records.append((od_pair, i, 0.0, 0))
            continue
        ps = 0.0
        for route in all_legs_per_alt[i]:
            n_shared = sum(1 for j in range(n) if route in all_legs_per_alt[j])
            ps += (1.0 / L_i) * (1.0 / n_shared)
        ps_leg_records.append((od_pair, i, np.log(max(ps, 1e-10)), L_i))

    if n_ods % 100000 == 0:
        print(f'  {n_ods:,} ODs processed... ({time.time()-t0:.0f}s)')

conn.close()

ps_stop_df = pd.DataFrame(ps_stop_records, columns=['od_pair', 'alt_idx', 'ln_ps_stop', 'n_stops'])
ps_leg_df = pd.DataFrame(ps_leg_records, columns=['od_pair', 'alt_idx', 'ln_ps_leg', 'n_legs'])
ps_df = ps_stop_df.merge(ps_leg_df, on=['od_pair', 'alt_idx'])

print(f'\n완료: {n_ods:,} ODs, {n_total_alts:,} alts ({time.time()-t0:.1f}s)')
for tag, col in [('stop-based', 'ln_ps_stop'), ('leg-based', 'ln_ps_leg')]:
    s = ps_df[col]
    print(f'\n  [{tag}] ln(PS):  mean={s.mean():.4f}, std={s.std():.4f}, '
          f'min={s.min():.4f}, max={s.max():.4f}')
    print(f'    ln(PS)=0: {(s==0).sum():,} ({(s==0).mean()*100:.1f}%)')

# ============================================================
# 2. Training data merge
# ============================================================
print('\n' + '=' * 70)
print('Step 2: Training data merge')
print('=' * 70)

df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')
if 'alt_idx' not in df.columns:
    df['alt_idx'] = df.groupby('od_pair').cumcount()

df = df.merge(ps_df[['od_pair', 'alt_idx', 'ln_ps_stop', 'ln_ps_leg']], on=['od_pair', 'alt_idx'], how='left')
df['ln_ps_stop'] = df['ln_ps_stop'].fillna(0.0)
df['ln_ps_leg'] = df['ln_ps_leg'].fillna(0.0)

print(f'Merge: {len(df):,} rows')
print(f'  ln_ps_stop==0: {(df["ln_ps_stop"]==0).mean()*100:.1f}%')
print(f'  ln_ps_leg==0:  {(df["ln_ps_leg"]==0).mean()*100:.1f}%')

# ============================================================
# 3. 전처리 + Split
# ============================================================
for col in ['in_vehicle_time', 'wait_time', 'access_time', 'egress_time', 'transfer_walk_time']:
    df[col + '_min'] = df[col] / 60
df['total_distance_km'] = df['total_distance'] / 1000
df['fare_1000won'] = df['fare'] / 1000

from sklearn.model_selection import train_test_split
od_dom = df.loc[df.groupby('od_pair')['choice_prob'].idxmax(),
                ['od_pair', 'transport_category']].set_index('od_pair')['transport_category']
coarsen = lambda c: 'gtx_related' if 'gtx' in c else c
od_strat = od_dom.map(coarsen)
tr_ods, te_ods = train_test_split(
    od_strat.index.to_numpy(), test_size=0.2, random_state=42, stratify=od_strat.values)
tr_set, te_set = set(tr_ods), set(te_ods)
for od in ['9007_9008', '9008_9007']:
    tr_set.discard(od); te_set.add(od)
train_df = df[df['od_pair'].isin(tr_set)].copy()
test_df = df[df['od_pair'].isin(te_set)].copy()
print(f'Train: {len(train_df):,} rows, Test: {len(test_df):,} rows')

# ============================================================
# 4. MNL engine
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

def run_model(train_df, test_df, features, labels, sign_neg, name):
    print(f'\n--- {name} ---')
    bounds = [(None, 0) if f in sign_neg else (None, None) for f in features]
    tr_fl = prepare_flat(train_df, features)
    te_fl = prepare_flat(test_df, features)

    t = time.time()
    res = minimize(neg_ll, np.zeros(len(features)), args=(tr_fl,), jac=grad_fn,
                   method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
    beta = res.x
    print(f'  converged={res.success}, iter={res.nit} ({time.time()-t:.1f}s)')

    # LL & rho
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

    # Top-1 & RMSE
    V = te_fl['X'] @ beta
    Vm = np.full(ng_t, -np.inf); np.maximum.at(Vm, gid_t, V)
    eV = np.exp(V - Vm[gid_t])
    se = np.bincount(gid_t, weights=eV, minlength=ng_t)
    pred = eV / se[gid_t]; actual = te_fl['y']
    top1 = sum(np.argmax(pred[gid_t == g]) == np.argmax(actual[gid_t == g]) for g in range(ng_t)) / ng_t
    rmse = np.sqrt(np.mean((pred - actual) ** 2))

    # SE & t-stat
    from scipy.optimize import approx_fprime
    n_p = len(beta)
    hess = np.zeros((n_p, n_p))
    for i in range(n_p):
        def _gi(b, _i=i): return grad_fn(b, tr_fl)[_i]
        hess[i, :] = approx_fprime(beta, _gi, 1e-5)
    hess = (hess + hess.T) / 2
    try:
        se_arr = np.sqrt(np.abs(np.diag(np.linalg.inv(hess))))
    except np.linalg.LinAlgError:
        se_arr = np.sqrt(np.abs(np.diag(np.linalg.pinv(hess))))
    t_stats = beta / np.where(se_arr > 0, se_arr, 1)

    on_bound = [f for f, b in zip(features, beta) if f in sign_neg and abs(b) < 1e-10]

    return dict(beta=beta, se=se_arr, t=t_stats, feats=features, labels=labels,
                rho_tr=rho_tr, rho_te=rho_te, ll_tr=ll_tr, ll_te=ll_te,
                top1=top1, rmse=rmse, on_bound=on_bound, n_te=ng_t)

# ============================================================
# 5. 실행
# ============================================================
SIGN_NEG = {
    'in_vehicle_time_min', 'wait_time_min', 'access_time_min', 'egress_time_min',
    'transfer_walk_time_min', 'total_distance_km', 'num_transfers', 'fare_1000won',
}

BASE_F = ['in_vehicle_time_min', 'wait_time_min', 'access_time_min', 'egress_time_min',
          'transfer_walk_time_min', 'total_distance_km', 'num_transfers', 'fare_1000won',
          'has_bus', 'has_train', 'has_gtx']
BASE_L = ['IVT (min)', 'Wait (min)', 'Access walk (min)', 'Egress walk (min)',
          'Transfer walk (min)', 'Total dist (km)', 'Transfers', 'Fare (1000won)',
          'Has bus', 'Has train', 'Has GTX']

PSL_STOP_F = BASE_F + ['ln_ps_stop']
PSL_STOP_L = BASE_L + ['ln(PS) stop']
PSL_LEG_F = BASE_F + ['ln_ps_leg']
PSL_LEG_L = BASE_L + ['ln(PS) leg']
PSL_BOTH_F = BASE_F + ['ln_ps_stop', 'ln_ps_leg']
PSL_BOTH_L = BASE_L + ['ln(PS) stop', 'ln(PS) leg']

print('\n' + '=' * 70)
print('Step 3: Model estimation')
print('=' * 70)

A = run_model(train_df, test_df, BASE_F, BASE_L, SIGN_NEG, 'MNL (baseline)')
B = run_model(train_df, test_df, PSL_STOP_F, PSL_STOP_L, SIGN_NEG, 'PSL (stop-based)')
C = run_model(train_df, test_df, PSL_LEG_F, PSL_LEG_L, SIGN_NEG, 'PSL (leg-based)')
D = run_model(train_df, test_df, PSL_BOTH_F, PSL_BOTH_L, SIGN_NEG, 'PSL (stop+leg)')

# ============================================================
# 6. 결과 비교 (4개 모형)
# ============================================================
models = [
    ('MNL', A),
    ('PSL-stop', B),
    ('PSL-leg', C),
    ('PSL-both', D),
]

print('\n' + '=' * 90)
print('          Model Comparison: MNL vs PSL variants')
print('=' * 90)

header = f'{"":>25}' + ''.join(f'{nm:>15}' for nm, _ in models)
print(f'\n{header}')
print('-' * 90)

for metric_name, key in [('Train rho-sq', 'rho_tr'), ('Test rho-sq', 'rho_te'),
                          ('Train LL', 'll_tr'), ('Test LL', 'll_te')]:
    vals = [m[key] for _, m in models]
    line = f'{metric_name:>25}' + ''.join(f'{v:>15.6f}' for v in vals)
    print(line)

vals_t1 = [m['top1'] * 100 for _, m in models]
print(f'{"Top-1 Accuracy":>25}' + ''.join(f'{v:>14.2f}%' for v in vals_t1))
vals_rmse = [m['rmse'] for _, m in models]
print(f'{"RMSE":>25}' + ''.join(f'{v:>15.6f}' for v in vals_rmse))
vals_bnd = [str(m['on_bound']) for _, m in models]
print(f'\n{"Bound":>25}' + ''.join(f'{v:>15}' for v in vals_bnd))

# vs MNL diff
print(f'\n--- vs MNL baseline ---')
print(f'{"":>25}' + ''.join(f'{nm:>15}' for nm, _ in models[1:]))
print('-' * 70)
for metric_name, key in [('dRho-sq (test)', 'rho_te'), ('dTop-1', 'top1'), ('dRMSE', 'rmse')]:
    base = A[key]
    diffs = [m[key] - base for _, m in models[1:]]
    if key == 'top1':
        print(f'{metric_name:>25}' + ''.join(f'{d*100:>+14.3f}%' for d in diffs))
    else:
        print(f'{metric_name:>25}' + ''.join(f'{d:>+15.6f}' for d in diffs))

# beta 비교
print('\n' + '=' * 90)
print('beta coefficients')
print('=' * 90)
print(f'{"Feature":<22}' + ''.join(f'{nm:>15}' for nm, _ in models))
print('-' * 90)

for i, (f, l) in enumerate(zip(BASE_F, BASE_L)):
    vals = [m['beta'][i] for _, m in models]
    on = ' *' if any(abs(v) < 1e-10 and f in SIGN_NEG for v in vals) else ''
    print(f'{l:<22}' + ''.join(f'{v:>15.6f}' for v in vals) + on)

# PS terms
ps_labels = [('ln(PS) stop', 'ln_ps_stop'), ('ln(PS) leg', 'ln_ps_leg')]
for label, feat in ps_labels:
    vals = []
    for nm, m in models:
        if feat in m['feats']:
            idx = m['feats'].index(feat)
            vals.append(f'{m["beta"][idx]:>12.6f} (t={m["t"][idx]:.1f})')
        else:
            vals.append(f'{"--":>12}       ')
    print(f'{label:<22}' + ''.join(f'{v:>22}' for v in vals))

# Walk/IVT ratio
print('\n=== Walk / IVT ratio ===')
ivt_idx = BASE_F.index('in_vehicle_time_min')
print(f'{"":>22}' + ''.join(f'{nm:>15}' for nm, _ in models))
for label, feat in [('Access walk', 'access_time_min'), ('Egress walk', 'egress_time_min'),
                     ('Transfer walk', 'transfer_walk_time_min')]:
    fi = BASE_F.index(feat)
    ratios = []
    for nm, m in models:
        ivt = m['beta'][ivt_idx]
        w = m['beta'][fi]
        ratios.append(abs(w / ivt) if abs(ivt) > 1e-10 else 0)
    print(f'{label:<22}' + ''.join(f'{r:>14.1f}x' for r in ratios) + '  (expected: 2-5x)')

# Wait
print(f'\n=== Wait beta ===')
wi = BASE_F.index('wait_time_min')
for nm, m in models:
    print(f'  {nm}: {m["beta"][wi]:.6f}')

# LR tests vs MNL
from scipy import stats
print(f'\n=== LR tests (vs MNL) ===')
for nm, m in models[1:]:
    lr = -2 * (A['ll_tr'] - m['ll_tr'])
    n_extra = len(m['feats']) - len(A['feats'])
    pv = 1 - stats.chi2.cdf(lr, df=n_extra)
    print(f'  {nm}: LR={lr:.4f}, df={n_extra}, p={pv:.2e} → {"YES" if pv < 0.001 else "NO"}')

print(f'\nTotal: {time.time()-t0:.1f}s')
