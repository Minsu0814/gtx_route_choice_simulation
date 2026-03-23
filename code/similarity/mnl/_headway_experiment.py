"""Headway 기반 대기시간 실험.

현재 wait_time은 total_duration - IVT - walk의 잔차로 계산 → β=0.
GTFS headway 기반 expected wait (headway/2)로 대체하여 β 활성화 시도.
"""
import sqlite3
import pickle
import csv
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / 'data' / 'training_set'
GTFS_DIR = ROOT / 'data' / 'gtfs' / 'a1'

# ============================================================
# 1. GTFS headway 룩업 빌드: route_short_name → median headway (min)
# ============================================================
print('=' * 70)
print('Step 1: Build headway lookup from GTFS')
print('=' * 70)
t0 = time.time()

# routes.txt: route_id → route_short_name
routes_df = pd.read_csv(GTFS_DIR / 'routes.txt', dtype={'route_id': str})
rid_to_name = dict(zip(routes_df['route_id'], routes_df['route_short_name'].str.strip()))

# trips.txt: trip_id → route_id
trips_df = pd.read_csv(GTFS_DIR / 'trips.txt', dtype={'route_id': str, 'trip_id': str})
tid_to_rid = dict(zip(trips_df['trip_id'], trips_df['route_id']))
print(f'  routes: {len(rid_to_name):,}, trips: {len(tid_to_rid):,}')

# stop_times.txt (22M rows): csv 모듈로 stop_sequence==1만 추출
print('  Reading stop_times.txt (csv module, stop_sequence==1 only)...')
first_deps = {}  # trip_id → departure_time
with open(GTFS_DIR / 'stop_times.txt', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['stop_sequence']) == 1:
            first_deps[row['trip_id']] = row['departure_time']

print(f'  First departures: {len(first_deps):,} trips')

# departure_time → minutes
def time_to_minutes(t):
    try:
        parts = str(t).split(':')
        return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
    except:
        return np.nan

# route_id별 departure minutes 수집
route_deps = {}  # route_id → [dep_min, ...]
for tid, dep in first_deps.items():
    rid = tid_to_rid.get(tid)
    if rid is None:
        continue
    dm = time_to_minutes(dep)
    if np.isnan(dm):
        continue
    route_deps.setdefault(rid, []).append(dm)
del first_deps  # 메모리 해제

# route_id별 headway 계산: 연속 출발시간 간격의 중앙값
def calc_headway(deps_list):
    deps = np.sort(deps_list)
    if len(deps) < 2:
        return np.nan
    gaps = np.diff(deps)
    # 이상값 제거 (gap > 120분 = 운행 안 하는 시간)
    gaps = gaps[gaps <= 120]
    if len(gaps) == 0:
        return np.nan
    return float(np.median(gaps))

headway_by_rid = {rid: calc_headway(deps) for rid, deps in route_deps.items()}
headway_by_rid = {k: v for k, v in headway_by_rid.items() if not np.isnan(v)}
del route_deps

# route_id → route_short_name 매핑, 같은 노선명의 route_id들 중 최소 headway
name_hws = {}  # route_name → [headway, ...]
for rid, hw in headway_by_rid.items():
    name = rid_to_name.get(rid, '')
    if name and str(name) != 'nan':
        name_hws.setdefault(name, []).append(hw)
del headway_by_rid

# 같은 route_name에 대해 여러 route_id가 있을 수 있음 → 중앙값
headway_lookup = {name: float(np.median(hws)) for name, hws in name_hws.items()}
del name_hws

# 서울 지하철 이름 정규화 (서울1호선 → 1호선)
extra = {}
for name, hw in headway_lookup.items():
    if name.startswith('서울') and '호선' in name:
        extra[name[2:]] = hw  # '1호선'
headway_lookup.update(extra)

print(f'  Headway lookup: {len(headway_lookup):,} route names')
hws = np.array(list(headway_lookup.values()))
print(f'  Headway stats: median={np.median(hws):.1f}min, '
      f'mean={np.mean(hws):.1f}min, min={np.min(hws):.1f}min, max={np.max(hws):.1f}min')
print(f'  ({time.time()-t0:.1f}s)')

# ============================================================
# 2. OTP cache에서 각 대안의 headway_wait 계산
# ============================================================
print('\n' + '=' * 70)
print('Step 2: Compute headway_wait per alternative from OTP cache')
print('=' * 70)
t1 = time.time()

conn = sqlite3.connect(str(DATA_DIR / 'otp_cache.db'))
cur = conn.execute('SELECT od_pair, n_alts, data FROM otp_cache')

hw_records = []
n_ods = 0
n_miss = 0
n_hit = 0
fallback_hw = np.median(hws)  # 매칭 안 되는 노선 → 전체 중앙값 사용

for od_pair, n_alts, blob in cur:
    data = pickle.loads(blob)
    parsed_list = data['otp_parsed']
    n_ods += 1

    for i, p in enumerate(parsed_list):
        legs = p.get('transit_legs', [])
        total_hw_wait = 0.0
        n_boardings = len(legs)

        for leg in legs:
            rn = leg['route_name'].strip()
            hw = headway_lookup.get(rn)
            if hw is None:
                hw = fallback_hw
                n_miss += 1
            else:
                n_hit += 1
            total_hw_wait += hw / 2.0  # expected wait = headway/2

        hw_records.append((od_pair, i, total_hw_wait, n_boardings))

    if n_ods % 100000 == 0:
        print(f'  {n_ods:,} ODs... ({time.time()-t1:.0f}s)')

conn.close()

hw_wait_df = pd.DataFrame(hw_records, columns=['od_pair', 'alt_idx', 'headway_wait_min', 'n_boardings'])

hit_rate = n_hit / (n_hit + n_miss) * 100 if (n_hit + n_miss) > 0 else 0
print(f'\n완료: {n_ods:,} ODs, {len(hw_records):,} alts ({time.time()-t1:.1f}s)')
print(f'  Headway match rate: {hit_rate:.1f}% ({n_hit:,} hit, {n_miss:,} miss)')
print(f'  headway_wait stats: mean={hw_wait_df["headway_wait_min"].mean():.2f}min, '
      f'std={hw_wait_df["headway_wait_min"].std():.2f}min')

# ============================================================
# 3. Training data merge
# ============================================================
print('\n' + '=' * 70)
print('Step 3: Training data merge')
print('=' * 70)

df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')
if 'alt_idx' not in df.columns:
    df['alt_idx'] = df.groupby('od_pair').cumcount()

df = df.merge(hw_wait_df[['od_pair', 'alt_idx', 'headway_wait_min']],
              on=['od_pair', 'alt_idx'], how='left')
df['headway_wait_min'] = df['headway_wait_min'].fillna(fallback_hw / 2)

print(f'Merge: {len(df):,} rows')
print(f'  headway_wait_min: mean={df["headway_wait_min"].mean():.2f}, '
      f'std={df["headway_wait_min"].std():.2f}')

# 기존 잔차 wait 분포 비교
wait_min = df['wait_time'] / 60
print(f'  residual wait_min: mean={wait_min.mean():.2f}, std={wait_min.std():.2f}')
print(f'  correlation(residual, headway): {wait_min.corr(df["headway_wait_min"]):.4f}')

# ============================================================
# 4. 전처리 + Split
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
# 5. MNL engine (PSL experiment에서 재사용)
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
# 6. 모델 비교: baseline / headway 대체 / headway 추가 / residual+headway 둘다
# ============================================================
SIGN_NEG = {
    'in_vehicle_time_min', 'wait_time_min', 'access_time_min', 'egress_time_min',
    'transfer_walk_time_min', 'total_distance_km', 'num_transfers', 'fare_1000won',
    'headway_wait_min',
}

# A: Baseline (기존 잔차 wait)
BASE_F = ['in_vehicle_time_min', 'wait_time_min', 'access_time_min', 'egress_time_min',
          'transfer_walk_time_min', 'total_distance_km', 'num_transfers', 'fare_1000won',
          'has_bus', 'has_train', 'has_gtx']
BASE_L = ['IVT (min)', 'Wait-residual (min)', 'Access walk (min)', 'Egress walk (min)',
          'Transfer walk (min)', 'Total dist (km)', 'Transfers', 'Fare (1000won)',
          'Has bus', 'Has train', 'Has GTX']

# B: headway wait 대체 (잔차 wait → headway wait)
HW_REPLACE_F = ['in_vehicle_time_min', 'headway_wait_min', 'access_time_min', 'egress_time_min',
                'transfer_walk_time_min', 'total_distance_km', 'num_transfers', 'fare_1000won',
                'has_bus', 'has_train', 'has_gtx']
HW_REPLACE_L = ['IVT (min)', 'Wait-headway (min)', 'Access walk (min)', 'Egress walk (min)',
                'Transfer walk (min)', 'Total dist (km)', 'Transfers', 'Fare (1000won)',
                'Has bus', 'Has train', 'Has GTX']

# C: headway wait 추가 (잔차 + headway 둘 다)
HW_BOTH_F = BASE_F + ['headway_wait_min']
HW_BOTH_L = BASE_L + ['Wait-headway (min)']

# D: headway wait만 (잔차 wait 제거)
HW_ONLY_F = ['in_vehicle_time_min', 'headway_wait_min', 'access_time_min', 'egress_time_min',
             'transfer_walk_time_min', 'total_distance_km', 'num_transfers', 'fare_1000won',
             'has_bus', 'has_train', 'has_gtx']

print('\n' + '=' * 70)
print('Step 4: Model estimation')
print('=' * 70)

A = run_model(train_df, test_df, BASE_F, BASE_L, SIGN_NEG, 'MNL Baseline (residual wait)')
B = run_model(train_df, test_df, HW_REPLACE_F, HW_REPLACE_L, SIGN_NEG, 'MNL Headway-replace')
C = run_model(train_df, test_df, HW_BOTH_F, HW_BOTH_L, SIGN_NEG, 'MNL Residual+Headway')

# ============================================================
# 7. 결과 비교
# ============================================================
models = [
    ('Baseline', A),
    ('HW-replace', B),
    ('Res+HW', C),
]

print('\n' + '=' * 90)
print('          Model Comparison: Wait Time Variants')
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

# vs Baseline diff
print(f'\n--- vs Baseline ---')
print(f'{"":>25}' + ''.join(f'{nm:>15}' for nm, _ in models[1:]))
print('-' * 70)
for metric_name, key in [('dRho-sq (test)', 'rho_te'), ('dTop-1', 'top1'), ('dRMSE', 'rmse')]:
    base = A[key]
    diffs = [m[key] - base for _, m in models[1:]]
    if key == 'top1':
        print(f'{metric_name:>25}' + ''.join(f'{d*100:>+14.3f}%' for d in diffs))
    else:
        print(f'{metric_name:>25}' + ''.join(f'{d:>+15.6f}' for d in diffs))

# beta 비교 (정렬된 피처)
print('\n' + '=' * 90)
print('Beta coefficients')
print('=' * 90)
print(f'{"Feature":<25} {"Label":<20}' + ''.join(f'{nm:>15}' for nm, _ in models))
print('-' * 100)

all_feats = ['in_vehicle_time_min', 'wait_time_min', 'headway_wait_min',
             'access_time_min', 'egress_time_min', 'transfer_walk_time_min',
             'total_distance_km', 'num_transfers', 'fare_1000won',
             'has_bus', 'has_train', 'has_gtx']
all_labels = ['IVT', 'Wait-residual', 'Wait-headway',
              'Access walk', 'Egress walk', 'Transfer walk',
              'Total dist', 'Transfers', 'Fare',
              'Has bus', 'Has train', 'Has GTX']

for feat, label in zip(all_feats, all_labels):
    vals = []
    for nm, m in models:
        if feat in m['feats']:
            idx = m['feats'].index(feat)
            b = m['beta'][idx]
            t = m['t'][idx]
            vals.append(f'{b:>9.4f} (t={t:>6.1f})')
        else:
            vals.append(f'{"--":>18}')
    on = ' *' if feat in SIGN_NEG and any(
        feat in m['feats'] and abs(m['beta'][m['feats'].index(feat)]) < 1e-10
        for _, m in models
    ) else ''
    print(f'{feat:<25} {label:<20}' + ''.join(f'{v:>20}' for v in vals) + on)

# Walk/IVT ratio
print('\n=== Walk / IVT ratio ===')
ivt_f = 'in_vehicle_time_min'
print(f'{"":>25}' + ''.join(f'{nm:>15}' for nm, _ in models))
for label, feat in [('Access walk', 'access_time_min'), ('Egress walk', 'egress_time_min'),
                     ('Transfer walk', 'transfer_walk_time_min')]:
    ratios = []
    for nm, m in models:
        ivt_idx = m['feats'].index(ivt_f)
        fi = m['feats'].index(feat)
        ivt = m['beta'][ivt_idx]
        w = m['beta'][fi]
        ratios.append(abs(w / ivt) if abs(ivt) > 1e-10 else 0)
    print(f'{label:<25}' + ''.join(f'{r:>14.1f}x' for r in ratios) + '  (expected: 2-5x)')

# Wait/IVT ratio
print(f'\n=== Wait / IVT ratio ===')
for nm, m in models:
    ivt_idx = m['feats'].index(ivt_f)
    ivt = m['beta'][ivt_idx]
    for wf in ['wait_time_min', 'headway_wait_min']:
        if wf in m['feats']:
            wi = m['feats'].index(wf)
            w = m['beta'][wi]
            ratio = abs(w / ivt) if abs(ivt) > 1e-10 else 0
            print(f'  {nm} {wf}: {w:.6f} / {ivt:.6f} = {ratio:.1f}x  (expected: 1-3x)')

# LR tests
from scipy import stats
print(f'\n=== LR tests (vs Baseline) ===')
for nm, m in models[1:]:
    lr = -2 * (A['ll_tr'] - m['ll_tr'])
    n_extra = len(m['feats']) - len(A['feats'])
    if n_extra > 0:
        pv = 1 - stats.chi2.cdf(lr, df=n_extra)
        print(f'  {nm}: LR={lr:.4f}, df={n_extra}, p={pv:.2e}')
    else:
        print(f'  {nm}: same #params, dLL={m["ll_tr"]-A["ll_tr"]:.4f}')

print(f'\nTotal: {time.time()-t0:.1f}s')
