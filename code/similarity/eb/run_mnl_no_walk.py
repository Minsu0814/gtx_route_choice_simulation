# -*- coding: utf-8 -*-
"""
EB 연동용 MNL 재추정: walk 피처 제거 버전

기존 K3 모형에서 ln_access, ln_egress, transfer_walk_time_min을 제거하고 재추정.
EB가 도보를 처리하므로 경로선택모형에서는 walk 이중 반영 방지.

기존 모형 vs walk 제거 모형 성능 비교 포함.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, approx_fprime
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / 'data' / 'training_set'
EB_DIR = ROOT / 'data' / 'eb'

# ============================================================
# 1. 데이터 로드 + 피처 생성
# ============================================================
print("=" * 60)
print("EB 연동 MNL: walk 피처 제거 버전")
print("=" * 60)

df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')
print(f'\n데이터: {len(df):,} rows, {df["od_pair"].nunique()} ODs')

if 'alt_idx' not in df.columns:
    df['alt_idx'] = df.groupby('od_pair').cumcount()
training_ods = set(df['od_pair'].unique())

# 모드별 IVT 추출
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
df = df.merge(ivt_df, on=['od_pair', 'alt_idx'], how='left')
df['bus_ivt_min'] = df['bus_ivt_min'].fillna(0)
df['train_ivt_min'] = df['train_ivt_min'].fillna(0)
df['gtx_ivt_min'] = df['gtx_ivt_min'].fillna(0)
del ivt_records, ivt_df

# 피처 생성
for col in ['access_time', 'egress_time', 'transfer_walk_time']:
    df[col + '_min'] = df[col] / 60
df['ln_access'] = np.log1p(df['access_time_min'])
df['ln_egress'] = np.log1p(df['egress_time_min'])
df['fare_1000won'] = df['fare'] / 1000
df['total_ivt_min'] = df['bus_ivt_min'] + df['train_ivt_min'] + df['gtx_ivt_min']

# ============================================================
# 2. 모형 사양 정의: 기존 K3 vs walk 제거
# ============================================================
SPECS = {
    'K3_original': {
        'features': ['total_ivt_min', 'ln_access', 'ln_egress', 'transfer_walk_time_min',
                      'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx'],
        'labels': ['Total IVT', 'ln(1+Access)', 'ln(1+Egress)', 'Transfer walk',
                    'Transfers', 'Fare', 'Has bus', 'Has train', 'Has GTX'],
        'neg_constrained': {'total_ivt_min', 'ln_access', 'ln_egress',
                            'transfer_walk_time_min', 'num_transfers', 'fare_1000won'},
    },
    'K3_no_walk': {
        'features': ['total_ivt_min', 'num_transfers', 'fare_1000won',
                      'has_bus', 'has_train', 'has_gtx'],
        'labels': ['Total IVT', 'Transfers', 'Fare',
                    'Has bus', 'Has train', 'Has GTX'],
        'neg_constrained': {'total_ivt_min', 'num_transfers', 'fare_1000won'},
    },
}

# ============================================================
# 3. Train/Test Split
# ============================================================
od_dominant = df.loc[
    df.groupby('od_pair')['choice_prob'].idxmax(),
    ['od_pair', 'transport_category']
].set_index('od_pair')['transport_category']


def coarsen(cat):
    return 'gtx_related' if 'gtx' in cat else cat


od_strat = od_dominant.map(coarsen)
od_list = od_strat.index.to_numpy()
strat_labels = od_strat.values

train_ods_arr, test_ods_arr = train_test_split(
    od_list, test_size=0.2, random_state=42, stratify=strat_labels
)

train_od_set = set(train_ods_arr)
test_od_set = set(test_ods_arr)

gtx_move_to_test = ['9007_9008', '9008_9007']
for od in gtx_move_to_test:
    train_od_set.discard(od)
    test_od_set.add(od)

train_df = df[df['od_pair'].isin(train_od_set)].copy()
test_df = df[df['od_pair'].isin(test_od_set)].copy()
print(f'\nTrain: {train_df["od_pair"].nunique()} ODs, Test: {test_df["od_pair"].nunique()} ODs')


# ============================================================
# 4. MNL 함수
# ============================================================
def prepare_flat_data(data, features):
    X_list, y_list, w_list, gid_list = [], [], [], []
    group_id = 0
    for od, grp in data.groupby('od_pair'):
        y = grp['choice_prob'].values.astype(np.float64)
        if abs(y.sum() - 1.0) > 0.01:
            continue
        X = grp[features].values.astype(np.float64)
        w = float(grp['n_total'].iloc[0])
        n = len(grp)
        X_list.append(X)
        y_list.append(y)
        w_list.append(np.full(n, w))
        gid_list.append(np.full(n, group_id, dtype=np.int64))
        group_id += 1
    return {
        'X': np.vstack(X_list), 'y': np.concatenate(y_list),
        'w': np.concatenate(w_list), 'gid': np.concatenate(gid_list),
        'n_groups': group_id,
    }


def mnl_neg_ll(beta, flat):
    X, y, w, gid, ng = flat['X'], flat['y'], flat['w'], flat['gid'], flat['n_groups']
    V = X @ beta
    V_max = np.full(ng, -np.inf)
    np.maximum.at(V_max, gid, V)
    V_shifted = V - V_max[gid]
    exp_V = np.exp(V_shifted)
    sum_exp = np.bincount(gid, weights=exp_V, minlength=ng)
    log_prob = V_shifted - np.log(sum_exp[gid])
    return -np.sum(w * y * log_prob)


def mnl_gradient(beta, flat):
    X, y, w, gid, ng = flat['X'], flat['y'], flat['w'], flat['gid'], flat['n_groups']
    V = X @ beta
    V_max = np.full(ng, -np.inf)
    np.maximum.at(V_max, gid, V)
    V_shifted = V - V_max[gid]
    exp_V = np.exp(V_shifted)
    sum_exp = np.bincount(gid, weights=exp_V, minlength=ng)
    prob = exp_V / sum_exp[gid]
    return X.T @ (w * (prob - y))


def fit_mnl(train_data, test_data, spec):
    features = spec['features']
    labels = spec['labels']
    neg_constrained = spec['neg_constrained']

    train_flat = prepare_flat_data(train_data, features)
    test_flat = prepare_flat_data(test_data, features)

    bounds = [(None, 0) if f in neg_constrained else (None, None) for f in features]
    beta0 = np.zeros(len(features))

    result = minimize(
        mnl_neg_ll, beta0, args=(train_flat,),
        jac=mnl_gradient, method='L-BFGS-B',
        bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12}
    )

    beta_hat = result.x

    # Hessian → SE → t-stat
    eps = 1e-5
    n_params = len(beta_hat)
    hessian = np.zeros((n_params, n_params))
    for i in range(n_params):
        def grad_i(b, _i=i):
            return mnl_gradient(b, train_flat)[_i]
        hessian[i, :] = approx_fprime(beta_hat, grad_i, eps)
    hessian = (hessian + hessian.T) / 2

    try:
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hessian)
    std_errors = np.sqrt(np.abs(np.diag(cov)))
    t_stats = beta_hat / np.where(std_errors > 0, std_errors, 1)

    # LL(0)
    gid = train_flat['gid']
    w = train_flat['w']
    ng = train_flat['n_groups']
    gsz = np.bincount(gid, minlength=ng)
    wpg = np.bincount(gid, weights=w, minlength=ng) / gsz
    ll_0 = -np.sum(wpg * np.log(gsz))
    ll_beta = -result.fun
    rho_sq = 1 - ll_beta / ll_0

    # Test ρ²
    test_ll = -mnl_neg_ll(beta_hat, test_flat)
    tgid = test_flat['gid']
    tng = test_flat['n_groups']
    tgsz = np.bincount(tgid, minlength=tng)
    twpg = np.bincount(tgid, weights=test_flat['w'], minlength=tng) / tgsz
    test_ll_0 = -np.sum(twpg * np.log(tgsz))
    test_rho_sq = 1 - test_ll / test_ll_0

    # Top-1 accuracy
    V = test_flat['X'] @ beta_hat
    V_max = np.full(tng, -np.inf)
    np.maximum.at(V_max, tgid, V)
    exp_V = np.exp(V - V_max[tgid])
    sum_exp = np.bincount(tgid, weights=exp_V, minlength=tng)
    pred_prob = exp_V / sum_exp[tgid]

    top1_correct = 0
    for g in range(tng):
        g_mask = tgid == g
        if np.argmax(pred_prob[g_mask]) == np.argmax(test_flat['y'][g_mask]):
            top1_correct += 1
    top1_acc = top1_correct / tng

    return {
        'beta': beta_hat, 'std_errors': std_errors, 't_stats': t_stats,
        'features': features, 'labels': labels,
        'train_rho_sq': rho_sq, 'test_rho_sq': test_rho_sq,
        'top1_acc': top1_acc, 'll_0': ll_0, 'll_beta': ll_beta,
        'test_ll': test_ll, 'test_ll_0': test_ll_0,
        'n_train_ods': train_flat['n_groups'], 'n_test_ods': test_flat['n_groups'],
    }


# ============================================================
# 5. 두 모형 추정 + 비교
# ============================================================
results = {}
for name, spec in SPECS.items():
    print(f'\n{"=" * 60}')
    print(f'모형: {name} ({len(spec["features"])}개 피처)')
    print(f'{"=" * 60}')

    r = fit_mnl(train_df, test_df, spec)
    results[name] = r

    print(f'\n{"Feature":<22} {"β":>12} {"SE":>12} {"t-stat":>10}')
    print('-' * 58)
    for label, b, se, t in zip(r['labels'], r['beta'], r['std_errors'], r['t_stats']):
        sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.960 else ''
        print(f'{label:<22} {b:>12.6f} {se:>12.6f} {t:>10.2f} {sig}')
    print('-' * 58)
    print(f'Train ρ²: {r["train_rho_sq"]:.4f}')
    print(f'Test ρ²:  {r["test_rho_sq"]:.4f}')
    print(f'Top-1:    {r["top1_acc"]:.4f} ({r["top1_acc"]*100:.1f}%)')

# ============================================================
# 6. 비교 테이블
# ============================================================
print(f'\n{"=" * 60}')
print('모형 비교: K3 (기존) vs K3 no-walk (EB 연동)')
print(f'{"=" * 60}')
print(f'{"Metric":<25} {"K3 original":>15} {"K3 no-walk":>15} {"차이":>10}')
print('-' * 68)

r1 = results['K3_original']
r2 = results['K3_no_walk']

metrics = [
    ('Train ρ²', r1['train_rho_sq'], r2['train_rho_sq']),
    ('Test ρ²', r1['test_rho_sq'], r2['test_rho_sq']),
    ('Top-1 Accuracy', r1['top1_acc'], r2['top1_acc']),
    ('LL(β) train', r1['ll_beta'], r2['ll_beta']),
    ('LL(β) test', r1['test_ll'], r2['test_ll']),
    ('N params', len(r1['features']), len(r2['features'])),
]

for name, v1, v2 in metrics:
    if isinstance(v1, int):
        print(f'{name:<25} {v1:>15} {v2:>15} {v2-v1:>+10}')
    else:
        print(f'{name:<25} {v1:>15.4f} {v2:>15.4f} {v2-v1:>+10.4f}')

# ============================================================
# 7. no-walk 계수 저장 (EB 적용용)
# ============================================================
no_walk = results['K3_no_walk']
coef_dict = {f: float(b) for f, b in zip(no_walk['features'], no_walk['beta'])}
coef_output = {
    'model': 'K3_no_walk',
    'description': 'MNL without walk features (for EB integration)',
    'coefficients': coef_dict,
    'train_rho_sq': float(no_walk['train_rho_sq']),
    'test_rho_sq': float(no_walk['test_rho_sq']),
    'top1_accuracy': float(no_walk['top1_acc']),
}

out_path = EB_DIR / 'mnl_no_walk_coefficients.json'
with open(out_path, 'w') as f:
    json.dump(coef_output, f, indent=2)
print(f'\n저장: {out_path}')

print('\n완료.')
