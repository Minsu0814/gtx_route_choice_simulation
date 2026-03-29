# -*- coding: utf-8 -*-
"""
4가지 MNL 모형 비교:
  1. K3 기존:       정류장 OD + walk 포함
  2. K3 no-walk:    정류장 OD + walk 제거
  3. K3 H3:         H3 재집계 + walk 포함
  4. K3 H3 no-walk: H3 재집계 + walk 제거
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
# MNL 함수
# ============================================================
def prepare_flat_data(data, features, od_col='od_pair', prob_col='choice_prob', weight_col='n_total'):
    X_list, y_list, w_list, gid_list = [], [], [], []
    group_id = 0
    for od, grp in data.groupby(od_col):
        y = grp[prob_col].values.astype(np.float64)
        if abs(y.sum() - 1.0) > 0.01:
            continue
        if y.sum() == 0:
            continue
        X = grp[features].values.astype(np.float64)
        w = float(grp[weight_col].iloc[0]) if weight_col in grp.columns else 1.0
        n = len(grp)
        X_list.append(X)
        y_list.append(y)
        w_list.append(np.full(n, w))
        gid_list.append(np.full(n, group_id, dtype=np.int64))
        group_id += 1
    if not X_list:
        return None
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


def fit_and_evaluate(train_flat, test_flat, features, labels, neg_constrained):
    bounds = [(None, 0) if f in neg_constrained else (None, None) for f in features]
    beta0 = np.zeros(len(features))

    result = minimize(
        mnl_neg_ll, beta0, args=(train_flat,),
        jac=mnl_gradient, method='L-BFGS-B',
        bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12}
    )
    beta_hat = result.x

    # SE, t-stat
    n_params = len(beta_hat)
    hessian = np.zeros((n_params, n_params))
    for i in range(n_params):
        def grad_i(b, _i=i):
            return mnl_gradient(b, train_flat)[_i]
        hessian[i, :] = approx_fprime(beta_hat, grad_i, 1e-5)
    hessian = (hessian + hessian.T) / 2
    try:
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hessian)
    std_errors = np.sqrt(np.abs(np.diag(cov)))
    t_stats = beta_hat / np.where(std_errors > 0, std_errors, 1)

    # Train ρ²
    gid, w, ng = train_flat['gid'], train_flat['w'], train_flat['n_groups']
    gsz = np.bincount(gid, minlength=ng)
    wpg = np.bincount(gid, weights=w, minlength=ng) / gsz
    ll_0 = -np.sum(wpg * np.log(gsz))
    ll_beta = -result.fun
    train_rho = 1 - ll_beta / ll_0

    # Test ρ²
    test_ll = -mnl_neg_ll(beta_hat, test_flat)
    tgid, tng = test_flat['gid'], test_flat['n_groups']
    tgsz = np.bincount(tgid, minlength=tng)
    twpg = np.bincount(tgid, weights=test_flat['w'], minlength=tng) / tgsz
    test_ll_0 = -np.sum(twpg * np.log(tgsz))
    test_rho = 1 - test_ll / test_ll_0

    # Top-1
    V = test_flat['X'] @ beta_hat
    V_max = np.full(tng, -np.inf)
    np.maximum.at(V_max, tgid, V)
    exp_V = np.exp(V - V_max[tgid])
    sum_exp = np.bincount(tgid, weights=exp_V, minlength=tng)
    pred_prob = exp_V / sum_exp[tgid]
    top1 = sum(
        np.argmax(pred_prob[tgid == g]) == np.argmax(test_flat['y'][tgid == g])
        for g in range(tng)
    ) / tng

    return {
        'beta': beta_hat, 'std_errors': std_errors, 't_stats': t_stats,
        'train_rho': train_rho, 'test_rho': test_rho, 'top1': top1,
        'n_train': train_flat['n_groups'], 'n_test': test_flat['n_groups'],
    }


# ============================================================
# 데이터 로드
# ============================================================
print("=" * 70)
print("4가지 MNL 모형 비교")
print("=" * 70)

# 1. 정류장 OD 데이터
print("\n[1] 정류장 OD 데이터 로드...")
stop_df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')
if 'alt_idx' not in stop_df.columns:
    stop_df['alt_idx'] = stop_df.groupby('od_pair').cumcount()

# 모드별 IVT
training_ods = set(stop_df['od_pair'].unique())
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
stop_df = stop_df.merge(ivt_df, on=['od_pair', 'alt_idx'], how='left')
for c in ['bus_ivt_min', 'train_ivt_min', 'gtx_ivt_min']:
    stop_df[c] = stop_df[c].fillna(0)
del ivt_records, ivt_df

# 피처 생성
for col in ['access_time', 'egress_time', 'transfer_walk_time']:
    stop_df[col + '_min'] = stop_df[col] / 60
stop_df['ln_access'] = np.log1p(stop_df['access_time_min'])
stop_df['ln_egress'] = np.log1p(stop_df['egress_time_min'])
stop_df['fare_1000won'] = stop_df['fare'] / 1000
stop_df['total_ivt_min'] = stop_df['bus_ivt_min'] + stop_df['train_ivt_min'] + stop_df['gtx_ivt_min']

print(f"  정류장 OD: {stop_df['od_pair'].nunique():,} ODs, {len(stop_df):,} rows")

# 2. H3 재집계 데이터
print("[2] H3 재집계 데이터 로드...")
h3_df = pd.read_parquet(EB_DIR / 'h3_choice_prob.parquet')

# 같은 IVT 데이터 병합
merge_cols = ['od_pair', 'alt_idx', 'bus_ivt_min', 'train_ivt_min', 'gtx_ivt_min',
              'total_ivt_min', 'ln_access', 'ln_egress', 'fare_1000won',
              'access_time_min', 'egress_time_min', 'transfer_walk_time_min']
h3_df = h3_df.merge(
    stop_df[merge_cols],
    on=['od_pair', 'alt_idx'], how='left', suffixes=('', '_dup')
)
# 중복 컬럼 제거
for c in list(h3_df.columns):
    if c.endswith('_dup'):
        h3_df.drop(columns=c, inplace=True)

# eb_walk 피처 확인
if 'ln_eb_access' not in h3_df.columns:
    if 'eb_access_time_min' in h3_df.columns:
        h3_df['ln_eb_access'] = np.log1p(h3_df['eb_access_time_min'])
        h3_df['ln_eb_egress'] = np.log1p(h3_df['eb_egress_time_min'])
    else:
        print("  WARNING: eb_access 피처 없음, aggregate_h3_choice.py를 다시 실행하세요.")

print(f"  H3 OD: {h3_df['h3_od'].nunique():,} ODs, {len(h3_df):,} rows")

# ============================================================
# Train/Test Split (동일한 기준)
# ============================================================
print("\n[3] Train/Test Split...")

# 정류장 OD 기준 split
od_dominant = stop_df.loc[
    stop_df.groupby('od_pair')['choice_prob'].idxmax(),
    ['od_pair', 'transport_category']
].set_index('od_pair')['transport_category']

od_strat = od_dominant.map(lambda c: 'gtx_related' if 'gtx' in c else c)
od_list = od_strat.index.to_numpy()
train_ods_arr, test_ods_arr = train_test_split(
    od_list, test_size=0.2, random_state=42, stratify=od_strat.values
)
train_od_set = set(train_ods_arr)
test_od_set = set(test_ods_arr)
for od in ['9007_9008', '9008_9007']:
    train_od_set.discard(od)
    test_od_set.add(od)

# 정류장 OD split
stop_train = stop_df[stop_df['od_pair'].isin(train_od_set)]
stop_test = stop_df[stop_df['od_pair'].isin(test_od_set)]

# H3 split: 정류장 OD 기준으로 split (H3 OD가 아닌 포함된 정류장 OD 기준)
h3_train = h3_df[h3_df['od_pair'].isin(train_od_set)]
h3_test = h3_df[h3_df['od_pair'].isin(test_od_set)]

print(f"  Stop OD - Train: {stop_train['od_pair'].nunique():,}, Test: {stop_test['od_pair'].nunique():,}")
print(f"  H3 OD   - Train: {h3_train['h3_od'].nunique():,}, Test: {h3_test['h3_od'].nunique():,}")

# ============================================================
# 모형 사양
# ============================================================
WALK_FEATURES = ['total_ivt_min', 'ln_access', 'ln_egress', 'transfer_walk_time_min',
                 'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx']
WALK_LABELS = ['Total IVT', 'ln(1+Access)', 'ln(1+Egress)', 'Transfer walk',
               'Transfers', 'Fare', 'Has bus', 'Has train', 'Has GTX']
WALK_NEG = {'total_ivt_min', 'ln_access', 'ln_egress', 'transfer_walk_time_min',
            'num_transfers', 'fare_1000won'}

NO_WALK_FEATURES = ['total_ivt_min', 'num_transfers', 'fare_1000won',
                    'has_bus', 'has_train', 'has_gtx']
NO_WALK_LABELS = ['Total IVT', 'Transfers', 'Fare', 'Has bus', 'Has train', 'Has GTX']
NO_WALK_NEG = {'total_ivt_min', 'num_transfers', 'fare_1000won'}

EB_WALK_FEATURES = ['total_ivt_min', 'ln_eb_access', 'ln_eb_egress',
                    'num_transfers', 'fare_1000won', 'has_bus', 'has_train', 'has_gtx']
EB_WALK_LABELS = ['Total IVT', 'ln(1+EB Access)', 'ln(1+EB Egress)',
                  'Transfers', 'Fare', 'Has bus', 'Has train', 'Has GTX']
EB_WALK_NEG = {'total_ivt_min', 'ln_eb_access', 'ln_eb_egress', 'num_transfers', 'fare_1000won'}

# ============================================================
# 4가지 모형 추정
# ============================================================
specs = [
    ('1. Stop OD + walk', stop_train, stop_test, 'od_pair', 'choice_prob', 'n_total',
     WALK_FEATURES, WALK_LABELS, WALK_NEG),
    ('2. Stop OD - walk', stop_train, stop_test, 'od_pair', 'choice_prob', 'n_total',
     NO_WALK_FEATURES, NO_WALK_LABELS, NO_WALK_NEG),
    ('3. H3 OD + OTP walk', h3_train, h3_test, 'h3_od', 'choice_prob', 'n_matched',
     WALK_FEATURES, WALK_LABELS, WALK_NEG),
    ('4. H3 OD + EB walk', h3_train, h3_test, 'h3_od', 'choice_prob', 'n_matched',
     EB_WALK_FEATURES, EB_WALK_LABELS, EB_WALK_NEG),
    ('5. H3 OD - walk', h3_train, h3_test, 'h3_od', 'choice_prob', 'n_matched',
     NO_WALK_FEATURES, NO_WALK_LABELS, NO_WALK_NEG),
]

all_results = {}

for name, train_data, test_data, od_col, prob_col, weight_col, features, labels, neg_con in specs:
    print(f'\n{"=" * 60}')
    print(f'{name} ({len(features)}개 피처)')
    print(f'{"=" * 60}')

    train_flat = prepare_flat_data(train_data, features, od_col, prob_col, weight_col)
    test_flat = prepare_flat_data(test_data, features, od_col, prob_col, weight_col)

    if train_flat is None or test_flat is None:
        print("  데이터 부족, 건너뜀")
        continue

    print(f'  Train: {train_flat["n_groups"]:,} groups, Test: {test_flat["n_groups"]:,} groups')

    r = fit_and_evaluate(train_flat, test_flat, features, labels, neg_con)
    all_results[name] = r

    print(f'\n  {"Feature":<22} {"β":>12} {"t-stat":>10}')
    print(f'  {"-"*46}')
    for label, b, t in zip(labels, r['beta'], r['t_stats']):
        sig = '***' if abs(t) > 2.576 else ''
        print(f'  {label:<22} {b:>12.6f} {t:>10.2f} {sig}')
    print(f'  {"-"*46}')
    print(f'  Train ρ²: {r["train_rho"]:.4f}')
    print(f'  Test ρ²:  {r["test_rho"]:.4f}')
    print(f'  Top-1:    {r["top1"]*100:.1f}%')

# ============================================================
# 비교 테이블
# ============================================================
print(f'\n{"=" * 70}')
print('모형 비교 요약')
print(f'{"=" * 70}')
print(f'{"Model":<25} {"Train ρ²":>10} {"Test ρ²":>10} {"Top-1":>8} {"Params":>8}')
print('-' * 63)
for name, r in all_results.items():
    n_params = len(r['beta'])
    print(f'{name:<25} {r["train_rho"]:>10.4f} {r["test_rho"]:>10.4f} {r["top1"]*100:>7.1f}% {n_params:>8}')

# 결과 저장
output = {}
for name, r in all_results.items():
    # 피처 목록 결정
    for sname, _, _, _, _, _, feats, _, _ in specs:
        if sname == name:
            feat_list = feats
            break
    output[name] = {
        'train_rho_sq': float(r['train_rho']),
        'test_rho_sq': float(r['test_rho']),
        'top1_accuracy': float(r['top1']),
        'n_train_groups': int(r['n_train']),
        'n_test_groups': int(r['n_test']),
        'coefficients': {f: float(b) for f, b in zip(feat_list, r['beta'])},
    }

out_path = EB_DIR / 'mnl_4way_comparison.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\n저장: {out_path}')
print('\n완료.')
