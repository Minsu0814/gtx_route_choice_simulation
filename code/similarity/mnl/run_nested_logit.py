"""Nested Logit 경로선택모형 — K3 피처 + nest(bus/train/gtx)."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import sqlite3
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('../../../')
DATA_DIR = ROOT / 'data' / 'training_set'

# ========== 1. 데이터 로드 ==========
df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')
print(f'집계 데이터: {len(df):,} rows, {df["od_pair"].nunique()} ODs')

# 모드별 IVT 추출
if 'alt_idx' not in df.columns:
    df['alt_idx'] = df.groupby('od_pair').cumcount()
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
df = df.merge(ivt_df, on=['od_pair', 'alt_idx'], how='left')
for c in ['bus_ivt_min', 'train_ivt_min', 'gtx_ivt_min']:
    df[c] = df[c].fillna(0)
del ivt_records, ivt_df

# 피처 생성 (K3)
for col in ['access_time', 'egress_time', 'transfer_walk_time']:
    df[col + '_min'] = df[col] / 60
df['ln_access'] = np.log1p(df['access_time_min'])
df['ln_egress'] = np.log1p(df['egress_time_min'])
df['fare_1000won'] = df['fare'] / 1000
df['total_ivt_min'] = df['bus_ivt_min'] + df['train_ivt_min'] + df['gtx_ivt_min']

MODEL_FEATURES = [
    'total_ivt_min',
    'ln_access', 'ln_egress',
    'transfer_walk_time_min',
    'num_transfers',
    'fare_1000won',
    'has_bus', 'has_train', 'has_gtx',
]
FEATURE_LABELS = [
    'Total IVT (min)',
    'ln(1+Access walk)', 'ln(1+Egress walk)',
    'Transfer walk (min)',
    'Transfers',
    'Fare (1000won)',
    'Has bus', 'Has train', 'Has GTX',
]
SIGN_CONSTRAINED_NEGATIVE = {
    'total_ivt_min', 'ln_access', 'ln_egress',
    'transfer_walk_time_min', 'num_transfers', 'fare_1000won',
}

# ========== Nest 할당 ==========
# 주 수단 기반: bus_only→bus, train_only→train, gtx 포함→gtx, 혼합→dominant IVT 기준
def assign_nest(row):
    cat = row['transport_category']
    if 'gtx' in cat:
        return 'gtx'
    elif cat == 'bus_only':
        return 'bus'
    elif cat == 'train_only':
        return 'train'
    else:  # bus+train → IVT가 큰 쪽
        if row['train_ivt_min'] >= row['bus_ivt_min']:
            return 'train'
        else:
            return 'bus'

df['nest'] = df.apply(assign_nest, axis=1)
print(f'\nNest 분포:')
print(df['nest'].value_counts().to_string())
print()

NESTS = ['bus', 'train', 'gtx']
nest_to_id = {n: i for i, n in enumerate(NESTS)}
df['nest_id'] = df['nest'].map(nest_to_id)

# ========== 2. Train/Test Split ==========
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

gtx_move = ['9007_9008', '9008_9007']
for od in gtx_move:
    train_od_set.discard(od)
    test_od_set.add(od)

train_df = df[df['od_pair'].isin(train_od_set)].copy()
test_df = df[df['od_pair'].isin(test_od_set)].copy()
print(f'Train: {len(train_df):,} rows, {train_df["od_pair"].nunique()} ODs')
print(f'Test:  {len(test_df):,} rows, {test_df["od_pair"].nunique()} ODs')

# ========== 3. Nested Logit 구현 ==========
N_FEATURES = len(MODEL_FEATURES)
N_NESTS = len(NESTS)
# 파라미터: beta(N_FEATURES) + mu(N_NESTS)
# mu = nest scale parameter, 0 < mu <= 1 (1이면 MNL과 동일)


def prepare_nested_data(data, features):
    """Nested Logit용 데이터 준비."""
    X_list, y_list, w_list, nest_list, gid_list = [], [], [], [], []
    group_id = 0
    for od, grp in data.groupby('od_pair'):
        y = grp['choice_prob'].values.astype(np.float64)
        if abs(y.sum() - 1.0) > 0.01:
            continue
        X = grp[features].values.astype(np.float64)
        w = float(grp['n_total'].iloc[0])
        n = len(grp)
        nests = grp['nest_id'].values.astype(np.int64)

        X_list.append(X)
        y_list.append(y)
        w_list.append(np.full(n, w))
        nest_list.append(nests)
        gid_list.append(np.full(n, group_id, dtype=np.int64))
        group_id += 1

    return {
        'X': np.vstack(X_list),
        'y': np.concatenate(y_list),
        'w': np.concatenate(w_list),
        'nests': np.concatenate(nest_list),
        'gid': np.concatenate(gid_list),
        'n_groups': group_id,
    }


def nested_logit_prob(params, flat):
    """Nested Logit 선택확률 계산.

    P(i) = [exp(V_i/mu_k) / sum_j_in_k exp(V_j/mu_k)]
          × [IV_k^mu_k / sum_m IV_m^mu_m]

    where IV_k = sum_j_in_k exp(V_j/mu_k)  (inclusive value)
    """
    beta = params[:N_FEATURES]
    mu_raw = params[N_FEATURES:]  # will be transformed to (0, 1]

    X, y, w, nests, gid, ng = (
        flat['X'], flat['y'], flat['w'], flat['nests'], flat['gid'], flat['n_groups']
    )

    # mu: sigmoid로 (0, 1] 범위 보장
    mu = 1.0 / (1.0 + np.exp(-mu_raw))  # sigmoid
    mu = np.clip(mu, 0.01, 1.0)

    V = X @ beta  # (N,) utility

    # 각 대안의 nest별 mu
    mu_alt = mu[nests]  # (N,)

    # V / mu
    V_scaled = V / mu_alt

    # 오버플로 방지: group별 max 빼기
    V_max = np.full(ng, -np.inf)
    np.maximum.at(V_max, gid, V_scaled)
    V_shifted = V_scaled - V_max[gid]

    exp_V = np.exp(V_shifted)

    # nest-group별 합 (inclusive value 분자)
    # key: (gid, nest_id) → sum of exp(V_shifted)
    nest_gid_key = gid * N_NESTS + nests  # unique key per (group, nest)
    max_key = ng * N_NESTS
    IV_sum = np.bincount(nest_gid_key, weights=exp_V, minlength=max_key).astype(np.float64)

    # P(i|nest_k) = exp(V_i/mu_k) / sum_j_in_k exp(V_j/mu_k)
    p_within = exp_V / np.maximum(IV_sum[nest_gid_key], 1e-30)

    # IV_k = log(sum_j_in_k exp(V_j/mu_k)) + V_max  (log scale)
    # But we need IV_k^mu_k for between-nest
    # log(IV_k^mu_k) = mu_k * log(IV_sum_k) + mu_k * V_max_gid
    # 간소화: IV_k_raw = IV_sum[gid, nest] (shifted scale에서)
    # IV_k^mu_k = IV_sum^mu_k * exp(mu_k * V_max)

    # nest별 inclusive value (log scale)
    log_IV = np.log(np.maximum(IV_sum, 1e-30))  # (ng * N_NESTS,)

    # 각 nest의 mu
    mu_nest = np.tile(mu, ng)  # (ng * N_NESTS,)

    # between-nest: log(IV_k^mu_k) = mu_k * (log_IV_k + V_max_gid)
    V_max_tiled = np.repeat(V_max, N_NESTS)
    log_IV_powered = mu_nest * (log_IV + V_max_tiled)

    # nest가 비어있으면 -inf 처리
    nest_exists = IV_sum > 0
    log_IV_powered = np.where(nest_exists, log_IV_powered, -1e30)

    # group별 between-nest logsumexp
    log_IV_powered_by_group = log_IV_powered.reshape(ng, N_NESTS)
    max_log_IV = np.max(log_IV_powered_by_group, axis=1, keepdims=True)
    log_denom = max_log_IV.squeeze() + np.log(
        np.sum(np.exp(log_IV_powered_by_group - max_log_IV) * (IV_sum.reshape(ng, N_NESTS) > 0), axis=1)
    )

    # P(nest_k) = IV_k^mu_k / sum_m IV_m^mu_m
    nest_gid_group = gid * N_NESTS + nests
    p_nest_log = log_IV_powered[nest_gid_key] - log_denom[gid]
    p_nest = np.exp(p_nest_log)

    prob = p_within * p_nest
    prob = np.clip(prob, 1e-30, 1.0)

    return prob


def nested_logit_neg_ll(params, flat):
    """Weighted negative log-likelihood."""
    prob = nested_logit_prob(params, flat)
    log_prob = np.log(np.maximum(prob, 1e-30))
    return -np.sum(flat['w'] * flat['y'] * log_prob)


print('\n[3/5] Preparing data...')
train_flat = prepare_nested_data(train_df, MODEL_FEATURES)
print(f'유효 choice sets (train): {train_flat["n_groups"]:,}')
print(f'총 대안 수: {len(train_flat["y"]):,}')

# Nest 분포
for i, n in enumerate(NESTS):
    cnt = np.sum(train_flat['nests'] == i)
    print(f'  {n}: {cnt:,} ({cnt/len(train_flat["y"])*100:.1f}%)')

# ========== 4. 추정 ==========
print('\n[4/5] Estimating Nested Logit...')

# 초기값: MNL K3 beta + mu=0 (sigmoid(0)=0.5)
beta0 = np.zeros(N_FEATURES)
mu0 = np.zeros(N_NESTS)  # sigmoid(0) = 0.5
params0 = np.concatenate([beta0, mu0])

# 부호 제약: beta는 부호 제약, mu는 자유 (sigmoid로 변환)
bounds = []
for feat in MODEL_FEATURES:
    if feat in SIGN_CONSTRAINED_NEGATIVE:
        bounds.append((None, 0))
    else:
        bounds.append((None, None))
for _ in range(N_NESTS):
    bounds.append((None, None))  # mu_raw: unconstrained (sigmoid 변환)

result = minimize(
    nested_logit_neg_ll, params0, args=(train_flat,),
    method='L-BFGS-B', bounds=bounds,
    options={'maxiter': 2000, 'ftol': 1e-12, 'disp': True}
)

print(f'\n수렴: {result.success} ({result.message})')
print(f'반복: {result.nit}')
print(f'LL(β): {-result.fun:.4f}')

# ========== 5. 결과 출력 ==========
beta_hat = result.x[:N_FEATURES]
mu_raw = result.x[N_FEATURES:]
mu_hat = 1.0 / (1.0 + np.exp(-mu_raw))
mu_hat = np.clip(mu_hat, 0.01, 1.0)

# Hessian → 표준오차
print('\n표준오차 계산 중...')
eps = 1e-5
n_params = len(result.x)
hessian = np.zeros((n_params, n_params))
for i in range(n_params):
    def grad_i(p, _i=i):
        delta = np.zeros(n_params)
        delta[_i] = eps
        return (nested_logit_neg_ll(p + delta, train_flat) -
                nested_logit_neg_ll(p - delta, train_flat)) / (2 * eps)
    hessian[i, :] = approx_fprime(result.x, grad_i, eps)

hessian = (hessian + hessian.T) / 2
try:
    cov = np.linalg.inv(hessian)
    std_errors = np.sqrt(np.abs(np.diag(cov)))
except np.linalg.LinAlgError:
    cov = np.linalg.pinv(hessian)
    std_errors = np.sqrt(np.abs(np.diag(cov)))

t_stats = result.x / np.where(std_errors > 0, std_errors, 1)

# LL(0)
gid = train_flat['gid']
w = train_flat['w']
ng = train_flat['n_groups']
group_sizes = np.bincount(gid, minlength=ng)
w_per_group = np.bincount(gid, weights=w, minlength=ng) / group_sizes
ll_0 = -np.sum(w_per_group * np.log(group_sizes))
ll_beta = -result.fun
rho_sq = 1 - ll_beta / ll_0

train_trips = train_df.groupby('od_pair')['n_total'].first().sum()

print('\n' + '=' * 80)
print('Nested Logit 추정 결과 (K3 피처 + bus/train/gtx nest)')
print('=' * 80)
print(f'{"Parameter":<25} {"Value":>12} {"Std.Err":>12} {"t-stat":>10} {"Sig":>5}')
print('-' * 80)

# Beta
for i, (label, b) in enumerate(zip(FEATURE_LABELS, beta_hat)):
    se = std_errors[i]
    t = t_stats[i]
    sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.960 else '*' if abs(t) > 1.645 else ''
    on_bound = MODEL_FEATURES[i] in SIGN_CONSTRAINED_NEGATIVE and abs(b) < 1e-10
    if on_bound:
        print(f'{label:<25} {b:>12.6f} {"N/A":>12} {"N/A":>10} {"":>5}  ← bound')
    else:
        print(f'{label:<25} {b:>12.6f} {se:>12.6f} {t:>10.3f} {sig:>5}')

# Mu (nest parameters)
print('-' * 80)
print('Nest Scale Parameters (mu):')
for i, nest_name in enumerate(NESTS):
    idx = N_FEATURES + i
    se = std_errors[idx]
    t = t_stats[idx]
    sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.960 else '*' if abs(t) > 1.645 else ''
    print(f'  mu_{nest_name:<20} {mu_hat[i]:>12.4f} {se:>12.6f} {t:>10.3f} {sig:>5}'
          f'  (raw={mu_raw[i]:.4f})')

print('-' * 80)
print(f'LL(0):       {ll_0:.4f}')
print(f'LL(β):       {ll_beta:.4f}')
print(f'McFadden ρ²: {rho_sq:.4f}')
print(f'N (ODs):     {train_flat["n_groups"]:,}')
print(f'N (trips):   {train_trips:,} (weighted)')
print(f'K (params):  {n_params} ({N_FEATURES} beta + {N_NESTS} mu)')

# mu 해석
print('\n=== Nest Parameter 해석 ===')
for i, n in enumerate(NESTS):
    if mu_hat[i] > 0.95:
        interp = 'MNL과 거의 동일 (nest 내 상관 낮음)'
    elif mu_hat[i] > 0.5:
        interp = 'nest 내 약한 상관'
    elif mu_hat[i] > 0.2:
        interp = 'nest 내 강한 상관 (IIA 위반 큼)'
    else:
        interp = 'nest 내 매우 강한 상관'
    print(f'  mu_{n} = {mu_hat[i]:.4f}: {interp}')

# 부호 검증
print('\n=== 부호 검증 ===')
all_ok = True
for feat, label, b in zip(MODEL_FEATURES, FEATURE_LABELS, beta_hat):
    if feat in SIGN_CONSTRAINED_NEGATIVE:
        ok = 'OK' if b <= 0 else 'FAIL'
        if ok != 'OK':
            all_ok = False
        print(f'  {label:<22}: β={b:>10.6f}  ≤ 0 → {ok}')
print(f'\n전체 부호: {"ALL CORRECT" if all_ok else "CHECK NEEDED"}')

# ========== 6. 테스트셋 평가 ==========
print('\n[5/5] 테스트셋 평가...')
test_flat = prepare_nested_data(test_df, MODEL_FEATURES)
print(f'유효 choice sets (test): {test_flat["n_groups"]:,}')

pred_prob = nested_logit_prob(result.x, test_flat)
actual_y = test_flat['y']
gid_test = test_flat['gid']
ng_test = test_flat['n_groups']

# Test LL
test_ll = np.sum(test_flat['w'] * test_flat['y'] * np.log(np.maximum(pred_prob, 1e-30)))
test_group_sizes = np.bincount(gid_test, minlength=ng_test)
test_w_per_group = np.bincount(gid_test, weights=test_flat['w'], minlength=ng_test) / test_group_sizes
test_ll_0 = -np.sum(test_w_per_group * np.log(test_group_sizes))
test_rho_sq = 1 - test_ll / test_ll_0

# Top-1
top1_correct = 0
for g in range(ng_test):
    g_mask = gid_test == g
    if np.argmax(pred_prob[g_mask]) == np.argmax(actual_y[g_mask]):
        top1_correct += 1
top1_acc = top1_correct / ng_test

# Top-3
top3_correct = 0
for g in range(ng_test):
    g_mask = gid_test == g
    g_pred = pred_prob[g_mask]
    g_actual = actual_y[g_mask]
    k = min(3, len(g_pred))
    top_k_idx = np.argsort(g_pred)[-k:]
    if np.argmax(g_actual) in top_k_idx:
        top3_correct += 1
top3_acc = top3_correct / ng_test

# RMSE
rmse = np.sqrt(np.mean((pred_prob - actual_y) ** 2))

print('\n' + '=' * 60)
print('테스트셋 평가 결과')
print('=' * 60)
print(f'Test LL(β):          {test_ll:.4f}')
print(f'Test LL(0):          {test_ll_0:.4f}')
print(f'Test McFadden ρ²:    {test_rho_sq:.4f}')
print(f'Top-1 Accuracy (FPR): {top1_acc:.4f} ({top1_correct}/{ng_test})')
print(f'Top-3 Accuracy:       {top3_acc:.4f} ({top3_correct}/{ng_test})')
print(f'RMSE:                 {rmse:.4f}')
print(f'Train ρ²:             {rho_sq:.4f} (과적합 확인)')

# MNL K3 vs Nested Logit 비교
print('\n=== MNL K3 vs Nested Logit 비교 ===')
print(f'{"지표":<20} {"MNL K3":>12} {"Nested":>12} {"변화":>12}')
print('-' * 60)
mnl_rho = 0.5304
mnl_top1 = 0.7250
mnl_rmse = 0.2451
print(f'{"Test ρ²":<20} {mnl_rho:>12.4f} {test_rho_sq:>12.4f} {test_rho_sq-mnl_rho:>+12.4f}')
print(f'{"Top-1 FPR":<20} {mnl_top1*100:>11.2f}% {top1_acc*100:>11.2f}% {(top1_acc-mnl_top1)*100:>+11.2f}%')
print(f'{"RMSE":<20} {mnl_rmse:>12.4f} {rmse:>12.4f} {rmse-mnl_rmse:>+12.4f}')

print('\nDone.')
