# -*- coding: utf-8 -*-
"""
Step 5: H3 OD MNL with external access/egress features

교수님 지시: 기존 OTP 결과에 first/last mile만 매핑, H3 셀 단위로 합침
- 기존 K3 피처 (IVT, transfers, fare, mode dummies)
- OTP walk (ln_access, ln_egress) → 기존 그대로
- 외부 접근/이탈 (ln_access_ext, ln_egress_ext) → 신규 추가

비교 사양:
  A. H3 OD + OTP walk only (기존)
  B. H3 OD + ext access/egress only (OTP walk 제거)
  C. H3 OD + OTP walk + ext access/egress (전부)

Usage:
    python run_h3_ext_access.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

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
# MNL core functions
# ============================================================
def prepare_flat_data(data, features, od_col='h3_od', prob_col='choice_prob', weight_col=None):
    """Convert data to flat arrays for vectorized MNL."""
    X_list, y_list, w_list, gid_list = [], [], [], []
    group_id = 0
    n_skipped = 0
    for _, grp in data.groupby(od_col):
        y = grp[prob_col].values.astype(np.float64)
        if abs(y.sum() - 1.0) > 0.01 or y.sum() == 0:
            n_skipped += 1
            continue
        X = grp[features].values.astype(np.float64)
        if weight_col and weight_col in grp.columns:
            w = float(grp[weight_col].iloc[0])
        else:
            w = 1.0
        n = len(grp)
        X_list.append(X)
        y_list.append(y)
        w_list.append(np.full(n, w))
        gid_list.append(np.full(n, group_id, dtype=np.int64))
        group_id += 1

    if not X_list:
        return None
    flat = {
        'X': np.vstack(X_list), 'y': np.concatenate(y_list),
        'w': np.concatenate(w_list), 'gid': np.concatenate(gid_list),
        'n_groups': group_id,
    }
    if n_skipped:
        print(f'  choice_prob 합 ≠ 1.0 → {n_skipped:,} groups 제외')
    return flat


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


def fit_mnl(flat, features, sign_neg, max_iter=2000):
    """Fit MNL with sign constraints, return results dict."""
    bounds = []
    for f in features:
        if f in sign_neg:
            bounds.append((None, 0))
        else:
            bounds.append((None, None))

    beta0 = np.zeros(len(features))
    result = minimize(
        mnl_neg_ll, beta0, args=(flat,),
        jac=mnl_gradient, method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12}
    )

    beta_hat = result.x
    n_params = len(beta_hat)

    # Hessian → SE → t-stat
    eps = 1e-5
    hessian = np.zeros((n_params, n_params))
    for i in range(n_params):
        def grad_i(b, _i=i):
            return mnl_gradient(b, flat)[_i]
        hessian[i, :] = approx_fprime(beta_hat, grad_i, eps)
    hessian = (hessian + hessian.T) / 2

    try:
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hessian)
    se = np.sqrt(np.abs(np.diag(cov)))
    t_stat = beta_hat / np.where(se > 0, se, 1)

    # LL(0), ρ²
    gid = flat['gid']
    w = flat['w']
    ng = flat['n_groups']
    gs = np.bincount(gid, minlength=ng)
    w_g = np.bincount(gid, weights=w, minlength=ng) / gs
    ll_0 = -np.sum(w_g * np.log(gs))
    ll_beta = -result.fun
    rho_sq = 1 - ll_beta / ll_0

    return {
        'beta': beta_hat, 'se': se, 't_stat': t_stat,
        'll_0': ll_0, 'll_beta': ll_beta, 'rho_sq': rho_sq,
        'converged': result.success, 'n_groups': flat['n_groups'],
        'bounds': bounds, 'sign_neg': sign_neg,
    }


def evaluate_model(beta, data, features, od_col='h3_od', prob_col='choice_prob'):
    """Top-1, Top-3 accuracy."""
    correct_1 = correct_3 = total = 0
    for _, grp in data.groupby(od_col):
        y = grp[prob_col].values
        if y.sum() == 0:
            continue
        X = grp[features].values.astype(np.float64)
        V = X @ beta
        pred_rank = np.argsort(-V)
        actual_best = np.argmax(y)
        if pred_rank[0] == actual_best:
            correct_1 += 1
        if actual_best in pred_rank[:3]:
            correct_3 += 1
        total += 1
    return {
        'top1': correct_1 / total if total else 0,
        'top3': correct_3 / total if total else 0,
        'n_ods': total,
    }


def print_results(name, features, labels, res, eval_train, eval_test):
    """Print formatted MNL results."""
    print(f'\n{"=" * 80}')
    print(f'  {name}')
    print(f'{"=" * 80}')
    print(f'{"Feature":<28} {"β":>12} {"SE":>12} {"t-stat":>10} {"Sig":>5}')
    print('-' * 72)
    for i, (f, l, b, se, t) in enumerate(zip(features, labels, res['beta'], res['se'], res['t_stat'])):
        sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.645 else ''
        on_bound = f in res['sign_neg'] and abs(b) < 1e-10
        if on_bound:
            print(f'{l:<28} {b:>12.6f} {"(bound)":>12} {"":>10} {"":>5}')
        else:
            print(f'{l:<28} {b:>12.6f} {se:>12.6f} {t:>10.3f} {sig:>5}')
    print('-' * 72)
    print(f'LL(0): {res["ll_0"]:.2f}  |  LL(β): {res["ll_beta"]:.2f}  |  ρ²: {res["rho_sq"]:.4f}')
    print(f'N(groups): {res["n_groups"]:,}  |  Converged: {res["converged"]}')
    print(f'Train Top-1: {eval_train["top1"]:.4f}  |  Train Top-3: {eval_train["top3"]:.4f}')
    print(f'Test  Top-1: {eval_test["top1"]:.4f}  |  Test  Top-3: {eval_test["top3"]:.4f}')

    # Sign check
    print('\nSign check:')
    for f, l, b in zip(features, labels, res['beta']):
        if f in res['sign_neg']:
            ok = 'OK' if b <= 1e-10 else 'FAIL'
            print(f'  {l:<28}: β={b:>10.6f} ≤ 0 → {ok}')


def main():
    # ============================================================
    # 1. 데이터 로드
    # ============================================================
    print('[1/5] 데이터 로드...')
    h3_path = EB_DIR / 'h3_choice_prob.parquet'
    df = pd.read_parquet(h3_path)
    print(f'  H3 집계 데이터: {len(df):,}행, {df["h3_od"].nunique():,} H3 OD')

    # 2. 모드별 IVT 추출
    print('\n[2/5] 모드별 IVT 추출...')
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
    print(f'  bus_ivt={df["bus_ivt_min"].mean():.2f}, train_ivt={df["train_ivt_min"].mean():.2f}, gtx_ivt={df["gtx_ivt_min"].mean():.2f}')

    # 3. 피처 생성
    print('\n[3/5] 피처 생성...')
    for col in ['access_time', 'egress_time', 'transfer_walk_time']:
        df[col + '_min'] = df[col] / 60
    df['ln_access'] = np.log1p(df['access_time_min'])
    df['ln_egress'] = np.log1p(df['egress_time_min'])
    df['fare_1000won'] = df['fare'] / 1000
    df['total_ivt_min'] = df['bus_ivt_min'] + df['train_ivt_min'] + df['gtx_ivt_min']

    # ext access/egress 로그 변환 (이미 있지만 이름 통일)
    df['ln_access_ext'] = df['ln_eb_access']
    df['ln_egress_ext'] = df['ln_eb_egress']

    # H3 OD 내 ext 변이 통계
    var_ext = df.groupby('h3_od')['eb_access_dist'].std().fillna(0)
    pct_vary = (var_ext > 0).mean() * 100
    print(f'  H3 OD 내 ext_access 변이 있는 비율: {pct_vary:.1f}%')

    # 4. Train/Test split (H3 OD 단위)
    print('\n[4/5] Train/Test split...')
    h3_ods = np.array(df['h3_od'].unique())
    train_ods, test_ods = train_test_split(h3_ods, test_size=0.2, random_state=42)
    train_set = set(train_ods)
    test_set = set(test_ods)
    train_df = df[df['h3_od'].isin(train_set)].copy()
    test_df = df[df['h3_od'].isin(test_set)].copy()
    print(f'  Train: {len(train_df):,}행, {len(train_set):,} H3 ODs')
    print(f'  Test:  {len(test_df):,}행, {len(test_set):,} H3 ODs')

    # Compute n_total per H3 OD for weighting
    # n_total = sum of n_matched across all routes in H3 OD
    h3_n_total = df.groupby('h3_od')['n_matched'].sum().rename('h3_n_total')
    train_df = train_df.merge(h3_n_total, on='h3_od', how='left')
    test_df = test_df.merge(h3_n_total, on='h3_od', how='left')

    # ============================================================
    # 5. 3 사양 MNL 비교
    # ============================================================
    print('\n[5/5] MNL 추정...')

    SIGN_NEG = {
        'total_ivt_min', 'ln_access', 'ln_egress',
        'transfer_walk_time_min', 'num_transfers', 'fare_1000won',
        'ln_access_ext', 'ln_egress_ext',
    }

    specs = {
        'A: H3 + OTP walk': {
            'features': [
                'total_ivt_min', 'ln_access', 'ln_egress',
                'transfer_walk_time_min', 'num_transfers',
                'fare_1000won', 'has_bus', 'has_train', 'has_gtx',
            ],
            'labels': [
                'Total IVT (min)', 'ln(1+OTP access)', 'ln(1+OTP egress)',
                'Transfer walk (min)', 'Transfers',
                'Fare (1000won)', 'Has bus', 'Has train', 'Has GTX',
            ],
        },
        'B: H3 + ext access/egress': {
            'features': [
                'total_ivt_min', 'ln_access_ext', 'ln_egress_ext',
                'transfer_walk_time_min', 'num_transfers',
                'fare_1000won', 'has_bus', 'has_train', 'has_gtx',
            ],
            'labels': [
                'Total IVT (min)', 'ln(1+ext access)', 'ln(1+ext egress)',
                'Transfer walk (min)', 'Transfers',
                'Fare (1000won)', 'Has bus', 'Has train', 'Has GTX',
            ],
        },
        'C: H3 + OTP walk + ext access': {
            'features': [
                'total_ivt_min', 'ln_access', 'ln_egress',
                'ln_access_ext', 'ln_egress_ext',
                'transfer_walk_time_min', 'num_transfers',
                'fare_1000won', 'has_bus', 'has_train', 'has_gtx',
            ],
            'labels': [
                'Total IVT (min)', 'ln(1+OTP access)', 'ln(1+OTP egress)',
                'ln(1+ext access)', 'ln(1+ext egress)',
                'Transfer walk (min)', 'Transfers',
                'Fare (1000won)', 'Has bus', 'Has train', 'Has GTX',
            ],
        },
    }

    results_summary = []

    for name, spec in specs.items():
        features = spec['features']
        labels = spec['labels']
        print(f'\n--- {name} ---')

        train_flat = prepare_flat_data(
            train_df, features, od_col='h3_od',
            prob_col='choice_prob', weight_col='h3_n_total'
        )
        if train_flat is None:
            print('  ERROR: 유효한 그룹 없음')
            continue

        print(f'  유효 그룹: {train_flat["n_groups"]:,}, 대안: {len(train_flat["y"]):,}')

        res = fit_mnl(train_flat, features, SIGN_NEG)
        eval_train = evaluate_model(res['beta'], train_df, features, od_col='h3_od')
        eval_test = evaluate_model(res['beta'], test_df, features, od_col='h3_od')

        print_results(name, features, labels, res, eval_train, eval_test)

        results_summary.append({
            'name': name,
            'n_features': len(features),
            'rho_sq': res['rho_sq'],
            'train_top1': eval_train['top1'],
            'test_top1': eval_test['top1'],
            'train_top3': eval_train['top3'],
            'test_top3': eval_test['top3'],
        })

    # ============================================================
    # 비교 요약 테이블
    # ============================================================
    print(f'\n\n{"=" * 80}')
    print('  비교 요약')
    print(f'{"=" * 80}')
    print(f'{"Spec":<35} {"K":>3} {"ρ²":>8} {"Train T1":>10} {"Test T1":>10} {"Test T3":>10}')
    print('-' * 80)
    for r in results_summary:
        print(f'{r["name"]:<35} {r["n_features"]:>3} {r["rho_sq"]:>8.4f} '
              f'{r["train_top1"]:>10.4f} {r["test_top1"]:>10.4f} {r["test_top3"]:>10.4f}')


if __name__ == '__main__':
    main()
