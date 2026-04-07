"""
MNL 4-Spec comparison (A/B/C/D x con/uncon = 8 specs).
Uses shared spec definitions from spec_config.py.
"""
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, approx_fprime

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code' / 'similarity'))

from spec_config import MNL_SPECS, CATEGORY_MAP, ASC_REF

DATA_PATH = ROOT / 'data' / 'training_set_new' / 'route_choice_filtered_training.parquet'
OUT_PATH = ROOT / 'data' / 'training_set_new' / 'mnl_4spec.json'


def load_data(path):
    df = pd.read_parquet(path)
    for col in ['ivt_bus', 'ivt_train', 'ivt_gtx',
                'waiting_time', 'transfer_walk_time',
                'in_vehicle_time', 'access_time', 'egress_time']:
        if col in df.columns:
            df[col] = df[col] / 60.0
    df = df.rename(columns={
        'waiting_time': 'wait_time_min',
        'transfer_walk_time': 'transfer_walk_time_min',
    })
    df['fare_1000won'] = df['fare'] / 1000.0
    df['total_ivt_min'] = df['in_vehicle_time']
    df['ln_access'] = np.log1p(df['access_time'])
    df['ln_egress'] = np.log1p(df['egress_time'])
    df['transport_category'] = df['transport_category'].map(CATEGORY_MAP)
    return df


def build_arrays(df, features):
    cats = sorted(df['transport_category'].unique())
    if ASC_REF in cats:
        cats.remove(ASC_REF)
    asc_names = [f'ASC_{c}' for c in cats]

    X_feat = df[features].values.astype(np.float64)
    asc_mat = np.zeros((len(df), len(cats)), dtype=np.float64)
    cat_vals = df['transport_category'].values
    for k, c in enumerate(cats):
        asc_mat[:, k] = (cat_vals == c).astype(np.float64)
    X = np.hstack([X_feat, asc_mat])

    y = df['chosen'].values.astype(np.float64)
    gid_codes, _ = pd.factorize(df['od_pair'], sort=False)
    gid = gid_codes.astype(np.int64)
    ng = gid.max() + 1
    return X, y, gid, ng, features + asc_names


def mnl_neg_ll(beta, X, y, gid, ng):
    V = X @ beta
    V_max = np.full(ng, -np.inf)
    np.maximum.at(V_max, gid, V)
    V_s = V - V_max[gid]
    exp_V = np.exp(V_s)
    sum_exp = np.bincount(gid, weights=exp_V, minlength=ng)
    log_p = V_s - np.log(sum_exp[gid])
    return -np.sum(y * log_p)


def mnl_grad(beta, X, y, gid, ng):
    V = X @ beta
    V_max = np.full(ng, -np.inf)
    np.maximum.at(V_max, gid, V)
    V_s = V - V_max[gid]
    exp_V = np.exp(V_s)
    sum_exp = np.bincount(gid, weights=exp_V, minlength=ng)
    prob = exp_V / sum_exp[gid]
    return X.T @ (prob - y)


def fit_mnl(X, y, gid, ng, feat_names, sign_neg):
    n_p = X.shape[1]
    bounds = []
    for f in feat_names:
        if f in sign_neg:
            bounds.append((None, 0))
        else:
            bounds.append((None, None))

    beta0 = np.zeros(n_p)
    res = minimize(
        mnl_neg_ll, beta0, args=(X, y, gid, ng),
        jac=mnl_grad, method='L-BFGS-B',
        bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12},
    )

    # Hessian -> SE
    eps = 1e-5
    H = np.zeros((n_p, n_p))
    for i in range(n_p):
        def grad_i(b, _i=i):
            return mnl_grad(b, X, y, gid, ng)[_i]
        H[i, :] = approx_fprime(res.x, grad_i, eps)
    H = (H + H.T) / 2
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t_stat = res.x / np.where(se > 0, se, 1.0)

    gs = np.bincount(gid, minlength=ng)
    ll_0 = -float(np.sum(np.log(gs)))
    ll_beta = -res.fun
    rho_sq = 1 - ll_beta / ll_0

    return res.x, se, t_stat, ll_0, ll_beta, rho_sq, res.success


def fast_top1(X, y, gid, ng, beta):
    V = X @ beta
    df_tmp = pd.DataFrame({'gid': gid, 'V': V, 'y': y})
    pred_idx = df_tmp.groupby('gid')['V'].idxmax()
    true_idx = df_tmp.groupby('gid')['y'].idxmax()
    return (pred_idx == true_idx).mean()


def run_spec(df, spec_name, spec_def):
    features = spec_def['features']
    sign_neg = spec_def['sign_neg']

    X, y, gid, ng, all_names = build_arrays(df, features)
    beta, se, t_stat, ll_0, ll_beta, rho_sq, success = fit_mnl(
        X, y, gid, ng, all_names, sign_neg
    )
    top1 = fast_top1(X, y, gid, ng, beta)

    return {
        'spec': spec_name, 'success': success,
        'beta': beta, 'se': se, 't_stat': t_stat,
        'names': all_names, 'sign_neg': sign_neg,
        'll_0': ll_0, 'll_beta': ll_beta,
        'rho_sq': rho_sq, 'top1': top1,
        'n_ods': ng, 'n_alts': len(y), 'n_params': len(beta),
    }


def save_results(results, path):
    """Save results as JSON (convert numpy types)."""
    out = []
    for r in results:
        out.append({
            'spec': r['spec'], 'success': bool(r['success']),
            'rho_sq': float(r['rho_sq']), 'top1': float(r['top1']),
            'll_0': float(r['ll_0']), 'll_beta': float(r['ll_beta']),
            'n_ods': int(r['n_ods']), 'n_alts': int(r['n_alts']),
            'n_params': int(r['n_params']),
            'names': r['names'],
            'beta': [float(b) for b in r['beta']],
            'se': [float(s) for s in r['se']],
            't_stat': [float(t) for t in r['t_stat']],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to {path}')


def print_summary(results):
    print('\n' + '=' * 90)
    print('SUMMARY')
    print('=' * 90)
    print(f'{"Spec":<28} {"rho-sq":>8} {"Top-1":>8} {"LL(b)":>14} {"K":>4}')
    print('-' * 70)
    for r in results:
        print(f'{r["spec"]:<28} {r["rho_sq"]:>8.4f} {r["top1"]*100:>7.1f}% '
              f'{r["ll_beta"]:>14,.2f} {r["n_params"]:>4}')


def print_betas(results):
    for r in results:
        print(f'\n--- {r["spec"]} ---')
        print(f'{"Parameter":<26} {"beta":>12} {"SE":>12} {"t-stat":>10}')
        print('-' * 62)
        for i, name in enumerate(r['names']):
            on_bound = name in r['sign_neg'] and abs(r['beta'][i]) < 1e-10
            if on_bound:
                print(f'{name:<26} {r["beta"][i]:>12.6f} {"BOUND":>12} {"":>10}')
            else:
                print(f'{name:<26} {r["beta"][i]:>12.6f} '
                      f'{r["se"][i]:>12.6f} {r["t_stat"][i]:>10.3f}')


if __name__ == '__main__':
    print('Loading data...')
    df = load_data(DATA_PATH)
    print(f'{len(df):,} rows, {df["od_pair"].nunique():,} ODs\n')

    results = []
    for spec_name, spec_def in MNL_SPECS.items():
        print(f'--- {spec_name} ---')
        t0 = time.time()
        r = run_spec(df, spec_name, spec_def)
        elapsed = time.time() - t0
        print(f'  rho-sq={r["rho_sq"]:.4f}, Top-1={r["top1"]:.4f}, '
              f'time={elapsed:.1f}s, converged={r["success"]}')
        results.append(r)

    print_summary(results)
    print_betas(results)
    save_results(results, OUT_PATH)
