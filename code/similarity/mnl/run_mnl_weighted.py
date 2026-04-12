"""
Weighted MNL estimation using choice_prob (A_total_uncon only).

Unlike binary MNL (chosen=0/1), this uses:
  y = choice_prob (0.0~1.0)
  w = n_total (trip count per OD, used as weight)
  LL = -sum(w * y * log_p)
  grad = X.T @ (w * (prob - y))
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

from spec_config import CATEGORY_MAP, ASC_REF

DATA_PATH = (ROOT / 'data' / 'choice_set' / 'otp'
             / 'route_choice_filtered_training_weighted.parquet')
OUT_PATH = ROOT / 'data' / 'results' / 'mnl_weighted.json'

# A_total_uncon: total IVT, no walk, unconstrained
FEATURES = ['total_ivt_min', 'transfer_walk_time_min',
            'num_transfers', 'fare_1000won']


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
    df['transport_category'] = df['transport_category'].map(CATEGORY_MAP)
    return df


def build_arrays(df, features):
    """Build X, y (choice_prob), w (n_total), gid arrays."""
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

    y = df['choice_prob'].values.astype(np.float64)
    w = df['n_total'].values.astype(np.float64)

    # Weight is constant within OD group, normalize to mean=1
    # to keep LL on similar scale
    w = w / w.mean()

    gid_codes, _ = pd.factorize(df['od_pair'], sort=False)
    gid = gid_codes.astype(np.int64)
    ng = gid.max() + 1
    return X, y, w, gid, ng, features + asc_names


def mnl_neg_ll(beta, X, y, w, gid, ng):
    """Weighted negative log-likelihood."""
    V = X @ beta
    V_max = np.full(ng, -np.inf)
    np.maximum.at(V_max, gid, V)
    V_s = V - V_max[gid]
    exp_V = np.exp(V_s)
    sum_exp = np.bincount(gid, weights=exp_V, minlength=ng)
    log_p = V_s - np.log(sum_exp[gid])
    return -np.sum(w * y * log_p)


def mnl_grad(beta, X, y, w, gid, ng):
    """Weighted gradient."""
    V = X @ beta
    V_max = np.full(ng, -np.inf)
    np.maximum.at(V_max, gid, V)
    V_s = V - V_max[gid]
    exp_V = np.exp(V_s)
    sum_exp = np.bincount(gid, weights=exp_V, minlength=ng)
    prob = exp_V / sum_exp[gid]

    # Weighted sum of y per group (for proper gradient)
    wy = w * y
    wy_sum = np.bincount(gid, weights=wy, minlength=ng)
    # grad = X.T @ (wy_sum[gid] * prob - w * y)
    return X.T @ (wy_sum[gid] * prob - wy)


def fit_mnl(X, y, w, gid, ng, feat_names):
    """Fit unconstrained weighted MNL."""
    n_p = X.shape[1]
    bounds = [(None, None)] * n_p  # unconstrained

    beta0 = np.zeros(n_p)
    res = minimize(
        mnl_neg_ll, beta0, args=(X, y, w, gid, ng),
        jac=mnl_grad, method='L-BFGS-B',
        bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12},
    )

    # Gradient check
    eps = 1e-5
    num_grad = approx_fprime(res.x, mnl_neg_ll, eps, X, y, w, gid, ng)
    ana_grad = mnl_grad(res.x, X, y, w, gid, ng)
    grad_diff = np.max(np.abs(num_grad - ana_grad))
    print(f'  Gradient check: max|num-ana| = {grad_diff:.2e}')

    # Hessian -> SE
    H = np.zeros((n_p, n_p))
    for i in range(n_p):
        def grad_i(b, _i=i):
            return mnl_grad(b, X, y, w, gid, ng)[_i]
        H[i, :] = approx_fprime(res.x, grad_i, eps)
    H = (H + H.T) / 2
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t_stat = res.x / np.where(se > 0, se, 1.0)

    # LL(0): equal probability model (weighted)
    gs = np.bincount(gid, minlength=ng)
    # wy_sum per group
    wy = w * y
    wy_sum = np.bincount(gid, weights=wy, minlength=ng)
    ll_0 = -float(np.sum(wy_sum * np.log(gs)))
    ll_beta = -res.fun
    rho_sq = 1 - ll_beta / ll_0

    return res.x, se, t_stat, ll_0, ll_beta, rho_sq, res.success


def weighted_top1(X, y, gid, ng, beta):
    """Top-1 accuracy: highest choice_prob alt vs highest predicted prob."""
    V = X @ beta
    df_tmp = pd.DataFrame({'gid': gid, 'V': V, 'y': y})
    pred_idx = df_tmp.groupby('gid')['V'].idxmax()
    true_idx = df_tmp.groupby('gid')['y'].idxmax()
    return (pred_idx == true_idx).mean()


def main():
    print('=== Weighted MNL: A_total_uncon ===\n')

    print('Loading data...')
    df = load_data(DATA_PATH)
    print(f'  {len(df):,} rows, {df["od_pair"].nunique():,} ODs')

    # Verify choice_prob exists
    if 'choice_prob' not in df.columns:
        print('ERROR: choice_prob not found. Run attach_trip_counts.py first.')
        return

    X, y, w, gid, ng, all_names = build_arrays(df, FEATURES)
    print(f'  Features: {all_names}')
    print(f'  X shape: {X.shape}')

    print('\nFitting weighted MNL...')
    t0 = time.time()
    beta, se, t_stat, ll_0, ll_beta, rho_sq, success = fit_mnl(
        X, y, w, gid, ng, all_names
    )
    elapsed = time.time() - t0

    top1 = weighted_top1(X, y, gid, ng, beta)

    print(f'\n  Converged: {success}')
    print(f'  Time: {elapsed:.1f}s')
    print(f'  LL(0): {ll_0:,.2f}')
    print(f'  LL(beta): {ll_beta:,.2f}')
    print(f'  rho-sq: {rho_sq:.4f}')
    print(f'  Top-1: {top1*100:.2f}%')

    print(f'\n  {"Parameter":<26} {"beta":>12} {"SE":>12} {"t-stat":>10}')
    print(f'  {"-" * 62}')
    for i, name in enumerate(all_names):
        print(f'  {name:<26} {beta[i]:>12.6f} '
              f'{se[i]:>12.6f} {t_stat[i]:>10.3f}')

    # Compare with binary MNL
    binary_path = ROOT / 'data' / 'results' / 'mnl_4spec.json'
    if binary_path.exists():
        with open(binary_path) as f:
            binary_results = json.load(f)
        binary_a = next(
            (r for r in binary_results if r['spec'] == 'A_total_uncon'),
            None
        )
        if binary_a:
            print('\n  --- Binary vs Weighted Comparison ---')
            print(f'  {"Parameter":<26} {"Binary":>12} {"Weighted":>12}')
            print(f'  {"-" * 52}')
            for i, name in enumerate(all_names):
                b_val = binary_a['beta'][i] if i < len(binary_a['beta']) else 0
                print(f'  {name:<26} {b_val:>12.6f} {beta[i]:>12.6f}')
            print(f'  {"rho-sq":<26} {binary_a["rho_sq"]:>12.4f} '
                  f'{rho_sq:>12.4f}')
            print(f'  {"Top-1":<26} '
                  f'{binary_a["top1"]*100:>11.2f}% {top1*100:>11.2f}%')

    # Save
    result = {
        'spec': 'A_total_uncon_weighted',
        'success': bool(success),
        'rho_sq': float(rho_sq),
        'top1': float(top1),
        'll_0': float(ll_0),
        'll_beta': float(ll_beta),
        'n_ods': int(ng),
        'n_alts': int(len(y)),
        'n_params': int(len(beta)),
        'names': all_names,
        'beta': [float(b) for b in beta],
        'se': [float(s) for s in se],
        't_stat': [float(t) for t in t_stat],
        'estimation': 'weighted',
        'weight': 'n_total (normalized)',
        'label': 'choice_prob',
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
