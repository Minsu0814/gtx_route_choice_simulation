"""
Mixed Logit 4-spec comparison using shared spec_config.
  ML_A: Total IVT, no walk
  ML_B: Total IVT + walk
  ML_C: IVT split, no walk
  ML_D: IVT split + walk

Usage:
    python run_ml_4spec.py [n_draws] [maxiter] [n_sample]
    python run_ml_4spec.py 200 300 200000
"""
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'similarity' / 'mnl'))
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'similarity'))

from mixed_logit import (
    generate_halton_draws, prepare_mixed_data, mixed_logit_ll,
    _unpack_params, _pack_params, predict_mixed_probs, print_results,
    RANDOM_LOGNORMAL,
)
from spec_config import ML_SPECS, CATEGORY_MAP

DATA_PATH = (PROJECT_ROOT / 'data' / 'training_set_new'
             / 'route_choice_filtered_training.parquet')
OUT_PATH = PROJECT_ROOT / 'data' / 'training_set_new' / 'ml_4spec.json'


def load_data(path, n_sample=200000):
    df = pd.read_parquet(path)
    print(f'Loaded {len(df):,} rows, {df["od_pair"].nunique():,} ODs')

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
    df['choice_prob'] = df['chosen'].astype(float)
    df['n_total'] = 1

    if n_sample > 0 and n_sample < df['od_pair'].nunique():
        rng = np.random.RandomState(42)
        ods = df['od_pair'].unique()
        sel = rng.choice(ods, size=n_sample, replace=False)
        df = df[df['od_pair'].isin(sel)].copy()
        print(f'Subsampled: {df["od_pair"].nunique():,} ODs, {len(df):,} rows')
    return df


def fit_spec(df, spec_name, spec_def, n_draws=200, maxiter=300):
    """Fit one Mixed Logit spec."""
    from scipy.optimize import minimize

    random_params = spec_def['random_ln'] + spec_def['random_normal']
    fixed_params = spec_def['fixed']
    n_ln = len(spec_def['random_ln'])

    print(f'\n{"="*70}')
    print(f'  {spec_name}')
    print(f'  Random (lognormal): {spec_def["random_ln"]}')
    print(f'  Random (normal):    {spec_def["random_normal"]}')
    print(f'  Fixed:              {fixed_params}')
    print(f'{"="*70}')

    flat = prepare_mixed_data(
        df, random_params, fixed_params,
        asc_column='transport_category', asc_base='bus_only',
        group_col='od_pair', choice_col='choice_prob', weight_col='n_total',
    )

    n_r = flat['n_random']
    n_f = flat['n_fixed']
    n_p = 2 * n_r + n_f
    draws = generate_halton_draws(n_draws, n_r, seed=12345)

    print(f'Groups: {flat["n_groups"]:,}, Alts: {len(flat["y"]):,}, '
          f'Params: {n_p} (2*{n_r} + {n_f})')

    # Initial values
    mu0 = np.full(n_r, -4.0)
    for k in range(n_ln, n_r):
        mu0[k] = -0.5
    sigma0 = np.full(n_r, 0.1)
    beta_f0 = np.zeros(n_f)
    beta_f0[0] = -0.5  # fare
    params0 = np.concatenate([mu0, sigma0, beta_f0])

    # Bounds
    bounds = (
        [(None, None)] * n_r +       # mu
        [(0.001, None)] * n_r +       # sigma > 0
        [(None, 0)] +                  # fare <= 0
        [(None, None)] * (n_f - 1)     # walk + ASCs
    )

    # Walk betas <= 0
    for i, name in enumerate(flat['fixed_param_names']):
        if name in ('ln_access', 'ln_egress'):
            bounds[2 * n_r + i] = (None, 0)

    best = [np.inf]
    t0 = time.time()
    icount = [0]

    def callback(xk):
        icount[0] += 1
        if icount[0] % 10 == 0:
            ll = mixed_logit_ll(xk, flat, draws, n_ln)
            elapsed = (time.time() - t0) / 60
            if ll < best[0]:
                best[0] = ll
            print(f'  Iter {icount[0]:4d}  LL={-ll:,.2f}  '
                  f'best={-best[0]:,.2f}  ({elapsed:.1f}m)')

    ll_init = mixed_logit_ll(params0, flat, draws, n_ln)
    print(f'Initial LL = {-ll_init:,.2f}')

    result = minimize(
        mixed_logit_ll, params0, args=(flat, draws, n_ln),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-6, 'maxcor': 20},
        callback=callback,
    )

    elapsed = (time.time() - t0) / 60
    print(f'Converged: {result.success} ({elapsed:.1f} min)')
    print(f'LL(beta): {-result.fun:,.2f}')

    mu_hat, sigma_hat, beta_f_hat = _unpack_params(result.x, n_r, n_f)

    # Mean betas
    mean_b = np.empty(n_r)
    for k in range(n_ln):
        mean_b[k] = -np.exp(mu_hat[k] + sigma_hat[k]**2 / 2)
    for k in range(n_ln, n_r):
        mean_b[k] = mu_hat[k]

    # LL(0)
    gid = flat['gid']
    ng = flat['n_groups']
    gs = np.bincount(gid, minlength=ng)
    ll_0 = -float(np.sum(np.log(gs)))
    ll_beta = -result.fun
    rho_sq = 1 - ll_beta / ll_0

    # Top-1 accuracy
    model_tmp = {
        'n_random': n_r, 'n_fixed': n_f, 'n_lognormal': n_ln,
        'mu': mu_hat, 'sigma': sigma_hat, 'beta_fixed': beta_f_hat,
        '_draws': draws,
    }
    pred = predict_mixed_probs(model_tmp, flat, draws)
    y = flat['y']
    top1 = sum(
        np.argmax(pred[gid == g]) == np.argmax(y[gid == g])
        for g in range(ng)
    )
    top1_acc = top1 / ng

    print(f'rho-sq: {rho_sq:.4f}, Top-1: {top1_acc*100:.1f}%')

    # Print betas
    print(f'\n  {"Parameter":<24} {"Mean b":>12} {"mu":>10} {"sigma":>10}')
    print(f'  {"-"*58}')
    rp = random_params
    for k in range(n_r):
        dist = 'LN' if k < n_ln else 'N'
        print(f'  {rp[k]:<20} ({dist}) {mean_b[k]:>12.6f} '
              f'{mu_hat[k]:>10.4f} {sigma_hat[k]:>10.4f}')
    fp = flat['fixed_param_names']
    for k in range(n_f):
        print(f'  {fp[k]:<24} {beta_f_hat[k]:>12.6f}')

    return {
        'spec': spec_name, 'success': bool(result.success),
        'rho_sq': float(rho_sq), 'top1': float(top1_acc),
        'll_0': float(ll_0), 'll_beta': float(ll_beta),
        'mean_betas': mean_b.tolist(), 'mu': mu_hat.tolist(),
        'sigma': sigma_hat.tolist(), 'beta_fixed': beta_f_hat.tolist(),
        'random_names': random_params,
        'fixed_names': list(flat['fixed_param_names']),
        'n_ln': n_ln, 'n_params': n_p,
        'elapsed_min': float(elapsed),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--draws', type=int, default=200)
    parser.add_argument('--maxiter', type=int, default=300)
    parser.add_argument('--sample', type=int, default=200000)
    parser.add_argument('--spec', type=str, default=None,
                        help='Run single spec, e.g. ML_C_split')
    args = parser.parse_args()

    n_draws, maxiter, n_sample = args.draws, args.maxiter, args.sample

    print(f'Mixed Logit 4-Spec Comparison')
    print(f'Draws={n_draws}, MaxIter={maxiter}, Sample={n_sample}\n')

    df = load_data(DATA_PATH, n_sample=n_sample)

    if args.spec:
        specs_to_run = {args.spec: ML_SPECS[args.spec]}
    else:
        specs_to_run = ML_SPECS

    results = []
    for spec_name, spec_def in specs_to_run.items():
        r = fit_spec(df, spec_name, spec_def, n_draws, maxiter)
        results.append(r)

    # Summary
    print(f'\n\n{"="*80}')
    print('MIXED LOGIT 4-SPEC SUMMARY')
    print(f'{"="*80}')
    print(f'{"Spec":<25} {"rho-sq":>8} {"Top-1":>8} '
          f'{"LL(b)":>14} {"K":>4} {"Time":>6}')
    print(f'{"-"*70}')
    for r in results:
        print(f'{r["spec"]:<25} {r["rho_sq"]:>8.4f} '
              f'{r["top1"]*100:>7.1f}% {r["ll_beta"]:>14,.2f} '
              f'{r["n_params"]:>4} {r["elapsed_min"]:>5.1f}m')

    # IVT sign check
    print(f'\n--- IVT Mean Beta Check ---')
    for r in results:
        print(f'\n  {r["spec"]}:')
        for k, name in enumerate(r['random_names']):
            if 'ivt' in name or 'total' in name:
                sign = '(-)' if r['mean_betas'][k] < 0 else '(+) !!!'
                print(f'    {name:<24} E[b]={r["mean_betas"][k]:>10.6f} {sign}')

    # Save JSON
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to {OUT_PATH}')
