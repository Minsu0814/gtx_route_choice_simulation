"""
Sensitivity Analysis: 2D Grid (spatial × route)
Usage:
    python sensitivity_2d_grid.py --step 0.05
    python sensitivity_2d_grid.py --step 0.01
    python sensitivity_2d_grid.py --step 0.05 0.01   # both
"""
import argparse
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no GUI
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'training_set'
OUT_DIR = ROOT / 'data' / 'sensitivity'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────
BASELINE_THRESHOLD = 0.6
BASELINE_NORM_DIST = 5000

TIME_RAW = ['in_vehicle_time', 'wait_time', 'access_time', 'egress_time', 'transfer_walk_time']
DIST_RAW = ['total_distance']

MODEL_FEATURES = [
    'in_vehicle_time_min', 'wait_time_min',
    'access_time_min', 'egress_time_min', 'transfer_walk_time_min',
    'total_distance_km', 'num_transfers', 'fare',
    'has_bus', 'has_train', 'has_gtx',
]
SIGN_CONSTRAINED = {
    'in_vehicle_time_min', 'wait_time_min',
    'access_time_min', 'egress_time_min', 'transfer_walk_time_min',
    'total_distance_km', 'num_transfers', 'fare',
}
KEY_BETAS = ['in_vehicle_time_min', 'access_time_min', 'num_transfers', 'has_bus', 'has_train']
KEY_LABELS = ['beta_IVT', 'beta_access', 'beta_transfers', 'beta_bus', 'beta_train']


# ── MNL functions ──────────────────────────────────────────────────
def prepare_flat_data(data, features):
    X_list, y_list, w_list, gid_list = [], [], [], []
    group_id = 0
    for _, grp in data.groupby('od_pair'):
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
    exp_V = np.exp(V - V_max[gid])
    sum_exp = np.bincount(gid, weights=exp_V, minlength=ng)
    log_prob = (V - V_max[gid]) - np.log(sum_exp[gid])
    return -np.sum(w * y * log_prob)


def mnl_gradient(beta, flat):
    X, y, w, gid, ng = flat['X'], flat['y'], flat['w'], flat['gid'], flat['n_groups']
    V = X @ beta
    V_max = np.full(ng, -np.inf)
    np.maximum.at(V_max, gid, V)
    exp_V = np.exp(V - V_max[gid])
    sum_exp = np.bincount(gid, weights=exp_V, minlength=ng)
    prob = exp_V / sum_exp[gid]
    return X.T @ (w * (prob - y))


def compute_ll0(flat):
    gid, w, ng = flat['gid'], flat['w'], flat['n_groups']
    gs = np.bincount(gid, minlength=ng)
    wpg = np.bincount(gid, weights=w, minlength=ng) / gs
    return -np.sum(wpg * np.log(gs))


def compute_fpr(beta, flat):
    X, y, gid, ng = flat['X'], flat['y'], flat['gid'], flat['n_groups']
    V = X @ beta
    V_max = np.full(ng, -np.inf)
    np.maximum.at(V_max, gid, V)
    exp_V = np.exp(V - V_max[gid])
    sum_exp = np.bincount(gid, weights=exp_V, minlength=ng)
    pred = exp_V / sum_exp[gid]
    correct = 0
    for g in range(ng):
        m = gid == g
        if np.argmax(pred[m]) == np.argmax(y[m]):
            correct += 1
    return correct / ng


def estimate_mnl(train_flat, test_flat):
    bounds = [(None, 0) if f in SIGN_CONSTRAINED else (None, None) for f in MODEL_FEATURES]
    res = minimize(mnl_neg_ll, np.zeros(len(MODEL_FEATURES)), args=(train_flat,),
                   jac=mnl_gradient, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 2000, 'ftol': 1e-12})
    b = res.x
    ll_b = -res.fun
    ll_0 = compute_ll0(train_flat)
    rho = 1 - ll_b / ll_0
    t_ll = -mnl_neg_ll(b, test_flat)
    t_ll0 = compute_ll0(test_flat)
    t_rho = 1 - t_ll / t_ll0
    fpr = compute_fpr(b, test_flat)
    # RMSE
    X_t, gid_t, ng_t = test_flat['X'], test_flat['gid'], test_flat['n_groups']
    V = X_t @ b
    V_max = np.full(ng_t, -np.inf)
    np.maximum.at(V_max, gid_t, V)
    exp_V = np.exp(V - V_max[gid_t])
    sum_exp = np.bincount(gid_t, weights=exp_V, minlength=ng_t)
    pred = exp_V / sum_exp[gid_t]
    rmse = np.sqrt(np.mean((pred - test_flat['y']) ** 2))
    return {
        'converged': bool(res.success), 'train_rho_sq': float(rho),
        'test_rho_sq': float(t_rho), 'test_fpr': float(fpr),
        'test_rmse': float(rmse),
        'betas': {f: float(v) for f, v in zip(MODEL_FEATURES, b)},
    }


# ── Grid helpers ───────────────────────────────────────────────────
def recompute_composite(df, weights):
    return (weights['mode'] * df['sim_mode']
            + weights['seq'] * df['sim_sequence']
            + weights['time'] * df['sim_time']
            + weights['route'] * df['sim_route']
            + weights['spatial'] * df['sim_spatial'])


def filter_by_threshold(df, composite, threshold):
    df = df.copy()
    df['new_composite'] = composite
    idx_dom = df.groupby('od_pair')['choice_prob'].idxmax()
    dom_comp = df.loc[idx_dom, ['od_pair', 'new_composite']].set_index('od_pair')['new_composite']
    valid = dom_comp[dom_comp >= threshold].index
    filt = df[df['od_pair'].isin(valid)]
    od_cnt = filt.groupby('od_pair').size()
    valid2 = od_cnt[od_cnt >= 2].index
    return filt[filt['od_pair'].isin(valid2)]


def run_2d_grid(df, spatial_values, route_values, step_label):
    results = []
    t_total = time.time()
    total = sum(1 for sp in spatial_values for rt in route_values if (1.0 - sp - rt) >= -1e-9)
    done = 0

    for sp in spatial_values:
        for rt in route_values:
            rest = 1.0 - sp - rt
            if rest < -1e-9:
                continue
            rest = max(rest, 0.0)
            each = rest / 3
            weights = {'mode': each, 'seq': each, 'time': each, 'route': rt, 'spatial': sp}
            label = f'sp{sp:.2f}_rt{rt:.2f}'

            t0 = time.time()
            comp = recompute_composite(df, weights)
            filt = filter_by_threshold(df, comp, BASELINE_THRESHOLD)
            n_ods = filt['od_pair'].nunique()

            if n_ods < 100:
                done += 1
                print(f'  [{done}/{total}] [{label}] SKIP (n_ods={n_ods})')
                continue

            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr_idx, te_idx = next(splitter.split(filt, groups=filt['od_pair']))
            train_flat = prepare_flat_data(filt.iloc[tr_idx], MODEL_FEATURES)
            test_flat = prepare_flat_data(filt.iloc[te_idx], MODEL_FEATURES)

            r = estimate_mnl(train_flat, test_flat)
            r.update({'scenario': label, 'spatial': sp, 'route': rt, 'rest_each': each,
                      'n_ods': n_ods, 'weights': weights})
            for feat, key in zip(KEY_BETAS, KEY_LABELS):
                r[key] = r['betas'].get(feat, 0.0)
            results.append(r)

            done += 1
            elapsed = time.time() - t0
            print(f'  [{done}/{total}] [{label}] ODs={n_ods:,} | '
                  f'rho={r["train_rho_sq"]:.4f}/{r["test_rho_sq"]:.4f} | '
                  f'FPR={r["test_fpr"]:.3f} | {elapsed:.0f}s')

    print(f'\nDone ({step_label}): {len(results)} scenarios, {(time.time() - t_total) / 60:.1f} min total')
    return results


# ── Save & plot ────────────────────────────────────────────────────
CSV_COLS = ['scenario', 'spatial', 'route', 'rest_each', 'n_ods',
            'train_rho_sq', 'test_rho_sq', 'test_fpr', 'test_rmse',
            'beta_IVT', 'beta_access', 'beta_transfers', 'beta_bus', 'beta_train',
            'converged']


def save_results(results, suffix):
    df_out = pd.DataFrame(results)
    df_out[CSV_COLS].to_csv(OUT_DIR / f'sensitivity_2d_grid_{suffix}.csv', index=False)
    with open(OUT_DIR / f'sensitivity_2d_grid_{suffix}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f'Saved {len(results)} scenarios → sensitivity_2d_grid_{suffix}.csv / .json')
    return df_out


def make_heatmap(df_2d, value_col, title, fmt, cmap, vmin=None, vmax=None):
    pivot = df_2d.pivot(index='spatial', columns='route', values=value_col)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax, origin='lower')
    if pivot.shape[0] <= 15 and pivot.shape[1] <= 10:
        fontsize = max(6, 10 - max(pivot.shape[0], pivot.shape[1]) // 3)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isnan(val):
                    continue
                vmin_eff = vmin or pivot.values[~np.isnan(pivot.values)].min()
                vmax_eff = vmax or pivot.values[~np.isnan(pivot.values)].max()
                rel = (val - vmin_eff) / (vmax_eff - vmin_eff + 1e-9)
                color = 'white' if rel > 0.6 else 'black'
                ax.text(j, i, fmt.format(val), ha='center', va='center',
                        fontsize=fontsize, fontweight='bold', color=color)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{v:.2f}' for v in pivot.columns], fontsize=7, rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{v:.2f}' for v in pivot.index], fontsize=7)
    ax.set_xlabel('Route weight ($w_{route}$)', fontsize=12)
    ax.set_ylabel('Spatial weight ($w_{spatial}$)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    return fig


def plot_beta_heatmaps(df_grid, step_label, out_suffix):
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    for ax, feat, lbl in zip(axes, KEY_BETAS, KEY_LABELS):
        df_grid[f'_tmp'] = df_grid['betas'].apply(lambda d: d.get(feat, 0.0))
        pivot = df_grid.pivot(index='spatial', columns='route', values='_tmp')
        im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', origin='lower')
        if pivot.shape[0] <= 15 and pivot.shape[1] <= 10:
            fs = max(5, 8 - max(pivot.shape[0], pivot.shape[1]) // 4)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=fs)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{v:.2f}' for v in pivot.columns], fontsize=6, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{v:.2f}' for v in pivot.index], fontsize=6)
        ax.set_xlabel('$w_{route}$', fontsize=9)
        ax.set_ylabel('$w_{spatial}$', fontsize=9)
        vals = pivot.values[~np.isnan(pivot.values)]
        cv = np.std(vals) / abs(np.mean(vals)) * 100 if abs(np.mean(vals)) > 1e-10 else 0
        ax.set_title(f'{lbl}\nCV={cv:.1f}%', fontsize=10, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)
        df_grid.drop(columns=['_tmp'], inplace=True)
    fig.suptitle(f'$\\beta$ Coefficient Stability ({step_label})', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f'fig_heatmap_betas_{out_suffix}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_plots(df_grid, suffix, step_label):
    fig = make_heatmap(df_grid, 'test_rho_sq',
                       f'Test McFadden $\\rho^2$ ({step_label})',
                       '{:.3f}', 'YlOrRd', vmin=0.43, vmax=0.50)
    fig.savefig(OUT_DIR / f'fig_heatmap_rho2_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = make_heatmap(df_grid, 'test_fpr',
                       f'First Preference Recovery ({step_label})',
                       '{:.1%}', 'YlGn', vmin=0.68, vmax=0.76)
    fig.savefig(OUT_DIR / f'fig_heatmap_fpr_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = make_heatmap(df_grid, 'n_ods',
                       f'Number of OD Pairs ({step_label})',
                       '{:.0f}', 'PuBu')
    fig.savefig(OUT_DIR / f'fig_heatmap_n_ods_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    plot_beta_heatmaps(df_grid, step_label, suffix)
    print(f'Plots saved ({suffix}).')


def print_robustness(df_grid, label):
    rho_vals = df_grid['test_rho_sq'].values
    fpr_vals = df_grid['test_fpr'].values
    print('=' * 65)
    print(f'2D Grid Robustness — {label}  ({len(df_grid)} scenarios)')
    print('=' * 65)
    print(f'rho2: [{rho_vals.min():.4f}, {rho_vals.max():.4f}]  spread={rho_vals.max() - rho_vals.min():.4f}')
    print(f'FPR:  [{fpr_vals.min() * 100:.1f}%, {fpr_vals.max() * 100:.1f}%]  '
          f'spread={(fpr_vals.max() - fpr_vals.min()) * 100:.1f}%p')
    best_rho = df_grid.loc[df_grid['test_rho_sq'].idxmax()]
    best_fpr = df_grid.loc[df_grid['test_fpr'].idxmax()]
    print(f'  Best rho2: {best_rho["scenario"]}  ({best_rho["test_rho_sq"]:.4f})')
    print(f'  Best FPR:  {best_fpr["scenario"]}  ({best_fpr["test_fpr"]:.3f})')
    for feat, lbl in zip(KEY_BETAS, KEY_LABELS):
        vals = df_grid['betas'].apply(lambda d: d.get(feat, 0.0)).values
        m = np.mean(vals)
        cv = np.std(vals) / abs(m) * 100 if abs(m) > 1e-10 else 0
        print(f'  {lbl:<18}: mean={m:>10.6f}  CV={cv:>6.1f}%  {"ROBUST" if cv < 20 else "VARIABLE"}')
    print(f'All rho2 > 0.2:     {"PASS" if all(rho_vals > 0.2) else "FAIL"}')
    print(f'All FPR > 60%:      {"PASS" if all(fpr_vals > 0.6) else "FAIL"}')
    print(f'rho2 spread < 0.05: {"PASS" if (rho_vals.max() - rho_vals.min()) < 0.05 else "FAIL"}')
    print(f'FPR spread < 5%p:   {"PASS" if (fpr_vals.max() - fpr_vals.min()) < 0.05 else "FAIL"}')
    print()


# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='2D Grid Sensitivity Analysis')
    parser.add_argument('--step', nargs='+', type=float, default=[0.05],
                        help='Grid step sizes (e.g. 0.05 0.01)')
    args = parser.parse_args()

    # Load data
    print('Loading data...')
    df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')
    print(f'{len(df):,} rows, {df["od_pair"].nunique():,} ODs')

    for col in TIME_RAW:
        df[col + '_min'] = df[col] / 60
    for col in DIST_RAW:
        df[col + '_km'] = df[col] / 1000
    df['raw_hausdorff'] = (1 - df['sim_spatial']) * BASELINE_NORM_DIST

    for step in args.step:
        suffix = f'step{str(step).replace(".", "")}'
        step_label = f'step={step}'

        spatial_vals = np.round(np.arange(0.0, 0.61, step), 2).tolist()
        route_vals = np.round(np.arange(0.10, 0.41, step), 2).tolist()
        total = sum(1 for sp in spatial_vals for rt in route_vals if (1.0 - sp - rt) >= -1e-9)

        print(f'\n{"=" * 65}')
        print(f'Grid {step_label}: spatial {len(spatial_vals)} × route {len(route_vals)} = {total} combos')
        print(f'{"=" * 65}')

        results = run_2d_grid(df, spatial_vals, route_vals, step_label)
        df_grid = save_results(results, suffix)
        save_plots(df_grid, suffix, step_label)
        print_robustness(df_grid, step_label)

    print('\nAll done!')


if __name__ == '__main__':
    main()
