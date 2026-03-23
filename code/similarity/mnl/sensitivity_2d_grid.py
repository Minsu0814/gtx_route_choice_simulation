"""
Sensitivity Analysis: 2D Grid (route × sequence)  — Optimized
Usage:
    python sensitivity_2d_grid.py --step 0.05
    python sensitivity_2d_grid.py --step 0.01
    python sensitivity_2d_grid.py --step 0.05 0.01
    python sensitivity_2d_grid.py --step 0.01 --workers 8
    python sensitivity_2d_grid.py --step 0.01 --no-cache   # ignore checkpoint, run from scratch
"""
import argparse
import time
import json
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / 'data' / 'training_set'
OUT_DIR = ROOT / 'data' / 'sensitivity'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUT_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────
BASELINE_THRESHOLD = 0.5
BASELINE_NORM_DIST = 5000
MIN_SEQ_GATE = 0.3  # minimum sim_sequence to filter detour/stopover routes

# K1 사양: 모드별 IVT + log(walk) + no distance/wait
MODEL_FEATURES = [
    'bus_ivt_min', 'train_ivt_min', 'gtx_ivt_min',
    'ln_access', 'ln_egress',
    'transfer_walk_time_min',
    'num_transfers', 'fare_1000won',
    'has_bus', 'has_train', 'has_gtx',
]
N_FEAT = len(MODEL_FEATURES)
SIGN_CONSTRAINED = {
    'bus_ivt_min', 'train_ivt_min', 'gtx_ivt_min',
    'ln_access', 'ln_egress',
    'transfer_walk_time_min',
    'num_transfers', 'fare_1000won',
}
BOUNDS = [(None, 0) if f in SIGN_CONSTRAINED else (None, None) for f in MODEL_FEATURES]
KEY_BETAS = ['bus_ivt_min', 'train_ivt_min', 'gtx_ivt_min', 'ln_access', 'num_transfers', 'has_train']
KEY_LABELS = ['beta_bus_IVT', 'beta_train_IVT', 'beta_gtx_IVT', 'beta_ln_access', 'beta_transfers', 'beta_train']


# ══════════════════════════════════════════════════════════════════
#  Pre-indexed numpy structure (built ONCE, reused every scenario)
# ══════════════════════════════════════════════════════════════════
class PreIndexed:
    """
    Sort df by od_pair once → store all arrays + group boundaries.
    Per-scenario: vectorized composite → mask → slice → MNL.
    No pandas groupby in the hot loop.
    """
    def __init__(self, df):
        t0 = time.time()
        # sort by od_pair for contiguous groups
        df = df.sort_values('od_pair').reset_index(drop=True)
        od = df['od_pair'].values
        # group boundaries
        change = np.concatenate([[True], od[1:] != od[:-1]])
        self.starts = np.where(change)[0]
        self.sizes = np.diff(np.append(self.starts, len(od)))
        self.n_groups = len(self.starts)
        # per-group od labels
        self.od_labels = od[self.starts]
        # similarity columns (float32 saves memory, enough precision)
        self.sim_mode = df['sim_mode'].values.astype(np.float32)
        self.sim_seq = df['sim_sequence'].values.astype(np.float32)
        self.sim_route = df['sim_route'].values.astype(np.float32)
        # choice prob & weight
        self.choice_prob = df['choice_prob'].values.astype(np.float64)
        self.n_total = df['n_total'].values.astype(np.float64)
        # model features
        self.X = df[MODEL_FEATURES].values.astype(np.float64)
        # per-group: dominant row index (argmax choice_prob within group)
        self.dom_idx = np.empty(self.n_groups, dtype=np.int64)
        # per-group: valid flag for choice_prob sum ≈ 1
        self.valid_prob = np.ones(self.n_groups, dtype=bool)
        # per-group weight (first row's n_total)
        self.group_weight = np.empty(self.n_groups, dtype=np.float64)
        for g in range(self.n_groups):
            s, sz = self.starts[g], self.sizes[g]
            prob_g = self.choice_prob[s:s+sz]
            self.dom_idx[g] = s + np.argmax(prob_g)
            self.group_weight[g] = self.n_total[s]
            if abs(prob_g.sum() - 1.0) > 0.01:
                self.valid_prob[g] = False
        # precompute train/test split at group level (80/20 by OD hash)
        rng = np.random.RandomState(42)
        perm = rng.permutation(self.n_groups)
        n_test = int(self.n_groups * 0.2)
        self.test_groups = np.sort(perm[:n_test])
        self.train_groups = np.sort(perm[n_test:])
        print(f'PreIndexed built: {self.n_groups:,} groups, {len(df):,} rows  ({time.time()-t0:.1f}s)')

    def composite_at_dom(self, w_mode, w_route, w_seq):
        """Composite score at each group's dominant row (3-level: mode+route+seq)."""
        idx = self.dom_idx
        return (w_mode * self.sim_mode[idx]
                + w_route * self.sim_route[idx]
                + w_seq * self.sim_seq[idx])

    def build_flat(self, group_mask):
        """Build MNL flat arrays from selected groups (boolean mask over n_groups)."""
        gids = np.where(group_mask)[0]
        n_sel = gids.shape[0]
        # estimate total rows
        total_rows = self.sizes[gids].sum()
        X_out = np.empty((total_rows, N_FEAT), dtype=np.float64)
        y_out = np.empty(total_rows, dtype=np.float64)
        w_out = np.empty(total_rows, dtype=np.float64)
        gid_out = np.empty(total_rows, dtype=np.int64)
        pos = 0
        new_gid = 0
        for g in gids:
            s, sz = self.starts[g], self.sizes[g]
            X_out[pos:pos+sz] = self.X[s:s+sz]
            y_out[pos:pos+sz] = self.choice_prob[s:s+sz]
            w_out[pos:pos+sz] = self.group_weight[g]
            gid_out[pos:pos+sz] = new_gid
            new_gid += 1
            pos += sz
        return {
            'X': X_out[:pos], 'y': y_out[:pos],
            'w': w_out[:pos], 'gid': gid_out[:pos],
            'n_groups': new_gid,
        }


# ══════════════════════════════════════════════════════════════════
#  MNL (vectorized, no python loops for FPR)
# ══════════════════════════════════════════════════════════════════
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


def compute_fpr_vectorized(beta, flat):
    """FPR without python for-loop: use bincount tricks."""
    X, y, gid, ng = flat['X'], flat['y'], flat['gid'], flat['n_groups']
    V = X @ beta
    V_max = np.full(ng, -np.inf)
    np.maximum.at(V_max, gid, V)
    exp_V = np.exp(V - V_max[gid])
    sum_exp = np.bincount(gid, weights=exp_V, minlength=ng)
    pred = exp_V / sum_exp[gid]
    # For each group, check argmax(pred) == argmax(y)
    # Trick: within each group, the predicted choice is the row with max pred
    # Encode row-within-group pred as: gid * BIG + pred, then argmax per group
    # Simpler: use np.maximum.at to find max pred per group, then check
    pred_max = np.full(ng, -np.inf)
    np.maximum.at(pred_max, gid, pred)
    pred_winner = (pred == pred_max[gid])  # might have ties
    y_max = np.full(ng, -np.inf)
    np.maximum.at(y_max, gid, y)
    y_winner = (y == y_max[gid])
    # Both are winner → correct
    both = pred_winner & y_winner
    correct = np.bincount(gid[both], minlength=ng)
    return (correct > 0).sum() / ng


def estimate_mnl(train_flat, test_flat):
    res = minimize(mnl_neg_ll, np.zeros(N_FEAT), args=(train_flat,),
                   jac=mnl_gradient, method='L-BFGS-B', bounds=BOUNDS,
                   options={'maxiter': 2000, 'ftol': 1e-12})
    b = res.x
    ll_b = -res.fun
    ll_0 = compute_ll0(train_flat)
    rho = 1 - ll_b / ll_0
    t_ll = -mnl_neg_ll(b, test_flat)
    t_ll0 = compute_ll0(test_flat)
    t_rho = 1 - t_ll / t_ll0
    fpr = compute_fpr_vectorized(b, test_flat)
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


# ══════════════════════════════════════════════════════════════════
#  Single scenario worker (for multiprocessing)
# ══════════════════════════════════════════════════════════════════
def run_one_scenario(pre, rt, sq):
    """Run one (route, sequence) scenario. Returns dict or None."""
    w_mode = 1.0 - rt - sq
    if w_mode < -1e-9:
        return None
    w_mode = max(w_mode, 0.0)

    # composite at dominant rows → threshold filter
    dom_comp = pre.composite_at_dom(w_mode, rt, sq)
    keep = (dom_comp >= BASELINE_THRESHOLD) & pre.valid_prob
    keep &= (pre.sizes >= 2)
    # minimum sequence gate: filter detour/stopover routes
    keep &= (pre.sim_seq[pre.dom_idx] >= MIN_SEQ_GATE)
    n_ods = keep.sum()

    if n_ods < 100:
        return None

    # train/test split (pre-computed group indices, intersect with keep)
    train_mask = keep.copy()
    test_mask = keep.copy()
    train_mask[pre.test_groups] = False
    test_mask[pre.train_groups] = False

    if train_mask.sum() < 50 or test_mask.sum() < 50:
        return None

    train_flat = pre.build_flat(train_mask)
    test_flat = pre.build_flat(test_mask)

    r = estimate_mnl(train_flat, test_flat)
    label = f'rt{rt:.2f}_sq{sq:.2f}'
    weights = {'mode': w_mode, 'route': rt, 'seq': sq}
    r.update({'scenario': label, 'route': rt, 'sequence': sq, 'mode_weight': w_mode,
              'n_ods': int(n_ods), 'weights': weights})
    for feat, key in zip(KEY_BETAS, KEY_LABELS):
        r[key] = r['betas'].get(feat, 0.0)
    return r


# Wrapper for ProcessPoolExecutor (pre must be global in worker)
_GLOBAL_PRE = None

def _init_worker(pre):
    global _GLOBAL_PRE
    _GLOBAL_PRE = pre

def _worker_fn(args):
    rt, sq = args
    return run_one_scenario(_GLOBAL_PRE, rt, sq)


# ══════════════════════════════════════════════════════════════════
#  Checkpoint helpers
# ══════════════════════════════════════════════════════════════════
def _checkpoint_path(suffix):
    return CHECKPOINT_DIR / f'ckpt_{suffix}.json'


def _load_checkpoint(suffix):
    """Load checkpoint: returns (results_list, set_of_done_keys)."""
    path = _checkpoint_path(suffix)
    if not path.exists():
        return [], set()
    with open(path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    done = set()
    for r in results:
        done.add((round(r['route'], 4), round(r['sequence'], 4)))
    # Also mark skipped combos
    skip_path = CHECKPOINT_DIR / f'ckpt_{suffix}_skipped.json'
    if skip_path.exists():
        with open(skip_path, 'r', encoding='utf-8') as f:
            skipped = json.load(f)
        for s in skipped:
            done.add((round(s[0], 4), round(s[1], 4)))
    print(f'Checkpoint loaded: {len(results)} results, {len(done)} total done (from {path.name})')
    return results, done


def _save_checkpoint(results, skipped, suffix):
    """Atomically save checkpoint (results + skipped combos)."""
    path = _checkpoint_path(suffix)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, default=str)
    tmp.replace(path)
    # Save skipped list too
    skip_path = CHECKPOINT_DIR / f'ckpt_{suffix}_skipped.json'
    tmp_skip = skip_path.with_suffix('.tmp')
    with open(tmp_skip, 'w', encoding='utf-8') as f:
        json.dump(skipped, f)
    tmp_skip.replace(skip_path)


# ══════════════════════════════════════════════════════════════════
#  Grid runner
# ══════════════════════════════════════════════════════════════════
def run_2d_grid(pre, route_values, sequence_values, step_label, n_workers=1, use_cache=True):
    suffix = f'step{str(step_label.split("=")[1]).replace(".", "")}'
    combos = [(rt, sq) for rt in route_values for sq in sequence_values
              if (1.0 - rt - sq) >= -1e-9]
    total = len(combos)

    # Load checkpoint
    results = []
    skipped = []
    done_keys = set()
    if use_cache:
        results, done_keys = _load_checkpoint(suffix)
        # Reconstruct skipped list from checkpoint
        skip_path = CHECKPOINT_DIR / f'ckpt_{suffix}_skipped.json'
        if skip_path.exists():
            with open(skip_path, 'r', encoding='utf-8') as f:
                skipped = json.load(f)

    remaining = [(rt, sq) for rt, sq in combos
                 if (round(rt, 4), round(sq, 4)) not in done_keys]

    print(f'Running {len(remaining)}/{total} combos ({step_label}), '
          f'{len(done_keys)} cached, workers={n_workers}')

    if not remaining:
        print('All combos already cached - skipping computation.')
        return results

    t_total = time.time()
    done_count = len(done_keys)
    ckpt_interval = 5  # save checkpoint every N scenarios

    if n_workers <= 1:
        for i, (rt, sq) in enumerate(remaining):
            t0 = time.time()
            r = run_one_scenario(pre, rt, sq)
            elapsed = time.time() - t0
            done_count += 1
            if r is None:
                skipped.append([rt, sq])
                print(f'  [{done_count}/{total}] rt{rt:.2f}_sq{sq:.2f} SKIP  ({elapsed:.0f}s)')
            else:
                results.append(r)
                print(f'  [{done_count}/{total}] {r["scenario"]} ODs={r["n_ods"]:,} | '
                      f'rho={r["train_rho_sq"]:.4f}/{r["test_rho_sq"]:.4f} | '
                      f'FPR={r["test_fpr"]:.3f} | {elapsed:.0f}s')
            # Periodic checkpoint
            if (i + 1) % ckpt_interval == 0:
                _save_checkpoint(results, skipped, suffix)
                print(f'    [checkpoint saved: {len(results)} results]')
    else:
        global _GLOBAL_PRE
        _GLOBAL_PRE = pre
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 initargs=(pre,)) as pool:
            futures = {pool.submit(_worker_fn, c): c for c in remaining}
            batch_done = 0
            for fut in as_completed(futures):
                batch_done += 1
                done_count += 1
                rt, sq = futures[fut]
                r = fut.result()
                if r is None:
                    skipped.append([rt, sq])
                    print(f'  [{done_count}/{total}] rt{rt:.2f}_sq{sq:.2f} SKIP')
                else:
                    results.append(r)
                    print(f'  [{done_count}/{total}] {r["scenario"]} ODs={r["n_ods"]:,} | '
                          f'rho={r["train_rho_sq"]:.4f}/{r["test_rho_sq"]:.4f} | '
                          f'FPR={r["test_fpr"]:.3f}')
                if batch_done % ckpt_interval == 0:
                    _save_checkpoint(results, skipped, suffix)
                    print(f'    [checkpoint saved: {len(results)} results]')

    # Final checkpoint
    _save_checkpoint(results, skipped, suffix)
    print(f'    [final checkpoint saved: {len(results)} results]')

    elapsed_total = (time.time() - t_total) / 60
    print(f'\nDone ({step_label}): {len(results)} scenarios, {elapsed_total:.1f} min total')
    if remaining:
        avg = elapsed_total * 60 / len(remaining)
        print(f'  Average: {avg:.1f}s per scenario (this run)')
    return results


# ══════════════════════════════════════════════════════════════════
#  Save & plot (unchanged logic)
# ══════════════════════════════════════════════════════════════════
CSV_COLS = ['scenario', 'route', 'sequence', 'mode_weight', 'n_ods',
            'train_rho_sq', 'test_rho_sq', 'test_fpr', 'test_rmse',
            'beta_bus_IVT', 'beta_train_IVT', 'beta_ln_access', 'beta_transfers', 'beta_train',
            'converged']


def save_results(results, suffix):
    df_out = pd.DataFrame(results)
    df_out[CSV_COLS].to_csv(OUT_DIR / f'sensitivity_2d_grid_{suffix}.csv', index=False)
    with open(OUT_DIR / f'sensitivity_2d_grid_{suffix}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f'Saved {len(results)} scenarios → sensitivity_2d_grid_{suffix}.csv / .json')
    return df_out


def make_heatmap(df_2d, value_col, title, fmt, cmap, vmin=None, vmax=None):
    pivot = df_2d.pivot(index='route', columns='sequence', values=value_col)
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
    ax.set_xlabel('Sequence weight ($w_{sequence}$)', fontsize=12)
    ax.set_ylabel('Route weight ($w_{route}$)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    return fig


def plot_beta_heatmaps(df_grid, step_label, out_suffix):
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    for ax, feat, lbl in zip(axes, KEY_BETAS, KEY_LABELS):
        df_grid['_tmp'] = df_grid['betas'].apply(lambda d: d.get(feat, 0.0))
        pivot = df_grid.pivot(index='route', columns='sequence', values='_tmp')
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
        ax.set_xlabel('$w_{sequence}$', fontsize=9)
        ax.set_ylabel('$w_{route}$', fontsize=9)
        vals = pivot.values[~np.isnan(pivot.values)]
        cv = np.std(vals) / abs(np.mean(vals)) * 100 if abs(np.mean(vals)) > 1e-10 else 0
        ax.set_title(f'{lbl}\nCV={cv:.1f}%', fontsize=10, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)
        df_grid.drop(columns=['_tmp'], inplace=True)
    fig.suptitle(f'$\\beta$ Coefficient Stability ({step_label})',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f'fig_heatmap_betas_{out_suffix}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_plots(df_grid, suffix, step_label):
    for col, title_tpl, fmt, cmap, vmin, vmax in [
        ('test_rho_sq', 'Test McFadden $\\rho^2$ ({})', '{:.3f}', 'YlOrRd', 0.43, 0.50),
        ('test_fpr', 'First Preference Recovery ({})', '{:.1%}', 'YlGn', 0.68, 0.76),
        ('n_ods', 'Number of OD Pairs ({})', '{:.0f}', 'PuBu', None, None),
    ]:
        fig = make_heatmap(df_grid, col, title_tpl.format(step_label),
                           fmt, cmap, vmin, vmax)
        fig.savefig(OUT_DIR / f'fig_heatmap_{col}_{suffix}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    plot_beta_heatmaps(df_grid, step_label, suffix)
    print(f'Plots saved ({suffix}).')


def print_robustness(df_grid, label):
    rho_vals = df_grid['test_rho_sq'].values
    fpr_vals = df_grid['test_fpr'].values
    print('=' * 65)
    print(f'2D Grid Robustness - {label}  ({len(df_grid)} scenarios)')
    print('=' * 65)
    print(f'rho2: [{rho_vals.min():.4f}, {rho_vals.max():.4f}]  '
          f'spread={rho_vals.max() - rho_vals.min():.4f}')
    print(f'FPR:  [{fpr_vals.min()*100:.1f}%, {fpr_vals.max()*100:.1f}%]  '
          f'spread={(fpr_vals.max()-fpr_vals.min())*100:.1f}%p')
    best_rho = df_grid.loc[df_grid['test_rho_sq'].idxmax()]
    best_fpr = df_grid.loc[df_grid['test_fpr'].idxmax()]
    print(f'  Best rho2: {best_rho["scenario"]}  (rho2={best_rho["test_rho_sq"]:.4f})')
    print(f'  Best FPR:  {best_fpr["scenario"]}  (FPR={best_fpr["test_fpr"]:.3f})')
    for feat, lbl in zip(KEY_BETAS, KEY_LABELS):
        vals = df_grid['betas'].apply(lambda d: d.get(feat, 0.0)).values
        m = np.mean(vals)
        cv = np.std(vals) / abs(m) * 100 if abs(m) > 1e-10 else 0
        print(f'  {lbl:<18}: mean={m:>10.6f}  CV={cv:>6.1f}%  '
              f'{"ROBUST" if cv < 20 else "VARIABLE"}')
    print(f'All rho2 > 0.2:     {"PASS" if all(rho_vals > 0.2) else "FAIL"}')
    print(f'All FPR > 60%:      {"PASS" if all(fpr_vals > 0.6) else "FAIL"}')
    print(f'rho2 spread < 0.05: {"PASS" if (rho_vals.max()-rho_vals.min()) < 0.05 else "FAIL"}')
    print(f'FPR spread < 5%p:   {"PASS" if (fpr_vals.max()-fpr_vals.min()) < 0.05 else "FAIL"}')
    print()


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='2D Grid Sensitivity Analysis (Optimized)')
    parser.add_argument('--step', nargs='+', type=float, default=[0.05],
                        help='Grid step sizes (e.g. 0.05 0.01)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore checkpoint and run from scratch')
    parser.add_argument('--route-range', nargs=2, type=float, default=None,
                        metavar=('MIN', 'MAX'),
                        help='Route weight range (e.g. --route-range 0.80 1.00)')
    parser.add_argument('--seq-range', nargs=2, type=float, default=None,
                        metavar=('MIN', 'MAX'),
                        help='Sequence weight range (e.g. --seq-range 0.00 0.20)')
    args = parser.parse_args()

    # Load data
    import sqlite3
    import pickle

    print('Loading data...')
    df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')
    print(f'{len(df):,} rows, {df["od_pair"].nunique():,} ODs')

    # K1 피처 생성: 모드별 IVT
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
    df['bus_ivt_min'] = df['bus_ivt_min'].fillna(0)
    df['train_ivt_min'] = df['train_ivt_min'].fillna(0)
    df['gtx_ivt_min'] = df['gtx_ivt_min'].fillna(0)
    del ivt_records, ivt_df
    print(f'모드별 IVT 추출 완료')

    # K1 피처 생성: log walk, fare
    for col in ['access_time', 'egress_time', 'transfer_walk_time']:
        df[col + '_min'] = df[col] / 60
    df['ln_access'] = np.log1p(df['access_time_min'])
    df['ln_egress'] = np.log1p(df['egress_time_min'])
    df['fare_1000won'] = df['fare'] / 1000

    # Build pre-indexed structure ONCE
    pre = PreIndexed(df)
    del df  # free memory

    for step in args.step:
        # Range defaults
        rt_min, rt_max = (args.route_range if args.route_range else [0.0, 1.0])
        sq_min, sq_max = (args.seq_range if args.seq_range else [0.0, 1.0])

        route_vals = np.round(np.arange(rt_min, rt_max + step/2, step), 2).tolist()
        seq_vals = np.round(np.arange(sq_min, sq_max + step/2, step), 2).tolist()
        total = sum(1 for rt in route_vals for sq in seq_vals if (1.0 - rt - sq) >= -1e-9)

        range_tag = ''
        if args.route_range or args.seq_range:
            range_tag = f'_rt{rt_min:.0e}-{rt_max:.0e}_sq{sq_min:.0e}-{sq_max:.0e}'
            range_tag = f'_rt{str(rt_min).replace(".","")}-{str(rt_max).replace(".","")}_sq{str(sq_min).replace(".","")}-{str(sq_max).replace(".","")}'
        suffix = f'step{str(step).replace(".", "")}{range_tag}'
        step_label = f'step={step}'
        if range_tag:
            step_label += f', route=[{rt_min},{rt_max}], seq=[{sq_min},{sq_max}]'

        print(f'\n{"="*65}')
        print(f'Grid {step_label}: route {len(route_vals)} × sequence {len(seq_vals)} = {total} combos')
        print(f'{"="*65}')

        results = run_2d_grid(pre, route_vals, seq_vals, step_label, args.workers,
                              use_cache=not args.no_cache)
        # sort by scenario for consistent output
        results.sort(key=lambda r: (r['route'], r['sequence']))
        df_grid = save_results(results, suffix)
        save_plots(df_grid, suffix, step_label)
        print_robustness(df_grid, step_label)

    print('\nAll done!')


if __name__ == '__main__':
    main()
