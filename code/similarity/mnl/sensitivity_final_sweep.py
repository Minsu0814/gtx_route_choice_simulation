"""
Final Sweep: threshold + sequence gate sensitivity
Confirmed weights: route=0.90, seq=0.08, mode=0.02

Usage:
    python sensitivity_final_sweep.py
    python sensitivity_final_sweep.py --no-cache
"""
import argparse
import time
import json
import warnings
from pathlib import Path

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

# ── Confirmed weights ─────────────────────────────────────────────
W_ROUTE = 0.90
W_SEQ = 0.08
W_MODE = 0.02

# ── Sweep parameters ──────────────────────────────────────────────
THRESHOLD_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SEQ_GATE_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

BASELINE_THRESHOLD = 0.5
BASELINE_SEQ_GATE = 0.3

# ── Model features ────────────────────────────────────────────────
MODEL_FEATURES = [
    'in_vehicle_time_min', 'wait_time_min',
    'access_time_min', 'egress_time_min', 'transfer_walk_time_min',
    'total_distance_km', 'num_transfers', 'fare',
    'has_bus', 'has_train', 'has_gtx',
]
N_FEAT = len(MODEL_FEATURES)
SIGN_CONSTRAINED = {
    'in_vehicle_time_min', 'wait_time_min',
    'access_time_min', 'egress_time_min', 'transfer_walk_time_min',
    'total_distance_km', 'num_transfers', 'fare',
}
BOUNDS = [(None, 0) if f in SIGN_CONSTRAINED else (None, None) for f in MODEL_FEATURES]
KEY_BETAS = ['in_vehicle_time_min', 'access_time_min', 'num_transfers', 'has_bus', 'has_train']
KEY_LABELS = ['beta_IVT', 'beta_access', 'beta_transfers', 'beta_bus', 'beta_train']


# ══════════════════════════════════════════════════════════════════
#  PreIndexed (reused from sensitivity_2d_grid.py)
# ══════════════════════════════════════════════════════════════════
class PreIndexed:
    def __init__(self, df):
        t0 = time.time()
        df = df.sort_values('od_pair').reset_index(drop=True)
        od = df['od_pair'].values
        change = np.concatenate([[True], od[1:] != od[:-1]])
        self.starts = np.where(change)[0]
        self.sizes = np.diff(np.append(self.starts, len(od)))
        self.n_groups = len(self.starts)
        self.od_labels = od[self.starts]
        self.sim_mode = df['sim_mode'].values.astype(np.float32)
        self.sim_seq = df['sim_sequence'].values.astype(np.float32)
        self.sim_route = df['sim_route'].values.astype(np.float32)
        self.choice_prob = df['choice_prob'].values.astype(np.float64)
        self.n_total = df['n_total'].values.astype(np.float64)
        self.X = df[MODEL_FEATURES].values.astype(np.float64)
        self.dom_idx = np.empty(self.n_groups, dtype=np.int64)
        self.valid_prob = np.ones(self.n_groups, dtype=bool)
        self.group_weight = np.empty(self.n_groups, dtype=np.float64)
        for g in range(self.n_groups):
            s, sz = self.starts[g], self.sizes[g]
            prob_g = self.choice_prob[s:s+sz]
            self.dom_idx[g] = s + np.argmax(prob_g)
            self.group_weight[g] = self.n_total[s]
            if abs(prob_g.sum() - 1.0) > 0.01:
                self.valid_prob[g] = False
        rng = np.random.RandomState(42)
        perm = rng.permutation(self.n_groups)
        n_test = int(self.n_groups * 0.2)
        self.test_groups = np.sort(perm[:n_test])
        self.train_groups = np.sort(perm[n_test:])
        print(f'PreIndexed built: {self.n_groups:,} groups, {len(df):,} rows  ({time.time()-t0:.1f}s)')

    def composite_at_dom(self):
        idx = self.dom_idx
        return (W_MODE * self.sim_mode[idx]
                + W_ROUTE * self.sim_route[idx]
                + W_SEQ * self.sim_seq[idx])

    def seq_at_dom(self):
        return self.sim_seq[self.dom_idx]

    def build_flat(self, group_mask):
        gids = np.where(group_mask)[0]
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
#  MNL
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


def compute_fpr(beta, flat):
    X, y, gid, ng = flat['X'], flat['y'], flat['gid'], flat['n_groups']
    V = X @ beta
    V_max = np.full(ng, -np.inf)
    np.maximum.at(V_max, gid, V)
    exp_V = np.exp(V - V_max[gid])
    sum_exp = np.bincount(gid, weights=exp_V, minlength=ng)
    pred = exp_V / sum_exp[gid]
    pred_max = np.full(ng, -np.inf)
    np.maximum.at(pred_max, gid, pred)
    pred_winner = (pred == pred_max[gid])
    y_max = np.full(ng, -np.inf)
    np.maximum.at(y_max, gid, y)
    y_winner = (y == y_max[gid])
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
    fpr = compute_fpr(b, test_flat)
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
#  Scenario runner
# ══════════════════════════════════════════════════════════════════
def run_scenario(pre, threshold, min_seq, label):
    t0 = time.time()
    dom_comp = pre.composite_at_dom()
    dom_seq = pre.seq_at_dom()

    keep = (dom_comp >= threshold) & pre.valid_prob & (pre.sizes >= 2)
    keep &= (dom_seq >= min_seq)
    n_ods = keep.sum()

    if n_ods < 100:
        print(f'  [{label}] SKIP (n_ods={n_ods})')
        return None

    train_mask = keep.copy()
    test_mask = keep.copy()
    train_mask[pre.test_groups] = False
    test_mask[pre.train_groups] = False

    if train_mask.sum() < 50 or test_mask.sum() < 50:
        print(f'  [{label}] SKIP (train/test too small)')
        return None

    train_flat = pre.build_flat(train_mask)
    test_flat = pre.build_flat(test_mask)
    r = estimate_mnl(train_flat, test_flat)
    r.update({
        'scenario': label, 'n_ods': int(n_ods),
        'threshold': threshold, 'min_seq': min_seq,
    })
    for feat, key in zip(KEY_BETAS, KEY_LABELS):
        r[key] = r['betas'].get(feat, 0.0)
    elapsed = time.time() - t0
    print(f'  [{label}] ODs={n_ods:,} | '
          f'rho={r["train_rho_sq"]:.4f}/{r["test_rho_sq"]:.4f} | '
          f'FPR={r["test_fpr"]:.3f} | {elapsed:.0f}s')
    return r


# ══════════════════════════════════════════════════════════════════
#  Checkpoint
# ══════════════════════════════════════════════════════════════════
CKPT_PATH = OUT_DIR / 'checkpoints' / 'ckpt_final_sweep.json'
CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_checkpoint():
    if not CKPT_PATH.exists():
        return {}
    with open(CKPT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Checkpoint loaded: {len(data)} sweeps')
    return data


def save_checkpoint(data):
    tmp = CKPT_PATH.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    tmp.replace(CKPT_PATH)


# ══════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════
def plot_sweep(df_sweep, x_col, x_label, suffix, baseline_val=None):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = range(len(df_sweep))
    xlabels = [f'{v}' for v in df_sweep[x_col]]

    color_rho = '#D32F2F'
    color_fpr = '#4CAF50'

    ax1.plot(x, df_sweep['test_rho_sq'], 'o-', color=color_rho,
             label=r'Test $\rho^2$', markersize=8, lw=2)
    ax1.set_ylabel(r'Test $\rho^2$', color=color_rho, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_rho)

    ax2 = ax1.twinx()
    ax2.plot(x, df_sweep['test_fpr'] * 100, 's-', color=color_fpr,
             label='FPR (%)', markersize=8, lw=2)
    ax2.set_ylabel('FPR (%)', color=color_fpr, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_fpr)

    # n_ods as bar background
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('axes', 1.15))
    ax3.bar(x, df_sweep['n_ods'] / 1000, color='gray', alpha=0.15, width=0.6)
    ax3.set_ylabel('N(OD) [K]', color='gray', fontsize=10)
    ax3.tick_params(axis='y', labelcolor='gray')

    if baseline_val is not None:
        idx = list(df_sweep[x_col]).index(baseline_val) if baseline_val in list(df_sweep[x_col]) else None
        if idx is not None:
            ax1.axvline(idx, color='navy', ls='--', alpha=0.5, lw=1.5)
            ax1.text(idx + 0.1, ax1.get_ylim()[1], 'baseline',
                     color='navy', fontsize=9, va='top')

    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=10)
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_title(f'Final Sweep: {x_label}\n(route={W_ROUTE}, seq={W_SEQ})',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT_DIR / f'fig_final_sweep_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot saved: fig_final_sweep_{suffix}.png')


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Final Sweep: threshold + seq gate')
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()

    print(f'Confirmed weights: route={W_ROUTE}, seq={W_SEQ}, mode={W_MODE}')
    print(f'Threshold values: {THRESHOLD_VALUES}')
    print(f'Seq gate values:  {SEQ_GATE_VALUES}')
    print()

    # Load data
    print('Loading data...')
    df = pd.read_parquet(DATA_DIR / 'route_choice_training.parquet')
    print(f'{len(df):,} rows, {df["od_pair"].nunique():,} ODs')

    for col in ['in_vehicle_time', 'wait_time', 'access_time', 'egress_time', 'transfer_walk_time']:
        df[col + '_min'] = df[col] / 60
    df['total_distance_km'] = df['total_distance'] / 1000

    pre = PreIndexed(df)
    del df

    cache = load_checkpoint() if not args.no_cache else {}

    # ── Sweep A: Threshold (fixed seq gate = 0.3) ────────────────
    sweep_key = 'threshold'
    print(f'\n{"="*65}')
    print(f'SWEEP A: Threshold (seq_gate={BASELINE_SEQ_GATE})')
    print(f'{"="*65}')

    if sweep_key in cache and not args.no_cache:
        results_thr = cache[sweep_key]
        print(f'  Loaded from cache: {len(results_thr)} results')
    else:
        results_thr = []
        for thr in THRESHOLD_VALUES:
            r = run_scenario(pre, thr, BASELINE_SEQ_GATE, f'thr_{thr:.1f}')
            if r:
                results_thr.append(r)
        cache[sweep_key] = results_thr
        save_checkpoint(cache)

    # ── Sweep B: Seq gate (fixed threshold = 0.5) ────────────────
    sweep_key = 'seq_gate'
    print(f'\n{"="*65}')
    print(f'SWEEP B: Sequence gate (threshold={BASELINE_THRESHOLD})')
    print(f'{"="*65}')

    if sweep_key in cache and not args.no_cache:
        results_sg = cache[sweep_key]
        print(f'  Loaded from cache: {len(results_sg)} results')
    else:
        results_sg = []
        for sg in SEQ_GATE_VALUES:
            r = run_scenario(pre, BASELINE_THRESHOLD, sg, f'seq_gate_{sg:.1f}')
            if r:
                results_sg.append(r)
        cache[sweep_key] = results_sg
        save_checkpoint(cache)

    # ── Save CSVs ─────────────────────────────────────────────────
    CSV_COLS = ['scenario', 'threshold', 'min_seq', 'n_ods',
                'train_rho_sq', 'test_rho_sq', 'test_fpr', 'test_rmse',
                'beta_IVT', 'beta_access', 'beta_transfers', 'beta_bus', 'beta_train',
                'converged']

    df_thr = pd.DataFrame(results_thr)
    df_thr[CSV_COLS].to_csv(OUT_DIR / 'final_sweep_threshold.csv', index=False)

    df_sg = pd.DataFrame(results_sg)
    df_sg[CSV_COLS].to_csv(OUT_DIR / 'final_sweep_seq_gate.csv', index=False)

    print(f'\nSaved: final_sweep_threshold.csv ({len(df_thr)} rows)')
    print(f'Saved: final_sweep_seq_gate.csv ({len(df_sg)} rows)')

    # ── Print results ─────────────────────────────────────────────
    print(f'\n{"="*65}')
    print('THRESHOLD SWEEP (seq_gate=0.3)')
    print(f'{"="*65}')
    print(f'{"Threshold":<10} {"N(OD)":>8} {"Train ρ²":>10} {"Test ρ²":>10} {"FPR":>8} {"RMSE":>8}')
    print('-' * 56)
    for _, r in df_thr.iterrows():
        mark = ' <--' if r['threshold'] == BASELINE_THRESHOLD else ''
        print(f'{r["threshold"]:<10.1f} {r["n_ods"]:>8,.0f} {r["train_rho_sq"]:>10.4f} '
              f'{r["test_rho_sq"]:>10.4f} {r["test_fpr"]*100:>7.1f}% {r["test_rmse"]:>8.4f}{mark}')

    print(f'\n{"="*65}')
    print('SEQ GATE SWEEP (threshold=0.5)')
    print(f'{"="*65}')
    print(f'{"Seq Gate":<10} {"N(OD)":>8} {"Train ρ²":>10} {"Test ρ²":>10} {"FPR":>8} {"RMSE":>8}')
    print('-' * 56)
    for _, r in df_sg.iterrows():
        mark = ' <--' if r['min_seq'] == BASELINE_SEQ_GATE else ''
        print(f'{r["min_seq"]:<10.1f} {r["n_ods"]:>8,.0f} {r["train_rho_sq"]:>10.4f} '
              f'{r["test_rho_sq"]:>10.4f} {r["test_fpr"]*100:>7.1f}% {r["test_rmse"]:>8.4f}{mark}')

    # ── Plots ─────────────────────────────────────────────────────
    plot_sweep(df_thr, 'threshold', 'Matching Threshold', 'threshold',
               baseline_val=BASELINE_THRESHOLD)
    plot_sweep(df_sg, 'min_seq', 'Minimum sim_sequence Gate', 'seq_gate',
               baseline_val=BASELINE_SEQ_GATE)

    # ── Final summary ─────────────────────────────────────────────
    baseline = df_thr[df_thr['threshold'] == BASELINE_THRESHOLD]
    if len(baseline) > 0:
        b = baseline.iloc[0]
        print(f'\n{"="*65}')
        print('FINAL CONFIRMED PARAMETERS')
        print(f'{"="*65}')
        print(f'  Weights:     route={W_ROUTE}, seq={W_SEQ}, mode={W_MODE}')
        print(f'  Threshold:   {BASELINE_THRESHOLD}')
        print(f'  Seq gate:    {BASELINE_SEQ_GATE}')
        print(f'  N(OD):       {b["n_ods"]:,.0f}')
        print(f'  Train ρ²:    {b["train_rho_sq"]:.4f}')
        print(f'  Test ρ²:     {b["test_rho_sq"]:.4f}')
        print(f'  FPR:         {b["test_fpr"]*100:.1f}%')
        print(f'  RMSE:        {b["test_rmse"]:.4f}')

    print('\nAll done!')


if __name__ == '__main__':
    main()
