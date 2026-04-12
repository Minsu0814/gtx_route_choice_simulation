"""
Raptor GC Parameter Calibration (v2).

MNL β-informed grid search with per-mode transit reluctance
and multi-objective scoring.

GC = boardCost + transferCost*n_transfers
     + waitTime*waitReluctance
     + busTime*busReluctance + trainTime*trainReluctance
     + walkTime*walkReluctance

MNL Spec A β ratios (anchored to IVT):
  walk_rel ≈ 10.1, transfer_cost ≈ 9387s, fare ≈ 45min/1000won
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

# Feature columns (all in seconds in the parquet)
TIME_COLS = ['waiting_time', 'in_vehicle_time',
             'access_time', 'egress_time', 'transfer_walk_time']
MODE_IVT_COLS = ['ivt_bus', 'ivt_train', 'ivt_gtx']


def load_labeled_data(assign_path: Path, train_path: Path) -> pd.DataFrame:
    """Load assignment + training labels with mode-specific IVT."""
    df_a = pd.read_parquet(assign_path)
    train_cols = ['od_pair', 'alt_idx', 'chosen',
                  'ivt_bus', 'ivt_train', 'ivt_gtx', 'transport_category']
    df_t = pd.read_parquet(train_path, columns=train_cols)
    df = df_a.merge(df_t[['od_pair', 'alt_idx', 'chosen',
                           'ivt_bus', 'ivt_train', 'ivt_gtx']],
                    on=['od_pair', 'alt_idx'], how='inner')

    for col in TIME_COLS + ['num_transfers'] + MODE_IVT_COLS:
        df[col] = df[col].fillna(0)
    return df


def compute_gc(df: pd.DataFrame, board: float, transfer: float,
               wait_rel: float, bus_rel: float, train_rel: float,
               walk_rel: float) -> np.ndarray:
    """Compute parametric generalized cost (seconds).

    Per-mode IVT: bus_rel applies to ivt_bus,
    train_rel applies to ivt_train + ivt_gtx (rail modes).
    """
    walk_time = (df['access_time'].values + df['egress_time'].values
                 + df['transfer_walk_time'].values)
    return (board
            + transfer * df['num_transfers'].values
            + wait_rel * df['waiting_time'].values
            + bus_rel * df['ivt_bus'].values
            + train_rel * (df['ivt_train'].values + df['ivt_gtx'].values)
            + walk_rel * walk_time)


class FastScorer:
    """Pre-indexed scorer for fast grid search iteration.

    Builds OD-group indices once, then scores GC vectors in O(n) time.
    """

    def __init__(self, df: pd.DataFrame):
        od = df['od_pair'].values
        chosen_mask = df['chosen'].values == 1

        codes, _uniques = pd.factorize(od)
        self.codes = codes
        self.n_ods = len(_uniques)
        self.chosen_mask = chosen_mask

        # Chosen row's OD code
        self.chosen_od_code = codes[chosen_mask]

        # For each OD, row indices (sorted by code)
        od_order = np.argsort(codes, kind='mergesort')
        codes_sorted = codes[od_order]
        splits = np.flatnonzero(np.diff(codes_sorted) != 0) + 1
        self.sort_order = od_order
        self.group_starts = np.r_[0, splits]
        self.group_sizes = np.diff(np.r_[self.group_starts, len(codes)])

        # Build chosen-row lookup: od_code → row index
        self.chosen_row_idx = np.empty(self.n_ods, dtype=np.intp)
        self.chosen_row_idx[:] = -1
        chosen_positions = np.flatnonzero(chosen_mask)
        for pos in chosen_positions:
            self.chosen_row_idx[codes[pos]] = pos

        # Pre-extract comparison arrays for chosen rows
        self.n_chosen = len(chosen_positions)
        self.chosen_cat = df['transport_category'].values[chosen_mask] \
            if 'transport_category' in df.columns else None
        self.chosen_dur = (df['in_vehicle_time'].values[chosen_mask]
                           + df['waiting_time'].values[chosen_mask])
        self.chosen_xfer = df['num_transfers'].values[chosen_mask]

        # Same arrays for all rows (for gc-min lookup)
        self.all_cat = df['transport_category'].values \
            if 'transport_category' in df.columns else None
        self.all_dur = (df['in_vehicle_time'].values
                        + df['waiting_time'].values)
        self.all_xfer = df['num_transfers'].values

    def gc_accuracy_fast(self, gc: np.ndarray) -> float:
        """Fast GC accuracy using reduceat (O(n), zero-alloc)."""
        gc_sorted = gc[self.sort_order]
        min_gc = np.minimum.reduceat(gc_sorted, self.group_starts)
        chosen_gc = gc[self.chosen_row_idx[self.chosen_od_code]]
        return float((chosen_gc == min_gc[self.chosen_od_code]).mean())

    def score(self, gc: np.ndarray,
              w_gc: float = 0.3, w_cat: float = 0.3,
              w_dur: float = 0.2, w_xfer: float = 0.2) -> dict:
        """Full multi-objective score using lexsort (pure numpy)."""
        order = np.lexsort((gc, self.codes))
        codes_sorted = self.codes[order]
        first_of_group = np.empty(len(codes_sorted), dtype=bool)
        first_of_group[0] = True
        first_of_group[1:] = codes_sorted[1:] != codes_sorted[:-1]
        gc_min_rows = order[first_of_group]

        gc_min_for_chosen = gc_min_rows[self.chosen_od_code]
        chosen_rows_flat = self.chosen_row_idx[self.chosen_od_code]

        gc_acc = float((gc_min_for_chosen == chosen_rows_flat).mean())

        if self.chosen_cat is not None:
            cat_match = float((self.all_cat[gc_min_for_chosen]
                               == self.chosen_cat).mean())
        else:
            cat_match = 0.0

        dur_180 = float((np.abs(self.all_dur[gc_min_for_chosen]
                                - self.chosen_dur) <= 180).mean())

        xfer_match = float((self.all_xfer[gc_min_for_chosen]
                            == self.chosen_xfer).mean())

        score = (w_gc * gc_acc + w_cat * cat_match
                 + w_dur * dur_180 + w_xfer * xfer_match)

        return {
            'gc_accuracy': gc_acc,
            'category_match': cat_match,
            'duration_180s': dur_180,
            'transfer_match': xfer_match,
            'score': score,
        }


def verify_current_gc(df: pd.DataFrame) -> dict:
    """Verify GC reconstruction against stored generalized_cost."""
    gc_recon = compute_gc(df, 60, 120, 1.0, 1.0, 1.0, 1.0)
    gc_stored = df['generalized_cost'].values

    diff = gc_recon - gc_stored
    return {
        'mean_diff': float(np.nanmean(diff)),
        'std_diff': float(np.nanstd(diff)),
        'max_abs_diff': float(np.nanmax(np.abs(diff))),
        'corr': float(np.corrcoef(gc_recon[~np.isnan(gc_stored)],
                                    gc_stored[~np.isnan(gc_stored)])[0, 1]),
    }


def grid_search(df: pd.DataFrame, param_grid: dict,
                per_group: int = 30) -> pd.DataFrame:
    """Two-phase grid search with diversity guarantee.

    Phase 1: reduceat GC accuracy for all combos (~25ms each).
    Phase 2: full lexsort scoring for diverse candidates.
             Top per_group per transfer value → ensures variety.
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    print('  Building OD index...', flush=True)
    scorer = FastScorer(df)

    # Phase 1: fast GC accuracy (reduceat)
    print(f'  Phase 1: GC accuracy for {len(combos):,} combos...',
          flush=True)
    t0 = time.time()
    phase1 = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        gc = compute_gc(df, **params)
        acc = scorer.gc_accuracy_fast(gc)
        phase1.append({**params, 'gc_accuracy': acc})
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(combos) - i - 1) / rate
            print(f'    {i + 1}/{len(combos)} '
                  f'({rate:.0f}/s, ETA {eta:.0f}s)', flush=True)

    # Diverse selection: top per_group per transfer value
    df_p1 = pd.DataFrame(phase1)
    candidates = (df_p1.groupby('transfer', group_keys=False)
                  .apply(lambda g: g.nlargest(per_group, 'gc_accuracy'))
                  .reset_index(drop=True))
    n_cand = len(candidates)
    print(f'  Phase 1 done: {n_cand} candidates '
          f'({per_group}/transfer group)', flush=True)

    # Phase 2: full multi-objective
    print(f'  Phase 2: full scoring for {n_cand} candidates...',
          flush=True)
    results = []
    for _, row in candidates.iterrows():
        params = {k: row[k] for k in keys}
        gc = compute_gc(df, **params)
        metrics = scorer.score(gc)
        results.append({**params, **metrics})

    return pd.DataFrame(results).sort_values('score', ascending=False)


def print_calibration_report(verify: dict, grid_df: pd.DataFrame,
                             current_metrics: dict):
    """Print formatted calibration report."""
    print('=' * 70)
    print('GC RECONSTRUCTION VERIFICATION')
    print('=' * 70)
    print(f'  Mean diff:    {verify["mean_diff"]:.2f}s')
    print(f'  Std diff:     {verify["std_diff"]:.2f}s')
    print(f'  Max |diff|:   {verify["max_abs_diff"]:.2f}s')
    print(f'  Correlation:  {verify["corr"]:.6f}')

    print(f'\n{"=" * 70}')
    print('CURRENT (DEFAULT) PARAMETERS')
    print('=' * 70)
    print(f'  board=60, transfer=120, wait=1.0, bus=1.0, train=1.0, walk=1.0')
    for k, v in current_metrics.items():
        print(f'  {k:20s}: {v:.3f}')

    print(f'\n{"=" * 70}')
    print('TOP 10 PARAMETER COMBINATIONS (by composite score)')
    print('=' * 70)
    show_cols = ['board', 'transfer', 'wait_rel', 'bus_rel', 'train_rel',
                 'walk_rel', 'gc_accuracy', 'category_match',
                 'duration_180s', 'transfer_match', 'score']
    top10 = grid_df.head(10)[show_cols]
    print(top10.to_string(index=False, float_format='%.3f'))

    best = grid_df.iloc[0]
    print(f'\n{"=" * 70}')
    print('BEST PARAMETERS')
    print('=' * 70)
    print(f'  board:       {best["board"]:.0f}s')
    print(f'  transfer:    {best["transfer"]:.0f}s')
    print(f'  wait_rel:    {best["wait_rel"]:.2f}')
    print(f'  bus_rel:     {best["bus_rel"]:.2f}')
    print(f'  train_rel:   {best["train_rel"]:.2f}')
    print(f'  walk_rel:    {best["walk_rel"]:.2f}')
    print(f'  Score:       {best["score"]:.3f}  '
          f'(+{best["score"] - current_metrics["score"]:.3f} vs current)')
    print(f'  GC accuracy: {best["gc_accuracy"]:.1%}')
    print(f'  Category:    {best["category_match"]:.1%}')
    print(f'  Duration:    {best["duration_180s"]:.1%}')
    print(f'  Transfers:   {best["transfer_match"]:.1%}')


def run_calibration(assign_path: Path, train_path: Path) -> dict:
    """Run full calibration pipeline. Returns results dict."""
    print('Loading data...')
    df = load_labeled_data(assign_path, train_path)
    print(f'  {len(df):,} labeled rows, '
          f'{df["od_pair"].nunique():,} ODs')

    # Step 1: Verify GC reconstruction
    verify = verify_current_gc(df)

    # Step 2: Current (default) metrics
    gc_current = compute_gc(df, 60, 120, 1.0, 1.0, 1.0, 1.0)
    scorer = FastScorer(df)
    current_metrics = scorer.score(gc_current)

    # Step 3: MNL β-informed grid search
    # MNL β ratios: walk_rel≈10, transfer≈9387s, bus/train분리
    param_grid = {
        'board': [0, 60, 120],
        'transfer': [120, 300, 600, 1200, 2400, 4800],
        'wait_rel': [1.0, 1.5, 2.0],
        'bus_rel': [0.3, 0.5, 0.7, 1.0, 1.5],
        'train_rel': [0.3, 0.5, 0.7, 1.0, 1.5],
        'walk_rel': [2.0, 4.0, 6.0, 8.0, 10.0],
    }
    n_combos = int(np.prod([len(v) for v in param_grid.values()]))
    print(f'\nGrid search: {n_combos:,} combinations '
          f'(MNL β-informed ranges)...')
    grid_df = grid_search(df, param_grid)

    print_calibration_report(verify, grid_df, current_metrics)

    return {
        'verification': verify,
        'current_metrics': current_metrics,
        'grid_results': grid_df.head(20).to_dict('records'),
        'best_params': grid_df.iloc[0].to_dict(),
    }
