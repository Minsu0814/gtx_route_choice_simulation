"""
Raptor GC Parameter Calibration.
Reconstructs generalized cost with different reluctance parameters
and finds the combination that best matches actual route choices.

Current Raptor GC formula (KoreanCostCalculator):
  GC = boardCost + transferCost*n_transfers
       + waitTime*waitReluctance
       + transitTime*modeReluctance
       + walkTime*walkReluctance

Current config (router-config.json): all reluctances = 1.0
  boardCost=60s, transferCost=120s
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

# Feature columns (all in seconds in the parquet)
TIME_COLS = ['waiting_time', 'in_vehicle_time',
             'access_time', 'egress_time', 'transfer_walk_time']


def load_labeled_data(assign_path: Path, train_path: Path) -> pd.DataFrame:
    """Load assignment + training labels, return labeled rows only."""
    df_a = pd.read_parquet(assign_path)
    df_t = pd.read_parquet(train_path, columns=['od_pair', 'alt_idx', 'chosen'])
    df = df_a.merge(df_t, on=['od_pair', 'alt_idx'], how='inner')

    # Fill NaN time columns with 0
    for col in TIME_COLS + ['num_transfers']:
        df[col] = df[col].fillna(0)
    return df


def compute_gc(df: pd.DataFrame, board: float, transfer: float,
               wait_rel: float, transit_rel: float,
               walk_rel: float) -> np.ndarray:
    """Compute parametric generalized cost (seconds)."""
    walk_time = (df['access_time'].values + df['egress_time'].values
                 + df['transfer_walk_time'].values)
    return (board
            + transfer * df['num_transfers'].values
            + wait_rel * df['waiting_time'].values
            + transit_rel * df['in_vehicle_time'].values
            + walk_rel * walk_time)


def gc_top1_accuracy(df: pd.DataFrame, gc: np.ndarray) -> float:
    """Fraction of chosen=1 rows that are GC-minimum in their OD."""
    df = df.copy()
    df['_gc'] = gc
    min_gc = df.groupby('od_pair')['_gc'].transform('min')
    df['_is_min_gc'] = df['_gc'] == min_gc

    chosen = df[df['chosen'] == 1]
    return chosen['_is_min_gc'].mean()


def verify_current_gc(df: pd.DataFrame) -> dict:
    """Verify GC reconstruction against stored generalized_cost."""
    gc_recon = compute_gc(df, 60, 120, 1.0, 1.0, 1.0)
    gc_stored = df['generalized_cost'].values

    diff = gc_recon - gc_stored
    return {
        'mean_diff': float(np.nanmean(diff)),
        'std_diff': float(np.nanstd(diff)),
        'max_abs_diff': float(np.nanmax(np.abs(diff))),
        'corr': float(np.corrcoef(gc_recon[~np.isnan(gc_stored)],
                                    gc_stored[~np.isnan(gc_stored)])[0, 1]),
    }


def grid_search(df: pd.DataFrame, param_grid: dict) -> pd.DataFrame:
    """Grid search over parameter combinations.

    Args:
        param_grid: dict with keys board, transfer, wait_rel,
                    transit_rel, walk_rel → lists of values
    Returns:
        DataFrame sorted by accuracy (descending)
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        gc = compute_gc(df, **params)
        acc = gc_top1_accuracy(df, gc)
        results.append({**params, 'accuracy': acc})

    return pd.DataFrame(results).sort_values('accuracy', ascending=False)


def print_calibration_report(verify: dict, grid_df: pd.DataFrame,
                             current_acc: float):
    """Print formatted calibration report."""
    print('=' * 70)
    print('GC RECONSTRUCTION VERIFICATION')
    print('=' * 70)
    print(f'  Mean diff:    {verify["mean_diff"]:.2f}s')
    print(f'  Std diff:     {verify["std_diff"]:.2f}s')
    print(f'  Max |diff|:   {verify["max_abs_diff"]:.2f}s')
    print(f'  Correlation:  {verify["corr"]:.6f}')

    print(f'\n{"=" * 70}')
    print('CURRENT PARAMETERS → Chosen=GC-min accuracy')
    print('=' * 70)
    print(f'  board=60, transfer=120, wait=1.0, transit=1.0, walk=1.0')
    print(f'  Accuracy: {current_acc:.1%}')

    print(f'\n{"=" * 70}')
    print('TOP 10 PARAMETER COMBINATIONS')
    print('=' * 70)
    top10 = grid_df.head(10)
    print(top10.to_string(index=False, float_format='%.3f'))

    best = grid_df.iloc[0]
    print(f'\n{"=" * 70}')
    print('BEST PARAMETERS')
    print('=' * 70)
    print(f'  board:       {best["board"]:.0f}s')
    print(f'  transfer:    {best["transfer"]:.0f}s')
    print(f'  wait_rel:    {best["wait_rel"]:.2f}')
    print(f'  transit_rel: {best["transit_rel"]:.2f}')
    print(f'  walk_rel:    {best["walk_rel"]:.2f}')
    print(f'  Accuracy:    {best["accuracy"]:.1%}  '
          f'(+{best["accuracy"] - current_acc:.1%} vs current)')


def run_calibration(assign_path: Path, train_path: Path) -> dict:
    """Run full calibration pipeline. Returns results dict."""
    print('Loading data...')
    df = load_labeled_data(assign_path, train_path)
    print(f'  {len(df):,} labeled rows, '
          f'{df["od_pair"].nunique():,} ODs')

    # Step 1: Verify GC reconstruction
    verify = verify_current_gc(df)

    # Step 2: Current accuracy
    gc_current = compute_gc(df, 60, 120, 1.0, 1.0, 1.0)
    current_acc = gc_top1_accuracy(df, gc_current)

    # Step 3: Grid search
    param_grid = {
        'board': [0, 30, 60, 90, 120],
        'transfer': [0, 60, 120, 180, 240, 360],
        'wait_rel': [0.5, 1.0, 1.5, 2.0],
        'transit_rel': [0.5, 1.0, 1.5],
        'walk_rel': [1.0, 1.5, 2.0, 3.0, 4.0],
    }
    print(f'\nGrid search: {np.prod([len(v) for v in param_grid.values()]):,} '
          f'combinations...')
    grid_df = grid_search(df, param_grid)

    print_calibration_report(verify, grid_df, current_acc)

    return {
        'verification': verify,
        'current_accuracy': current_acc,
        'grid_results': grid_df.head(20).to_dict('records'),
        'best_params': grid_df.iloc[0].to_dict(),
    }
