"""
Route Choice Criterion Analysis.
Labels alternatives by what criterion they minimize, then computes
statistics on actual choices vs model predictions.
"""
import numpy as np
import pandas as pd


# Criteria to analyze: (column_name, label, minimize=True)
CRITERIA = [
    ('generalized_cost', 'min_gc', True),
    ('num_transfers', 'min_transfers', True),
    ('walk_distance', 'min_walk', True),
    ('total_duration', 'min_duration', True),
    ('fare', 'min_fare', True),
]


def label_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean columns for each criterion (is_min_*).

    For each OD group, marks alternatives that achieve the minimum
    value for each criterion. Ties → all tied alts marked True.
    """
    df = df.copy()
    for col, label, minimize in CRITERIA:
        if col not in df.columns:
            df[f'is_{label}'] = False
            continue
        col_vals = df[col].fillna(np.inf if minimize else -np.inf)
        if minimize:
            group_min = col_vals.groupby(df['od_pair']).transform('min')
            df[f'is_{label}'] = col_vals == group_min
        else:
            group_max = col_vals.groupby(df['od_pair']).transform('max')
            df[f'is_{label}'] = col_vals == group_max
    return df


def add_model_top1(df: pd.DataFrame) -> pd.DataFrame:
    """Add model_top1 column: 1 for highest-probability alt per OD."""
    df = df.copy()
    max_prob = df.groupby('od_pair')['prob'].transform('max')
    df['model_top1'] = (df['prob'] == max_prob).astype(int)
    return df


def compute_single_criterion_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute % of chosen/model_top1 that match each criterion.

    Returns DataFrame with columns: criterion, chosen_pct, model_pct
    """
    has_chosen = 'chosen' in df.columns
    rows = []
    for _, label, _ in CRITERIA:
        flag_col = f'is_{label}'
        if flag_col not in df.columns:
            continue

        if has_chosen:
            chosen_mask = df['chosen'] == 1
            chosen_pct = df.loc[chosen_mask, flag_col].mean() * 100
        else:
            chosen_pct = None

        top1_mask = df['model_top1'] == 1
        model_pct = df.loc[top1_mask, flag_col].mean() * 100

        rows.append({
            'criterion': label,
            'chosen_pct': chosen_pct,
            'model_top1_pct': model_pct,
        })
    return pd.DataFrame(rows)


def compute_cross_table(df: pd.DataFrame, target='chosen') -> pd.DataFrame:
    """Cross-tabulation: for chosen (or model_top1) rows,
    what % also satisfy each other criterion.

    Returns square DataFrame: rows = primary criterion, cols = secondary.
    """
    if target == 'chosen':
        mask = df['chosen'] == 1
    else:
        mask = df['model_top1'] == 1
    subset = df[mask]

    labels = [l for _, l, _ in CRITERIA if f'is_{l}' in df.columns]
    cross = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for primary in labels:
        primary_rows = subset[subset[f'is_{primary}']]
        n_primary = len(primary_rows)
        if n_primary == 0:
            continue
        for secondary in labels:
            cross.loc[primary, secondary] = (
                primary_rows[f'is_{secondary}'].mean() * 100
            )
    return cross


def compute_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute criterion stats broken down by a grouping column.

    Returns long-form DataFrame: group_val, criterion, chosen_pct, model_pct
    """
    has_chosen = 'chosen' in df.columns
    rows = []
    for gval, gdf in df.groupby(group_col):
        for _, label, _ in CRITERIA:
            flag_col = f'is_{label}'
            if flag_col not in gdf.columns:
                continue

            if has_chosen:
                cmask = gdf['chosen'] == 1
                cpct = gdf.loc[cmask, flag_col].mean() * 100 if cmask.any() else None
            else:
                cpct = None

            tmask = gdf['model_top1'] == 1
            mpct = gdf.loc[tmask, flag_col].mean() * 100 if tmask.any() else None

            rows.append({
                group_col: gval,
                'criterion': label,
                'chosen_pct': cpct,
                'model_top1_pct': mpct,
            })
    return pd.DataFrame(rows)


def print_report(single: pd.DataFrame, cross_chosen: pd.DataFrame,
                 cross_model: pd.DataFrame, by_cs: pd.DataFrame):
    """Print formatted analysis report."""
    print('\n' + '=' * 70)
    print('TABLE 1: Single Criterion Match Rate')
    print('=' * 70)
    print(f'{"Criterion":<20} {"Chosen %":>12} {"Model Top-1 %":>15}')
    print('-' * 50)
    for _, r in single.iterrows():
        ch = f'{r["chosen_pct"]:.1f}%' if r['chosen_pct'] is not None else 'N/A'
        mo = f'{r["model_top1_pct"]:.1f}%'
        print(f'{r["criterion"]:<20} {ch:>12} {mo:>15}')

    print('\n' + '=' * 70)
    print('TABLE 2: Cross-tabulation (Chosen)')
    print('=' * 70)
    print(cross_chosen.round(1).to_string())

    print('\n' + '=' * 70)
    print('TABLE 3: Cross-tabulation (Model Top-1)')
    print('=' * 70)
    print(cross_model.round(1).to_string())

    if by_cs is not None and not by_cs.empty:
        print('\n' + '=' * 70)
        print('TABLE 4: By Choice Set Size (Chosen %)')
        print('=' * 70)
        pivot = by_cs.pivot_table(
            index='choice_set_size', columns='criterion',
            values='chosen_pct', aggfunc='first',
        )
        print(pivot.round(1).to_string())
