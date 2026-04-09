"""
Run route choice criterion analysis.
Merges assignment probabilities with training labels to analyze
what criteria people optimize when choosing routes.

Usage:
    python run_choice_criterion.py
    python run_choice_criterion.py --spec A_total_uncon
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code' / 'similarity' / 'analysis'))

from choice_criterion import (
    label_criteria, add_model_top1,
    compute_single_criterion_stats, compute_cross_table,
    compute_by_group, print_report,
)

DATA_DIR = ROOT / 'data' / 'training_set_new'
TRAIN_PATH = DATA_DIR / 'route_choice_filtered_training.parquet'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', default='A_total_uncon')
    args = parser.parse_args()

    assign_path = DATA_DIR / f'assignment_{args.spec}.parquet'
    print(f'Loading assignment: {assign_path}')
    df_assign = pd.read_parquet(assign_path)
    print(f'  {len(df_assign):,} rows, {df_assign["od_pair"].nunique():,} ODs')

    print(f'Loading training: {TRAIN_PATH}')
    df_train = pd.read_parquet(TRAIN_PATH, columns=['od_pair', 'alt_idx', 'chosen'])
    print(f'  {len(df_train):,} rows')

    # Merge chosen labels
    df = df_assign.merge(df_train, on=['od_pair', 'alt_idx'], how='left')
    n_labeled = df['chosen'].notna().sum()
    print(f'  Labeled: {n_labeled:,} / {len(df):,} ({n_labeled/len(df)*100:.1f}%)')

    # Add choice_set_size
    cs_size = df.groupby('od_pair')['alt_idx'].transform('count')
    df['choice_set_size'] = cs_size

    # Label criteria & model top-1
    df = label_criteria(df)
    df = add_model_top1(df)

    # Filter to labeled ODs for chosen analysis
    labeled = df[df['chosen'].notna()].copy()
    print(f'\nAnalyzing {labeled["od_pair"].nunique():,} labeled ODs...')

    # Compute stats
    single = compute_single_criterion_stats(labeled)
    cross_chosen = compute_cross_table(labeled, target='chosen')
    cross_model = compute_cross_table(labeled, target='model_top1')
    by_cs = compute_by_group(labeled, 'choice_set_size')

    print_report(single, cross_chosen, cross_model, by_cs)

    # By transport category
    print('\n' + '=' * 70)
    print('TABLE 5: By Transport Category (Chosen %)')
    print('=' * 70)
    by_cat = compute_by_group(labeled, 'transport_category')
    if not by_cat.empty:
        pivot = by_cat.pivot_table(
            index='transport_category', columns='criterion',
            values='chosen_pct', aggfunc='first',
        )
        print(pivot.round(1).to_string())

    # Save results
    out_path = DATA_DIR / 'choice_criterion_analysis.json'
    results = {
        'spec': args.spec,
        'n_ods_total': int(df['od_pair'].nunique()),
        'n_ods_labeled': int(labeled['od_pair'].nunique()),
        'single_criterion': single.to_dict('records'),
        'cross_chosen': cross_chosen.to_dict(),
        'cross_model': cross_model.to_dict(),
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved: {out_path}')


if __name__ == '__main__':
    main()
