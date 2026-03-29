# -*- coding: utf-8 -*-
"""
Phase 2-2: EB prior/posterior 계산

Prior:      exp(-β × distance)
Likelihood: SC 관측 빈도
Posterior:  Prior × Likelihood (정규화)

Usage:
    python eb_allocation.py                # 기본
    python eb_allocation.py --alpha 1.0    # prior 강도 조절
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'data')
EB_DIR = os.path.join(DATA_DIR, 'eb')


def compute_prior(distances, beta):
    """거리 감쇠 기반 prior 확률"""
    log_prior = -beta * distances
    log_prior -= np.logaddexp.reduce(log_prior)
    return np.exp(log_prior)


def compute_posterior(prior_probs, sc_counts, alpha=1.0):
    """Prior × Likelihood → Posterior

    Args:
        prior_probs: 정규화된 prior 확률
        sc_counts: SC 관측 건수
        alpha: prior 강도 (클수록 prior 비중 증가)
    """
    total = sc_counts.sum()
    if total == 0:
        # SC 데이터 없음 → posterior = prior (GTX 신설역 등)
        return prior_probs

    likelihood = sc_counts / total
    posterior = alpha * prior_probs + total * likelihood
    posterior = posterior / posterior.sum()
    return posterior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Prior 강도 (default: 1.0)')
    args = parser.parse_args()

    # 1. β 로드
    print("[1/3] β 로드...")
    beta_path = os.path.join(EB_DIR, 'eb_beta.json')
    if not os.path.exists(beta_path):
        print(f"  ERROR: {beta_path} 없음. estimate_beta.py를 먼저 실행하세요.")
        sys.exit(1)

    with open(beta_path) as f:
        beta_info = json.load(f)
    beta = beta_info['beta']
    resolution = beta_info['resolution']
    print(f"  β = {beta:.6f}, resolution = {resolution}")

    # 2. 병합 데이터 로드
    print("[2/3] H3-정류장-SC 데이터 로드...")
    merged_path = os.path.join(EB_DIR, 'h3_stop_sc_merged.csv')
    if not os.path.exists(merged_path):
        print(f"  ERROR: {merged_path} 없음. estimate_beta.py를 먼저 실행하세요.")
        sys.exit(1)

    merged = pd.read_csv(merged_path)

    # SC 없는 셀도 포함하기 위해 전체 거리 데이터 로드
    dist_path = os.path.join(EB_DIR, 'h3_stop_distance.csv')
    all_dist = pd.read_csv(dist_path)
    all_dist = all_dist[all_dist['h3_resolution'] == resolution]

    # SC counts 병합
    sc_cols = merged[['h3_cell', 'stop_id', 'sc_count']].copy()
    sc_cols['stop_id'] = sc_cols['stop_id'].astype(str)
    all_dist['stop_id'] = all_dist['stop_id'].astype(str)

    full = all_dist.merge(sc_cols[['h3_cell', 'stop_id', 'sc_count']],
                          on=['h3_cell', 'stop_id'], how='left')
    full['sc_count'] = full['sc_count'].fillna(0).astype(int)

    # 3. Prior / Posterior 계산
    print("[3/3] Prior / Posterior 계산...")
    results = []
    for cell, group in full.groupby('h3_cell'):
        distances = group['distance_m'].values
        sc_counts = group['sc_count'].values

        prior = compute_prior(distances, beta)
        posterior = compute_posterior(prior, sc_counts, alpha=args.alpha)

        for i, (_, row) in enumerate(group.iterrows()):
            results.append({
                'h3_cell': cell,
                'stop_id': row['stop_id'],
                'distance_m': row['distance_m'],
                'sc_count': int(sc_counts[i]),
                'prior': round(prior[i], 6),
                'posterior': round(posterior[i], 6),
            })

    result_df = pd.DataFrame(results)

    # 통계
    n_cells = result_df['h3_cell'].nunique()
    n_no_sc = result_df.groupby('h3_cell')['sc_count'].sum()
    n_prior_only = (n_no_sc == 0).sum()

    print(f"  총 {n_cells} 셀")
    print(f"  SC 있는 셀: {n_cells - n_prior_only}")
    print(f"  SC 없는 셀 (prior only): {n_prior_only}")

    # 저장
    prior_path = os.path.join(EB_DIR, 'eb_prior.csv')
    posterior_path = os.path.join(EB_DIR, 'eb_posterior.csv')

    result_df[['h3_cell', 'stop_id', 'distance_m', 'prior']].to_csv(prior_path, index=False)
    result_df.to_csv(posterior_path, index=False)
    print(f"  저장: {prior_path}")
    print(f"  저장: {posterior_path}")

    print("\nPhase 2-2 (EB allocation) 완료.")


if __name__ == '__main__':
    main()
