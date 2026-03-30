# -*- coding: utf-8 -*-
"""
Phase 2-2: EB prior/posterior 계산

M2 다변량 prior: exp(-β₁×distance + β₂×ln(1+n_routes))
Likelihood: SC 관측 빈도
Posterior: Prior × Likelihood (정규화)

Usage:
    python eb_allocation.py                # 기본
    python eb_allocation.py --alpha 1.0    # prior 강도 조절
    python eb_allocation.py --distance-only # 거리만 사용 (M1)
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


def compute_prior_m2(distances, ln_n_routes, beta_dist, beta_routes):
    """M2 다변량 prior: 거리 + 노선 수"""
    utility = beta_dist * distances + beta_routes * ln_n_routes
    utility -= np.logaddexp.reduce(utility)
    return np.exp(utility)


def compute_prior_m1(distances, beta):
    """M1 거리만 prior (기존)"""
    log_prior = -beta * distances
    log_prior -= np.logaddexp.reduce(log_prior)
    return np.exp(log_prior)


def compute_posterior(prior_probs, sc_counts, alpha=1.0):
    """Prior × Likelihood → Posterior"""
    total = sc_counts.sum()
    if total == 0:
        return prior_probs

    likelihood = sc_counts / total
    posterior = alpha * prior_probs + total * likelihood
    posterior = posterior / posterior.sum()
    return posterior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--distance-only', action='store_true', help='M1 거리만 사용')
    args = parser.parse_args()

    # 1. β 로드
    print("[1/4] β 로드...")
    multi_path = os.path.join(EB_DIR, 'eb_multi_beta.json')
    single_path = os.path.join(EB_DIR, 'eb_beta.json')

    if not args.distance_only and os.path.exists(multi_path):
        with open(multi_path) as f:
            multi = json.load(f)
        # M2 사용
        m2 = multi.get('M2: distance + n_routes', {})
        beta_dist = m2['params'].get('distance_m', 0)
        beta_routes = m2['params'].get('ln_n_routes', 0)
        use_m2 = True
        print(f"  M2 prior: β_distance={beta_dist:.6f}, β_ln_n_routes={beta_routes:.6f}")
    else:
        with open(single_path) as f:
            beta_info = json.load(f)
        beta_dist = -beta_info['beta']
        beta_routes = 0
        use_m2 = False
        print(f"  M1 prior: β_distance={beta_dist:.6f}")

    with open(single_path) as f:
        resolution = json.load(f)['resolution']

    # 2. 거리 데이터 + SC + GTFS 노선 수 로드
    print("[2/4] 데이터 로드...")
    dist_path = os.path.join(EB_DIR, 'h3_stop_distance.csv')
    all_dist = pd.read_csv(dist_path)
    all_dist = all_dist[all_dist['h3_resolution'] == resolution]
    all_dist['stop_id'] = all_dist['stop_id'].astype(str)

    # SC counts
    merged_path = os.path.join(EB_DIR, 'h3_stop_sc_merged.csv')
    if os.path.exists(merged_path):
        sc_merged = pd.read_csv(merged_path)
        sc_cols = sc_merged[['h3_cell', 'stop_id', 'sc_count']].copy()
        sc_cols['stop_id'] = sc_cols['stop_id'].astype(str)
    else:
        sc_cols = pd.DataFrame(columns=['h3_cell', 'stop_id', 'sc_count'])

    full = all_dist.merge(sc_cols, on=['h3_cell', 'stop_id'], how='left')
    full['sc_count'] = full['sc_count'].fillna(0).astype(int)

    # GTFS 노선 수
    if use_m2:
        match_path = os.path.join(EB_DIR, 'otp_gtfs_stop_matching.csv')
        if os.path.exists(match_path):
            gtfs_match = pd.read_csv(match_path)
            gtfs_match['otp_stop_id'] = gtfs_match['otp_stop_id'].astype(str)
            full = full.merge(
                gtfs_match[['otp_stop_id', 'n_routes']],
                left_on='stop_id', right_on='otp_stop_id', how='left'
            )
            full['n_routes'] = full['n_routes'].fillna(1)
            if 'otp_stop_id' in full.columns:
                full.drop(columns='otp_stop_id', inplace=True)
        else:
            full['n_routes'] = 1

        full['ln_n_routes'] = np.log1p(full['n_routes'])

    # 3. Prior / Posterior 계산
    print("[3/4] Prior / Posterior 계산...")
    results = []
    for cell, group in full.groupby('h3_cell'):
        distances = group['distance_m'].values
        sc_counts_arr = group['sc_count'].values

        if use_m2:
            ln_routes = group['ln_n_routes'].values
            prior = compute_prior_m2(distances, ln_routes, beta_dist, beta_routes)
        else:
            prior = compute_prior_m1(distances, -beta_dist)

        posterior = compute_posterior(prior, sc_counts_arr, alpha=args.alpha)

        for i, (_, row) in enumerate(group.iterrows()):
            result_row = {
                'h3_cell': cell,
                'stop_id': row['stop_id'],
                'distance_m': row['distance_m'],
                'sc_count': int(sc_counts_arr[i]),
                'prior': round(prior[i], 6),
                'posterior': round(posterior[i], 6),
            }
            if use_m2:
                result_row['n_routes'] = int(row.get('n_routes', 1))
            results.append(result_row)

    result_df = pd.DataFrame(results)

    # 통계
    n_cells = result_df['h3_cell'].nunique()
    n_no_sc = result_df.groupby('h3_cell')['sc_count'].sum()
    n_prior_only = (n_no_sc == 0).sum()

    print(f"  총 {n_cells} 셀")
    print(f"  SC 있는 셀: {n_cells - n_prior_only}")
    print(f"  SC 없는 셀 (prior only): {n_prior_only}")
    print(f"  prior 모형: {'M2 (distance + n_routes)' if use_m2 else 'M1 (distance only)'}")

    # 4. 저장
    print("[4/4] 저장...")
    prior_path = os.path.join(EB_DIR, 'eb_prior.csv')
    posterior_path = os.path.join(EB_DIR, 'eb_posterior.csv')

    result_df[['h3_cell', 'stop_id', 'distance_m', 'prior']].to_csv(prior_path, index=False)
    result_df.to_csv(posterior_path, index=False)
    print(f"  저장: {prior_path}")
    print(f"  저장: {posterior_path}")

    print("\nPhase 2-2 (EB allocation) 완료.")


if __name__ == '__main__':
    main()
