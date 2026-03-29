# -*- coding: utf-8 -*-
"""
Phase 2-1: EB β 파라미터 추정 (MLE)

SC 이용빈도 ~ 도보거리 관계에서 거리 감쇠 파라미터 β를 추정한다.
P(stop k | H3 cell i) ∝ exp(-β × distance(centroid_i, stop_k))

Usage:
    python estimate_beta.py                # 기본 (res 8)
    python estimate_beta.py --res 8        # resolution 지정
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'data')
EB_DIR = os.path.join(DATA_DIR, 'eb')
TCN_DIR = os.path.join(DATA_DIR, 'tcn')


def load_sc_stop_counts():
    """SC 데이터에서 정류장별 이용 건수 집계"""
    all_counts = []
    tcn_dirs = sorted([
        d for d in os.listdir(TCN_DIR)
        if os.path.isdir(os.path.join(TCN_DIR, d)) and d.startswith('20')
    ])

    print(f"  TCN 폴더: {len(tcn_dirs)}일치")
    for dname in tcn_dirs:
        fpath = os.path.join(TCN_DIR, dname, f'TCN_{dname}_route.parquet')
        if not os.path.exists(fpath):
            continue
        tcn = pd.read_parquet(fpath, columns=['승차정류장ID'])
        counts = tcn['승차정류장ID'].value_counts().reset_index()
        counts.columns = ['stop_id', 'sc_count']
        all_counts.append(counts)

    combined = pd.concat(all_counts).groupby('stop_id')['sc_count'].sum().reset_index()
    print(f"  정류장 {len(combined)}개, 총 {combined['sc_count'].sum():,}건")
    return combined


def merge_distance_and_counts(h3_stop_dist, sc_counts, resolution):
    """H3 셀-정류장 거리와 SC 이용 건수 병합"""
    df = h3_stop_dist[h3_stop_dist['h3_resolution'] == resolution].copy()

    # stop_id 타입 통일
    df['stop_id'] = df['stop_id'].astype(str)
    sc_counts['stop_id'] = sc_counts['stop_id'].astype(str)

    merged = df.merge(sc_counts, on='stop_id', how='left')
    merged['sc_count'] = merged['sc_count'].fillna(0).astype(int)

    # SC 데이터가 있는 셀만 사용 (β 추정에 의미 있는 데이터)
    cells_with_data = merged.groupby('h3_cell')['sc_count'].sum()
    valid_cells = cells_with_data[cells_with_data > 0].index
    merged = merged[merged['h3_cell'].isin(valid_cells)]

    # 셀 내 정류장 2개 이상인 셀만 (1개면 추정 의미 없음)
    cell_stop_counts = merged.groupby('h3_cell').size()
    multi_stop_cells = cell_stop_counts[cell_stop_counts >= 2].index
    merged = merged[merged['h3_cell'].isin(multi_stop_cells)]

    print(f"  β 추정용: {merged['h3_cell'].nunique()} 셀, "
          f"{len(merged)} 정류장-셀 쌍, "
          f"SC {merged['sc_count'].sum():,}건")
    return merged


def estimate_beta(merged_df):
    """MLE로 β 추정: 각 셀에서 P(stop) ∝ exp(-β × distance)"""

    def neg_log_likelihood(beta):
        total_nll = 0
        for cell, group in merged_df.groupby('h3_cell'):
            distances = group['distance_m'].values
            counts = group['sc_count'].values
            if counts.sum() == 0:
                continue

            log_probs = -beta * distances
            log_probs -= np.logaddexp.reduce(log_probs)  # log-softmax
            total_nll -= np.sum(counts * log_probs)
        return total_nll

    result = minimize_scalar(
        neg_log_likelihood,
        bounds=(0.0001, 0.05),
        method='bounded',
    )

    beta_hat = result.x
    nll = result.fun
    print(f"  β = {beta_hat:.6f} (NLL = {nll:,.0f})")
    print(f"  해석: 거리 100m 증가 시 이용확률 {(1 - np.exp(-beta_hat * 100)) * 100:.1f}% 감소")
    return beta_hat, nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=8, help='H3 resolution (default: 8)')
    args = parser.parse_args()

    # 1. H3-정류장 거리 로드
    print("[1/3] H3-정류장 거리 로드...")
    dist_path = os.path.join(EB_DIR, 'h3_stop_distance.csv')
    if not os.path.exists(dist_path):
        print(f"  ERROR: {dist_path} 없음. build_h3_mapping.py를 먼저 실행하세요.")
        sys.exit(1)
    h3_stop_dist = pd.read_csv(dist_path)

    # 2. SC 이용 건수 집계
    print("[2/3] SC 이용 건수 집계...")
    sc_counts = load_sc_stop_counts()

    # 3. β 추정
    print(f"[3/3] β 추정 (resolution {args.res})...")
    merged = merge_distance_and_counts(h3_stop_dist, sc_counts, args.res)
    beta_hat, nll = estimate_beta(merged)

    # 저장
    result = {
        'resolution': args.res,
        'beta': beta_hat,
        'neg_log_likelihood': nll,
        'n_cells': int(merged['h3_cell'].nunique()),
        'n_pairs': len(merged),
        'total_sc_count': int(merged['sc_count'].sum()),
    }
    out_path = os.path.join(EB_DIR, 'eb_beta.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  저장: {out_path}")

    # 병합 데이터도 저장 (검증용)
    merged_path = os.path.join(EB_DIR, 'h3_stop_sc_merged.csv')
    merged.to_csv(merged_path, index=False)
    print(f"  저장: {merged_path}")

    print("\nPhase 2-1 (β 추정) 완료.")


if __name__ == '__main__':
    main()
