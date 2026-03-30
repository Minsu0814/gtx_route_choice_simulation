# -*- coding: utf-8 -*-
"""
Leave-One-Out 검증: EB prior의 walk 파라미터 신빙성 검증

각 H3 셀에서 정류장 하나를 "없다고 가정"하고,
EB prior(도보거리)만으로 배분을 예측한 뒤,
실제 SC 관측 비율과 비교한다.

이것이 맞으면 → "도보 거리 기반 배분이 실제 행태를 설명한다"는 근거.
이것이 맞으면 → GTX 신설역에도 prior만으로 배분 가능하다는 근거.

Usage:
    python validate_eb_loo.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'data')
EB_DIR = os.path.join(DATA_DIR, 'eb')


def main():
    print("=" * 60)
    print("Leave-One-Out 검증: EB prior walk 신빙성")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1/4] 데이터 로드...")
    beta_path = os.path.join(EB_DIR, 'eb_beta.json')
    with open(beta_path) as f:
        beta_info = json.load(f)
    beta = beta_info['beta']
    print(f"  β = {beta:.6f}")

    merged_path = os.path.join(EB_DIR, 'h3_stop_sc_merged.csv')
    df = pd.read_csv(merged_path)
    print(f"  {df['h3_cell'].nunique()} 셀, {len(df)} 정류장-셀 쌍")

    # 2. LOO 검증: 셀 내 정류장 3개 이상인 셀만 (1개 빼도 2개 남아야 비교 가능)
    print("\n[2/4] LOO 대상 셀 선정...")
    cell_sizes = df.groupby('h3_cell').size()
    loo_cells = cell_sizes[cell_sizes >= 3].index
    print(f"  정류장 3개 이상 셀: {len(loo_cells)} ({len(loo_cells)/df['h3_cell'].nunique()*100:.1f}%)")

    # SC 데이터가 충분한 셀만 (최소 50건)
    cell_sc = df.groupby('h3_cell')['sc_count'].sum()
    loo_cells = loo_cells.intersection(cell_sc[cell_sc >= 50].index)
    print(f"  SC 50건 이상 셀: {len(loo_cells)}")

    # 3. LOO 실행
    print("\n[3/4] LOO 검증 실행...")
    results = []

    for cell in loo_cells:
        group = df[df['h3_cell'] == cell].copy()
        n_stops = len(group)
        total_sc = group['sc_count'].sum()

        # 실제 SC 비율
        actual_share = group['sc_count'].values / total_sc

        for leave_out_idx in range(n_stops):
            # 제외할 정류장
            left_out = group.iloc[leave_out_idx]
            left_out_stop = left_out['stop_id']
            left_out_actual = actual_share[leave_out_idx]

            if left_out_actual == 0:
                continue  # SC 0건인 정류장은 건너뜀

            # 나머지 정류장으로 prior 계산 (제외된 정류장 포함)
            distances = group['distance_m'].values

            # prior: 모든 정류장에 대해 계산 (제외된 것 포함)
            prior = np.exp(-beta * distances)
            prior = prior / prior.sum()
            predicted_share = prior[leave_out_idx]

            # 나머지 정류장만으로 prior 계산 후 제외된 정류장 예측
            # → "이 정류장이 없었다면" prior만으로 얼마를 배분했을까
            remaining_mask = np.ones(n_stops, dtype=bool)
            remaining_mask[leave_out_idx] = False

            remaining_prior = np.exp(-beta * distances)
            remaining_prior_norm = remaining_prior / remaining_prior.sum()
            predicted_share_full = remaining_prior_norm[leave_out_idx]

            results.append({
                'h3_cell': cell,
                'stop_id': left_out_stop,
                'distance_m': left_out['distance_m'],
                'sc_count': int(left_out['sc_count']),
                'total_sc': int(total_sc),
                'actual_share': round(left_out_actual, 6),
                'prior_share': round(predicted_share, 6),
                'abs_error': round(abs(predicted_share - left_out_actual), 6),
                'n_stops_in_cell': n_stops,
            })

    result_df = pd.DataFrame(results)
    print(f"  LOO 케이스: {len(result_df):,}")

    # 4. 검증 결과 분석
    print("\n[4/4] 검증 결과 분석...")

    mae = result_df['abs_error'].mean()
    rmse = np.sqrt((result_df['abs_error'] ** 2).mean())
    corr = result_df[['actual_share', 'prior_share']].corr().iloc[0, 1]

    print(f"\n=== 전체 검증 결과 ===")
    print(f"  LOO 케이스: {len(result_df):,}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  상관계수 (actual vs prior): {corr:.4f}")

    # 오차 분포
    print(f"\n  절대 오차 분포:")
    print(f"    25%ile: {result_df['abs_error'].quantile(0.25):.4f}")
    print(f"    50%ile: {result_df['abs_error'].quantile(0.50):.4f}")
    print(f"    75%ile: {result_df['abs_error'].quantile(0.75):.4f}")
    print(f"    90%ile: {result_df['abs_error'].quantile(0.90):.4f}")

    # 정류장 수별 분석
    print(f"\n  셀 내 정류장 수별:")
    for n in sorted(result_df['n_stops_in_cell'].unique()):
        sub = result_df[result_df['n_stops_in_cell'] == n]
        if len(sub) < 10:
            continue
        print(f"    {n}개: MAE={sub['abs_error'].mean():.4f}, "
              f"corr={sub[['actual_share','prior_share']].corr().iloc[0,1]:.4f}, "
              f"n={len(sub)}")

    # SC 건수별 분석 (SC 많은 셀일수록 actual이 안정적)
    print(f"\n  SC 건수별:")
    for lo, hi, label in [(50, 100, '50-100'), (100, 500, '100-500'),
                           (500, 2000, '500-2K'), (2000, 999999, '2K+')]:
        sub = result_df[(result_df['total_sc'] >= lo) & (result_df['total_sc'] < hi)]
        if len(sub) < 10:
            continue
        c = sub[['actual_share', 'prior_share']].corr().iloc[0, 1]
        print(f"    {label}: MAE={sub['abs_error'].mean():.4f}, corr={c:.4f}, n={len(sub)}")

    # 거리별 분석
    print(f"\n  정류장 거리별:")
    for lo, hi, label in [(0, 200, '0-200m'), (200, 400, '200-400m'),
                           (400, 600, '400-600m'), (600, 99999, '600m+')]:
        sub = result_df[(result_df['distance_m'] >= lo) & (result_df['distance_m'] < hi)]
        if len(sub) < 10:
            continue
        print(f"    {label}: MAE={sub['abs_error'].mean():.4f}, "
              f"actual={sub['actual_share'].mean():.4f}, "
              f"prior={sub['prior_share'].mean():.4f}, n={len(sub)}")

    # 저장
    out_path = os.path.join(EB_DIR, 'loo_validation.csv')
    result_df.to_csv(out_path, index=False)
    print(f"\n  저장: {out_path}")

    # 요약 저장
    summary = {
        'n_cases': len(result_df),
        'n_cells': int(result_df['h3_cell'].nunique()),
        'mae': float(mae),
        'rmse': float(rmse),
        'correlation': float(corr),
        'beta': beta,
    }
    summary_path = os.path.join(EB_DIR, 'loo_validation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  저장: {summary_path}")

    print("\nLOO 검증 완료.")


if __name__ == '__main__':
    main()
