# -*- coding: utf-8 -*-
"""
다변량 EB prior 추정: 거리 + 노선 수 + 배차(trip 수)

Prior: P(stop k | H3 cell i) ∝ exp(-β₁×distance + β₂×ln(n_routes) + β₃×ln(n_trips))

기존 거리 단독 prior와 성능 비교 (LOO 검증 포함).

Usage:
    python estimate_beta_multi.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'data')
EB_DIR = os.path.join(DATA_DIR, 'eb')
GTFS_DIR = os.path.join(DATA_DIR, 'gtfs', 'a1')
TCN_DIR = os.path.join(DATA_DIR, 'tcn')


def load_stop_features():
    """GTFS에서 정류장별 노선 수, trip 수 추출"""
    stop_times = pd.read_csv(os.path.join(GTFS_DIR, 'stop_times.txt'), usecols=['trip_id', 'stop_id'])
    trips = pd.read_csv(os.path.join(GTFS_DIR, 'trips.txt'), usecols=['trip_id', 'route_id'])

    st = stop_times.merge(trips, on='trip_id')

    # 정류장별 노선 수
    n_routes = st.groupby('stop_id')['route_id'].nunique().reset_index()
    n_routes.columns = ['gtfs_stop_id', 'n_routes']

    # 정류장별 trip 수 (배차 빈도 proxy)
    n_trips = st.groupby('stop_id')['trip_id'].nunique().reset_index()
    n_trips.columns = ['gtfs_stop_id', 'n_trips']

    features = n_routes.merge(n_trips, on='gtfs_stop_id')
    print(f"  GTFS 정류장 피처: {len(features)}개")
    print(f"  n_routes: mean={features['n_routes'].mean():.1f}, median={features['n_routes'].median():.0f}")
    print(f"  n_trips: mean={features['n_trips'].mean():.1f}, median={features['n_trips'].median():.0f}")
    return features


def load_sc_stop_counts():
    """SC 정류장별 이용 건수"""
    all_counts = []
    for dname in sorted(os.listdir(TCN_DIR)):
        fpath = os.path.join(TCN_DIR, dname, f'TCN_{dname}_route.parquet')
        if not os.path.exists(fpath):
            continue
        tcn = pd.read_parquet(fpath, columns=['승차정류장ID'])
        counts = tcn['승차정류장ID'].value_counts().reset_index()
        counts.columns = ['stop_id', 'sc_count']
        all_counts.append(counts)
    return pd.concat(all_counts).groupby('stop_id')['sc_count'].sum().reset_index()


def build_merged_data(resolution=8):
    """H3-정류장 거리 + GTFS 피처 + SC 건수 병합"""
    # 거리
    dist = pd.read_csv(os.path.join(EB_DIR, 'h3_stop_distance.csv'))
    dist = dist[dist['h3_resolution'] == resolution]
    dist['stop_id'] = dist['stop_id'].astype(str)

    # GTFS 피처
    gtfs_feat = load_stop_features()
    gtfs_feat['gtfs_stop_id'] = gtfs_feat['gtfs_stop_id'].astype(str)

    # SC 건수
    sc = load_sc_stop_counts()
    sc['stop_id'] = sc['stop_id'].astype(str)

    # GTFS stop_id 매핑 시도 (OTP stop_id와 GTFS stop_id가 다를 수 있음)
    # 먼저 직접 매칭
    merged = dist.merge(sc, on='stop_id', how='left')
    merged['sc_count'] = merged['sc_count'].fillna(0).astype(int)

    # 좌표 기반 GTFS 매칭 결과 로드
    match_path = os.path.join(EB_DIR, 'otp_gtfs_stop_matching.csv')
    if os.path.exists(match_path):
        gtfs_match = pd.read_csv(match_path)
        gtfs_match['otp_stop_id'] = gtfs_match['otp_stop_id'].astype(str)
        merged = merged.merge(
            gtfs_match[['otp_stop_id', 'n_routes', 'n_trips']],
            left_on='stop_id', right_on='otp_stop_id', how='left'
        )
        merged['n_routes'] = merged['n_routes'].fillna(1)
        merged['n_trips'] = merged['n_trips'].fillna(1)
        if 'otp_stop_id' in merged.columns:
            merged.drop(columns='otp_stop_id', inplace=True)
    else:
        print("  WARNING: otp_gtfs_stop_matching.csv 없음, GTFS 피처 없이 진행")
        merged['n_routes'] = 1
        merged['n_trips'] = 1

    # log 변환
    merged['ln_n_routes'] = np.log1p(merged['n_routes'])
    merged['ln_n_trips'] = np.log1p(merged['n_trips'])

    # SC 있고 정류장 2개 이상인 셀만
    cell_sc = merged.groupby('h3_cell')['sc_count'].sum()
    valid = cell_sc[cell_sc > 0].index
    merged = merged[merged['h3_cell'].isin(valid)]

    cell_sizes = merged.groupby('h3_cell').size()
    multi = cell_sizes[cell_sizes >= 2].index
    merged = merged[merged['h3_cell'].isin(multi)]

    print(f"  병합 데이터: {merged['h3_cell'].nunique()} 셀, {len(merged)} 행")
    print(f"  GTFS 매칭률: {(merged['n_routes'] > 1).mean()*100:.1f}%")
    return merged


def estimate_multi_beta(merged, feature_cols, feature_signs):
    """다변량 β 추정 (MLE)

    Args:
        merged: 병합 데이터
        feature_cols: 피처 컬럼명 리스트
        feature_signs: 각 피처의 부호 (-1: 음수 제약, +1: 양수 제약, 0: 무제약)
    """
    def neg_log_likelihood(params):
        total_nll = 0
        for cell, group in merged.groupby('h3_cell'):
            counts = group['sc_count'].values
            if counts.sum() == 0:
                continue

            # 효용 = Σ βᵢ × xᵢ
            utility = np.zeros(len(group))
            for p, col in zip(params, feature_cols):
                utility += p * group[col].values

            log_probs = utility - np.logaddexp.reduce(utility)
            total_nll -= np.sum(counts * log_probs)
        return total_nll

    # 초기값 & 범위
    x0 = []
    bounds = []
    for sign in feature_signs:
        if sign < 0:
            x0.append(-0.001)
            bounds.append((None, 0))
        elif sign > 0:
            x0.append(0.5)
            bounds.append((0, None))
        else:
            x0.append(0.0)
            bounds.append((None, None))

    result = minimize(neg_log_likelihood, x0, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})

    return result.x, result.fun


def loo_validate(merged, params, feature_cols, min_stops=3, min_sc=50):
    """LOO 검증"""
    cell_sizes = merged.groupby('h3_cell').size()
    cell_sc = merged.groupby('h3_cell')['sc_count'].sum()
    valid_cells = cell_sizes[(cell_sizes >= min_stops)].index
    valid_cells = valid_cells.intersection(cell_sc[cell_sc >= min_sc].index)

    results = []
    for cell in valid_cells:
        group = merged[merged['h3_cell'] == cell]
        total_sc = group['sc_count'].sum()
        actual_share = group['sc_count'].values / total_sc

        # prior 계산
        utility = np.zeros(len(group))
        for p, col in zip(params, feature_cols):
            utility += p * group[col].values
        prior = np.exp(utility - np.logaddexp.reduce(utility))

        for i in range(len(group)):
            if actual_share[i] == 0:
                continue
            results.append({
                'actual_share': actual_share[i],
                'prior_share': prior[i],
                'abs_error': abs(prior[i] - actual_share[i]),
                'n_stops': len(group),
            })

    rdf = pd.DataFrame(results)
    mae = rdf['abs_error'].mean()
    corr = rdf[['actual_share', 'prior_share']].corr().iloc[0, 1]

    # 셀 내 상관
    within_corrs = []
    for cell in valid_cells:
        group = merged[merged['h3_cell'] == cell]
        total_sc = group['sc_count'].sum()
        actual = group['sc_count'].values / total_sc

        utility = np.zeros(len(group))
        for p, col in zip(params, feature_cols):
            utility += p * group[col].values
        prior = np.exp(utility - np.logaddexp.reduce(utility))

        if len(actual) >= 3 and np.std(actual) > 0 and np.std(prior) > 0:
            c = np.corrcoef(actual, prior)[0, 1]
            if not np.isnan(c):
                within_corrs.append(c)

    within_corr_mean = np.mean(within_corrs) if within_corrs else 0

    return mae, corr, within_corr_mean, len(rdf)


def main():
    print("=" * 60)
    print("다변량 EB prior 추정 + LOO 검증")
    print("=" * 60)

    # 1. 데이터 준비
    print("\n[1/3] 데이터 준비...")
    merged = build_merged_data()

    # 2. 모형별 추정
    print("\n[2/3] 모형 추정...")

    specs = [
        ('M1: distance only',
         ['distance_m'], [-1]),
        ('M2: distance + n_routes',
         ['distance_m', 'ln_n_routes'], [-1, +1]),
        ('M3: distance + n_trips',
         ['distance_m', 'ln_n_trips'], [-1, +1]),
        ('M4: distance + n_routes + n_trips',
         ['distance_m', 'ln_n_routes', 'ln_n_trips'], [-1, +1, +1]),
    ]

    all_results = {}
    for name, feat_cols, signs in specs:
        print(f"\n  --- {name} ---")
        params, nll = estimate_multi_beta(merged, feat_cols, signs)

        for col, p in zip(feat_cols, params):
            print(f"    β_{col} = {p:.6f}")
        print(f"    NLL = {nll:,.0f}")

        mae, corr, within_corr, n = loo_validate(merged, params, feat_cols)
        print(f"    LOO: MAE={mae:.4f}, corr={corr:.4f}, within_corr={within_corr:.4f}, n={n:,}")

        all_results[name] = {
            'features': feat_cols,
            'params': {col: float(p) for col, p in zip(feat_cols, params)},
            'nll': float(nll),
            'loo_mae': float(mae),
            'loo_corr': float(corr),
            'loo_within_corr': float(within_corr),
            'loo_n': n,
        }

    # 3. 비교 테이블
    print(f"\n{'=' * 60}")
    print("모형 비교")
    print(f"{'=' * 60}")
    print(f"{'Model':<35} {'NLL':>12} {'MAE':>8} {'Corr':>8} {'Within':>8}")
    print('-' * 73)
    for name, r in all_results.items():
        print(f"{name:<35} {r['nll']:>12,.0f} {r['loo_mae']:>8.4f} {r['loo_corr']:>8.4f} {r['loo_within_corr']:>8.4f}")

    # 저장
    out_path = os.path.join(EB_DIR, 'eb_multi_beta.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n저장: {out_path}")

    print("\n완료.")


if __name__ == '__main__':
    main()
