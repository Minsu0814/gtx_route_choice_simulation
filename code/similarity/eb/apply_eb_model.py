# -*- coding: utf-8 -*-
"""
Phase 5: EB + 경로선택모형 적용 파이프라인

H3→H3 수요를 입력받아:
1. EB posterior로 정류장별 인원 배분
2. 각 정류장 OD에 경로선택모형 적용
3. 경로별 최종 수요 산출

Usage:
    python apply_eb_model.py --demand demand.csv
    python apply_eb_model.py --o-h3 882696a5a3fffff --d-h3 882696b59bfffff --demand-count 1000
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
TRAINING_DIR = os.path.join(DATA_DIR, 'training_set')


def load_mnl_coefficients():
    """MNL 계수 로드"""
    coef_path = os.path.join(TRAINING_DIR, 'mnl_coefficients.json')
    with open(coef_path) as f:
        return json.load(f)


def load_eb_posterior():
    """EB posterior 로드"""
    path = os.path.join(EB_DIR, 'eb_posterior.csv')
    return pd.read_csv(path)


def load_h3_choice_data():
    """H3 재집계 학습 데이터 로드"""
    path = os.path.join(EB_DIR, 'h3_choice_prob.parquet')
    return pd.read_parquet(path)


def mnl_predict(features, coefficients):
    """MNL 선택확률 예측

    Args:
        features: DataFrame, 각 행 = 대안, 컬럼 = 피처
        coefficients: dict, {feature_name: beta}

    Returns:
        np.array, 각 대안의 선택확률
    """
    utility = np.zeros(len(features))
    for feat, beta in coefficients.items():
        if feat in features.columns:
            utility += beta * features[feat].values

    # softmax
    utility -= utility.max()
    exp_u = np.exp(utility)
    return exp_u / exp_u.sum()


def allocate_demand(o_h3, d_h3, demand_count, posterior_df, choice_data, coefficients):
    """단일 H3 OD에 대해 EB 배분 + 경로선택 적용

    Args:
        o_h3: 출발 H3 셀
        d_h3: 도착 H3 셀
        demand_count: 총 수요 (명)
        posterior_df: EB posterior
        choice_data: H3 재집계 학습 데이터
        coefficients: MNL 계수

    Returns:
        DataFrame, 경로별 배분 결과
    """
    h3_od = f"{o_h3}_{d_h3}"

    # 1. 해당 H3 OD의 경로 데이터 가져오기
    routes = choice_data[choice_data['h3_od'] == h3_od]

    if len(routes) == 0:
        return pd.DataFrame()

    # 2. 출발 셀의 EB posterior (정류장 배분)
    o_posterior = posterior_df[posterior_df['h3_cell'] == o_h3]

    if len(o_posterior) == 0:
        # EB 데이터 없으면 균등 배분
        o_stops = routes['o_stop'].unique()
        stop_shares = {s: 1.0 / len(o_stops) for s in o_stops}
    else:
        total_post = o_posterior['posterior'].sum()
        stop_shares = dict(zip(
            o_posterior['stop_id'].astype(str),
            o_posterior['posterior'] / total_post
        ))

    # 3. 정류장별 배분 + 경로선택
    results = []
    for o_stop, share in stop_shares.items():
        stop_demand = demand_count * share

        # 이 정류장에서 출발하는 경로들
        stop_routes = routes[routes['o_stop'] == o_stop]
        if len(stop_routes) == 0:
            continue

        # MNL 예측 (피처 기반)
        feature_cols = [c for c in coefficients.keys() if c in stop_routes.columns]
        if feature_cols:
            probs = mnl_predict(stop_routes[feature_cols], coefficients)
        else:
            # 계수와 매칭되는 피처 없으면 관측 확률 사용
            probs = stop_routes['stop_od_choice_prob'].values
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(len(stop_routes)) / len(stop_routes)

        for i, (_, route) in enumerate(stop_routes.iterrows()):
            results.append({
                'h3_od': h3_od,
                'o_stop': o_stop,
                'd_stop': route.get('d_stop', ''),
                'od_pair': route.get('od_pair', ''),
                'alt_idx': route.get('alt_idx', i),
                'transport_category': route.get('transport_category', ''),
                'main_route': route.get('main_route', ''),
                'eb_share': round(share, 4),
                'route_prob': round(probs[i], 4),
                'demand': round(stop_demand * probs[i], 1),
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--o-h3', type=str, help='출발 H3 셀')
    parser.add_argument('--d-h3', type=str, help='도착 H3 셀')
    parser.add_argument('--demand-count', type=int, default=1000, help='수요 (명)')
    parser.add_argument('--demand', type=str, help='수요 CSV (o_h3, d_h3, demand 컬럼)')
    args = parser.parse_args()

    # 데이터 로드
    print("데이터 로드...")
    posterior_df = load_eb_posterior()
    choice_data = load_h3_choice_data()
    coefficients = load_mnl_coefficients()

    # 단일 OD 모드
    if args.o_h3 and args.d_h3:
        print(f"\n{args.o_h3} → {args.d_h3}: {args.demand_count}명")
        result = allocate_demand(
            args.o_h3, args.d_h3, args.demand_count,
            posterior_df, choice_data, coefficients
        )
        if len(result) > 0:
            print(result.to_string(index=False))
            print(f"\n총 배분: {result['demand'].sum():.0f}명")
        else:
            print("해당 H3 OD에 대한 경로 데이터 없음")
        return

    # CSV 수요 모드
    if args.demand:
        print(f"수요 CSV 로드: {args.demand}")
        demand_df = pd.read_csv(args.demand)
        all_results = []
        for _, row in demand_df.iterrows():
            result = allocate_demand(
                row['o_h3'], row['d_h3'], row['demand'],
                posterior_df, choice_data, coefficients
            )
            all_results.append(result)

        final = pd.concat(all_results, ignore_index=True)
        out_path = os.path.join(EB_DIR, 'eb_allocation_result.csv')
        final.to_csv(out_path, index=False)
        print(f"저장: {out_path}")
        print(f"총 {len(final)}개 경로, {final['demand'].sum():.0f}명 배분")
        return

    print("--o-h3/--d-h3 또는 --demand를 지정하세요.")


if __name__ == '__main__':
    main()
