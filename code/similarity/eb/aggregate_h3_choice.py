# -*- coding: utf-8 -*-
"""
Phase 3: H3 단위 choice_prob 재집계

기존 정류장 OD 학습 데이터에 H3 키를 부여하고,
H3 OD 단위로 choice_prob을 재집계한다.

각 경로는 원래 정류장 OD 기준 피처를 그대로 유지하며,
choice_prob만 H3 OD 전체 기준으로 재계산한다.

Usage:
    python aggregate_h3_choice.py
"""

import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', 'data')
EB_DIR = os.path.join(DATA_DIR, 'eb')
TRAINING_PATH = os.path.join(DATA_DIR, 'training_set', 'route_choice_training.parquet')


def main():
    # 1. 학습 데이터 로드
    print("[1/4] 학습 데이터 로드...")
    df = pd.read_parquet(TRAINING_PATH)
    print(f"  {len(df):,}행, {df['od_pair'].nunique():,} OD")

    # 2. 정류장-H3 매핑 로드
    print("[2/4] H3 매핑 로드...")
    mapping_path = os.path.join(EB_DIR, 'stop_h3_mapping.csv')
    if not os.path.exists(mapping_path):
        print(f"  ERROR: {mapping_path} 없음. build_h3_mapping.py를 먼저 실행하세요.")
        sys.exit(1)

    mapping = pd.read_csv(mapping_path)
    # stop_id → h3_cell 딕셔너리
    stop_to_h3 = dict(zip(mapping['stop_id'].astype(str), mapping['h3_res8']))

    # 3. od_pair에서 승/하차 정류장 추출 → H3 매핑
    print("[3/4] od_pair → H3 OD 매핑...")
    df['o_stop'] = df['od_pair'].str.split('_').str[0]
    df['d_stop'] = df['od_pair'].str.split('_').str[1]
    df['o_h3'] = df['o_stop'].map(stop_to_h3)
    df['d_h3'] = df['d_stop'].map(stop_to_h3)

    # H3 매핑 실패 제거
    before = len(df)
    df = df.dropna(subset=['o_h3', 'd_h3'])
    print(f"  H3 매핑 성공: {len(df):,}행 ({len(df)/before*100:.1f}%)")

    df['h3_od'] = df['o_h3'] + '_' + df['d_h3']

    # 4. H3 OD 단위 choice_prob 재집계
    print("[4/4] H3 OD 단위 choice_prob 재집계...")

    # 각 행의 SC 매칭 건수 (n_matched)를 기반으로 재집계
    # n_matched = 해당 경로에 매칭된 SC 통행 수
    h3_groups = df.groupby('h3_od')

    # H3 OD별 총 매칭 건수
    h3_totals = h3_groups['n_matched'].transform('sum')

    # H3 기준 choice_prob 재계산
    df['h3_choice_prob'] = df['n_matched'] / h3_totals.replace(0, 1)

    # 기존 choice_prob 보존
    df = df.rename(columns={'choice_prob': 'stop_od_choice_prob'})
    df = df.rename(columns={'h3_choice_prob': 'choice_prob'})

    # H3 OD별 통계
    n_h3_od = df['h3_od'].nunique()
    n_stop_od = df['od_pair'].nunique()
    avg_routes = df.groupby('h3_od').size().mean()

    print(f"  정류장 OD: {n_stop_od:,} → H3 OD: {n_h3_od:,}")
    print(f"  H3 OD당 평균 경로 수: {avg_routes:.1f}")

    # 저장
    out_path = os.path.join(EB_DIR, 'h3_choice_prob.parquet')
    df.to_parquet(out_path, index=False)
    print(f"  저장: {out_path}")

    # 요약 통계
    print(f"\n=== 요약 ===")
    print(f"  H3 OD 수: {n_h3_od:,}")
    print(f"  정류장 OD 수: {n_stop_od:,}")
    print(f"  평균 경로/H3 OD: {avg_routes:.1f}")
    print(f"  choice_prob 분포:")
    print(f"    mean: {df['choice_prob'].mean():.4f}")
    print(f"    std:  {df['choice_prob'].std():.4f}")
    print(f"    max:  {df['choice_prob'].max():.4f}")

    print("\nPhase 3 완료.")


if __name__ == '__main__':
    main()
