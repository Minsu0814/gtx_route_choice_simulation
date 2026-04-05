# -*- coding: utf-8 -*-
"""
Step 4: 개인 SC 통행 단위 H3 OD 학습 데이터 구축

기존 route_choice_individual.parquet에:
1. 정류장 → H3 매핑 (o_h3, d_h3, h3_od)
2. 외부 접근/이탈 거리 (eb_access_dist, eb_egress_dist)
3. 로그 변환 도보시간 (ln_access_ext, ln_egress_ext)

을 추가하여 MNL 추정용 데이터를 생성한다.

127M행 처리를 위해 청크 단위로 처리.

Usage:
    python build_h3_individual.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')
EB_DIR = os.path.join(DATA_DIR, 'eb')
TRAINING_DIR = os.path.join(DATA_DIR, 'training_set')

INPUT_PATH = os.path.join(TRAINING_DIR, 'route_choice_individual.parquet')
OUTPUT_PATH = os.path.join(TRAINING_DIR, 'route_choice_individual_h3.parquet')

CHUNK_SIZE = 5_000_000  # 500만 행씩 처리


def build_od_lookup():
    """od_pair별 H3 매핑 + 외부 접근/이탈 거리 룩업 테이블 생성"""

    # 1. 정류장 → H3 매핑
    mapping = pd.read_csv(os.path.join(EB_DIR, 'stop_h3_mapping.csv'))
    stop_to_h3 = dict(zip(mapping['stop_id'].astype(str), mapping['h3_res8']))
    print(f"  정류장 {len(stop_to_h3):,}개 → H3 {mapping['h3_res8'].nunique():,}셀")

    # 2. OSRM 도보거리 (H3 centroid → 정류장)
    dist_path = os.path.join(EB_DIR, 'h3_stop_distance.csv')
    dist_lookup = {}
    if os.path.exists(dist_path):
        dist_df = pd.read_csv(dist_path)
        dist_df = dist_df[dist_df['h3_resolution'] == 8]
        dist_df['stop_id'] = dist_df['stop_id'].astype(str)
        dist_lookup = dict(zip(
            zip(dist_df['h3_cell'], dist_df['stop_id']),
            dist_df['distance_m']
        ))
        print(f"  OSRM 거리 {len(dist_lookup):,}쌍 로드")
    else:
        print(f"  WARNING: {dist_path} 없음 — 직선거리 사용")
        # fallback: use dist_to_centroid from mapping
        for _, row in mapping.iterrows():
            h3 = row['h3_res8']
            sid = str(row['stop_id'])
            dist_lookup[(h3, sid)] = row['dist_to_centroid_res8']

    # 3. 개별 데이터의 고유 od_pair 추출
    print("  고유 od_pair 추출 중...")
    df_od = pd.read_parquet(INPUT_PATH, columns=['od_pair'])
    unique_ods = df_od['od_pair'].unique()
    del df_od
    print(f"  고유 OD: {len(unique_ods):,}")

    # 4. od_pair별 룩업 테이블 구축
    records = []
    n_mapped = 0
    n_failed = 0
    for od in unique_ods:
        parts = od.split('_')
        o_stop, d_stop = parts[0], parts[1]
        o_h3 = stop_to_h3.get(o_stop)
        d_h3 = stop_to_h3.get(d_stop)

        if o_h3 is None or d_h3 is None:
            n_failed += 1
            continue

        h3_od = f"{o_h3}_{d_h3}"
        access_dist = dist_lookup.get((o_h3, o_stop), 0)
        egress_dist = dist_lookup.get((d_h3, d_stop), 0)

        records.append({
            'od_pair': od,
            'o_h3': o_h3,
            'd_h3': d_h3,
            'h3_od': h3_od,
            'ext_access_dist': access_dist,
            'ext_egress_dist': egress_dist,
        })
        n_mapped += 1

    print(f"  H3 매핑 성공: {n_mapped:,} ({n_mapped/(n_mapped+n_failed)*100:.1f}%)")
    print(f"  H3 매핑 실패: {n_failed:,}")

    lookup = pd.DataFrame(records)
    # 도보시간 + 로그변환
    lookup['ext_access_time_min'] = lookup['ext_access_dist'] / 80  # 4.8km/h
    lookup['ext_egress_time_min'] = lookup['ext_egress_dist'] / 80
    lookup['ln_access_ext'] = np.log1p(lookup['ext_access_time_min'])
    lookup['ln_egress_ext'] = np.log1p(lookup['ext_egress_time_min'])

    return lookup


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 4: 개인 통행 단위 H3 OD 학습 데이터 구축")
    print("=" * 60)

    # 1. 룩업 테이블 구축
    print("\n[1/3] OD → H3 룩업 테이블 구축...")
    lookup = build_od_lookup()
    print(f"  룩업 테이블: {len(lookup):,}행")

    # 외부 접근/이탈 거리 통계
    print(f"\n  ext_access_dist: mean={lookup['ext_access_dist'].mean():.0f}m, "
          f"median={lookup['ext_access_dist'].median():.0f}m")
    print(f"  ext_egress_dist: mean={lookup['ext_egress_dist'].mean():.0f}m, "
          f"median={lookup['ext_egress_dist'].median():.0f}m")

    # 2. 청크 단위 처리
    print(f"\n[2/3] 개인 통행 데이터 처리 (청크={CHUNK_SIZE:,})...")
    df_full = pd.read_parquet(INPUT_PATH)
    n_total = len(df_full)
    print(f"  전체: {n_total:,}행")

    # merge
    print("  H3 정보 병합 중...")
    df_merged = df_full.merge(lookup, on='od_pair', how='inner')
    n_merged = len(df_merged)
    print(f"  병합 결과: {n_merged:,}행 ({n_merged/n_total*100:.1f}%)")

    del df_full

    # 3. 저장
    print(f"\n[3/3] 저장: {OUTPUT_PATH}")
    df_merged.to_parquet(OUTPUT_PATH, index=False)

    # 요약
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"완료 ({elapsed:.0f}초)")
    print(f"{'=' * 60}")
    print(f"  입력: {n_total:,}행")
    print(f"  출력: {n_merged:,}행 (H3 매핑 성공)")
    print(f"  고유 H3 OD: {df_merged['h3_od'].nunique():,}")
    print(f"  고유 통행: {df_merged['trip_id'].nunique():,}")
    print(f"  chosen=1: {df_merged['chosen'].sum():,}")
    print(f"  chosen=0: {(df_merged['chosen'] == 0).sum():,}")
    print(f"\n  새 피처:")
    print(f"    ext_access_dist: mean={df_merged['ext_access_dist'].mean():.0f}m")
    print(f"    ext_egress_dist: mean={df_merged['ext_egress_dist'].mean():.0f}m")
    print(f"    ln_access_ext:   mean={df_merged['ln_access_ext'].mean():.3f}")
    print(f"    ln_egress_ext:   mean={df_merged['ln_egress_ext'].mean():.3f}")

    # 선택된 경로 vs 비선택 경로 비교
    chosen = df_merged[df_merged['chosen'] == 1]
    unchosen = df_merged[df_merged['chosen'] == 0]
    print(f"\n  선택 경로 vs 비선택 경로 (주요 피처 평균):")
    for col in ['in_vehicle_time', 'num_transfers', 'fare',
                'access_time', 'egress_time', 'ext_access_dist', 'ext_egress_dist']:
        if col in df_merged.columns:
            c_mean = chosen[col].mean()
            u_mean = unchosen[col].mean()
            print(f"    {col:25s}: chosen={c_mean:.1f}, unchosen={u_mean:.1f}")

    print(f"\n  파일 크기: {os.path.getsize(OUTPUT_PATH)/1e9:.2f} GB")


if __name__ == '__main__':
    main()
