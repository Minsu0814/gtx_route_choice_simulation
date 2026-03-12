"""
데이터 샘플링 스크립트
OD 10,000개 기준으로 4개 데이터를 샘플링하여 경량 버전 생성

Usage:
    python sample_data.py
    python sample_data.py --n-ods 1000      # OD 수 변경
    python sample_data.py --date 20250218   # TCN 날짜 변경
"""

import argparse
import json
import os
import shutil

import ijson
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data_sample')


def sample_data(n_ods=10000, tcn_date='20250217', seed=42):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================================================================
    # 1. OTP input: OD 샘플링 (기준)
    # ================================================================
    print(f'\n[1/4] OTP input에서 {n_ods:,}개 OD 샘플링...')
    otp_input_path = os.path.join(DATA_DIR, 'otp', 'input', 'otp_od_input_over13.csv')
    df_otp = pd.read_csv(otp_input_path)
    print(f'  전체 OD: {len(df_otp):,}')

    df_sampled = df_otp.sample(n=min(n_ods, len(df_otp)), random_state=seed)
    sampled_od_pairs = set(df_sampled['od_pair'].astype(str))
    # 원본 인덱스(= similarity.json의 id) 보존
    sampled_indices = set(df_sampled.index)

    out_otp_dir = os.path.join(OUTPUT_DIR, 'otp', 'input')
    os.makedirs(out_otp_dir, exist_ok=True)
    df_sampled_reset = df_sampled.reset_index(drop=True)
    out_otp_path = os.path.join(out_otp_dir, 'otp_od_input_over13.csv')
    df_sampled_reset.to_csv(out_otp_path, index=False)
    print(f'  저장: {out_otp_path} ({len(df_sampled_reset):,} rows)')

    # 인덱스 매핑 저장 (원본 index → 새 index)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(sampled_indices))}

    # ================================================================
    # 2. similarity.json: 샘플된 OD만 추출
    # ================================================================
    print(f'\n[2/4] similarity.json에서 샘플된 OD 추출...')
    sim_input = os.path.join(DATA_DIR, 'otp', 'output', 'similarity.json')
    out_sim_dir = os.path.join(OUTPUT_DIR, 'otp', 'output')
    os.makedirs(out_sim_dir, exist_ok=True)
    out_sim_path = os.path.join(out_sim_dir, 'similarity.json')

    extracted = 0
    with open(sim_input, 'rb') as fin, open(out_sim_path, 'w', encoding='utf-8') as fout:
        fout.write('[\n')
        first = True
        for item in tqdm(ijson.items(fin, 'item', use_float=True),
                         desc='  스트리밍', total=len(df_otp)):
            item_id = item.get('id')
            if item_id in sampled_indices:
                # id를 새 인덱스로 변환
                item['id'] = index_map[item_id]
                if not first:
                    fout.write(',\n')
                json.dump(item, fout, ensure_ascii=False)
                first = False
                extracted += 1
                if extracted >= n_ods:
                    break
        fout.write('\n]')

    sim_size_mb = os.path.getsize(out_sim_path) / 1024 / 1024
    print(f'  추출: {extracted:,}개 OD, {sim_size_mb:.1f} MB')

    # ================================================================
    # 3. TCN: 해당 OD + 1일치만 추출
    # ================================================================
    print(f'\n[3/4] TCN {tcn_date} 에서 샘플된 OD 필터...')
    tcn_path = os.path.join(DATA_DIR, 'tcn', tcn_date, f'TCN_{tcn_date}_route.parquet')
    df_tcn = pd.read_parquet(tcn_path)
    print(f'  전체 TCN 행: {len(df_tcn):,}')

    df_tcn_filtered = df_tcn[df_tcn['od_pair'].astype(str).isin(sampled_od_pairs)]
    out_tcn_dir = os.path.join(OUTPUT_DIR, 'tcn', tcn_date)
    os.makedirs(out_tcn_dir, exist_ok=True)
    out_tcn_path = os.path.join(out_tcn_dir, f'TCN_{tcn_date}_route.parquet')
    df_tcn_filtered.to_parquet(out_tcn_path, index=False)
    tcn_size_mb = os.path.getsize(out_tcn_path) / 1024 / 1024
    print(f'  필터된 TCN: {len(df_tcn_filtered):,} rows, {tcn_size_mb:.1f} MB')

    # ================================================================
    # 4. GTFS: 캐시 파일 + 작은 원본만 복사
    # ================================================================
    print(f'\n[4/4] GTFS 파일 복사...')
    gtfs_src = os.path.join(DATA_DIR, 'gtfs', 'a1')
    gtfs_dst = os.path.join(OUTPUT_DIR, 'gtfs', 'a1')
    os.makedirs(gtfs_dst, exist_ok=True)

    # 캐시 파일 (필수)
    for fname in ['route_stops_cache.parquet', 'shape_polylines_cache.parquet']:
        src = os.path.join(gtfs_src, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(gtfs_dst, fname))
            size_mb = os.path.getsize(src) / 1024 / 1024
            print(f'  복사: {fname} ({size_mb:.1f} MB)')

    # 작은 원본 파일 (모델에서 참조할 수 있음)
    for fname in ['agency.txt', 'calendar.txt', 'routes.txt', 'stops.txt', 'trips.txt']:
        src = os.path.join(gtfs_src, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(gtfs_dst, fname))
            size_mb = os.path.getsize(src) / 1024 / 1024
            print(f'  복사: {fname} ({size_mb:.1f} MB)')

    # shapes.txt, stop_times.txt는 너무 큼 → 스킵 (캐시로 대체)
    print('  스킵: shapes.txt (1.5GB), stop_times.txt (1.6GB) → 캐시로 대체')

    # ================================================================
    # 결과 요약
    # ================================================================
    print('\n' + '=' * 60)
    print('샘플링 완료!')
    print('=' * 60)
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    total_mb = total_size / 1024 / 1024
    print(f'  출력 폴더: {OUTPUT_DIR}')
    print(f'  총 크기: {total_mb:.1f} MB')
    print(f'  샘플 OD: {n_ods:,}개')
    print(f'  TCN 날짜: {tcn_date}')
    print()

    # 폴더별 크기
    for item in sorted(os.listdir(OUTPUT_DIR)):
        item_path = os.path.join(OUTPUT_DIR, item)
        if os.path.isdir(item_path):
            size = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _, files in os.walk(item_path) for f in files
            )
            print(f'  {item}/: {size / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='데이터 샘플링')
    parser.add_argument('--n-ods', type=int, default=10000, help='샘플 OD 수 (default: 10000)')
    parser.add_argument('--date', default='20250217', help='TCN 날짜 (default: 20250217)')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    args = parser.parse_args()
    sample_data(n_ods=args.n_ods, tcn_date=args.date, seed=args.seed)
