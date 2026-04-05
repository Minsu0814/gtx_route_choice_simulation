# -*- coding: utf-8 -*-
"""
H3 OD 경로선택 학습 데이터 구축

H3 centroid → Raptor 결과를 파싱하여:
1. OTP 캐시 빌드 (raptor_output_h3.json → SQLite)
2. TCN SC 매칭 (stop OD → H3 OD 매핑 후 매칭)
3. OD 집계 + 저장

Usage:
    python build_h3_training_set.py                    # 전체 실행
    python build_h3_training_set.py --max-ods 1000     # 테스트
    python build_h3_training_set.py --force-rebuild     # OTP 캐시 재생성
    python build_h3_training_set.py --force-rematch     # 매칭 재실행
    python build_h3_training_set.py --force-rematch --clean  # 처음부터
"""

import argparse
import csv
import gc
import glob
import os
import pickle
import sqlite3
import sys

import ijson
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.gtfs_lookup import GTFSRouteLookup
from module.route_features import extract_itinerary_features, extract_trip_context, fix_missing_distances, load_bus_type_map
from module.similarity import (
    parse_otp_itinerary,
    parse_smartcard_trip,
    deduplicate_itineraries,
    compute_all_metrics,
    compute_composite_similarity,
)

# ============================================================
# 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')

H3_OTP_INPUT_CSV = os.path.join(DATA_DIR, 'otp', 'input', 'otp_h3_od_input.csv')
H3_OTP_JSON_PATH = os.path.join(DATA_DIR, 'otp', 'output', 'raptor_output_h3.json')
TCN_DIR = os.path.join(DATA_DIR, 'tcn')
OUTPUT_DIR = os.path.join(DATA_DIR, 'training_set_h3')
GTFS_DIR = os.path.join(DATA_DIR, 'gtfs', 'a1')
EB_DIR = os.path.join(DATA_DIR, 'eb')

SIMILARITY_THRESHOLD = 0.5
MIN_CHOICE_SET_SIZE = 2
CHECKPOINT_INTERVAL = 2000

SIM_WEIGHTS = {
    'mode': 0.39, 'route': 0.40, 'sequence': 0.21,
}

TCN_COLUMNS = [
    'od_pair', 'transport_category',
    '환승횟수', '정류장명칭시퀀스', '정류장시퀀스',
    '총탑승시간', '노선명', '노선ID',
    '승차정류장 X 좌표', '승차정류장 Y 좌표',
    '하차정류장 X 좌표', '하차정류장 Y 좌표',
    '정류장lat시퀀스', '정류장lon시퀀스', '승차일시',
]

ROUTE_FEATURES = [
    'total_duration', 'in_vehicle_time', 'walk_time', 'wait_time',
    'access_time', 'egress_time', 'transfer_walk_time',
    'walk_distance', 'total_distance', 'bus_distance',
    'subway_distance', 'gtx_distance',
    'num_transfers', 'num_legs', 'fare', 'generalized_cost',
    'transport_category', 'has_bus', 'has_train', 'has_gtx', 'main_route',
    'bus_subtype',
]


# ============================================================
# Step 0: stop OD → H3 OD 매핑 로드
# ============================================================
def load_stop_to_h3_od():
    """stop od_pair → h3_od 매핑 딕셔너리"""
    mapping = pd.read_csv(os.path.join(EB_DIR, 'stop_h3_mapping.csv'))
    stop_to_h3 = dict(zip(mapping['stop_id'].astype(str), mapping['h3_res8']))

    # h3_choice_prob에서 실제 사용된 stop OD → H3 OD 매핑
    h3_data = pd.read_parquet(
        os.path.join(EB_DIR, 'h3_choice_prob.parquet'),
        columns=['od_pair', 'h3_od']
    )
    stop_od_to_h3_od = dict(zip(h3_data['od_pair'], h3_data['h3_od']))

    # 중복 제거 (같은 stop OD는 같은 h3_od)
    stop_od_to_h3_od_unique = {}
    for od, h3od in stop_od_to_h3_od.items():
        stop_od_to_h3_od_unique[od] = h3od

    print(f'  stop OD → H3 OD 매핑: {len(stop_od_to_h3_od_unique):,}')
    return stop_od_to_h3_od_unique, stop_to_h3


# ============================================================
# Step 1: OTP 캐시 빌드 (Raptor H3 JSON → SQLite)
# ============================================================
def build_otp_cache(otp_cache_db, force_rebuild=False, max_ods=None):
    """Raptor H3 JSON → SQLite 캐시"""
    gtfs_lookup = GTFSRouteLookup(GTFS_DIR)
    load_bus_type_map(os.path.join(GTFS_DIR, 'routes.txt'))

    if not force_rebuild and os.path.exists(otp_cache_db):
        if os.path.getmtime(otp_cache_db) >= os.path.getmtime(H3_OTP_JSON_PATH):
            conn = sqlite3.connect(otp_cache_db)
            n_ods, n_itins = conn.execute(
                'SELECT COUNT(*), SUM(n_alts) FROM otp_cache').fetchone()
            conn.close()
            print(f'H3 OTP 캐시 확인: {n_ods:,} ODs, {n_itins:,} itineraries')
            return

    # CSV에서 id → h3_od 매핑
    od_map = {}
    with open(H3_OTP_INPUT_CSV) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            od_map[i] = row['h3_od']
    print(f'H3 OD CSV: {len(od_map):,}')

    # DB 초기화
    if os.path.exists(otp_cache_db):
        os.remove(otp_cache_db)
    conn = sqlite3.connect(otp_cache_db)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''CREATE TABLE otp_cache (
        h3_od   TEXT PRIMARY KEY,
        n_alts  INTEGER,
        data    BLOB
    )''')

    batch = []
    BATCH_SIZE = 2000
    loaded = 0
    skipped_empty = 0
    skipped_small = 0
    total_itins = 0

    with open(H3_OTP_JSON_PATH, 'rb') as f:
        for item in tqdm(ijson.items(f, 'item', use_float=True),
                         desc='H3 Raptor 스트리밍+피처추출', total=len(od_map)):
            item_id = item.get('id')
            itins = item.get('data', {}).get('plan', {}).get('itineraries', [])
            if not itins:
                skipped_empty += 1
                continue

            h3_od = od_map.get(item_id)
            if h3_od is None:
                continue

            for itin in itins:
                fix_missing_distances(itin)

            deduped = deduplicate_itineraries(itins)
            if len(deduped) < MIN_CHOICE_SET_SIZE:
                skipped_small += 1
                continue

            parsed_list = []
            for itin in deduped:
                parsed = parse_otp_itinerary(itin, gtfs_lookup=gtfs_lookup)
                parsed.pop('route_polyline', None)
                parsed.pop('shape_polyline', None)
                parsed.pop('full_stop_coords', None)
                parsed_list.append(parsed)

            cache_entry = {
                'alt_features': [extract_itinerary_features(itin) for itin in deduped],
                'otp_parsed': parsed_list,
            }
            batch.append((h3_od, len(deduped),
                          pickle.dumps(cache_entry, protocol=pickle.HIGHEST_PROTOCOL)))
            total_itins += len(deduped)
            loaded += 1

            if len(batch) >= BATCH_SIZE:
                conn.executemany('INSERT OR REPLACE INTO otp_cache VALUES (?,?,?)', batch)
                conn.commit()
                batch = []

            if max_ods and loaded >= max_ods:
                break

    if batch:
        conn.executemany('INSERT OR REPLACE INTO otp_cache VALUES (?,?,?)', batch)
        conn.commit()
    conn.close()

    del od_map
    gc.collect()

    db_mb = os.path.getsize(otp_cache_db) / 1024 / 1024
    print(f'\nH3 OTP 캐시 빌드 완료:')
    print(f'  처리 ODs: {loaded:,}, itineraries: {total_itins:,}')
    print(f'  빈 결과 skip: {skipped_empty:,}')
    print(f'  대안 부족 skip (<{MIN_CHOICE_SET_SIZE}): {skipped_small:,}')
    print(f'  DB: {otp_cache_db} ({db_mb:.0f} MB)')


# ============================================================
# Step 2: TCN × H3 OTP 매칭
# ============================================================
def run_matching(otp_cache_db, stop_od_to_h3, force_rematch=False, clean=False):
    """SC 매칭: stop OD → H3 OD로 변환 후 Raptor 경로와 매칭"""
    gtfs_lookup = GTFSRouteLookup(GTFS_DIR)
    load_bus_type_map(os.path.join(GTFS_DIR, 'routes.txt'))

    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    individual_path = os.path.join(OUTPUT_DIR, 'route_choice_h3_individual.parquet')

    if not force_rematch and os.path.exists(individual_path):
        training_df = pd.read_parquet(individual_path)
        print(f'최종 결과 로드: {individual_path}')
        print(f'  {len(training_df):,}행, {training_df["trip_id"].nunique():,} trips')
        return training_df

    if force_rematch and clean:
        existing_ckpts = glob.glob(os.path.join(checkpoint_dir, 'day_*.parquet'))
        for f in existing_ckpts:
            os.remove(f)
        if os.path.exists(individual_path):
            os.remove(individual_path)
        print(f'clean: 체크포인트 {len(existing_ckpts)}개 + 결과 삭제')

    tcn_dates = sorted([d for d in os.listdir(TCN_DIR)
                        if os.path.isdir(os.path.join(TCN_DIR, d))])
    print(f'TCN 날짜: {tcn_dates} ({len(tcn_dates)}일)')

    cache_conn = sqlite3.connect(otp_cache_db)
    h3_od_set = set(r[0] for r in cache_conn.execute('SELECT h3_od FROM otp_cache'))
    print(f'H3 OTP 캐시: {len(h3_od_set):,} H3 ODs')

    os.makedirs(checkpoint_dir, exist_ok=True)

    processed_dates = set()
    trip_counter = 0
    existing_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, 'day_*.parquet')))
    if existing_ckpts:
        for ckpt_file in existing_ckpts:
            # day_20250217_c0.parquet → 20250217
            fname = os.path.basename(ckpt_file).replace('day_', '').replace('.parquet', '')
            date_str = fname.split('_c')[0]  # chunk suffix 제거
            processed_dates.add(date_str)
        last_ckpt = pd.read_parquet(existing_ckpts[-1], columns=['trip_id'])
        trip_counter = max(int(tid.split('_')[-1]) for tid in last_ckpt['trip_id'].unique()) + 1
        del last_ckpt
        n_existing_rows = sum(pq.read_metadata(f).num_rows for f in existing_ckpts)
        print(f'체크포인트 로드: {len(existing_ckpts)}일 완료, {n_existing_rows:,}행')

    skip_count = 0
    match_count = 0
    no_h3_count = 0
    total_sc = 0

    for date in tcn_dates:
        if date in processed_dates:
            print(f'\n--- {date}: 이미 처리됨, skip ---')
            continue

        print(f'\n--- {date} 처리 중 ---')
        tcn_files = glob.glob(os.path.join(TCN_DIR, date, '*.parquet'))
        if not tcn_files:
            continue

        tcn_day = pd.read_parquet(tcn_files[0], columns=TCN_COLUMNS)
        n_day_total = len(tcn_day)

        # stop OD → H3 OD 매핑
        tcn_day['h3_od'] = tcn_day['od_pair'].map(stop_od_to_h3)
        tcn_day = tcn_day.dropna(subset=['h3_od'])
        tcn_day = tcn_day[tcn_day['h3_od'].isin(h3_od_set)]
        n_day_matched = len(tcn_day)

        if n_day_matched == 0:
            print(f'  {date}: {n_day_total:,}건 → 매칭 대상 0건, skip')
            del tcn_day
            continue

        print(f'  {date}: {n_day_total:,}건 → H3 매핑 {n_day_matched:,}건')
        total_sc += n_day_matched

        # H3 OD별 batch pre-fetch
        day_h3_list = tcn_day['h3_od'].unique().tolist()
        otp_cache_batch = {}
        for chunk_start in range(0, len(day_h3_list), 999):
            chunk = day_h3_list[chunk_start:chunk_start + 999]
            ph = ','.join(['?'] * len(chunk))
            for row in cache_conn.execute(
                    f'SELECT h3_od, n_alts, data FROM otp_cache WHERE h3_od IN ({ph})', chunk):
                otp_cache_batch[row[0]] = (row[1], row[2])

        day_rows = []
        day_match = 0
        day_skip = 0
        chunk_idx = 0
        CHUNK_FLUSH = 500_000  # 50만 행마다 flush

        for h3_od, sc_trips in tcn_day.groupby('h3_od'):
            cached = otp_cache_batch.get(h3_od)
            if cached is None:
                continue
            n_alts, cache_blob = cached
            cache = pickle.loads(cache_blob)
            alt_features = cache['alt_features']
            otp_parsed_list = cache['otp_parsed']

            score_cache = {}
            sc_records = sc_trips.to_dict('records')
            for sc_dict in sc_records:
                cache_key = (str(sc_dict.get('노선명', '')),
                             str(sc_dict.get('정류장명칭시퀀스', '')))
                if cache_key not in score_cache:
                    sc_row = pd.Series(sc_dict)
                    sc_parsed = parse_smartcard_trip(sc_row, gtfs_lookup=gtfs_lookup)
                    scores = []
                    for idx, otp_p in enumerate(otp_parsed_list):
                        metrics = compute_all_metrics(otp_p, sc_parsed, skip_diagnostics=True)
                        score_result = compute_composite_similarity(metrics, weights=SIM_WEIGHTS)
                        scores.append({'idx': idx, **score_result})
                    best = max(scores, key=lambda x: x['composite'])
                    score_cache[cache_key] = (scores, best)

            for sc_dict in sc_records:
                cache_key = (str(sc_dict.get('노선명', '')),
                             str(sc_dict.get('정류장명칭시퀀스', '')))
                scores, best = score_cache[cache_key]

                if best['composite'] < SIMILARITY_THRESHOLD:
                    day_skip += 1
                    continue

                day_match += 1
                sc_row = pd.Series(sc_dict)
                context = extract_trip_context(sc_row)
                trip_id = f'{h3_od}_{date}_{trip_counter}'
                trip_counter += 1

                for s in scores:
                    alt_idx = s['idx']
                    day_rows.append({
                        'trip_id': trip_id,
                        'h3_od': h3_od,
                        'stop_od': sc_dict['od_pair'],
                        'alt_idx': alt_idx,
                        'choice_set_size': n_alts,
                        **alt_features[alt_idx],
                        **context,
                        'chosen': int(alt_idx == best['idx']),
                        'sim_composite': round(s['composite'], 4),
                        'sim_mode': round(s['mode_score'], 4),
                        'sim_route': round(s['route_score'], 4),
                        'sim_sequence': round(s['sequence_score'], 4),
                        'sim_time': round(s['time_score'], 4),
                        'sim_spatial': round(s['spatial_score'], 4),
                    })
            del score_cache

            # 메모리 관리: 50만 행마다 청크 저장
            if len(day_rows) >= CHUNK_FLUSH:
                ckpt_path = os.path.join(checkpoint_dir, f'day_{date}_c{chunk_idx}.parquet')
                pd.DataFrame(day_rows).to_parquet(ckpt_path, index=False)
                print(f'    chunk {chunk_idx}: {len(day_rows):,}행 flush')
                day_rows = []
                chunk_idx += 1
                gc.collect()

        del otp_cache_batch

        match_count += day_match
        skip_count += day_skip
        print(f'  {date} 완료: 매칭={day_match:,}, skip={day_skip:,}')

        del tcn_day
        if day_rows:
            ckpt_path = os.path.join(checkpoint_dir, f'day_{date}_c{chunk_idx}.parquet')
            pd.DataFrame(day_rows).to_parquet(ckpt_path, index=False)
            print(f'    chunk {chunk_idx}: {len(day_rows):,}행 flush (마지막)')
        del day_rows
        gc.collect()

    cache_conn.close()

    # 병합
    all_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, 'day_*.parquet')))
    if all_ckpts:
        training_df = pd.concat(
            [pd.read_parquet(f) for f in all_ckpts], ignore_index=True)
    else:
        training_df = pd.DataFrame()

    print(f'\n=== 매칭 완료 ===')
    print(f'  총 SC 대상: {total_sc:,}건')
    print(f'  매칭 성공: {match_count:,}건')
    print(f'  매칭 실패 (threshold): {skip_count:,}건')
    total = match_count + skip_count
    if total > 0:
        print(f'  매칭률: {match_count / total * 100:.1f}%')

    if len(training_df) > 0:
        training_df.to_parquet(
            os.path.join(OUTPUT_DIR, 'route_choice_h3_individual.parquet'), index=False)
        print(f'  individual 저장: {len(training_df):,}행, '
              f'{training_df["trip_id"].nunique():,} trips, '
              f'{training_df["h3_od"].nunique():,} H3 ODs')

    return training_df


# ============================================================
# Step 3: OD 집계
# ============================================================
def aggregate_and_save(training_df):
    """trip 단위 → H3 OD 단위 집계"""
    if len(training_df) == 0:
        print('데이터 없음')
        return

    print('\n=== H3 OD 단위 집계 ===')

    # OD × alt별 경로 피처
    available_features = [f for f in ROUTE_FEATURES if f in training_df.columns]
    od_alt_features = (
        training_df
        .groupby(['h3_od', 'alt_idx', 'choice_set_size'])
        [available_features]
        .first()
        .reset_index()
    )

    # OD × alt별 매칭 통계
    match_stats = (
        training_df
        .groupby(['h3_od', 'alt_idx'])
        .agg(
            n_matched=('chosen', 'sum'),
            n_total=('trip_id', 'nunique'),
            sim_composite=('sim_composite', 'mean'),
            sim_mode=('sim_mode', 'mean'),
            sim_route=('sim_route', 'mean'),
            sim_sequence=('sim_sequence', 'mean'),
            sim_time=('sim_time', 'mean'),
            sim_spatial=('sim_spatial', 'mean'),
        )
        .reset_index()
    )

    # 합치기
    result = od_alt_features.merge(match_stats, on=['h3_od', 'alt_idx'], how='left')

    # n_total은 OD 전체에서의 총 통행 수 (모든 alt에서 같아야)
    od_totals = result.groupby('h3_od')['n_matched'].sum().rename('od_n_total')
    result = result.merge(od_totals, on='h3_od')
    result['n_total'] = result['od_n_total']
    result = result.drop(columns='od_n_total')

    # choice_prob
    result['choice_prob'] = result['n_matched'] / result['n_total'].replace(0, 1)

    # OD distance (context)
    if 'od_distance' in training_df.columns:
        od_dist = training_df.groupby('h3_od')['od_distance'].first()
        result = result.merge(od_dist, on='h3_od', how='left')

    # 저장
    out_path = os.path.join(OUTPUT_DIR, 'route_choice_h3_training.parquet')
    result.to_parquet(out_path, index=False)

    n_ods = result['h3_od'].nunique()
    avg_alts = result.groupby('h3_od').size().mean()
    n_trips = result.groupby('h3_od')['n_total'].first().sum()

    print(f'  H3 ODs: {n_ods:,}')
    print(f'  평균 대안 수: {avg_alts:.1f}')
    print(f'  총 통행 수: {n_trips:,}')
    print(f'  저장: {out_path}')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-ods', type=int, default=None)
    parser.add_argument('--force-rebuild', action='store_true')
    parser.add_argument('--force-rematch', action='store_true')
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    otp_cache_db = os.path.join(OUTPUT_DIR, 'h3_otp_cache.db')

    print('=' * 60)
    print('H3 OD 경로선택 학습 데이터 구축')
    print('=' * 60)

    # Step 0: stop OD → H3 OD 매핑
    print('\n[Step 0] stop OD → H3 OD 매핑 로드...')
    stop_od_to_h3, stop_to_h3 = load_stop_to_h3_od()

    # Step 1: OTP 캐시
    print('\n[Step 1] H3 OTP 캐시 빌드...')
    build_otp_cache(otp_cache_db, args.force_rebuild, args.max_ods)

    # Step 2: SC 매칭
    print('\n[Step 2] TCN × H3 OTP 매칭...')
    training_df = run_matching(otp_cache_db, stop_od_to_h3,
                                args.force_rematch, args.clean)

    # Step 3: 집계
    if len(training_df) > 0:
        aggregate_and_save(training_df)

    print('\n완료!')


if __name__ == '__main__':
    main()
