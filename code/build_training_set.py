"""
Route Choice Model Training Dataset 구축
OTP K-best 경로 × 스마트카드 실제 통행 → OD별 선택 확률 데이터셋

Usage:
    python build_training_set.py                    # 전체 실행
    python build_training_set.py --max-ods 1000     # 테스트 (1000 OD)
    python build_training_set.py --force-rebuild     # OTP 캐시 재생성
    python build_training_set.py --force-rematch     # 매칭 재실행
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

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.similarity.route_features import extract_itinerary_features, extract_trip_context, fix_missing_distances
from module.similarity.similarity import (
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
OTP_INPUT_CSV = os.path.join(BASE_DIR, '..', 'data', 'otp', 'input', 'otp_od_input_over13.csv')
OTP_JSON_PATH = os.path.join(BASE_DIR, '..', 'data', 'otp', 'output', 'similarity.json')
TCN_DIR = os.path.join(BASE_DIR, '..', 'data', 'tcn')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'data', 'training_set')

SIMILARITY_THRESHOLD = 0.6
MIN_CHOICE_SET_SIZE = 2
CHECKPOINT_INTERVAL = 2000

SIM_WEIGHTS = {
    'mode': 0.20, 'sequence': 0.20, 'time': 0.20,
    'route': 0.30, 'spatial': 0.10,
}

# TCN 매칭에 필요한 컬럼
TCN_COLUMNS = [
    'od_pair', 'transport_category',
    '환승횟수', '정류장명칭시퀀스', '정류장시퀀스',
    '총탑승시간', '노선명', '노선ID',
    '승차정류장 X 좌표', '승차정류장 Y 좌표',
    '하차정류장 X 좌표', '하차정류장 Y 좌표',
    '정류장lat시퀀스', '정류장lon시퀀스', '승차일시',
]

# 경로 피처 (OD × alt별 동일)
ROUTE_FEATURES = [
    'total_duration', 'in_vehicle_time', 'walk_time', 'wait_time',
    'access_time', 'egress_time', 'transfer_walk_time',
    'walk_distance', 'total_distance', 'bus_distance',
    'subway_distance', 'gtx_distance',
    'num_transfers', 'num_legs', 'fare', 'generalized_cost',
    'transport_category', 'has_bus', 'has_train', 'has_gtx', 'main_route',
]


# ============================================================
# Step 1: OTP 캐시 빌드 (SQLite)
# ============================================================
def build_otp_cache(otp_cache_db, force_rebuild=False, max_ods=None):
    """OTP JSON → SQLite 캐시 (메모리 최소화)"""

    if not force_rebuild and os.path.exists(otp_cache_db):
        if os.path.getmtime(otp_cache_db) >= os.path.getmtime(OTP_JSON_PATH):
            conn = sqlite3.connect(otp_cache_db)
            n_ods, n_itins = conn.execute(
                'SELECT COUNT(*), SUM(n_alts) FROM otp_cache').fetchone()
            conn.close()
            print(f'OTP 캐시 확인: {n_ods:,} ODs, {n_itins:,} itineraries')
            return
        else:
            print('JSON이 캐시보다 최신 → 재생성')

    # CSV에서 id → od_pair 매핑
    od_map = {}
    with open(OTP_INPUT_CSV) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            od_map[i] = row['od_pair']
    print(f'CSV OD pairs: {len(od_map):,}')

    # DB 초기화
    if os.path.exists(otp_cache_db):
        os.remove(otp_cache_db)
    conn = sqlite3.connect(otp_cache_db)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''CREATE TABLE otp_cache (
        od_pair TEXT PRIMARY KEY,
        n_alts  INTEGER,
        data    BLOB
    )''')

    # Single-pass 스트리밍
    batch = []
    BATCH_SIZE = 2000
    loaded = 0
    skipped_empty = 0
    skipped_small = 0
    total_itins = 0

    with open(OTP_JSON_PATH, 'rb') as f:
        for item in tqdm(ijson.items(f, 'item', use_float=True),
                         desc='OTP 스트리밍+피처추출', total=len(od_map)):
            item_id = item.get('id')
            itins = item.get('data', {}).get('plan', {}).get('itineraries', [])
            if not itins:
                skipped_empty += 1
                continue

            od_pair = od_map.get(item_id)
            if od_pair is None:
                continue

            for itin in itins:
                fix_missing_distances(itin)

            deduped = deduplicate_itineraries(itins)
            if len(deduped) < MIN_CHOICE_SET_SIZE:
                skipped_small += 1
                continue

            parsed_list = []
            for itin in deduped:
                parsed = parse_otp_itinerary(itin, gtfs_lookup=None)
                parsed.pop('route_polyline', None)
                parsed.pop('shape_polyline', None)
                parsed.pop('full_stop_coords', None)
                parsed_list.append(parsed)

            cache_entry = {
                'alt_features': [extract_itinerary_features(itin) for itin in deduped],
                'otp_parsed': parsed_list,
            }
            batch.append((od_pair, len(deduped),
                          pickle.dumps(cache_entry, protocol=pickle.HIGHEST_PROTOCOL)))
            total_itins += len(deduped)
            loaded += 1

            if len(batch) >= BATCH_SIZE:
                conn.executemany('INSERT INTO otp_cache VALUES (?,?,?)', batch)
                conn.commit()
                batch = []

            if max_ods and loaded >= max_ods:
                break

    if batch:
        conn.executemany('INSERT INTO otp_cache VALUES (?,?,?)', batch)
        conn.commit()
    conn.close()

    del od_map
    gc.collect()

    db_mb = os.path.getsize(otp_cache_db) / 1024 / 1024
    print(f'\nOTP 캐시 빌드 완료:')
    print(f'  처리 ODs: {loaded:,}, itineraries: {total_itins:,}')
    print(f'  빈 결과 skip: {skipped_empty:,}')
    print(f'  대안 부족 skip (<{MIN_CHOICE_SET_SIZE}): {skipped_small:,}')
    print(f'  DB: {otp_cache_db} ({db_mb:.0f} MB)')


# ============================================================
# Step 2: TCN × OTP 매칭
# ============================================================
def run_matching(otp_cache_db, force_rematch=False):
    """날짜별 TCN → OTP 매칭 → 체크포인트 저장"""

    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    individual_path = os.path.join(OUTPUT_DIR, 'route_choice_individual.parquet')

    # 이미 결과가 있으면 스킵
    if not force_rematch and os.path.exists(individual_path):
        training_df = pd.read_parquet(individual_path)
        print(f'최종 결과 로드: {individual_path}')
        print(f'  {len(training_df):,}행, {training_df["trip_id"].nunique():,} trips, '
              f'{training_df["od_pair"].nunique():,} ODs')
        return training_df

    tcn_dates = sorted([d for d in os.listdir(TCN_DIR)
                        if os.path.isdir(os.path.join(TCN_DIR, d))])
    print(f'TCN 날짜: {tcn_dates} ({len(tcn_dates)}일)')

    # DB 연결
    cache_conn = sqlite3.connect(otp_cache_db)
    otp_od_set = set(r[0] for r in cache_conn.execute('SELECT od_pair FROM otp_cache'))
    print(f'otp_cache DB: {len(otp_od_set):,} ODs')

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 이어하기: 기존 체크포인트 확인
    processed_dates = set()
    trip_counter = 0
    existing_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, 'day_*.parquet')))
    if existing_ckpts:
        for ckpt_file in existing_ckpts:
            date_str = os.path.basename(ckpt_file).replace('day_', '').replace('.parquet', '')
            processed_dates.add(date_str)
        last_ckpt = pd.read_parquet(existing_ckpts[-1], columns=['trip_id'])
        trip_counter = max(int(tid.split('_')[-1]) for tid in last_ckpt['trip_id'].unique()) + 1
        del last_ckpt
        n_existing_rows = sum(pq.read_metadata(f).num_rows for f in existing_ckpts)
        print(f'체크포인트 로드: {len(existing_ckpts)}일 완료, {n_existing_rows:,}행')
        print(f'  완료된 날짜: {sorted(processed_dates)}')
    else:
        print('체크포인트 없음, 처음부터 시작')

    skip_count = 0
    match_count = 0
    total_sc = 0

    for date in tcn_dates:
        if date in processed_dates:
            print(f'\n--- {date}: 이미 처리됨, skip ---')
            continue

        print(f'\n--- {date} 처리 중 ---')
        tcn_files = glob.glob(os.path.join(TCN_DIR, date, '*.parquet'))
        if not tcn_files:
            print(f'  파일 없음, skip')
            continue

        tcn_day = pd.read_parquet(tcn_files[0], columns=TCN_COLUMNS)
        n_day_total = len(tcn_day)
        tcn_day = tcn_day[tcn_day['od_pair'].isin(otp_od_set)]
        n_day_matched = len(tcn_day)

        if n_day_matched == 0:
            print(f'  {date}: {n_day_total:,}건 중 매칭 대상 0건, skip')
            del tcn_day
            continue

        mem_mb = tcn_day.memory_usage(deep=True).sum() / 1024**2
        print(f'  {date}: {n_day_total:,}건 → 매칭 대상 {n_day_matched:,}건 ({mem_mb:.0f} MB)')
        total_sc += n_day_matched

        day_rows = []
        day_match = 0
        day_skip = 0

        for od_pair, sc_trips in tcn_day.groupby('od_pair'):
            row = cache_conn.execute(
                'SELECT n_alts, data FROM otp_cache WHERE od_pair = ?',
                (od_pair,)).fetchone()
            if row is None:
                continue
            n_alts = row[0]
            cache = pickle.loads(row[1])
            alt_features = cache['alt_features']
            otp_parsed_list = cache['otp_parsed']

            for _, sc_row in sc_trips.iterrows():
                sc_parsed = parse_smartcard_trip(sc_row, gtfs_lookup=None)

                scores = []
                for idx, otp_p in enumerate(otp_parsed_list):
                    metrics = compute_all_metrics(otp_p, sc_parsed)
                    score_result = compute_composite_similarity(metrics, weights=SIM_WEIGHTS)
                    scores.append({'idx': idx, **score_result})

                best = max(scores, key=lambda x: x['composite'])
                if best['composite'] < SIMILARITY_THRESHOLD:
                    day_skip += 1
                    continue

                day_match += 1
                context = extract_trip_context(sc_row)
                trip_id = f'{od_pair}_{date}_{trip_counter}'
                trip_counter += 1

                for s in scores:
                    alt_idx = s['idx']
                    day_rows.append({
                        'trip_id': trip_id,
                        'od_pair': od_pair,
                        'alt_idx': alt_idx,
                        'choice_set_size': n_alts,
                        **alt_features[alt_idx],
                        **context,
                        'chosen': int(alt_idx == best['idx']),
                        'sim_composite': round(s['composite'], 4),
                        'sim_mode': round(s['mode_score'], 4),
                        'sim_sequence': round(s['sequence_score'], 4),
                        'sim_time': round(s['time_score'], 4),
                        'sim_route': round(s['route_score'], 4),
                        'sim_spatial': round(s['spatial_score'], 4),
                    })

        match_count += day_match
        skip_count += day_skip
        print(f'  {date} 완료: 매칭={day_match:,}, skip={day_skip:,}')

        del tcn_day
        if day_rows:
            day_df = pd.DataFrame(day_rows)
            ckpt_path = os.path.join(checkpoint_dir, f'day_{date}.parquet')
            day_df.to_parquet(ckpt_path, index=False)
            print(f'  체크포인트 저장: {len(day_rows):,}행 → {ckpt_path}')
            del day_df
        del day_rows
        gc.collect()

    cache_conn.close()

    # 전체 날짜 병합
    all_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, 'day_*.parquet')))
    if all_ckpts:
        training_df = pd.concat(
            [pd.read_parquet(f) for f in all_ckpts],
            ignore_index=True,
        )
    else:
        training_df = pd.DataFrame()

    print(f'\n=== 매칭 완료 ===')
    print(f'  총 SC 대상: {total_sc:,}건')
    print(f'  매칭 성공: {match_count:,}건')
    print(f'  매칭 실패: {skip_count:,}건')
    total = match_count + skip_count
    if total > 0:
        print(f'  매칭률: {match_count / total * 100:.1f}%')
    print(f'  총 행 수: {len(training_df):,}')
    if len(training_df) > 0:
        print(f'  선택상황(trips): {training_df["trip_id"].nunique():,}')
        print(f'  OD pairs: {training_df["od_pair"].nunique():,}')

    return training_df


# ============================================================
# Step 3: OD 단위 집계 + 저장
# ============================================================
def aggregate_and_save(training_df):
    """trip 단위 → OD 단위 집계, choice_prob 계산, 저장"""

    if len(training_df) == 0:
        print('데이터 없음, 저장 skip')
        return

    print('\n=== OD 단위 집계 ===')

    # 1. OD × alt별 경로 피처
    od_alt_features = (
        training_df
        .groupby(['od_pair', 'alt_idx', 'choice_set_size'])
        [ROUTE_FEATURES]
        .first()
        .reset_index()
    )

    # 2. OD × alt별 chosen 집계
    chosen_counts = (
        training_df[training_df['chosen'] == 1]
        .groupby(['od_pair', 'alt_idx'])
        .size()
        .reset_index(name='n_matched')
    )

    # 3. OD별 전체 trip 수
    total_trips = (
        training_df
        .groupby('od_pair')['trip_id']
        .nunique()
        .reset_index(name='n_total')
    )

    # 4. OD별 od_distance
    od_dist = (
        training_df
        .groupby('od_pair')['od_distance']
        .first()
        .reset_index()
    )

    # 5. 병합
    agg_df = od_alt_features.merge(chosen_counts, on=['od_pair', 'alt_idx'], how='left')
    agg_df = agg_df.merge(total_trips, on='od_pair', how='left')
    agg_df = agg_df.merge(od_dist, on='od_pair', how='left')
    agg_df['n_matched'] = agg_df['n_matched'].fillna(0).astype(int)

    # 6. choice_prob
    agg_df['choice_prob'] = agg_df['n_matched'] / agg_df['n_total']

    # 7. 평균 유사도 (검증용)
    sim_cols = ['sim_composite', 'sim_mode', 'sim_sequence', 'sim_time', 'sim_route', 'sim_spatial']
    avg_sim = (
        training_df[training_df['chosen'] == 1]
        .groupby(['od_pair', 'alt_idx'])
        [sim_cols]
        .mean()
        .reset_index()
    )
    agg_df = agg_df.merge(avg_sim, on=['od_pair', 'alt_idx'], how='left')
    for col in sim_cols:
        agg_df[col] = agg_df[col].fillna(0)

    agg_df = agg_df.sort_values(['od_pair', 'alt_idx']).reset_index(drop=True)

    # 검증
    prob_sums = agg_df.groupby('od_pair')['choice_prob'].sum()
    not_one = prob_sums[(prob_sums - 1.0).abs() > 0.001]
    print(f'  행 수: {len(agg_df):,} (OD × alt)')
    print(f'  OD pairs: {agg_df["od_pair"].nunique():,}')
    print(f'  choice_prob 합=1.0 검증: {"PASS" if len(not_one) == 0 else f"FAIL ({len(not_one)} ODs)"}')

    # 저장
    parquet_path = os.path.join(OUTPUT_DIR, 'route_choice_training.parquet')
    csv_path = os.path.join(OUTPUT_DIR, 'route_choice_training.csv')
    individual_path = os.path.join(OUTPUT_DIR, 'route_choice_individual.parquet')

    agg_df.to_parquet(parquet_path, index=False)
    agg_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    training_df.to_parquet(individual_path, index=False)

    # 체크포인트 정리
    checkpoint_path = os.path.join(OUTPUT_DIR, 'training_checkpoint.parquet')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print('  체크포인트 삭제')

    print(f'\n=== 저장 완료 ===')
    print(f'  [메인] {parquet_path}')
    print(f'    {len(agg_df):,}행, {agg_df["od_pair"].nunique():,} ODs, {len(agg_df.columns)} 컬럼')
    print(f'  [개인] {individual_path}')
    print(f'    {len(training_df):,}행')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Build route choice training set')
    parser.add_argument('--max-ods', type=int, default=None, help='테스트용 OD 수 제한')
    parser.add_argument('--force-rebuild', action='store_true', help='OTP 캐시 재생성')
    parser.add_argument('--force-rematch', action='store_true', help='매칭 재실행')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    otp_cache_db = os.path.join(OUTPUT_DIR, 'otp_cache.db')

    print('=' * 60)
    print('Route Choice Training Set Builder')
    print('=' * 60)

    # Step 1: OTP 캐시
    print('\n[Step 1] OTP 캐시 빌드')
    build_otp_cache(otp_cache_db, force_rebuild=args.force_rebuild, max_ods=args.max_ods)

    # Step 2: 매칭
    print('\n[Step 2] TCN × OTP 매칭')
    training_df = run_matching(otp_cache_db, force_rematch=args.force_rematch)

    # Step 3: 집계 + 저장
    print('\n[Step 3] OD 집계 + 저장')
    aggregate_and_save(training_df)

    print('\n완료!')


if __name__ == '__main__':
    main()
