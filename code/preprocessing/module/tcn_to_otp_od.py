"""
TCN to OTP Input Data Module

TCN 데이터를 OTP(OpenTripPlanner) 요청용 OD pair로 그룹화

Output columns:
- od_pair: OD pair 키 (승차정류장ID_하차정류장ID)
- o_stop_id, d_stop_id: 정류장 ID
- o_lat, o_lon, d_lat, d_lon: 좌표
- departure_time: 대표 출발시간
- trip_count: 통행 건수
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import os


def filter_gtx_trips(tcn: pd.DataFrame) -> pd.DataFrame:
    """
    GTX 이용 통행만 필터링

    transport_category 컬럼에서 'gtx' 문자열이 포함된 행만 반환
    (gtx_only, bus+gtx, train+gtx, bus+train+gtx)

    Args:
        tcn: TCN DataFrame (transport_category 컬럼 필요)

    Returns:
        GTX 이용 통행만 포함된 DataFrame
    """
    mask = tcn['transport_category'].str.contains('gtx', case=False, na=False)
    return tcn[mask].copy()


def create_otp_od_data(tcn: pd.DataFrame) -> pd.DataFrame:
    """
    단일 TCN DataFrame을 OTP 요청용 OD pair로 그룹화

    Args:
        tcn: TCN DataFrame (od_pair, 좌표, 승차일시 컬럼 필요)

    Returns:
        OD pair로 그룹화된 DataFrame
    """
    od_grouped = tcn.groupby(['od_pair', '승차정류장ID', '하차정류장ID']).agg({
        '승차정류장 X 좌표': 'first',
        '승차정류장 Y 좌표': 'first',
        '하차정류장 X 좌표': 'first',
        '하차정류장 Y 좌표': 'first',
        '승차일시': 'first',
        'group_key': 'count'
    }).reset_index()

    od_grouped = od_grouped.rename(columns={
        '승차정류장ID': 'o_stop_id',
        '하차정류장ID': 'd_stop_id',
        '승차정류장 X 좌표': 'o_lat',
        '승차정류장 Y 좌표': 'o_lon',
        '하차정류장 X 좌표': 'd_lat',
        '하차정류장 Y 좌표': 'd_lon',
        '승차일시': 'departure_time',
        'group_key': 'trip_count'
    })

    return od_grouped


def merge_multiple_tcn(tcn_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    여러 날짜의 TCN을 합치고 OD pair로 그룹화

    Args:
        tcn_list: TCN DataFrame 리스트

    Returns:
        통합 OD pair DataFrame
    """
    # 각 TCN을 OD로 그룹화
    od_list = [create_otp_od_data(tcn) for tcn in tcn_list]

    # 전체 합치기
    all_od = pd.concat(od_list, ignore_index=True)

    # 다시 OD pair로 집계 (여러 날짜 통합)
    final_od = all_od.groupby(['od_pair', 'o_stop_id', 'd_stop_id']).agg({
        'o_lat': 'first',
        'o_lon': 'first',
        'd_lat': 'first',
        'd_lon': 'first',
        'departure_time': 'first',  # 대표 시간
        'trip_count': 'sum'         # 총 통행 건수
    }).reset_index()

    return final_od


def load_and_merge_tcn_files(tcn_paths: List[str], filter_gtx: bool = False) -> pd.DataFrame:
    """
    여러 TCN parquet 파일을 로드하여 OD pair로 그룹화

    Args:
        tcn_paths: TCN parquet 파일 경로 리스트
        filter_gtx: True이면 GTX 이용 통행만 필터링

    Returns:
        통합 OD pair DataFrame
    """
    tcn_list = []
    for path in tcn_paths:
        print(f"Loading {path}...")
        tcn = pd.read_parquet(path)
        if filter_gtx:
            tcn = filter_gtx_trips(tcn)
            print(f"  {len(tcn):,} GTX trips")
        else:
            print(f"  {len(tcn):,} trips")
        tcn_list.append(tcn)

    print(f"\nMerging {len(tcn_list)} files...")
    od_data = merge_multiple_tcn(tcn_list)
    print(f"Total OD pairs: {len(od_data):,}")
    print(f"Total trips: {od_data['trip_count'].sum():,}")

    return od_data


def save_otp_input(od_data: pd.DataFrame, output_path: str):
    """
    OTP 입력 데이터 저장

    Args:
        od_data: OD pair DataFrame
        output_path: 출력 파일 경로 (.csv 또는 .parquet)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.endswith('.parquet'):
        od_data.to_parquet(output_path, index=False)
    else:
        od_data.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def process_tcn_to_otp_input(
    tcn_paths: List[str],
    output_path: Optional[str] = None,
    filter_gtx: bool = False
) -> pd.DataFrame:
    """
    TCN 파일들을 OTP 입력용 OD pair 데이터로 변환

    Args:
        tcn_paths: TCN parquet 파일 경로 리스트
        output_path: 출력 경로 (None이면 저장 안 함)
        filter_gtx: True이면 GTX 이용 통행만 필터링

    Returns:
        OD pair DataFrame

    Example:
        >>> tcn_paths = [f'data/TCN_{d}.parquet' for d in dates]
        >>> od_data = process_tcn_to_otp_input(tcn_paths, 'data/otp_input.csv')
        >>> # GTX 통행만 필터링
        >>> od_gtx = process_tcn_to_otp_input(tcn_paths, 'data/otp_gtx.csv', filter_gtx=True)
    """
    od_data = load_and_merge_tcn_files(tcn_paths, filter_gtx=filter_gtx)

    if output_path:
        save_otp_input(od_data, output_path)

    return od_data
