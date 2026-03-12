"""
TCD (Transaction Card Data) to TCN (Trip Chain Network) Conversion Module
- Route-based version: ROUT/ROUTESTTN 기반 정류장 매칭

기존 tcd_to_tcn.py와의 차이:
- 기존: TCD + STTN → 정산지역코드 + 정류장ID로 정류장 좌표 매칭
- 변경: TCD + ROUTESTTN → 노선ID + 정류장ID로 노선별 정류장 좌표 매칭
- ROUTESTTN에서 교통수단유형, 노선명, 정류장순서, 누적거리 등 추가 정보 활용

TCD: 개별 환승 레코드 (한 통행에서 환승이 있으면 여러 행)
TCN: 한 사람의 한 통행을 하나의 행으로 묶은 데이터

Features:
- 병렬 리스트 형식으로 leg 정보 저장
- 이용거리, 탑승시간, 정류장 시퀀스 포함
- 왕복 통행 분리 기능
- 노선별 정류장 순서/누적거리 정보 포함
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from typing import Optional, List, Dict, Tuple
from collections import defaultdict, deque
from itertools import combinations
import os

_TYPE_TO_MODE = {'B': 'bus', 'T': 'train', 'G': 'gtx'}

def _mode_set_to_category(mode_set):
    """모드 집합 → 7개 카테고리 분류"""
    modes = mode_set - {'unknown'}
    if not modes:
        return 'unknown'
    if modes == {'bus'}:
        return 'bus_only'
    if modes == {'train'}:
        return 'train_only'
    if modes == {'gtx'}:
        return 'gtx_only'
    if modes == {'bus', 'train'}:
        return 'bus+train'
    if modes == {'bus', 'gtx'}:
        return 'bus+gtx'
    if modes == {'train', 'gtx'}:
        return 'train+gtx'
    if modes >= {'bus', 'train', 'gtx'}:
        return 'bus+train+gtx'
    return 'unknown'


def haversine_vectorized(lat1: np.ndarray, lon1: np.ndarray,
                         lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    벡터화된 Haversine 거리 계산 (미터 단위)

    Args:
        lat1, lon1: 시작점 좌표
        lat2, lon2: 끝점 좌표

    Returns:
        거리 배열 (미터)
    """
    R = 6371000  # 지구 반지름 (미터)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def fix_swapped_coords_tcn(df: pd.DataFrame) -> pd.DataFrame:
    """
    TCN 데이터에서 위경도가 뒤바뀐 행 수정

    한국 좌표 기준: 위도 33-43, 경도 124-132
    lat > 100 또는 lon < 100이면 뒤바뀐 것으로 판단

    Args:
        df: TCN DataFrame (승차/하차정류장 X/Y 좌표 컬럼 필요)

    Returns:
        수정된 DataFrame
    """
    df = df.copy()

    # 승차 좌표 수정
    o_swapped = (df['승차정류장 X 좌표'] > 100) | (df['승차정류장 Y 좌표'] < 100)
    df.loc[o_swapped, ['승차정류장 X 좌표', '승차정류장 Y 좌표']] = \
        df.loc[o_swapped, ['승차정류장 Y 좌표', '승차정류장 X 좌표']].values

    # 하차 좌표 수정
    d_swapped = (df['하차정류장 X 좌표'] > 100) | (df['하차정류장 Y 좌표'] < 100)
    df.loc[d_swapped, ['하차정류장 X 좌표', '하차정류장 Y 좌표']] = \
        df.loc[d_swapped, ['하차정류장 Y 좌표', '하차정류장 X 좌표']].values

    print(f"  Origin coords fixed: {o_swapped.sum()}")
    print(f"  Destination coords fixed: {d_swapped.sum()}")

    return df


def filter_short_trips(df: pd.DataFrame, min_distance: float = 500) -> pd.DataFrame:
    """
    최소 거리 미만의 짧은 통행 필터링

    Args:
        df: TCN DataFrame
        min_distance: 최소 OD 거리 (미터)

    Returns:
        필터링된 DataFrame
    """
    # Haversine 거리 계산
    df = df.copy()
    df['od_distance'] = haversine_vectorized(
        df['승차정류장 X 좌표'].values,
        df['승차정류장 Y 좌표'].values,
        df['하차정류장 X 좌표'].values,
        df['하차정류장 Y 좌표'].values
    )

    before_count = len(df)
    filtered = df[df['od_distance'] >= min_distance].copy()

    print(f"  Before distance filter: {before_count:,}")
    print(f"  After distance filter (>= {min_distance}m): {len(filtered):,}")
    print(f"  Removed short trips: {before_count - len(filtered):,}")

    return filtered


def split_round_trips(df: pd.DataFrame, max_transfer_dist: int = 1000) -> pd.DataFrame:
    """
    왕복 통행 분리 (방향 반전 + 출발지로 접근 중 감지)

    Args:
        df: TCD DataFrame (구분코드, 승차일시, 좌표 컬럼 필요)
        max_transfer_dist: 최대 환승 거리 (미터), 이 이상이면 새 통행으로 간주

    Returns:
        sub_trip 컬럼이 추가된 DataFrame
    """
    df = df.sort_values(['구분코드', '승차일시']).reset_index(drop=True)

    prev_구분코드 = df['구분코드'].shift(1)

    # 1. 환승 거리 계산
    transfer_dist = haversine_vectorized(
        df['하차정류장 X 좌표'].shift(1).values,
        df['하차정류장 Y 좌표'].shift(1).values,
        df['승차정류장 X 좌표'].values,
        df['승차정류장 Y 좌표'].values
    )

    # 2. 방향 벡터 계산
    vec_lat = df['하차정류장 X 좌표'] - df['승차정류장 X 좌표']
    vec_lon = df['하차정류장 Y 좌표'] - df['승차정류장 Y 좌표']

    dot_product = (vec_lat * vec_lat.shift(1)) + (vec_lon * vec_lon.shift(1))
    direction_reversed = dot_product < 0

    # 3. 출발지와의 거리 변화
    df['_first_lat'] = df.groupby('구분코드')['승차정류장 X 좌표'].transform('first')
    df['_first_lon'] = df.groupby('구분코드')['승차정류장 Y 좌표'].transform('first')

    curr_dist_to_origin = haversine_vectorized(
        df['하차정류장 X 좌표'].values, df['하차정류장 Y 좌표'].values,
        df['_first_lat'].values, df['_first_lon'].values
    )

    prev_dist_to_origin = np.roll(curr_dist_to_origin, 1)
    prev_dist_to_origin[0] = np.nan

    heading_to_origin = curr_dist_to_origin < prev_dist_to_origin

    # 4. 새 통행 조건
    new_trip = (
        (df['구분코드'] != prev_구분코드) |
        (transfer_dist > max_transfer_dist) |
        (direction_reversed & heading_to_origin)
    )
    new_trip.iloc[0] = True

    df = df.drop(columns=['_first_lat', '_first_lon'])
    df['_new_trip'] = new_trip.astype(int)
    df['sub_trip'] = df.groupby('구분코드')['_new_trip'].cumsum() - 1
    df = df.drop(columns='_new_trip')

    return df


class SubwayTransferGraph:
    """지하철 노선 간 환승 그래프 (ROUTESTTN 기반)"""

    def __init__(self, routesttn: pd.DataFrame, rail_types: Optional[List[str]] = None):
        """
        Args:
            routesttn: ROUTESTTN DataFrame (전처리 전/후 모두 가능)
            rail_types: 지하철/GTX 교통수단유형 목록
        """
        if rail_types is None:
            rail_types = ['T', 'G']

        rs_rail = routesttn[routesttn['교통수단유형'].isin(rail_types)].copy()

        # 정류장 명칭 → 노선명 매핑
        station_to_lines = defaultdict(set)
        for _, row in rs_rail[['정류장 명칭', '노선명(short)']].drop_duplicates().iterrows():
            station_to_lines[row['정류장 명칭']].add(row['노선명(short)'])

        # 노선 간 연결 그래프 (환승역 공유)
        self.line_graph = defaultdict(set)
        self.transfer_stations = defaultdict(set)  # (line_a, line_b) -> set of station names

        for station, lines in station_to_lines.items():
            if len(lines) >= 2:
                for a, b in combinations(lines, 2):
                    self.line_graph[a].add(b)
                    self.line_graph[b].add(a)
                    self.transfer_stations[(a, b)].add(station)
                    self.transfer_stations[(b, a)].add(station)

        # 모든 노선 쌍에 대해 최소 환승 횟수 + 경로 사전 계산
        all_lines = list(set(rs_rail['노선명(short)'].unique()))
        self._min_transfer_cache = {}
        self._path_cache = {}  # (src, dst) -> [src, ..., dst] 노선 경로
        for src in all_lines:
            self._bfs_all(src, all_lines)

        n_lines = len(all_lines)
        n_transfers = sum(1 for s, l in station_to_lines.items() if len(l) >= 2)
        print(f"  SubwayTransferGraph: {n_lines} lines, {n_transfers} transfer stations")

    def _bfs_all(self, src: str, all_lines: List[str]):
        """BFS로 src에서 모든 노선까지의 최소 환승 횟수 + 경로 계산"""
        self._min_transfer_cache[(src, src)] = 0
        self._path_cache[(src, src)] = [src]
        visited = {src}
        prev = {src: None}
        queue = deque([(src, 0)])
        while queue:
            curr, dist = queue.popleft()
            for neighbor in self.line_graph[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    prev[neighbor] = curr
                    self._min_transfer_cache[(src, neighbor)] = dist + 1
                    # 경로 복원
                    path = []
                    node = neighbor
                    while node is not None:
                        path.append(node)
                        node = prev[node]
                    self._path_cache[(src, neighbor)] = path[::-1]
                    queue.append((neighbor, dist + 1))

    def min_transfers(self, src_line: str, dst_line: str) -> int:
        """
        두 노선 간 최소 환승 횟수 반환

        Returns:
            환승 횟수 (같은 노선이면 0, 연결 불가면 -1)
        """
        if src_line == dst_line:
            return 0
        return self._min_transfer_cache.get((src_line, dst_line), -1)

    def get_transfer_station_names(self, src_line: str, dst_line: str) -> List[str]:
        """
        두 노선 간 숨겨진 환승역 명칭 리스트 반환

        BFS 최단 경로의 연속 노선 쌍마다 공유 환승역 중 하나를 선택.
        예: 수인분당선→2호선 경로가 [수인분당선, 2호선]이면 → [선릉]
            A→B→C 경로면 → [A-B 환승역, B-C 환승역]

        Returns:
            환승역 명칭 리스트 (같은 노선이면 빈 리스트, 연결 불가면 빈 리스트)
        """
        if src_line == dst_line:
            return []
        path = self._path_cache.get((src_line, dst_line))
        if path is None:
            return []

        transfer_names = []
        for i in range(len(path) - 1):
            shared = self.transfer_stations.get((path[i], path[i + 1]), set())
            if shared:
                transfer_names.append(sorted(shared)[0])  # 알파벳순 첫 번째
        return transfer_names

    def calc_hidden_transfers_series(self, o_lines: pd.Series, d_lines: pd.Series,
                                     o_types: pd.Series, d_types: pd.Series,
                                     rail_types: Optional[List[str]] = None
                                     ) -> Tuple[pd.Series, pd.Series]:
        """
        TCD 레코드별 숨겨진 환승 횟수 + 환승역 명칭 벡터 계산

        승차/하차 모두 지하철(T/G)이고 노선이 다른 경우에만 계산.

        Args:
            o_lines: 승차노선명 Series
            d_lines: 하차노선명 Series
            o_types: 승차교통수단유형 Series
            d_types: 하차교통수단유형 Series

        Returns:
            (숨겨진환승횟수 Series, 숨겨진환승역 Series[list of str])
        """
        if rail_types is None:
            rail_types = ['T', 'G']

        counts = pd.Series(0, index=o_lines.index)
        stations = pd.Series([[] for _ in range(len(o_lines))], index=o_lines.index)

        # 승차/하차 모두 지하철이고 노선이 다른 경우만 계산
        mask = (
            o_types.isin(rail_types) &
            d_types.isin(rail_types) &
            (o_lines != d_lines) &
            o_lines.notna() &
            d_lines.notna()
        )

        if mask.any():
            transfer_counts = []
            transfer_stations = []
            for o, d in zip(o_lines[mask], d_lines[mask]):
                transfer_counts.append(self.min_transfers(o, d))
                transfer_stations.append(self.get_transfer_station_names(o, d))

            counts.loc[mask] = transfer_counts
            stations.loc[mask] = transfer_stations

        return counts, stations


class TCDLoader:
    """TCD, ROUT, ROUTESTTN 데이터 로더 (Route 기반)"""

    def __init__(self, base_path: str = r'C:\Folder\Research\0. DATA\tcd_2025_parquet'):
        """
        Args:
            base_path: 데이터 기본 경로 (날짜 폴더 상위)
        """
        self.base_path = base_path

    def load_data(self, date: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        특정 날짜의 TCD, ROUT, ROUTESTTN 데이터 로드

        Args:
            date: 날짜 문자열 (예: '20250217')

        Returns:
            (tcd_df, rout_df, routesttn_df) 튜플
        """
        path = os.path.join(self.base_path, date)

        tcd = pd.read_parquet(os.path.join(path, f'TCD_{date}.parquet'))
        route = pd.read_parquet(os.path.join(path, f'ROUTE_{date}.parquet'))
        routesttn = pd.read_parquet(os.path.join(path, f'ROUTESTTN_{date}.parquet'))

        return tcd, route, routesttn


class TCDPreprocessor:
    """TCD 데이터 전처리 클래스 (Route 기반)"""

    @staticmethod
    def preprocess_tcd(tcd: pd.DataFrame) -> pd.DataFrame:
        """
        TCD 데이터 전처리
        - 불필요한 컬럼 제거
        - 컬럼명 변경
        - 타입 변환
        """
        # 제거할 컬럼 (없으면 무시)
        drop_cols = ['차량ID(국토부표준)', '노선ID(국토부표준)',
                     '승차정류장ID(국토부표준)', '하차정류장ID(국토부표준)']
        drop_cols = [c for c in drop_cols if c in tcd.columns]

        tcd_pre = tcd.drop(columns=drop_cols, errors='ignore')

        # 컬럼명 변경
        rename_map = {
            '승차정류장ID(정산사업자)': '승차정류장ID',
            '하차정류장ID(정산사업자)': '하차정류장ID',
            '노선ID(정산사업자)': '노선ID',
        }
        tcd_pre = tcd_pre.rename(columns={k: v for k, v in rename_map.items() if k in tcd_pre.columns})

        # 타입 변환
        for col in ['하차정류장ID', '하차일시']:
            if col in tcd_pre.columns:
                tcd_pre[col] = tcd_pre[col].astype('Int64')

        if '교통수단코드' in tcd_pre.columns:
            tcd_pre['교통수단코드'] = tcd_pre['교통수단코드'].astype(str)
        if '노선ID' in tcd_pre.columns:
            tcd_pre['노선ID'] = tcd_pre['노선ID'].astype(str)

        tcd_pre['운행일자'] = tcd_pre['운행일자'].astype(str)
        tcd_pre['승차정류장ID'] = tcd_pre['승차정류장ID'].astype(str)
        tcd_pre['하차정류장ID'] = tcd_pre['하차정류장ID'].astype(str)

        return tcd_pre

    @staticmethod
    def preprocess_routesttn(routesttn: pd.DataFrame) -> pd.DataFrame:
        """
        ROUTESTTN (노선별 정류장) 데이터 전처리
        """
        drop_cols = ['정류장 ARS번호']
        drop_cols = [c for c in drop_cols if c in routesttn.columns]
        rs_pre = routesttn.drop(columns=drop_cols, errors='ignore')

        rs_pre['정류장 ID'] = pd.to_numeric(rs_pre['정류장 ID'], errors='coerce').fillna(0).astype('int64')

        # 노선ID + 정류장ID 기준 중복 제거 (같은 노선에서 같은 정류장이 여러 번 나올 수 있음 - 첫번째만)
        rs_pre = rs_pre.drop_duplicates(subset=['운행일자', '노선ID', '정류장 ID'], keep='first')

        rs_pre['운행일자'] = rs_pre['운행일자'].astype(str)
        rs_pre['노선ID'] = rs_pre['노선ID'].astype(str)
        rs_pre['정류장 ID'] = rs_pre['정류장 ID'].astype(str)

        return rs_pre

    @staticmethod
    def preprocess_rout(rout: pd.DataFrame) -> pd.DataFrame:
        """
        ROUT (노선) 데이터 전처리
        """
        rout_pre = rout.copy()
        rout_pre['운행일자'] = rout_pre['운행일자'].astype(str)
        rout_pre['노선ID'] = rout_pre['노선ID'].astype(str)

        return rout_pre

    @staticmethod
    def merge_tcd_routesttn(tcd: pd.DataFrame, routesttn: pd.DataFrame) -> pd.DataFrame:
        """
        TCD와 ROUTESTTN 병합 (노선ID + 정류장ID 기반 승차/하차 정류장 정보 추가)

        지하철 무환승 환승 대응:
        - 1차: 노선ID + 정류장ID로 매칭 (버스/같은 노선 지하철)
        - 2차: 1차 미매칭 시, 정류장ID로 ROUTESTTN 역조회 + 지하철(T/G) 필터
               (예: 수인분당선 승차 → 2호선 하차, 하차정류장이 다른 노선에 있는 경우)
        """
        merge_cols = ['운행일자', '노선ID', '정류장 ID', '정류장 명칭',
                      '정류장 X 좌표', '정류장 Y 좌표',
                      '교통수단유형', '노선명(short)', '정류장순서', '누적거리(m)']
        merge_cols = [c for c in merge_cols if c in routesttn.columns]

        # 지하철 역조회용 테이블 (정류장ID 기준, 지하철/GTX만)
        rail_types = ['T', 'G']
        rs_rail = routesttn[routesttn['교통수단유형'].isin(rail_types)].copy()
        fallback_cols = [c for c in merge_cols if c != '노선ID']
        # 정류장ID 기준 중복 제거 (같은 정류장이 여러 노선에 있을 수 있으나 지하철은 대부분 1:1)
        rs_rail_dedup = rs_rail.drop_duplicates(subset=['운행일자', '정류장 ID'], keep='first')

        # ── 승차 정류장 매칭 ──
        # 1차: 노선ID + 승차정류장ID
        merged = tcd.merge(
            routesttn[merge_cols],
            left_on=['운행일자', '노선ID', '승차정류장ID'],
            right_on=['운행일자', '노선ID', '정류장 ID'],
            how='left'
        )

        rename_o = {
            '정류장 명칭': '승차정류장 명칭',
            '정류장 X 좌표': '승차정류장 X 좌표',
            '정류장 Y 좌표': '승차정류장 Y 좌표',
            '교통수단유형': '승차교통수단유형',
            '노선명(short)': '승차노선명',
            '정류장순서': '승차정류장순서',
            '누적거리(m)': '승차누적거리',
        }
        rename_o = {k: v for k, v in rename_o.items() if k in merged.columns}
        merged = merged.rename(columns=rename_o)
        merged = merged.drop(columns=['정류장 ID'], errors='ignore')

        # 2차 fallback: 승차 미매칭 → 정류장ID로 지하철 역조회
        o_na = merged['승차정류장 X 좌표'].isna()
        if o_na.any():
            fb = merged.loc[o_na].drop(
                columns=[v for v in rename_o.values() if v in merged.columns], errors='ignore'
            ).merge(
                rs_rail_dedup[fallback_cols],
                left_on=['운행일자', '승차정류장ID'],
                right_on=['운행일자', '정류장 ID'],
                how='left'
            )
            fb = fb.rename(columns=rename_o)
            fb = fb.drop(columns=['정류장 ID'], errors='ignore')

            for col in rename_o.values():
                if col in fb.columns:
                    merged.loc[o_na, col] = fb[col].values

        # ── 하차 정류장 매칭 ──
        # 1차: 노선ID + 하차정류장ID
        merged = merged.merge(
            routesttn[merge_cols],
            left_on=['운행일자', '노선ID', '하차정류장ID'],
            right_on=['운행일자', '노선ID', '정류장 ID'],
            how='left'
        )

        rename_d = {
            '정류장 명칭': '하차정류장 명칭',
            '정류장 X 좌표': '하차정류장 X 좌표',
            '정류장 Y 좌표': '하차정류장 Y 좌표',
            '교통수단유형': '하차교통수단유형',
            '노선명(short)': '하차노선명',
            '정류장순서': '하차정류장순서',
            '누적거리(m)': '하차누적거리',
        }
        rename_d = {k: v for k, v in rename_d.items() if k in merged.columns}
        merged = merged.rename(columns=rename_d)
        merged = merged.drop(columns=['정류장 ID'], errors='ignore')

        # 2차 fallback: 하차 미매칭 → 정류장ID로 지하철 역조회
        d_na = merged['하차정류장 X 좌표'].isna()
        if d_na.any():
            fb = merged.loc[d_na].drop(
                columns=[v for v in rename_d.values() if v in merged.columns], errors='ignore'
            ).merge(
                rs_rail_dedup[fallback_cols],
                left_on=['운행일자', '하차정류장ID'],
                right_on=['운행일자', '정류장 ID'],
                how='left'
            )
            fb = fb.rename(columns=rename_d)
            fb = fb.drop(columns=['정류장 ID'], errors='ignore')

            for col in rename_d.values():
                if col in fb.columns:
                    merged.loc[d_na, col] = fb[col].values

        return merged

    @staticmethod
    def create_trip_id(df: pd.DataFrame) -> pd.DataFrame:
        """
        통행 구분 코드 생성
        """
        df = df.copy()
        df['가상카드번호'] = df['가상카드번호'].astype(str)
        df['트랜잭션ID'] = df['트랜잭션ID'].astype(str)
        df['환승횟수_재계산'] = df.groupby(['가상카드번호', '트랜잭션ID']).cumcount()
        df['구분코드'] = df['가상카드번호'] + '_' + df['트랜잭션ID']

        return df

    @staticmethod
    def filter_gtx_origin(
        df: pd.DataFrame,
        sig_path: str,
        ctprvn_path: str,
        o_lon_col: str = 'o_lon',
        o_lat_col: str = 'o_lat',
        gtx_cities: Optional[List[str]] = None,
        encoding: str = 'cp949',
        crs_epsg: int = 4326,
        predicate: str = 'within',
        keep_boundary_cols: bool = False
    ) -> pd.DataFrame:
        """
        GTX 영향권 지역 필터링 (출발지 기준)

        Args:
            df: 입력 DataFrame
            sig_path: 시군구 SHP 파일 경로
            ctprvn_path: 시도 SHP 파일 경로
            o_lon_col, o_lat_col: 출발지 좌표 컬럼명
            gtx_cities: 필터링할 지역 목록 (None이면 기본값)
            encoding: SHP 파일 인코딩
            crs_epsg: 좌표계
            predicate: 공간 조인 방식
            keep_boundary_cols: 경계 정보 컬럼 유지 여부

        Returns:
            필터링된 DataFrame
        """
        if gtx_cities is None:
            gtx_cities = [
                "서울특별시",
                "파주시", "고양시", "양주시", "의정부시", "남양주시",
                "성남시", "용인시", "화성시", "수원시",
                "과천시", "안양시", "군포시", "의왕시",
                "인천광역시", "부천시"
            ]

        # SHP 파일 로드
        sig = gpd.read_file(sig_path, encoding=encoding).to_crs(epsg=crs_epsg)
        ctprvn = gpd.read_file(ctprvn_path, encoding=encoding).to_crs(epsg=crs_epsg)

        # 지역 필터링
        pattern = "|".join(map(str, gtx_cities))

        gtx_sig = sig[sig["SIG_KOR_NM"].astype(str).str.contains(pattern, na=False, regex=True)].copy()
        gtx_ctprvn = ctprvn[ctprvn["CTP_KOR_NM"].astype(str).str.contains(pattern, na=False, regex=True)].copy()

        # 컬럼명 통일
        gtx_ctprvn = gtx_ctprvn.rename(columns={
            "CTPRVN_CD": "cd", "CTP_ENG_NM": "eng_nm", "CTP_KOR_NM": "kor_nm",
        })
        gtx_sig = gtx_sig.rename(columns={
            "SIG_CD": "cd", "SIG_ENG_NM": "eng_nm", "SIG_KOR_NM": "kor_nm",
        })

        gtx_combined = pd.concat([gtx_ctprvn, gtx_sig], ignore_index=True)
        gtx_combined = gpd.GeoDataFrame(gtx_combined, geometry="geometry", crs=f"EPSG:{crs_epsg}")

        # 포인트 생성 및 공간 조인
        geometry = [Point(xy) for xy in zip(df[o_lon_col], df[o_lat_col])]
        df_gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=f"EPSG:{crs_epsg}")

        filtered = gpd.sjoin(df_gdf, gtx_combined, how="inner", predicate=predicate)

        # 불필요한 컬럼 제거
        drop_cols = ["geometry", "index_right"]
        if not keep_boundary_cols:
            drop_cols += ["cd", "eng_nm", "kor_nm"]
        drop_cols = [c for c in drop_cols if c in filtered.columns]
        filtered = filtered.drop(columns=drop_cols).reset_index(drop=True)

        return filtered



class TCDtoTCNConverter:
    """TCD를 TCN으로 변환하는 클래스 (Route 기반)"""

    def __init__(self, split_round_trip: bool = True, max_transfer_dist: int = 1000,
                 fix_coords: bool = True, min_od_distance: Optional[float] = 500,
                 transfer_graph: Optional[SubwayTransferGraph] = None):
        """
        Args:
            split_round_trip: 왕복 통행 분리 여부
            max_transfer_dist: 왕복 분리 시 최대 환승 거리
            fix_coords: 뒤바뀐 좌표 자동 수정 여부
            min_od_distance: 최소 OD 거리 (미터), None이면 필터링 안 함
            transfer_graph: 지하철 환승 그래프 (None이면 숨겨진 환승 계산 안 함)
        """
        self.split_round_trip = split_round_trip
        self.max_transfer_dist = max_transfer_dist
        self.fix_coords = fix_coords
        self.min_od_distance = min_od_distance
        self.transfer_graph = transfer_graph

    def convert(self, tcd: pd.DataFrame) -> pd.DataFrame:
        """
        TCD DataFrame을 TCN으로 변환

        병렬 리스트 형식으로 저장:
        - 교통수단코드: list
        - 노선ID: list
        - 노선명: list
        - 교통수단유형: list
        - 이용거리: list (각 leg별)
        - 탑승시간: list (각 leg별)
        - 정류장시퀀스: list (승차→환승→하차)

        Args:
            tcd: 전처리된 TCD DataFrame

        Returns:
            TCN DataFrame
        """
        df = tcd.copy()

        # TCD 레코드별 숨겨진 환승 횟수 + 환승역 계산 (지하철 노선 간 무환승 환승)
        if (self.transfer_graph is not None
                and '승차노선명' in df.columns and '하차노선명' in df.columns
                and '승차교통수단유형' in df.columns and '하차교통수단유형' in df.columns):
            df['숨겨진환승횟수'], df['숨겨진환승역'] = self.transfer_graph.calc_hidden_transfers_series(
                df['승차노선명'], df['하차노선명'],
                df['승차교통수단유형'], df['하차교통수단유형']
            )
        else:
            df['숨겨진환승횟수'] = 0
            df['숨겨진환승역'] = [[] for _ in range(len(df))]

        # 왕복 통행 분리 여부에 따라 그룹 키 결정
        if self.split_round_trip:
            df = split_round_trips(df, self.max_transfer_dist)
            df['group_key'] = df['구분코드'] + '_' + df['sub_trip'].astype(str)
        else:
            # 왕복 분리 없이 원래 구분코드 사용
            df['group_key'] = df['구분코드']

        # 정류장ID/명칭을 문자열로 변환 (시퀀스 생성용)
        df['승차정류장ID_str'] = df['승차정류장ID'].astype(str)
        df['하차정류장ID_str'] = df['하차정류장ID'].astype(str)
        df['승차정류장명_str'] = df['승차정류장 명칭'].astype(str) if '승차정류장 명칭' in df.columns else df['승차정류장ID_str']
        df['하차정류장명_str'] = df['하차정류장 명칭'].astype(str) if '하차정류장 명칭' in df.columns else df['하차정류장ID_str']

        # 좌표 시퀀스 생성용 (모든 하차 좌표를 리스트로 수집)
        df['_하차lat'] = df['하차정류장 X 좌표']
        df['_하차lon'] = df['하차정류장 Y 좌표']

        # 그룹별 집계
        agg_dict = {
            # 기본 정보
            '승차정류장ID': 'first',
            '하차정류장ID': 'last',
            '승차정류장 명칭': 'first',
            '하차정류장 명칭': 'last',

            # 좌표
            '승차정류장 X 좌표': 'first',
            '승차정류장 Y 좌표': 'first',
            '하차정류장 X 좌표': 'last',
            '하차정류장 Y 좌표': 'last',

            # 시간
            '승차일시': 'first',
            '하차일시': 'last',

            # 병렬 리스트 (leg별 정보)
            '교통수단코드': list,
            '노선ID': list,
            '이용거리': list,
            '탑승시간': list,

            # 정류장 시퀀스용 (첫 승차 + 모든 하차)
            '승차정류장ID_str': 'first',
            '하차정류장ID_str': list,
            '승차정류장명_str': 'first',
            '하차정류장명_str': list,

            # 좌표 시퀀스용 (모든 하차 좌표 리스트)
            '_하차lat': list,
            '_하차lon': list,

            # 숨겨진 환승 (leg별)
            '숨겨진환승횟수': 'sum',
            '숨겨진환승역': list,  # list of lists
        }

        # Route 기반 추가 컬럼 (존재하는 경우만)
        optional_list_cols = ['승차노선명', '승차교통수단유형', '하차노선명', '하차교통수단유형']
        for col in optional_list_cols:
            if col in df.columns:
                agg_dict[col] = list

        optional_first_cols = ['승차정류장순서', '승차누적거리']
        for col in optional_first_cols:
            if col in df.columns:
                agg_dict[col] = 'first'

        optional_last_cols = ['하차정류장순서', '하차누적거리']
        for col in optional_last_cols:
            if col in df.columns:
                agg_dict[col] = 'last'

        tcn = df.groupby('group_key').agg(agg_dict).reset_index()

        # 노선명 리스트 컬럼 정리
        if '승차노선명' in tcn.columns:
            tcn = tcn.rename(columns={'승차노선명': '노선명'})
        if '승차교통수단유형' in tcn.columns:
            tcn = tcn.rename(columns={'승차교통수단유형': '교통수단유형'})

        # 정류장 시퀀스 생성 (첫 승차 + 모든 하차)
        tcn['정류장시퀀스'] = tcn.apply(
            lambda row: [row['승차정류장ID_str']] + row['하차정류장ID_str'],
            axis=1
        )

        # 정류장 명칭 시퀀스 생성 (숨겨진 환승역 포함)
        # 각 leg별: [승차명칭, 숨겨진환승역..., 하차명칭]
        def build_name_seq(row):
            seq = [row['승차정류장명_str']]
            hidden_list = row['숨겨진환승역']  # list of lists
            alighting_list = row['하차정류장명_str']  # list
            for i, dest_name in enumerate(alighting_list):
                # i번째 leg의 숨겨진 환승역 삽입
                if i < len(hidden_list) and hidden_list[i]:
                    seq.extend(hidden_list[i])
                seq.append(dest_name)
            return seq

        tcn['정류장명칭시퀀스'] = tcn.apply(build_name_seq, axis=1)

        # 정류장 좌표 시퀀스 생성 (정류장명칭시퀀스와 1:1 대응, 숨겨진환승역 포함)
        # 숨겨진 환승역 좌표 룩업 (TCD 승/하차 정류장에서 구축)
        _o = df[['승차정류장명_str', '승차정류장 X 좌표', '승차정류장 Y 좌표']].drop_duplicates('승차정류장명_str')
        _d = df[['하차정류장명_str', '하차정류장 X 좌표', '하차정류장 Y 좌표']].drop_duplicates('하차정류장명_str')
        _stn_coords = {}
        for _, r in _o.iterrows():
            if pd.notna(r['승차정류장 X 좌표']):
                _stn_coords[r['승차정류장명_str']] = (r['승차정류장 X 좌표'], r['승차정류장 Y 좌표'])
        for _, r in _d.iterrows():
            if r['하차정류장명_str'] not in _stn_coords and pd.notna(r['하차정류장 X 좌표']):
                _stn_coords[r['하차정류장명_str']] = (r['하차정류장 X 좌표'], r['하차정류장 Y 좌표'])

        def build_coord_seq(row):
            lat_seq = [row['승차정류장 X 좌표']]
            lon_seq = [row['승차정류장 Y 좌표']]
            hidden_list = row['숨겨진환승역']  # list of lists (before flatten)
            a_lats = row['_하차lat']
            a_lons = row['_하차lon']
            for i, (a_lat, a_lon) in enumerate(zip(a_lats, a_lons)):
                if i < len(hidden_list) and hidden_list[i]:
                    for h_name in hidden_list[i]:
                        h = _stn_coords.get(h_name)
                        if h:
                            lat_seq.append(h[0])
                            lon_seq.append(h[1])
                lat_seq.append(a_lat)
                lon_seq.append(a_lon)
            return pd.Series([lat_seq, lon_seq])

        tcn[['정류장lat시퀀스', '정류장lon시퀀스']] = tcn.apply(build_coord_seq, axis=1)

        # 숨겨진환승역 flatten (list of lists → single list)
        tcn['숨겨진환승역'] = tcn['숨겨진환승역'].apply(
            lambda lol: [s for sub in lol for s in sub] if lol else []
        )

        # 임시 컬럼 제거
        tcn = tcn.drop(columns=['승차정류장ID_str', '하차정류장ID_str',
                                '승차정류장명_str', '하차정류장명_str',
                                '_하차lat', '_하차lon'])

        # 환승 횟수 계산
        tcn['명시적환승횟수'] = tcn['교통수단코드'].str.len() - 1
        tcn['환승횟수'] = tcn['명시적환승횟수'] + tcn['숨겨진환승횟수']

        # 총 이용거리 및 탑승시간 (벡터화)
        tcn['총이용거리'] = tcn['이용거리'].apply(lambda x: sum(v for v in x if pd.notna(v)))
        tcn['총탑승시간'] = tcn['탑승시간'].apply(lambda x: sum(v for v in x if pd.notna(v)))

        # 노선 구간거리 계산 (ROUTESTTN 누적거리 기반)
        if '승차누적거리' in tcn.columns and '하차누적거리' in tcn.columns:
            tcn['노선구간거리'] = (tcn['하차누적거리'] - tcn['승차누적거리']).abs()

        # 교통수단 카테고리 추가
        if '교통수단유형' in tcn.columns:
            # ROUTESTTN 교통수단유형 (B/T/G) → mode_set → 7개 카테고리
            tcn['transport_category'] = tcn['교통수단유형'].apply(
                lambda types: _mode_set_to_category(
                    {_TYPE_TO_MODE.get(t, 'unknown') for t in types}
                )
            )

        # OD pair 키 추가
        tcn['od_pair'] = (
            tcn['승차정류장ID'].astype(str) + '_' +
            tcn['하차정류장ID'].astype(str)
        )

        # 좌표 수정 (뒤바뀐 위경도)
        if self.fix_coords:
            tcn = fix_swapped_coords_tcn(tcn)

        # 짧은 거리 통행 필터링
        if self.min_od_distance is not None:
            tcn = filter_short_trips(tcn, self.min_od_distance)

        return tcn


def process_date(date: str,
                 base_path: str = r'C:\Folder\Research\0. DATA\tcd_2025_parquet',
                 output_path: Optional[str] = None,
                 split_round_trip: bool = True,
                 sig_path: Optional[str] = None,
                 ctprvn_path: Optional[str] = None) -> pd.DataFrame:
    """
    특정 날짜의 TCD를 TCN으로 변환 (Route 기반)

    Args:
        date: 날짜 문자열 (예: '20250217')
        base_path: 데이터 기본 경로
        output_path: 출력 경로 (None이면 저장 안 함)
        split_round_trip: 왕복 통행 분리 여부
        sig_path: 시군구 SHP 파일 경로 (None이면 지역 필터링 안 함)
        ctprvn_path: 시도 SHP 파일 경로 (None이면 지역 필터링 안 함)

    Returns:
        TCN DataFrame
    """
    import gc
    print(f"Processing {date} (route-based)...")

    # 1. 데이터 로드
    loader = TCDLoader(base_path)
    tcd, route, routesttn = loader.load_data(date)
    print(f"  Loaded TCD: {len(tcd):,} records")
    print(f"  Loaded ROUT: {len(route):,} routes")
    print(f"  Loaded ROUTESTTN: {len(routesttn):,} route-station records")
    del route; gc.collect()

    # 2. 전처리
    preprocessor = TCDPreprocessor()

    # 승하차 동일 제거
    mask_same_stop = tcd["승차정류장ID(정산사업자)"] == tcd["하차정류장ID(정산사업자)"]
    invalid_trips = tcd.loc[mask_same_stop, ["가상카드번호", "트랜잭션ID"]].drop_duplicates()

    tcd = tcd.merge(
        invalid_trips, on=["가상카드번호", "트랜잭션ID"],
        how="left", indicator=True
    ).query("_merge == 'left_only'").drop(columns="_merge")
    del invalid_trips, mask_same_stop; gc.collect()
    print(f"  After removing same O-D: {len(tcd):,} records")

    # 정류장ID NA인 통행 제거 (승차 또는 하차 정류장ID가 NA면 해당 통행 전체 제거)
    sttn_cols = ["승차정류장ID(정산사업자)", "하차정류장ID(정산사업자)"]
    mask_na_sttn = tcd[sttn_cols].isna().any(axis=1)
    invalid_na_sttn = tcd.loc[mask_na_sttn, ["가상카드번호", "트랜잭션ID"]].drop_duplicates()

    tcd = tcd.merge(
        invalid_na_sttn, on=["가상카드번호", "트랜잭션ID"],
        how="left", indicator=True
    ).query("_merge == 'left_only'").drop(columns="_merge")
    print(f"  After removing NA station ID: {len(tcd):,} records (removed {len(invalid_na_sttn):,} trips)")
    del invalid_na_sttn, mask_na_sttn; gc.collect()

    # TCD 전처리
    tcd = preprocessor.preprocess_tcd(tcd)
    gc.collect()

    # ROUTESTTN 전처리 및 병합 — 환승 그래프는 원본 routesttn으로 먼저 생성
    transfer_graph = SubwayTransferGraph(routesttn)
    routesttn_pre = preprocessor.preprocess_routesttn(routesttn)
    del routesttn; gc.collect()

    tcd = preprocessor.merge_tcd_routesttn(tcd, routesttn_pre)
    del routesttn_pre; gc.collect()
    tcd = preprocessor.create_trip_id(tcd)

    # 좌표 매칭 현황 출력
    coord_cols = ['승차정류장 X 좌표', '승차정류장 Y 좌표', '하차정류장 X 좌표', '하차정류장 Y 좌표']
    o_matched = tcd['승차정류장 X 좌표'].notna().sum()
    d_matched = tcd['하차정류장 X 좌표'].notna().sum()
    print(f"  Route-station match: origin {o_matched:,}/{len(tcd):,} ({o_matched/len(tcd)*100:.1f}%), "
          f"dest {d_matched:,}/{len(tcd):,} ({d_matched/len(tcd)*100:.1f}%)")

    # 좌표 0/NA인 통행 제거 (좌표값이 0이거나 NA이면 해당 통행 전체 제거)
    mask_bad = (tcd[coord_cols] == 0).any(axis=1) | tcd[coord_cols].isna().any(axis=1)
    invalid_bad = tcd.loc[mask_bad, ["가상카드번호", "트랜잭션ID"]].drop_duplicates()

    tcd = tcd.merge(
        invalid_bad, on=["가상카드번호", "트랜잭션ID"],
        how="left", indicator=True
    ).query("_merge == 'left_only'").drop(columns="_merge")
    print(f"  After removing bad coords: {len(tcd):,} records (removed {len(invalid_bad):,} trips)")
    del invalid_bad, mask_bad; gc.collect()

    # 3. TCN 변환
    converter = TCDtoTCNConverter(
        split_round_trip=split_round_trip,
        transfer_graph=transfer_graph
    )
    tcn = converter.convert(tcd)
    del tcd, converter, transfer_graph; gc.collect()

    # NaN 제거
    tcn = tcn.dropna().reset_index(drop=True)
    print(f"  Final TCN: {len(tcn):,} trips")

    # 지역 필터링 (옵션)
    if sig_path and ctprvn_path:
        tcn = TCDPreprocessor.filter_gtx_origin(
            tcn,
            sig_path=sig_path,
            ctprvn_path=ctprvn_path,
            o_lon_col='승차정류장 Y 좌표',
            o_lat_col='승차정류장 X 좌표'
        )
        print(f"  After region filter: {len(tcn):,}")

    # 4. 저장
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tcn.to_parquet(output_path)
        print(f"  Saved to {output_path}")

    return tcn


def process_multiple_dates(dates: List[str],
                           base_path: str = r'C:\Folder\Research\0. DATA\tcd_2025_parquet',
                           output_dir: Optional[str] = None,
                           split_round_trip: bool = False,
                           sig_path: Optional[str] = None,
                           ctprvn_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    여러 날짜의 TCD를 TCN으로 변환 (Route 기반)

    Args:
        dates: 날짜 문자열 리스트
        base_path: 데이터 기본 경로
        output_dir: 출력 디렉토리 (None이면 저장 안 함)
        split_round_trip: 왕복 통행 분리 여부
        sig_path: 시군구 SHP 파일 경로 (None이면 지역 필터링 안 함)
        ctprvn_path: 시도 SHP 파일 경로 (None이면 지역 필터링 안 함)

    Returns:
        {date: TCN DataFrame} 딕셔너리
    """
    results = {}

    for date in dates:
        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, date, f'TCN_{date}_route.parquet')

        tcn = process_date(date, base_path, output_path, split_round_trip, sig_path, ctprvn_path)

        if output_path:
            # 저장 완료 시 메모리 해제, 경로만 보관
            results[date] = output_path
            del tcn
        else:
            results[date] = tcn

    return results
