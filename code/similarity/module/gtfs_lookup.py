# -*- coding: utf-8 -*-
"""
GTFS 기반 경로 정류장 시퀀스 룩업 모듈

GTFS 데이터를 로드하여 노선명 → 전체 정류장 시퀀스(좌표 포함)
룩업을 빌드한다. OTP/스마트카드 경로의 중간 정류장 확장에 사용.

주요 기능:
- stop_times.txt (2000만 줄) 대응: 대표 trip_id 1개만 선택 + chunked 읽기
- parquet 캐싱으로 재실행 시 수 초 이내 로드
- 정류장명 정규화 (괄호 내용 제거) 로 TCN/OTP 명칭 매칭
- 방향 자동 선택 (from→to 순서가 맞는 방향)
"""

import os
import re
import json
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict


def _normalize_stop_name(name: str) -> str:
    """
    정류장명 정규화: 괄호 내용 제거 + 공백 제거

    예: '교대(법원·검찰청)' → '교대'
        '서울대입구(관악구청)' → '서울대입구'
        '왕십리(성동구청)' → '왕십리'
        '강변(동서울터미널)' → '강변'
    """
    if not name:
        return name
    # 괄호와 그 내용 제거 (중첩 괄호 포함)
    normalized = re.sub(r'\([^)]*\)', '', name).strip()
    return normalized if normalized else name


class GTFSRouteLookup:
    """
    GTFS 데이터 기반 노선별 정류장 시퀀스 룩업

    Usage:
        lookup = GTFSRouteLookup('data/gtfs/a1')
        stops = lookup.expand_route('2호선', '강남', '서울대입구')
        # → [('강남', 37.497, 127.027), ('역삼', ...), ..., ('서울대입구', ...)]
    """

    def __init__(self, gtfs_dir: str, cache_path: Optional[str] = None,
                 shape_cache_path: Optional[str] = None):
        """
        Args:
            gtfs_dir: GTFS 파일 디렉토리 (stops.txt, routes.txt, trips.txt, stop_times.txt)
            cache_path: 캐시 parquet 경로 (None이면 gtfs_dir/route_stops_cache.parquet)
            shape_cache_path: shape 캐시 경로 (None이면 gtfs_dir/shape_polylines_cache.parquet)
        """
        self.gtfs_dir = gtfs_dir
        if cache_path is None:
            cache_path = os.path.join(gtfs_dir, 'route_stops_cache.parquet')
        self.cache_path = cache_path
        if shape_cache_path is None:
            shape_cache_path = os.path.join(gtfs_dir, 'shape_polylines_cache.parquet')
        self.shape_cache_path = shape_cache_path

        # 로드
        self._stops_df = self._load_stops()
        self._routes_df = self._load_routes()
        self._route_stops_cache = self._load_or_build_route_stops()

        # 룩업 빌드
        self._stop_id_to_info = self._build_stop_id_lookup()
        self._route_name_to_ids = self._build_route_name_lookup()
        self._route_id_to_stops = self._build_route_stops_lookup()

        # Shape polyline 캐시 로드/빌드
        self._shape_cache = self._load_or_build_shape_polylines()
        self._route_id_to_shape = self._build_route_shape_lookup()

        n_routes = len(self._route_id_to_stops)
        n_stops = len(self._stop_id_to_info)
        n_shapes = len(self._route_id_to_shape)
        print(f"GTFSRouteLookup: {n_routes} routes, {n_stops} stops, {n_shapes} shapes loaded")

    def _load_stops(self) -> pd.DataFrame:
        """stops.txt 로드"""
        path = os.path.join(self.gtfs_dir, 'stops.txt')
        return pd.read_csv(path, dtype={'stop_id': str})

    def _load_routes(self) -> pd.DataFrame:
        """routes.txt 로드"""
        path = os.path.join(self.gtfs_dir, 'routes.txt')
        return pd.read_csv(path, dtype={'route_id': str})

    def _load_or_build_route_stops(self) -> pd.DataFrame:
        """
        route_stops 캐시 로드 또는 빌드

        캐시가 있으면 바로 로드, 없으면 trips.txt + stop_times.txt에서 빌드 후 캐싱.
        """
        if os.path.exists(self.cache_path):
            print(f"  Loading cached route stops from {self.cache_path}")
            return pd.read_parquet(self.cache_path)

        print(f"  Building route stops cache (this may take a few minutes)...")
        cache_df = self._build_route_stops_from_gtfs()

        # 캐시 저장
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        cache_df.to_parquet(self.cache_path, index=False)
        print(f"  Cached to {self.cache_path} ({len(cache_df):,} rows)")

        return cache_df

    def _build_route_stops_from_gtfs(self) -> pd.DataFrame:
        """
        trips.txt + stop_times.txt에서 route_id별 대표 trip의 정류장 시퀀스 빌드

        전략:
        1. trips.txt에서 route_id별 대표 trip_id 1개 선택 (첫 번째)
        2. stop_times.txt를 chunked 읽기, 대표 trip_id만 필터링
        """
        # 1. trips.txt 로드 → route_id별 대표 trip_id
        trips_path = os.path.join(self.gtfs_dir, 'trips.txt')
        trips_df = pd.read_csv(trips_path, dtype={'route_id': str, 'trip_id': str})
        representative_trips = trips_df.groupby('route_id')['trip_id'].first().reset_index()
        rep_trip_ids = set(representative_trips['trip_id'].values)
        route_for_trip = dict(zip(representative_trips['trip_id'],
                                  representative_trips['route_id']))
        print(f"    Representative trips: {len(rep_trip_ids):,} (from {len(trips_df):,} total)")

        # 2. stop_times.txt chunked 읽기
        st_path = os.path.join(self.gtfs_dir, 'stop_times.txt')
        chunks = []
        chunk_size = 500_000

        for chunk in pd.read_csv(st_path, chunksize=chunk_size,
                                  dtype={'trip_id': str, 'stop_id': str,
                                         'stop_sequence': int}):
            filtered = chunk[chunk['trip_id'].isin(rep_trip_ids)]
            if len(filtered) > 0:
                chunks.append(filtered[['trip_id', 'stop_id', 'stop_sequence']])

        if not chunks:
            return pd.DataFrame(columns=['route_id', 'stop_id', 'stop_sequence'])

        st_filtered = pd.concat(chunks, ignore_index=True)
        st_filtered['route_id'] = st_filtered['trip_id'].map(route_for_trip)
        st_filtered = st_filtered.sort_values(['route_id', 'stop_sequence'])

        result = st_filtered[['route_id', 'stop_id', 'stop_sequence']].copy()
        print(f"    Route stops built: {len(result):,} rows")

        return result

    # ==============================================================
    # Shape polyline 캐시 빌드/로드
    # ==============================================================
    def _load_or_build_shape_polylines(self) -> pd.DataFrame:
        """
        shape polyline 캐시 로드 또는 빌드

        캐시 구조: shape_id, route_id, coords_json
        coords_json = JSON string of [(lat, lon), ...]
        """
        if os.path.exists(self.shape_cache_path):
            print(f"  Loading cached shape polylines from {self.shape_cache_path}")
            return pd.read_parquet(self.shape_cache_path)

        shapes_path = os.path.join(self.gtfs_dir, 'shapes.txt')
        if not os.path.exists(shapes_path):
            print(f"  shapes.txt not found, skipping shape polyline cache")
            return pd.DataFrame(columns=['shape_id', 'route_id', 'coords_json'])

        print(f"  Building shape polyline cache (this may take a few minutes)...")
        cache_df = self._build_shape_polylines_from_gtfs()

        os.makedirs(os.path.dirname(self.shape_cache_path), exist_ok=True)
        cache_df.to_parquet(self.shape_cache_path, index=False)
        print(f"  Cached to {self.shape_cache_path} ({len(cache_df):,} rows)")

        return cache_df

    def _build_shape_polylines_from_gtfs(self) -> pd.DataFrame:
        """
        trips.txt에서 route_id별 대표 trip의 shape_id 추출 후
        shapes.txt를 chunked 읽기하여 대표 shape만 필터 → polyline 구성
        """
        # 1. trips.txt → route_id별 대표 trip의 shape_id
        trips_path = os.path.join(self.gtfs_dir, 'trips.txt')
        trips_df = pd.read_csv(trips_path,
                                dtype={'route_id': str, 'trip_id': str, 'shape_id': str})
        representative = trips_df.groupby('route_id').first().reset_index()
        shape_to_route = dict(zip(representative['shape_id'], representative['route_id']))
        rep_shape_ids = set(representative['shape_id'].dropna().values)
        print(f"    Representative shapes: {len(rep_shape_ids):,}")

        # 2. shapes.txt chunked 읽기, 대표 shape_id만 필터
        shapes_path = os.path.join(self.gtfs_dir, 'shapes.txt')
        chunk_size = 500_000
        chunks = []

        for chunk in pd.read_csv(shapes_path, chunksize=chunk_size,
                                  dtype={'shape_id': str,
                                         'shape_pt_lat': float,
                                         'shape_pt_lon': float,
                                         'shape_pt_sequence': int}):
            filtered = chunk[chunk['shape_id'].isin(rep_shape_ids)]
            if len(filtered) > 0:
                chunks.append(filtered[['shape_id', 'shape_pt_lat',
                                        'shape_pt_lon', 'shape_pt_sequence']])

        if not chunks:
            return pd.DataFrame(columns=['shape_id', 'route_id', 'coords_json'])

        shapes_filtered = pd.concat(chunks, ignore_index=True)
        print(f"    Filtered shape points: {len(shapes_filtered):,}")

        # 3. shape_id별 정렬 → polyline 구성
        rows = []
        for shape_id, group in shapes_filtered.groupby('shape_id'):
            sorted_pts = group.sort_values('shape_pt_sequence')
            coords = list(zip(sorted_pts['shape_pt_lat'].values,
                              sorted_pts['shape_pt_lon'].values))
            route_id = shape_to_route.get(shape_id, '')
            rows.append({
                'shape_id': shape_id,
                'route_id': route_id,
                'coords_json': json.dumps(coords),
            })

        result = pd.DataFrame(rows)
        print(f"    Shape polylines built: {len(result):,}")
        return result

    def _build_route_shape_lookup(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        route_id → shape polyline [(lat, lon), ...] 룩업 빌드
        """
        lookup: Dict[str, List[Tuple[float, float]]] = {}
        for _, row in self._shape_cache.iterrows():
            route_id = row['route_id']
            coords = json.loads(row['coords_json'])
            # 튜플로 변환
            lookup[route_id] = [(float(lat), float(lon)) for lat, lon in coords]
        return lookup

    def _build_stop_id_lookup(self) -> Dict[str, Tuple[str, float, float]]:
        """stop_id → (name, lat, lon) 룩업"""
        lookup = {}
        for _, row in self._stops_df.iterrows():
            lookup[row['stop_id']] = (
                row['stop_name'],
                float(row['stop_lat']),
                float(row['stop_lon']),
            )
        return lookup

    def _build_route_name_lookup(self) -> Dict[str, List[str]]:
        """
        노선명 → route_id(s) 룩업 빌드

        OTP/TCN 노선명 ('2호선', '공항철도') → GTFS route_short_name ('서울2호선', '공항철도')
        """
        lookup: Dict[str, List[str]] = {}
        for _, row in self._routes_df.iterrows():
            route_id = row['route_id']
            short_name = str(row.get('route_short_name', ''))

            if short_name and short_name != 'nan':
                # 원본 이름
                lookup.setdefault(short_name, []).append(route_id)

                # '서울' 접두사 제거 버전 (서울2호선 → 2호선)
                if short_name.startswith('서울') and '호선' in short_name:
                    stripped = short_name[2:]  # '2호선'
                    lookup.setdefault(stripped, []).append(route_id)

        return lookup

    def _build_route_stops_lookup(self) -> Dict[str, List[Tuple[str, str, float, float]]]:
        """
        route_id → 정렬된 정류장 목록 [(stop_id, name, lat, lon), ...] 빌드
        """
        lookup: Dict[str, List[Tuple[str, str, float, float]]] = {}

        for route_id, group in self._route_stops_cache.groupby('route_id'):
            sorted_stops = group.sort_values('stop_sequence')
            stops = []
            for _, row in sorted_stops.iterrows():
                stop_id = row['stop_id']
                info = self._stop_id_to_info.get(stop_id)
                if info:
                    name, lat, lon = info
                    stops.append((stop_id, name, lat, lon))
            if stops:
                lookup[route_id] = stops

        return lookup

    def _match_route_ids(self, route_name: str) -> List[str]:
        """
        TCN/OTP 노선명 → GTFS route_id(s) 매칭

        Args:
            route_name: '2호선', '공항철도', '3호선' 등

        Returns:
            매칭된 route_id 리스트
        """
        if not route_name:
            return []

        # 직접 매칭
        if route_name in self._route_name_to_ids:
            return self._route_name_to_ids[route_name]

        # '서울' 접두사 추가 시도
        prefixed = '서울' + route_name
        if prefixed in self._route_name_to_ids:
            return self._route_name_to_ids[prefixed]

        return []

    def _find_stop_indices(self, route_stops: List[Tuple[str, str, float, float]],
                           stop_name: str) -> List[int]:
        """
        노선 정류장 목록에서 정류장명 매칭 (정규화 기반)

        Args:
            route_stops: [(stop_id, name, lat, lon), ...]
            stop_name: 찾을 정류장명 (예: '교대')

        Returns:
            매칭된 인덱스 리스트
        """
        normalized_target = _normalize_stop_name(stop_name)
        indices = []

        for i, (_, name, _, _) in enumerate(route_stops):
            if _normalize_stop_name(name) == normalized_target:
                indices.append(i)

        return indices

    def expand_route(self, route_name: str, from_stop: str, to_stop: str
                     ) -> List[Tuple[str, float, float]]:
        """
        노선명 + 출발/도착 정류장 → 전체 중간 정류장 시퀀스 (좌표 포함)

        방향 자동 선택: from→to 순서가 맞는 route_id를 선택.
        순환선(2호선)의 경우 내선/외선 중 from→to가 순방향인 것을 선택.

        Args:
            route_name: 노선명 (예: '2호선', '공항철도')
            from_stop: 출발 정류장명 (예: '강남')
            to_stop: 도착 정류장명 (예: '서울대입구')

        Returns:
            [(name, lat, lon), ...] 출발~도착 전체 정류장 (양 끝 포함)
            매칭 실패 시 빈 리스트
        """
        route_ids = self._match_route_ids(route_name)
        if not route_ids:
            return []

        best_result = []
        best_length = float('inf')

        for route_id in route_ids:
            stops = self._route_id_to_stops.get(route_id, [])
            if not stops:
                continue

            from_indices = self._find_stop_indices(stops, from_stop)
            to_indices = self._find_stop_indices(stops, to_stop)

            if not from_indices or not to_indices:
                continue

            # 모든 from→to 조합 중 가장 짧은 유효 구간 선택
            for fi in from_indices:
                for ti in to_indices:
                    if fi == ti:
                        continue

                    if fi < ti:
                        # 순방향: from_index < to_index
                        segment = stops[fi:ti + 1]
                    else:
                        # 순환선 대응: from 이후 끝까지 + 처음부터 to까지
                        # (2호선 내선에서 강남(idx=112) → 시청(idx=134+) 등)
                        segment = stops[fi:] + stops[:ti + 1]

                    if len(segment) < best_length and len(segment) >= 2:
                        best_length = len(segment)
                        best_result = [(name, lat, lon)
                                       for _, name, lat, lon in segment]

        return best_result

    def _nearest_shape_index(self, shape_coords: List[Tuple[float, float]],
                              lat: float, lon: float) -> int:
        """shape polyline에서 주어진 좌표에 가장 가까운 점의 인덱스 반환"""
        best_idx = 0
        best_dist = float('inf')
        for i, (slat, slon) in enumerate(shape_coords):
            # 빠른 유클리드 근사 (lat/lon 소수점 비교, 한국 내 충분)
            d = (slat - lat) ** 2 + (slon - lon) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def expand_route_shape(self, route_name: str, from_stop: str, to_stop: str
                           ) -> List[Tuple[float, float]]:
        """
        노선명 + 출발/도착 정류장 → dense shape polyline 구간 반환

        shapes.txt 기반의 고밀도 좌표 시퀀스를 반환. expand_route()와 동일한
        방향 자동 선택 로직 (여러 route_id/shape 중 from→to 순방향인 것 선택).

        Args:
            route_name: 노선명 (예: '2호선', '공항철도')
            from_stop: 출발 정류장명 (예: '강남')
            to_stop: 도착 정류장명 (예: '서울대입구')

        Returns:
            [(lat, lon), ...] 출발~도착 구간 dense polyline
            매칭 실패 시 빈 리스트
        """
        route_ids = self._match_route_ids(route_name)
        if not route_ids:
            return []

        best_result = []
        best_length = float('inf')

        for route_id in route_ids:
            # shape polyline 확인
            shape_coords = self._route_id_to_shape.get(route_id)
            if not shape_coords or len(shape_coords) < 2:
                continue

            # 정류장 목록에서 from/to 좌표 조회
            stops = self._route_id_to_stops.get(route_id, [])
            if not stops:
                continue

            from_indices = self._find_stop_indices(stops, from_stop)
            to_indices = self._find_stop_indices(stops, to_stop)

            if not from_indices or not to_indices:
                continue

            # from/to 정류장 좌표 → shape에서 nearest point 찾기
            for fi in from_indices:
                for ti in to_indices:
                    if fi == ti:
                        continue

                    _, _, from_lat, from_lon = stops[fi]
                    _, _, to_lat, to_lon = stops[ti]

                    from_shape_idx = self._nearest_shape_index(
                        shape_coords, from_lat, from_lon)
                    to_shape_idx = self._nearest_shape_index(
                        shape_coords, to_lat, to_lon)

                    if from_shape_idx == to_shape_idx:
                        continue

                    if from_shape_idx < to_shape_idx:
                        segment = shape_coords[from_shape_idx:to_shape_idx + 1]
                    else:
                        # 순환선 대응
                        segment = shape_coords[from_shape_idx:] + \
                                  shape_coords[:to_shape_idx + 1]

                    if len(segment) >= 2 and len(segment) < best_length:
                        best_length = len(segment)
                        best_result = segment

        return best_result

    def get_route_all_stops(self, route_name: str) -> Dict[str, List[Tuple[str, float, float]]]:
        """
        노선명의 모든 방향별 전체 정류장 반환 (디버깅/확인용)

        Returns:
            {route_id: [(name, lat, lon), ...]}
        """
        route_ids = self._match_route_ids(route_name)
        result = {}
        for route_id in route_ids:
            stops = self._route_id_to_stops.get(route_id, [])
            result[route_id] = [(name, lat, lon) for _, name, lat, lon in stops]
        return result
