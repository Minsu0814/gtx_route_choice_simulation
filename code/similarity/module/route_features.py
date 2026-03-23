# -*- coding: utf-8 -*-
"""
Route Choice Model 피처 추출 모듈

OTP itinerary에서 경로 선택 모델(MNL/Mixed Logit/NN) 학습용 피처를 추출한다.
- 시간 분해: total_duration, in_vehicle, walk, wait, access, egress, transfer_walk
- 거리: total, bus, subway, gtx, walk
- 구조: num_transfers, num_legs, transport_category, has_bus/train/gtx
- 비용: fare (모드+거리+버스유형 기반), generalized_cost (OTP)
- 컨텍스트: od_distance, departure_hour, departure_dow, is_peak
"""

import csv
import math
import os
import re
import pandas as pd

from .similarity import (
    OTP_MODE_MAP,
    GTX_ROUTE_KEYWORDS,
    _normalize_route_name,
    _mode_set_to_category,
    _haversine,
)


# ============================================================
# 버스 세부 유형 상수
# ============================================================
BUS_TRUNK = 'BUS_TRUNK'           # 간선 (Blue)       — 1,500원
BUS_BRANCH = 'BUS_BRANCH'         # 지선 (Green)      — 1,500원
BUS_CIRCULAR = 'BUS_CIRCULAR'     # 순환간선 (Yellow)  — 1,400원
BUS_VILLAGE = 'BUS_VILLAGE'       # 마을               — 1,200원
BUS_NIGHT = 'BUS_NIGHT'           # 심야 (N)          — 2,500원
BUS_EXPRESS = 'BUS_EXPRESS'       # 광역급행 (M-bus)   — 3,000원
BUS_AIRPORT = 'BUS_AIRPORT'       # 공항리무진 (6xxx)  — 노선별 고정요금
BUS_INTERCITY = 'BUS_INTERCITY'   # 시외               — 2,500원
BUS_DEFAULT = 'BUS_TRUNK'         # 분류 불가 시 기본

# KTDB route_type → 대중교통 유형
# 0: 시내/농어촌/마을, 1: 도시철도/경전철, 2: 해운, 3: 시외버스,
# 4: 일반철도, 5: 공항리무진버스, 6: 고속철도, 7: 항공, 8: GTX
KTDB_ROUTE_TYPE_BUS = {0, 3, 5}

# 모듈 레벨 GTFS route_type 룩업 (선택적 로드)
_ROUTE_TYPE_MAP = {}   # {route_id: route_type_int}


# ============================================================
# Polyline 거리 복원 (distance=0인 transit leg 보정)
# ============================================================
def _decode_polyline(encoded):
    """Google Encoded Polyline -> [(lat, lon), ...]"""
    coords = []
    index, lat, lng = 0, 0, 0
    while index < len(encoded):
        for is_lat in [True, False]:
            shift, result = 0, 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if is_lat:
                lat += delta
            else:
                lng += delta
        coords.append((lat / 1e5, lng / 1e5))
    return coords


def _polyline_distance_m(encoded):
    """Encoded polyline -> 총 거리 (미터)"""
    if not encoded:
        return 0.0
    coords = _decode_polyline(encoded)
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(coords)):
        total += float(_haversine(
            coords[i - 1][0], coords[i - 1][1],
            coords[i][0], coords[i][1]
        ))
    return total


def fix_missing_distances(itinerary):
    """
    distance=0인 transit leg의 거리를 legGeometry polyline에서 복원.
    원본 itinerary를 in-place 수정한다.

    Args:
        itinerary: OTP itinerary dict

    Returns:
        itinerary (수정된 원본)
    """
    for leg in itinerary.get('legs', []):
        dist = leg.get('distance')
        if dist is not None:
            dist = float(dist)
        if (dist is None or dist == 0) and leg.get('mode') != 'WALK':
            geom = (leg.get('legGeometry') or {}).get('points', '')
            if geom:
                leg['distance'] = round(_polyline_distance_m(geom), 1)
    return itinerary


# ============================================================
# GTFS route_type 룩업 로드
# ============================================================
def load_bus_type_map(routes_txt_path):
    """
    GTFS routes.txt를 읽어 모듈 레벨 route_type 룩업을 초기화한다.

    Args:
        routes_txt_path: routes.txt 파일 경로

    호출 예:
        load_bus_type_map('data_sample/gtfs/a1/routes.txt')
    """
    global _ROUTE_TYPE_MAP
    with open(routes_txt_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        _ROUTE_TYPE_MAP = {
            row['route_id']: int(row['route_type'])
            for row in reader
        }


# ============================================================
# 버스 세부 유형 분류
# ============================================================
# 마을버스 route_id 패턴: 서울 BR_1100_10X9... (4번째 자리 = 9)
_RE_SEOUL_VILLAGE = re.compile(r'^BR_\d{4}_\d{3}9')
# 심야버스 shortName: N + 숫자
_RE_NIGHT_BUS = re.compile(r'^N\d')
# M-버스 (경기 광역급행): M + 숫자
_RE_M_BUS = re.compile(r'^M\d')


def _classify_bus_subtype(leg):
    """
    OTP BUS leg → 버스 세부 유형 분류.

    분류 우선순위:
    1. GTFS route_type (3=시외, 5=광역/공항리무진)
    2. shortName 패턴 (N##=심야, M##=광역급행)
    3. route_id 패턴 (마을버스)
    4. shortName 자릿수 (간선 3자리, 지선 4자리, 순환 0x)

    Args:
        leg: OTP leg dict (mode='BUS')

    Returns:
        str: BUS_TRUNK / BUS_BRANCH / BUS_CIRCULAR / BUS_VILLAGE /
             BUS_NIGHT / BUS_EXPRESS / BUS_INTERCITY
    """
    route_info = leg.get('route') or {}
    gtfs_id = route_info.get('gtfsId', '')
    short_name = _normalize_route_name(route_info.get('shortName', ''))

    # OTP feed prefix 제거 ("1:BR_..." → "BR_...")
    route_id = gtfs_id.split(':', 1)[-1] if ':' in gtfs_id else gtfs_id

    # 1) GTFS route_type 룩업
    route_type = _ROUTE_TYPE_MAP.get(route_id)
    if route_type == 3:
        return BUS_INTERCITY
    if route_type == 5:
        return BUS_AIRPORT

    # 2) shortName 패턴
    sn = short_name or ''
    if _RE_NIGHT_BUS.match(sn):
        return BUS_NIGHT
    if _RE_M_BUS.match(sn):
        return BUS_EXPRESS

    # 3) route_id 패턴 — 마을버스 (서울)
    if _RE_SEOUL_VILLAGE.match(route_id):
        return BUS_VILLAGE

    # 4) shortName 자릿수 → 간선/지선/순환
    digits = ''.join(c for c in sn if c.isdigit())
    if sn.startswith('0') and len(digits) <= 3:
        return BUS_CIRCULAR
    if len(digits) >= 4:
        return BUS_BRANCH
    if len(digits) == 3:
        return BUS_TRUNK

    return BUS_DEFAULT


# ============================================================
# 요금 설정 (성인 카드 기준, 2023.08~)
# ============================================================
FARE_CFG = {
    "base_fare": {
        # 버스 유형별
        BUS_TRUNK: 1500,        # 간선
        BUS_BRANCH: 1500,       # 지선
        BUS_CIRCULAR: 1400,     # 순환간선
        BUS_VILLAGE: 1200,      # 마을
        BUS_NIGHT: 2500,        # 심야
        BUS_EXPRESS: 3000,      # 광역급행 (M-bus)
        BUS_AIRPORT: 7000,      # 공항리무진 (노선별 5,000~16,000원, 대표값)
        BUS_INTERCITY: 2500,    # 시외
        # 하위호환
        "BUS": 1500,
        # 철도
        "SUBWAY": 1550,
        "GTX": 3200,
    },
    "integrated": {
        "base_is_max": True,
        "base_km": 10,
        "block_km": 5,
        "block_won": 100,       # 성인, GTX 미포함
        "block_won_gtx": 250,   # 성인, GTX 포함
    },
    "bus_rule": {
        # 간선/지선/순환/마을 공통
        "free_km": 10,
        "block_km": 5,
        "block_won": 100,
    },
    "bus_express_rule": {
        # 광역버스 단독
        "free_km": 30,
        "block_km": 5,
        "block_won": 100,
    },
    "subway_rule": {
        "free_km": 10,
        "mid_to_km": 50,
        "mid_block_km": 5,
        "mid_block_won": 100,
        "long_block_km": 8,
        "long_block_won": 100,
    },
    "gtx_rule": {
        "free_km": 10,
        "block_km": 5,
        "block_won": 250,
    },
}


# ============================================================
# 요금 계산 내부 함수
# ============================================================
def _fare_integrated(bus_km, sub_km, gtx_km, bus_subtype=BUS_TRUNK):
    """
    통합환승요금: 기본요금 = 이용 수단 중 최고 기본요금, 10km 초과 5km당 가산.

    버스 유형별 기본요금 차이를 반영한다 (마을 1,200 / 광역 3,000 등).
    """
    cfg = FARE_CFG
    fare_keys = []
    if bus_km > 0:
        fare_keys.append(bus_subtype)
    if sub_km > 0:
        fare_keys.append("SUBWAY")
    if gtx_km > 0:
        fare_keys.append("GTX")

    base = (max(cfg["base_fare"].get(k, 1500) for k in fare_keys)
            if cfg["integrated"]["base_is_max"] and fare_keys
            else cfg["base_fare"]["SUBWAY"])
    d_over = max(0.0, bus_km + sub_km + gtx_km - cfg["integrated"]["base_km"])
    blocks = int(d_over // cfg["integrated"]["block_km"])
    per_block = (cfg["integrated"]["block_won_gtx"] if "GTX" in fare_keys
                 else cfg["integrated"]["block_won"])
    return int(base + blocks * per_block)


def _fare_bus_only(bus_km, bus_subtype=BUS_TRUNK):
    """
    버스 단독 요금: 유형별 기본요금 + 거리비례 가산.

    - 간선/지선/순환/마을/심야/시외: 10km 초과 5km당 100원
    - 광역급행(M-bus): 30km 초과 5km당 100원
    - 공항리무진: 고정요금 (거리비례 없음, 통합환승 미적용)
    """
    cfg = FARE_CFG
    base = cfg["base_fare"].get(bus_subtype, 1500)

    # 공항리무진: 노선별 고정요금, 거리 가산 없음
    if bus_subtype == BUS_AIRPORT:
        return int(base)

    if bus_subtype == BUS_EXPRESS:
        r = cfg["bus_express_rule"]
    else:
        r = cfg["bus_rule"]

    d = max(0.0, bus_km)
    if d <= r["free_km"]:
        return int(base)
    return int(base + int((d - r["free_km"]) // r["block_km"]) * r["block_won"])


def _fare_subway_only(sub_km):
    """전철 단독: 10km 무료 / 10~50km 5km당 100원 / 50km 초과 8km당 100원"""
    cfg = FARE_CFG
    base = cfg["base_fare"]["SUBWAY"]
    r = cfg["subway_rule"]
    d = max(0.0, sub_km)
    if d <= r["free_km"]:
        return int(base)
    if d <= r["mid_to_km"]:
        return int(base + int((d - r["free_km"]) // r["mid_block_km"]) * r["mid_block_won"])
    mid_b = int((r["mid_to_km"] - r["free_km"]) // r["mid_block_km"])
    long_b = int((d - r["mid_to_km"]) // r["long_block_km"])
    return int(base + mid_b * r["mid_block_won"] + long_b * r["long_block_won"])


def _fare_gtx_only(gtx_km):
    """GTX 단독: 10km 초과분 5km당 250원"""
    cfg = FARE_CFG
    base = cfg["base_fare"]["GTX"]
    r = cfg["gtx_rule"]
    d = max(0.0, gtx_km)
    if d <= r["free_km"]:
        return int(base)
    return int(base + int((d - r["free_km"]) // r["block_km"]) * r["block_won"])


# ============================================================
# Leg 모드 분류
# ============================================================
def _classify_leg(leg):
    """OTP leg -> fare_mode/internal_mode/bus_subtype/duration/distance 분류 dict"""
    mode = leg.get('mode', '')
    duration = float(leg.get('duration', 0) or 0)
    distance = float(leg.get('distance', 0) or 0)
    route_info = leg.get('route') or {}
    route_name = _normalize_route_name(route_info.get('shortName', ''))

    if mode == 'WALK':
        return {
            'fare_mode': 'WALK', 'internal_mode': 'walk',
            'bus_subtype': None,
            'duration': duration, 'distance': distance,
            'route_name': '', 'is_transit': False,
        }

    is_gtx = any(kw in (route_name or '') for kw in GTX_ROUTE_KEYWORDS)
    bus_subtype = None
    if is_gtx:
        fare_mode, internal_mode = 'GTX', 'gtx'
    elif mode == 'BUS':
        fare_mode, internal_mode = 'BUS', 'bus'
        bus_subtype = _classify_bus_subtype(leg)
    else:  # SUBWAY, RAIL, TRAM
        fare_mode, internal_mode = 'SUBWAY', 'train'

    return {
        'fare_mode': fare_mode, 'internal_mode': internal_mode,
        'bus_subtype': bus_subtype,
        'duration': duration, 'distance': distance,
        'route_name': route_name or '', 'is_transit': True,
    }


# ============================================================
# 핵심 함수 1: OTP 대안경로 피처 추출
# ============================================================
def extract_itinerary_features(itinerary):
    """
    OTP itinerary -> 경로 선택 모델용 피처 dict (21개)

    Args:
        itinerary: OTP itinerary dict

    Returns:
        dict with 21 route choice features
    """
    legs = itinerary.get('legs', [])
    classified = [_classify_leg(leg) for leg in legs]
    transit_idx = [i for i, c in enumerate(classified) if c['is_transit']]

    # --- 시간 분해 ---
    total_duration = float(itinerary.get('duration', 0) or 0)
    in_vehicle_time = sum(c['duration'] for c in classified if c['is_transit'])
    walk_time = sum(c['duration'] for c in classified if not c['is_transit'])
    wait_time = max(0, total_duration - in_vehicle_time - walk_time)

    access_time = egress_time = transfer_walk_time = 0
    if transit_idx:
        first_t, last_t = transit_idx[0], transit_idx[-1]
        access_time = sum(
            c['duration'] for c in classified[:first_t] if not c['is_transit']
        )
        egress_time = sum(
            c['duration'] for c in classified[last_t + 1:] if not c['is_transit']
        )
        transfer_walk_time = sum(
            c['duration'] for c in classified[first_t + 1:last_t]
            if not c['is_transit']
        )
    else:
        access_time = walk_time

    # --- 거리 ---
    walk_distance = float(itinerary.get('walkDistance') or 0) or sum(
        c['distance'] for c in classified if not c['is_transit']
    )
    total_distance = sum(c['distance'] for c in classified)
    bus_distance = sum(c['distance'] for c in classified if c['fare_mode'] == 'BUS')
    subway_distance = sum(c['distance'] for c in classified if c['fare_mode'] == 'SUBWAY')
    gtx_distance = sum(c['distance'] for c in classified if c['fare_mode'] == 'GTX')

    # --- 구조 ---
    num_transfers = max(0, len(transit_idx) - 1)
    num_legs = len(transit_idx)
    mode_set = {c['internal_mode'] for c in classified if c['is_transit']}
    transport_category = _mode_set_to_category(mode_set)

    # 주 노선 (최장 transit leg)
    main_route = ''
    if transit_idx:
        longest = max((classified[i] for i in transit_idx), key=lambda c: c['duration'])
        main_route = longest['route_name']

    # 버스 유형 (가장 비싼 기본요금의 subtype)
    bus_subtypes = [c['bus_subtype'] for c in classified if c['bus_subtype']]
    if bus_subtypes:
        bus_subtype = max(
            bus_subtypes,
            key=lambda s: FARE_CFG["base_fare"].get(s, 0),
        )
    else:
        bus_subtype = None

    # --- 요금 ---
    fare = calc_fare_from_itinerary(itinerary)
    generalized_cost = float(itinerary.get('generalizedCost', 0) or 0)

    return {
        'total_duration': total_duration,
        'in_vehicle_time': in_vehicle_time,
        'walk_time': walk_time,
        'wait_time': wait_time,
        'access_time': access_time,
        'egress_time': egress_time,
        'transfer_walk_time': transfer_walk_time,
        'walk_distance': round(walk_distance, 1),
        'total_distance': round(total_distance, 1),
        'bus_distance': round(bus_distance, 1),
        'subway_distance': round(subway_distance, 1),
        'gtx_distance': round(gtx_distance, 1),
        'num_transfers': num_transfers,
        'num_legs': num_legs,
        'fare': fare,
        'generalized_cost': generalized_cost,
        'transport_category': transport_category,
        'has_bus': int('bus' in mode_set),
        'has_train': int('train' in mode_set),
        'has_gtx': int('gtx' in mode_set),
        'main_route': main_route,
        'bus_subtype': bus_subtype,
    }


# ============================================================
# 핵심 함수 2: SC 통행 컨텍스트 피처 추출
# ============================================================
def extract_trip_context(sc_row):
    """
    스마트카드 통행 row -> 시공간 컨텍스트 피처 dict (5개)

    Args:
        sc_row: TCN DataFrame row (pandas Series)

    Returns:
        dict with 5 context features
    """
    # OD 직선거리 (좌표 기반 계산)
    od_distance = 0.0
    o_lat = sc_row.get('승차정류장 X 좌표')
    o_lon = sc_row.get('승차정류장 Y 좌표')
    d_lat = sc_row.get('하차정류장 X 좌표')
    d_lon = sc_row.get('하차정류장 Y 좌표')
    if all(v is not None and not _is_nan(v) for v in [o_lat, o_lon, d_lat, d_lon]):
        od_distance = float(
            _haversine(float(o_lat), float(o_lon), float(d_lat), float(d_lon))
        )

    # 출발시각
    departure = sc_row.get('승차일시')
    departure_hour = 0
    departure_dow = 0
    is_peak = 0
    if departure is not None and not _is_nan(departure):
        if isinstance(departure, str):
            departure = pd.to_datetime(departure)
        if hasattr(departure, 'hour'):
            departure_hour = int(departure.hour)
            departure_dow = int(departure.dayofweek)
            is_peak = int(7 <= departure_hour <= 9 or 17 <= departure_hour <= 19)

    # SC transport category
    sc_transport_category = str(sc_row.get('transport_category', 'unknown'))

    return {
        'od_distance': round(od_distance, 1),
        'departure_hour': departure_hour,
        'departure_dow': departure_dow,
        'is_peak': is_peak,
        'sc_transport_category': sc_transport_category,
    }


def _is_nan(v):
    """NaN 체크 (숫자가 아닌 경우 False 반환)"""
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


# ============================================================
# 핵심 함수 3: 요금 계산
# ============================================================
def calc_fare_from_itinerary(itinerary):
    """
    OTP itinerary -> 요금 (원)

    모드별 거리를 합산하여 통합/단독 요금 규칙을 적용한다.
    버스 유형(간선/지선/마을/심야/광역 등)에 따라 기본요금이 다르다.
    공항리무진은 통합환승 미적용 → 별도 고정요금 합산.

    Args:
        itinerary: OTP itinerary dict

    Returns:
        int: 요금 (원)
    """
    legs = itinerary.get('legs', [])
    bus_d = sub_d = gtx_d = 0.0
    airport_fare = 0  # 공항리무진 별도 합산
    # 통합환승 대상 버스 중 가장 비싼 기본요금의 유형을 대표로 사용
    best_bus_subtype = BUS_DEFAULT
    best_bus_base = 0

    for leg in legs:
        mode = leg.get('mode', '')
        if mode == 'WALK':
            continue
        distance = float(leg.get('distance', 0) or 0)
        route_info = leg.get('route') or {}
        route_name = _normalize_route_name(route_info.get('shortName', ''))

        is_gtx = any(kw in (route_name or '') for kw in GTX_ROUTE_KEYWORDS)
        if is_gtx:
            gtx_d += distance
        elif mode == 'BUS':
            subtype = _classify_bus_subtype(leg)
            if subtype == BUS_AIRPORT:
                # 공항리무진: 통합환승 미적용, 고정요금 별도
                airport_fare += FARE_CFG["base_fare"][BUS_AIRPORT]
            else:
                bus_d += distance
                sub_base = FARE_CFG["base_fare"].get(subtype, 1500)
                if sub_base > best_bus_base:
                    best_bus_base = sub_base
                    best_bus_subtype = subtype
        else:  # SUBWAY, RAIL, TRAM
            sub_d += distance

    bus_km, sub_km, gtx_km = bus_d / 1000, sub_d / 1000, gtx_d / 1000
    has_bus, has_sub, has_gtx = bus_km > 0, sub_km > 0, gtx_km > 0
    used = sum([has_bus, has_sub, has_gtx])

    # 통합환승 대상 수단 요금 계산
    if used >= 2:
        integrated = _fare_integrated(bus_km, sub_km, gtx_km, best_bus_subtype)
    elif has_gtx:
        integrated = _fare_gtx_only(gtx_km)
    elif has_sub:
        integrated = _fare_subway_only(sub_km)
    elif has_bus:
        integrated = _fare_bus_only(bus_km, best_bus_subtype)
    else:
        integrated = 0

    return integrated + airport_fare
