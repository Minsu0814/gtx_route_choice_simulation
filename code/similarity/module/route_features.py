# -*- coding: utf-8 -*-
"""
Route Choice Model 피처 추출 모듈

OTP itinerary에서 경로 선택 모델(MNL/Mixed Logit/NN) 학습용 피처를 추출한다.
- 시간 분해: total_duration, in_vehicle, walk, wait, access, egress, transfer_walk
- 거리: total, bus, subway, gtx, walk
- 구조: num_transfers, num_legs, transport_category, has_bus/train/gtx
- 비용: fare (모드+거리 기반), generalized_cost (OTP)
- 컨텍스트: od_distance, departure_hour, departure_dow, is_peak
"""

import math
import pandas as pd

from .similarity import (
    OTP_MODE_MAP,
    GTX_ROUTE_KEYWORDS,
    _normalize_route_name,
    _mode_set_to_category,
    _haversine,
)


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
# 요금 설정 (성인 기본) — fare_calc.py 요금 정책 적용
# ============================================================
FARE_CFG = {
    "base_fare": {"BUS": 1500, "SUBWAY": 1550, "GTX": 3200},
    "integrated": {
        "base_is_max": True,
        "base_km": 10,
        "block_km": 5,
        "block_won": 100,       # 성인, GTX 미포함
        "block_won_gtx": 250,   # 성인, GTX 포함
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
def _fare_integrated(bus_km, sub_km, gtx_km):
    """BUS/SUBWAY/GTX 통합요금: 기본요금=최고 기본요금, 10km 초과 5km당 가산"""
    cfg = FARE_CFG
    used = [m for m, k in [("BUS", bus_km), ("SUBWAY", sub_km), ("GTX", gtx_km)] if k > 0]
    base = (max(cfg["base_fare"][m] for m in used)
            if cfg["integrated"]["base_is_max"]
            else cfg["base_fare"]["SUBWAY"])
    d_over = max(0.0, bus_km + sub_km + gtx_km - cfg["integrated"]["base_km"])
    blocks = int(d_over // cfg["integrated"]["block_km"])
    per_block = cfg["integrated"]["block_won_gtx"] if "GTX" in used else cfg["integrated"]["block_won"]
    return int(base + blocks * per_block)


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
    """OTP leg -> fare_mode/internal_mode/duration/distance 분류 dict"""
    mode = leg.get('mode', '')
    duration = float(leg.get('duration', 0) or 0)
    distance = float(leg.get('distance', 0) or 0)
    route_info = leg.get('route') or {}
    route_name = _normalize_route_name(route_info.get('shortName', ''))

    if mode == 'WALK':
        return {
            'fare_mode': 'WALK', 'internal_mode': 'walk',
            'duration': duration, 'distance': distance,
            'route_name': '', 'is_transit': False,
        }

    is_gtx = any(kw in (route_name or '') for kw in GTX_ROUTE_KEYWORDS)
    if is_gtx:
        fare_mode, internal_mode = 'GTX', 'gtx'
    elif mode == 'BUS':
        fare_mode, internal_mode = 'BUS', 'bus'
    else:  # SUBWAY, RAIL, TRAM
        fare_mode, internal_mode = 'SUBWAY', 'train'

    return {
        'fare_mode': fare_mode, 'internal_mode': internal_mode,
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
    GTX 판별: route shortName에 GTX/290 키워드 포함 여부

    Args:
        itinerary: OTP itinerary dict

    Returns:
        int: 요금 (원)
    """
    legs = itinerary.get('legs', [])
    bus_d = sub_d = gtx_d = 0.0

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
            bus_d += distance
        else:  # SUBWAY, RAIL, TRAM
            sub_d += distance

    bus_km, sub_km, gtx_km = bus_d / 1000, sub_d / 1000, gtx_d / 1000
    has_bus, has_sub, has_gtx = bus_km > 0, sub_km > 0, gtx_km > 0
    used = sum([has_bus, has_sub, has_gtx])

    if used >= 2:
        return _fare_integrated(bus_km, sub_km, gtx_km)
    if has_gtx:
        return _fare_gtx_only(gtx_km)
    if has_sub:
        return _fare_subway_only(sub_km)
    if has_bus:
        return FARE_CFG['base_fare']['BUS']
    return 0
