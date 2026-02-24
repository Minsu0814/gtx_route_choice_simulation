# -*- coding: utf-8 -*-
"""
OTP-스마트카드 유사도 비교 모듈
6 Level Similarity Metrics + Composite Score

폴리라인 기반 공간 비교 포함:
- OTP legGeometry polyline → 실제 경로 좌표 (GTFS 불필요)
- GTFS 가용 시: 양방향 Hausdorff 거리 기반 유사도
- GTFS 미가용 시: SC 좌표 → OTP polyline 근접도 (points-on-path)
"""

import numpy as np

# OTP 모드 → 내부 모드 변환
OTP_MODE_MAP = {
    'BUS': 'bus',
    'SUBWAY': 'train',
    'RAIL': 'train',
    'TRAM': 'train',
}

# GTX 노선 식별용 키워드
GTX_ROUTE_KEYWORDS = ['GTX', '290']


def _normalize_route_name(name):
    """노선명 정규화: OTP '서울2호선' ↔ SC '2호선' / SC '5531번(...)' → '5531' 매칭용"""
    if not name:
        return name
    name = str(name).strip()
    # 서울 지하철: OTP는 '서울1호선', SC는 '1호선'
    if name.startswith('서울') and '호선' in name:
        name = name[2:]  # '서울' 제거
    # SC 버스: '5531번(군포동행정복지센터방면)' → '5531'
    if '번(' in name:
        name = name.split('번(')[0]
    # SC 버스: '5531번' (괄호 없는 경우)
    elif name.endswith('번'):
        name = name[:-1]
    return name


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


# ============================================================
# 유틸리티 함수
# ============================================================
def _decode_polyline(encoded):
    """Google Encoded Polyline → [(lat, lon), ...]"""
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


def _stop_name_match(otp_name, sc_name):
    """
    정류장명 fuzzy 매칭.
    OTP: '군포1동행정복지센터.군포역'  SC: '군포역' → True
    '.'으로 분리된 부분 중 하나라도 상대방 이름에 포함되면 일치.
    """
    if otp_name == sc_name:
        return True
    # '.'으로 분리된 부분명 비교
    otp_parts = [p.strip() for p in otp_name.split('.') if p.strip()]
    sc_parts = [p.strip() for p in sc_name.split('.') if p.strip()]
    for op in otp_parts:
        for sp in sc_parts:
            if op == sp or op in sp or sp in op:
                return True
    return False


def _normalize_stops_for_comparison(otp_stops, sc_stops):
    """
    OTP/SC 정류장 시퀀스를 비교 가능한 형태로 정규화.
    OTP 정류장명 중 SC 정류장명과 fuzzy 매칭되면 SC 이름으로 통일.
    """
    # SC 이름 → 정규화된 이름 매핑 (빠른 lookup용)
    normalized_otp = []
    for otp_s in otp_stops:
        matched = False
        for sc_s in sc_stops:
            if _stop_name_match(otp_s, sc_s):
                normalized_otp.append(sc_s)
                matched = True
                break
        if not matched:
            normalized_otp.append(otp_s)
    return normalized_otp, list(sc_stops)


def _lcs_length(seq1, seq2):
    """Longest Common Subsequence 길이 (DP)"""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _levenshtein_distance(seq1, seq2):
    """Levenshtein 편집 거리"""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 삭제
                dp[i][j - 1] + 1,      # 삽입
                dp[i - 1][j - 1] + cost  # 치환
            )
    return dp[m][n]


def _haversine(lat1, lon1, lat2, lon2):
    """두 좌표 사이 거리 (미터)"""
    R = 6_371_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ============================================================
# OTP 응답 파싱
# ============================================================
def parse_otp_itinerary(itinerary, gtfs_lookup=None):
    """
    OTP itinerary → 비교용 딕셔너리

    OTP 응답의 from.name / route.shortName을 직접 사용하여 명칭 기반 비교.
    gtfs_lookup이 주어지면 각 leg의 from→to를 GTFS로 확장하여 full_stop_coords 생성.

    Args:
        itinerary: OTP itinerary dict
        gtfs_lookup: GTFSRouteLookup 인스턴스 (None이면 GTFS 확장 안 함)

    Returns:
        dict with keys: modes, transfer_count, stops, stop_coords,
                        full_stop_coords, total_time, routes, main_route,
                        od_coords, transit_legs
    """
    legs = itinerary.get('legs', [])

    modes = set()
    stops = []
    stop_coords = []
    routes = []
    transit_legs = []
    route_polyline = []  # legGeometry 기반 전체 경로 좌표

    for leg in legs:
        mode = leg.get('mode', '')

        # legGeometry polyline 추출 (WALK 포함 — 전체 경로 형태 복원)
        geom = (leg.get('legGeometry') or {}).get('points', '')
        if geom:
            decoded = _decode_polyline(geom)
            # 이전 leg와 연결점 중복 제거
            if route_polyline and decoded:
                last = route_polyline[-1]
                first = decoded[0]
                if abs(last[0] - first[0]) < 1e-5 and abs(last[1] - first[1]) < 1e-5:
                    decoded = decoded[1:]
            route_polyline.extend(decoded)

        if mode == 'WALK':
            continue

        # 노선 명칭 (OTP shortName → 정규화)
        route_info = leg.get('route') or {}
        route_name = _normalize_route_name(route_info.get('shortName', ''))

        # 모드 분류
        is_gtx = any(kw in route_name for kw in GTX_ROUTE_KEYWORDS)
        if is_gtx:
            modes.add('gtx')
        elif mode in OTP_MODE_MAP:
            modes.add(OTP_MODE_MAP[mode])

        # 정류장 추출 (OTP from.name / to.name 직접 사용)
        from_obj = leg.get('from', {})
        to_obj = leg.get('to', {})

        if from_obj.get('stop'):
            stops.append(from_obj.get('name', ''))
            stop_coords.append((from_obj.get('lat'), from_obj.get('lon')))

        if to_obj.get('stop'):
            stops.append(to_obj.get('name', ''))
            stop_coords.append((to_obj.get('lat'), to_obj.get('lon')))

        # 노선 추출
        if route_name:
            routes.append(route_name)

        transit_legs.append({
            'mode': mode,
            'mapped_mode': 'gtx' if is_gtx else OTP_MODE_MAP.get(mode, mode),
            'route_name': route_name,
            'duration': leg.get('duration', 0),
            'distance': leg.get('distance', 0),
        })

    # 중복 제거 (순서 유지)
    seen = set()
    unique_stops = []
    unique_coords = []
    for s, c in zip(stops, stop_coords):
        if s not in seen:
            seen.add(s)
            unique_stops.append(s)
            unique_coords.append(c)

    # 주 노선: 가장 긴 leg의 노선
    main_route = ''
    if transit_legs:
        longest = max(transit_legs, key=lambda x: x['duration'])
        main_route = longest['route_name']

    # 총 소요시간 (초)
    total_time = itinerary.get('duration', 0)

    # OD 좌표
    od_coords = None
    if legs:
        first = legs[0]
        last = legs[-1]
        od_coords = {
            'o_lat': first['from'].get('lat'),
            'o_lon': first['from'].get('lon'),
            'd_lat': last['to'].get('lat'),
            'd_lon': last['to'].get('lon'),
        }

    # 7개 카테고리 분류
    transport_category = _mode_set_to_category(modes)

    # GTFS 기반 전체 정류장 좌표 확장
    # OTP leg에서 route_name + from.name + to.name → GTFS expand_route
    full_stop_coords = []
    if gtfs_lookup and unique_stops and len(unique_stops) >= 2:
        route_list = [tl['route_name'] for tl in transit_legs]
        for leg_idx in range(len(route_list)):
            from_stop = unique_stops[leg_idx] if leg_idx < len(unique_stops) else None
            to_stop = unique_stops[leg_idx + 1] if (leg_idx + 1) < len(unique_stops) else None
            route_name = route_list[leg_idx] if leg_idx < len(route_list) else None

            if from_stop and to_stop and route_name:
                expanded = gtfs_lookup.expand_route(route_name, from_stop, to_stop)
                if expanded:
                    # 이전 leg와 연결점 중복 제거
                    if full_stop_coords and expanded:
                        last = full_stop_coords[-1]
                        first = expanded[0]
                        if (abs(last[1] - first[1]) < 0.001 and
                                abs(last[2] - first[2]) < 0.001):
                            expanded = expanded[1:]
                    full_stop_coords.extend(expanded)

    return {
        'modes': modes,
        'transport_category': transport_category,
        'transfer_count': max(0, len(transit_legs) - 1),
        'stops': unique_stops,
        'stop_coords': unique_coords,
        'full_stop_coords': full_stop_coords,  # [(name, lat, lon), ...]
        'route_polyline': route_polyline,  # [(lat, lon), ...] from legGeometry
        'total_time': total_time,
        'routes': list(set(routes)),
        'main_route': main_route,
        'od_coords': od_coords,
        'transit_legs': transit_legs,
    }


def _itinerary_route_key(itinerary):
    """
    OTP itinerary의 경로 구조 키 생성 (출발시간 무시).
    모드 + 노선 + 정류장 시퀀스가 같으면 동일 경로로 판단.
    """
    legs = itinerary.get('legs', [])
    parts = []
    for leg in legs:
        mode = leg.get('mode', '')
        if mode == 'WALK':
            continue
        route_info = leg.get('route') or {}
        route_id = route_info.get('gtfsId', '') if isinstance(route_info, dict) else ''
        from_stop = leg.get('from', {}).get('stop') or {}
        to_stop = leg.get('to', {}).get('stop') or {}
        from_id = from_stop.get('gtfsId', '') or from_stop.get('code', '')
        to_id = to_stop.get('gtfsId', '') or to_stop.get('code', '')
        parts.append(f"{mode}|{route_id}|{from_id}>{to_id}")
    return '::'.join(parts)


def deduplicate_itineraries(itineraries):
    """
    OTP itineraries에서 경로 구조가 같은 것을 하나로 합침.
    출발시간만 다른 중복 경로를 제거하고, 소요시간은 평균으로 대표.

    Args:
        itineraries: OTP 응답의 itineraries 리스트

    Returns:
        중복 제거된 itineraries 리스트 (각 항목에 'count' 추가)
    """
    seen = {}  # route_key → (itinerary, durations)

    for itin in itineraries:
        key = _itinerary_route_key(itin)
        # all-walk 경로 제거 (대중교통 leg이 없는 경로)
        if not key:
            continue
        if key in seen:
            seen[key][1].append(itin.get('duration', 0))
        else:
            seen[key] = (itin, [itin.get('duration', 0)])

    deduped = []
    for key, (itin, durations) in seen.items():
        itin_copy = dict(itin)
        itin_copy['duration'] = int(np.mean(durations))  # 평균 소요시간
        itin_copy['_route_key'] = key
        itin_copy['_count'] = len(durations)  # 동일 경로가 몇 개 있었는지
        deduped.append(itin_copy)

    return deduped


def parse_smartcard_trip(trip_row, col_map=None, gtfs_lookup=None):
    """
    TCN DataFrame row → 비교용 딕셔너리

    Args:
        trip_row: TCN DataFrame의 한 행
        col_map: 컬럼명 매핑 (인코딩 깨짐 대비, index 기반 fallback)
        gtfs_lookup: GTFSRouteLookup 인스턴스 (None이면 GTFS 확장 안 함)

    Returns:
        dict with keys: modes, transfer_count, stops, total_time,
                        routes, od_coords, full_stop_coords
    """
    if col_map is None:
        col_map = {}

    # 컬럼 접근 (이름 또는 인덱스)
    def get_col(name, idx=None):
        if name in trip_row.index:
            return trip_row[name]
        if col_map.get(name) and col_map[name] in trip_row.index:
            return trip_row[col_map[name]]
        if idx is not None:
            return trip_row.iloc[idx]
        return None

    # 모드 set: TCN의 transport_category에서 추출
    transport_category = get_col('transport_category')
    cat = str(transport_category) if transport_category else ''
    modes = set()
    if cat and cat not in ('', 'nan', 'unknown'):
        for m in ('bus', 'train', 'gtx'):
            if m in cat:
                modes.add(m)

    # 환승횟수
    transfer_count = get_col('환승횟수', 18)
    if transfer_count is None:
        transfer_count = 0
    transfer_count = int(transfer_count)

    # 정류장명칭시퀀스 (숨겨진 환승역 포함, 명칭 기반 - OTP와 동일 체계로 비교)
    stop_seq = get_col('정류장명칭시퀀스')
    if stop_seq is None:
        # fallback: 기존 정류장시퀀스 (ID 기반)
        stop_seq = get_col('정류장시퀀스', 17)
    stops = []
    if stop_seq is not None:
        if isinstance(stop_seq, str):
            stop_seq = stop_seq.strip("[]'\" ").split("', '")
            if len(stop_seq) == 1:
                stop_seq = stop_seq[0].split()
        stops = [str(s).strip("' ") for s in stop_seq]

    # 총탑승시간 (초)
    total_time = get_col('총탑승시간', 20)
    if total_time is None:
        total_time = 0
    total_time = int(total_time)

    # 노선명 리스트 (명칭 기반 - OTP shortName과 동일 체계로 비교)
    route_names = get_col('노선명')
    if route_names is None:
        # fallback: 노선ID
        route_names = get_col('노선ID', 14)
    routes = []
    if route_names is not None:
        if isinstance(route_names, str):
            route_names = route_names.strip("[]'\" ").split("', '")
            if len(route_names) == 1:
                route_names = route_names[0].split()
        routes = [_normalize_route_name(str(r).strip("' ")) for r in route_names]

    # OD 좌표
    o_lat = get_col('승차정류장 X 좌표', 7)
    o_lon = get_col('승차정류장 Y 좌표', 8)
    d_lat = get_col('하차정류장 X 좌표', 9)
    d_lon = get_col('하차정류장 Y 좌표', 10)

    od_coords = {
        'o_lat': float(o_lat) if o_lat is not None else None,
        'o_lon': float(o_lon) if o_lon is not None else None,
        'd_lat': float(d_lat) if d_lat is not None else None,
        'd_lon': float(d_lon) if d_lon is not None else None,
    }

    # 정류장 좌표 시퀀스 → 중간 환승 정류장 좌표 추출
    # 정류장lat/lon시퀀스 = [origin, alight1, alight2, ...], 중간 = [1:-1]
    stop_coords = []
    lat_seq = get_col('정류장lat시퀀스')
    lon_seq = get_col('정류장lon시퀀스')
    if lat_seq is not None and lon_seq is not None:
        if isinstance(lat_seq, (list, np.ndarray)) and isinstance(lon_seq, (list, np.ndarray)):
            # 중간 정류장만 (origin/destination은 od_coords에서 처리)
            for lat, lon in zip(lat_seq[1:-1], lon_seq[1:-1]):
                if lat is not None and lon is not None:
                    try:
                        stop_coords.append((float(lat), float(lon)))
                    except (TypeError, ValueError):
                        pass

    # GTFS 기반 전체 정류장 좌표 확장
    # TCN 데이터 구조:
    #   stops = ['강남', '교대', '고속터미널']  (origin, alight1, alight2)
    #   routes = ['2호선', '3호선']             (route1, route2)
    #   Leg 0: route='2호선', from='강남', to='교대'
    #   Leg 1: route='3호선', from='교대', to='고속터미널'
    full_stop_coords = []
    if gtfs_lookup and routes and len(stops) >= 2:
        for leg_idx in range(len(routes)):
            from_stop = stops[leg_idx] if leg_idx < len(stops) else None
            to_stop = stops[leg_idx + 1] if (leg_idx + 1) < len(stops) else None
            route_name = routes[leg_idx] if leg_idx < len(routes) else None

            if from_stop and to_stop and route_name:
                expanded = gtfs_lookup.expand_route(route_name, from_stop, to_stop)
                if expanded:
                    # 이전 leg와 연결점 중복 제거
                    if full_stop_coords and expanded:
                        last = full_stop_coords[-1]
                        first = expanded[0]
                        if (abs(last[1] - first[1]) < 0.001 and
                                abs(last[2] - first[2]) < 0.001):
                            expanded = expanded[1:]
                    full_stop_coords.extend(expanded)

    return {
        'modes': modes,
        'transport_category': cat if cat and cat not in ('', 'nan', 'unknown') else _mode_set_to_category(modes),
        'transfer_count': transfer_count,
        'stops': stops,
        'stop_coords': stop_coords,  # [(lat, lon), ...] 중간 환승 정류장 좌표
        'full_stop_coords': full_stop_coords,  # [(name, lat, lon), ...]
        'total_time': total_time,
        'routes': routes,
        'od_coords': od_coords,
    }


# ============================================================
# Level 1: 모드 조합 (Mode Combination) - 가중치 0.15
# ============================================================
def compute_mode_metrics(otp_parsed, sc_parsed):
    """모드 조합 유사도 지표"""
    otp_modes = otp_parsed['modes']
    sc_modes = sc_parsed['modes']

    if not otp_modes and not sc_modes:
        return {'mode_exact': 1.0, 'mode_jaccard': 1.0, 'mode_contains': 1.0}
    if not otp_modes or not sc_modes:
        return {'mode_exact': 0.0, 'mode_jaccard': 0.0, 'mode_contains': 0.0}

    mode_exact = 1.0 if otp_modes == sc_modes else 0.0

    intersection = len(otp_modes & sc_modes)
    union = len(otp_modes | sc_modes)
    mode_jaccard = intersection / union if union > 0 else 0.0

    mode_contains = 1.0 if sc_modes.issubset(otp_modes) else 0.0

    return {
        'mode_exact': mode_exact,
        'mode_jaccard': mode_jaccard,
        'mode_contains': mode_contains,
    }


# ============================================================
# Level 2: 환승 (Transfer) - 가중치 0.15
# ============================================================
def compute_transfer_metrics(otp_parsed, sc_parsed):
    """환승 유사도 지표"""
    otp_t = otp_parsed['transfer_count']
    sc_t = sc_parsed['transfer_count']

    transfer_exact = 1.0 if otp_t == sc_t else 0.0

    max_t = max(otp_t, sc_t, 1)
    transfer_diff = 1.0 - abs(otp_t - sc_t) / max_t

    def cat(t):
        if t == 0: return 0
        if t == 1: return 1
        return 2
    transfer_cat = 1.0 if cat(otp_t) == cat(sc_t) else 0.0

    return {
        'transfer_exact': transfer_exact,
        'transfer_diff': transfer_diff,
        'transfer_cat': transfer_cat,
    }


# ============================================================
# Level 3: 정류장 시퀀스 (Stop Sequence) - 가중치 0.30
# ============================================================
def compute_sequence_metrics(otp_parsed, sc_parsed):
    """
    정류장 시퀀스 유사도 지표

    명칭 기반 비교: OTP from.name과 스마트카드 정류장명칭시퀀스를 직접 비교.
    """
    otp_stops_raw = otp_parsed['stops']
    sc_stops_raw = sc_parsed['stops']

    if not otp_stops_raw and not sc_stops_raw:
        return {
            'seq_jaccard': 1.0, 'seq_lcs': 1.0, 'seq_levenshtein': 1.0,
            'seq_prefix': 1.0, 'seq_suffix': 1.0,
        }
    if not otp_stops_raw or not sc_stops_raw:
        return {
            'seq_jaccard': 0.0, 'seq_lcs': 0.0, 'seq_levenshtein': 0.0,
            'seq_prefix': 0.0, 'seq_suffix': 0.0,
        }

    # 정류장명 정규화 (OTP '군포1동행정복지센터.군포역' ↔ SC '군포역')
    otp_stops, sc_stops = _normalize_stops_for_comparison(otp_stops_raw, sc_stops_raw)

    # Jaccard (집합 기반)
    otp_set = set(otp_stops)
    sc_set = set(sc_stops)
    union = len(otp_set | sc_set)
    seq_jaccard = len(otp_set & sc_set) / union if union > 0 else 0.0

    # LCS
    lcs_len = _lcs_length(otp_stops, sc_stops)
    max_len = max(len(otp_stops), len(sc_stops))
    seq_lcs = lcs_len / max_len if max_len > 0 else 0.0

    # Levenshtein
    lev_dist = _levenshtein_distance(otp_stops, sc_stops)
    seq_levenshtein = 1.0 - lev_dist / max_len if max_len > 0 else 0.0

    # Prefix (시작부 일치율)
    prefix_match = 0
    for a, b in zip(otp_stops, sc_stops):
        if a == b:
            prefix_match += 1
        else:
            break
    seq_prefix = prefix_match / max_len if max_len > 0 else 0.0

    # Suffix (도착부 일치율)
    suffix_match = 0
    for a, b in zip(reversed(otp_stops), reversed(sc_stops)):
        if a == b:
            suffix_match += 1
        else:
            break
    seq_suffix = suffix_match / max_len if max_len > 0 else 0.0

    return {
        'seq_jaccard': seq_jaccard,
        'seq_lcs': seq_lcs,
        'seq_levenshtein': seq_levenshtein,
        'seq_prefix': seq_prefix,
        'seq_suffix': seq_suffix,
    }



# ============================================================
# Level 4: 소요시간 (Travel Time) - 가중치 0.15
# ============================================================
def compute_time_metrics(otp_parsed, sc_parsed, threshold_sec=1800):
    """
    소요시간 유사도 지표

    Args:
        threshold_sec: 정규화 기준 시간 (초), 기본 30분
    """
    otp_time = otp_parsed['total_time']  # 초
    sc_time = sc_parsed['total_time']    # 초

    if otp_time == 0 and sc_time == 0:
        return {
            'time_diff_abs': 0.0, 'time_ratio': 1.0,
            'time_band': 1.0, 'time_score': 1.0,
        }

    diff = abs(otp_time - sc_time)

    # 절대 차이 (분)
    time_diff_abs = diff / 60.0

    # 시간 비율
    max_time = max(otp_time, sc_time, 1)
    min_time = min(otp_time, sc_time)
    time_ratio = min_time / max_time if max_time > 0 else 0.0

    # 동일 시간대 (±5분 = 300초)
    time_band = 1.0 if diff <= 300 else 0.0

    # 정규화 점수
    time_score = max(0.0, 1.0 - diff / threshold_sec)

    return {
        'time_diff_abs': time_diff_abs,
        'time_ratio': time_ratio,
        'time_band': time_band,
        'time_score': time_score,
    }


# ============================================================
# Level 5: 노선 (Route) - 가중치 0.15
# ============================================================
def compute_route_metrics(otp_parsed, sc_parsed):
    """노선 유사도 지표 (명칭 기반 비교)"""
    otp_routes = set(otp_parsed['routes'])
    sc_routes = set(sc_parsed['routes'])

    if not otp_routes and not sc_routes:
        return {'route_exact': 1.0, 'route_jaccard': 1.0, 'route_main': 1.0}
    if not otp_routes or not sc_routes:
        return {'route_exact': 0.0, 'route_jaccard': 0.0, 'route_main': 0.0}

    route_exact = 1.0 if otp_routes == sc_routes else 0.0

    intersection = len(otp_routes & sc_routes)
    union = len(otp_routes | sc_routes)
    route_jaccard = intersection / union if union > 0 else 0.0

    # 주 노선 일치 (OTP의 main_route가 SC routes에 포함)
    main_route = otp_parsed.get('main_route', '')
    route_main = 1.0 if main_route and main_route in sc_routes else 0.0

    return {
        'route_exact': route_exact,
        'route_jaccard': route_jaccard,
        'route_main': route_main,
    }


# ============================================================
# Level 6: 공간 (Spatial) - 가중치 0.10
# ============================================================
def compute_polyline_similarity(coords1, coords2, normalize_dist=5000):
    """
    두 폴리라인 간 평균 Hausdorff 거리 기반 유사도

    Args:
        coords1: [(name, lat, lon), ...] 또는 [(lat, lon), ...]
        coords2: [(name, lat, lon), ...] 또는 [(lat, lon), ...]
        normalize_dist: 정규화 기준 거리 (미터), 기본 5km

    Returns:
        0~1 유사도 점수 (1 = 완전 일치, 0 = 5km 이상 차이)
    """
    if not coords1 or not coords2:
        return 0.0

    # 좌표 추출 (name 포함/미포함 모두 대응)
    def to_latlon(coords):
        result = []
        for c in coords:
            if len(c) == 3:
                result.append((float(c[1]), float(c[2])))
            elif len(c) == 2:
                result.append((float(c[0]), float(c[1])))
        return result

    pts1 = to_latlon(coords1)
    pts2 = to_latlon(coords2)

    if not pts1 or not pts2:
        return 0.0

    # 방향 1: pts1의 각 점 → pts2의 최근접 점 거리 평균
    def avg_min_distance(from_pts, to_pts):
        total = 0.0
        for lat1, lon1 in from_pts:
            min_dist = float('inf')
            for lat2, lon2 in to_pts:
                d = _haversine(lat1, lon1, lat2, lon2)
                if d < min_dist:
                    min_dist = d
            total += min_dist
        return total / len(from_pts)

    avg1 = avg_min_distance(pts1, pts2)
    avg2 = avg_min_distance(pts2, pts1)

    # 양방향 평균의 최대값 (modified Hausdorff)
    hausdorff_avg = max(avg1, avg2)

    # 정규화: 5km 이상이면 0
    score = max(0.0, 1.0 - hausdorff_avg / normalize_dist)
    return score


def _extract_sc_known_coords(sc_parsed):
    """
    SC parsed dict에서 사용 가능한 좌표 추출.
    OD 좌표 + (있으면) stop_coords → [(lat, lon), ...]
    """
    coords = []
    od = sc_parsed.get('od_coords') or {}

    # 출발지
    if od.get('o_lat') is not None and od.get('o_lon') is not None:
        coords.append((float(od['o_lat']), float(od['o_lon'])))

    # stop_coords (환승 정류장 등, 있으면)
    for c in sc_parsed.get('stop_coords', []):
        if c and len(c) >= 2 and c[0] is not None and c[1] is not None:
            coords.append((float(c[0]), float(c[1])))

    # 도착지
    if od.get('d_lat') is not None and od.get('d_lon') is not None:
        coords.append((float(od['d_lat']), float(od['d_lon'])))

    # 중복 제거 (순서 유지)
    seen = set()
    unique = []
    for c in coords:
        key = (round(c[0], 5), round(c[1], 5))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def _point_to_segment_distance(pt, seg_start, seg_end):
    """점 pt에서 선분 (seg_start, seg_end)까지의 최근접 거리 (미터)"""
    lat_p, lon_p = pt
    lat_a, lon_a = seg_start
    lat_b, lon_b = seg_end

    # 선분 길이가 0이면 점-점 거리
    seg_len_sq = (lat_b - lat_a) ** 2 + (lon_b - lon_a) ** 2
    if seg_len_sq < 1e-14:
        return _haversine(lat_p, lon_p, lat_a, lon_a)

    # 투영 비율 t (0~1 사이로 클램핑)
    t = ((lat_p - lat_a) * (lat_b - lat_a) + (lon_p - lon_a) * (lon_b - lon_a)) / seg_len_sq
    t = max(0.0, min(1.0, t))

    # 투영점
    proj_lat = lat_a + t * (lat_b - lat_a)
    proj_lon = lon_a + t * (lon_b - lon_a)

    return _haversine(lat_p, lon_p, proj_lat, proj_lon)


def compute_points_on_path_similarity(sc_coords, otp_polyline, normalize_dist=2000):
    """
    SC의 알려진 좌표들이 OTP 경로 polyline에 얼마나 가까운지 측정.

    각 SC 좌표 → OTP polyline의 최근접 선분까지 거리의 평균 → 정규화.
    같은 OD라도 다른 경로의 환승 정류장은 OTP 경로에서 멀어짐 → 변별력.

    Args:
        sc_coords: [(lat, lon), ...] SC의 알려진 좌표들
        otp_polyline: [(lat, lon), ...] OTP legGeometry 디코딩 결과
        normalize_dist: 정규화 기준 거리 (미터), 기본 2km

    Returns:
        0~1 유사도 점수 (1 = 모든 SC 좌표가 OTP 경로 위에 있음)
    """
    if not sc_coords or not otp_polyline or len(otp_polyline) < 2:
        return 0.0

    total_dist = 0.0
    for pt in sc_coords:
        min_dist = float('inf')
        for j in range(len(otp_polyline) - 1):
            d = _point_to_segment_distance(pt, otp_polyline[j], otp_polyline[j + 1])
            if d < min_dist:
                min_dist = d
        total_dist += min_dist

    avg_dist = total_dist / len(sc_coords)
    return max(0.0, 1.0 - avg_dist / normalize_dist)


def compute_spatial_metrics(otp_parsed, sc_parsed):
    """
    공간 유사도 지표 (OTP legGeometry polyline 기반)

    - GTFS 있음: OTP polyline vs SC GTFS확장 좌표 → 양방향 Hausdorff
    - GTFS 없음: SC의 알려진 정류장 좌표 → OTP polyline 위 근접도
    """
    otp_od = otp_parsed.get('od_coords') or {}
    sc_od = sc_parsed.get('od_coords') or {}

    # OD 직선거리 유사도 (참고용 유지)
    od_distance_sim = 0.0
    if all(otp_od.get(k) is not None for k in ['o_lat', 'o_lon', 'd_lat', 'd_lon']) and \
       all(sc_od.get(k) is not None for k in ['o_lat', 'o_lon', 'd_lat', 'd_lon']):
        otp_od_dist = _haversine(otp_od['o_lat'], otp_od['o_lon'], otp_od['d_lat'], otp_od['d_lon'])
        sc_od_dist = _haversine(sc_od['o_lat'], sc_od['o_lon'], sc_od['d_lat'], sc_od['d_lon'])

        max_dist = max(otp_od_dist, sc_od_dist, 1)
        od_distance_sim = 1.0 - abs(otp_od_dist - sc_od_dist) / max_dist

    # 환승 정류장 일치율 (참고용 유지)
    transfer_loc_match = _compute_transfer_loc_match(otp_parsed, sc_parsed, threshold=500)

    # OTP route_polyline (legGeometry 기반)
    otp_polyline = otp_parsed.get('route_polyline', [])

    # GTFS 기반 폴리라인 유사도
    otp_full = otp_parsed.get('full_stop_coords', [])
    sc_full = sc_parsed.get('full_stop_coords', [])

    if otp_full and sc_full:
        # GTFS 있음: 기존 양방향 Hausdorff 방식 유지
        polyline_similarity = compute_polyline_similarity(otp_full, sc_full)
    elif otp_polyline:
        # GTFS 없음, legGeometry 있음: SC 알려진 좌표 → OTP polyline 근접도
        sc_coords = _extract_sc_known_coords(sc_parsed)
        polyline_similarity = compute_points_on_path_similarity(sc_coords, otp_polyline)
    else:
        polyline_similarity = 0.0

    return {
        'od_distance_sim': od_distance_sim,
        'transfer_loc_match': transfer_loc_match,
        'polyline_similarity': polyline_similarity,
    }


def _compute_transfer_loc_match(otp_parsed, sc_parsed, threshold=500):
    """환승 정류장 위치 일치율 (좌표 기반)"""
    otp_coords = otp_parsed.get('stop_coords', [])
    sc_od = sc_parsed.get('od_coords', {})

    # OTP 환승 정류장 = 중간 정류장들 (첫/끝 제외)
    if len(otp_coords) <= 2:
        # 환승 없음 → 일치
        if sc_parsed['transfer_count'] == 0:
            return 1.0
        return 0.0

    otp_transfer_coords = otp_coords[1:-1]

    # SC 환승 정류장 좌표 → TCN에는 OD만 있고 중간 환승 좌표가 없음
    # 정류장시퀀스의 중간 ID만 있으므로, 좌표 매칭이 제한적
    # → 환승 횟수가 같으면 0.5 부여, 다르면 0.0
    if sc_parsed['transfer_count'] > 0:
        if otp_parsed['transfer_count'] == sc_parsed['transfer_count']:
            return 0.5
        else:
            return 0.0

    return 1.0 if len(otp_transfer_coords) == 0 else 0.0


# ============================================================
# 복합 유사도 점수 (Composite Similarity Score)
# ============================================================
def compute_all_metrics(otp_parsed, sc_parsed):
    """모든 레벨의 유사도 지표 계산"""
    metrics = {}
    metrics.update(compute_mode_metrics(otp_parsed, sc_parsed))
    metrics.update(compute_transfer_metrics(otp_parsed, sc_parsed))
    metrics.update(compute_sequence_metrics(otp_parsed, sc_parsed))
    metrics.update(compute_time_metrics(otp_parsed, sc_parsed))
    metrics.update(compute_route_metrics(otp_parsed, sc_parsed))
    metrics.update(compute_spatial_metrics(otp_parsed, sc_parsed))
    return metrics


def compute_composite_similarity(metrics, weights=None):
    """
    모든 지표를 종합한 복합 유사도 계산

    Args:
        metrics: compute_all_metrics()의 결과
        weights: 레벨별 가중치 (기본값 사용 시 None)

    Returns:
        float: 0~1 사이의 복합 유사도 점수
    """
    if weights is None:
        weights = {
            'mode': 0.10,
            'sequence': 0.15,
            'time': 0.10,
            'route': 0.15,
            'spatial': 0.50,
        }

    mode_score = metrics.get('mode_jaccard', 0)

    sequence_score = metrics.get('seq_lcs', 0)

    time_score = metrics.get('time_ratio', 0)

    route_score = metrics.get('route_jaccard', 0)

    spatial_score = metrics.get('polyline_similarity', 0)

    composite = (
        weights['mode'] * mode_score +
        weights['sequence'] * sequence_score +
        weights['time'] * time_score +
        weights['route'] * route_score +
        weights['spatial'] * spatial_score
    )

    return {
        'composite': composite,
        'mode_score': mode_score,
        'sequence_score': sequence_score,
        'time_score': time_score,
        'route_score': route_score,
        'spatial_score': spatial_score,
    }


def grade_similarity(composite_score):
    """유사도 등급 판정"""
    if composite_score >= 0.8:
        return '우수'
    elif composite_score >= 0.6:
        return '양호'
    elif composite_score >= 0.4:
        return '보통'
    else:
        return '불량'


# ============================================================
# 매칭 함수
# ============================================================
def match_smartcard_to_otp(sc_parsed, otp_itineraries, threshold=0.6, gtfs_lookup=None):
    """
    스마트카드 통행과 OTP 경로 매칭

    Args:
        sc_parsed: parse_smartcard_trip()의 결과
        otp_itineraries: OTP 응답의 itineraries 리스트
        threshold: 매칭 임계값
        gtfs_lookup: GTFSRouteLookup 인스턴스 (None이면 GTFS 확장 안 함)

    Returns:
        매칭 결과 dict
    """
    scores = []

    for i, itin in enumerate(otp_itineraries):
        otp_parsed = parse_otp_itinerary(itin, gtfs_lookup=gtfs_lookup)
        metrics = compute_all_metrics(otp_parsed, sc_parsed)
        score_result = compute_composite_similarity(metrics)

        scores.append({
            'itinerary_idx': i,
            'composite': score_result['composite'],
            'level_scores': score_result,
            'metrics': metrics,
            'otp_parsed': otp_parsed,
        })

    scores.sort(key=lambda x: x['composite'], reverse=True)

    best = scores[0]
    return {
        'matched': best['composite'] >= threshold,
        'best_idx': best['itinerary_idx'],
        'best_score': best['composite'],
        'best_grade': grade_similarity(best['composite']),
        'best_metrics': best['metrics'],
        'best_level_scores': best['level_scores'],
        'all_scores': scores,
    }
