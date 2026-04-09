/// Encode a sequence of (lat, lon) coordinates into a Google Encoded Polyline string.
pub fn encode_polyline(coords: &[(f64, f64)]) -> String {
    let mut result = String::new();
    let mut prev_lat = 0i64;
    let mut prev_lon = 0i64;

    for &(lat, lon) in coords {
        let lat_e5 = (lat * 1e5).round() as i64;
        let lon_e5 = (lon * 1e5).round() as i64;

        encode_value(lat_e5 - prev_lat, &mut result);
        encode_value(lon_e5 - prev_lon, &mut result);

        prev_lat = lat_e5;
        prev_lon = lon_e5;
    }

    result
}

/// Decode a Google Encoded Polyline string into (lat, lon) coordinates.
pub fn decode_polyline(encoded: &str) -> Vec<(f64, f64)> {
    let mut coords = Vec::new();
    let mut lat = 0i64;
    let mut lon = 0i64;
    let bytes = encoded.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        lat += decode_next(bytes, &mut i);
        if i >= bytes.len() {
            break;
        }
        lon += decode_next(bytes, &mut i);
        coords.push((lat as f64 / 1e5, lon as f64 / 1e5));
    }
    coords
}

fn decode_next(bytes: &[u8], i: &mut usize) -> i64 {
    let mut result = 0u64;
    let mut shift = 0;
    loop {
        if *i >= bytes.len() {
            break;
        }
        let raw = bytes[*i];
        if raw < 63 {
            // Invalid polyline byte — stop decoding this value
            *i += 1;
            break;
        }
        let b = (raw - 63) as u64;
        *i += 1;
        result |= (b & 0x1f) << shift;
        shift += 5;
        if b < 0x20 {
            break;
        }
    }
    if result & 1 != 0 {
        !(result >> 1) as i64
    } else {
        (result >> 1) as i64
    }
}

fn encode_value(value: i64, result: &mut String) {
    let mut v = if value < 0 {
        (!value) << 1 | 1
    } else {
        value << 1
    } as u64;

    loop {
        let mut chunk = (v & 0x1f) as u8;
        v >>= 5;
        if v > 0 {
            chunk |= 0x20;
        }
        result.push((chunk + 63) as char);
        if v == 0 {
            break;
        }
    }
}
