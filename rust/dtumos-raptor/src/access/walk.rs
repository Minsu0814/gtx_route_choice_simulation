use crate::types::haversine_distance;

/// Default walking speed (m/s).
pub const WALK_SPEED_MPS: f64 = 1.2;

/// Detour factor: real walking distance ≈ haversine × 1.35.
/// Empirical value for urban grids (Manhattan factor ~1.41, typical ~1.3-1.4).
pub const WALK_DETOUR_FACTOR: f64 = 1.35;

/// Estimate walking time using haversine distance × detour factor.
/// Returns (walk_time_secs, estimated_road_distance_m).
pub fn estimate_walk(
    from_lat: f64,
    from_lon: f64,
    to_lat: f64,
    to_lon: f64,
    walk_speed: f64,
) -> (u32, f64) {
    let haversine = haversine_distance(from_lat, from_lon, to_lat, to_lon);
    let dist = haversine * WALK_DETOUR_FACTOR;
    let time = (dist / walk_speed).ceil() as u32;
    (time, dist)
}
