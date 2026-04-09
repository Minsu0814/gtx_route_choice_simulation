use std::collections::HashMap;

use crate::access::street_graph::StreetGraph;
use crate::access::walk;
use crate::raptor::cost::TaxiCostConfig;
use crate::types::{haversine_distance, StopPoint, TransitData};

/// How the passenger reaches/leaves a transit stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    Walk,
    Taxi,
}

/// A stop reachable from an origin/destination by walking or taxi.
#[derive(Debug, Clone)]
pub struct AccessStop {
    pub stop_index: u32,
    pub walk_time_secs: u32,   // for Walk: actual walk time; for Taxi: drive_time + wait_time
    pub walk_distance_m: f64,  // for Walk: haversine × detour; for Taxi: road distance
    pub orig_lat: f64,         // origin/destination coordinate (for path reconstruction)
    pub orig_lon: f64,
    pub mode: AccessMode,
    // Taxi-specific fields (only set when mode == Taxi)
    pub taxi_fare_krw: u32,
    pub taxi_wait_secs: u32,
    pub taxi_drive_secs: u32,
}

/// Find nearby walking-accessible stops within `max_distance_m` meters.
///
/// Always uses haversine × detour_factor (1.35). The street graph is a car
/// road network — missing alleys, parks, pedestrian paths — so it's not
/// suitable for pedestrian distance estimation.
pub fn find_access_stops(
    data: &TransitData,
    lat: f64,
    lon: f64,
    max_distance_m: f64,
    max_stops: usize,
    walk_speed: f64,
) -> Vec<AccessStop> {
    find_access_stops_haversine(data, lat, lon, max_distance_m, max_stops, walk_speed)
}

/// Haversine-based walk access stop finder.
fn find_access_stops_haversine(
    data: &TransitData,
    lat: f64,
    lon: f64,
    max_distance_m: f64,
    max_stops: usize,
    walk_speed: f64,
) -> Vec<AccessStop> {
    let lat_range = max_distance_m / 111_000.0;
    let lon_range = max_distance_m / (111_000.0 * lat.to_radians().cos());

    let envelope = rstar::AABB::from_corners(
        [lat - lat_range, lon - lon_range],
        [lat + lat_range, lon + lon_range],
    );

    let mut stops: Vec<AccessStop> = data
        .stop_rtree
        .locate_in_envelope(&envelope)
        .filter_map(|point: &StopPoint| {
            let dist = haversine_distance(lat, lon, point.lat, point.lon);
            if dist <= max_distance_m {
                let (walk_time, walk_dist) =
                    walk::estimate_walk(lat, lon, point.lat, point.lon, walk_speed);
                Some(AccessStop {
                    stop_index: point.index,
                    walk_time_secs: walk_time,
                    walk_distance_m: walk_dist,
                    orig_lat: lat,
                    orig_lon: lon,
                    mode: AccessMode::Walk,
                    taxi_fare_krw: 0,
                    taxi_wait_secs: 0,
                    taxi_drive_secs: 0,
                })
            } else {
                None
            }
        })
        .collect();

    stops.sort_by_key(|s| s.walk_time_secs);
    stops.truncate(max_stops);
    stops
}

/// Find stops reachable by taxi within [min_distance_m, max_distance_m].
///
/// When `street_graph` is provided, uses actual road network distances.
/// Falls back to haversine × road_factor when no graph is available.
pub fn find_taxi_access_stops(
    data: &TransitData,
    lat: f64,
    lon: f64,
    min_distance_m: f64,
    max_distance_m: f64,
    max_stops: usize,
    taxi_config: &TaxiCostConfig,
    street_graph: Option<&StreetGraph>,
) -> Vec<AccessStop> {
    if let Some(graph) = street_graph {
        find_taxi_stops_network(
            data, graph, lat, lon, min_distance_m, max_distance_m, max_stops, taxi_config,
        )
    } else {
        find_taxi_stops_haversine(
            data, lat, lon, min_distance_m, max_distance_m, max_stops, taxi_config,
        )
    }
}

/// Network-based taxi access stop finder.
fn find_taxi_stops_network(
    data: &TransitData,
    graph: &StreetGraph,
    lat: f64,
    lon: f64,
    min_distance_m: f64,
    max_distance_m: f64,
    max_stops: usize,
    taxi_config: &TaxiCostConfig,
) -> Vec<AccessStop> {
    let (src_node, snap_dist) = match graph.snap(lat, lon, max_distance_m) {
        Some(r) => r,
        None => {
            return find_taxi_stops_haversine(
                data, lat, lon, min_distance_m, max_distance_m, max_stops, taxi_config,
            );
        }
    };

    // Bounded Dijkstra with driving weights
    let max_time_ms = (max_distance_m / taxi_config.taxi_speed_mps * 1000.0) as u32;
    let reachable = graph.dijkstra_bounded(src_node, max_time_ms, false);

    // Sparse map: only reachable nodes
    let mut node_times: HashMap<u32, (u32, f32)> = HashMap::with_capacity(reachable.len() + 1);
    node_times.insert(src_node, (0, 0.0));
    for &(node, time_ms, dist_m) in &reachable {
        node_times.insert(node, (time_ms, dist_m));
    }

    let lat_range = max_distance_m / 111_000.0;
    let lon_range = max_distance_m / (111_000.0 * lat.to_radians().cos());
    let envelope = rstar::AABB::from_corners(
        [lat - lat_range, lon - lon_range],
        [lat + lat_range, lon + lon_range],
    );

    let mut candidates: Vec<AccessStop> = data
        .stop_rtree
        .locate_in_envelope(&envelope)
        .filter_map(|point: &StopPoint| {
            let haversine_dist = haversine_distance(lat, lon, point.lat, point.lon);
            if haversine_dist < min_distance_m || haversine_dist > max_distance_m {
                return None;
            }

            // Haversine baseline
            let hav_road_distance = haversine_dist * taxi_config.road_factor;

            // Try network-based distance, use it only if shorter
            if let Some((stop_node, _)) = graph.snap(point.lat, point.lon, 300.0) {
                if let Some(&(time_ms, dist_m)) = node_times.get(&stop_node) {
                    let net_road_distance = snap_dist + dist_m as f64;
                    if net_road_distance < hav_road_distance {
                        let drive_secs =
                            (time_ms / 1000) + (snap_dist / taxi_config.taxi_speed_mps) as u32;
                        let fare_krw = taxi_config.estimate_fare(net_road_distance);
                        let total_duration = taxi_config.default_wait_secs + drive_secs;

                        return Some(AccessStop {
                            stop_index: point.index,
                            walk_time_secs: total_duration,
                            walk_distance_m: net_road_distance,
                            orig_lat: lat,
                            orig_lon: lon,
                            mode: AccessMode::Taxi,
                            taxi_fare_krw: fare_krw,
                            taxi_wait_secs: taxi_config.default_wait_secs,
                            taxi_drive_secs: drive_secs,
                        });
                    }
                }
            }

            // Use haversine × road_factor estimate
            let road_distance = hav_road_distance;
            let drive_secs = (road_distance / taxi_config.taxi_speed_mps).ceil() as u32;
            let fare_krw = taxi_config.estimate_fare(road_distance);
            let total_duration = taxi_config.default_wait_secs + drive_secs;

            Some(AccessStop {
                stop_index: point.index,
                walk_time_secs: total_duration,
                walk_distance_m: road_distance,
                orig_lat: lat,
                orig_lon: lon,
                mode: AccessMode::Taxi,
                taxi_fare_krw: fare_krw,
                taxi_wait_secs: taxi_config.default_wait_secs,
                taxi_drive_secs: drive_secs,
            })
        })
        .collect();

    candidates.sort_by_key(|s| {
        let time_cost_cs = s.walk_time_secs as u64 * 100;
        let fare_cost_cs = s.taxi_fare_krw as u64 * taxi_config.vot_factor as u64;
        (time_cost_cs + fare_cost_cs).min(u32::MAX as u64) as u32
    });
    candidates.truncate(max_stops);
    candidates
}

/// Haversine-based taxi access fallback.
fn find_taxi_stops_haversine(
    data: &TransitData,
    lat: f64,
    lon: f64,
    min_distance_m: f64,
    max_distance_m: f64,
    max_stops: usize,
    taxi_config: &TaxiCostConfig,
) -> Vec<AccessStop> {
    let lat_range = max_distance_m / 111_000.0;
    let lon_range = max_distance_m / (111_000.0 * lat.to_radians().cos());

    let envelope = rstar::AABB::from_corners(
        [lat - lat_range, lon - lon_range],
        [lat + lat_range, lon + lon_range],
    );

    let mut candidates: Vec<AccessStop> = data
        .stop_rtree
        .locate_in_envelope(&envelope)
        .filter_map(|point: &StopPoint| {
            let dist = haversine_distance(lat, lon, point.lat, point.lon);
            if dist >= min_distance_m && dist <= max_distance_m {
                let road_distance = dist * taxi_config.road_factor;
                let drive_secs = (road_distance / taxi_config.taxi_speed_mps).ceil() as u32;
                let fare_krw = taxi_config.estimate_fare(road_distance);
                let total_duration = taxi_config.default_wait_secs + drive_secs;

                Some(AccessStop {
                    stop_index: point.index,
                    walk_time_secs: total_duration,
                    walk_distance_m: road_distance,
                    orig_lat: lat,
                    orig_lon: lon,
                    mode: AccessMode::Taxi,
                    taxi_fare_krw: fare_krw,
                    taxi_wait_secs: taxi_config.default_wait_secs,
                    taxi_drive_secs: drive_secs,
                })
            } else {
                None
            }
        })
        .collect();

    candidates.sort_by_key(|s| {
        let time_cost_cs = s.walk_time_secs as u64 * 100;
        let fare_cost_cs = s.taxi_fare_krw as u64 * taxi_config.vot_factor as u64;
        (time_cost_cs + fare_cost_cs).min(u32::MAX as u64) as u32
    });
    candidates.truncate(max_stops);
    candidates
}
