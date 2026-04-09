use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use h3o::{LatLng, Resolution};
use rayon::prelude::*;
use serde_json::Value;

use crate::access::street_graph::StreetGraph;
use crate::access::{find_access_stops, find_taxi_access_stops, AccessStop};
use crate::output::paths_to_otp_json;
use crate::raptor::cost::{CostConfig, TaxiCostConfig};
use crate::raptor::engine::{raptor_compute, raptor_extract, raptor_search};
use crate::raptor::fare::FareConfig;
use crate::raptor::mc_raptor::{mc_raptor_compute, mc_raptor_extract, mc_raptor_search};
use crate::raptor::range::range_raptor_search;
use crate::types::TransitData;

/// Search mode for batch routing.
#[derive(Debug, Clone, Copy)]
pub enum SearchMode {
    Standard,
    MultiCriteria,
    Research,
    /// Range-RAPTOR: search across a departure time window.
    /// (step_secs, window_secs) — e.g., (60, 1800) = every 1 min, 30 min window.
    Range { step_secs: u32, window_secs: u32 },
}

/// Batch routing configuration.
pub struct BatchConfig {
    pub max_access_walk_m: f64,
    pub max_egress_walk_m: f64,
    pub max_access_stops: usize,
    pub max_egress_stops: usize,
    pub walk_speed: f64,
    pub max_results: usize,
    pub search_mode: SearchMode,
    pub mc_relax_ratio: f64,
    pub mc_relax_slack: u32,
    // Taxi access/egress configuration
    pub enable_taxi_access: bool,
    pub max_taxi_distance_m: f64,
    pub min_taxi_distance_m: f64,
    pub max_taxi_stops: usize,
    // Fare configuration
    pub fare_config: FareConfig,
    // Geometry configuration
    pub use_route_shapes: bool,
    /// Force routing strategy: None = auto, true = grouped, false = individual.
    pub force_grouped: Option<bool>,
    /// H3 resolution for spatial bucketing (0 = disabled, 7-9 = enabled).
    /// Queries in the same H3 cell share RAPTOR computation.
    ///   res 9: ~175m edge (fine, ~balanced)
    ///   res 8: ~460m edge (medium, ~fast)
    ///   res 7: ~1.2km edge (coarse, ~turbo)
    pub h3_resolution: u8,
    /// Time bucket size in seconds for departure time grouping.
    /// 0 = exact second match (default).
    pub time_bucket_secs: u32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_access_walk_m: 800.0,
            max_egress_walk_m: 800.0,
            max_access_stops: 30,
            max_egress_stops: 30,
            walk_speed: 1.2,
            max_results: 5,
            search_mode: SearchMode::MultiCriteria,
            mc_relax_ratio: 1.0,
            mc_relax_slack: 0,
            enable_taxi_access: false,
            max_taxi_distance_m: 5000.0,
            min_taxi_distance_m: 1500.0,
            max_taxi_stops: 20,
            fare_config: FareConfig::default(),
            use_route_shapes: false,
            force_grouped: None,
            h3_resolution: 0,
            time_bucket_secs: 0,
        }
    }
}

/// A single OD query.
pub struct OdQuery {
    pub from_lat: f64,
    pub from_lon: f64,
    pub to_lat: f64,
    pub to_lon: f64,
    pub departure_time: u32, // seconds since midnight
    /// Per-OD taxi wait time override (from ServiceFeed). None = use default.
    pub taxi_wait_secs: Option<u32>,
    /// Per-OD taxi surge multiplier override (from ServiceFeed). None = use default.
    pub taxi_surge: Option<f64>,
}

/// Minimum batch size to enable grouped optimization.
/// Below this threshold, grouping overhead exceeds benefit.
const GROUPED_THRESHOLD: usize = 64;

/// Run batch routing on multiple OD pairs.
///
/// Automatically selects between per-query parallel routing (small batches)
/// and grouped routing (large batches) for optimal performance.
/// Grouped routing deduplicates RAPTOR computations for queries sharing
/// the same origin and departure time.
pub fn route_batch(
    data: &TransitData,
    queries: &[OdQuery],
    cost_config: &CostConfig,
    taxi_config: &TaxiCostConfig,
    batch_config: &BatchConfig,
    street_graph: Option<&StreetGraph>,
) -> Vec<Value> {
    // Determine routing strategy
    let use_grouped = match batch_config.force_grouped {
        Some(true) => !matches!(batch_config.search_mode, SearchMode::Range { .. }),
        Some(false) => false,
        None => {
            queries.len() >= GROUPED_THRESHOLD
                && !matches!(batch_config.search_mode, SearchMode::Range { .. })
        }
    };

    if use_grouped {
        route_batch_grouped(data, queries, cost_config, taxi_config, batch_config, street_graph)
    } else {
        route_batch_individual(data, queries, cost_config, taxi_config, batch_config, street_graph)
    }
}

/// Per-query parallel routing (original approach).
fn route_batch_individual(
    data: &TransitData,
    queries: &[OdQuery],
    cost_config: &CostConfig,
    taxi_config: &TaxiCostConfig,
    batch_config: &BatchConfig,
    street_graph: Option<&StreetGraph>,
) -> Vec<Value> {
    queries
        .par_iter()
        .enumerate()
        .map(|(idx, query)| {
            route_single(
                data,
                query,
                idx,
                cost_config,
                taxi_config,
                batch_config,
                street_graph,
            )
        })
        .collect()
}

// ── H3 / spatial bucketing helpers ────────────────────────────────

/// Convert (lat, lon) to H3 cell u64 at given resolution.
/// Returns 0 if resolution is 0 (disabled) or coordinates are invalid.
#[inline]
fn to_h3_cell(lat: f64, lon: f64, resolution: u8) -> u64 {
    if resolution == 0 {
        return 0;
    }
    let res = match Resolution::try_from(resolution) {
        Ok(r) => r,
        Err(_) => return 0,
    };
    match LatLng::new(lat, lon) {
        Ok(ll) => u64::from(ll.to_cell(res)),
        Err(_) => 0,
    }
}

/// Get the center (lat, lon) of an H3 cell.
#[inline]
fn h3_cell_center(cell: u64) -> (f64, f64) {
    if cell == 0 {
        return (0.0, 0.0);
    }
    match h3o::CellIndex::try_from(cell) {
        Ok(idx) => {
            let ll = LatLng::from(idx);
            (f64::from(ll.lat()), f64::from(ll.lng()))
        }
        Err(_) => (0.0, 0.0),
    }
}

/// Approximate H3 cell edge length in meters for a given resolution.
#[inline]
fn h3_edge_length_m(resolution: u8) -> f64 {
    match resolution {
        0 => 1107_000.0, 1 => 418_000.0, 2 => 158_000.0, 3 => 59_800.0,
        4 => 22_600.0, 5 => 8_540.0, 6 => 3_230.0, 7 => 1_220.0,
        8 => 461.0, 9 => 174.0, 10 => 65.9, 11 => 24.9,
        12 => 9.4, 13 => 3.6, 14 => 1.3, 15 => 0.5,
        _ => 200.0,
    }
}

/// Quantize a coordinate to a spatial key:
/// - h3_resolution > 0: H3 cell u64
/// - h3_resolution == 0: exact coordinate as quantized i64 pair packed into u64
#[inline]
fn spatial_key(lat: f64, lon: f64, h3_resolution: u8) -> u64 {
    if h3_resolution > 0 {
        to_h3_cell(lat, lon, h3_resolution)
    } else {
        // Exact: pack two i32 into u64 (~0.1m precision)
        let lat_q = (lat * 1_000_000.0).round() as i32;
        let lon_q = (lon * 1_000_000.0).round() as i32;
        ((lat_q as u64) << 32) | (lon_q as u32 as u64)
    }
}

/// Reconstruct representative (lat, lon) from a spatial key.
#[inline]
fn spatial_key_to_latlon(key: u64, h3_resolution: u8) -> (f64, f64) {
    if h3_resolution > 0 {
        h3_cell_center(key)
    } else {
        let lat_q = (key >> 32) as i32;
        let lon_q = key as i32;
        (lat_q as f64 / 1_000_000.0, lon_q as f64 / 1_000_000.0)
    }
}

// ── Grouped batch routing ──────────────────────────────────────────

/// Group key: queries with the same key share a single RAPTOR computation.
/// Uses H3 cell (when h3_resolution > 0) or exact coordinates for spatial grouping.
#[derive(Clone)]
struct GroupKey {
    /// Origin spatial key (H3 cell u64 or packed lat/lon)
    from_spatial: u64,
    departure_time: u32,
    taxi_wait_secs: Option<u32>,
    taxi_surge_q: Option<i64>,
}

impl GroupKey {
    fn new(query: &OdQuery, h3_resolution: u8, time_bucket_secs: u32) -> Self {
        let dep = if time_bucket_secs > 0 {
            (query.departure_time / time_bucket_secs) * time_bucket_secs
        } else {
            query.departure_time
        };

        Self {
            from_spatial: spatial_key(query.from_lat, query.from_lon, h3_resolution),
            departure_time: dep,
            taxi_wait_secs: query.taxi_wait_secs,
            taxi_surge_q: query.taxi_surge.map(|s| (s * 1_000.0).round() as i64),
        }
    }

    fn representative_lat_lon(&self, h3_resolution: u8) -> (f64, f64) {
        spatial_key_to_latlon(self.from_spatial, h3_resolution)
    }
}

impl PartialEq for GroupKey {
    fn eq(&self, other: &Self) -> bool {
        self.from_spatial == other.from_spatial
            && self.departure_time == other.departure_time
            && self.taxi_wait_secs == other.taxi_wait_secs
            && self.taxi_surge_q == other.taxi_surge_q
    }
}
impl Eq for GroupKey {}

impl Hash for GroupKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.from_spatial.hash(state);
        self.departure_time.hash(state);
        self.taxi_wait_secs.hash(state);
        self.taxi_surge_q.hash(state);
    }
}

/// Pre-computed per-query data.
struct QueryInfo {
    idx: usize,
    access_stops: Vec<AccessStop>,
    egress_stops: Vec<AccessStop>,
}

/// Grouped batch routing: deduplicate RAPTOR across queries sharing the same origin.
///
/// Phase 1: Pre-compute access/egress stops for all queries (parallel)
/// Phase 2: Group by origin + departure time
/// Phase 3: Run RAPTOR once per group with merged egress for early termination (parallel)
/// Phase 4: Extract paths per query from shared RAPTOR state (parallel within group)
fn route_batch_grouped(
    data: &TransitData,
    queries: &[OdQuery],
    cost_config: &CostConfig,
    taxi_config: &TaxiCostConfig,
    batch_config: &BatchConfig,
    street_graph: Option<&StreetGraph>,
) -> Vec<Value> {
    let use_bucketing = batch_config.h3_resolution > 0 || batch_config.time_bucket_secs > 0;

    // Phase 1: Pre-compute egress stops per query + access stops
    // With bucketing: access stops are computed per-group (from representative point)
    // Without bucketing: access stops are computed per-query (exact coordinates)
    let query_infos: Vec<QueryInfo> = queries
        .par_iter()
        .enumerate()
        .map(|(idx, query)| {
            let effective_taxi_config = effective_taxi(taxi_config, query);

            // Access stops: skip per-query computation when bucketing (will compute per-group)
            let access_stops = if use_bucketing {
                Vec::new() // placeholder — filled per-group in Phase 3
            } else {
                let mut stops = find_access_stops(
                    data,
                    query.from_lat,
                    query.from_lon,
                    batch_config.max_access_walk_m,
                    batch_config.max_access_stops,
                    batch_config.walk_speed,
                );
                if batch_config.enable_taxi_access {
                    let taxi_access = find_taxi_access_stops(
                        data,
                        query.from_lat,
                        query.from_lon,
                        batch_config.min_taxi_distance_m,
                        batch_config.max_taxi_distance_m,
                        batch_config.max_taxi_stops,
                        &effective_taxi_config,
                        street_graph,
                    );
                    stops.extend(taxi_access);
                }
                stops
            };

            // Egress stops: always per-query (each query has a unique destination)
            let mut egress_stops = find_access_stops(
                data,
                query.to_lat,
                query.to_lon,
                batch_config.max_egress_walk_m,
                batch_config.max_egress_stops,
                batch_config.walk_speed,
            );
            if batch_config.enable_taxi_access {
                let taxi_egress = find_taxi_access_stops(
                    data,
                    query.to_lat,
                    query.to_lon,
                    batch_config.min_taxi_distance_m,
                    batch_config.max_taxi_distance_m,
                    batch_config.max_taxi_stops,
                    &effective_taxi_config,
                    street_graph,
                );
                egress_stops.extend(taxi_egress);
            }

            QueryInfo {
                idx,
                access_stops,
                egress_stops,
            }
        })
        .collect();

    // Phase 2: Group by origin + departure time (with optional bucketing)
    let mut groups: HashMap<GroupKey, Vec<usize>> = HashMap::new();
    for (qi_idx, query) in queries.iter().enumerate() {
        let key = GroupKey::new(query, batch_config.h3_resolution, batch_config.time_bucket_secs);
        groups.entry(key).or_default().push(qi_idx);
    }

    // Phase 3 + 4: Process groups in parallel
    let group_items: Vec<(GroupKey, Vec<usize>)> = groups.into_iter().collect();

    let scattered: Vec<Vec<(usize, Value)>> = group_items
        .par_iter()
        .map(|(key, member_indices)| {
            // Compute access stops for this group
            let group_access_stops = if use_bucketing {
                // Bucketing mode: compute access from representative cell center
                let (rep_lat, rep_lon) = key.representative_lat_lon(batch_config.h3_resolution);
                let effective_taxi_config = effective_taxi(taxi_config, &queries[member_indices[0]]);
                let mut stops = find_access_stops(
                    data,
                    rep_lat,
                    rep_lon,
                    batch_config.max_access_walk_m + h3_edge_length_m(batch_config.h3_resolution) * 0.7,
                    batch_config.max_access_stops,
                    batch_config.walk_speed,
                );
                if batch_config.enable_taxi_access {
                    let taxi_access = find_taxi_access_stops(
                        data,
                        rep_lat,
                        rep_lon,
                        batch_config.min_taxi_distance_m,
                        batch_config.max_taxi_distance_m,
                        batch_config.max_taxi_stops,
                        &effective_taxi_config,
                        street_graph,
                    );
                    stops.extend(taxi_access);
                }
                stops
            } else {
                query_infos[member_indices[0]].access_stops.clone()
            };

            // Skip group if no access stops
            if group_access_stops.is_empty() {
                return member_indices
                    .iter()
                    .map(|&qi_idx| (query_infos[qi_idx].idx, empty_result(query_infos[qi_idx].idx)))
                    .collect();
            }

            // Single-member group: use direct route (no grouping overhead)
            if member_indices.len() == 1 {
                let qi = &query_infos[member_indices[0]];
                let query = &queries[member_indices[0]];
                if qi.egress_stops.is_empty() {
                    return vec![(qi.idx, empty_result(qi.idx))];
                }
                let effective_taxi = effective_taxi(taxi_config, query);
                let paths = run_search(
                    data,
                    &group_access_stops,
                    &qi.egress_stops,
                    key.departure_time,
                    cost_config,
                    &effective_taxi,
                    batch_config,
                );
                let json = paths_to_otp_json(
                    data,
                    &paths,
                    qi.idx,
                    query.from_lat,
                    query.from_lon,
                    query.to_lat,
                    query.to_lon,
                    query.departure_time,
                    &batch_config.fare_config,
                    batch_config.use_route_shapes,
                    if batch_config.use_route_shapes { street_graph } else { None },
                );
                return vec![(qi.idx, json)];
            }

            // Multi-member group: run RAPTOR once, extract per query
            let effective_taxi = effective_taxi(taxi_config, &queries[member_indices[0]]);

            // Merge all members' egress stops as upper bound for early termination.
            // Using the union gives a loose but valid bound — RAPTOR won't prune
            // labels that could reach ANY member's destination.
            let mut merged_egress: Vec<AccessStop> = Vec::new();
            let mut seen_stops = std::collections::HashSet::new();
            for &qi_idx in member_indices {
                for eg in &query_infos[qi_idx].egress_stops {
                    if seen_stops.insert(eg.stop_index) {
                        merged_egress.push(eg.clone());
                    }
                }
            }

            // Run shared RAPTOR computation
            match batch_config.search_mode {
                SearchMode::Standard => {
                    let state = raptor_compute(
                        data,
                        &group_access_stops,
                        &merged_egress,
                        key.departure_time,
                        cost_config,
                    );

                    member_indices
                        .iter()
                        .map(|&qi_idx| {
                            let qi = &query_infos[qi_idx];
                            let query = &queries[qi_idx];
                            if qi.egress_stops.is_empty() {
                                return (qi.idx, empty_result(qi.idx));
                            }
                            let paths = raptor_extract(
                                data,
                                &state,
                                &group_access_stops,
                                &qi.egress_stops,
                                key.departure_time,
                                cost_config,
                                batch_config.max_results,
                            );
                            let json = paths_to_otp_json(
                                data,
                                &paths,
                                qi.idx,
                                query.from_lat,
                                query.from_lon,
                                query.to_lat,
                                query.to_lon,
                                query.departure_time,
                                &batch_config.fare_config,
                                batch_config.use_route_shapes,
                                if batch_config.use_route_shapes { street_graph } else { None },
                            );
                            (qi.idx, json)
                        })
                        .collect()
                }
                SearchMode::MultiCriteria | SearchMode::Research => {
                    let (relax_ratio, relax_slack) = match batch_config.search_mode {
                        SearchMode::Research => (1.3, 0),
                        _ => (batch_config.mc_relax_ratio, batch_config.mc_relax_slack),
                    };

                    let state = mc_raptor_compute(
                        data,
                        &group_access_stops,
                        &merged_egress,
                        key.departure_time,
                        cost_config,
                        &effective_taxi,
                        relax_ratio,
                        relax_slack,
                    );

                    member_indices
                        .iter()
                        .map(|&qi_idx| {
                            let qi = &query_infos[qi_idx];
                            let query = &queries[qi_idx];
                            if qi.egress_stops.is_empty() {
                                return (qi.idx, empty_result(qi.idx));
                            }
                            let paths = mc_raptor_extract(
                                data,
                                &state,
                                &group_access_stops,
                                &qi.egress_stops,
                                key.departure_time,
                                cost_config,
                                &effective_taxi,
                                batch_config.max_results,
                            );
                            let json = paths_to_otp_json(
                                data,
                                &paths,
                                qi.idx,
                                query.from_lat,
                                query.from_lon,
                                query.to_lat,
                                query.to_lon,
                                query.departure_time,
                                &batch_config.fare_config,
                                batch_config.use_route_shapes,
                                if batch_config.use_route_shapes { street_graph } else { None },
                            );
                            (qi.idx, json)
                        })
                        .collect()
                }
                SearchMode::Range { .. } => {
                    // Range mode: fall back to per-query (shouldn't reach here)
                    member_indices
                        .iter()
                        .map(|&qi_idx| {
                            let qi = &query_infos[qi_idx];
                            let query = &queries[qi_idx];
                            let json = route_single(
                                data,
                                query,
                                qi.idx,
                                cost_config,
                                taxi_config,
                                batch_config,
                                street_graph,
                            );
                            (qi.idx, json)
                        })
                        .collect()
                }
            }
        })
        .collect();

    // Scatter results back to original order
    let mut results = vec![empty_result(0); queries.len()];
    for group in scattered {
        for (idx, value) in group {
            results[idx] = value;
        }
    }
    results
}

/// Build effective taxi config with per-query overrides.
fn effective_taxi(base: &TaxiCostConfig, query: &OdQuery) -> TaxiCostConfig {
    if query.taxi_wait_secs.is_some() || query.taxi_surge.is_some() {
        let mut cfg = base.clone();
        if let Some(w) = query.taxi_wait_secs {
            cfg.default_wait_secs = w;
        }
        if let Some(s) = query.taxi_surge {
            cfg.default_surge = s;
        }
        cfg
    } else {
        base.clone()
    }
}

/// Run the appropriate RAPTOR search (non-grouped, for single queries or fallback).
fn run_search(
    data: &TransitData,
    access_stops: &[AccessStop],
    egress_stops: &[AccessStop],
    departure_time: u32,
    cost_config: &CostConfig,
    taxi_config: &TaxiCostConfig,
    config: &BatchConfig,
) -> Vec<crate::raptor::engine::RaptorPath> {
    match config.search_mode {
        SearchMode::Standard => raptor_search(
            data,
            access_stops,
            egress_stops,
            departure_time,
            cost_config,
            config.max_results,
        ),
        SearchMode::MultiCriteria => mc_raptor_search(
            data,
            access_stops,
            egress_stops,
            departure_time,
            cost_config,
            taxi_config,
            config.max_results,
            config.mc_relax_ratio,
            config.mc_relax_slack,
        ),
        SearchMode::Research => mc_raptor_search(
            data,
            access_stops,
            egress_stops,
            departure_time,
            cost_config,
            taxi_config,
            config.max_results,
            1.3,
            0,
        ),
        SearchMode::Range {
            step_secs,
            window_secs,
        } => {
            let earliest = departure_time;
            let latest = departure_time + window_secs;
            range_raptor_search(
                data,
                access_stops,
                egress_stops,
                earliest,
                latest,
                step_secs,
                cost_config,
                taxi_config,
                config.max_results,
            )
        }
    }
}

// ── Per-query routing (unchanged logic) ─────────────────────────────

/// Route a single OD pair.
fn route_single(
    data: &TransitData,
    query: &OdQuery,
    query_id: usize,
    cost_config: &CostConfig,
    taxi_config: &TaxiCostConfig,
    config: &BatchConfig,
    street_graph: Option<&StreetGraph>,
) -> Value {
    let effective_taxi_config = effective_taxi(taxi_config, query);

    let mut access_stops = find_access_stops(
        data,
        query.from_lat,
        query.from_lon,
        config.max_access_walk_m,
        config.max_access_stops,
        config.walk_speed,
    );

    if config.enable_taxi_access {
        let taxi_access = find_taxi_access_stops(
            data,
            query.from_lat,
            query.from_lon,
            config.min_taxi_distance_m,
            config.max_taxi_distance_m,
            config.max_taxi_stops,
            &effective_taxi_config,
            street_graph,
        );
        access_stops.extend(taxi_access);
    }

    if access_stops.is_empty() {
        return empty_result(query_id);
    }

    let mut egress_stops = find_access_stops(
        data,
        query.to_lat,
        query.to_lon,
        config.max_egress_walk_m,
        config.max_egress_stops,
        config.walk_speed,
    );

    if config.enable_taxi_access {
        let taxi_egress = find_taxi_access_stops(
            data,
            query.to_lat,
            query.to_lon,
            config.min_taxi_distance_m,
            config.max_taxi_distance_m,
            config.max_taxi_stops,
            &effective_taxi_config,
            street_graph,
        );
        egress_stops.extend(taxi_egress);
    }

    if egress_stops.is_empty() {
        return empty_result(query_id);
    }

    let paths = run_search(
        data,
        &access_stops,
        &egress_stops,
        query.departure_time,
        cost_config,
        &effective_taxi_config,
        config,
    );

    paths_to_otp_json(
        data,
        &paths,
        query_id,
        query.from_lat,
        query.from_lon,
        query.to_lat,
        query.to_lon,
        query.departure_time,
        &config.fare_config,
        config.use_route_shapes,
        if config.use_route_shapes { street_graph } else { None },
    )
}

fn empty_result(query_id: usize) -> Value {
    serde_json::json!({
        "id": query_id,
        "data": {
            "plan": {
                "itineraries": []
            }
        }
    })
}

// ═══════════════════════════════════════════════════════════════════
// Simulation-mode: attributes-only batch (no JSON/polyline overhead)
// ═══════════════════════════════════════════════════════════════════

/// Flat simulation attributes for the best (lowest GC) itinerary per query.
/// 15 f32 fields × N queries → returned as columnar numpy arrays.
#[derive(Clone, Copy, Default)]
pub struct SimAttrs {
    pub duration_sec: f32,
    pub generalized_cost: f32,
    pub num_transfers: f32,
    pub fare_krw: f32,
    pub access_time_sec: f32,
    pub egress_time_sec: f32,
    pub wait_time_sec: f32,
    pub walk_time_sec: f32,
    pub ivt_total_sec: f32,
    pub ivt_bus_sec: f32,
    pub ivt_subway_sec: f32,
    pub ivt_rail_sec: f32,
    pub walk_distance_m: f32,
    pub taxi_access_fare: f32,
    pub taxi_egress_fare: f32,
}

/// Extract SimAttrs from the best path (lowest GC), skipping JSON entirely.
fn path_to_sim_attrs(
    data: &TransitData,
    path: &crate::raptor::engine::RaptorPath,
    departure_time: u32,
    fare_config: &FareConfig,
) -> SimAttrs {
    use crate::raptor::engine::PathLeg;
    use crate::types::RouteMode;

    let mut a = SimAttrs {
        duration_sec: path.total_duration as f32,
        generalized_cost: path.generalized_cost as f32,
        num_transfers: path.num_transfers as f32,
        walk_distance_m: path.total_walk_dist as f32,
        ..Default::default()
    };

    // Fare
    let fare = fare_config.compute_fare(data, &path.legs);
    a.fare_krw = fare.total_krw as f32;

    // Decompose legs
    let first_transit = path.legs.iter().position(|l| matches!(l, PathLeg::Transit { .. }));
    let last_transit = path.legs.iter().rposition(|l| matches!(l, PathLeg::Transit { .. }));
    let mut current_time = departure_time;
    let mut seen_transit = false;

    for (i, leg) in path.legs.iter().enumerate() {
        match leg {
            PathLeg::Walk { duration_secs, .. } | PathLeg::Taxi { duration_secs, .. } => {
                let dur = *duration_secs as f32;
                let is_taxi = matches!(leg, PathLeg::Taxi { .. });

                if first_transit.is_none() || i < first_transit.unwrap() {
                    a.access_time_sec += dur;
                    if is_taxi {
                        if let PathLeg::Taxi { fare_krw, .. } = leg {
                            a.taxi_access_fare = *fare_krw as f32;
                        }
                    }
                } else if last_transit.is_some() && i > last_transit.unwrap() {
                    a.egress_time_sec += dur;
                    if is_taxi {
                        if let PathLeg::Taxi { fare_krw, .. } = leg {
                            a.taxi_egress_fare = *fare_krw as f32;
                        }
                    }
                } else {
                    a.walk_time_sec += dur;
                }
                current_time += *duration_secs;
            }
            PathLeg::Transit { route_index, board_time, alight_time, .. } => {
                let wait = board_time.saturating_sub(current_time) as f32;
                a.wait_time_sec += wait;

                let ivt = alight_time.saturating_sub(*board_time) as f32;
                a.ivt_total_sec += ivt;

                let route = &data.routes[*route_index as usize];
                match route.mode {
                    RouteMode::Bus => a.ivt_bus_sec += ivt,
                    RouteMode::Subway => a.ivt_subway_sec += ivt,
                    RouteMode::Rail | RouteMode::Gtx => a.ivt_rail_sec += ivt,
                    _ => {}
                }

                current_time = *alight_time;
                seen_transit = true;
            }
        }
    }

    // walk_time = only pure walking (exclude taxi access/egress)
    if !seen_transit {
        a.walk_time_sec = a.access_time_sec;
    }

    a
}

/// Full OD group key: origin_cell + dest_cell + time_bucket.
/// Uses H3 cells for spatial grouping, consistent with ServiceFeed and demand models.
#[derive(Clone, PartialEq, Eq, Hash)]
struct OdGroupKey {
    from_spatial: u64,
    to_spatial: u64,
    departure_time: u32,
}

impl OdGroupKey {
    fn new(query: &OdQuery, h3_resolution: u8, time_bucket_secs: u32) -> Self {
        let dep = if time_bucket_secs > 0 {
            (query.departure_time / time_bucket_secs) * time_bucket_secs
        } else {
            query.departure_time
        };

        Self {
            from_spatial: spatial_key(query.from_lat, query.from_lon, h3_resolution),
            to_spatial: spatial_key(query.to_lat, query.to_lon, h3_resolution),
            departure_time: dep,
        }
    }

    fn origin_lat_lon(&self, h3_resolution: u8) -> (f64, f64) {
        spatial_key_to_latlon(self.from_spatial, h3_resolution)
    }

    fn dest_lat_lon(&self, h3_resolution: u8) -> (f64, f64) {
        spatial_key_to_latlon(self.to_spatial, h3_resolution)
    }
}

/// Batch routing returning only simulation attributes (no JSON/polyline).
///
/// Groups by (origin_cell, dest_cell, time_bucket):
/// - Same OD cell pair + time → ONE compute + ONE extract → shared result
/// - Eliminates both compute AND extract redundancy
///
/// With 200m+120s bucketing, 100K queries collapse to ~few thousand unique OD-time groups.
pub fn route_batch_attrs(
    data: &TransitData,
    queries: &[OdQuery],
    cost_config: &CostConfig,
    taxi_config: &TaxiCostConfig,
    batch_config: &BatchConfig,
    _street_graph: Option<&StreetGraph>,
) -> Vec<SimAttrs> {
    if queries.is_empty() {
        return Vec::new();
    }

    // Phase 1: Group by (origin_cell, dest_cell, time_bucket)
    let mut groups: HashMap<OdGroupKey, Vec<usize>> = HashMap::new();
    for (qi, query) in queries.iter().enumerate() {
        let key = OdGroupKey::new(query, batch_config.h3_resolution, batch_config.time_bucket_secs);
        groups.entry(key).or_default().push(qi);
    }

    let group_items: Vec<(OdGroupKey, Vec<usize>)> = groups.into_iter().collect();

    // Phase 2: Sub-group by origin (for RAPTOR compute sharing)
    // Multiple OD groups can share the same origin → share RAPTOR compute
    let mut origin_groups: HashMap<(u64, u32), Vec<usize>> = HashMap::new();
    for (gi, (key, _)) in group_items.iter().enumerate() {
        let origin_key = (key.from_spatial, key.departure_time);
        origin_groups.entry(origin_key).or_default().push(gi);
    }

    let origin_items: Vec<((u64, u32), Vec<usize>)> = origin_groups.into_iter().collect();

    // Phase 3: Process origin groups in parallel
    // For each origin: ONE compute, then extract per unique destination
    let scattered: Vec<Vec<(usize, SimAttrs)>> = origin_items
        .par_iter()
        .map(|(_, od_group_indices)| {
            let first_key = &group_items[od_group_indices[0]].0;
            let (rep_lat, rep_lon) = first_key.origin_lat_lon(batch_config.h3_resolution);

            let access_radius = if batch_config.h3_resolution > 0 {
                batch_config.max_access_walk_m + h3_edge_length_m(batch_config.h3_resolution) * 0.7
            } else {
                batch_config.max_access_walk_m
            };
            let access_stops = find_access_stops(
                data, rep_lat, rep_lon, access_radius,
                batch_config.max_access_stops, batch_config.walk_speed,
            );

            if access_stops.is_empty() {
                return od_group_indices.iter().flat_map(|&gi| {
                    group_items[gi].1.iter().map(|&qi| (qi, SimAttrs::default()))
                }).collect();
            }

            // Compute egress for each unique destination in this origin group
            let egress_radius = if batch_config.h3_resolution > 0 {
                batch_config.max_egress_walk_m + h3_edge_length_m(batch_config.h3_resolution) * 0.7
            } else {
                batch_config.max_egress_walk_m
            };
            let dest_egress: Vec<Vec<AccessStop>> = od_group_indices.iter().map(|&gi| {
                let (dlat, dlon) = group_items[gi].0.dest_lat_lon(batch_config.h3_resolution);
                find_access_stops(
                    data, dlat, dlon, egress_radius,
                    batch_config.max_egress_stops, batch_config.walk_speed,
                )
            }).collect();

            // Merge all egress stops for early termination bound
            let mut merged_egress: Vec<AccessStop> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for eg_list in &dest_egress {
                for eg in eg_list {
                    if seen.insert(eg.stop_index) {
                        merged_egress.push(eg.clone());
                    }
                }
            }

            let effective_taxi = effective_taxi(taxi_config, &queries[group_items[od_group_indices[0]].1[0]]);

            // ONE RAPTOR compute for all destinations from this origin
            match batch_config.search_mode {
                SearchMode::Standard => {
                    let state = raptor_compute(
                        data, &access_stops, &merged_egress,
                        first_key.departure_time, cost_config,
                    );

                    // Extract once per unique destination, share result with all members
                    od_group_indices.iter().enumerate().flat_map(|(di, &gi)| {
                        let members = &group_items[gi].1;
                        if dest_egress[di].is_empty() {
                            return members.iter().map(|&qi| (qi, SimAttrs::default())).collect::<Vec<_>>();
                        }
                        let paths = raptor_extract(
                            data, &state, &access_stops, &dest_egress[di],
                            first_key.departure_time, cost_config, batch_config.max_results,
                        );
                        let attrs = paths.first()
                            .map(|p| path_to_sim_attrs(data, p, first_key.departure_time, &batch_config.fare_config))
                            .unwrap_or_default();
                        // All members of this OD group get the same result
                        members.iter().map(|&qi| (qi, attrs)).collect::<Vec<_>>()
                    }).collect()
                }
                _ => {
                    let (relax_ratio, relax_slack) = match batch_config.search_mode {
                        SearchMode::Research => (1.3, 0),
                        _ => (batch_config.mc_relax_ratio, batch_config.mc_relax_slack),
                    };

                    let state = mc_raptor_compute(
                        data, &access_stops, &merged_egress,
                        first_key.departure_time, cost_config,
                        &effective_taxi, relax_ratio, relax_slack,
                    );

                    od_group_indices.iter().enumerate().flat_map(|(di, &gi)| {
                        let members = &group_items[gi].1;
                        if dest_egress[di].is_empty() {
                            return members.iter().map(|&qi| (qi, SimAttrs::default())).collect::<Vec<_>>();
                        }
                        let paths = mc_raptor_extract(
                            data, &state, &access_stops, &dest_egress[di],
                            first_key.departure_time, cost_config,
                            &effective_taxi, batch_config.max_results,
                        );
                        let attrs = paths.first()
                            .map(|p| path_to_sim_attrs(data, p, first_key.departure_time, &batch_config.fare_config))
                            .unwrap_or_default();
                        members.iter().map(|&qi| (qi, attrs)).collect::<Vec<_>>()
                    }).collect()
                }
            }
        })
        .collect();

    // Scatter back to original order
    let mut results = vec![SimAttrs::default(); queries.len()];
    for group in scattered {
        for (qi, attrs) in group {
            results[qi] = attrs;
        }
    }
    results
}
