use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use rstar::RTree;
use serde::{Deserialize, Serialize};

use crate::error::RaptorError;
use crate::gtfs::loader;
use crate::gtfs::types::{GtfsFrequency, GtfsTransfer};
use crate::types::*;

/// Serializable subset of TransitData (RTree excluded — rebuilt on load).
#[derive(Serialize, Deserialize)]
struct TransitDataCache {
    stop_ids: Vec<String>,
    stop_names: Vec<String>,
    stop_lats: Vec<f64>,
    stop_lons: Vec<f64>,
    routes: Vec<Route>,
    transfers_from: Vec<Vec<Transfer>>,
    routes_by_stop: Vec<Vec<(u32, u16)>>,
    service_start: u32,
    service_end: u32,
}

const CACHE_FILE: &str = ".raptor_cache.bin";
const CACHE_VERSION: u32 = 4;

/// Max walking transfer distance between stops (meters).
const MAX_TRANSFER_DISTANCE_M: f64 = 500.0;

/// Walking speed for transfer time calculation (m/s).
const WALK_SPEED_MPS: f64 = 1.2;

/// Maximum haversine distance between first/last stops for round-trip detection (meters).
const ROUND_TRIP_THRESHOLD_M: f64 = 500.0;

/// Minimum stops required to attempt round-trip splitting.
const ROUND_TRIP_MIN_STOPS: usize = 10;

/// Build TransitData from a GTFS directory, with binary cache.
///
/// First load: parses CSV (~5s for Seoul) → saves `.raptor_cache.bin`.
/// Subsequent loads: reads binary cache (<0.5s).
pub fn build_transit_data(gtfs_dir: &Path) -> Result<TransitData, RaptorError> {
    // Try binary cache first
    if let Some(data) = try_load_cache(gtfs_dir) {
        return Ok(data);
    }

    let data = build_transit_data_from_csv(gtfs_dir)?;

    // Save cache for next time
    save_cache(gtfs_dir, &data);

    Ok(data)
}

fn try_load_cache(gtfs_dir: &Path) -> Option<TransitData> {
    let cache_path = gtfs_dir.join(CACHE_FILE);
    if !cache_path.exists() {
        return None;
    }

    // Check if any GTFS txt file is newer than cache
    let cache_mtime = fs::metadata(&cache_path).ok()?.modified().ok()?;
    for entry in fs::read_dir(gtfs_dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("txt") {
            if let Ok(meta) = fs::metadata(&path) {
                if let Ok(mtime) = meta.modified() {
                    if mtime > cache_mtime {
                        eprintln!("[RAPTOR] Cache stale ({}), rebuilding...", path.display());
                        return None;
                    }
                }
            }
        }
    }

    let start = Instant::now();
    let bytes = fs::read(&cache_path).ok()?;

    // Version check
    if bytes.len() < 4 {
        return None;
    }
    let version = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if version != CACHE_VERSION {
        eprintln!("[RAPTOR] Cache version mismatch (got {}, need {}), rebuilding...", version, CACHE_VERSION);
        return None;
    }

    let cache: TransitDataCache = bincode::deserialize(&bytes[4..]).ok()?;

    // Rebuild RTree from cached coordinates
    let stop_points: Vec<StopPoint> = (0..cache.stop_ids.len())
        .map(|i| StopPoint {
            lat: cache.stop_lats[i],
            lon: cache.stop_lons[i],
            index: i as u32,
        })
        .collect();
    let stop_rtree = RTree::bulk_load(stop_points);

    let data = TransitData {
        stop_ids: cache.stop_ids,
        stop_names: cache.stop_names,
        stop_lats: cache.stop_lats,
        stop_lons: cache.stop_lons,
        routes: cache.routes,
        transfers_from: cache.transfers_from,
        routes_by_stop: cache.routes_by_stop,
        service_start: cache.service_start,
        service_end: cache.service_end,
        stop_rtree,
    };

    eprintln!(
        "[RAPTOR] Loaded from cache: {} stops, {} routes ({:.1}ms)",
        data.stop_count(),
        data.route_count(),
        start.elapsed().as_secs_f64() * 1000.0
    );

    Some(data)
}

fn save_cache(gtfs_dir: &Path, data: &TransitData) {
    let cache = TransitDataCache {
        stop_ids: data.stop_ids.clone(),
        stop_names: data.stop_names.clone(),
        stop_lats: data.stop_lats.clone(),
        stop_lons: data.stop_lons.clone(),
        routes: data.routes.clone(),
        transfers_from: data.transfers_from.clone(),
        routes_by_stop: data.routes_by_stop.clone(),
        service_start: data.service_start,
        service_end: data.service_end,
    };

    if let Ok(bytes) = bincode::serialize(&cache) {
        let cache_path = gtfs_dir.join(CACHE_FILE);
        let mut out = CACHE_VERSION.to_le_bytes().to_vec();
        out.extend(bytes);
        if let Err(e) = fs::write(&cache_path, &out) {
            eprintln!("[RAPTOR] Warning: failed to write cache: {}", e);
        } else {
            eprintln!(
                "[RAPTOR] Cache saved: {:.1} MB → {}",
                out.len() as f64 / 1e6,
                cache_path.display()
            );
        }
    }
}

/// Build TransitData by parsing GTFS CSV files.
fn build_transit_data_from_csv(gtfs_dir: &Path) -> Result<TransitData, RaptorError> {
    let total_start = Instant::now();

    // 1. Load raw GTFS files
    eprintln!("[RAPTOR] Loading GTFS from {:?}...", gtfs_dir);

    let start = Instant::now();
    let raw_stops = loader::load_stops(gtfs_dir)?;
    eprintln!(
        "[RAPTOR]   stops.txt: {} stops ({:.1}ms)",
        raw_stops.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );

    let start = Instant::now();
    let raw_routes = loader::load_routes(gtfs_dir)?;
    eprintln!(
        "[RAPTOR]   routes.txt: {} routes ({:.1}ms)",
        raw_routes.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );

    let start = Instant::now();
    let raw_trips = loader::load_trips(gtfs_dir)?;
    eprintln!(
        "[RAPTOR]   trips.txt: {} trips ({:.1}ms)",
        raw_trips.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );

    let start = Instant::now();
    let stop_times_by_trip = loader::load_stop_times(gtfs_dir)?;
    eprintln!(
        "[RAPTOR]   stop_times.txt: {} trips with times ({:.1}ms)",
        stop_times_by_trip.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );

    let start = Instant::now();
    let gtfs_transfers = loader::load_transfers(gtfs_dir)?;
    eprintln!(
        "[RAPTOR]   transfers.txt: {} transfers ({:.1}ms)",
        gtfs_transfers.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );

    let start = Instant::now();
    let gtfs_frequencies = loader::load_frequencies(gtfs_dir)?;
    eprintln!(
        "[RAPTOR]   frequencies.txt: {} trips ({:.1}ms)",
        gtfs_frequencies.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );

    // 2. Build stop index
    let start = Instant::now();
    let stop_id_to_index: HashMap<&str, u32> = raw_stops
        .iter()
        .enumerate()
        .map(|(i, s)| (s.stop_id.as_str(), i as u32))
        .collect();

    let stop_ids: Vec<String> = raw_stops.iter().map(|s| s.stop_id.clone()).collect();
    let stop_names: Vec<String> = raw_stops.iter().map(|s| s.stop_name.clone()).collect();
    let stop_lats: Vec<f64> = raw_stops.iter().map(|s| s.stop_lat).collect();
    let stop_lons: Vec<f64> = raw_stops.iter().map(|s| s.stop_lon).collect();
    let stop_count = raw_stops.len();
    eprintln!(
        "[RAPTOR]   Stop index: {} stops ({:.1}ms)",
        stop_count,
        start.elapsed().as_secs_f64() * 1000.0
    );

    // 2b. Load shapes.txt (optional)
    let start = Instant::now();
    let shapes = loader::load_shapes(gtfs_dir)?;
    if !shapes.is_empty() {
        eprintln!(
            "[RAPTOR]   shapes.txt: {} shapes ({:.1}ms)",
            shapes.len(),
            start.elapsed().as_secs_f64() * 1000.0
        );
    }

    // 3. Group trips into patterns and build routes
    let start = Instant::now();
    let routes = build_routes(
        &raw_trips,
        &raw_routes,
        &stop_times_by_trip,
        &stop_id_to_index,
        &gtfs_frequencies,
        &shapes,
        &stop_lats,
        &stop_lons,
    );
    let total_trips: usize = routes.iter().map(|r| r.timetable.schedules.len()).sum();
    eprintln!(
        "[RAPTOR]   Patterns: {} routes, {} trips ({:.1}ms)",
        routes.len(),
        total_trips,
        start.elapsed().as_secs_f64() * 1000.0
    );

    // 4. Build routes_by_stop reverse index
    let start = Instant::now();
    let routes_by_stop = build_routes_by_stop(stop_count, &routes);
    eprintln!(
        "[RAPTOR]   Routes-by-stop index ({:.1}ms)",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // 5. Build spatial index (R-tree) for stops
    let start = Instant::now();
    let stop_points: Vec<StopPoint> = (0..stop_count)
        .map(|i| StopPoint {
            lat: stop_lats[i],
            lon: stop_lons[i],
            index: i as u32,
        })
        .collect();
    let stop_rtree = RTree::bulk_load(stop_points);
    eprintln!(
        "[RAPTOR]   R-tree built ({:.1}ms)",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // 6. Build transfers using R-tree (rayon parallel) + merge GTFS transfers.txt
    let start = Instant::now();
    let mut transfers_from =
        build_transfers(stop_count, &stop_lats, &stop_lons, &stop_rtree);
    let generated_count: usize = transfers_from.iter().map(|v| v.len()).sum();

    // Merge GTFS transfers.txt metadata (transfer_type, min_transfer_time)
    let gtfs_applied = merge_gtfs_transfers(
        &mut transfers_from,
        &gtfs_transfers,
        &stop_id_to_index,
        &stop_lats,
        &stop_lons,
    );
    let transfer_count: usize = transfers_from.iter().map(|v| v.len()).sum();
    eprintln!(
        "[RAPTOR]   Transfers: {} generated, {} GTFS overrides, {} total ({:.1}ms)",
        generated_count,
        gtfs_applied,
        transfer_count,
        start.elapsed().as_secs_f64() * 1000.0
    );

    // 7. Determine service time window
    let (service_start, service_end) = compute_service_window(&routes);

    eprintln!(
        "[RAPTOR] TransitData built: {} stops, {} routes, {} trips, {} transfers ({:.1}s total)",
        stop_count,
        routes.len(),
        total_trips,
        transfer_count,
        total_start.elapsed().as_secs_f64()
    );

    Ok(TransitData {
        stop_ids,
        stop_names,
        stop_lats,
        stop_lons,
        routes,
        transfers_from,
        routes_by_stop,
        service_start,
        service_end,
        stop_rtree,
    })
}

/// Group trips by (route_id, stop_sequence) pattern → build Route objects.
fn build_routes(
    raw_trips: &HashMap<String, crate::gtfs::types::GtfsTrip>,
    raw_routes: &HashMap<String, crate::gtfs::types::GtfsRoute>,
    stop_times_by_trip: &HashMap<String, Vec<crate::gtfs::types::GtfsStopTime>>,
    stop_id_to_index: &HashMap<&str, u32>,
    gtfs_frequencies: &HashMap<String, Vec<GtfsFrequency>>,
    shapes: &HashMap<String, Vec<(f64, f64)>>,
    stop_lats: &[f64],
    stop_lons: &[f64],
) -> Vec<Route> {
    // pattern_key → list of (trip_id, sorted stop_times)
    let mut pattern_groups: HashMap<String, Vec<(&str, &Vec<crate::gtfs::types::GtfsStopTime>)>> =
        HashMap::new();

    for (trip_id, stop_times) in stop_times_by_trip {
        if stop_times.is_empty() {
            continue;
        }

        let trip = match raw_trips.get(trip_id) {
            Some(t) => t,
            None => continue,
        };

        // Pattern key: route_id + ordered stop IDs
        let mut key = trip.route_id.clone();
        key.push(':');
        for st in stop_times {
            key.push_str(&st.stop_id);
            key.push(',');
        }

        pattern_groups
            .entry(key)
            .or_default()
            .push((trip_id.as_str(), stop_times));
    }

    let mut routes = Vec::with_capacity(pattern_groups.len());
    let mut split_count = 0u32;

    for (_key, trips) in &pattern_groups {
        if trips.is_empty() {
            continue;
        }

        let first_trip_id = trips[0].0;
        let first_stop_times = trips[0].1;

        // Look up GTFS route info
        let trip_info = match raw_trips.get(first_trip_id) {
            Some(t) => t,
            None => continue,
        };
        let gtfs_route = match raw_routes.get(&trip_info.route_id) {
            Some(r) => r,
            None => continue,
        };

        // Build stop index array
        let mut stop_indices = Vec::with_capacity(first_stop_times.len());
        let mut valid = true;
        for st in first_stop_times {
            match stop_id_to_index.get(st.stop_id.as_str()) {
                Some(&idx) => stop_indices.push(idx),
                None => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid || stop_indices.is_empty() {
            continue;
        }

        let mode = RouteMode::from_gtfs_route_type(gtfs_route.route_type);

        let pattern = TripPattern {
            stop_indices,
            slack_index: mode.slack_index(),
        };

        // Build schedules for all trips in this pattern
        let mut schedules = Vec::with_capacity(trips.len());
        for &(trip_id, stop_times) in trips {
            if stop_times.len() != pattern.num_stops() {
                continue; // mismatched stop count, skip
            }

            let mut arrival_times = Vec::with_capacity(stop_times.len());
            let mut departure_times = Vec::with_capacity(stop_times.len());
            let mut valid = true;

            for st in stop_times {
                if st.arrival_time == 0 && st.departure_time == 0 {
                    valid = false;
                    break;
                }
                arrival_times.push(st.arrival_time);
                departure_times.push(st.departure_time);
            }

            if !valid {
                continue;
            }

            schedules.push(TripSchedule {
                trip_id: trip_id.to_string(),
                route_short_name: gtfs_route.route_short_name.clone(),
                arrival_times,
                departure_times,
            });
        }

        if schedules.is_empty() {
            continue;
        }

        // Sort by first departure time (position 0 is guaranteed sorted)
        schedules.sort_by_key(|s| s.departure_times[0]);

        // Collect frequency entries for trips in this pattern
        let mut frequencies: Vec<FrequencyEntry> = Vec::new();
        for &(trip_id, _) in trips {
            if let Some(freqs) = gtfs_frequencies.get(trip_id) {
                for f in freqs {
                    frequencies.push(FrequencyEntry {
                        start_time: f.start_time,
                        end_time: f.end_time,
                        headway_secs: f.headway_secs,
                        exact_times: f.exact_times,
                    });
                }
            }
        }

        // For frequency-based routes with exact_times, expand to synthetic schedules
        if !frequencies.is_empty() && frequencies.iter().any(|f| f.exact_times) {
            expand_frequency_schedules(&mut schedules, &frequencies);
            schedules.sort_by_key(|s| s.departure_times[0]);
        }

        // Split round-trip patterns (A→B→A) into outbound + return FIRST,
        // then resolve shapes for each half separately.
        // This avoids shape mapping issues where the same geographic location
        // appears at different positions in a round-trip shape.
        if let Some(split) = try_split_round_trip(
            &pattern,
            &schedules,
            &frequencies,
            trip_info,
            shapes,
            stop_lats,
            stop_lons,
            mode,
            &gtfs_route.route_id,
            &gtfs_route.route_short_name,
            &gtfs_route.route_long_name,
            gtfs_route.route_type,
        ) {
            split_count += 1;
            routes.extend(split);
        } else {
            // Non-round-trip: resolve shape for the full pattern
            let (shape_coords, shape_stop_indices) = resolve_shape_for_pattern(
                trip_info, &pattern, shapes, stop_lats, stop_lons,
            );
            let mut timetable = TimeTable {
                schedules,
                frequencies,
                sorted_at_all_positions: false,
            };
            timetable.validate_sort_order();

            routes.push(Route {
                pattern,
                timetable,
                route_id: gtfs_route.route_id.clone(),
                route_short_name: gtfs_route.route_short_name.clone(),
                route_long_name: gtfs_route.route_long_name.clone(),
                route_type: gtfs_route.route_type,
                mode,
                reliability: None,
                crowding: None,
                shape_coords,
                shape_stop_indices,
            });
        }
    }

    if split_count > 0 {
        eprintln!(
            "[RAPTOR]   Round-trip split: {} patterns → {} routes",
            split_count,
            split_count * 2
        );
    }

    // Report overtaking statistics
    let unsorted_count = routes
        .iter()
        .filter(|r| !r.timetable.sorted_at_all_positions)
        .count();
    if unsorted_count > 0 {
        eprintln!(
            "[RAPTOR]   Trip overtaking: {}/{} routes have non-monotonic departure times (using linear scan fallback)",
            unsorted_count,
            routes.len()
        );
    }

    routes
}

/// Detect if a stop pattern forms a round-trip (first stop ≈ last stop).
fn is_round_trip_pattern(stop_indices: &[u32], stop_lats: &[f64], stop_lons: &[f64]) -> bool {
    if stop_indices.len() < ROUND_TRIP_MIN_STOPS {
        return false;
    }
    let first = stop_indices[0] as usize;
    let last = *stop_indices.last().unwrap() as usize;
    if first == last {
        return true;
    }
    let dist = haversine_distance(
        stop_lats[first],
        stop_lons[first],
        stop_lats[last],
        stop_lons[last],
    );
    dist < ROUND_TRIP_THRESHOLD_M
}

/// Find the turnaround point: the stop position farthest from the first stop.
fn find_turnaround_point(
    stop_indices: &[u32],
    stop_lats: &[f64],
    stop_lons: &[f64],
) -> usize {
    let first = stop_indices[0] as usize;
    let (lat0, lon0) = (stop_lats[first], stop_lons[first]);

    let mut best_pos = 0usize;
    let mut best_dist = 0.0f64;

    for (pos, &si) in stop_indices.iter().enumerate() {
        let si = si as usize;
        let d = haversine_distance(lat0, lon0, stop_lats[si], stop_lons[si]);
        if d > best_dist {
            best_dist = d;
            best_pos = pos;
        }
    }

    best_pos
}

/// If the pattern is a round-trip, split into outbound + return Route pair.
/// The turnaround stop is shared (last outbound = first return).
fn try_split_round_trip(
    pattern: &TripPattern,
    schedules: &[TripSchedule],
    frequencies: &[FrequencyEntry],
    trip_info: &crate::gtfs::types::GtfsTrip,
    shapes: &HashMap<String, Vec<(f64, f64)>>,
    stop_lats: &[f64],
    stop_lons: &[f64],
    mode: RouteMode,
    route_id: &str,
    route_short_name: &str,
    route_long_name: &str,
    route_type: u16,
) -> Option<Vec<Route>> {
    if !is_round_trip_pattern(&pattern.stop_indices, stop_lats, stop_lons) {
        return None;
    }

    let mid = find_turnaround_point(&pattern.stop_indices, stop_lats, stop_lons);
    let n = pattern.stop_indices.len();
    if mid < 2 || n - mid < 3 {
        return None;
    }

    // --- Split patterns (turnaround stop shared) ---
    let out_stops = pattern.stop_indices[..=mid].to_vec();
    let ret_stops = pattern.stop_indices[mid..].to_vec();

    // --- Split schedules ---
    let mut out_schedules = Vec::with_capacity(schedules.len());
    let mut ret_schedules = Vec::with_capacity(schedules.len());
    for s in schedules {
        out_schedules.push(TripSchedule {
            trip_id: format!("{}_O", s.trip_id),
            route_short_name: s.route_short_name.clone(),
            arrival_times: s.arrival_times[..=mid].to_vec(),
            departure_times: s.departure_times[..=mid].to_vec(),
        });
        ret_schedules.push(TripSchedule {
            trip_id: format!("{}_R", s.trip_id),
            route_short_name: s.route_short_name.clone(),
            arrival_times: s.arrival_times[mid..].to_vec(),
            departure_times: s.departure_times[mid..].to_vec(),
        });
    }
    out_schedules.sort_by_key(|s| s.departure_times[0]);
    ret_schedules.sort_by_key(|s| s.departure_times[0]);

    // --- Resolve shapes for each half using shape slicing ---
    // The original shape covers the full round-trip (A→B→A).
    // We split the shape at its midpoint and resolve each half independently.
    let (out_shape, out_shape_idx, ret_shape, ret_shape_idx) = {
        let shape_id = trip_info.shape_id.as_deref().and_then(|id| shapes.get(id));
        if let Some(full_coords) = shape_id {
            if full_coords.len() >= 4 {
                // Find shape midpoint: farthest point from shape start
                let (slat, slon) = full_coords[0];
                let shape_mid = full_coords
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        let da = (a.0 - slat).powi(2) + (a.1 - slon).powi(2);
                        let db = (b.0 - slat).powi(2) + (b.1 - slon).powi(2);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(full_coords.len() / 2);

                let out_half = &full_coords[..=shape_mid];
                let ret_half = &full_coords[shape_mid..];

                // Resolve outbound pattern against first half of shape
                let out_pattern_tmp = TripPattern {
                    stop_indices: out_stops.clone(),
                    slack_index: pattern.slack_index,
                };
                let out_si = resolve_shape_indices(&out_pattern_tmp, out_half, stop_lats, stop_lons);

                // Resolve return pattern against second half of shape
                let ret_pattern_tmp = TripPattern {
                    stop_indices: ret_stops.clone(),
                    slack_index: pattern.slack_index,
                };
                let ret_si = resolve_shape_indices(&ret_pattern_tmp, ret_half, stop_lats, stop_lons);

                (out_half.to_vec(), out_si, ret_half.to_vec(), ret_si)
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            }
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        }
    };

    let mut out_timetable = TimeTable {
        schedules: out_schedules,
        frequencies: frequencies.to_vec(),
        sorted_at_all_positions: false,
    };
    out_timetable.validate_sort_order();

    let mut ret_timetable = TimeTable {
        schedules: ret_schedules,
        frequencies: frequencies.to_vec(),
        sorted_at_all_positions: false,
    };
    ret_timetable.validate_sort_order();

    let out_route = Route {
        pattern: TripPattern {
            stop_indices: out_stops,
            slack_index: mode.slack_index(),
        },
        timetable: out_timetable,
        route_id: format!("{}_O", route_id),
        route_short_name: route_short_name.to_string(),
        route_long_name: route_long_name.to_string(),
        route_type,
        mode,
        reliability: None,
        crowding: None,
        shape_coords: out_shape,
        shape_stop_indices: out_shape_idx,
    };

    let ret_route = Route {
        pattern: TripPattern {
            stop_indices: ret_stops,
            slack_index: mode.slack_index(),
        },
        timetable: ret_timetable,
        route_id: format!("{}_R", route_id),
        route_short_name: route_short_name.to_string(),
        route_long_name: route_long_name.to_string(),
        route_type,
        mode,
        reliability: None,
        crowding: None,
        shape_coords: ret_shape,
        shape_stop_indices: ret_shape_idx,
    };

    Some(vec![out_route, ret_route])
}

/// Resolve shape polyline for a route pattern.
///
/// For each stop in the pattern, finds the nearest shape point — searching
/// strictly forward from the previous match. This ensures monotonically
/// increasing indices even on round-trip routes (A→B→A) where the same
/// geographic location appears at different positions in the shape.
///
/// Returns (shape_coords, shape_stop_indices) or empty vecs if no shape available.
fn resolve_shape_for_pattern(
    trip_info: &crate::gtfs::types::GtfsTrip,
    pattern: &TripPattern,
    shapes: &HashMap<String, Vec<(f64, f64)>>,
    stop_lats: &[f64],
    stop_lons: &[f64],
) -> (Vec<(f64, f64)>, Vec<usize>) {
    if shapes.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let shape_id = match &trip_info.shape_id {
        Some(id) => id,
        None => return (Vec::new(), Vec::new()),
    };

    let coords = match shapes.get(shape_id) {
        Some(c) if c.len() >= 2 => c,
        _ => return (Vec::new(), Vec::new()),
    };

    // For each stop, find nearest shape point searching forward only.
    // search_from advances strictly so round-trip shapes work correctly:
    // stop A(pos 10) → stop B(pos 50) → stop A(pos 90) all get unique indices.
    let mut stop_indices = Vec::with_capacity(pattern.stop_indices.len());
    let mut search_from = 0usize;

    for &si in &pattern.stop_indices {
        let si = si as usize;
        let slat = stop_lats[si];
        let slon = stop_lons[si];

        let mut best_idx = search_from;
        let mut best_dist = f64::MAX;
        for i in search_from..coords.len() {
            let (clat, clon) = coords[i];
            let d = (clat - slat).powi(2) + (clon - slon).powi(2);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        stop_indices.push(best_idx);
        // Advance past current match for next stop (strictly monotonic)
        search_from = best_idx + 1;
        if search_from >= coords.len() {
            // If we've exhausted the shape, clamp remaining stops to the last point
            for _ in stop_indices.len()..pattern.stop_indices.len() {
                stop_indices.push(coords.len() - 1);
            }
            break;
        }
    }

    (coords.clone(), stop_indices)
}

/// Resolve shape stop indices for a pattern against a shape coordinate slice.
/// Returns Vec of indices into the coords slice, one per pattern stop.
fn resolve_shape_indices(
    pattern: &TripPattern,
    coords: &[(f64, f64)],
    stop_lats: &[f64],
    stop_lons: &[f64],
) -> Vec<usize> {
    if coords.len() < 2 {
        return Vec::new();
    }
    let mut stop_indices = Vec::with_capacity(pattern.stop_indices.len());
    let mut search_from = 0usize;

    for &si in &pattern.stop_indices {
        let si = si as usize;
        let slat = stop_lats[si];
        let slon = stop_lons[si];

        // Search from current position to end of shape (no window limit).
        // Since shape is already split to one direction, full search is safe.
        let mut best_idx = search_from;
        let mut best_dist = f64::MAX;
        for i in search_from..coords.len() {
            let (clat, clon) = coords[i];
            let d = (clat - slat).powi(2) + (clon - slon).powi(2);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        stop_indices.push(best_idx);
        search_from = best_idx + 1;
        if search_from >= coords.len() {
            for _ in stop_indices.len()..pattern.stop_indices.len() {
                stop_indices.push(coords.len() - 1);
            }
            break;
        }
    }
    stop_indices
}

/// Expand frequency-based schedules into explicit synthetic trips.
fn expand_frequency_schedules(
    schedules: &mut Vec<TripSchedule>,
    frequencies: &[FrequencyEntry],
) {
    if schedules.is_empty() {
        return;
    }

    // Use the first schedule as a template (relative offsets)
    let template = &schedules[0];
    let base_dep = template.departure_times[0];
    let offsets_arr: Vec<u32> = template
        .arrival_times
        .iter()
        .map(|&t| t.saturating_sub(base_dep))
        .collect();
    let offsets_dep: Vec<u32> = template
        .departure_times
        .iter()
        .map(|&t| t.saturating_sub(base_dep))
        .collect();
    let route_name = template.route_short_name.clone();
    let num_stops = template.arrival_times.len();

    for freq in frequencies.iter().filter(|f| f.exact_times) {
        let mut dep = freq.start_time;
        let mut trip_counter = 0u32;
        while dep < freq.end_time {
            // Skip if already covered by an existing schedule
            let already_exists = schedules.iter().any(|s| {
                (s.departure_times[0] as i64 - dep as i64).unsigned_abs() < 30
            });
            if !already_exists {
                let arr_times: Vec<u32> = offsets_arr.iter().map(|&o| dep + o).collect();
                let dep_times: Vec<u32> = offsets_dep.iter().map(|&o| dep + o).collect();

                if arr_times.len() == num_stops {
                    schedules.push(TripSchedule {
                        trip_id: format!("freq_{}_{}", route_name, trip_counter),
                        route_short_name: route_name.clone(),
                        arrival_times: arr_times,
                        departure_times: dep_times,
                    });
                }
            }
            dep += freq.headway_secs;
            trip_counter += 1;
        }
    }
}

/// Build reverse index: stop → [(route_index, position_in_route)].
///
/// Pre-computes the stop's position within each route pattern,
/// avoiding O(n) .position() search during routing (Solari optimization).
fn build_routes_by_stop(stop_count: usize, routes: &[Route]) -> Vec<Vec<(u32, u16)>> {
    let mut by_stop: Vec<Vec<(u32, u16)>> = vec![Vec::new(); stop_count];

    for (route_idx, route) in routes.iter().enumerate() {
        for (pos, &stop_idx) in route.pattern.stop_indices.iter().enumerate() {
            let si = stop_idx as usize;
            let ri = route_idx as u32;
            // A stop may appear multiple times in a route (loops); store first occurrence
            if !by_stop[si].iter().any(|&(r, _)| r == ri) {
                by_stop[si].push((ri, pos as u16));
            }
        }
    }

    by_stop
}

/// Build walking transfers between nearby stops using R-tree.
fn build_transfers(
    stop_count: usize,
    stop_lats: &[f64],
    stop_lons: &[f64],
    rtree: &RTree<StopPoint>,
) -> Vec<Vec<Transfer>> {
    // Approximate degree range for MAX_TRANSFER_DISTANCE_M
    let lat_range = MAX_TRANSFER_DISTANCE_M / 111_000.0;
    let lon_range_approx = MAX_TRANSFER_DISTANCE_M / 85_000.0; // conservative at ~37°N

    // Build transfers in parallel per stop
    (0..stop_count)
        .into_par_iter()
        .map(|i| {
            let lat = stop_lats[i];
            let lon = stop_lons[i];

            let envelope = rstar::AABB::from_corners(
                [lat - lat_range, lon - lon_range_approx],
                [lat + lat_range, lon + lon_range_approx],
            );

            let mut transfers = Vec::new();
            for point in rtree.locate_in_envelope(&envelope) {
                if point.index == i as u32 {
                    continue; // skip self
                }

                let dist = haversine_distance(lat, lon, point.lat, point.lon);
                if dist <= MAX_TRANSFER_DISTANCE_M {
                    let duration = (dist / WALK_SPEED_MPS).ceil() as u32;
                    transfers.push(Transfer {
                        to_stop: point.index,
                        duration_secs: duration,
                        distance_m: dist as f32,
                        transfer_type: TransferType::Recommended,
                    });
                }
            }

            transfers
        })
        .collect()
}

/// Merge GTFS transfers.txt metadata into generated transfers.
///
/// - Updates transfer_type for known stop pairs
/// - Overrides min_transfer_time when specified
/// - Adds new transfers for pairs not covered by distance-based generation
/// - Removes transfers marked as NotPossible (type=3)
fn merge_gtfs_transfers(
    transfers_from: &mut Vec<Vec<Transfer>>,
    gtfs_transfers: &[GtfsTransfer],
    stop_id_to_index: &HashMap<&str, u32>,
    stop_lats: &[f64],
    stop_lons: &[f64],
) -> usize {
    let mut applied = 0usize;

    for gt in gtfs_transfers {
        let from_idx = match stop_id_to_index.get(gt.from_stop_id.as_str()) {
            Some(&i) => i as usize,
            None => continue,
        };
        let to_idx = match stop_id_to_index.get(gt.to_stop_id.as_str()) {
            Some(&i) => i,
            None => continue,
        };

        let transfer_type = TransferType::from_gtfs(gt.transfer_type);

        // Remove transfers marked as not possible
        if transfer_type == TransferType::NotPossible {
            transfers_from[from_idx].retain(|t| t.to_stop != to_idx);
            applied += 1;
            continue;
        }

        // Find existing transfer and update, or add new one
        if let Some(existing) = transfers_from[from_idx]
            .iter_mut()
            .find(|t| t.to_stop == to_idx)
        {
            existing.transfer_type = transfer_type;
            if let Some(min_time) = gt.min_transfer_time {
                existing.duration_secs = existing.duration_secs.max(min_time);
            }
            // In-station transfers get reduced walking time
            if transfer_type == TransferType::InStation || transfer_type == TransferType::Timed {
                if let Some(min_time) = gt.min_transfer_time {
                    existing.duration_secs = min_time;
                }
            }
        } else {
            // Add new transfer not covered by distance-based generation
            let dist = haversine_distance(
                stop_lats[from_idx],
                stop_lons[from_idx],
                stop_lats[to_idx as usize],
                stop_lons[to_idx as usize],
            );
            let duration = gt
                .min_transfer_time
                .unwrap_or_else(|| (dist / WALK_SPEED_MPS).ceil() as u32);
            transfers_from[from_idx].push(Transfer {
                to_stop: to_idx,
                duration_secs: duration,
                distance_m: dist as f32,
                transfer_type,
            });
        }
        applied += 1;
    }

    applied
}

/// Compute service time window from all trip schedules.
fn compute_service_window(routes: &[Route]) -> (u32, u32) {
    let mut min_time = u32::MAX;
    let mut max_time = 0u32;

    for route in routes {
        for schedule in &route.timetable.schedules {
            if let Some(&first_dep) = schedule.departure_times.first() {
                min_time = min_time.min(first_dep);
            }
            if let Some(&last_arr) = schedule.arrival_times.last() {
                max_time = max_time.max(last_arr);
            }
        }
    }

    if min_time == u32::MAX {
        (14400, 93600) // fallback: 04:00 ~ 26:00
    } else {
        (min_time, max_time)
    }
}

/// Refine pre-computed transfers using the street network graph.
///
/// For each stop, runs a single bounded Dijkstra on the walk network,
/// then validates existing haversine-based transfers:
/// - Reachable: update duration/distance to actual network values
/// - Unreachable (physical barrier): remove the transfer
///
/// This runs at setup time (after both GTFS and StreetGraph are loaded).
/// Zero query-time cost — results are stored in `transfers_from`.
///
/// Returns (validated, removed, total_before).
pub fn refine_transfers_with_graph(
    data: &mut TransitData,
    graph: &crate::access::street_graph::StreetGraph,
) -> (usize, usize, usize) {
    let total_before: usize = data.transfers_from.iter().map(|v| v.len()).sum();
    let stop_count = data.stop_count();

    let max_walk_time_ms = (MAX_TRANSFER_DISTANCE_M / WALK_SPEED_MPS * 1000.0) as u32;

    // Phase 1: For each stop, snap to graph and run bounded walk Dijkstra.
    // Collect reachable node sets in parallel.
    let snap_results: Vec<Option<(u32, f64)>> = (0..stop_count)
        .into_par_iter()
        .map(|i| graph.snap(data.stop_lats[i], data.stop_lons[i], 300.0))
        .collect();

    // Phase 2: Refine transfers in parallel per stop.
    // For each stop, run Dijkstra and validate its outgoing transfers.
    let refined: Vec<Vec<Transfer>> = (0..stop_count)
        .into_par_iter()
        .map(|from_stop| {
            let snap = match snap_results[from_stop] {
                Some(s) => s,
                None => {
                    // Can't snap this stop to graph — keep haversine transfers as-is
                    return data.transfers_from[from_stop].clone();
                }
            };

            let (src_node, _snap_dist) = snap;
            let reachable = graph.dijkstra_bounded(src_node, max_walk_time_ms, true);

            // Build lookup: graph_node → (walk_time_ms, walk_dist_m)
            let mut node_map: std::collections::HashMap<u32, (u32, f32)> =
                std::collections::HashMap::with_capacity(reachable.len());
            for &(node, time_ms, dist_m) in &reachable {
                node_map.insert(node, (time_ms, dist_m));
            }

            // Validate each existing transfer
            data.transfers_from[from_stop]
                .iter()
                .filter_map(|transfer| {
                    let to = transfer.to_stop as usize;

                    // Try to snap destination stop to graph
                    let to_snap = snap_results[to]?;
                    let (to_node, _) = to_snap;

                    if src_node == to_node {
                        // Same graph node — keep with minimal time
                        return Some(Transfer {
                            duration_secs: 30, // ~30m walk minimum
                            distance_m: 30.0,
                            ..transfer.clone()
                        });
                    }

                    // Check if destination node is reachable via walk network
                    if let Some(&(walk_time_ms, walk_dist_m)) = node_map.get(&to_node) {
                        let walk_time_secs = (walk_time_ms / 1000).max(1);
                        Some(Transfer {
                            duration_secs: walk_time_secs,
                            distance_m: walk_dist_m,
                            ..transfer.clone()
                        })
                    } else {
                        // Not reachable via walk network — physical barrier
                        None
                    }
                })
                .collect()
        })
        .collect();

    let total_after: usize = refined.iter().map(|v| v.len()).sum();
    let removed = total_before.saturating_sub(total_after);
    let validated = total_after;

    data.transfers_from = refined;

    (validated, removed, total_before)
}
