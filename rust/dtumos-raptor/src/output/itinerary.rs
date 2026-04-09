use serde_json::{json, Value};

use crate::output::polyline::{decode_polyline, encode_polyline};
use crate::raptor::engine::{PathLeg, RaptorPath};
use crate::raptor::fare::FareConfig;
use crate::types::{haversine_distance, RouteMode, TransitData};

/// Ramer-Douglas-Peucker simplification for (lat, lon) coordinates.
/// `epsilon` is in degrees (~0.00005 ≈ 5m at mid-latitudes).
fn simplify_coords(coords: &[(f64, f64)], epsilon: f64) -> Vec<(f64, f64)> {
    if coords.len() <= 2 {
        return coords.to_vec();
    }
    // Find the point with maximum perpendicular distance from the line (first→last)
    let (first, last) = (coords[0], coords[coords.len() - 1]);
    let mut max_dist = 0.0f64;
    let mut max_idx = 0;
    let dx = last.0 - first.0;
    let dy = last.1 - first.1;
    let line_len_sq = dx * dx + dy * dy;

    for (i, &(lat, lon)) in coords.iter().enumerate().skip(1).take(coords.len() - 2) {
        let d = if line_len_sq < 1e-20 {
            let a = lat - first.0;
            let b = lon - first.1;
            (a * a + b * b).sqrt()
        } else {
            let t = ((lat - first.0) * dx + (lon - first.1) * dy) / line_len_sq;
            let t = t.clamp(0.0, 1.0);
            let proj_lat = first.0 + t * dx;
            let proj_lon = first.1 + t * dy;
            let a = lat - proj_lat;
            let b = lon - proj_lon;
            (a * a + b * b).sqrt()
        };
        if d > max_dist {
            max_dist = d;
            max_idx = i;
        }
    }

    if max_dist > epsilon {
        let mut left = simplify_coords(&coords[..=max_idx], epsilon);
        let right = simplify_coords(&coords[max_idx..], epsilon);
        left.pop(); // remove duplicate pivot
        left.extend(right);
        left
    } else {
        vec![first, last]
    }
}

/// Simplification epsilon in degrees. ~5m at 37°N latitude.
const SHAPE_SIMPLIFY_EPSILON: f64 = 0.00005;

/// Convert RAPTOR paths to OTP-compatible JSON format.
///
/// Output matches the format expected by TransitSimulator and mode_choice.py:
/// ```json
/// {
///   "id": 0,
///   "data": {
///     "plan": {
///       "itineraries": [{ "duration": ..., "legs": [...], "attributes": {...} }]
///     }
///   }
/// }
/// ```
pub fn paths_to_otp_json(
    data: &TransitData,
    paths: &[RaptorPath],
    query_id: usize,
    from_lat: f64,
    from_lon: f64,
    to_lat: f64,
    to_lon: f64,
    departure_time: u32,
    fare_config: &FareConfig,
    use_route_shapes: bool,
    street_graph: Option<&crate::access::StreetGraph>,
) -> Value {
    let itineraries: Vec<Value> = paths
        .iter()
        .map(|path| {
            path_to_itinerary(
                data,
                path,
                from_lat,
                from_lon,
                to_lat,
                to_lon,
                departure_time,
                fare_config,
                use_route_shapes,
                street_graph,
            )
        })
        .collect();

    json!({
        "id": query_id,
        "data": {
            "plan": {
                "itineraries": itineraries
            }
        }
    })
}

fn path_to_itinerary(
    data: &TransitData,
    path: &RaptorPath,
    _from_lat: f64,
    _from_lon: f64,
    _to_lat: f64,
    _to_lon: f64,
    departure_time: u32,
    fare_config: &FareConfig,
    use_route_shapes: bool,
    street_graph: Option<&crate::access::StreetGraph>,
) -> Value {
    let raw_legs: Vec<Value> = path
        .legs
        .iter()
        .map(|leg| leg_to_json(data, leg, use_route_shapes, street_graph))
        .collect();

    // Merge consecutive WALK legs (transfer walk + egress walk → single walk)
    let legs = merge_consecutive_walks(raw_legs);

    let transfers = if path.num_transfers > 0 {
        path.num_transfers as i32
    } else {
        legs.iter()
            .filter(|l| {
                let mode = l["mode"].as_str().unwrap_or("");
                mode != "WALK" && mode != "TAXI"
            })
            .count()
            .saturating_sub(1) as i32
    };

    let attributes = compute_attributes(data, &path.legs, departure_time);
    let fare = fare_config.compute_fare(data, &path.legs);

    let mut itinerary = json!({
        "duration": path.total_duration,
        "walkDistance": path.total_walk_dist,
        "generalizedCost": path.generalized_cost,
        "transfers": transfers,
        "legs": legs,
        "attributes": attributes,
        "fare": {
            "total_krw": fare.total_krw,
            "base_fare_krw": fare.base_fare_krw,
            "distance_surcharge_krw": fare.distance_surcharge_krw,
            "free_transfers_used": fare.free_transfers_used,
            "cumulative_distance_m": fare.cumulative_distance_m,
        }
    });

    if let Some(ref cat) = path.category {
        itinerary["category"] = json!(cat);
    }

    itinerary
}

/// Compute decomposed LOS attributes for SP/RP modeling and DL mode choice.
///
/// Walks the leg sequence to extract:
/// - Time decomposition: access, egress, transfer walk, wait, IVT (total + by mode)
/// - Mode indicators: access_mode, egress_mode
/// - Taxi fares: access/egress
/// - Transfer count
///
/// Wait time = time between arriving at a stop and boarding the next transit vehicle.
fn compute_attributes(data: &TransitData, legs: &[PathLeg], departure_time: u32) -> Value {
    let mut access_time_sec: u32 = 0;
    let mut egress_time_sec: u32 = 0;
    let mut transfer_walk_sec: u32 = 0;
    let mut total_wait_sec: u32 = 0;
    let mut first_wait_sec: u32 = 0;

    // IVT by mode
    let mut ivt_total_sec: u32 = 0;
    let mut ivt_bus_sec: u32 = 0;
    let mut ivt_subway_sec: u32 = 0;
    let mut ivt_rail_sec: u32 = 0;
    let mut ivt_gtx_sec: u32 = 0;
    let mut ivt_ferry_sec: u32 = 0;
    let mut ivt_air_sec: u32 = 0;

    let mut num_transit_legs: u32 = 0;
    let mut access_mode = "WALK";
    let mut egress_mode = "WALK";
    let mut taxi_access_fare: u32 = 0;
    let mut taxi_egress_fare: u32 = 0;
    let mut taxi_access_wait_sec: u32 = 0;
    let mut taxi_access_drive_sec: u32 = 0;
    let mut taxi_egress_wait_sec: u32 = 0;
    let mut taxi_egress_drive_sec: u32 = 0;

    // Track position to classify access/transfer/egress
    // Legs are chronological: [access] [transit transfer...]* [egress]
    let mut seen_transit = false;
    let mut _last_transit_idx: Option<usize> = None;

    // Find first and last transit leg indices
    let first_transit = legs.iter().position(|l| matches!(l, PathLeg::Transit { .. }));
    let last_transit = legs.iter().rposition(|l| matches!(l, PathLeg::Transit { .. }));

    // Current time tracker for wait time computation
    let mut current_time = departure_time;

    for (i, leg) in legs.iter().enumerate() {
        match leg {
            PathLeg::Walk { duration_secs, .. } | PathLeg::Taxi { duration_secs, .. } => {
                let is_taxi = matches!(leg, PathLeg::Taxi { .. });
                let dur = *duration_secs;

                if first_transit.is_none() || i < first_transit.unwrap() {
                    // Access leg (before first transit)
                    access_time_sec += dur;
                    if is_taxi {
                        access_mode = "TAXI";
                        if let PathLeg::Taxi {
                            fare_krw,
                            wait_secs,
                            drive_secs,
                            ..
                        } = leg
                        {
                            taxi_access_fare = *fare_krw;
                            taxi_access_wait_sec = *wait_secs;
                            taxi_access_drive_sec = *drive_secs;
                        }
                    }
                    current_time += dur;
                } else if last_transit.is_some() && i > last_transit.unwrap() {
                    // Egress leg (after last transit)
                    egress_time_sec += dur;
                    if is_taxi {
                        egress_mode = "TAXI";
                        if let PathLeg::Taxi {
                            fare_krw,
                            wait_secs,
                            drive_secs,
                            ..
                        } = leg
                        {
                            taxi_egress_fare = *fare_krw;
                            taxi_egress_wait_sec = *wait_secs;
                            taxi_egress_drive_sec = *drive_secs;
                        }
                    }
                    // No need to track current_time after last transit for wait calc
                } else {
                    // Transfer walk (between transit legs)
                    transfer_walk_sec += dur;
                    current_time += dur;
                }
            }

            PathLeg::Transit {
                route_index,
                board_time,
                alight_time,
                ..
            } => {
                // Wait time: gap between current_time and board_time
                let wait = board_time.saturating_sub(current_time);
                total_wait_sec += wait;
                if !seen_transit {
                    first_wait_sec = wait;
                    seen_transit = true;
                }

                // IVT
                let ivt = alight_time.saturating_sub(*board_time);
                ivt_total_sec += ivt;
                num_transit_legs += 1;

                // IVT by mode
                let route = &data.routes[*route_index as usize];
                match route.mode {
                    RouteMode::Bus => ivt_bus_sec += ivt,
                    RouteMode::Subway => ivt_subway_sec += ivt,
                    RouteMode::Rail => ivt_rail_sec += ivt,
                    RouteMode::Gtx => ivt_gtx_sec += ivt,
                    RouteMode::Ferry => ivt_ferry_sec += ivt,
                    RouteMode::Air => ivt_air_sec += ivt,
                }

                // Update current_time to alight time
                current_time = *alight_time;
                _last_transit_idx = Some(i);
            }
        }
    }

    let walking_time_sec = access_time_sec + egress_time_sec + transfer_walk_sec;
    // Subtract taxi durations from walking_time if access/egress is taxi
    // (walking_time should only count actual walking)
    let pure_walk_sec = if access_mode == "TAXI" {
        walking_time_sec.saturating_sub(access_time_sec)
    } else {
        walking_time_sec
    };
    let pure_walk_sec = if egress_mode == "TAXI" {
        pure_walk_sec.saturating_sub(egress_time_sec)
    } else {
        pure_walk_sec
    };

    json!({
        // Java-compatible fields (11 fields)
        "in_vehicle_time_sec": ivt_total_sec,
        "waiting_time_sec": total_wait_sec,
        "walking_time_sec": pure_walk_sec,
        "access_walk_sec": if access_mode == "WALK" { access_time_sec } else { 0 },
        "egress_walk_sec": if egress_mode == "WALK" { egress_time_sec } else { 0 },
        "transfer_walk_sec": transfer_walk_sec,
        "num_transit_legs": num_transit_legs,
        "access_mode": access_mode,
        "egress_mode": egress_mode,
        "taxi_access_fare": taxi_access_fare,
        "taxi_egress_fare": taxi_egress_fare,

        // Extended fields for DL mode choice / SP survey
        "first_wait_sec": first_wait_sec,
        "access_time_sec": access_time_sec,
        "egress_time_sec": egress_time_sec,
        "taxi_access_wait_sec": taxi_access_wait_sec,
        "taxi_access_drive_sec": taxi_access_drive_sec,
        "taxi_egress_wait_sec": taxi_egress_wait_sec,
        "taxi_egress_drive_sec": taxi_egress_drive_sec,
        "ivt_bus_sec": ivt_bus_sec,
        "ivt_subway_sec": ivt_subway_sec,
        "ivt_rail_sec": ivt_rail_sec,
        "ivt_gtx_sec": ivt_gtx_sec,
        "ivt_ferry_sec": ivt_ferry_sec,
        "ivt_air_sec": ivt_air_sec
    })
}

/// Build from/to JSON objects for walk/taxi legs.
/// Multi-candidate shape segment finder (ported from Java ShapeGeometryService).
///
/// For round-trip shapes where the same location appears twice, finds all
/// candidate indices near from/to, then picks the (start, end) pair with
/// the smallest gap. Returns None if no valid segment or ratio exceeds 5x.
fn find_shape_segment(
    shape: &[(f64, f64)],
    from_lat: f64,
    from_lon: f64,
    to_lat: f64,
    to_lon: f64,
    _stop_distance_m: f64,
) -> Option<Vec<(f64, f64)>> {
    if shape.len() < 2 {
        return None;
    }

    let start_candidates = find_all_close_indices(shape, from_lat, from_lon);
    let end_candidates = find_all_close_indices(shape, to_lat, to_lon);

    if start_candidates.is_empty() || end_candidates.is_empty() {
        return None;
    }

    // Pick (start, end) pair with minimum positive gap
    let mut best_start = 0usize;
    let mut best_end = 0usize;
    let mut best_gap = usize::MAX;

    for &si in &start_candidates {
        for &ei in &end_candidates {
            if ei > si {
                let gap = ei - si;
                if gap < best_gap {
                    best_gap = gap;
                    best_start = si;
                    best_end = ei;
                }
            }
        }
    }

    if best_gap == usize::MAX || best_gap < 1 {
        return None;
    }

    // Sanity check: path distance vs direct distance (MAX_PATH_RATIO = 5.0)
    let direct_dist = haversine_distance(from_lat, from_lon, to_lat, to_lon);
    if direct_dist > 300.0 {
        let mut path_dist = 0.0f64;
        for i in best_start..best_end {
            path_dist += haversine_distance(
                shape[i].0, shape[i].1, shape[i + 1].0, shape[i + 1].1,
            );
        }
        if path_dist / direct_dist > 5.0 {
            return None;
        }
    }

    Some(shape[best_start..=best_end].to_vec())
}

/// Find all shape indices close to a coordinate (multi-candidate).
///
/// Collects indices within threshold of the minimum distance.
/// For consecutive close points, keeps only the closest as representative.
fn find_all_close_indices(shape: &[(f64, f64)], lat: f64, lon: f64) -> Vec<usize> {
    // Pass 1: find minimum distance²
    let mut min_dist_sq = f64::MAX;
    for &(slat, slon) in shape {
        let d = (slat - lat).powi(2) + (slon - lon).powi(2);
        if d < min_dist_sq {
            min_dist_sq = d;
        }
    }

    // Threshold: max(min_dist² × 4, ~200m²)
    let threshold = f64::max(min_dist_sq * 4.0, 0.002 * 0.002);

    // Pass 2: collect candidates (one per continuous run)
    let mut candidates = Vec::new();
    let mut in_run = false;
    let mut run_best_idx = 0usize;
    let mut run_best_dist = f64::MAX;

    for (i, &(slat, slon)) in shape.iter().enumerate() {
        let d = (slat - lat).powi(2) + (slon - lon).powi(2);
        if d <= threshold {
            if !in_run {
                in_run = true;
                run_best_idx = i;
                run_best_dist = d;
            } else if d < run_best_dist {
                run_best_idx = i;
                run_best_dist = d;
            }
        } else if in_run {
            candidates.push(run_best_idx);
            in_run = false;
            run_best_dist = f64::MAX;
        }
    }
    if in_run {
        candidates.push(run_best_idx);
    }

    candidates
}

/// Merge consecutive WALK legs into a single leg.
///
/// RAPTOR can produce transfer_walk + egress_walk as separate legs
/// (e.g., Walk → Bus → Walk → Walk). This merges them by concatenating
/// polylines, summing durations/distances, and keeping the first "from"
/// and last "to".
fn merge_consecutive_walks(legs: Vec<Value>) -> Vec<Value> {
    if legs.len() < 2 {
        return legs;
    }

    let mut merged: Vec<Value> = Vec::with_capacity(legs.len());

    let mut i = 0;
    while i < legs.len() {
        if legs[i]["mode"].as_str() != Some("WALK") {
            merged.push(legs[i].clone());
            i += 1;
            continue;
        }

        // Start of a WALK sequence — collect all consecutive WALKs
        let start = i;
        while i < legs.len() && legs[i]["mode"].as_str() == Some("WALK") {
            i += 1;
        }

        if i - start == 1 {
            // Single WALK — no merge needed
            merged.push(legs[start].clone());
        } else {
            // Merge consecutive WALKs
            let first = &legs[start];
            let last = &legs[i - 1];

            let mut total_duration = 0.0f64;
            let mut total_distance = 0.0f64;
            let mut all_coords: Vec<(f64, f64)> = Vec::new();

            for j in start..i {
                total_duration += legs[j]["duration"].as_f64().unwrap_or(0.0);
                total_distance += legs[j]["distance"].as_f64().unwrap_or(0.0);

                if let Some(pts) = legs[j]["legGeometry"]["points"].as_str() {
                    let coords = decode_polyline(pts);
                    if !all_coords.is_empty() && !coords.is_empty() {
                        // Skip first point if it's the same as the last (avoid duplicate)
                        let last_coord = all_coords.last().unwrap();
                        let first_coord = &coords[0];
                        if (last_coord.0 - first_coord.0).abs() < 1e-6
                            && (last_coord.1 - first_coord.1).abs() < 1e-6
                        {
                            all_coords.extend_from_slice(&coords[1..]);
                        } else {
                            all_coords.extend_from_slice(&coords);
                        }
                    } else {
                        all_coords.extend_from_slice(&coords);
                    }
                }
            }

            let polyline = if all_coords.len() >= 2 {
                encode_polyline(&all_coords)
            } else {
                first["legGeometry"]["points"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            };

            merged.push(json!({
                "mode": "WALK",
                "duration": total_duration,
                "distance": total_distance,
                "from": first["from"].clone(),
                "to": last["to"].clone(),
                "legGeometry": {
                    "points": polyline,
                    "length": all_coords.len().max(2)
                }
            }));
        }
    }

    merged
}

fn stop_json(data: &TransitData, stop: u32, lat: f64, lon: f64, is_origin: bool) -> Value {
    let name = if stop == u32::MAX {
        if is_origin {
            "Origin".to_string()
        } else {
            "Destination".to_string()
        }
    } else {
        data.stop_names
            .get(stop as usize)
            .cloned()
            .unwrap_or_else(|| format!("Stop {}", stop))
    };

    let mut obj = json!({
        "name": name,
        "lat": lat,
        "lon": lon,
    });
    if stop != u32::MAX {
        obj["stop"] = json!({
            "gtfsId": format!("1:{}", data.stop_ids.get(stop as usize).unwrap_or(&String::new()))
        });
    }
    obj
}

fn leg_to_json(
    data: &TransitData,
    leg: &PathLeg,
    use_route_shapes: bool,
    street_graph: Option<&crate::access::StreetGraph>,
) -> Value {
    match leg {
        PathLeg::Walk {
            from_stop,
            to_stop,
            from_lat,
            from_lon,
            to_lat,
            to_lon,
            duration_secs,
            distance_m,
        } => {
            // Use street graph for walk path if available and use_route_shapes is on
            let (polyline, n_points, walk_dist) =
                if use_route_shapes {
                    if let Some(graph) = street_graph {
                        let path = graph.walk_path(*from_lat, *from_lon, *to_lat, *to_lon, 1500.0);
                        if path.len() >= 2 {
                            let mut d = 0.0f32;
                            for w in path.windows(2) {
                                d += haversine_distance(w[0].0, w[0].1, w[1].0, w[1].1) as f32;
                            }
                            let pl = encode_polyline(&path);
                            let n = path.len();
                            (pl, n, d)
                        } else {
                            (encode_polyline(&[(*from_lat, *from_lon), (*to_lat, *to_lon)]), 2usize, *distance_m as f32)
                        }
                    } else {
                        (encode_polyline(&[(*from_lat, *from_lon), (*to_lat, *to_lon)]), 2usize, *distance_m as f32)
                    }
                } else {
                    (encode_polyline(&[(*from_lat, *from_lon), (*to_lat, *to_lon)]), 2usize, *distance_m as f32)
                };

            let from_obj = stop_json(data, *from_stop, *from_lat, *from_lon, true);
            let to_obj = stop_json(data, *to_stop, *to_lat, *to_lon, false);

            json!({
                "mode": "WALK",
                "duration": *duration_secs as f64,
                "distance": walk_dist,
                "from": from_obj,
                "to": to_obj,
                "legGeometry": {
                    "points": polyline,
                    "length": n_points
                }
            })
        }

        PathLeg::Taxi {
            from_stop,
            to_stop,
            from_lat,
            from_lon,
            to_lat,
            to_lon,
            duration_secs,
            distance_m,
            fare_krw,
            wait_secs,
            drive_secs,
        } => {
            let polyline = encode_polyline(&[(*from_lat, *from_lon), (*to_lat, *to_lon)]);
            let from_obj = stop_json(data, *from_stop, *from_lat, *from_lon, true);
            let to_obj = stop_json(data, *to_stop, *to_lat, *to_lon, false);

            json!({
                "mode": "TAXI",
                "duration": *duration_secs as f64,
                "distance": *distance_m,
                "fare": *fare_krw,
                "waitTime": *wait_secs,
                "driveTime": *drive_secs,
                "from": from_obj,
                "to": to_obj,
                "legGeometry": {
                    "points": polyline,
                    "length": 2
                }
            })
        }

        PathLeg::Transit {
            route_index,
            trip_index,
            board_stop,
            alight_stop,
            board_stop_pos,
            alight_stop_pos,
            board_time,
            alight_time,
        } => {
            let route = &data.routes[*route_index as usize];
            let schedule = &route.timetable.schedules[*trip_index as usize];
            let mode = route.mode.to_otp_mode();

            let board_si = *board_stop as usize;
            let alight_si = *alight_stop as usize;

            let from_name = data
                .stop_names
                .get(board_si)
                .cloned()
                .unwrap_or_default();
            let to_name = data
                .stop_names
                .get(alight_si)
                .cloned()
                .unwrap_or_default();

            let from_lat = data.stop_lats.get(board_si).copied().unwrap_or(0.0);
            let from_lon = data.stop_lons.get(board_si).copied().unwrap_or(0.0);
            let to_lat = data.stop_lats.get(alight_si).copied().unwrap_or(0.0);
            let to_lon = data.stop_lons.get(alight_si).copied().unwrap_or(0.0);

            let duration = alight_time.saturating_sub(*board_time);

            // Build polyline and compute distance
            let bp = *board_stop_pos as usize;
            let ap = *alight_stop_pos as usize;

            // Distance: always computed from stop coordinates (reliable)
            let mut leg_distance_m: f64 = 0.0;
            let mut stop_coords: Vec<(f64, f64)> = Vec::new();
            for pos in bp..=ap {
                let si = route.pattern.stop_indices[pos] as usize;
                let lat = data.stop_lats[si];
                let lon = data.stop_lons[si];
                if let Some(&(prev_lat, prev_lon)) = stop_coords.last() {
                    leg_distance_m += haversine_distance(prev_lat, prev_lon, lat, lon);
                }
                stop_coords.push((lat, lon));
            }

            // Polyline: Java ShapeGeometryService-style multi-candidate matching.
            // For each leg, find the from/to coordinates on the shape using
            // all close candidates, then pick the pair with minimum gap.
            let coords: Vec<(f64, f64)> = if use_route_shapes
                && !route.shape_coords.is_empty()
            {
                match find_shape_segment(
                    &route.shape_coords, from_lat, from_lon, to_lat, to_lon, leg_distance_m,
                ) {
                    Some(slice) => simplify_coords(&slice, SHAPE_SIMPLIFY_EPSILON),
                    None => stop_coords.clone(),
                }
            } else {
                stop_coords.clone()
            };
            let polyline = encode_polyline(&coords);

            // Build intermediate stops array
            let mut intermediate_stops: Vec<Value> = Vec::new();
            for pos in (bp + 1)..ap {
                let si = route.pattern.stop_indices[pos] as usize;
                intermediate_stops.push(json!({
                    "name": data.stop_names.get(si).unwrap_or(&String::new()),
                    "lat": data.stop_lats[si],
                    "lon": data.stop_lons[si],
                    "stop": {
                        "gtfsId": format!("1:{}", data.stop_ids.get(si).unwrap_or(&String::new()))
                    },
                    "arrivalTime": schedule.arrival_times.get(pos).copied().unwrap_or(0),
                    "departureTime": schedule.departure_times.get(pos).copied().unwrap_or(0)
                }));
            }

            json!({
                "mode": mode,
                "duration": duration as f64,
                "distance": leg_distance_m,
                "startTime": *board_time,
                "endTime": *alight_time,
                "from": {
                    "name": from_name,
                    "lat": from_lat,
                    "lon": from_lon,
                    "stop": {
                        "gtfsId": format!("1:{}", data.stop_ids.get(board_si).unwrap_or(&String::new()))
                    }
                },
                "to": {
                    "name": to_name,
                    "lat": to_lat,
                    "lon": to_lon,
                    "stop": {
                        "gtfsId": format!("1:{}", data.stop_ids.get(alight_si).unwrap_or(&String::new()))
                    }
                },
                "route": {
                    "shortName": route.route_short_name,
                    "longName": route.route_long_name,
                    "type": route.route_type
                },
                "trip": {
                    "gtfsId": format!("1:{}", schedule.trip_id)
                },
                "intermediateStops": intermediate_stops,
                "legGeometry": {
                    "points": polyline,
                    "length": coords.len()
                }
            })
        }
    }
}
