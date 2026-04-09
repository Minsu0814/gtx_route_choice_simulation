use crate::access::{AccessMode, AccessStop};
use crate::raptor::cost::CostConfig;
use crate::types::{haversine_distance, TransitData};

/// Maximum number of rounds (transfers).
const MAX_ROUNDS: usize = 6;

/// Result of a single RAPTOR query: the journey legs.
#[derive(Debug, Clone)]
pub struct RaptorPath {
    pub legs: Vec<PathLeg>,
    pub total_duration: u32,   // seconds
    pub total_walk_dist: f64,  // meters
    pub generalized_cost: u32, // centi-seconds
    pub num_transfers: u8,
    /// Route category: "optimal", "min_time", "min_transfer", "min_walk", or None.
    pub category: Option<String>,
}

#[derive(Debug, Clone)]
pub enum PathLeg {
    Walk {
        from_stop: u32, // u32::MAX for origin
        to_stop: u32,   // u32::MAX for destination
        from_lat: f64,
        from_lon: f64,
        to_lat: f64,
        to_lon: f64,
        duration_secs: u32,
        distance_m: f64,
    },
    Taxi {
        from_stop: u32, // u32::MAX for origin
        to_stop: u32,   // u32::MAX for destination
        from_lat: f64,
        from_lon: f64,
        to_lat: f64,
        to_lon: f64,
        duration_secs: u32, // drive_time + wait_time
        distance_m: f64,    // road distance
        fare_krw: u32,
        wait_secs: u32,
        drive_secs: u32,
    },
    Transit {
        route_index: u32,
        trip_index: u32,
        board_stop: u32,
        alight_stop: u32,
        board_stop_pos: u32,
        alight_stop_pos: u32,
        board_time: u32,
        alight_time: u32,
    },
}

/// Shared standard RAPTOR state: round times and backtrack at all stops.
/// Produced by `raptor_compute`, consumed by `raptor_extract`.
pub struct RaptorState {
    pub round_time: Vec<Vec<u32>>,
    pub backtrack: Vec<Vec<Option<BacktrackEntry>>>,
}

/// Run standard RAPTOR computation, returning shared state reusable for multiple egress sets.
///
/// Uses dual-criterion optimization:
/// - `best_time[stop]`: earliest arrival (used for trip lookups via `earliest_trip`)
/// - `best_gc[stop]`: lowest generalized cost (used for improvement decisions)
/// A stop is re-explored when EITHER time or GC improves. This produces
/// cost-aware results while keeping single-value-per-stop speed (no Pareto bags).
///
/// `bound_egress_stops` is used only for early termination.
pub fn raptor_compute(
    data: &TransitData,
    access_stops: &[AccessStop],
    bound_egress_stops: &[AccessStop],
    departure_time: u32,
    cost_config: &CostConfig,
) -> RaptorState {
    let n_stops = data.stop_count();
    let n_routes = data.route_count();

    let mut best_time = vec![u32::MAX; n_stops];
    let mut best_gc = vec![u32::MAX; n_stops];
    let mut round_time = vec![vec![u32::MAX; n_stops]; MAX_ROUNDS + 1];
    let mut round_gc = vec![vec![u32::MAX; n_stops]; MAX_ROUNDS + 1];
    let mut backtrack: Vec<Vec<Option<BacktrackEntry>>> =
        vec![vec![None; n_stops]; MAX_ROUNDS + 1];

    // Dirty-set pattern: track which stops are marked instead of scanning all
    let mut marked_set: Vec<u32> = Vec::with_capacity(512);
    let mut marked = vec![false; n_stops];

    for access in access_stops {
        let arr = departure_time + access.walk_time_secs;
        let gc = cost_config.walk_cost(access.walk_time_secs);
        let si = access.stop_index as usize;
        if arr < best_time[si] || gc < best_gc[si] {
            best_time[si] = best_time[si].min(arr);
            best_gc[si] = best_gc[si].min(gc);
            round_time[0][si] = arr;
            round_gc[0][si] = gc;
            if !marked[si] {
                marked[si] = true;
                marked_set.push(si as u32);
            }
            backtrack[0][si] = Some(BacktrackEntry {
                prev_stop: u32::MAX,
                route_index: u32::MAX,
                trip_index: u32::MAX,
                board_stop_pos: u32::MAX,
                alight_stop_pos: u32::MAX,
            });
        }
    }

    const MAX_STEP_DELTA: usize = 2;
    let mut first_target_round: Option<usize> = None;
    let egress_set: Vec<usize> = bound_egress_stops
        .iter()
        .map(|e| e.stop_index as usize)
        .collect();

    // Pre-allocate buffers once, reuse across rounds (dirty-set reset)
    let mut new_marked = vec![false; n_stops];
    let mut new_marked_set: Vec<u32> = Vec::with_capacity(512);
    let mut route_min_pos: Vec<usize> = vec![usize::MAX; n_routes];
    let mut dirty_routes: Vec<u32> = Vec::with_capacity(512);

    for k in 1..=MAX_ROUNDS {
        if let Some(ftr) = first_target_round {
            if k > ftr + MAX_STEP_DELTA {
                break;
            }
        }

        // Clear only dirty entries from previous round
        for &si in &new_marked_set {
            new_marked[si as usize] = false;
        }
        new_marked_set.clear();
        for &ri in &dirty_routes {
            route_min_pos[ri as usize] = usize::MAX;
        }
        dirty_routes.clear();
        let mut any_improvement = false;

        // Collect routes from marked stops (iterate dirty set only)
        for &stop_idx in &marked_set {
            let stop_idx = stop_idx as usize;
            for &(route_idx, pos) in &data.routes_by_stop[stop_idx] {
                let ri = route_idx as usize;
                let p = pos as usize;
                if route_min_pos[ri] == usize::MAX {
                    dirty_routes.push(route_idx);
                }
                if p < route_min_pos[ri] {
                    route_min_pos[ri] = p;
                }
            }
        }

        let queue: Vec<(u32, usize)> = dirty_routes
            .iter()
            .map(|&ri| (ri, route_min_pos[ri as usize]))
            .collect();

        for &(route_idx, board_from_pos) in &queue {
            let route = &data.routes[route_idx as usize];
            let pattern = &route.pattern;
            let timetable = &route.timetable;
            let slack_idx = pattern.slack_index;

            let board_slack = cost_config.board_slack_secs(slack_idx);
            let alight_slack = cost_config.alight_slack_secs(slack_idx);

            let mut current_trip: Option<usize> = None;
            let mut current_board_pos: usize = 0;
            let mut current_board_gc: u32 = u32::MAX;

            for pos in board_from_pos..pattern.num_stops() {
                let stop_idx = pattern.stop_indices[pos] as usize;

                // Try to board: use best_time for trip lookup (earliest arrival)
                let prev_arr = round_time[k - 1][stop_idx];
                if prev_arr < u32::MAX {
                    let min_board_time = prev_arr + board_slack;
                    if let Some(ti) = timetable.earliest_trip(pos, min_board_time) {
                        match current_trip {
                            None => {
                                current_trip = Some(ti);
                                current_board_pos = pos;
                                current_board_gc = round_gc[k - 1][stop_idx];
                            }
                            Some(ct) => {
                                if timetable.schedules[ti].departure_times[pos]
                                    < timetable.schedules[ct].departure_times[current_board_pos]
                                {
                                    current_trip = Some(ti);
                                    current_board_pos = pos;
                                    current_board_gc = round_gc[k - 1][stop_idx];
                                }
                            }
                        }
                    }
                }

                // Alight and check improvement
                if let Some(ti) = current_trip {
                    let schedule = &timetable.schedules[ti];
                    let arr_time = schedule.arrival_times[pos] + alight_slack;

                    // Compute incremental GC for this arrival
                    let is_first_board = k == 1
                        && backtrack[k - 1][pattern.stop_indices[current_board_pos] as usize]
                            .as_ref()
                            .map(|b| b.route_index == u32::MAX)
                            .unwrap_or(true);

                    let board_cost = cost_config.boarding_cost(
                        is_first_board,
                        round_time[k - 1][pattern.stop_indices[current_board_pos] as usize],
                        schedule.departure_times[current_board_pos],
                    );
                    let transit_time =
                        arr_time.saturating_sub(schedule.departure_times[current_board_pos]);
                    let transit_cost =
                        cost_config.transit_arrival_cost(board_cost, transit_time, slack_idx);
                    let gc = current_board_gc.saturating_add(transit_cost);

                    // Dual criterion: improve if EITHER time or GC is better
                    let time_improves = arr_time < best_time[stop_idx];
                    let gc_improves = gc < best_gc[stop_idx];

                    if time_improves || gc_improves {
                        best_time[stop_idx] = best_time[stop_idx].min(arr_time);
                        best_gc[stop_idx] = best_gc[stop_idx].min(gc);
                        round_time[k][stop_idx] = arr_time;
                        round_gc[k][stop_idx] = gc;
                        if !new_marked[stop_idx] {
                            new_marked[stop_idx] = true;
                            new_marked_set.push(stop_idx as u32);
                        }
                        any_improvement = true;

                        backtrack[k][stop_idx] = Some(BacktrackEntry {
                            prev_stop: pattern.stop_indices[current_board_pos],
                            route_index: route_idx,
                            trip_index: ti as u32,
                            board_stop_pos: current_board_pos as u32,
                            alight_stop_pos: pos as u32,
                        });
                    }
                }
            }
        }

        // Transfer phase: iterate only newly marked stops
        for &stop_idx in &new_marked_set.clone() {
            let stop_idx = stop_idx as usize;
            let arr_at_stop = round_time[k][stop_idx];
            let gc_at_stop = round_gc[k][stop_idx];

            if let Some(ref bt) = backtrack[k][stop_idx] {
                if bt.route_index == u32::MAX && bt.prev_stop != u32::MAX {
                    continue;
                }
            }

            for transfer in &data.transfers_from[stop_idx] {
                let new_arr =
                    arr_at_stop + transfer.duration_secs + cost_config.transfer_slack;
                let transfer_gc =
                    gc_at_stop + cost_config.transfer_cost(transfer);
                let to = transfer.to_stop as usize;

                let time_improves = new_arr < best_time[to];
                let gc_improves = transfer_gc < best_gc[to];

                if time_improves || gc_improves {
                    best_time[to] = best_time[to].min(new_arr);
                    best_gc[to] = best_gc[to].min(transfer_gc);
                    round_time[k][to] = new_arr;
                    round_gc[k][to] = transfer_gc;
                    if !new_marked[to] {
                        new_marked[to] = true;
                        new_marked_set.push(to as u32);
                    }
                    any_improvement = true;

                    backtrack[k][to] = Some(BacktrackEntry {
                        prev_stop: stop_idx as u32,
                        route_index: u32::MAX,
                        trip_index: u32::MAX,
                        board_stop_pos: u32::MAX,
                        alight_stop_pos: u32::MAX,
                    });
                }
            }
        }

        // Swap marked ↔ new_marked (reuse buffers)
        std::mem::swap(&mut marked, &mut new_marked);
        std::mem::swap(&mut marked_set, &mut new_marked_set);

        if first_target_round.is_none() {
            for &si in &egress_set {
                if best_time[si] < u32::MAX {
                    first_target_round = Some(k);
                    break;
                }
            }
        }

        if !any_improvement {
            break;
        }
    }

    RaptorState {
        round_time,
        backtrack,
    }
}

/// Extract paths from shared standard RAPTOR state for specific egress stops.
pub fn raptor_extract(
    data: &TransitData,
    state: &RaptorState,
    access_stops: &[AccessStop],
    egress_stops: &[AccessStop],
    departure_time: u32,
    cost_config: &CostConfig,
    max_results: usize,
) -> Vec<RaptorPath> {
    let mut results: Vec<(u32, RaptorPath)> = Vec::new();

    for egress in egress_stops {
        let si = egress.stop_index as usize;

        for k in 0..=MAX_ROUNDS {
            if state.round_time[k][si] == u32::MAX {
                continue;
            }

            let arrival_at_dest = state.round_time[k][si] + egress.walk_time_secs;
            let total_duration = arrival_at_dest.saturating_sub(departure_time);

            let legs = reconstruct_path(
                data,
                &state.backtrack,
                k,
                si,
                egress,
                access_stops,
                departure_time,
            );

            let total_walk_dist = legs
                .iter()
                .map(|l| match l {
                    PathLeg::Walk { distance_m, .. } => *distance_m,
                    PathLeg::Taxi { .. } | PathLeg::Transit { .. } => 0.0,
                })
                .sum();

            let gc = compute_path_gc(data, &legs, departure_time, cost_config);

            let path = RaptorPath {
                legs,
                total_duration,
                total_walk_dist,
                generalized_cost: gc,
                num_transfers: k.saturating_sub(1) as u8,
                category: None,
            };

            results.push((arrival_at_dest, path));
        }
    }

    // Filter out paths with excessive transit leg detours (Java ShapeGeometryService equivalent)
    results.retain(|(_, path)| !has_excessive_detour(data, path));

    results.sort_by_key(|(arr, _)| *arr);
    results.dedup_by(|(a1, p1), (a2, p2)| {
        *a1 == *a2 && p1.num_transfers == p2.num_transfers
    });
    results.truncate(max_results);

    results.into_iter().map(|(_, p)| p).collect()
}

/// Standard RAPTOR search (single criterion: earliest arrival).
///
/// Returns up to `max_results` paths sorted by arrival time.
pub fn raptor_search(
    data: &TransitData,
    access_stops: &[AccessStop],
    egress_stops: &[AccessStop],
    departure_time: u32,
    cost_config: &CostConfig,
    max_results: usize,
) -> Vec<RaptorPath> {
    let state = raptor_compute(data, access_stops, egress_stops, departure_time, cost_config);
    raptor_extract(data, &state, access_stops, egress_stops, departure_time, cost_config, max_results)
}

#[derive(Clone)]
pub struct BacktrackEntry {
    pub prev_stop: u32,
    pub route_index: u32,
    pub trip_index: u32,
    pub board_stop_pos: u32,
    pub alight_stop_pos: u32,
}

fn reconstruct_path(
    data: &TransitData,
    backtrack: &[Vec<Option<BacktrackEntry>>],
    round: usize,
    egress_stop: usize,
    egress: &AccessStop,
    access_stops: &[AccessStop],
    _departure_time: u32,
) -> Vec<PathLeg> {
    let mut legs = Vec::new();

    // Absorb trailing transfer walks into egress:
    // Trace back from egress_stop through any transfer walks to find the
    // actual last transit stop. Build a single egress walk from there.
    let mut actual_egress_stop = egress_stop;
    let mut extra_walk_time = 0u32;
    let mut extra_walk_dist = 0.0f64;
    let trace_round = round;
    let mut absorb_iters = 0u32;

    while trace_round > 0 {
        // Guard against circular transfers in GTFS data
        absorb_iters += 1;
        if absorb_iters > MAX_ROUNDS as u32 {
            break;
        }
        let entry = match &backtrack[trace_round][actual_egress_stop] {
            Some(e) => e.clone(),
            None => break,
        };
        if entry.route_index != u32::MAX {
            break; // Hit a transit leg — stop absorbing
        }
        // Transfer walk — absorb into egress
        let from = entry.prev_stop as usize;
        let (dur, dist) = data.transfers_from[from]
            .iter()
            .find(|t| t.to_stop == actual_egress_stop as u32)
            .map(|t| (t.duration_secs, t.distance_m as f64))
            .unwrap_or((0, 0.0));
        extra_walk_time += dur;
        extra_walk_dist += dist;
        actual_egress_stop = from;
        // Stay in same round for transfer walks
    }

    // Egress leg: from actual last transit stop to destination
    let total_egress_time = egress.walk_time_secs + extra_walk_time;
    let total_egress_dist = egress.walk_distance_m + extra_walk_dist;
    if total_egress_time > 0 {
        match egress.mode {
            AccessMode::Taxi => {
                legs.push(PathLeg::Taxi {
                    from_stop: actual_egress_stop as u32,
                    to_stop: u32::MAX,
                    from_lat: data.stop_lats[actual_egress_stop],
                    from_lon: data.stop_lons[actual_egress_stop],
                    to_lat: egress.orig_lat,
                    to_lon: egress.orig_lon,
                    duration_secs: total_egress_time,
                    distance_m: total_egress_dist,
                    fare_krw: egress.taxi_fare_krw,
                    wait_secs: egress.taxi_wait_secs,
                    drive_secs: egress.taxi_drive_secs,
                });
            }
            AccessMode::Walk => {
                legs.push(PathLeg::Walk {
                    from_stop: actual_egress_stop as u32,
                    to_stop: u32::MAX,
                    from_lat: data.stop_lats[actual_egress_stop],
                    from_lon: data.stop_lons[actual_egress_stop],
                    to_lat: egress.orig_lat,
                    to_lon: egress.orig_lon,
                    duration_secs: total_egress_time,
                    distance_m: total_egress_dist,
                });
            }
        }
    }

    // Walk backward through rounds from the actual egress stop
    let mut current_stop = actual_egress_stop;
    let mut current_round = round;

    while current_round > 0 {
        let entry = match &backtrack[current_round][current_stop] {
            Some(e) => e.clone(),
            None => break,
        };

        if entry.route_index == u32::MAX {
            // Walking transfer (between two transit legs)
            let from = entry.prev_stop as usize;
            legs.push(PathLeg::Walk {
                from_stop: entry.prev_stop,
                to_stop: current_stop as u32,
                from_lat: data.stop_lats[from],
                from_lon: data.stop_lons[from],
                to_lat: data.stop_lats[current_stop],
                to_lon: data.stop_lons[current_stop],
                duration_secs: data.transfers_from[from]
                    .iter()
                    .find(|t| t.to_stop == current_stop as u32)
                    .map(|t| t.duration_secs)
                    .unwrap_or(0),
                distance_m: data.transfers_from[from]
                    .iter()
                    .find(|t| t.to_stop == current_stop as u32)
                    .map(|t| t.distance_m as f64)
                    .unwrap_or(0.0),
            });
            current_stop = from;
            // Stay in same round for transfer walks
        } else {
            // Transit leg
            legs.push(PathLeg::Transit {
                route_index: entry.route_index,
                trip_index: entry.trip_index,
                board_stop: entry.prev_stop,
                alight_stop: current_stop as u32,
                board_stop_pos: entry.board_stop_pos,
                alight_stop_pos: entry.alight_stop_pos,
                board_time: data.routes[entry.route_index as usize].timetable.schedules
                    [entry.trip_index as usize]
                    .departure_times[entry.board_stop_pos as usize],
                alight_time: data.routes[entry.route_index as usize].timetable.schedules
                    [entry.trip_index as usize]
                    .arrival_times[entry.alight_stop_pos as usize],
            });
            current_stop = entry.prev_stop as usize;
            current_round -= 1;
        }
    }

    // Access leg (origin → first stop): walk or taxi
    if let Some(access) = access_stops
        .iter()
        .find(|a| a.stop_index == current_stop as u32)
    {
        if access.walk_time_secs > 0 {
            match access.mode {
                AccessMode::Taxi => {
                    legs.push(PathLeg::Taxi {
                        from_stop: u32::MAX,
                        to_stop: current_stop as u32,
                        from_lat: access.orig_lat,
                        from_lon: access.orig_lon,
                        to_lat: data.stop_lats[current_stop],
                        to_lon: data.stop_lons[current_stop],
                        duration_secs: access.walk_time_secs,
                        distance_m: access.walk_distance_m,
                        fare_krw: access.taxi_fare_krw,
                        wait_secs: access.taxi_wait_secs,
                        drive_secs: access.taxi_drive_secs,
                    });
                }
                AccessMode::Walk => {
                    legs.push(PathLeg::Walk {
                        from_stop: u32::MAX,
                        to_stop: current_stop as u32,
                        from_lat: access.orig_lat,
                        from_lon: access.orig_lon,
                        to_lat: data.stop_lats[current_stop],
                        to_lon: data.stop_lons[current_stop],
                        duration_secs: access.walk_time_secs,
                        distance_m: access.walk_distance_m,
                    });
                }
            }
        }
    }

    // Reverse to get chronological order
    legs.reverse();
    legs
}

/// Maximum allowed ratio of path distance to direct distance for a transit leg.
/// Matches Java ShapeGeometryService MAX_PATH_RATIO threshold.
const MAX_DETOUR_RATIO: f64 = 5.0;

/// Minimum direct distance (meters) to apply detour check.
/// Very short legs are exempt (noisy coordinates can inflate ratios).
const MIN_DETOUR_CHECK_DIST_M: f64 = 300.0;

/// Check if any transit leg in a path has an excessive detour
/// (stop-to-stop cumulative distance / direct haversine > threshold).
pub fn has_excessive_detour(data: &TransitData, path: &RaptorPath) -> bool {
    for leg in &path.legs {
        if let PathLeg::Transit {
            route_index,
            board_stop,
            alight_stop,
            board_stop_pos,
            alight_stop_pos,
            ..
        } = leg
        {
            let b = *board_stop as usize;
            let a = *alight_stop as usize;

            let direct_dist = haversine_distance(
                data.stop_lats[b],
                data.stop_lons[b],
                data.stop_lats[a],
                data.stop_lons[a],
            );

            if direct_dist < MIN_DETOUR_CHECK_DIST_M {
                continue;
            }

            let route = &data.routes[*route_index as usize];
            let stops = &route.pattern.stop_indices;
            let bp = *board_stop_pos as usize;
            let ap = *alight_stop_pos as usize;

            // Sum haversine between consecutive stops (lower bound of actual path)
            let mut path_dist = 0.0;
            for i in bp..ap {
                let s1 = stops[i] as usize;
                let s2 = stops[i + 1] as usize;
                path_dist += haversine_distance(
                    data.stop_lats[s1],
                    data.stop_lons[s1],
                    data.stop_lats[s2],
                    data.stop_lons[s2],
                );
            }

            if path_dist > direct_dist * MAX_DETOUR_RATIO {
                return true;
            }
        }
    }
    false
}

/// Compute generalized cost for a reconstructed path (Standard RAPTOR).
///
/// Walks the leg sequence and applies CostConfig's reluctance weights:
/// - Walk legs: walk_reluctance
/// - Wait time (gap before boarding): wait_reluctance
/// - Transit legs: transit_reluctance[mode]
/// - Boarding: first_board_cost or transfer_cost
fn compute_path_gc(
    data: &TransitData,
    legs: &[PathLeg],
    departure_time: u32,
    cost_config: &CostConfig,
) -> u32 {
    let mut gc: u32 = 0;
    let mut current_time = departure_time;
    let mut seen_transit = false;

    for leg in legs {
        match leg {
            PathLeg::Walk { duration_secs, .. } => {
                gc += cost_config.walk_cost(*duration_secs);
                current_time += duration_secs;
            }
            PathLeg::Taxi { duration_secs, .. } => {
                // Taxi legs: use duration as-is (taxi cost is handled separately in McRAPTOR)
                gc += *duration_secs * 100;
                current_time += duration_secs;
            }
            PathLeg::Transit {
                route_index,
                board_time,
                alight_time,
                ..
            } => {
                // Wait cost
                let wait_time = board_time.saturating_sub(current_time);
                let wait_cost = (wait_time as f64 * 100.0 * cost_config.wait_reluctance) as u32;
                gc += wait_cost;

                // Boarding penalty
                gc += if !seen_transit {
                    cost_config.first_board_cost_cs()
                } else {
                    cost_config.transfer_cost_cs()
                };
                seen_transit = true;

                // Transit IVT with mode-specific reluctance
                let route = &data.routes[*route_index as usize];
                let slack_idx = route.pattern.slack_index;
                let transit_time = alight_time.saturating_sub(*board_time);
                let reluctance = if slack_idx < cost_config.transit_reluctance.len() {
                    cost_config.transit_reluctance[slack_idx]
                } else {
                    1.0
                };
                gc += (transit_time as f64 * 100.0 * reluctance) as u32;

                current_time = *alight_time;
            }
        }
    }

    gc
}
