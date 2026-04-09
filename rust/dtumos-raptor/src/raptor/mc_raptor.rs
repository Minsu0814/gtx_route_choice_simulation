use smallvec::SmallVec;

use crate::access::{AccessMode, AccessStop};
use crate::raptor::cost::{CostConfig, TaxiCostConfig};
use crate::raptor::engine::{has_excessive_detour, PathLeg, RaptorPath};
use crate::raptor::label::{Label, LabelArena, ParetoBag};
use crate::types::{TransferType, TransitData};

/// Maximum rounds for McRAPTOR (= max transfers + 1).
/// 6 rounds = up to 5 transfers (aligned with Standard RAPTOR).
const MAX_ROUNDS: usize = 6;

/// Shared McRAPTOR state: Pareto bags + label arena.
/// Produced by `mc_raptor_compute`, consumed by `mc_raptor_extract`.
pub struct McRaptorState {
    pub best_bags: Vec<ParetoBag>,
    pub arena: LabelArena,
}

/// Run McRAPTOR computation, returning shared state reusable for multiple egress sets.
///
/// `bound_egress_stops` is used only for early termination (global upper bound).
/// Pass the union of all egress stops when sharing across multiple queries.
pub fn mc_raptor_compute(
    data: &TransitData,
    access_stops: &[AccessStop],
    bound_egress_stops: &[AccessStop],
    departure_time: u32,
    cost_config: &CostConfig,
    taxi_config: &TaxiCostConfig,
    relax_ratio: f64,
    relax_slack: u32,
) -> McRaptorState {
    let n_stops = data.stop_count();
    let n_routes = data.route_count();

    let mut arena = LabelArena::new();
    let mut best_bags: Vec<ParetoBag> = (0..n_stops).map(|_| ParetoBag::new()).collect();

    // Dirty-set: track which stops were marked
    let mut marked_set: Vec<u32> = Vec::with_capacity(512);
    let mut marked = vec![false; n_stops];

    for access in access_stops {
        let arr = departure_time + access.walk_time_secs;
        let access_cost = match access.mode {
            AccessMode::Walk => cost_config.walk_cost(access.walk_time_secs),
            AccessMode::Taxi => {
                taxi_config.taxi_cost_cs(
                    access.taxi_drive_secs,
                    access.taxi_wait_secs,
                    access.taxi_fare_krw,
                )
            }
        };
        let si = access.stop_index as usize;

        let idx = arena.push(Label::walk_access(arr, access_cost));
        if best_bags[si].add(idx, &arena) {
            if !marked[si] {
                marked[si] = true;
                marked_set.push(si as u32);
            }
        }
    }

    let mut global_upper_bound: u32 = data.service_end;
    const MAX_STEP_DELTA: usize = 2;
    let mut first_target_round: Option<usize> = None;

    // Pre-allocate buffers once, reuse across rounds
    let mut new_bags: Vec<ParetoBag> = (0..n_stops).map(|_| ParetoBag::new()).collect();
    let mut new_marked = vec![false; n_stops];
    let mut new_marked_set: Vec<u32> = Vec::with_capacity(512);
    let mut route_min_pos: Vec<usize> = vec![usize::MAX; n_routes];
    let mut dirty_routes: Vec<u32> = Vec::with_capacity(512);
    let mut transfer_labels: Vec<(usize, u32)> = Vec::new();

    for _k in 1..=MAX_ROUNDS {
        if let Some(ftr) = first_target_round {
            if _k > ftr + MAX_STEP_DELTA {
                break;
            }
        }

        // Clear only dirty entries from previous round
        for &si in &new_marked_set {
            new_bags[si as usize].clear();
            new_marked[si as usize] = false;
        }
        new_marked_set.clear();
        for &ri in &dirty_routes {
            route_min_pos[ri as usize] = usize::MAX;
        }
        dirty_routes.clear();
        let mut any_improvement = false;

        // (a) Collect routes from marked stops (iterate dirty set only)
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

        // (b) Traverse routes
        for &(route_idx, board_from_pos) in &queue {
            let route = &data.routes[route_idx as usize];
            let pattern = &route.pattern;
            let timetable = &route.timetable;
            let slack_idx = pattern.slack_index;
            let board_slack = cost_config.board_slack_secs(slack_idx);
            let alight_slack = cost_config.alight_slack_secs(slack_idx);

            // on_trip: (trip_index, board_pos, label_index_in_arena)
            let mut on_trip: SmallVec<[(usize, usize, u32); 4]> = SmallVec::new();

            for pos in board_from_pos..pattern.num_stops() {
                let stop_idx = pattern.stop_indices[pos] as usize;

                // Step 1: ALIGHT
                if pos > board_from_pos {
                    for &(ti, bp, board_idx) in &on_trip {
                        let schedule = &timetable.schedules[ti];
                        let arr_time = schedule.arrival_times[pos] + alight_slack;

                        if arr_time > global_upper_bound {
                            continue;
                        }

                        // Copy fields from arena before any mutation
                        let bl = arena.get(board_idx);
                        let bl_transfers = bl.num_transfers;
                        let bl_gc = bl.generalized_cost;
                        let bl_arr = bl.arrival_time;
                        let bl_route = bl.route_index;

                        let board_cost = cost_config.boarding_cost(
                            bl_transfers == 0 && bl_route == u32::MAX,
                            bl_arr,
                            schedule.departure_times[bp],
                        );
                        let transit_time =
                            arr_time.saturating_sub(schedule.departure_times[bp]);
                        let gc = bl_gc
                            + cost_config
                                .transit_arrival_cost(board_cost, transit_time, slack_idx);

                        let relaxed_gc = if relax_ratio > 1.0 {
                            (gc as f64 * relax_ratio) as u32 + relax_slack * 100
                        } else {
                            gc
                        };

                        // Check dominance BEFORE allocating label
                        let new_transfers = bl_transfers + 1;
                        let dominated =
                            best_bags[stop_idx].labels.iter().any(|&ei| {
                                let e = arena.get(ei);
                                e.arrival_time <= arr_time
                                    && e.num_transfers <= new_transfers
                                    && e.generalized_cost <= relaxed_gc
                            });

                        if !dominated {
                            let new_idx = arena.push(Label {
                                arrival_time: arr_time,
                                num_transfers: new_transfers,
                                generalized_cost: gc,
                                prev_stop: pattern.stop_indices[bp],
                                route_index: route_idx,
                                trip_index: ti as u32,
                                board_stop_pos: bp as u32,
                                alight_stop_pos: pos as u32,
                                prev_label: board_idx,
                            });
                            if new_bags[stop_idx].add(new_idx, &arena) {
                                best_bags[stop_idx].add(new_idx, &arena);
                                if !new_marked[stop_idx] {
                                    new_marked[stop_idx] = true;
                                    new_marked_set.push(stop_idx as u32);
                                }
                                any_improvement = true;
                            }
                        }
                    }
                }

                // Step 2: BOARD — try boarding from all labels in best_bags
                for &label_idx in &best_bags[stop_idx].labels {
                    let label = arena.get(label_idx);
                    if label.arrival_time > global_upper_bound {
                        continue;
                    }

                    let l_arr = label.arrival_time;
                    let l_transfers = label.num_transfers;
                    let l_gc = label.generalized_cost;

                    let min_board_time = l_arr + board_slack;

                    if let Some(ti) = timetable.earliest_trip(pos, min_board_time) {
                        let dominated_on_trip = on_trip.iter().any(
                            |&(ex_ti, _, ex_idx)| {
                                let ex = arena.get(ex_idx);
                                ex_ti <= ti
                                    && ex.arrival_time <= l_arr
                                    && ex.num_transfers <= l_transfers
                                    && ex.generalized_cost <= l_gc
                            },
                        );

                        if !dominated_on_trip {
                            on_trip.retain(|(ex_ti, _, ex_idx)| {
                                let ex = arena.get(*ex_idx);
                                !(ti <= *ex_ti
                                    && l_arr <= ex.arrival_time
                                    && l_transfers <= ex.num_transfers
                                    && l_gc <= ex.generalized_cost
                                    && (ti < *ex_ti
                                        || l_arr < ex.arrival_time
                                        || l_transfers < ex.num_transfers
                                        || l_gc < ex.generalized_cost))
                            });

                            on_trip.push((ti, pos, label_idx));
                        }
                    }
                }
            }
        }

        // (c) Transfers — iterate dirty set only
        transfer_labels.clear();
        for &stop_idx in &new_marked_set {
            let stop_idx = stop_idx as usize;

            for &label_idx in &new_bags[stop_idx].labels {
                let label = arena.get(label_idx);
                if label.route_index == u32::MAX && label.prev_label != Label::NONE {
                    continue;
                }
                let l_arr = label.arrival_time;
                let l_transfers = label.num_transfers;
                let l_gc = label.generalized_cost;

                for transfer in &data.transfers_from[stop_idx] {
                    let slack = match transfer.transfer_type {
                        TransferType::InStation | TransferType::Timed => 0,
                        _ => cost_config.transfer_slack,
                    };
                    let new_arr = l_arr + transfer.duration_secs + slack;
                    let transfer_walk_cost = cost_config.transfer_cost(transfer);
                    let new_gc = l_gc + transfer_walk_cost;
                    let to = transfer.to_stop as usize;

                    // Check dominance before allocating
                    let dominated = best_bags[to].labels.iter().any(|&ei| {
                        let e = arena.get(ei);
                        e.arrival_time <= new_arr
                            && e.num_transfers <= l_transfers
                            && e.generalized_cost <= new_gc
                    });
                    if dominated {
                        continue;
                    }

                    let new_idx = arena.push(Label {
                        arrival_time: new_arr,
                        num_transfers: l_transfers,
                        generalized_cost: new_gc,
                        prev_stop: stop_idx as u32,
                        route_index: u32::MAX,
                        trip_index: u32::MAX,
                        board_stop_pos: u32::MAX,
                        alight_stop_pos: u32::MAX,
                        prev_label: label_idx,
                    });

                    transfer_labels.push((to, new_idx));
                }
            }
        }

        for &(to, new_idx) in &transfer_labels {
            if new_bags[to].add(new_idx, &arena) {
                best_bags[to].add(new_idx, &arena);
                if !new_marked[to] {
                    new_marked[to] = true;
                    new_marked_set.push(to as u32);
                }
                any_improvement = true;
            }
        }

        // Swap marked ↔ new_marked (reuse buffers)
        std::mem::swap(&mut marked, &mut new_marked);
        std::mem::swap(&mut marked_set, &mut new_marked_set);

        for egress in bound_egress_stops {
            let si = egress.stop_index as usize;
            let best_arr = best_bags[si].best_arrival_time(&arena);
            if best_arr < u32::MAX {
                let arr_at_dest = best_arr + egress.walk_time_secs;
                if arr_at_dest < global_upper_bound {
                    global_upper_bound = arr_at_dest;
                }
                if first_target_round.is_none() {
                    first_target_round = Some(_k);
                }
            }
        }

        if !any_improvement {
            break;
        }
    }

    McRaptorState { best_bags, arena }
}

/// Extract Pareto-optimal paths from shared McRAPTOR state for specific egress stops.
pub fn mc_raptor_extract(
    data: &TransitData,
    state: &McRaptorState,
    access_stops: &[AccessStop],
    egress_stops: &[AccessStop],
    departure_time: u32,
    cost_config: &CostConfig,
    taxi_config: &TaxiCostConfig,
    max_results: usize,
) -> Vec<RaptorPath> {
    let mut results: Vec<RaptorPath> = Vec::new();

    for egress in egress_stops {
        let si = egress.stop_index as usize;

        for &label_idx in &state.best_bags[si].labels {
            let label = state.arena.get(label_idx);
            let arrival_at_dest = label.arrival_time + egress.walk_time_secs;
            let total_duration = arrival_at_dest.saturating_sub(departure_time);
            let egress_cost = match egress.mode {
                AccessMode::Walk => cost_config.walk_cost(egress.walk_time_secs),
                AccessMode::Taxi => {
                    taxi_config.taxi_cost_cs(
                        egress.taxi_drive_secs,
                        egress.taxi_wait_secs,
                        egress.taxi_fare_krw,
                    )
                }
            };
            let gc = label.generalized_cost + egress_cost;

            let legs = reconstruct_mc_path(
                data,
                &state.arena,
                label_idx,
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

            results.push(RaptorPath {
                legs,
                total_duration,
                total_walk_dist,
                generalized_cost: gc,
                num_transfers: label.num_transfers.saturating_sub(1),
                category: None,
            });
        }
    }

    // Filter out paths with excessive transit leg detours
    results.retain(|path| !has_excessive_detour(data, path));

    categorize_results(&mut results, max_results)
}

/// Multi-criteria RAPTOR search.
///
/// Returns Pareto-optimal paths considering (arrival_time, transfers, generalized_cost).
pub fn mc_raptor_search(
    data: &TransitData,
    access_stops: &[AccessStop],
    egress_stops: &[AccessStop],
    departure_time: u32,
    cost_config: &CostConfig,
    taxi_config: &TaxiCostConfig,
    max_results: usize,
    relax_ratio: f64,
    relax_slack: u32,
) -> Vec<RaptorPath> {
    let state = mc_raptor_compute(
        data,
        access_stops,
        egress_stops,
        departure_time,
        cost_config,
        taxi_config,
        relax_ratio,
        relax_slack,
    );
    mc_raptor_extract(
        data,
        &state,
        access_stops,
        egress_stops,
        departure_time,
        cost_config,
        taxi_config,
        max_results,
    )
}

/// Transit leg signature for deduplication.
fn transit_signature(path: &RaptorPath) -> Vec<(u32, u32, u32, u32)> {
    path.legs
        .iter()
        .filter_map(|l| match l {
            PathLeg::Transit {
                route_index,
                trip_index,
                board_stop,
                alight_stop,
                ..
            } => Some((*route_index, *trip_index, *board_stop, *alight_stop)),
            _ => None,
        })
        .collect()
}

/// Select categorized routes in order: optimal → min_time → min_transfer → min_walk.
fn categorize_results(results: &mut Vec<RaptorPath>, max_results: usize) -> Vec<RaptorPath> {
    if results.is_empty() {
        return Vec::new();
    }

    results.sort_by_key(|p| p.generalized_cost);

    let mut categorized: Vec<RaptorPath> = Vec::new();
    let mut used_sigs: Vec<Vec<(u32, u32, u32, u32)>> = Vec::new();

    // 1. 최적경로
    {
        let sig = transit_signature(&results[0]);
        let mut p = results[0].clone();
        p.category = Some("optimal".into());
        used_sigs.push(sig);
        categorized.push(p);
    }

    // 2. 최소시간
    if let Some(best) = results
        .iter()
        .filter(|p| !used_sigs.contains(&transit_signature(p)))
        .min_by_key(|p| (p.total_duration, p.generalized_cost))
    {
        let sig = transit_signature(best);
        let mut p = best.clone();
        p.category = Some("min_time".into());
        used_sigs.push(sig);
        categorized.push(p);
    }

    // 3. 최소환승
    if let Some(best) = results
        .iter()
        .filter(|p| !used_sigs.contains(&transit_signature(p)))
        .min_by_key(|p| (p.num_transfers as u32, p.generalized_cost))
    {
        let sig = transit_signature(best);
        let mut p = best.clone();
        p.category = Some("min_transfer".into());
        used_sigs.push(sig);
        categorized.push(p);
    }

    // 4. 최소도보
    if let Some(best) = results
        .iter()
        .filter(|p| !used_sigs.contains(&transit_signature(p)))
        .min_by(|a, b| {
            a.total_walk_dist
                .partial_cmp(&b.total_walk_dist)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.generalized_cost.cmp(&b.generalized_cost))
        })
    {
        let sig = transit_signature(best);
        let mut p = best.clone();
        p.category = Some("min_walk".into());
        used_sigs.push(sig);
        categorized.push(p);
    }

    // Fill remaining slots
    for r in results.iter() {
        if categorized.len() >= max_results {
            break;
        }
        let sig = transit_signature(r);
        if !used_sigs.contains(&sig) {
            let mut p = r.clone();
            p.category = None;
            used_sigs.push(sig);
            categorized.push(p);
        }
    }

    categorized
}

fn reconstruct_mc_path(
    data: &TransitData,
    arena: &LabelArena,
    start_idx: u32,
    egress: &AccessStop,
    access_stops: &[AccessStop],
    _departure_time: u32,
) -> Vec<PathLeg> {
    let mut legs = Vec::new();

    // Absorb trailing transfer walks into egress
    let mut current_idx = start_idx;
    let mut egress_stop = egress.stop_index as usize;
    let mut extra_walk_time = 0u32;
    let mut extra_walk_dist = 0.0f64;

    loop {
        let label = arena.get(current_idx);
        if label.route_index != u32::MAX || label.prev_stop == u32::MAX {
            break;
        }
        // Transfer walk — absorb into egress
        let from = label.prev_stop as usize;
        let (dur, dist) = data.transfers_from[from]
            .iter()
            .find(|t| t.to_stop == egress_stop as u32)
            .map(|t| (t.duration_secs, t.distance_m as f64))
            .unwrap_or((0, 0.0));
        extra_walk_time += dur;
        extra_walk_dist += dist;
        egress_stop = from;
        if label.prev_label == Label::NONE {
            break;
        }
        current_idx = label.prev_label;
    }

    // Egress leg
    let total_egress_time = egress.walk_time_secs + extra_walk_time;
    let total_egress_dist = egress.walk_distance_m + extra_walk_dist;
    if total_egress_time > 0 {
        match egress.mode {
            AccessMode::Taxi => {
                legs.push(PathLeg::Taxi {
                    from_stop: egress_stop as u32,
                    to_stop: u32::MAX,
                    from_lat: data.stop_lats[egress_stop],
                    from_lon: data.stop_lons[egress_stop],
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
                    from_stop: egress_stop as u32,
                    to_stop: u32::MAX,
                    from_lat: data.stop_lats[egress_stop],
                    from_lon: data.stop_lons[egress_stop],
                    to_lat: egress.orig_lat,
                    to_lon: egress.orig_lon,
                    duration_secs: total_egress_time,
                    distance_m: total_egress_dist,
                });
            }
        }
    }

    // Walk the label chain backward via arena indices
    let mut current_stop = egress_stop;
    let mut visited = 0;

    loop {
        let label = arena.get(current_idx);
        if label.prev_stop == u32::MAX || visited > MAX_ROUNDS * 2 {
            break;
        }
        visited += 1;

        if label.route_index == u32::MAX {
            // Walking transfer
            let from = label.prev_stop as usize;
            let to = current_stop;

            let (dur, dist) = data.transfers_from[from]
                .iter()
                .find(|t| t.to_stop == to as u32)
                .map(|t| (t.duration_secs, t.distance_m as f64))
                .unwrap_or((0, 0.0));

            legs.push(PathLeg::Walk {
                from_stop: from as u32,
                to_stop: to as u32,
                from_lat: data.stop_lats[from],
                from_lon: data.stop_lons[from],
                to_lat: data.stop_lats[to],
                to_lon: data.stop_lons[to],
                duration_secs: dur,
                distance_m: dist,
            });

            current_stop = from;
        } else {
            // Transit leg
            let ri = label.route_index as usize;
            let ti = label.trip_index as usize;
            let schedule = &data.routes[ri].timetable.schedules[ti];

            legs.push(PathLeg::Transit {
                route_index: label.route_index,
                trip_index: label.trip_index,
                board_stop: label.prev_stop,
                alight_stop: current_stop as u32,
                board_stop_pos: label.board_stop_pos,
                alight_stop_pos: label.alight_stop_pos,
                board_time: schedule.departure_times[label.board_stop_pos as usize],
                alight_time: schedule.arrival_times[label.alight_stop_pos as usize],
            });

            current_stop = label.prev_stop as usize;
        }

        if label.prev_label == Label::NONE {
            break;
        }
        current_idx = label.prev_label;
    }

    // Access leg
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

    legs.reverse();
    legs
}
