//! Korean T-money fare calculator.
//!
//! Implements the integrated distance-based fare system used by Korean transit:
//! - Base fare: 1250 KRW (card) for bus/subway
//! - Distance surcharge: after 10km, +100 KRW per 5km (bus) or per 8km (subway)
//! - Transfer discount: within 30 min and ≤5 transfers, only distance surcharge applies
//! - Inter-mode: bus↔subway transfers are combined under single fare journey

use serde::{Deserialize, Serialize};

use crate::raptor::engine::PathLeg;
use crate::types::{RouteMode, TransitData};

/// T-money fare configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FareConfig {
    /// Base fare for bus (KRW, card)
    pub bus_base_fare: u32,
    /// Base fare for subway (KRW, card)
    pub subway_base_fare: u32,
    /// Base fare for rail (KTX/SRT, KRW)
    pub rail_base_fare: u32,
    /// Distance included in base fare (meters)
    pub base_distance_m: f64,
    /// Bus surcharge: KRW per extra_unit_m after base_distance_m
    pub bus_surcharge_krw: u32,
    /// Bus surcharge unit distance (meters)
    pub bus_surcharge_unit_m: f64,
    /// Subway surcharge: KRW per extra_unit_m after base_distance_m
    pub subway_surcharge_krw: u32,
    /// Subway surcharge unit distance (meters)
    pub subway_surcharge_unit_m: f64,
    /// Transfer time window (seconds) — free transfer within this window
    pub transfer_window_secs: u32,
    /// Maximum number of free transfers
    pub max_free_transfers: u8,
    /// Whether fare calculation is enabled
    pub enabled: bool,
}

impl Default for FareConfig {
    fn default() -> Self {
        Self {
            bus_base_fare: 1200,
            subway_base_fare: 1250,
            rail_base_fare: 1250,
            base_distance_m: 10_000.0,
            bus_surcharge_krw: 100,
            bus_surcharge_unit_m: 5_000.0,
            subway_surcharge_krw: 100,
            subway_surcharge_unit_m: 8_000.0,
            transfer_window_secs: 1800, // 30 minutes
            max_free_transfers: 5,
            enabled: true,
        }
    }
}

/// Computed fare breakdown for an itinerary.
#[derive(Debug, Clone, Default)]
pub struct FareBreakdown {
    /// Total fare (KRW)
    pub total_krw: u32,
    /// Base fare portion
    pub base_fare_krw: u32,
    /// Distance surcharge portion
    pub distance_surcharge_krw: u32,
    /// Number of fare-free transfers used
    pub free_transfers_used: u32,
    /// Cumulative transit distance (meters)
    pub cumulative_distance_m: f64,
}

impl FareConfig {
    /// Compute T-money fare for a complete itinerary (post-search).
    ///
    /// Walks through legs chronologically, tracking:
    /// - Cumulative distance across transfer-linked segments
    /// - Transfer timing for free-transfer eligibility
    /// - Mode-specific surcharge rates
    pub fn compute_fare(
        &self,
        data: &TransitData,
        legs: &[PathLeg],
    ) -> FareBreakdown {
        if !self.enabled {
            return FareBreakdown::default();
        }

        let mut result = FareBreakdown::default();
        let mut journey_distance_m: f64 = 0.0;
        let mut last_alight_time: u32 = 0;
        let mut journey_started = false;
        let mut transfers_in_journey: u8 = 0;
        // Track the "highest" base fare in the current journey
        // (bus 1200 vs subway 1250 — the higher one applies)
        let mut journey_base_fare: u32 = 0;
        // Track predominant surcharge rate (weighted by distance)
        let mut bus_distance: f64 = 0.0;
        let mut subway_distance: f64 = 0.0;

        for leg in legs {
            if let PathLeg::Transit {
                route_index,
                board_time,
                alight_time,
                board_stop_pos,
                alight_stop_pos,
                ..
            } = leg
            {
                let route = &data.routes[*route_index as usize];
                let mode = route.mode;

                // Check if this is a continuation of an existing journey (transfer)
                let is_transfer = journey_started
                    && (*board_time).saturating_sub(last_alight_time) <= self.transfer_window_secs
                    && transfers_in_journey < self.max_free_transfers;

                if !journey_started || !is_transfer {
                    // New fare journey — settle previous journey if any
                    if journey_started {
                        result.total_krw += self.compute_journey_fare(
                            journey_base_fare,
                            journey_distance_m,
                            bus_distance,
                            subway_distance,
                        );
                        result.cumulative_distance_m += journey_distance_m;
                    }
                    // Start new journey
                    journey_distance_m = 0.0;
                    bus_distance = 0.0;
                    subway_distance = 0.0;
                    transfers_in_journey = 0;
                    journey_started = true;
                    journey_base_fare = self.base_fare_for_mode(mode);
                } else {
                    // Free transfer — update base fare if higher mode encountered
                    let mode_fare = self.base_fare_for_mode(mode);
                    journey_base_fare = journey_base_fare.max(mode_fare);
                    transfers_in_journey += 1;
                    result.free_transfers_used += 1;
                }

                // Compute leg distance from stop positions
                let leg_dist = compute_leg_distance(
                    data,
                    route,
                    *board_stop_pos as usize,
                    *alight_stop_pos as usize,
                );
                journey_distance_m += leg_dist;

                match mode {
                    RouteMode::Bus => bus_distance += leg_dist,
                    RouteMode::Subway | RouteMode::Gtx => subway_distance += leg_dist,
                    RouteMode::Rail => subway_distance += leg_dist,
                    _ => bus_distance += leg_dist,
                }

                last_alight_time = *alight_time;
            }
        }

        // Settle final journey
        if journey_started {
            let journey_fare = self.compute_journey_fare(
                journey_base_fare,
                journey_distance_m,
                bus_distance,
                subway_distance,
            );
            result.total_krw += journey_fare;
            result.cumulative_distance_m += journey_distance_m;
        }

        // Decompose
        result.base_fare_krw = if journey_started {
            journey_base_fare
        } else {
            0
        };
        result.distance_surcharge_krw = result.total_krw.saturating_sub(result.base_fare_krw);

        result
    }

    fn base_fare_for_mode(&self, mode: RouteMode) -> u32 {
        match mode {
            RouteMode::Bus => self.bus_base_fare,
            RouteMode::Subway | RouteMode::Gtx => self.subway_base_fare,
            RouteMode::Rail => self.rail_base_fare,
            _ => self.bus_base_fare,
        }
    }

    /// Compute fare for a single fare journey (base + distance surcharge).
    fn compute_journey_fare(
        &self,
        base_fare: u32,
        total_distance_m: f64,
        bus_distance_m: f64,
        subway_distance_m: f64,
    ) -> u32 {
        if total_distance_m <= self.base_distance_m {
            return base_fare;
        }

        let extra_m = total_distance_m - self.base_distance_m;
        let total_transit = bus_distance_m + subway_distance_m;

        // Weighted surcharge rate based on distance proportion per mode
        let surcharge = if total_transit > 0.0 {
            let bus_fraction = bus_distance_m / total_transit;
            let subway_fraction = subway_distance_m / total_transit;

            let bus_surcharge = if self.bus_surcharge_unit_m > 0.0 {
                (extra_m * bus_fraction / self.bus_surcharge_unit_m).ceil() as u32
                    * self.bus_surcharge_krw
            } else {
                0
            };
            let subway_surcharge = if self.subway_surcharge_unit_m > 0.0 {
                (extra_m * subway_fraction / self.subway_surcharge_unit_m).ceil() as u32
                    * self.subway_surcharge_krw
            } else {
                0
            };
            bus_surcharge + subway_surcharge
        } else {
            0
        };

        base_fare + surcharge
    }
}

/// Compute transit leg distance.
///
/// Uses route shape polyline when available (accurate road distance),
/// falls back to stop-to-stop haversine sum (underestimates by ~20-30%).
fn compute_leg_distance(
    data: &TransitData,
    route: &crate::types::Route,
    board_pos: usize,
    alight_pos: usize,
) -> f64 {
    // Use shape polyline if available
    if !route.shape_coords.is_empty()
        && route.shape_stop_indices.len() > alight_pos
    {
        let si_start = route.shape_stop_indices[board_pos];
        let si_end = route.shape_stop_indices[alight_pos];
        if si_start <= si_end && si_end < route.shape_coords.len() {
            let mut dist = 0.0;
            for w in route.shape_coords[si_start..=si_end].windows(2) {
                dist += crate::types::haversine_distance(w[0].0, w[0].1, w[1].0, w[1].1);
            }
            return dist;
        }
    }

    // Fallback: stop-to-stop haversine
    let mut dist = 0.0;
    for pos in board_pos..alight_pos {
        let si1 = route.pattern.stop_indices[pos] as usize;
        let si2 = route.pattern.stop_indices[pos + 1] as usize;
        dist += crate::types::haversine_distance(
            data.stop_lats[si1],
            data.stop_lons[si1],
            data.stop_lats[si2],
            data.stop_lons[si2],
        );
    }
    dist
}
