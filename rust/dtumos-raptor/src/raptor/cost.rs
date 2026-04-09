use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::types::{Transfer, TransferType};

/// Taxi fare/time estimation configuration.
/// Ported from Java TaxiFeedData constants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxiCostConfig {
    /// Base fare (KRW)
    pub base_fare_krw: u32,
    /// Distance fare per unit (KRW)
    pub dist_fare_krw: u32,
    /// Distance unit (meters) for fare calculation
    pub dist_unit_m: u32,
    /// Road factor: haversine → road distance multiplier
    pub road_factor: f64,
    /// Average taxi speed (m/s), ~30 km/h
    pub taxi_speed_mps: f64,
    /// Default wait time (seconds)
    pub default_wait_secs: u32,
    /// Default surge multiplier
    pub default_surge: f64,
    /// VOT factor: centi-seconds per KRW (6000 / vot_krw_per_minute)
    /// Converts monetary cost to time cost for generalized cost
    pub vot_factor: u32,
    /// Taxi reluctance multiplier for generalized cost
    pub taxi_reluctance: f64,
}

impl Default for TaxiCostConfig {
    fn default() -> Self {
        Self {
            base_fare_krw: 4800,
            dist_fare_krw: 100,
            dist_unit_m: 132,
            road_factor: 1.3,
            taxi_speed_mps: 8.33,       // ~30 km/h
            default_wait_secs: 300,     // 5 minutes
            default_surge: 1.0,
            vot_factor: 36,             // 6000 / 167 ≈ 36
            taxi_reluctance: 1.0,
        }
    }
}

impl TaxiCostConfig {
    /// Estimate taxi fare (KRW) for given road distance in meters.
    pub fn estimate_fare(&self, road_distance_m: f64) -> u32 {
        let fare = self.base_fare_krw as f64
            + (road_distance_m / self.dist_unit_m as f64) * self.dist_fare_krw as f64;
        (fare * self.default_surge) as u32
    }

    /// Compute taxi access/egress generalized cost in centi-seconds.
    /// c1 = (drive_time + wait_time) * reluctance * 100 + fare * vot_factor
    pub fn taxi_cost_cs(&self, drive_secs: u32, wait_secs: u32, fare_krw: u32) -> u32 {
        let time_cs = ((drive_secs + wait_secs) as f64 * 100.0 * self.taxi_reluctance) as u32;
        let fare_cs = fare_krw * self.vot_factor;
        time_cs + fare_cs
    }
}

/// Cost configuration ported from Java KoreanCostCalculator + KoreanSlackProvider.
/// All costs in centi-seconds (1 second = 100 units).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostConfig {
    /// First boarding penalty (seconds → stored as centi-seconds internally)
    pub first_board_cost_secs: u32,
    /// Transfer penalty (seconds)
    pub transfer_cost_secs: u32,
    /// Wait time reluctance multiplier
    pub wait_reluctance: f64,
    /// Walk reluctance multiplier
    pub walk_reluctance: f64,
    /// Per-mode transit riding reluctance [SUBWAY, BUS, RAIL, AIR, FERRY, GTX]
    pub transit_reluctance: [f64; 6],
    /// Board slack per mode (seconds) [SUBWAY, BUS, RAIL, AIR, FERRY, GTX]
    pub board_slack: [u32; 6],
    /// Alight slack per mode (seconds) [SUBWAY, BUS, RAIL, AIR, FERRY, GTX]
    pub alight_slack: [u32; 6],
    /// Transfer slack (seconds)
    pub transfer_slack: u32,

    // ── Tier 4: Personalization (#16) ──────────────────────────────

    /// Per-route avoidance set: route indices to penalize heavily.
    /// Empty = no avoidance. Populated from user preferences.
    #[serde(default)]
    pub avoid_routes: HashSet<u32>,
    /// Avoidance penalty (centi-seconds) applied to avoided routes.
    #[serde(default = "default_avoid_penalty")]
    pub avoid_penalty_cs: u32,

    // ── Tier 4: Accessibility (#15) ────────────────────────────────

    /// Whether to require step-free (wheelchair accessible) routes only.
    #[serde(default)]
    pub wheelchair_accessible: bool,
    /// Elevation reluctance multiplier (>1.0 penalizes uphill walks).
    /// Requires DEM data to be loaded. Default 1.0 = no elevation effect.
    #[serde(default = "default_one")]
    pub elevation_reluctance: f64,
}

fn default_avoid_penalty() -> u32 {
    360000 // 1 hour in centi-seconds
}

fn default_one() -> f64 {
    1.0
}

impl Default for CostConfig {
    fn default() -> Self {
        Self {
            first_board_cost_secs: 60,
            transfer_cost_secs: 180,
            wait_reluctance: 1.0,
            walk_reluctance: 1.0,
            transit_reluctance: [1.0, 1.0, 0.7, 1.0, 1.0, 0.5], // SUBWAY, BUS, RAIL, AIR, FERRY, GTX
            board_slack: [60, 30, 120, 180, 180, 90],
            alight_slack: [30, 10, 60, 120, 120, 45],
            transfer_slack: 60,
            avoid_routes: HashSet::new(),
            avoid_penalty_cs: 360000,
            wheelchair_accessible: false,
            elevation_reluctance: 1.0,
        }
    }
}

impl CostConfig {
    /// Check if a route is in the avoidance set.
    #[inline]
    pub fn route_penalty(&self, route_index: u32) -> u32 {
        if self.avoid_routes.contains(&route_index) {
            self.avoid_penalty_cs
        } else {
            0
        }
    }

    /// First boarding cost in centi-seconds.
    #[inline]
    pub fn first_board_cost_cs(&self) -> u32 {
        self.first_board_cost_secs * 100
    }

    /// Transfer cost in centi-seconds.
    #[inline]
    pub fn transfer_cost_cs(&self) -> u32 {
        self.transfer_cost_secs * 100
    }

    /// Compute boarding cost (centi-seconds).
    #[inline]
    pub fn boarding_cost(
        &self,
        first_boarding: bool,
        prev_arrival_time: u32,
        board_time: u32,
    ) -> u32 {
        let wait_time = board_time.saturating_sub(prev_arrival_time);
        let wait_cost = (wait_time as f64 * 100.0 * self.wait_reluctance) as u32;
        let board_cost = if first_boarding {
            self.first_board_cost_cs()
        } else {
            self.transfer_cost_cs()
        };
        board_cost + wait_cost
    }

    /// Compute transit arrival cost (centi-seconds).
    #[inline]
    pub fn transit_arrival_cost(
        &self,
        board_cost: u32,
        transit_time: u32,
        slack_index: usize,
    ) -> u32 {
        let reluctance = if slack_index < self.transit_reluctance.len() {
            self.transit_reluctance[slack_index]
        } else {
            1.0
        };
        board_cost + (transit_time as f64 * 100.0 * reluctance) as u32
    }

    /// Walking cost for access/egress (centi-seconds).
    #[inline]
    pub fn walk_cost(&self, walk_time_secs: u32) -> u32 {
        (walk_time_secs as f64 * 100.0 * self.walk_reluctance) as u32
    }

    /// Board slack for a given mode (seconds).
    #[inline]
    pub fn board_slack_secs(&self, slack_index: usize) -> u32 {
        self.board_slack
            .get(slack_index)
            .or_else(|| self.board_slack.get(1))
            .copied()
            .unwrap_or(30)
    }

    /// Alight slack for a given mode (seconds).
    #[inline]
    pub fn alight_slack_secs(&self, slack_index: usize) -> u32 {
        self.alight_slack
            .get(slack_index)
            .or_else(|| self.alight_slack.get(1))
            .copied()
            .unwrap_or(10)
    }

    /// Transfer walk cost, adjusted by transfer type.
    ///
    /// - InStation/Timed: walk_time × 100 (no reluctance penalty — same complex)
    /// - Others: walk_time × walk_reluctance × 100
    #[inline]
    pub fn transfer_cost(&self, transfer: &Transfer) -> u32 {
        match transfer.transfer_type {
            TransferType::InStation | TransferType::Timed => transfer.duration_secs * 100,
            _ => self.walk_cost(transfer.duration_secs),
        }
    }
}
