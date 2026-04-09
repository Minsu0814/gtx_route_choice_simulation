use rstar::{PointDistance, RTreeObject, AABB};
use serde::{Deserialize, Serialize};

/// Transit mode classification matching Korean GTFS route_type → slackIndex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RouteMode {
    Subway = 0, // slackIndex 0: 지하철/도시철도
    Bus = 1,    // slackIndex 1: 버스
    Rail = 2,   // slackIndex 2: 철도
    Air = 3,    // slackIndex 3: 항공
    Ferry = 4,  // slackIndex 4: 해운
    Gtx = 5,    // slackIndex 5: GTX
}

impl RouteMode {
    /// Map GTFS route_type to RouteMode.
    ///
    /// Korean GTFS (TAGO API) uses codes 0-8 which differ from standard GTFS:
    ///   0=시내/마을버스, 1=도시철도, 2=해운, 3=시외버스, 4=일반철도,
    ///   5=공항리무진, 6=고속철도, 7=항공, 8=GTX
    ///
    /// Also supports extended GTFS types (100+) for international data.
    pub fn from_gtfs_route_type(route_type: u16) -> Self {
        match route_type {
            // Korean TAGO API codes (0-8)
            0 => RouteMode::Bus,    // 시내/농어촌/마을버스
            1 => RouteMode::Subway, // 도시철도/경전철
            2 => RouteMode::Ferry,  // 해운
            3 => RouteMode::Bus,    // 시외버스
            4 => RouteMode::Rail,   // 일반철도/광역철도
            5 => RouteMode::Bus,    // 공항리무진버스
            6 => RouteMode::Rail,   // 고속철도 (KTX/SRT)
            7 => RouteMode::Air,    // 항공
            8 => RouteMode::Gtx,    // GTX

            // Extended GTFS route_type ranges (100+)
            rt if (100..200).contains(&rt) => RouteMode::Rail,   // Railway
            rt if (200..300).contains(&rt) => RouteMode::Rail,   // Coach
            rt if (400..500).contains(&rt) => RouteMode::Subway, // Urban Rail
            rt if (700..800).contains(&rt) => RouteMode::Bus,    // Bus Service
            rt if (900..1000).contains(&rt) => RouteMode::Subway, // Tram
            rt if (1100..1200).contains(&rt) => RouteMode::Air,  // Air Service
            _ => RouteMode::Bus, // Default fallback
        }
    }

    pub fn slack_index(self) -> usize {
        self as usize
    }

    /// OTP-compatible mode string for JSON output.
    pub fn to_otp_mode(&self) -> &'static str {
        match self {
            RouteMode::Subway => "SUBWAY",
            RouteMode::Bus => "BUS",
            RouteMode::Rail => "RAIL",
            RouteMode::Air => "AIRPLANE",
            RouteMode::Ferry => "FERRY",
            RouteMode::Gtx => "RAIL", // GTX is a rail variant
        }
    }

    /// Human-readable Korean name.
    pub fn korean_name(&self) -> &'static str {
        match self {
            RouteMode::Subway => "지하철",
            RouteMode::Bus => "버스",
            RouteMode::Rail => "철도",
            RouteMode::Air => "항공",
            RouteMode::Ferry => "해운",
            RouteMode::Gtx => "GTX",
        }
    }
}

/// Immutable transit data — built once from GTFS, read-only during routing.
pub struct TransitData {
    // Stops (SoA layout for cache efficiency)
    pub stop_ids: Vec<String>,
    pub stop_names: Vec<String>,
    pub stop_lats: Vec<f64>,
    pub stop_lons: Vec<f64>,

    // Route patterns with timetables
    pub routes: Vec<Route>,

    // Transfers indexed by stop (from_stop → transfers)
    pub transfers_from: Vec<Vec<Transfer>>,

    // Reverse index: stop → [(route_index, position_in_route)]
    // Pre-computed to avoid O(n) .position() search during routing.
    pub routes_by_stop: Vec<Vec<(u32, u16)>>,

    // Service time window (seconds since midnight)
    pub service_start: u32,
    pub service_end: u32,

    // Spatial index for stop lookup
    pub stop_rtree: rstar::RTree<StopPoint>,
}

impl TransitData {
    pub fn stop_count(&self) -> usize {
        self.stop_ids.len()
    }

    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    pub fn total_trip_count(&self) -> usize {
        self.routes.iter().map(|r| r.timetable.schedules.len()).sum()
    }
}

/// A route = one trip pattern + its timetable.
#[derive(Clone, Serialize, Deserialize)]
pub struct Route {
    pub pattern: TripPattern,
    pub timetable: TimeTable,
    pub route_id: String,
    pub route_short_name: String,
    pub route_long_name: String,
    pub route_type: u16,
    pub mode: RouteMode,
    /// On-time performance (0.0–1.0). None = unknown.
    /// Used by Tier 3 reliability-aware routing.
    pub reliability: Option<f32>,
    /// Average crowding level (0.0–1.0, 0=empty, 1=crush load). None = unknown.
    /// Used by Tier 3 congestion-aware routing.
    pub crowding: Option<f32>,
    /// Shape polyline coordinates (lat, lon) from shapes.txt. Empty if unavailable.
    #[serde(default)]
    pub shape_coords: Vec<(f64, f64)>,
    /// For each stop position in the pattern, the index into shape_coords of the
    /// nearest shape point. Used to extract sub-polylines for individual legs.
    /// Empty if shape_coords is empty.
    #[serde(default)]
    pub shape_stop_indices: Vec<usize>,
}

/// Ordered sequence of stops that trips follow.
#[derive(Clone, Serialize, Deserialize)]
pub struct TripPattern {
    pub stop_indices: Vec<u32>,
    pub slack_index: usize,
}

impl TripPattern {
    pub fn num_stops(&self) -> usize {
        self.stop_indices.len()
    }
}

/// Sorted list of trips following one pattern.
#[derive(Clone, Serialize, Deserialize)]
pub struct TimeTable {
    pub schedules: Vec<TripSchedule>,
    /// Frequency entries for headway-based services (empty for scheduled services).
    pub frequencies: Vec<FrequencyEntry>,
    /// Whether departure_times are monotonically non-decreasing at ALL stop positions.
    /// When true, binary search is safe at every position.
    /// When false, binary search is only safe at position 0 and a linear scan
    /// fallback is used for other positions.
    /// Set by `validate_sort_order()` after building.
    #[serde(default = "default_false")]
    pub sorted_at_all_positions: bool,
}

fn default_false() -> bool {
    false
}

impl TimeTable {
    /// Find the earliest trip departing at or after `min_dep_time` from stop position `stop_pos`.
    /// Returns the schedule index, or None.
    ///
    /// Uses binary search when departure times are known to be sorted at `stop_pos`,
    /// otherwise falls back to a linear scan to correctly handle overtaking trips.
    pub fn earliest_trip(&self, stop_pos: usize, min_dep_time: u32) -> Option<usize> {
        if self.sorted_at_all_positions || stop_pos == 0 {
            // Binary search: departure_times[stop_pos] is sorted across schedules
            let mut lo = 0usize;
            let mut hi = self.schedules.len();

            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if self.schedules[mid].departure_times[stop_pos] < min_dep_time {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            if lo < self.schedules.len() {
                Some(lo)
            } else {
                None
            }
        } else {
            // Linear scan fallback for non-monotonic stop positions.
            // Find the trip with the earliest departure >= min_dep_time.
            let mut best_idx: Option<usize> = None;
            let mut best_dep = u32::MAX;
            for (i, schedule) in self.schedules.iter().enumerate() {
                let dep = schedule.departure_times[stop_pos];
                if dep >= min_dep_time && dep < best_dep {
                    best_dep = dep;
                    best_idx = Some(i);
                }
            }
            best_idx
        }
    }

    /// Validate whether departure times are sorted at every stop position.
    /// Call after building/sorting schedules. Sets `sorted_at_all_positions`.
    ///
    /// Returns the number of stop positions where ordering is violated
    /// (0 = fully sorted, binary search safe everywhere).
    pub fn validate_sort_order(&mut self) -> usize {
        if self.schedules.len() <= 1 {
            self.sorted_at_all_positions = true;
            return 0;
        }

        let num_stops = self.schedules[0].departure_times.len();
        let mut violations = 0usize;

        // Position 0 is always sorted (we sort by it).
        // Check positions 1..num_stops.
        for pos in 1..num_stops {
            for w in self.schedules.windows(2) {
                if w[0].departure_times[pos] > w[1].departure_times[pos] {
                    violations += 1;
                    break; // one violation at this position is enough
                }
            }
        }

        self.sorted_at_all_positions = violations == 0;
        violations
    }

    /// Expected wait time for frequency-based service at the given time.
    /// Returns headway/2 if within an active frequency period, or None.
    pub fn expected_wait_secs(&self, time: u32) -> Option<u32> {
        for freq in &self.frequencies {
            if time >= freq.start_time && time < freq.end_time {
                return Some(freq.headway_secs / 2);
            }
        }
        None
    }

    /// Whether this timetable has frequency-based service.
    pub fn is_frequency_based(&self) -> bool {
        !self.frequencies.is_empty()
    }
}

/// A single trip's timing at each stop.
#[derive(Clone, Serialize, Deserialize)]
pub struct TripSchedule {
    pub trip_id: String,
    pub route_short_name: String,
    pub arrival_times: Vec<u32>,   // seconds since midnight, indexed by stop position
    pub departure_times: Vec<u32>, // seconds since midnight, indexed by stop position
}

/// GTFS transfer type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum TransferType {
    /// Recommended transfer point (default)
    Recommended = 0,
    /// Timed transfer — vehicle waits
    Timed = 1,
    /// Minimum time required
    MinTime = 2,
    /// Transfer not possible
    NotPossible = 3,
    /// In-station transfer (same complex, e.g. subway ↔ subway)
    InStation = 4,
}

impl TransferType {
    pub fn from_gtfs(val: u8) -> Self {
        match val {
            0 => Self::Recommended,
            1 => Self::Timed,
            2 => Self::MinTime,
            3 => Self::NotPossible,
            4 => Self::InStation,
            _ => Self::Recommended,
        }
    }
}

/// Walking transfer between two stops.
#[derive(Clone, Serialize, Deserialize)]
pub struct Transfer {
    pub to_stop: u32,
    pub duration_secs: u32,
    pub distance_m: f32,
    /// Transfer type from GTFS transfers.txt (default: Recommended)
    pub transfer_type: TransferType,
}

/// Frequency entry for headway-based services (GTFS frequencies.txt).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyEntry {
    pub start_time: u32,
    pub end_time: u32,
    pub headway_secs: u32,
    pub exact_times: bool,
}

/// R-tree point for spatial stop queries.
#[derive(Debug, Clone, Copy)]
pub struct StopPoint {
    pub lat: f64,
    pub lon: f64,
    pub index: u32,
}

impl RTreeObject for StopPoint {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.lat, self.lon])
    }
}

impl PointDistance for StopPoint {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let dlat = self.lat - point[0];
        let dlon = self.lon - point[1];
        dlat * dlat + dlon * dlon
    }
}

/// Fast equirectangular distance approximation in meters.
///
/// Accurate to <0.1% for distances under ~5km at mid-latitudes (30°–60°).
/// Uses only multiplications — no trig functions — making it 5-10x faster
/// than haversine. Suitable for transfer generation, access stop filtering,
/// and detour checks where points are nearby.
#[inline]
pub fn fast_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const DEG_TO_M: f64 = 111_319.5; // meters per degree latitude
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    // cos(mid_lat) approximation using the average latitude
    let mid_lat_rad = ((lat1 + lat2) * 0.5).to_radians();
    let cos_lat = mid_lat_rad.cos();
    let x = dlon * cos_lat;
    let y = dlat;
    (x * x + y * y).sqrt() * DEG_TO_M
}

/// Haversine distance in meters between two (lat, lon) points.
///
/// For short distances (<5km), delegates to the faster equirectangular
/// approximation which avoids expensive trig calls.
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    // Short-circuit: if degree delta is small, use fast approximation.
    // 0.05° ≈ 5.5km — well within equirectangular accuracy range.
    let dlat = (lat2 - lat1).abs();
    let dlon = (lon2 - lon1).abs();
    if dlat < 0.05 && dlon < 0.05 {
        return fast_distance(lat1, lon1, lat2, lon2);
    }

    const R: f64 = 6_371_000.0; // Earth radius in meters

    let dlat_r = (lat2 - lat1).to_radians();
    let dlon_r = (lon2 - lon1).to_radians();

    let a = (dlat_r / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon_r / 2.0).sin().powi(2);

    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    R * c
}
