use serde::Deserialize;

/// Deserialize helper: GTFS time "HH:MM:SS" → seconds since midnight.
/// Handles times > 24:00 (e.g., "25:30:00" = 91800).
pub fn parse_gtfs_time(s: &str) -> Option<u32> {
    let parts: Vec<&str> = s.trim().split(':').collect();
    if parts.len() < 2 {
        return None;
    }
    let h: u32 = parts[0].parse().ok()?;
    let m: u32 = parts[1].parse().ok()?;
    let sec: u32 = if parts.len() >= 3 {
        parts[2].parse().ok()?
    } else {
        0
    };
    Some(h * 3600 + m * 60 + sec)
}

/// Raw GTFS stop record.
#[derive(Debug, Deserialize)]
pub struct GtfsStop {
    pub stop_id: String,
    #[serde(default)]
    pub stop_name: String,
    pub stop_lat: f64,
    pub stop_lon: f64,
}

/// Raw GTFS route record.
#[derive(Debug, Deserialize)]
pub struct GtfsRoute {
    pub route_id: String,
    #[serde(default)]
    pub agency_id: String,
    #[serde(default)]
    pub route_short_name: String,
    #[serde(default)]
    pub route_long_name: String,
    pub route_type: u16,
}

/// Raw GTFS trip record.
#[derive(Debug, Deserialize)]
pub struct GtfsTrip {
    pub route_id: String,
    #[serde(default)]
    pub service_id: String,
    pub trip_id: String,
    #[serde(default)]
    pub direction_id: Option<u8>,
    #[serde(default)]
    pub shape_id: Option<String>,
}

/// Raw GTFS stop_time record.
#[derive(Debug)]
pub struct GtfsStopTime {
    pub trip_id: String,
    pub arrival_time: u32,   // seconds since midnight
    pub departure_time: u32, // seconds since midnight
    pub stop_id: String,
    pub stop_sequence: u32,
}

/// Raw GTFS transfer record (transfers.txt).
#[derive(Debug)]
pub struct GtfsTransfer {
    pub from_stop_id: String,
    pub to_stop_id: String,
    pub transfer_type: u8,
    /// Minimum transfer time in seconds (optional, for transfer_type=2).
    pub min_transfer_time: Option<u32>,
}

/// Raw GTFS frequency record (frequencies.txt).
#[derive(Debug)]
pub struct GtfsFrequency {
    pub trip_id: String,
    pub start_time: u32,
    pub end_time: u32,
    pub headway_secs: u32,
    pub exact_times: bool,
}
