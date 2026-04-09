//! GTFS-RT (Real-Time) framework.
//!
//! Provides the data structures and application logic for applying
//! real-time trip updates (delays, cancellations) to the static TransitData.
//!
//! # Architecture
//!
//! ```text
//! External GTFS-RT Feed (protobuf)
//!   → Python parses via gtfs-realtime-bindings
//!   → Calls apply_trip_updates() with per-trip delay arrays
//!   → Rust applies to TransitData (arc-swap atomic update)
//! ```
//!
//! # Status: Framework only
//!
//! Full implementation requires:
//! 1. GTFS-RT protobuf parsing (Python-side, pass to Rust as arrays)
//! 2. Trip matching (trip_id → schedule index)
//! 3. Schedule adjustment (arrival/departure + delay)

use std::collections::HashMap;

/// A single trip delay update.
#[derive(Debug, Clone)]
pub struct TripUpdate {
    pub trip_id: String,
    /// Delay in seconds (positive = late, negative = early, 0 = on time)
    pub delay_secs: i32,
    /// Per-stop delays, indexed by stop_sequence. None = use trip-level delay.
    pub stop_delays: Option<Vec<Option<i32>>>,
    /// Whether this trip is cancelled
    pub cancelled: bool,
}

/// Collection of real-time updates to apply.
#[derive(Debug, Clone, Default)]
pub struct RealtimeUpdates {
    pub trip_updates: Vec<TripUpdate>,
    /// Timestamp of the feed (seconds since epoch)
    pub feed_timestamp: u64,
}

/// Apply real-time updates to a TransitData snapshot.
///
/// Returns a modified copy of the transit data with delays applied.
/// The original data is not modified (immutable design for arc-swap).
///
/// # Current Implementation
///
/// Creates the framework for delay application. Full implementation
/// would clone schedules and adjust arrival/departure times.
pub fn apply_updates(
    data: &crate::types::TransitData,
    updates: &RealtimeUpdates,
) -> ApplyResult {
    // Build trip_id → (route_idx, schedule_idx) lookup
    let mut trip_index: HashMap<&str, (usize, usize)> = HashMap::new();
    for (ri, route) in data.routes.iter().enumerate() {
        for (si, schedule) in route.timetable.schedules.iter().enumerate() {
            trip_index.insert(schedule.trip_id.as_str(), (ri, si));
        }
    }

    let mut applied = 0u32;
    let mut not_found = 0u32;
    let mut cancelled = 0u32;

    for update in &updates.trip_updates {
        if update.cancelled {
            cancelled += 1;
            // TODO: mark trip as cancelled in TransitData
            // For now, just count
        }

        match trip_index.get(update.trip_id.as_str()) {
            Some(&(_ri, _si)) => {
                applied += 1;
                // TODO: Clone schedule and apply delay
                // data.routes[ri].timetable.schedules[si].arrival_times += delay
                // data.routes[ri].timetable.schedules[si].departure_times += delay
            }
            None => {
                not_found += 1;
            }
        }
    }

    ApplyResult {
        applied,
        not_found,
        cancelled,
        feed_timestamp: updates.feed_timestamp,
    }
}

/// Result of applying real-time updates.
#[derive(Debug, Clone)]
pub struct ApplyResult {
    pub applied: u32,
    pub not_found: u32,
    pub cancelled: u32,
    pub feed_timestamp: u64,
}
