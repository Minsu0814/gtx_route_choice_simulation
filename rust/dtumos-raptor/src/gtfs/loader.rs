use std::collections::HashMap;
use std::path::Path;

use crate::error::RaptorError;
use crate::gtfs::types::*;

/// Load stops.txt → Vec<GtfsStop>
pub fn load_stops(gtfs_dir: &Path) -> Result<Vec<GtfsStop>, RaptorError> {
    let path = gtfs_dir.join("stops.txt");
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(&path)
        .map_err(|e| RaptorError::GtfsLoad(format!("stops.txt: {}", e)))?;

    let headers = reader
        .headers()
        .map_err(|e| RaptorError::GtfsLoad(format!("stops.txt headers: {}", e)))?
        .clone();

    let idx_id = find_col(&headers, "stop_id")?;
    let idx_name = find_col_opt(&headers, "stop_name");
    let idx_lat = find_col(&headers, "stop_lat")?;
    let idx_lon = find_col(&headers, "stop_lon")?;

    let mut stops = Vec::new();
    for result in reader.records() {
        let record = result?;
        let lat: f64 = record
            .get(idx_lat)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);
        let lon: f64 = record
            .get(idx_lon)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        // Skip stops without valid coordinates
        if lat == 0.0 && lon == 0.0 {
            continue;
        }

        stops.push(GtfsStop {
            stop_id: record.get(idx_id).unwrap_or("").trim().to_string(),
            stop_name: idx_name
                .and_then(|i| record.get(i))
                .unwrap_or("")
                .trim()
                .to_string(),
            stop_lat: lat,
            stop_lon: lon,
        });
    }

    Ok(stops)
}

/// Load routes.txt → HashMap<route_id, GtfsRoute>
pub fn load_routes(gtfs_dir: &Path) -> Result<HashMap<String, GtfsRoute>, RaptorError> {
    let path = gtfs_dir.join("routes.txt");
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(&path)
        .map_err(|e| RaptorError::GtfsLoad(format!("routes.txt: {}", e)))?;

    let headers = reader.headers()?.clone();
    let idx_id = find_col(&headers, "route_id")?;
    let idx_agency = find_col_opt(&headers, "agency_id");
    let idx_short = find_col_opt(&headers, "route_short_name");
    let idx_long = find_col_opt(&headers, "route_long_name");
    let idx_type = find_col(&headers, "route_type")?;

    let mut routes = HashMap::new();
    for result in reader.records() {
        let record = result?;
        let route_id = record.get(idx_id).unwrap_or("").trim().to_string();
        let route_type: u16 = record
            .get(idx_type)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(3);

        routes.insert(
            route_id.clone(),
            GtfsRoute {
                route_id,
                agency_id: idx_agency
                    .and_then(|i| record.get(i))
                    .unwrap_or("")
                    .trim()
                    .to_string(),
                route_short_name: idx_short
                    .and_then(|i| record.get(i))
                    .unwrap_or("")
                    .trim()
                    .to_string(),
                route_long_name: idx_long
                    .and_then(|i| record.get(i))
                    .unwrap_or("")
                    .trim()
                    .to_string(),
                route_type,
            },
        );
    }

    Ok(routes)
}

/// Load trips.txt → HashMap<trip_id, GtfsTrip>
pub fn load_trips(gtfs_dir: &Path) -> Result<HashMap<String, GtfsTrip>, RaptorError> {
    let path = gtfs_dir.join("trips.txt");
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(&path)
        .map_err(|e| RaptorError::GtfsLoad(format!("trips.txt: {}", e)))?;

    let headers = reader.headers()?.clone();
    let idx_route = find_col(&headers, "route_id")?;
    let idx_service = find_col_opt(&headers, "service_id");
    let idx_trip = find_col(&headers, "trip_id")?;
    let idx_dir = find_col_opt(&headers, "direction_id");
    let idx_shape = find_col_opt(&headers, "shape_id");

    let mut trips = HashMap::new();
    for result in reader.records() {
        let record = result?;
        let trip_id = record.get(idx_trip).unwrap_or("").trim().to_string();
        let shape_id = idx_shape
            .and_then(|i| record.get(i))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        trips.insert(
            trip_id.clone(),
            GtfsTrip {
                route_id: record.get(idx_route).unwrap_or("").trim().to_string(),
                service_id: idx_service
                    .and_then(|i| record.get(i))
                    .unwrap_or("")
                    .trim()
                    .to_string(),
                trip_id,
                direction_id: idx_dir
                    .and_then(|i| record.get(i))
                    .and_then(|s| s.trim().parse().ok()),
                shape_id,
            },
        );
    }

    Ok(trips)
}

/// Load stop_times.txt → Vec<GtfsStopTime>, grouped by trip_id for efficient processing.
/// This is typically the largest file (72MB-470MB).
pub fn load_stop_times(gtfs_dir: &Path) -> Result<HashMap<String, Vec<GtfsStopTime>>, RaptorError> {
    let path = gtfs_dir.join("stop_times.txt");
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(&path)
        .map_err(|e| RaptorError::GtfsLoad(format!("stop_times.txt: {}", e)))?;

    let headers = reader.headers()?.clone();
    let idx_trip = find_col(&headers, "trip_id")?;
    let idx_arr = find_col(&headers, "arrival_time")?;
    let idx_dep = find_col(&headers, "departure_time")?;
    let idx_stop = find_col(&headers, "stop_id")?;
    let idx_seq = find_col(&headers, "stop_sequence")?;

    let mut by_trip: HashMap<String, Vec<GtfsStopTime>> = HashMap::new();

    for result in reader.records() {
        let record = result?;

        let trip_id = record.get(idx_trip).unwrap_or("").trim().to_string();
        let arr_str = record.get(idx_arr).unwrap_or("").trim();
        let dep_str = record.get(idx_dep).unwrap_or("").trim();

        let arrival_time = match parse_gtfs_time(arr_str) {
            Some(t) => t,
            None => continue, // skip rows with invalid times
        };
        let departure_time = match parse_gtfs_time(dep_str) {
            Some(t) => t,
            None => continue,
        };

        let stop_id = record.get(idx_stop).unwrap_or("").trim().to_string();
        let stop_sequence: u32 = record
            .get(idx_seq)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        by_trip.entry(trip_id.clone()).or_default().push(GtfsStopTime {
            trip_id,
            arrival_time,
            departure_time,
            stop_id,
            stop_sequence,
        });
    }

    // Sort each trip's stop_times by stop_sequence
    for stop_times in by_trip.values_mut() {
        stop_times.sort_by_key(|st| st.stop_sequence);
    }

    Ok(by_trip)
}

/// Load transfers.txt → Vec<GtfsTransfer> (optional file).
pub fn load_transfers(gtfs_dir: &Path) -> Result<Vec<GtfsTransfer>, RaptorError> {
    let path = gtfs_dir.join("transfers.txt");
    if !path.exists() {
        return Ok(Vec::new());
    }

    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(&path)
        .map_err(|e| RaptorError::GtfsLoad(format!("transfers.txt: {}", e)))?;

    let headers = reader.headers()?.clone();
    let idx_from = find_col(&headers, "from_stop_id")?;
    let idx_to = find_col(&headers, "to_stop_id")?;
    let idx_type = find_col_opt(&headers, "transfer_type");
    let idx_min_time = find_col_opt(&headers, "min_transfer_time");

    let mut transfers = Vec::new();
    for result in reader.records() {
        let record = result?;
        let transfer_type: u8 = idx_type
            .and_then(|i| record.get(i))
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        transfers.push(GtfsTransfer {
            from_stop_id: record.get(idx_from).unwrap_or("").trim().to_string(),
            to_stop_id: record.get(idx_to).unwrap_or("").trim().to_string(),
            transfer_type,
            min_transfer_time: idx_min_time
                .and_then(|i| record.get(i))
                .and_then(|s| s.trim().parse().ok()),
        });
    }

    Ok(transfers)
}

/// Load frequencies.txt → HashMap<trip_id, Vec<GtfsFrequency>> (optional file).
pub fn load_frequencies(
    gtfs_dir: &Path,
) -> Result<HashMap<String, Vec<GtfsFrequency>>, RaptorError> {
    let path = gtfs_dir.join("frequencies.txt");
    if !path.exists() {
        return Ok(HashMap::new());
    }

    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(&path)
        .map_err(|e| RaptorError::GtfsLoad(format!("frequencies.txt: {}", e)))?;

    let headers = reader.headers()?.clone();
    let idx_trip = find_col(&headers, "trip_id")?;
    let idx_start = find_col(&headers, "start_time")?;
    let idx_end = find_col(&headers, "end_time")?;
    let idx_headway = find_col(&headers, "headway_secs")?;
    let idx_exact = find_col_opt(&headers, "exact_times");

    let mut by_trip: HashMap<String, Vec<GtfsFrequency>> = HashMap::new();

    for result in reader.records() {
        let record = result?;

        let trip_id = record.get(idx_trip).unwrap_or("").trim().to_string();
        let start_time = record
            .get(idx_start)
            .and_then(|s| parse_gtfs_time(s))
            .unwrap_or(0);
        let end_time = record
            .get(idx_end)
            .and_then(|s| parse_gtfs_time(s))
            .unwrap_or(0);
        let headway_secs: u32 = record
            .get(idx_headway)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(600);
        let exact_times: bool = idx_exact
            .and_then(|i| record.get(i))
            .and_then(|s| s.trim().parse::<u8>().ok())
            .unwrap_or(0)
            == 1;

        by_trip.entry(trip_id.clone()).or_default().push(GtfsFrequency {
            trip_id,
            start_time,
            end_time,
            headway_secs,
            exact_times,
        });
    }

    Ok(by_trip)
}

/// Load shapes.txt → HashMap<shape_id, Vec<(lat, lon)>> sorted by sequence (optional file).
pub fn load_shapes(gtfs_dir: &Path) -> Result<HashMap<String, Vec<(f64, f64)>>, RaptorError> {
    let path = gtfs_dir.join("shapes.txt");
    if !path.exists() {
        return Ok(HashMap::new());
    }

    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(&path)
        .map_err(|e| RaptorError::GtfsLoad(format!("shapes.txt: {}", e)))?;

    let headers = reader.headers()?.clone();
    let idx_id = find_col(&headers, "shape_id")?;
    let idx_lat = find_col(&headers, "shape_pt_lat")?;
    let idx_lon = find_col(&headers, "shape_pt_lon")?;
    let idx_seq = find_col(&headers, "shape_pt_sequence")?;

    // Collect raw points with sequence for sorting
    let mut raw: HashMap<String, Vec<(u32, f64, f64)>> = HashMap::new();
    for result in reader.records() {
        let record = result?;
        let shape_id = record.get(idx_id).unwrap_or("").trim().to_string();
        let lat: f64 = record.get(idx_lat).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
        let lon: f64 = record.get(idx_lon).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
        let seq: u32 = record.get(idx_seq).and_then(|s| s.trim().parse().ok()).unwrap_or(0);

        if lat != 0.0 || lon != 0.0 {
            raw.entry(shape_id).or_default().push((seq, lat, lon));
        }
    }

    // Sort by sequence and extract (lat, lon) only
    let mut shapes: HashMap<String, Vec<(f64, f64)>> = HashMap::with_capacity(raw.len());
    for (id, mut pts) in raw {
        pts.sort_by_key(|(seq, _, _)| *seq);
        shapes.insert(id, pts.into_iter().map(|(_, lat, lon)| (lat, lon)).collect());
    }

    eprintln!("[RAPTOR] Loaded {} shapes ({} total points)",
        shapes.len(),
        shapes.values().map(|v| v.len()).sum::<usize>(),
    );

    Ok(shapes)
}

fn find_col(headers: &csv::StringRecord, name: &str) -> Result<usize, RaptorError> {
    headers
        .iter()
        .position(|h| h.trim().trim_start_matches('\u{feff}') == name)
        .ok_or_else(|| RaptorError::GtfsLoad(format!("missing column: {}", name)))
}

fn find_col_opt(headers: &csv::StringRecord, name: &str) -> Option<usize> {
    headers
        .iter()
        .position(|h| h.trim().trim_start_matches('\u{feff}') == name)
}
