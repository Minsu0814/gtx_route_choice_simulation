pub mod access;
pub mod batch;
pub mod error;
pub mod gtfs;
pub mod output;
pub mod raptor;
pub mod realtime;
pub mod types;

use std::sync::Arc;
use std::time::Instant;

use arc_swap::ArcSwap;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::access::street_graph::StreetGraph;
use crate::batch::{BatchConfig, OdQuery, SearchMode};
use crate::error::RaptorError;
use crate::gtfs::build_transit_data;
use crate::output::pydict::values_to_pylist;
use crate::raptor::cost::{CostConfig, TaxiCostConfig};
use crate::raptor::fare::FareConfig;
use crate::types::TransitData;

/// Main PyO3 class: Rust RAPTOR transit routing engine.
#[pyclass]
pub struct DtumosRaptor {
    data: Arc<ArcSwap<Option<TransitData>>>,
    street_graph: Arc<ArcSwap<Option<StreetGraph>>>,
    cost_config: CostConfig,
    taxi_config: TaxiCostConfig,
    fare_config: FareConfig,
    num_threads: usize,
    // Walk access/egress configuration
    max_access_walk_m: f64,
    max_egress_walk_m: f64,
    max_access_stops: usize,
    max_egress_stops: usize,
    walk_speed: f64,
    max_results: usize,
    // Geometry configuration
    use_route_shapes: bool,
    // Taxi access/egress configuration
    enable_taxi_access: bool,
    max_taxi_distance_m: f64,
    min_taxi_distance_m: f64,
    max_taxi_stops: usize,
    // Spatial/time bucketing for large batch optimization
    h3_resolution: u8,
    time_bucket_secs: u32,
}

#[pymethods]
impl DtumosRaptor {
    #[new]
    #[pyo3(signature = (num_threads=None))]
    fn new(num_threads: Option<usize>) -> Self {
        let n = num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

        // Initialize rayon thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok(); // Ignore if already initialized

        Self {
            data: Arc::new(ArcSwap::from_pointee(None)),
            street_graph: Arc::new(ArcSwap::from_pointee(None)),
            cost_config: CostConfig::default(),
            taxi_config: TaxiCostConfig::default(),
            fare_config: FareConfig::default(),
            num_threads: n,
            max_access_walk_m: 800.0,
            max_egress_walk_m: 800.0,
            max_access_stops: 30,
            max_egress_stops: 30,
            walk_speed: 1.2,
            max_results: 5,
            use_route_shapes: false,
            enable_taxi_access: false,
            max_taxi_distance_m: 5000.0,
            min_taxi_distance_m: 1500.0,
            max_taxi_stops: 20,
            h3_resolution: 0,
            time_bucket_secs: 0,
        }
    }

    /// Load GTFS data and build TransitData.
    fn load_gtfs(&self, gtfs_dir: &str) -> PyResult<()> {
        let path = std::path::Path::new(gtfs_dir);
        let transit_data =
            build_transit_data(path).map_err(|e| RaptorError::GtfsLoad(e.to_string()))?;
        self.data.store(Arc::new(Some(transit_data)));
        Ok(())
    }

    /// Load street network graph for walking/taxi distance computation.
    ///
    /// Uses the same graph format as dtumos-routing (NetworkGraph).
    /// When loaded, access/egress distances use Dijkstra on the road network
    /// instead of haversine estimates.
    #[pyo3(signature = (node_lats, node_lons, edge_sources, edge_targets, edge_lengths_m, edge_speeds_kmh))]
    fn load_street_graph<'py>(
        &self,
        py: Python<'py>,
        node_lats: PyReadonlyArray1<'py, f64>,
        node_lons: PyReadonlyArray1<'py, f64>,
        edge_sources: PyReadonlyArray1<'py, u32>,
        edge_targets: PyReadonlyArray1<'py, u32>,
        edge_lengths_m: PyReadonlyArray1<'py, f64>,
        edge_speeds_kmh: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyObject> {
        let start = Instant::now();

        let lats = node_lats.as_slice()?;
        let lons = node_lons.as_slice()?;
        let srcs = edge_sources.as_slice()?;
        let tgts = edge_targets.as_slice()?;
        let lens = edge_lengths_m.as_slice()?;
        let spds = edge_speeds_kmh.as_slice()?;

        let graph = StreetGraph::from_arrays(
            lats,
            lons,
            srcs,
            tgts,
            lens,
            spds,
            self.walk_speed,
        );

        let n_nodes = graph.num_nodes();
        let n_edges = graph.num_edges();

        self.street_graph.store(Arc::new(Some(graph)));

        let elapsed = start.elapsed().as_millis() as u64;

        let dict = PyDict::new_bound(py);
        dict.set_item("nodes", n_nodes)?;
        dict.set_item("edges", n_edges)?;
        dict.set_item("build_time_ms", elapsed)?;
        Ok(dict.into_any().unbind())
    }

    /// Hot-reload GTFS: build new TransitData and atomically swap.
    /// In-progress routing uses old data; new requests use new data.
    fn reload_gtfs(&self, py: Python<'_>, gtfs_dir: &str) -> PyResult<PyObject> {
        let start = Instant::now();
        let path = std::path::Path::new(gtfs_dir);

        // Build new data (can be slow, but doesn't block routing)
        let new_data =
            build_transit_data(path).map_err(|e| RaptorError::GtfsLoad(e.to_string()))?;

        let stops = new_data.stop_count();
        let routes = new_data.route_count();
        let trips = new_data.total_trip_count();

        // Atomic swap
        self.data.store(Arc::new(Some(new_data)));

        let elapsed = start.elapsed().as_millis() as u64;

        let dict = PyDict::new_bound(py);
        dict.set_item("build_time_ms", elapsed)?;
        dict.set_item("stops", stops)?;
        dict.set_item("routes", routes)?;
        dict.set_item("trips", trips)?;
        Ok(dict.into_any().unbind())
    }

    /// Update cost configuration from a Python dict.
    fn update_config(&mut self, config: &Bound<'_, PyDict>) -> PyResult<()> {
        // Cost config
        if let Some(v) = config.get_item("first_board_cost_secs")? {
            self.cost_config.first_board_cost_secs = v.extract()?;
        }
        if let Some(v) = config.get_item("transfer_cost_secs")? {
            self.cost_config.transfer_cost_secs = v.extract()?;
        }
        if let Some(v) = config.get_item("wait_reluctance")? {
            self.cost_config.wait_reluctance = v.extract()?;
        }
        if let Some(v) = config.get_item("walk_reluctance")? {
            self.cost_config.walk_reluctance = v.extract()?;
        }
        if let Some(v) = config.get_item("transit_reluctance")? {
            let arr: Vec<f64> = v.extract()?;
            if arr.len() == 6 {
                self.cost_config.transit_reluctance = [
                    arr[0], arr[1], arr[2], arr[3], arr[4], arr[5],
                ];
            }
        }
        if let Some(v) = config.get_item("board_slack")? {
            let arr: Vec<u32> = v.extract()?;
            if arr.len() == 6 {
                self.cost_config.board_slack = [
                    arr[0], arr[1], arr[2], arr[3], arr[4], arr[5],
                ];
            }
        }
        if let Some(v) = config.get_item("alight_slack")? {
            let arr: Vec<u32> = v.extract()?;
            if arr.len() == 6 {
                self.cost_config.alight_slack = [
                    arr[0], arr[1], arr[2], arr[3], arr[4], arr[5],
                ];
            }
        }
        if let Some(v) = config.get_item("transfer_slack")? {
            self.cost_config.transfer_slack = v.extract()?;
        }

        // Walk/routing config
        if let Some(v) = config.get_item("max_access_walk_m")? {
            self.max_access_walk_m = v.extract()?;
        }
        if let Some(v) = config.get_item("max_egress_walk_m")? {
            self.max_egress_walk_m = v.extract()?;
        }
        if let Some(v) = config.get_item("max_access_stops")? {
            self.max_access_stops = v.extract()?;
        }
        if let Some(v) = config.get_item("max_egress_stops")? {
            self.max_egress_stops = v.extract()?;
        }
        if let Some(v) = config.get_item("walk_speed")? {
            self.walk_speed = v.extract()?;
        }
        if let Some(v) = config.get_item("max_results")? {
            self.max_results = v.extract()?;
        }

        // Geometry config
        if let Some(v) = config.get_item("use_route_shapes")? {
            self.use_route_shapes = v.extract()?;
        }

        // Taxi config
        if let Some(v) = config.get_item("enable_taxi_access")? {
            self.enable_taxi_access = v.extract()?;
        }
        if let Some(v) = config.get_item("max_taxi_distance_m")? {
            self.max_taxi_distance_m = v.extract()?;
        }
        if let Some(v) = config.get_item("min_taxi_distance_m")? {
            self.min_taxi_distance_m = v.extract()?;
        }
        if let Some(v) = config.get_item("max_taxi_stops")? {
            self.max_taxi_stops = v.extract()?;
        }
        if let Some(v) = config.get_item("taxi_base_fare_krw")? {
            self.taxi_config.base_fare_krw = v.extract()?;
        }
        if let Some(v) = config.get_item("taxi_dist_fare_krw")? {
            self.taxi_config.dist_fare_krw = v.extract()?;
        }
        if let Some(v) = config.get_item("taxi_dist_unit_m")? {
            self.taxi_config.dist_unit_m = v.extract()?;
        }
        if let Some(v) = config.get_item("taxi_road_factor")? {
            self.taxi_config.road_factor = v.extract()?;
        }
        if let Some(v) = config.get_item("taxi_speed_mps")? {
            self.taxi_config.taxi_speed_mps = v.extract()?;
        }
        if let Some(v) = config.get_item("taxi_default_wait_secs")? {
            self.taxi_config.default_wait_secs = v.extract()?;
        }
        if let Some(v) = config.get_item("taxi_default_surge")? {
            self.taxi_config.default_surge = v.extract()?;
        }
        if let Some(v) = config.get_item("taxi_vot_factor")? {
            self.taxi_config.vot_factor = v.extract()?;
        }
        if let Some(v) = config.get_item("taxi_reluctance")? {
            self.taxi_config.taxi_reluctance = v.extract()?;
        }

        // Fare config
        if let Some(v) = config.get_item("fare_enabled")? {
            self.fare_config.enabled = v.extract()?;
        }
        if let Some(v) = config.get_item("fare_bus_base")? {
            self.fare_config.bus_base_fare = v.extract()?;
        }
        if let Some(v) = config.get_item("fare_subway_base")? {
            self.fare_config.subway_base_fare = v.extract()?;
        }
        if let Some(v) = config.get_item("fare_base_distance_m")? {
            self.fare_config.base_distance_m = v.extract()?;
        }
        if let Some(v) = config.get_item("fare_transfer_window_secs")? {
            self.fare_config.transfer_window_secs = v.extract()?;
        }
        if let Some(v) = config.get_item("fare_max_free_transfers")? {
            self.fare_config.max_free_transfers = v.extract()?;
        }

        // Bucketing config (large batch optimization)
        if let Some(v) = config.get_item("h3_resolution")? {
            self.h3_resolution = v.extract()?;
        }
        if let Some(v) = config.get_item("time_bucket_secs")? {
            self.time_bucket_secs = v.extract()?;
        }

        Ok(())
    }

    /// Route a single OD pair.
    #[pyo3(signature = (from_lat, from_lon, to_lat, to_lon, departure_time_secs, mode=None))]
    fn route(
        &self,
        py: Python<'_>,
        from_lat: f64,
        from_lon: f64,
        to_lat: f64,
        to_lon: f64,
        departure_time_secs: u32,
        mode: Option<&str>,
    ) -> PyResult<PyObject> {
        let data_guard = self.data.load();
        let data = data_guard
            .as_ref()
            .as_ref()
            .ok_or(RaptorError::NoData)?;

        let graph_guard = self.street_graph.load();
        let graph_ref = graph_guard.as_ref().as_ref();

        let search_mode = parse_mode(mode);
        let batch_config = self.make_batch_config(search_mode);
        let taxi_config = self.taxi_config.clone();

        let query = OdQuery {
            from_lat,
            from_lon,
            to_lat,
            to_lon,
            departure_time: departure_time_secs,
            taxi_wait_secs: None,
            taxi_surge: None,
        };

        let result = batch::route_batch(
            data,
            &[query],
            &self.cost_config,
            &taxi_config,
            &batch_config,
            graph_ref,
        );

        crate::output::pydict::value_to_pyobject(py, &result[0])
    }

    /// Batch routing with numpy arrays. Releases GIL for parallel execution.
    ///
    /// Optional `taxi_wait_secs` and `taxi_surge` arrays provide per-OD
    /// taxi parameters from ServiceFeed (H3 zone-based dynamic data).
    /// When None, global TaxiCostConfig defaults are used.
    #[pyo3(signature = (from_lats, from_lons, to_lats, to_lons, departure_times, mode=None, taxi_wait_secs=None, taxi_surge=None, grouped=None))]
    fn route_batch<'py>(
        &self,
        py: Python<'py>,
        from_lats: PyReadonlyArray1<'py, f64>,
        from_lons: PyReadonlyArray1<'py, f64>,
        to_lats: PyReadonlyArray1<'py, f64>,
        to_lons: PyReadonlyArray1<'py, f64>,
        departure_times: PyReadonlyArray1<'py, u32>,
        mode: Option<&str>,
        taxi_wait_secs: Option<PyReadonlyArray1<'py, u32>>,
        taxi_surge: Option<PyReadonlyArray1<'py, f64>>,
        grouped: Option<bool>,
    ) -> PyResult<PyObject> {
        let fl = from_lats.as_slice()?;
        let flo = from_lons.as_slice()?;
        let tl = to_lats.as_slice()?;
        let tlo = to_lons.as_slice()?;
        let dt = departure_times.as_slice()?;

        let n = fl.len();
        if flo.len() != n || tl.len() != n || tlo.len() != n || dt.len() != n {
            return Err(RaptorError::Config("array lengths must match".into()).into());
        }

        // Extract optional per-OD taxi parameters
        let tw: Option<Vec<u32>> = taxi_wait_secs
            .map(|arr| arr.as_slice().map(|s| s.to_vec()))
            .transpose()?;
        let ts: Option<Vec<f64>> = taxi_surge
            .map(|arr| arr.as_slice().map(|s| s.to_vec()))
            .transpose()?;

        let queries: Vec<OdQuery> = (0..n)
            .map(|i| OdQuery {
                from_lat: fl[i],
                from_lon: flo[i],
                to_lat: tl[i],
                to_lon: tlo[i],
                departure_time: dt[i],
                taxi_wait_secs: tw.as_ref().map(|v| v[i]),
                taxi_surge: ts.as_ref().map(|v| v[i]),
            })
            .collect();

        let data_guard = self.data.load();
        let data = data_guard
            .as_ref()
            .as_ref()
            .ok_or(RaptorError::NoData)?;

        let graph_guard = self.street_graph.load();
        let graph_ref = graph_guard.as_ref().as_ref();

        let search_mode = parse_mode(mode);
        let mut batch_config = self.make_batch_config(search_mode);
        batch_config.force_grouped = grouped;
        let cost_config = self.cost_config.clone();
        let taxi_config = self.taxi_config.clone();

        // Release GIL for parallel computation
        let results = py.allow_threads(|| {
            batch::route_batch(
                data,
                &queries,
                &cost_config,
                &taxi_config,
                &batch_config,
                graph_ref,
            )
        });

        // Convert to Python objects directly (bypasses json.loads round-trip)
        values_to_pylist(py, &results)
    }

    /// Simulation-mode batch: returns only LOS attributes as dict of numpy arrays.
    ///
    /// ~2x faster than route_batch for large batches by skipping JSON/polyline.
    /// Returns dict with 15 numpy float32 arrays (one element per query):
    ///   duration_sec, generalized_cost, num_transfers, fare_krw,
    ///   access_time_sec, egress_time_sec, wait_time_sec, walk_time_sec,
    ///   ivt_total_sec, ivt_bus_sec, ivt_subway_sec, ivt_rail_sec,
    ///   walk_distance_m, taxi_access_fare, taxi_egress_fare
    #[pyo3(signature = (from_lats, from_lons, to_lats, to_lons, departure_times, mode=None, taxi_wait_secs=None, taxi_surge=None))]
    fn route_batch_attrs<'py>(
        &self,
        py: Python<'py>,
        from_lats: PyReadonlyArray1<'py, f64>,
        from_lons: PyReadonlyArray1<'py, f64>,
        to_lats: PyReadonlyArray1<'py, f64>,
        to_lons: PyReadonlyArray1<'py, f64>,
        departure_times: PyReadonlyArray1<'py, u32>,
        mode: Option<&str>,
        taxi_wait_secs: Option<PyReadonlyArray1<'py, u32>>,
        taxi_surge: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<PyObject> {
        let fl = from_lats.as_slice()?;
        let flo = from_lons.as_slice()?;
        let tl = to_lats.as_slice()?;
        let tlo = to_lons.as_slice()?;
        let dt = departure_times.as_slice()?;

        let n = fl.len();
        if flo.len() != n || tl.len() != n || tlo.len() != n || dt.len() != n {
            return Err(RaptorError::Config("array lengths must match".into()).into());
        }

        let tw: Option<Vec<u32>> = taxi_wait_secs
            .map(|arr| arr.as_slice().map(|s| s.to_vec()))
            .transpose()?;
        let ts: Option<Vec<f64>> = taxi_surge
            .map(|arr| arr.as_slice().map(|s| s.to_vec()))
            .transpose()?;

        let queries: Vec<batch::OdQuery> = (0..n)
            .map(|i| batch::OdQuery {
                from_lat: fl[i],
                from_lon: flo[i],
                to_lat: tl[i],
                to_lon: tlo[i],
                departure_time: dt[i],
                taxi_wait_secs: tw.as_ref().map(|v| v[i]),
                taxi_surge: ts.as_ref().map(|v| v[i]),
            })
            .collect();

        let data_guard = self.data.load();
        let data = data_guard
            .as_ref()
            .as_ref()
            .ok_or(RaptorError::NoData)?;

        let graph_guard = self.street_graph.load();
        let graph_ref = graph_guard.as_ref().as_ref();

        let search_mode = parse_mode(mode);
        let batch_config = self.make_batch_config(search_mode);
        let cost_config = self.cost_config.clone();
        let taxi_config = self.taxi_config.clone();

        // Release GIL for parallel computation
        let attrs = py.allow_threads(|| {
            batch::route_batch_attrs(
                data,
                &queries,
                &cost_config,
                &taxi_config,
                &batch_config,
                graph_ref,
            )
        });

        // Convert to dict of numpy arrays (columnar, zero-copy friendly)
        let dict = PyDict::new_bound(py);
        macro_rules! col {
            ($name:ident) => {
                {
                    let arr: Vec<f32> = attrs.iter().map(|a| a.$name).collect();
                    let np_arr = numpy::PyArray1::from_vec_bound(py, arr);
                    dict.set_item(stringify!($name), np_arr)?;
                }
            };
        }
        col!(duration_sec);
        col!(generalized_cost);
        col!(num_transfers);
        col!(fare_krw);
        col!(access_time_sec);
        col!(egress_time_sec);
        col!(wait_time_sec);
        col!(walk_time_sec);
        col!(ivt_total_sec);
        col!(ivt_bus_sec);
        col!(ivt_subway_sec);
        col!(ivt_rail_sec);
        col!(walk_distance_m);
        col!(taxi_access_fare);
        col!(taxi_egress_fare);

        Ok(dict.into_any().unbind())
    }

    /// Route a single OD pair with Range-RAPTOR (time window search).
    #[pyo3(signature = (from_lat, from_lon, to_lat, to_lon, earliest_dep, latest_dep, step_secs=None))]
    fn route_range(
        &self,
        py: Python<'_>,
        from_lat: f64,
        from_lon: f64,
        to_lat: f64,
        to_lon: f64,
        earliest_dep: u32,
        latest_dep: u32,
        step_secs: Option<u32>,
    ) -> PyResult<PyObject> {
        let data_guard = self.data.load();
        let data = data_guard
            .as_ref()
            .as_ref()
            .ok_or(RaptorError::NoData)?;

        let graph_guard = self.street_graph.load();
        let graph_ref = graph_guard.as_ref().as_ref();

        let mut batch_config = self.make_batch_config(SearchMode::Range {
            step_secs: step_secs.unwrap_or(60),
            window_secs: 0, // Not used — explicit earliest/latest
        });
        batch_config.search_mode = SearchMode::Range {
            step_secs: step_secs.unwrap_or(60),
            window_secs: latest_dep.saturating_sub(earliest_dep),
        };
        let taxi_config = self.taxi_config.clone();

        let query = OdQuery {
            from_lat,
            from_lon,
            to_lat,
            to_lon,
            departure_time: earliest_dep,
            taxi_wait_secs: None,
            taxi_surge: None,
        };

        let result = batch::route_batch(
            data,
            &[query],
            &self.cost_config,
            &taxi_config,
            &batch_config,
            graph_ref,
        );

        crate::output::pydict::value_to_pyobject(py, &result[0])
    }

    /// Refine walking transfers using the street network graph.
    ///
    /// Validates haversine-based transfers against actual walkable paths.
    /// Transfers blocked by physical barriers (rivers, highways) are removed.
    /// Reachable transfers get accurate network-based walk times.
    ///
    /// Must be called after both `load_gtfs()` and `load_street_graph()`.
    /// Runs once at setup time — zero query-time cost.
    fn refine_transfers(&self, py: Python<'_>) -> PyResult<PyObject> {
        let start = Instant::now();

        let graph_guard = self.street_graph.load();
        let graph = graph_guard
            .as_ref()
            .as_ref()
            .ok_or(RaptorError::Config("street graph not loaded".into()))?;

        // Clone current TransitData, refine, then atomically swap
        let data_guard = self.data.load();
        let current = data_guard
            .as_ref()
            .as_ref()
            .ok_or(RaptorError::NoData)?;

        let mut new_data = TransitData {
            stop_ids: current.stop_ids.clone(),
            stop_names: current.stop_names.clone(),
            stop_lats: current.stop_lats.clone(),
            stop_lons: current.stop_lons.clone(),
            routes: current.routes.clone(),
            transfers_from: current.transfers_from.clone(),
            routes_by_stop: current.routes_by_stop.clone(),
            service_start: current.service_start,
            service_end: current.service_end,
            stop_rtree: {
                let pts: Vec<crate::types::StopPoint> = (0..current.stop_count())
                    .map(|i| crate::types::StopPoint {
                        lat: current.stop_lats[i],
                        lon: current.stop_lons[i],
                        index: i as u32,
                    })
                    .collect();
                rstar::RTree::bulk_load(pts)
            },
        };

        let (validated, removed, total_before) =
            crate::gtfs::refine_transfers_with_graph(&mut new_data, graph);

        // Atomic swap
        self.data.store(Arc::new(Some(new_data)));

        let elapsed = start.elapsed().as_millis() as u64;

        eprintln!(
            "[RAPTOR] Transfers refined: {}/{} validated, {} removed ({:.1}s)",
            validated, total_before, removed, elapsed as f64 / 1000.0
        );

        let dict = PyDict::new_bound(py);
        dict.set_item("validated", validated)?;
        dict.set_item("removed", removed)?;
        dict.set_item("total_before", total_before)?;
        dict.set_item("elapsed_ms", elapsed)?;
        Ok(dict.into_any().unbind())
    }

    /// Update route reliability scores.
    ///
    /// `route_scores`: dict mapping route_id → reliability (0.0–1.0)
    fn update_reliability(&self, route_scores: &Bound<'_, PyDict>) -> PyResult<()> {
        // Note: this requires mutable access to TransitData which is behind ArcSwap.
        // For now, log the intent — full implementation needs TransitData clone + swap.
        let _count = route_scores.len();
        eprintln!(
            "[RAPTOR] update_reliability: {} routes (framework only — requires TransitData rebuild)",
            _count
        );
        Ok(())
    }

    /// Update route crowding levels.
    ///
    /// `route_levels`: dict mapping route_id → crowding (0.0–1.0)
    fn update_crowding(&self, route_levels: &Bound<'_, PyDict>) -> PyResult<()> {
        let _count = route_levels.len();
        eprintln!(
            "[RAPTOR] update_crowding: {} routes (framework only — requires TransitData rebuild)",
            _count
        );
        Ok(())
    }

    /// Get engine statistics.
    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        let data_guard = self.data.load();

        if let Some(data) = data_guard.as_ref().as_ref() {
            dict.set_item("loaded", true)?;
            dict.set_item("stops", data.stop_count())?;
            dict.set_item("routes", data.route_count())?;
            dict.set_item("trips", data.total_trip_count())?;
            dict.set_item("service_start", data.service_start)?;
            dict.set_item("service_end", data.service_end)?;

            let transfer_count: usize = data.transfers_from.iter().map(|v| v.len()).sum();
            dict.set_item("transfers", transfer_count)?;
        } else {
            dict.set_item("loaded", false)?;
        }

        let graph_guard = self.street_graph.load();
        if let Some(graph) = graph_guard.as_ref().as_ref() {
            dict.set_item("street_graph_loaded", true)?;
            dict.set_item("street_nodes", graph.num_nodes())?;
            dict.set_item("street_edges", graph.num_edges())?;
        } else {
            dict.set_item("street_graph_loaded", false)?;
        }

        dict.set_item("num_threads", self.num_threads)?;
        dict.set_item("engine", "rust")?;
        dict.set_item("taxi_access_enabled", self.enable_taxi_access)?;
        Ok(dict.into_any().unbind())
    }
}

impl DtumosRaptor {
    fn make_batch_config(&self, search_mode: SearchMode) -> BatchConfig {
        BatchConfig {
            max_access_walk_m: self.max_access_walk_m,
            max_egress_walk_m: self.max_egress_walk_m,
            max_access_stops: self.max_access_stops,
            max_egress_stops: self.max_egress_stops,
            walk_speed: self.walk_speed,
            max_results: self.max_results,
            search_mode,
            mc_relax_ratio: 1.0,
            mc_relax_slack: 0,
            enable_taxi_access: self.enable_taxi_access,
            max_taxi_distance_m: self.max_taxi_distance_m,
            min_taxi_distance_m: self.min_taxi_distance_m,
            max_taxi_stops: self.max_taxi_stops,
            fare_config: self.fare_config.clone(),
            use_route_shapes: self.use_route_shapes,
            force_grouped: None,
            h3_resolution: self.h3_resolution,
            time_bucket_secs: self.time_bucket_secs,
        }
    }
}

fn parse_mode(mode: Option<&str>) -> SearchMode {
    match mode {
        Some("standard") => SearchMode::Standard,
        Some("multi_criteria") | None => SearchMode::MultiCriteria,
        Some("research") => SearchMode::Research,
        Some("range") => SearchMode::Range {
            step_secs: 60,
            window_secs: 1800,
        },
        _ => SearchMode::MultiCriteria,
    }
}

/// PyO3 module definition.
#[pymodule]
fn dtumos_raptor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DtumosRaptor>()?;
    Ok(())
}
