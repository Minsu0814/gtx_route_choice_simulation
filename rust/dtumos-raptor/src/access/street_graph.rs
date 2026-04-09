//! Street network graph for walking & taxi distance computation.
//!
//! Loaded from the same NetworkGraph data that `dtumos-routing` uses.
//! Supports two modes:
//! - Walking: edge weight = length_m / walk_speed_mps (seconds)
//! - Driving: edge weight = length_m / speed_kmh * 3.6 (seconds)
//!
//! Uses bounded Dijkstra (early termination at max_distance) for efficiency.
//! Thread-local scratch buffers avoid per-query allocations on large graphs.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rstar::{PointDistance, RTree, RTreeObject, AABB};

/// Node in the street network.
#[derive(Debug, Clone)]
pub struct StreetNode {
    pub lat: f64,
    pub lon: f64,
}

/// Directed edge in the street network.
#[derive(Debug, Clone)]
pub struct StreetEdge {
    pub target: u32,
    /// Walking time in milliseconds.
    pub walk_weight_ms: u32,
    /// Driving time in milliseconds (u32::MAX for walk-only reverse edges).
    pub drive_weight_ms: u32,
    /// Length in meters.
    pub length_m: f32,
}

/// R-tree entry for snapping coordinates to nearest graph node.
#[derive(Debug, Clone)]
struct SnapPoint {
    lon: f64,
    lat: f64,
    node_id: u32,
}

impl RTreeObject for SnapPoint {
    type Envelope = AABB<[f64; 2]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.lat, self.lon])
    }
}

impl PointDistance for SnapPoint {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let dx = self.lat - point[0];
        let dy = self.lon - point[1];
        dx * dx + dy * dy
    }
}

// ─── Thread-local Dijkstra scratch ───────────────────────────────────

/// Reusable scratch buffers for bounded Dijkstra.
/// Avoids allocating 120K+ entry vectors on every query.
/// Uses a dirty-list to reset only touched entries — O(k) instead of O(n).
struct DijkstraScratch {
    dist: Vec<u32>,
    dist_m: Vec<f32>,
    prev: Vec<u32>,
    dirty: Vec<u32>,
    heap: BinaryHeap<Reverse<(u32, u32)>>,
}

impl DijkstraScratch {
    fn new() -> Self {
        Self {
            dist: Vec::new(),
            dist_m: Vec::new(),
            prev: Vec::new(),
            dirty: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// Grow buffers if needed (first use or graph size changed).
    fn ensure_size(&mut self, n: usize) {
        if self.dist.len() < n {
            self.dist.resize(n, u32::MAX);
            self.dist_m.resize(n, 0.0);
            self.prev.resize(n, u32::MAX);
        }
    }

    /// Reset only the entries modified by the last query.
    fn reset(&mut self) {
        for &idx in &self.dirty {
            let i = idx as usize;
            self.dist[i] = u32::MAX;
            self.dist_m[i] = 0.0;
            self.prev[i] = u32::MAX;
        }
        self.dirty.clear();
        // heap should already be empty after Dijkstra, but clear defensively
        self.heap.clear();
    }
}

thread_local! {
    static SCRATCH: RefCell<DijkstraScratch> = RefCell::new(DijkstraScratch::new());
}

// ─── StreetGraph ─────────────────────────────────────────────────────

/// Compact street network with bidirectional adjacency + R-tree snapping.
pub struct StreetGraph {
    pub nodes: Vec<StreetNode>,
    /// Forward adjacency list: adj[source] = [StreetEdge, ...]
    adj: Vec<Vec<StreetEdge>>,
    /// R-tree for coordinate → node snapping.
    snap_tree: RTree<SnapPoint>,
}

impl StreetGraph {
    /// Build from parallel arrays (same format as dtumos-routing load_graph).
    ///
    /// `walk_speed_mps`: walking speed in m/s (typically 1.2).
    pub fn from_arrays(
        node_lats: &[f64],
        node_lons: &[f64],
        edge_sources: &[u32],
        edge_targets: &[u32],
        edge_lengths_m: &[f64],
        edge_speeds_kmh: &[f64],
        walk_speed_mps: f64,
    ) -> Self {
        let n_nodes = node_lats.len();

        let nodes: Vec<StreetNode> = node_lats
            .iter()
            .zip(node_lons.iter())
            .map(|(&lat, &lon)| StreetNode { lat, lon })
            .collect();

        let mut adj: Vec<Vec<StreetEdge>> = vec![Vec::new(); n_nodes];

        // Track which (src, tgt) pairs exist to avoid duplicate reverse edges.
        let mut edge_set = std::collections::HashSet::new();

        for i in 0..edge_sources.len() {
            let src = edge_sources[i] as usize;
            let tgt = edge_targets[i];
            if src >= n_nodes || (tgt as usize) >= n_nodes {
                continue;
            }
            let length = edge_lengths_m[i];
            let speed = edge_speeds_kmh[i].max(1.0);

            let walk_ms = (length / walk_speed_mps * 1000.0) as u32;
            let drive_ms = (length / (speed / 3.6) * 1000.0) as u32;

            adj[src].push(StreetEdge {
                target: tgt,
                walk_weight_ms: walk_ms.max(1),
                drive_weight_ms: drive_ms.max(1),
                length_m: length as f32,
            });
            edge_set.insert((src as u32, tgt));
        }

        // Add reverse edges for walkability on one-way streets.
        // Pedestrians can walk against traffic, so every edge should be
        // traversable in both directions for walking. Reverse edges get
        // drive_weight_ms = u32::MAX so they are ignored for taxi routing.
        let mut reverse_added = 0usize;
        for i in 0..edge_sources.len() {
            let src = edge_sources[i];
            let tgt = edge_targets[i] as usize;
            if !edge_set.contains(&(edge_targets[i], src)) {
                let length = edge_lengths_m[i];
                let walk_ms = (length / walk_speed_mps * 1000.0) as u32;
                adj[tgt].push(StreetEdge {
                    target: src,
                    walk_weight_ms: walk_ms.max(1),
                    drive_weight_ms: u32::MAX,
                    length_m: length as f32,
                });
                reverse_added += 1;
            }
        }

        if reverse_added > 0 {
            eprintln!(
                "[RAPTOR] StreetGraph: {} fwd + {} reverse walk edges ({} nodes)",
                edge_sources.len(),
                reverse_added,
                n_nodes,
            );
        }

        // Build R-tree for snapping
        let snap_points: Vec<SnapPoint> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| SnapPoint {
                lat: n.lat,
                lon: n.lon,
                node_id: i as u32,
            })
            .collect();

        let snap_tree = RTree::bulk_load(snap_points);

        Self {
            nodes,
            adj,
            snap_tree,
        }
    }

    /// Snap a coordinate to the nearest graph node within `max_radius_m`.
    /// Returns (node_id, snap_distance_m).
    pub fn snap(&self, lat: f64, lon: f64, max_radius_m: f64) -> Option<(u32, f64)> {
        let nearest = self.snap_tree.nearest_neighbor(&[lat, lon])?;
        let dist = haversine_m(lat, lon, nearest.lat, nearest.lon);
        if dist <= max_radius_m {
            Some((nearest.node_id, dist))
        } else {
            None
        }
    }

    /// Bounded Dijkstra from a snapped node, returning all reachable nodes
    /// within `max_time_ms` (walking or driving).
    ///
    /// Uses thread-local scratch buffers with dirty-list reset to avoid
    /// allocating O(n) vectors on every call.
    ///
    /// Returns: Vec<(node_id, time_ms, distance_m)> sorted by time.
    pub fn dijkstra_bounded(
        &self,
        source: u32,
        max_time_ms: u32,
        use_walk: bool,
    ) -> Vec<(u32, u32, f32)> {
        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            s.ensure_size(self.nodes.len());

            let src = source as usize;
            if src >= self.nodes.len() {
                return Vec::new();
            }

            // Initialize source
            s.dist[src] = 0;
            s.dirty.push(source);
            s.heap.push(Reverse((0, source)));

            // Explore
            while let Some(Reverse((d, u))) = s.heap.pop() {
                let ui = u as usize;
                if d > s.dist[ui] || d > max_time_ms {
                    continue;
                }

                for edge in &self.adj[ui] {
                    let w = if use_walk {
                        edge.walk_weight_ms
                    } else {
                        edge.drive_weight_ms
                    };
                    let new_d = d.saturating_add(w);
                    let ti = edge.target as usize;
                    if new_d <= max_time_ms && new_d < s.dist[ti] {
                        if s.dist[ti] == u32::MAX {
                            // First visit — track for reset
                            s.dirty.push(edge.target);
                        }
                        s.dist[ti] = new_d;
                        s.dist_m[ti] = s.dist_m[ui] + edge.length_m;
                        s.prev[ti] = u;
                        s.heap.push(Reverse((new_d, edge.target)));
                    }
                }
            }

            // Collect results from dirty list only — O(k) not O(n)
            let mut results: Vec<(u32, u32, f32)> = Vec::with_capacity(s.dirty.len());
            for &node in &s.dirty {
                let i = node as usize;
                if i != src {
                    results.push((node, s.dist[i], s.dist_m[i]));
                }
            }
            results.sort_by_key(|&(_, t, _)| t);

            // Reset only touched entries for next use
            s.reset();

            results
        })
    }

    /// Compute walking distance and time from (lat, lon) to a target node.
    ///
    /// Returns (walk_time_secs, walk_distance_m) or None if unreachable.
    pub fn walk_distance(
        &self,
        from_lat: f64,
        from_lon: f64,
        to_node: u32,
        max_walk_m: f64,
    ) -> Option<(u32, f64)> {
        let (src_node, snap_dist) = self.snap(from_lat, from_lon, max_walk_m)?;
        if src_node == to_node {
            let walk_time = (snap_dist / 1.2) as u32;
            return Some((walk_time, snap_dist));
        }

        let max_time_ms = (max_walk_m / 1.2 * 1000.0) as u32;
        let reachable = self.dijkstra_bounded(src_node, max_time_ms, true);

        for &(node, time_ms, dist_m) in &reachable {
            if node == to_node {
                let total_dist = snap_dist + dist_m as f64;
                let total_time = (snap_dist / 1.2) as u32 + time_ms / 1000;
                return Some((total_time, total_dist));
            }
        }

        None
    }

    /// Compute driving distance and time from (lat, lon) to a target node.
    ///
    /// Returns (drive_time_secs, road_distance_m) or None if unreachable.
    pub fn drive_distance(
        &self,
        from_lat: f64,
        from_lon: f64,
        to_node: u32,
        max_drive_m: f64,
    ) -> Option<(u32, f64)> {
        let (src_node, snap_dist) = self.snap(from_lat, from_lon, max_drive_m)?;
        if src_node == to_node {
            let drive_time = (snap_dist / 8.33) as u32;
            return Some((drive_time, snap_dist));
        }

        let max_time_ms = (max_drive_m / 8.33 * 1000.0) as u32;
        let reachable = self.dijkstra_bounded(src_node, max_time_ms, false);

        for &(node, time_ms, dist_m) in &reachable {
            if node == to_node {
                let total_dist = snap_dist + dist_m as f64;
                let total_time = (snap_dist / 8.33) as u32 + time_ms / 1000;
                return Some((total_time, total_dist));
            }
        }

        None
    }

    /// Compute walking path geometry between two coordinates.
    ///
    /// Returns Vec<(lat, lon)> following the street network, or empty if unreachable.
    /// Uses bounded Dijkstra with predecessor tracking — efficient for short walks (<1km).
    pub fn walk_path(
        &self,
        from_lat: f64,
        from_lon: f64,
        to_lat: f64,
        to_lon: f64,
        max_walk_m: f64,
    ) -> Vec<(f64, f64)> {
        let src_snap = match self.snap(from_lat, from_lon, max_walk_m) {
            Some((node, _)) => node,
            None => return Vec::new(),
        };
        let tgt_snap = match self.snap(to_lat, to_lon, max_walk_m) {
            Some((node, _)) => node,
            None => return Vec::new(),
        };

        if src_snap == tgt_snap {
            return vec![(from_lat, from_lon), (to_lat, to_lon)];
        }

        let max_time_ms = (max_walk_m / 1.2 * 1000.0) as u32;

        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            s.ensure_size(self.nodes.len());

            let src = src_snap as usize;
            s.dist[src] = 0;
            s.prev[src] = u32::MAX;
            s.dirty.push(src_snap);
            s.heap.push(Reverse((0, src_snap)));

            let tgt = tgt_snap as usize;
            let mut found = false;

            while let Some(Reverse((d, u))) = s.heap.pop() {
                let ui = u as usize;
                if d > s.dist[ui] || d > max_time_ms {
                    continue;
                }
                if ui == tgt {
                    found = true;
                    break;
                }
                for edge in &self.adj[ui] {
                    let new_d = d.saturating_add(edge.walk_weight_ms);
                    let ti = edge.target as usize;
                    if new_d <= max_time_ms && new_d < s.dist[ti] {
                        if s.dist[ti] == u32::MAX {
                            s.dirty.push(edge.target);
                        }
                        s.dist[ti] = new_d;
                        s.prev[ti] = u;
                        s.heap.push(Reverse((new_d, edge.target)));
                    }
                }
            }

            let path = if found {
                // Reconstruct path: tgt → ... → src
                let mut nodes = Vec::new();
                let mut cur = tgt_snap;
                while cur != u32::MAX {
                    nodes.push(cur);
                    cur = s.prev[cur as usize];
                }
                nodes.reverse();

                // Convert to coordinates, prepend from_lat/lon, append to_lat/lon
                let mut coords = Vec::with_capacity(nodes.len() + 2);
                coords.push((from_lat, from_lon));
                for &nid in &nodes {
                    let n = &self.nodes[nid as usize];
                    coords.push((n.lat, n.lon));
                }
                coords.push((to_lat, to_lon));
                coords
            } else {
                Vec::new()
            };

            s.reset();
            path
        })
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_edges(&self) -> usize {
        self.adj.iter().map(|a| a.len()).sum()
    }

    /// Number of drivable edges (excludes walk-only reverse edges).
    pub fn num_drive_edges(&self) -> usize {
        self.adj
            .iter()
            .flat_map(|a| a.iter())
            .filter(|e| e.drive_weight_ms < u32::MAX)
            .count()
    }
}

/// Haversine distance between two points in meters.
fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().asin()
}
