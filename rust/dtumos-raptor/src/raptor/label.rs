/// A label represents one way to reach a stop.
/// Stored in a `LabelArena`; referenced by u32 index.
#[derive(Debug, Clone, Copy)]
pub struct Label {
    /// Arrival time at stop (seconds since midnight)
    pub arrival_time: u32,
    /// Number of transfers taken so far (0 = direct)
    pub num_transfers: u8,
    /// Generalized cost (centi-seconds)
    pub generalized_cost: u32,
    /// Previous stop index (for path reconstruction), u32::MAX if origin
    pub prev_stop: u32,
    /// Route index used to reach this stop (u32::MAX if walk)
    pub route_index: u32,
    /// Trip index within the route's timetable
    pub trip_index: u32,
    /// Board stop position within the trip pattern
    pub board_stop_pos: u32,
    /// Alight stop position within the trip pattern
    pub alight_stop_pos: u32,
    /// Parent label index in the arena (u32::MAX = none)
    pub prev_label: u32,
}

impl Label {
    pub const NONE: u32 = u32::MAX;

    pub fn walk_access(arrival_time: u32, walk_cost: u32) -> Self {
        Self {
            arrival_time,
            num_transfers: 0,
            generalized_cost: walk_cost,
            prev_stop: Self::NONE,
            route_index: Self::NONE,
            trip_index: Self::NONE,
            board_stop_pos: Self::NONE,
            alight_stop_pos: Self::NONE,
            prev_label: Self::NONE,
        }
    }
}

/// Flat arena for Label storage. All labels created during one RAPTOR call
/// live here, referenced by u32 index. Eliminates per-label heap allocation
/// and Arc atomic reference counting.
pub struct LabelArena {
    labels: Vec<Label>,
}

impl LabelArena {
    pub fn new() -> Self {
        Self {
            labels: Vec::with_capacity(4096),
        }
    }

    /// Push a label and return its index.
    #[inline]
    pub fn push(&mut self, label: Label) -> u32 {
        let idx = self.labels.len() as u32;
        self.labels.push(label);
        idx
    }

    /// Get a label by index.
    #[inline]
    pub fn get(&self, idx: u32) -> &Label {
        debug_assert!(
            (idx as usize) < self.labels.len(),
            "Label index {} out of bounds (len={})", idx, self.labels.len()
        );
        &self.labels[idx as usize]
    }
}

/// Maximum labels per stop bag — prevents label explosion on large networks.
const MAX_BAG_SIZE: usize = 5;

/// A Pareto bag stores non-dominated label indices at a stop.
#[derive(Debug, Clone, Default)]
pub struct ParetoBag {
    pub labels: Vec<u32>,
}

impl ParetoBag {
    pub fn new() -> Self {
        Self { labels: Vec::new() }
    }

    /// Try to add a label index. Returns true if added (not dominated).
    pub fn add(&mut self, new_idx: u32, arena: &LabelArena) -> bool {
        let new_label = arena.get(new_idx);

        // Check if new label is dominated by any existing label
        for &existing_idx in &self.labels {
            let existing = arena.get(existing_idx);
            if dominates(existing, new_label) {
                return false;
            }
        }

        // Remove labels dominated by the new one
        let nl = *new_label; // copy to avoid borrow conflict
        self.labels
            .retain(|&existing_idx| !dominates(&nl, arena.get(existing_idx)));

        // Cap bag size: if full, only add if better than worst
        if self.labels.len() >= MAX_BAG_SIZE {
            if let Some(worst_pos) = self
                .labels
                .iter()
                .enumerate()
                .max_by_key(|(_, &idx)| arena.get(idx).generalized_cost)
                .map(|(i, _)| i)
            {
                if nl.generalized_cost < arena.get(self.labels[worst_pos]).generalized_cost {
                    self.labels.swap_remove(worst_pos);
                } else {
                    return false;
                }
            }
        }

        self.labels.push(new_idx);
        true
    }

    /// Best arrival time in the bag.
    pub fn best_arrival_time(&self, arena: &LabelArena) -> u32 {
        self.labels
            .iter()
            .map(|&idx| arena.get(idx).arrival_time)
            .min()
            .unwrap_or(u32::MAX)
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Clear all labels without deallocating the underlying Vec.
    #[inline]
    pub fn clear(&mut self) {
        self.labels.clear();
    }
}

/// Label a dominates label b iff a is <= in all criteria and < in at least one.
#[inline]
fn dominates(a: &Label, b: &Label) -> bool {
    a.arrival_time <= b.arrival_time
        && a.num_transfers <= b.num_transfers
        && a.generalized_cost <= b.generalized_cost
        && (a.arrival_time < b.arrival_time
            || a.num_transfers < b.num_transfers
            || a.generalized_cost < b.generalized_cost)
}
