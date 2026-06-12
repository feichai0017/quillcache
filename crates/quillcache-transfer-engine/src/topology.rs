//! Topology (Mooncake's `topology.h`) — the NIC / GPU affinity matrix that drives
//! topology-aware path selection and multi-NIC striping. On a laptop there is one
//! TCP "device"; this keeps the seam so an RDMA backend can stripe across NICs by
//! a buffer's `location` (which GPU / NUMA node it is near) without changing any
//! caller. Mooncake calls this the `priority_matrix`.

use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct Topology {
    /// location (e.g. `"cpu:0"`, `"cuda:0"`) → preferred device names, best first.
    matrix: HashMap<String, Vec<String>>,
}

impl Topology {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_preference(&mut self, location: impl Into<String>, devices: Vec<String>) {
        self.matrix.insert(location.into(), devices);
    }

    /// The preferred device for a buffer's location (first in its priority list).
    pub fn select_device(&self, location: &str) -> Option<&str> {
        self.matrix
            .get(location)
            .and_then(|devices| devices.first())
            .map(|s| s.as_str())
    }

    pub fn is_empty(&self) -> bool {
        self.matrix.is_empty()
    }
}
