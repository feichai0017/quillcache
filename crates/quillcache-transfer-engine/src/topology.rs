//! Topology (Mooncake's `topology.h`) — the NIC / GPU affinity matrix that drives
//! topology-aware path selection and multi-NIC striping. On a laptop there is one
//! TCP "device"; this keeps the seam so an RDMA backend can stripe across NICs by
//! a buffer's `location` (which GPU / NUMA node it is near) without changing any
//! caller. Mooncake calls this the `priority_matrix`.

use std::collections::HashMap;

/// PCIe proximity between a GPU and a NIC — the affinity that decides whether
/// GPUDirect RDMA is worth it. The Mooncake #1459 lesson: a NIC far from the GPU
/// makes GDR *slower* than CPU-staged RDMA, so affinity (not "GDR on/off") drives
/// the path. Ordered closest-first: `SameSwitch < SameNuma < CrossNuma`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PcieAffinity {
    /// Under the same PCIe switch as the GPU — ideal for GPUDirect.
    SameSwitch,
    /// Same NUMA node / socket, different switch — GDR still pays off.
    SameNuma,
    /// Across the inter-socket link — GDR commonly loses to CPU-staged.
    CrossNuma,
}

#[derive(Debug, Clone, Default)]
pub struct Topology {
    /// location (e.g. `"cpu:0"`, `"cuda:0"`) → preferred device names, best first.
    matrix: HashMap<String, Vec<String>>,
    /// GPU location → its NICs tagged with PCIe affinity, used to derive the matrix
    /// and the GPUDirect-vs-CPU-staged decision.
    affinity: HashMap<String, Vec<(String, PcieAffinity)>>,
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

    /// All preferred devices for a location, best first — for striping a transfer
    /// across multiple NICs near the buffer (Mooncake's multi-NIC striping).
    pub fn select_devices(&self, location: &str) -> &[String] {
        self.matrix.get(location).map_or(&[], |v| v.as_slice())
    }

    pub fn is_empty(&self) -> bool {
        self.matrix.is_empty()
    }

    /// Record the PCIe `affinity` of `nic` to a GPU `location` (updates if present).
    pub fn set_affinity(
        &mut self,
        location: impl Into<String>,
        nic: impl Into<String>,
        affinity: PcieAffinity,
    ) {
        let entry = self.affinity.entry(location.into()).or_default();
        let nic = nic.into();
        match entry.iter_mut().find(|(n, _)| *n == nic) {
            Some(slot) => slot.1 = affinity,
            None => entry.push((nic, affinity)),
        }
    }

    /// NICs for a GPU `location`, ranked by PCIe affinity (closest first) — the order
    /// to stripe a GPUDirect transfer across.
    pub fn affine_nics(&self, location: &str) -> Vec<&str> {
        let Some(nics) = self.affinity.get(location) else {
            return Vec::new();
        };
        let mut ranked: Vec<&(String, PcieAffinity)> = nics.iter().collect();
        ranked.sort_by_key(|(_, a)| *a);
        ranked.into_iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Whether GPUDirect RDMA is worth it for a GPU `location`: true only if its best
    /// NIC is at least as close as `max_affinity`. Otherwise the caller should
    /// CPU-stage — a far NIC makes GDR slower than staging (Mooncake #1459).
    pub fn prefers_gpudirect(&self, location: &str, max_affinity: PcieAffinity) -> bool {
        self.affinity
            .get(location)
            .and_then(|nics| nics.iter().map(|(_, a)| *a).min())
            .is_some_and(|best| best <= max_affinity)
    }

    /// Populate the priority matrix from recorded affinities (closest NIC first), so
    /// `select_device` / `select_devices` become affinity-aware with no caller change.
    pub fn rebuild_matrix_from_affinity(&mut self) {
        let locations: Vec<String> = self.affinity.keys().cloned().collect();
        for loc in locations {
            let ranked: Vec<String> = self
                .affine_nics(&loc)
                .into_iter()
                .map(String::from)
                .collect();
            if !ranked.is_empty() {
                self.matrix.insert(loc, ranked);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_preferred_nics_by_location() {
        let mut t = Topology::new();
        t.set_preference("cuda:0", vec!["mlx5_0".into(), "mlx5_1".into()]);
        assert_eq!(t.select_device("cuda:0"), Some("mlx5_0"));
        assert_eq!(t.select_devices("cuda:0"), ["mlx5_0", "mlx5_1"]);
        // Unknown location → no device (caller falls back to default routing).
        assert_eq!(t.select_device("cpu:9"), None);
        assert!(t.select_devices("cpu:9").is_empty());
    }

    #[test]
    fn affinity_ranks_nics_and_gates_gpudirect() {
        let mut t = Topology::new();
        // gpu0 has a same-switch NIC and a cross-NUMA NIC; gpu1 only a far one.
        t.set_affinity("cuda:0", "mlx5_0", PcieAffinity::SameSwitch);
        t.set_affinity("cuda:0", "mlx5_3", PcieAffinity::CrossNuma);
        t.set_affinity("cuda:1", "mlx5_3", PcieAffinity::CrossNuma);

        // Ranked closest-first; unknown GPU → empty (caller falls back).
        assert_eq!(t.affine_nics("cuda:0"), ["mlx5_0", "mlx5_3"]);
        assert!(t.affine_nics("cuda:9").is_empty());

        // GDR worth it for gpu0 (best is same-switch), not for gpu1 (best cross-NUMA).
        assert!(t.prefers_gpudirect("cuda:0", PcieAffinity::SameNuma));
        assert!(!t.prefers_gpudirect("cuda:1", PcieAffinity::SameNuma));
        assert!(!t.prefers_gpudirect("cuda:9", PcieAffinity::CrossNuma));

        // Rebuilding the matrix makes the existing selector affinity-aware.
        t.rebuild_matrix_from_affinity();
        assert_eq!(t.select_device("cuda:0"), Some("mlx5_0"));
        assert_eq!(t.select_devices("cuda:0"), ["mlx5_0", "mlx5_3"]);
    }

    #[test]
    fn set_affinity_updates_an_existing_edge() {
        let mut t = Topology::new();
        t.set_affinity("cuda:0", "mlx5_0", PcieAffinity::CrossNuma);
        t.set_affinity("cuda:0", "mlx5_0", PcieAffinity::SameSwitch); // upgrade in place
        assert_eq!(t.affine_nics("cuda:0"), ["mlx5_0"]);
        assert!(t.prefers_gpudirect("cuda:0", PcieAffinity::SameSwitch));
    }
}
