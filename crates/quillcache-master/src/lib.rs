//! The master metadata service for the distributed KV pool.
//!
//! Promotes the residency index out of any single gateway process into a shared
//! service, so node A can discover that node B holds a block. Two pieces of
//! state, matching the master/etcd split discussed in the docs:
//!
//! - a **shared residency index** — which node (and tier) holds each block. A
//!   pluggable [`IndexBackend`] (Memory now; Holt-ART for a persistent,
//!   fast-recovering master; this is the large, high-churn, rebuildable state).
//! - a **node registry** — node id → transfer-engine address. Small, low-churn
//!   coordination state (the etcd analogue; an etcd-backed registry plugs in
//!   later without touching the read path).
//!
//! Gateways `register` themselves, report `placed` blocks, and `locate` blocks
//! here; the cross-node read path is then Conductor → `locate` → registry addr →
//! transfer engine (see `quillcache_store::PooledStore` / `EngineConnector`).

use quillcache_core::{CacheResidency, IndexBackend, KvBlockKey, MemoryIndex};
use std::collections::HashMap;

/// The master: a shared residency index plus a node registry.
#[derive(Debug)]
pub struct Master {
    index: Box<dyn IndexBackend>,
    nodes: HashMap<String, String>,
}

impl Master {
    /// A master backed by the in-memory reference index.
    pub fn new() -> Self {
        Self {
            index: Box::new(MemoryIndex::new()),
            nodes: HashMap::new(),
        }
    }

    /// A master backed by a chosen index (e.g. Holt-ART for persistence + fast
    /// recovery). The index is soft state — rebuildable from node block reports —
    /// so persistence buys fast restart, not correctness.
    pub fn with_index(index: Box<dyn IndexBackend>) -> Self {
        Self {
            index,
            nodes: HashMap::new(),
        }
    }

    /// A node joins the pool, announcing its transfer-engine address.
    pub fn register(&mut self, node_id: impl Into<String>, transfer_addr: impl Into<String>) {
        self.nodes.insert(node_id.into(), transfer_addr.into());
    }

    /// A node leaves the pool; its residency is dropped (peers stop targeting it).
    pub fn deregister(&mut self, node_id: &str) {
        self.nodes.remove(node_id);
        self.index.clear_worker(node_id);
    }

    /// The transfer-engine address for a node, if registered.
    pub fn node_addr(&self, node_id: &str) -> Option<&str> {
        self.nodes.get(node_id).map(String::as_str)
    }

    /// The full node id → address map (what a gateway loads into its registry).
    pub fn nodes(&self) -> &HashMap<String, String> {
        &self.nodes
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// A node reports it placed (offloaded) a block into its pool.
    pub fn placed(&mut self, residency: CacheResidency) {
        self.index.put(residency);
    }

    /// A node reports a batch of placements (a block report).
    pub fn placed_batch(&mut self, residencies: impl IntoIterator<Item = CacheResidency>) {
        for residency in residencies {
            self.index.put(residency);
        }
    }

    /// A node reports a block was evicted from its pool.
    pub fn evicted(&mut self, node_id: &str, key: &KvBlockKey) {
        let scope = quillcache_core::IdentityScope::from_key(key);
        self.index.remove_block(&scope, node_id, &key.block_hash);
    }

    /// Every residency for a block — which nodes / tiers hold it.
    pub fn locate(&self, key: &KvBlockKey) -> Vec<CacheResidency> {
        self.index.locate(key)
    }

    /// The node ids that hold a block, for the cross-node fetch path.
    pub fn locate_nodes(&self, key: &KvBlockKey) -> Vec<String> {
        let mut nodes: Vec<String> = self
            .index
            .locate(key)
            .into_iter()
            .map(|residency| residency.worker_id)
            .collect();
        nodes.dedup();
        nodes
    }

    /// Number of residency records held (for reports / metrics).
    pub fn resident_blocks(&self) -> usize {
        self.index.len()
    }
}

impl Default for Master {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quillcache_core::{CacheResidency, KvBlockKey};

    fn key(tenant: &str, hash: &str) -> KvBlockKey {
        KvBlockKey::new("m", "t", tenant, "p", hash, 0, 64)
    }

    #[test]
    fn register_locate_and_evict() {
        let mut master = Master::new();
        master.register("node-a", "127.0.0.1:7001");
        master.register("node-b", "127.0.0.1:7002");
        assert_eq!(master.node_count(), 2);
        assert_eq!(master.node_addr("node-b"), Some("127.0.0.1:7002"));

        let k = key("ten-a", "blk-1");
        master.placed(CacheResidency {
            key: k.clone(),
            worker_id: "node-b".into(),
            tier: quillcache_core::CacheTier::CpuDram,
            bytes: 16,
            last_access_ms: 0,
            ref_count: 0,
            pinned: false,
        });
        assert_eq!(master.locate_nodes(&k), vec!["node-b".to_string()]);
        assert_eq!(master.resident_blocks(), 1);

        master.evicted("node-b", &k);
        assert!(master.locate_nodes(&k).is_empty());

        // Deregistering a node drops its residency so peers stop targeting it.
        master.placed(CacheResidency {
            key: k.clone(),
            worker_id: "node-b".into(),
            tier: quillcache_core::CacheTier::CpuDram,
            bytes: 16,
            last_access_ms: 0,
            ref_count: 0,
            pinned: false,
        });
        master.deregister("node-b");
        assert_eq!(master.node_count(), 1);
        assert!(master.locate_nodes(&k).is_empty());
    }
}
