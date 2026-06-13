//! Conductor — QuillCache's cache-aware routing brain, a faithful port of
//! Mooncake's **OSS Conductor** (the KV-cache indexer for cache-aware routers).
//! The FAST'25 paper's full prefill/decode scheduler is proprietary; the open
//! piece — "for this request's prefix, which instance has the best KV-cache
//! locality?" — is what this module mirrors.
//!
//! Component mapping (Mooncake Conductor → here):
//!
//! | Mooncake | here |
//! | --- | --- |
//! | `ModelContext{tenant, model, lora, block_size, salt, …}` | [`ModelContext`] |
//! | `PrefixCacheTable` (global prefix → instances) | [`PrefixCacheTable`] |
//! | `KVEventHandler` (normalize `BlockStored`/`BlockRemoved`) | [`KVEventHandler`] |
//! | the cache-aware router (cost over overlap) | [`crate::router`] (`DynamoCostRouter`) |
//!
//! The walk: a request's prefix is a chain of complete-block hashes
//! `[p0, p1, …]`; [`PrefixCacheTable::query_overlap`] returns, per instance, the
//! longest **contiguous** leading run it has cached — the overlap the cost
//! function credits. The table is keyed by [`ModelContext`], so the identity
//! guard holds at the routing layer too: a request never gets prefix credit for a
//! different tenant / model / adapter's cache.

use crate::router::DynamoCostRouter;
use crate::{IdentityScope, WorkerState};
use std::collections::{HashMap, HashSet};

/// A serving instance (engine / worker) id.
pub type InstanceId = String;
/// A cumulative hash of a prefix's first `i+1` complete blocks (Mooncake's
/// complete-block prefix hash).
pub type PrefixHash = String;

/// The scope a prefix-cache entry lives in (Mooncake's `ModelContext`). Cached
/// prefix blocks are shared only within the same context — QuillCache's
/// [`IdentityScope`] plus the block size and an optional cache-busting `salt`,
/// made the table's key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelContext {
    pub tenant_id: String,
    pub model_id: String,
    pub tokenizer_id: String,
    pub lora_id: Option<String>,
    pub block_size: u32,
    pub salt: Option<String>,
}

impl ModelContext {
    /// Build from an [`IdentityScope`] (the LoRA adapter is Mooncake's `lora`).
    pub fn from_scope(scope: &IdentityScope, block_size: u32) -> Self {
        Self {
            tenant_id: scope.tenant_id.clone(),
            model_id: scope.model_id.clone(),
            tokenizer_id: scope.tokenizer_id.clone(),
            lora_id: scope.adapter_id.clone(),
            block_size,
            salt: None,
        }
    }
}

/// Global prefix → instances index (Mooncake's `PrefixCacheTable`). Per
/// [`ModelContext`], maps each cumulative-block-prefix hash to the instances that
/// have it cached.
#[derive(Debug, Default)]
pub struct PrefixCacheTable {
    tables: HashMap<ModelContext, HashMap<PrefixHash, HashSet<InstanceId>>>,
}

impl PrefixCacheTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `instance` has cached the prefix up to each of `prefix_hashes`.
    pub fn insert(
        &mut self,
        ctx: &ModelContext,
        prefix_hashes: &[PrefixHash],
        instance: &InstanceId,
    ) {
        let table = self.tables.entry(ctx.clone()).or_default();
        for ph in prefix_hashes {
            table
                .entry(ph.clone())
                .or_default()
                .insert(instance.clone());
        }
    }

    /// A block (at one prefix position) was evicted on `instance`.
    pub fn remove(&mut self, ctx: &ModelContext, prefix_hash: &PrefixHash, instance: &InstanceId) {
        if let Some(table) = self.tables.get_mut(ctx) {
            if let Some(set) = table.get_mut(prefix_hash) {
                set.remove(instance);
                if set.is_empty() {
                    table.remove(prefix_hash);
                }
            }
        }
    }

    /// Drop everything an instance had cached (it went away).
    pub fn remove_instance(&mut self, instance: &InstanceId) {
        for table in self.tables.values_mut() {
            for set in table.values_mut() {
                set.remove(instance);
            }
            table.retain(|_, set| !set.is_empty());
        }
        self.tables.retain(|_, table| !table.is_empty());
    }

    /// For a request's cumulative prefix hashes, the longest **contiguous** cached
    /// prefix length per instance (Mooncake's walk-until-first-miss) — the overlap
    /// the cost function credits.
    pub fn query_overlap(
        &self,
        ctx: &ModelContext,
        prefix_hashes: &[PrefixHash],
    ) -> HashMap<InstanceId, usize> {
        let mut result = HashMap::new();
        let Some(table) = self.tables.get(ctx) else {
            return result;
        };
        let Some(first) = prefix_hashes.first().and_then(|p| table.get(p)) else {
            return result;
        };
        for instance in first {
            let mut len = 1;
            while len < prefix_hashes.len()
                && table
                    .get(&prefix_hashes[len])
                    .is_some_and(|set| set.contains(instance))
            {
                len += 1;
            }
            result.insert(instance.clone(), len);
        }
        result
    }

    pub fn context_count(&self) -> usize {
        self.tables.len()
    }
}

/// A normalized KV-cache event from an engine (Mooncake's `KVEventHandler` input;
/// vLLM emits `BlockStored` / `BlockRemoved` over ZMQ msgpack — normalized to this
/// vendor-neutral shape by the events bridge).
#[derive(Debug, Clone)]
pub enum KvCacheEvent {
    BlockStored {
        ctx: ModelContext,
        prefix_hashes: Vec<PrefixHash>,
        instance: InstanceId,
    },
    BlockRemoved {
        ctx: ModelContext,
        prefix_hash: PrefixHash,
        instance: InstanceId,
    },
    InstanceGone {
        instance: InstanceId,
    },
}

/// Applies normalized KV events to a [`PrefixCacheTable`] (Mooncake's
/// `KVEventHandler`).
#[derive(Debug, Default)]
pub struct KVEventHandler {
    pub table: PrefixCacheTable,
}

impl KVEventHandler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn handle(&mut self, event: KvCacheEvent) {
        match event {
            KvCacheEvent::BlockStored {
                ctx,
                prefix_hashes,
                instance,
            } => self.table.insert(&ctx, &prefix_hashes, &instance),
            KvCacheEvent::BlockRemoved {
                ctx,
                prefix_hash,
                instance,
            } => self.table.remove(&ctx, &prefix_hash, &instance),
            KvCacheEvent::InstanceGone { instance } => self.table.remove_instance(&instance),
        }
    }
}

/// The cache-aware routing **Conductor**: a [`PrefixCacheTable`] (fed by KV events
/// via the [`KVEventHandler`]) queried by the Dynamo cost function to pick the
/// instance with the best KV-cache locality, traded off against load. This is
/// Mooncake's `Conductor` job — "route this request where its prefix is already
/// cached, unless that instance is too busy".
#[derive(Debug, Default)]
pub struct Conductor {
    events: KVEventHandler,
    router: DynamoCostRouter,
}

impl Conductor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply an engine's KV-cache event to the prefix table.
    pub fn observe(&mut self, event: KvCacheEvent) {
        self.events.handle(event);
    }

    pub fn table(&self) -> &PrefixCacheTable {
        &self.events.table
    }

    /// Pick the lowest-cost instance for a request: the Dynamo cost over each
    /// instance's contiguous prefix overlap (from the table) plus its load.
    pub fn route(
        &self,
        ctx: &ModelContext,
        prefix_hashes: &[PrefixHash],
        workers: &[WorkerState],
    ) -> Option<InstanceId> {
        if workers.is_empty() {
            return None;
        }
        let overlap = self.events.table.query_overlap(ctx, prefix_hashes);
        let prompt_blocks = prefix_hashes.len();
        let cost = |w: &WorkerState| {
            self.router.cost_with_overlap(
                prompt_blocks,
                w.queued_prefill_tokens,
                w.running_decodes,
                overlap.get(&w.id).copied().unwrap_or(0),
            )
        };
        workers
            .iter()
            .min_by(|a, b| cost(a).total_cmp(&cost(b)))
            .map(|w| w.id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(tenant: &str) -> ModelContext {
        ModelContext {
            tenant_id: tenant.into(),
            model_id: "m".into(),
            tokenizer_id: "t".into(),
            lora_id: None,
            block_size: 16,
            salt: None,
        }
    }
    fn ph(s: &str) -> PrefixHash {
        s.into()
    }

    #[test]
    fn longest_contiguous_prefix_overlap_per_instance() {
        let mut table = PrefixCacheTable::new();
        let c = ctx("ten-a");
        // gpu-0 cached the first 3 prefix blocks; gpu-1 only the first.
        table.insert(&c, &[ph("p0"), ph("p1"), ph("p2")], &"gpu-0".into());
        table.insert(&c, &[ph("p0")], &"gpu-1".into());
        // A request whose prefix is p0,p1,p2,p3.
        let overlap = table.query_overlap(&c, &[ph("p0"), ph("p1"), ph("p2"), ph("p3")]);
        assert_eq!(overlap["gpu-0"], 3);
        assert_eq!(overlap["gpu-1"], 1);
    }

    #[test]
    fn overlap_is_model_context_scoped() {
        // The identity guard at the routing layer: tenant-b gets no credit for
        // tenant-a's cached prefix even though the block hashes are identical.
        let mut table = PrefixCacheTable::new();
        table.insert(&ctx("ten-a"), &[ph("p0"), ph("p1")], &"gpu-0".into());
        assert!(table
            .query_overlap(&ctx("ten-b"), &[ph("p0"), ph("p1")])
            .is_empty());
    }

    #[test]
    fn removing_a_block_shortens_the_contiguous_overlap() {
        let mut table = PrefixCacheTable::new();
        let c = ctx("ten-a");
        table.insert(&c, &[ph("p0"), ph("p1"), ph("p2")], &"gpu-0".into());
        table.remove(&c, &ph("p2"), &"gpu-0".into());
        // p2 gone → the contiguous match is only p0,p1.
        let overlap = table.query_overlap(&c, &[ph("p0"), ph("p1"), ph("p2")]);
        assert_eq!(overlap["gpu-0"], 2);
    }

    #[test]
    fn kv_event_handler_applies_stored_and_instance_gone() {
        let mut handler = KVEventHandler::new();
        let c = ctx("ten-a");
        handler.handle(KvCacheEvent::BlockStored {
            ctx: c.clone(),
            prefix_hashes: vec![ph("p0"), ph("p1")],
            instance: "gpu-0".into(),
        });
        assert_eq!(
            handler.table.query_overlap(&c, &[ph("p0"), ph("p1")])["gpu-0"],
            2
        );
        handler.handle(KvCacheEvent::InstanceGone {
            instance: "gpu-0".into(),
        });
        assert!(handler
            .table
            .query_overlap(&c, &[ph("p0"), ph("p1")])
            .is_empty());
    }

    #[test]
    fn conductor_routes_to_the_instance_with_the_best_cached_prefix() {
        let mut conductor = Conductor::new();
        let c = ctx("ten-a");
        conductor.observe(KvCacheEvent::BlockStored {
            ctx: c.clone(),
            prefix_hashes: vec![ph("p0"), ph("p1"), ph("p2")],
            instance: "gpu-0".into(),
        });
        let workers = vec![
            WorkerState::new("gpu-0", "dc"),
            WorkerState::new("gpu-1", "dc"),
        ];
        let prefix = vec![ph("p0"), ph("p1"), ph("p2")];
        // gpu-0 has the whole prefix cached → it wins on locality.
        assert_eq!(
            conductor.route(&c, &prefix, &workers).as_deref(),
            Some("gpu-0")
        );
    }

    #[test]
    fn heavy_decode_load_spills_off_the_cache_hot_instance() {
        let mut conductor = Conductor::new();
        let c = ctx("ten-a");
        conductor.observe(KvCacheEvent::BlockStored {
            ctx: c.clone(),
            prefix_hashes: vec![ph("p0"), ph("p1")],
            instance: "gpu-0".into(),
        });
        let prefix = vec![ph("p0"), ph("p1")];
        // gpu-0 has the cache but is swamped (100 running decodes); gpu-1 is idle.
        // cost(gpu-0) = max(0, 2−2) + 100 = 100; cost(gpu-1) = max(0, 2−0) + 0 = 2.
        let workers = vec![
            WorkerState::new("gpu-0", "dc").with_load(0, 100),
            WorkerState::new("gpu-1", "dc"),
        ];
        assert_eq!(
            conductor.route(&c, &prefix, &workers).as_deref(),
            Some("gpu-1")
        );
    }
}
