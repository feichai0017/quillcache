use quillcache_core::{
    BlockRemovedEvent, BlockStoredEvent, CacheResidency, CacheTier, DataPlane, EngineEndpoint,
    ExternalKvBlockKey, IdentityScope, IndexBackend, KvBlockKey, KvEvent, KvEventBatch,
    MemoryIndex, NoDataPlane, RequestShape, ReuseViolation, WorkerState,
};
use quillcache_router::{GreedyStatePlaneRouter, RouteDecision, RouterError, RoutingPolicy};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ControlError {
    #[error("unknown engine: {0}")]
    UnknownEngine(String),
    #[error(transparent)]
    Router(#[from] RouterError),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestSummary {
    pub stored_blocks: usize,
    pub removed_blocks: usize,
    pub cleared: bool,
    pub total_resident_blocks: usize,
}

/// What the identity guard did for one request: how many blocks were safe to
/// reuse, and how many content-matching blocks were *refused* because they are
/// resident only under a different identity (a naive content cache would have
/// served them — see `quillcache_core::ReuseViolation`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReuseAudit {
    /// Request blocks whose exact identity is resident — safe to reuse.
    pub safe_reusable: usize,
    /// Request blocks whose content is resident only under a *different*
    /// identity — refused.
    pub refused_unsafe: usize,
    pub refused_cross_tenant: usize,
    pub refused_cross_adapter: usize,
    pub refused_cross_model: usize,
    pub refused_cross_tokenizer: usize,
}

/// Translate a batch of engine KV events into residency updates against any
/// [`IndexBackend`].
///
/// Backend-agnostic on purpose: the same path drives the in-memory index, Holt
/// (ART), RocksDB (LSM), or a filesystem index. Block identity
/// (model/tokenizer/adapter/tenant) is resolved here, once, so every backend
/// sees identity-scoped `put` / `remove_block` / `clear_worker` calls and no
/// backend has to re-implement vLLM/SGLang event parsing.
pub fn ingest_batch(
    backend: &mut dyn IndexBackend,
    batch: KvEventBatch,
    engines: &[EngineEndpoint],
) -> Result<IngestSummary, ControlError> {
    let engine = engines
        .iter()
        .find(|engine| engine.id == batch.engine_id)
        .ok_or_else(|| ControlError::UnknownEngine(batch.engine_id.clone()))?;

    let mut stored_blocks = 0;
    let mut removed_blocks = 0;
    let mut cleared = false;

    for event in batch.events.clone() {
        match event {
            KvEvent::BlockStored(event) => {
                stored_blocks += apply_stored(backend, engine, &batch, event);
            }
            KvEvent::BlockRemoved(event) => {
                removed_blocks += apply_removed(backend, engine, &batch, event);
            }
            KvEvent::AllBlocksCleared => {
                backend.clear_worker(&engine.id);
                cleared = true;
            }
        }
    }

    Ok(IngestSummary {
        stored_blocks,
        removed_blocks,
        cleared,
        total_resident_blocks: backend.len(),
    })
}

fn apply_stored(
    backend: &mut dyn IndexBackend,
    engine: &EngineEndpoint,
    batch: &KvEventBatch,
    event: BlockStoredEvent,
) -> usize {
    let tier = event
        .medium
        .as_deref()
        .and_then(|medium| CacheTier::from_str(medium).ok())
        .unwrap_or(CacheTier::Hbm);
    let block_bytes = event
        .bytes_per_block
        .or(batch.bytes_per_block)
        .unwrap_or(4 * 1024 * 1024);
    let model_id = batch.model_id.as_deref().unwrap_or(&engine.model_id);
    let tokenizer_id = batch
        .tokenizer_id
        .as_deref()
        .unwrap_or(&engine.tokenizer_id);
    let tenant_id = batch.tenant_id.as_deref().unwrap_or(&engine.tenant_id);
    let adapter_id = batch.adapter_id.clone().or(event.lora_name.clone());
    let mut parent = event
        .parent_block_hash
        .clone()
        .unwrap_or_else(|| "root".to_string());

    let mut stored = 0;
    for (idx, block_hash) in event.block_hashes.into_iter().enumerate() {
        let key = KvBlockKey::external_hash(ExternalKvBlockKey {
            model_id: model_id.to_string(),
            tokenizer_id: tokenizer_id.to_string(),
            adapter_id: adapter_id.clone(),
            tenant_id: tenant_id.to_string(),
            prefix_hash: parent.clone(),
            block_hash: block_hash.clone(),
            block_index: idx as u32,
            token_count: event.block_size,
        });
        parent = block_hash;
        backend.put(CacheResidency {
            key,
            worker_id: engine.id.clone(),
            tier,
            bytes: block_bytes,
            last_access_ms: batch.ts_ms.unwrap_or(0),
            ref_count: 0,
            pinned: false,
        });
        stored += 1;
    }

    stored
}

fn apply_removed(
    backend: &mut dyn IndexBackend,
    engine: &EngineEndpoint,
    batch: &KvEventBatch,
    event: BlockRemovedEvent,
) -> usize {
    let scope = IdentityScope {
        model_id: batch
            .model_id
            .clone()
            .unwrap_or_else(|| engine.model_id.clone()),
        tokenizer_id: batch
            .tokenizer_id
            .clone()
            .unwrap_or_else(|| engine.tokenizer_id.clone()),
        adapter_id: batch.adapter_id.clone(),
        tenant_id: batch
            .tenant_id
            .clone()
            .unwrap_or_else(|| engine.tenant_id.clone()),
    };
    let mut removed = 0;
    for block_hash in event.block_hashes {
        removed += backend.remove_block(&scope, &engine.id, &block_hash);
    }
    removed
}

/// The control plane: configured engines, derived worker state, a routing
/// policy, and a pluggable residency index backend.
#[derive(Debug)]
pub struct ControlPlane {
    engines: Vec<EngineEndpoint>,
    workers: Vec<WorkerState>,
    router: Box<dyn RoutingPolicy>,
    residency: Box<dyn IndexBackend>,
    data_plane: Box<dyn DataPlane>,
}

impl ControlPlane {
    /// New control plane backed by the default in-memory index ([`MemoryIndex`]).
    pub fn new(engines: Vec<EngineEndpoint>) -> Self {
        Self::with_index(engines, Box::new(MemoryIndex::new()))
    }

    /// New control plane with a specific index backend (memory, Holt/ART,
    /// RocksDB/LSM, filesystem) and the default cache-aware router.
    pub fn with_index(engines: Vec<EngineEndpoint>, residency: Box<dyn IndexBackend>) -> Self {
        Self::with_index_and_policy(
            engines,
            residency,
            Box::new(GreedyStatePlaneRouter::default()),
        )
    }

    /// New control plane with a specific index backend and routing policy — the
    /// runtime seam for Online mode to pick both by config (e.g. prefix-affinity
    /// vs round-robin across a real engine fleet).
    pub fn with_index_and_policy(
        engines: Vec<EngineEndpoint>,
        residency: Box<dyn IndexBackend>,
        router: Box<dyn RoutingPolicy>,
    ) -> Self {
        let workers = engines
            .iter()
            .map(EngineEndpoint::worker_state)
            .collect::<Vec<_>>();
        Self {
            engines,
            workers,
            router,
            residency,
            data_plane: Box::new(NoDataPlane),
        }
    }

    /// Attach a KV-tensor data plane (LMCache / KVBM / FlexKV adapter). By default
    /// there is none and QuillCache infers residency from routing + KV events;
    /// this is the seam where a real tensor store plugs in under the control plane.
    pub fn with_data_plane(mut self, data_plane: Box<dyn DataPlane>) -> Self {
        self.data_plane = data_plane;
        self
    }

    pub fn data_plane(&self) -> &dyn DataPlane {
        self.data_plane.as_ref()
    }

    pub fn route(&self, request: &RequestShape) -> Result<RouteDecision, ControlError> {
        let residency = self.residency.snapshot();
        self.router
            .route(request, &self.workers, &residency)
            .map_err(ControlError::from)
    }

    /// Record that a request's blocks were placed on `engine_id` — *inferred*
    /// residency from the control plane's own routing decision. This closes the
    /// online loop: the engine runs with prefix caching, so after it serves a
    /// request its prefix blocks are resident there, and the next request for the
    /// same prefix should see a local hit. Without this the index only learns
    /// from `/v1/kv-events`, so cache-aware routing is blind until a bridge is
    /// wired. KV events (Tier 2) later *correct* this inference (e.g. on
    /// eviction); inferred residency is the floor, ground truth the upgrade.
    pub fn observe_placement(&mut self, engine_id: &str, request: &RequestShape, block_bytes: u64) {
        for block in &request.blocks {
            self.residency.put(CacheResidency::hbm(
                engine_id.to_string(),
                block.clone(),
                block_bytes,
            ));
        }
    }

    pub fn ingest(&mut self, batch: KvEventBatch) -> Result<IngestSummary, ControlError> {
        ingest_batch(self.residency.as_mut(), batch, &self.engines)
    }

    /// Audit identity-governed safe reuse for a request against current
    /// residency: which blocks are safe to reuse, and which content-matching
    /// blocks are refused because they belong to another identity. The router
    /// already only reuses on exact identity (`KvBlockKey` equality), so this
    /// never *changes* a decision — it makes the guard's refusals observable, so
    /// the online path can report the unsafe reuse it prevented (the same
    /// property the `safe-reuse` experiment measures offline).
    pub fn audit_reuse(&self, request: &RequestShape) -> ReuseAudit {
        let snapshot = self.residency.snapshot();
        let mut by_content: HashMap<&str, Vec<&KvBlockKey>> = HashMap::new();
        for residency in &snapshot {
            by_content
                .entry(residency.key.block_hash.as_str())
                .or_default()
                .push(&residency.key);
        }

        let mut audit = ReuseAudit::default();
        for block in &request.blocks {
            let Some(resident_keys) = by_content.get(block.block_hash.as_str()) else {
                continue;
            };
            let scope = IdentityScope::from_key(block);
            if resident_keys.iter().any(|key| scope.matches(key)) {
                audit.safe_reusable += 1;
                continue;
            }
            // Content is resident, but only under other identities: refused.
            if let Some(violation) = resident_keys
                .iter()
                .find_map(|key| scope.reuse_violation(key))
            {
                audit.refused_unsafe += 1;
                match violation {
                    ReuseViolation::Tenant => audit.refused_cross_tenant += 1,
                    ReuseViolation::Adapter => audit.refused_cross_adapter += 1,
                    ReuseViolation::Model => audit.refused_cross_model += 1,
                    ReuseViolation::Tokenizer => audit.refused_cross_tokenizer += 1,
                }
            }
        }
        audit
    }

    pub fn engine(&self, id: &str) -> Option<&EngineEndpoint> {
        self.engines.iter().find(|engine| engine.id == id)
    }

    pub fn engines(&self) -> &[EngineEndpoint] {
        &self.engines
    }

    pub fn workers(&self) -> &[WorkerState] {
        &self.workers
    }

    pub fn residency(&self) -> &dyn IndexBackend {
        self.residency.as_ref()
    }

    /// Persist the residency index (checkpoint a persistent backend; no-op for
    /// in-memory). Call periodically and on shutdown so a persistent index
    /// survives a restart.
    pub fn flush(&self) {
        self.residency.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quillcache_core::{EngineKind, SloTarget};

    fn engine() -> EngineEndpoint {
        EngineEndpoint {
            id: "vllm-a".to_string(),
            kind: EngineKind::Vllm,
            base_url: "http://127.0.0.1:8001".to_string(),
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
            tenant_id: "tenant-a".to_string(),
            locality_domain: "local".to_string(),
        }
    }

    #[test]
    fn ingest_stored_block_updates_residency() {
        let mut index = MemoryIndex::new();
        let summary = ingest_batch(
            &mut index,
            KvEventBatch {
                engine_id: "vllm-a".to_string(),
                ts_ms: Some(42),
                model_id: None,
                tokenizer_id: None,
                adapter_id: None,
                tenant_id: None,
                bytes_per_block: Some(1024),
                events: vec![KvEvent::BlockStored(BlockStoredEvent {
                    block_hashes: vec!["h0".to_string()],
                    parent_block_hash: None,
                    token_ids: vec![1, 2, 3],
                    block_size: 3,
                    medium: Some("gpu".to_string()),
                    lora_name: None,
                    group_idx: None,
                    bytes_per_block: None,
                })],
            },
            &[engine()],
        )
        .unwrap();

        assert_eq!(summary.stored_blocks, 1);
        assert_eq!(index.len(), 1);
        assert_eq!(index.snapshot()[0].worker_id, "vllm-a");
    }

    #[test]
    fn control_plane_routes_to_worker_with_resident_block() {
        let mut control = ControlPlane::new(vec![
            engine(),
            EngineEndpoint {
                id: "vllm-b".to_string(),
                base_url: "http://127.0.0.1:8002".to_string(),
                locality_domain: "local".to_string(),
                ..engine()
            },
        ]);
        control
            .ingest(KvEventBatch {
                engine_id: "vllm-b".to_string(),
                ts_ms: None,
                model_id: None,
                tokenizer_id: None,
                adapter_id: None,
                tenant_id: None,
                bytes_per_block: Some(1024),
                events: vec![KvEvent::BlockStored(BlockStoredEvent {
                    block_hashes: vec!["h0".to_string()],
                    parent_block_hash: None,
                    token_ids: vec![1],
                    block_size: 1,
                    medium: Some("gpu".to_string()),
                    lora_name: None,
                    group_idx: None,
                    bytes_per_block: None,
                })],
            })
            .unwrap();

        let request = RequestShape {
            id: "req-1".to_string(),
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
            adapter_id: None,
            tenant_id: "tenant-a".to_string(),
            session_id: None,
            blocks: vec![KvBlockKey::external_hash(ExternalKvBlockKey {
                model_id: "Qwen/Qwen3-0.6B".to_string(),
                tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
                adapter_id: None,
                tenant_id: "tenant-a".to_string(),
                prefix_hash: "root".to_string(),
                block_hash: "h0".to_string(),
                block_index: 0,
                token_count: 1,
            })],
            estimated_decode_tokens: 16,
            slo: SloTarget::default(),
        };

        let decision = control.route(&request).unwrap();
        assert_eq!(decision.worker_id, "vllm-b");
        assert_eq!(decision.local_hits.len(), 1);
    }

    #[test]
    fn observe_placement_closes_the_routing_loop() {
        let mut control = ControlPlane::new(vec![
            engine(),
            EngineEndpoint {
                id: "vllm-b".to_string(),
                base_url: "http://127.0.0.1:8002".to_string(),
                ..engine()
            },
        ]);

        let block = KvBlockKey::external_hash(ExternalKvBlockKey {
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
            adapter_id: None,
            tenant_id: "tenant-a".to_string(),
            prefix_hash: "root".to_string(),
            block_hash: "shared-prefix".to_string(),
            block_index: 0,
            token_count: 64,
        });
        let request = RequestShape {
            id: "req-1".to_string(),
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
            adapter_id: None,
            tenant_id: "tenant-a".to_string(),
            session_id: None,
            blocks: vec![block],
            estimated_decode_tokens: 16,
            slo: SloTarget::default(),
        };

        // Cold index: the first request has no local hit anywhere.
        let first = control.route(&request).unwrap();
        assert_eq!(first.local_hits.len(), 0);

        // The gateway records where it placed the blocks...
        control.observe_placement(&first.worker_id, &request, 4 * 1024 * 1024);

        // ...so a second request for the same prefix now lands on that engine
        // with a real local hit — without any /v1/kv-events traffic.
        let second = control.route(&request).unwrap();
        assert_eq!(second.worker_id, first.worker_id);
        assert_eq!(second.local_hits.len(), 1);
    }

    #[test]
    fn audit_reuse_flags_cross_identity_content_as_refused() {
        let mut control = ControlPlane::new(vec![engine()]);

        // Tenant A places a block with content hash "shared".
        let a_block = KvBlockKey::external_hash(ExternalKvBlockKey {
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
            adapter_id: None,
            tenant_id: "tenant-a".to_string(),
            prefix_hash: "root".to_string(),
            block_hash: "shared".to_string(),
            block_index: 0,
            token_count: 64,
        });
        let a_request = RequestShape {
            id: "a".to_string(),
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
            adapter_id: None,
            tenant_id: "tenant-a".to_string(),
            session_id: None,
            blocks: vec![a_block],
            estimated_decode_tokens: 16,
            slo: SloTarget::default(),
        };
        control.observe_placement("vllm-a", &a_request, 1024);

        // Tenant A re-requesting the same content: a safe reuse.
        let a_audit = control.audit_reuse(&a_request);
        assert_eq!(a_audit.safe_reusable, 1);
        assert_eq!(a_audit.refused_unsafe, 0);

        // Tenant B with the SAME content hash: content is resident only under
        // tenant A, so the guard refuses it as a cross-tenant privacy leak.
        let b_request = RequestShape {
            id: "b".to_string(),
            tenant_id: "tenant-b".to_string(),
            blocks: vec![KvBlockKey::external_hash(ExternalKvBlockKey {
                model_id: "Qwen/Qwen3-0.6B".to_string(),
                tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
                adapter_id: None,
                tenant_id: "tenant-b".to_string(),
                prefix_hash: "root".to_string(),
                block_hash: "shared".to_string(),
                block_index: 0,
                token_count: 64,
            })],
            ..a_request.clone()
        };
        let b_audit = control.audit_reuse(&b_request);
        assert_eq!(b_audit.safe_reusable, 0);
        assert_eq!(b_audit.refused_unsafe, 1);
        assert_eq!(b_audit.refused_cross_tenant, 1);
        assert_eq!(b_audit.refused_cross_adapter, 0);
    }

    #[test]
    fn kv_events_correct_inferred_residency_on_eviction() {
        let mut control = ControlPlane::new(vec![engine()]);
        let block = |hash: &str| {
            KvBlockKey::external_hash(ExternalKvBlockKey {
                model_id: "Qwen/Qwen3-0.6B".to_string(),
                tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
                adapter_id: None,
                tenant_id: "tenant-a".to_string(),
                prefix_hash: "root".to_string(),
                block_hash: hash.to_string(),
                block_index: 0,
                token_count: 64,
            })
        };
        let request = RequestShape {
            id: "r".to_string(),
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
            adapter_id: None,
            tenant_id: "tenant-a".to_string(),
            session_id: None,
            blocks: vec![block("blk-1"), block("blk-2")],
            estimated_decode_tokens: 16,
            slo: SloTarget::default(),
        };

        // Inferred floor: the gateway records the placement after routing.
        control.observe_placement("vllm-a", &request, 1024);
        assert_eq!(control.residency().len(), 2);

        // Precise correction: the engine evicts blk-1 and emits a KV event.
        // Inferred residency would stay stale; the event corrects it.
        control
            .ingest(KvEventBatch {
                engine_id: "vllm-a".to_string(),
                ts_ms: None,
                model_id: None,
                tokenizer_id: None,
                adapter_id: None,
                tenant_id: None,
                bytes_per_block: None,
                events: vec![KvEvent::BlockRemoved(BlockRemovedEvent {
                    block_hashes: vec!["blk-1".to_string()],
                    medium: None,
                    group_idx: None,
                })],
            })
            .unwrap();

        assert_eq!(control.residency().len(), 1);
        assert_eq!(control.residency().snapshot()[0].key.block_hash, "blk-2");

        // A re-request now sees only blk-2 as a hit; blk-1 correctly recomputes.
        let decision = control.route(&request).unwrap();
        assert_eq!(decision.local_hits.len(), 1);
        assert_eq!(decision.recomputes.len(), 1);
    }

    #[test]
    fn control_plane_accepts_a_custom_index_backend() {
        // The runtime seam: any IndexBackend can back the control plane.
        let control = ControlPlane::with_index(vec![engine()], Box::new(MemoryIndex::new()));
        assert_eq!(control.residency().name(), "memory");
        assert!(!control.residency().persistent());
    }

    #[test]
    fn control_plane_data_plane_defaults_to_none_and_is_pluggable() {
        use quillcache_core::MockDataPlane;
        let control = ControlPlane::new(vec![engine()]);
        // By default there is no tensor data plane (infer from events).
        assert_eq!(control.data_plane().name(), "none");
        // A data plane (LMCache/KVBM/FlexKV adapter) plugs in at this seam.
        let control = control.with_data_plane(Box::new(MockDataPlane::new()));
        assert_eq!(control.data_plane().name(), "mock");
    }
}
