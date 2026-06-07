use quillcache_core::{
    BlockRemovedEvent, BlockStoredEvent, CacheResidency, CacheTier, EngineEndpoint,
    ExternalKvBlockKey, IdentityScope, IndexBackend, KvBlockKey, KvEvent, KvEventBatch,
    MemoryIndex, RequestShape, WorkerState,
};
use quillcache_router::{GreedyStatePlaneRouter, RouteDecision, RouterError};
use serde::{Deserialize, Serialize};
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
    router: GreedyStatePlaneRouter,
    residency: Box<dyn IndexBackend>,
}

impl ControlPlane {
    /// New control plane backed by the default in-memory index ([`MemoryIndex`]).
    pub fn new(engines: Vec<EngineEndpoint>) -> Self {
        Self::with_index(engines, Box::new(MemoryIndex::new()))
    }

    /// New control plane with a specific index backend (memory, Holt/ART,
    /// RocksDB/LSM, filesystem). This is the runtime seam for Online mode to
    /// pick a backend by config and for Experiment mode to compare them.
    pub fn with_index(engines: Vec<EngineEndpoint>, residency: Box<dyn IndexBackend>) -> Self {
        let workers = engines
            .iter()
            .map(EngineEndpoint::worker_state)
            .collect::<Vec<_>>();
        Self {
            engines,
            workers,
            router: GreedyStatePlaneRouter::default(),
            residency,
        }
    }

    pub fn route(&self, request: &RequestShape) -> Result<RouteDecision, ControlError> {
        let residency = self.residency.snapshot();
        self.router
            .route(request, &self.workers, &residency)
            .map_err(ControlError::from)
    }

    pub fn ingest(&mut self, batch: KvEventBatch) -> Result<IngestSummary, ControlError> {
        ingest_batch(self.residency.as_mut(), batch, &self.engines)
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
    fn control_plane_accepts_a_custom_index_backend() {
        // The runtime seam: any IndexBackend can back the control plane.
        let control = ControlPlane::with_index(vec![engine()], Box::new(MemoryIndex::new()));
        assert_eq!(control.residency().name(), "memory");
        assert!(!control.residency().persistent());
    }
}
