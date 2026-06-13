use crate::replication::{
    plan_replications, HotnessTracker, PrefixResidency, ReplicationAction, ReplicationConfig,
    WorkerLoad,
};
use crate::router::{
    plan_for_worker, GreedyStatePlaneRouter, RouteDecision, RouterError, RoutingPolicy,
};
use crate::{
    BlockRemovedEvent, BlockStoredEvent, CacheResidency, CacheTier, Conductor, CostModel,
    DataPlane, DataPlaneAction, DataPlaneUpdate, EngineEndpoint, EngineRole, ExternalKvBlockKey,
    IdentityScope, IndexBackend, KvBlockKey, KvCacheEvent, KvEvent, KvEventBatch, MemoryIndex,
    ModelContext, NoDataPlane, RequestShape, ReuseViolation, WorkerState,
};
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
/// served them — see `crate::ReuseViolation`).
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServingMode {
    Aggregated,
    Disaggregated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanActionKind {
    UseLocal,
    Fetch,
    Recompute,
    RunPrefill,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanAction {
    pub kind: PlanActionKind,
    pub worker_id: String,
    pub source_worker_id: Option<String>,
    pub key: Option<KvBlockKey>,
    pub tier: Option<CacheTier>,
    pub estimated_us: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestPlan {
    pub mode: ServingMode,
    pub execution_worker_id: String,
    pub prefill_worker_id: Option<String>,
    pub decode_worker_id: String,
    pub route: RouteDecision,
    pub actions: Vec<PlanAction>,
}

/// The control plane's admission decision (Mooncake's overload-oriented early
/// rejection): admit with a plan, or reject when the SLO can't be met.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdmissionDecision {
    Admit(Box<RequestPlan>),
    Reject {
        reason: String,
        best_slo_violation_us: u64,
    },
}

/// The prefill → decode KV handoff a disaggregated plan implies (Mooncake's P/D
/// data path): the freshly-prefilled blocks the `prefill_worker` computes and
/// publishes to the store, which the `decode_worker` then reads over the
/// Transfer Engine before continuing generation. `None` in aggregated mode (the
/// same worker prefills and decodes, so no KV crosses the wire).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvHandoff {
    pub request_id: String,
    pub prefill_worker_id: String,
    pub decode_worker_id: String,
    pub blocks: Vec<KvBlockKey>,
}

impl RequestPlan {
    /// The disaggregated prefill → decode KV handoff this plan implies, or `None`
    /// when serving aggregated. The handoff blocks are the plan's `RunPrefill`
    /// actions — the KV prefill computes that decode must receive.
    pub fn kv_handoff(&self) -> Option<KvHandoff> {
        if self.mode != ServingMode::Disaggregated {
            return None;
        }
        let prefill_worker_id = self.prefill_worker_id.clone()?;
        let blocks = self
            .actions
            .iter()
            .filter(|action| action.kind == PlanActionKind::RunPrefill)
            .filter_map(|action| action.key.clone())
            .collect();
        Some(KvHandoff {
            request_id: self.route.request_id.clone(),
            prefill_worker_id,
            decode_worker_id: self.decode_worker_id.clone(),
            blocks,
        })
    }
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
    /// The Mooncake Conductor's prefix-cache index, fed by the same KV events +
    /// inferred placement. When `use_conductor` is on, the worker pick comes from
    /// [`Conductor::route`] (contiguous-prefix overlap → Dynamo cost) instead of
    /// the residency-snapshot router; the per-block plan still uses residency.
    conductor: Conductor,
    use_conductor: bool,
    cost_model: CostModel,
    /// Overload admission: reject a request if the best worker would still
    /// violate its SLO by more than this many microseconds. `None` admits all
    /// (Mooncake's overload-oriented early rejection).
    max_slo_violation_us: Option<u64>,
    /// Per-(identity, prefix) access counts, so [`Self::replication_plan`] can spot
    /// hot prefixes worth replicating to spread a cache hotspot's load.
    hotness: HotnessTracker,
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
            conductor: Conductor::new(),
            use_conductor: false,
            cost_model: CostModel::default(),
            max_slo_violation_us: None,
            hotness: HotnessTracker::default(),
        }
    }

    /// Route via the Mooncake Conductor (the prefix-cache table + the Dynamo cost
    /// function) instead of the residency-snapshot router. Off by default (the
    /// residency router is unchanged); the gateway turns it on by config
    /// (`conductor: true`).
    pub fn with_conductor_routing(mut self, on: bool) -> Self {
        self.use_conductor = on;
        self
    }

    /// Enable overload admission: reject a request when even the best worker
    /// would violate its SLO by more than `max_violation_us` (Mooncake's
    /// overload-oriented early rejection). Off by default (admit all).
    pub fn with_admission_slo_limit(mut self, max_violation_us: u64) -> Self {
        self.max_slo_violation_us = Some(max_violation_us);
        self
    }

    /// Plan the request, then either admit it or **reject it early** if even the
    /// best worker would violate the SLO past the configured limit (overload).
    /// With no limit set, always admits.
    pub fn admit(&self, request: &RequestShape) -> Result<AdmissionDecision, ControlError> {
        let plan = self.plan(request)?;
        if let Some(limit) = self.max_slo_violation_us {
            if plan.route.slo_violation_us > limit {
                return Ok(AdmissionDecision::Reject {
                    reason: format!(
                        "overloaded: best worker '{}' would violate SLO by {}us (limit {}us)",
                        plan.route.worker_id, plan.route.slo_violation_us, limit
                    ),
                    best_slo_violation_us: plan.route.slo_violation_us,
                });
            }
        }
        Ok(AdmissionDecision::Admit(Box::new(plan)))
    }

    /// The request's `ModelContext` + its prefix-inclusive block hashes — the
    /// Conductor's query key.
    fn request_prefix(request: &RequestShape) -> (ModelContext, Vec<String>) {
        let scope = IdentityScope {
            model_id: request.model_id.clone(),
            tokenizer_id: request.tokenizer_id.clone(),
            adapter_id: request.adapter_id.clone(),
            tenant_id: request.tenant_id.clone(),
        };
        let block_size = request.blocks.first().map_or(0, |block| block.token_count);
        let prefix_hashes = request
            .blocks
            .iter()
            .map(|block| block.block_hash.clone())
            .collect();
        (ModelContext::from_scope(&scope, block_size), prefix_hashes)
    }

    /// Pick a route: via the Conductor when enabled (its worker pick + the
    /// residency-based per-block plan), else the residency-snapshot router.
    fn decide(
        &self,
        request: &RequestShape,
        route_workers: &[WorkerState],
        residency: &[CacheResidency],
    ) -> Result<RouteDecision, ControlError> {
        if self.use_conductor {
            let (ctx, prefix_hashes) = Self::request_prefix(request);
            if let Some(worker_id) = self.conductor.route(&ctx, &prefix_hashes, route_workers) {
                if let Some(target) = route_workers.iter().find(|w| w.id == worker_id) {
                    let worker_by_id: HashMap<&str, &WorkerState> =
                        route_workers.iter().map(|w| (w.id.as_str(), w)).collect();
                    return Ok(plan_for_worker(
                        &self.cost_model,
                        request,
                        target,
                        &worker_by_id,
                        residency,
                    ));
                }
            }
        }
        self.router
            .route(request, route_workers, residency)
            .map_err(ControlError::from)
    }

    /// Feed the Conductor's prefix table from a KV-event batch (BlockStored +
    /// AllBlocksCleared). Precise per-block removal from the Conductor is a
    /// refinement — the residency index already handles precise eviction for the
    /// per-block plan + the reuse audit; the prefix table refreshes on re-store /
    /// clear.
    fn feed_conductor(&mut self, batch: &KvEventBatch) {
        let Some(engine) = self
            .engines
            .iter()
            .find(|engine| engine.id == batch.engine_id)
            .cloned()
        else {
            return;
        };
        for event in &batch.events {
            match event {
                KvEvent::BlockStored(stored) => {
                    let scope = IdentityScope {
                        model_id: batch
                            .model_id
                            .clone()
                            .unwrap_or_else(|| engine.model_id.clone()),
                        tokenizer_id: batch
                            .tokenizer_id
                            .clone()
                            .unwrap_or_else(|| engine.tokenizer_id.clone()),
                        adapter_id: batch.adapter_id.clone().or(stored.lora_name.clone()),
                        tenant_id: batch
                            .tenant_id
                            .clone()
                            .unwrap_or_else(|| engine.tenant_id.clone()),
                    };
                    self.conductor.observe(KvCacheEvent::BlockStored {
                        ctx: ModelContext::from_scope(&scope, stored.block_size),
                        prefix_hashes: stored.block_hashes.clone(),
                        instance: engine.id.clone(),
                    });
                }
                KvEvent::AllBlocksCleared => {
                    self.conductor.observe(KvCacheEvent::InstanceGone {
                        instance: engine.id.clone(),
                    });
                }
                KvEvent::BlockRemoved(_) => {}
            }
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
        self.decide(request, &self.workers, &residency)
    }

    pub fn plan(&self, request: &RequestShape) -> Result<RequestPlan, ControlError> {
        // Count this request's prefixes for hot-prefix replication scheduling.
        self.hotness.record_request(request);
        // The router only ever inspects residency for *this request's* blocks
        // (local hit / transfer / recompute per block), so look those up
        // directly instead of dumping the whole index. O(request blocks) point
        // lookups — O(1) on the flat map, O(log n)/O(matches) on ART/LSM — keep
        // the online routing decision off the index's O(N) snapshot path.
        let residency: Vec<CacheResidency> = request
            .blocks
            .iter()
            .flat_map(|block| self.residency.locate(block))
            .collect();
        let decode_workers = self
            .workers
            .iter()
            .filter(|worker| worker.role.can_decode())
            .cloned()
            .collect::<Vec<_>>();
        let route_workers = if decode_workers.is_empty() {
            self.workers.clone()
        } else {
            decode_workers
        };
        let route = self.decide(request, &route_workers, &residency)?;
        let decode_worker_id = route.worker_id.clone();
        let decode_role = self
            .workers
            .iter()
            .find(|worker| worker.id == decode_worker_id)
            .map(|worker| worker.role)
            .unwrap_or(EngineRole::Aggregated);

        let prefill_worker_id = if decode_role == EngineRole::Decode && !route.recomputes.is_empty()
        {
            self.workers
                .iter()
                .filter(|worker| worker.role.can_prefill())
                .min_by_key(|worker| {
                    u64::from(worker.queued_prefill_tokens)
                        + u64::from(worker.running_decodes) * 1_000
                })
                .map(|worker| worker.id.clone())
        } else {
            None
        };
        let mode = if prefill_worker_id.is_some() {
            ServingMode::Disaggregated
        } else {
            ServingMode::Aggregated
        };
        let execution_worker_id = decode_worker_id.clone();
        let mut actions = Vec::new();

        for key in &route.local_hits {
            actions.push(PlanAction {
                kind: PlanActionKind::UseLocal,
                worker_id: decode_worker_id.clone(),
                source_worker_id: None,
                key: Some(key.clone()),
                tier: Some(CacheTier::Hbm),
                estimated_us: 0,
            });
        }
        for transfer in &route.transfers {
            actions.push(PlanAction {
                kind: PlanActionKind::Fetch,
                worker_id: decode_worker_id.clone(),
                source_worker_id: Some(transfer.from_worker_id.clone()),
                key: Some(transfer.key.clone()),
                tier: Some(transfer.tier),
                estimated_us: transfer.estimated_us,
            });
        }
        for recompute in &route.recomputes {
            actions.push(PlanAction {
                kind: if prefill_worker_id.is_some() {
                    PlanActionKind::RunPrefill
                } else {
                    PlanActionKind::Recompute
                },
                worker_id: prefill_worker_id
                    .clone()
                    .unwrap_or_else(|| execution_worker_id.clone()),
                source_worker_id: None,
                key: Some(recompute.key.clone()),
                tier: None,
                estimated_us: recompute.estimated_us,
            });
        }
        actions.push(PlanAction {
            kind: PlanActionKind::Decode,
            worker_id: decode_worker_id.clone(),
            source_worker_id: prefill_worker_id.clone(),
            key: None,
            tier: None,
            estimated_us: route.estimated_tpot_us,
        });

        Ok(RequestPlan {
            mode,
            execution_worker_id,
            prefill_worker_id,
            decode_worker_id,
            route,
            actions,
        })
    }

    /// Record that a request's blocks were placed on `engine_id` — *inferred*
    /// residency from the control plane's own routing decision. This closes the
    /// online loop: the engine runs with prefix caching, so after it serves a
    /// request its prefix blocks are resident there, and the next request for the
    /// same prefix should see a local hit. Without this the index only learns
    /// from `/v1/kv-events`, so cache-aware routing is blind until a bridge is
    /// wired. KV events (Tier 2) later *correct* this inference (e.g. on
    /// eviction); inferred residency is the floor, ground truth the upgrade.
    pub fn observe_placement(
        &mut self,
        engine_id: &str,
        request: &RequestShape,
        block_bytes: u64,
    ) -> Vec<DataPlaneAction> {
        let mut actions = Vec::new();
        for block in &request.blocks {
            let residency = CacheResidency::hbm(engine_id.to_string(), block.clone(), block_bytes);
            if self.data_plane.name() == "none" {
                self.residency.put(residency);
            } else {
                let update = self.data_plane.place(residency);
                actions.extend(self.mirror_data_plane_update(update));
            }
        }
        // Mirror the inferred placement into the Conductor's prefix table so
        // cache-aware routing learns from routing too (the floor; KV events refine).
        let (ctx, prefix_hashes) = Self::request_prefix(request);
        if !prefix_hashes.is_empty() {
            self.conductor.observe(KvCacheEvent::BlockStored {
                ctx,
                prefix_hashes,
                instance: engine_id.to_string(),
            });
        }
        actions
    }

    pub fn ingest(&mut self, batch: KvEventBatch) -> Result<IngestSummary, ControlError> {
        let mut summary = ingest_batch(self.residency.as_mut(), batch.clone(), &self.engines)?;
        self.feed_conductor(&batch);
        if self.data_plane.name() != "none" {
            let update = self.apply_batch_to_data_plane(&batch)?;
            self.mirror_data_plane_update(update);
            summary.total_resident_blocks = self.residency.len();
        }
        Ok(summary)
    }

    /// Audit identity-governed safe reuse for a request against current
    /// residency: which blocks are safe to reuse, and which content-matching
    /// blocks are refused because they belong to another identity. The router
    /// already only reuses on exact identity (`KvBlockKey` equality), so this
    /// never *changes* a decision — it makes the guard's refusals observable, so
    /// the online path can report the unsafe reuse it prevented (the same
    /// property the `safe-reuse` experiment measures offline).
    pub fn audit_reuse(&self, request: &RequestShape) -> ReuseAudit {
        let mut audit = ReuseAudit::default();
        for block in &request.blocks {
            // Only the blocks sharing *this* block's content hash can be a
            // (possibly cross-identity) reuse candidate — seek them directly
            // rather than scanning a full snapshot per request.
            let resident = self.residency.residency_by_content_hash(&block.block_hash);
            if resident.is_empty() {
                continue;
            }
            let scope = IdentityScope::from_key(block);
            if resident.iter().any(|r| scope.matches(&r.key)) {
                audit.safe_reusable += 1;
                continue;
            }
            // Content is resident, but only under other identities: refused.
            if let Some(violation) = resident.iter().find_map(|r| scope.reuse_violation(&r.key)) {
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

    /// Mark a request as in flight on an engine, bumping that worker's live
    /// decode load. The online router reads static `WorkerState` load, so without
    /// this the cost function's load term is inert and a cache-aware policy
    /// over-pins every request to the one cache-hot engine. Feeding the gateway's
    /// own in-flight count back into the worker state makes load-aware routing
    /// real on the live path — the gateway-side analogue of the engine metrics
    /// Dynamo/llm-d route on. Call [`Self::end_request`] when the request drains.
    pub fn begin_request(&mut self, engine_id: &str) {
        if let Some(worker) = self.workers.iter_mut().find(|w| w.id == engine_id) {
            worker.running_decodes = worker.running_decodes.saturating_add(1);
        }
    }

    /// Release a request's in-flight load on an engine (see [`Self::begin_request`]).
    pub fn end_request(&mut self, engine_id: &str) {
        if let Some(worker) = self.workers.iter_mut().find(|w| w.id == engine_id) {
            worker.running_decodes = worker.running_decodes.saturating_sub(1);
        }
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

    /// Decide which hot prefixes to replicate to spread cache hotspots: live
    /// access counts (the request path feeds the hotness tracker) crossed with
    /// where each hot prefix is resident (HBM) and current worker load. The cost
    /// router places one request at a time; this is the global balancing the
    /// Mooncake Conductor adds. The byte copies are the Transfer Engine's to
    /// execute. See [`crate::replication`].
    pub fn replication_plan(&self, cfg: &ReplicationConfig) -> Vec<ReplicationAction> {
        let workers: Vec<WorkerLoad> = self
            .workers
            .iter()
            .map(|w| WorkerLoad {
                id: w.id.clone(),
                load: w.running_decodes + w.queued_prefill_tokens,
            })
            .collect();
        let prefixes: Vec<PrefixResidency> = self
            .hotness
            .hot(cfg.hotness_threshold)
            .into_iter()
            .map(|(scope, prefix_hash, accesses)| {
                let mut holders: Vec<String> = self
                    .residency
                    .prefix_scan(&scope, &prefix_hash)
                    .into_iter()
                    .filter(|r| matches!(r.tier, CacheTier::Hbm | CacheTier::RemoteHbm))
                    .map(|r| r.worker_id)
                    .collect();
                holders.sort();
                holders.dedup();
                PrefixResidency {
                    scope,
                    prefix_hash,
                    holders,
                    accesses,
                }
            })
            .collect();
        plan_replications(&prefixes, &workers, cfg)
    }

    /// Decay the prefix hotness counts (call periodically) so replication tracks
    /// recent load rather than all-time totals.
    pub fn decay_hotness(&self) {
        self.hotness.decay();
    }

    /// Persist the residency index (checkpoint a persistent backend; no-op for
    /// in-memory). Call periodically and on shutdown so a persistent index
    /// survives a restart.
    pub fn flush(&self) {
        self.residency.flush();
    }

    fn mirror_data_plane_update(&mut self, update: DataPlaneUpdate) -> Vec<DataPlaneAction> {
        for removed in &update.removed {
            let scope = IdentityScope::from_key(&removed.key);
            self.residency
                .remove_block(&scope, &removed.worker_id, &removed.key.block_hash);
        }
        for resident in &update.resident {
            let scope = IdentityScope::from_key(&resident.key);
            self.residency
                .remove_block(&scope, &resident.worker_id, &resident.key.block_hash);
            self.residency.put(resident.clone());
        }
        update.actions
    }

    fn apply_batch_to_data_plane(
        &mut self,
        batch: &KvEventBatch,
    ) -> Result<DataPlaneUpdate, ControlError> {
        let engine = self
            .engines
            .iter()
            .find(|engine| engine.id == batch.engine_id)
            .ok_or_else(|| ControlError::UnknownEngine(batch.engine_id.clone()))?
            .clone();
        let mut out = DataPlaneUpdate::default();
        for event in batch.events.clone() {
            match event {
                KvEvent::BlockStored(event) => {
                    let update = self.apply_stored_to_data_plane(&engine, batch, event);
                    out.actions.extend(update.actions);
                    out.resident.extend(update.resident);
                    out.removed.extend(update.removed);
                }
                KvEvent::BlockRemoved(event) => {
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
                    for block_hash in event.block_hashes {
                        let update =
                            self.data_plane
                                .remove_block(&scope, &engine.id, block_hash.as_str());
                        out.actions.extend(update.actions);
                        out.resident.extend(update.resident);
                        out.removed.extend(update.removed);
                    }
                }
                KvEvent::AllBlocksCleared => {
                    let update = self.data_plane.clear_worker(&engine.id);
                    out.actions.extend(update.actions);
                    out.resident.extend(update.resident);
                    out.removed.extend(update.removed);
                }
            }
        }
        Ok(out)
    }

    fn apply_stored_to_data_plane(
        &mut self,
        engine: &EngineEndpoint,
        batch: &KvEventBatch,
        event: BlockStoredEvent,
    ) -> DataPlaneUpdate {
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
        let mut out = DataPlaneUpdate::default();

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
            let mut residency = CacheResidency::hbm(engine.id.clone(), key, block_bytes);
            residency.tier = tier;
            residency.last_access_ms = batch.ts_ms.unwrap_or(0);
            let update = self.data_plane.place(residency);
            out.actions.extend(update.actions);
            out.resident.extend(update.resident);
            out.removed.extend(update.removed);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineKind, SloTarget};

    fn engine() -> EngineEndpoint {
        EngineEndpoint {
            id: "vllm-a".to_string(),
            kind: EngineKind::Vllm,
            role: EngineRole::Aggregated,
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
    fn replication_plan_spreads_a_hot_singly_held_prefix() {
        let key = KvBlockKey {
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
            adapter_id: None,
            tenant_id: "tenant-a".to_string(),
            prefix_hash: "hotpre".to_string(),
            block_hash: "blk0".to_string(),
            block_index: 0,
            token_count: 16,
        };
        // The hot prefix is resident on vllm-a only.
        let mut index = MemoryIndex::new();
        index.put(CacheResidency::hbm("vllm-a".to_string(), key.clone(), 1024));
        let control = ControlPlane::with_index(
            vec![
                engine(),
                EngineEndpoint {
                    id: "vllm-b".to_string(),
                    base_url: "http://127.0.0.1:8002".to_string(),
                    ..engine()
                },
            ],
            Box::new(index),
        );

        // Drive enough requests for the prefix to count as hot (record runs first
        // in plan(), so it counts even if routing returns an error).
        let req = RequestShape {
            id: "r".to_string(),
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
            adapter_id: None,
            tenant_id: "tenant-a".to_string(),
            session_id: None,
            blocks: vec![key.clone()],
            estimated_decode_tokens: 8,
            slo: crate::SloTarget::default(),
        };
        for _ in 0..10 {
            let _ = control.plan(&req);
        }

        let actions = control.replication_plan(&ReplicationConfig::default());
        assert_eq!(actions.len(), 1, "one copy to reach target_replicas=2");
        assert_eq!(actions[0].prefix_hash, "hotpre");
        assert_eq!(actions[0].from_worker, "vllm-a");
        assert_eq!(actions[0].to_worker, "vllm-b");

        // After decay (10 -> 5, still >= threshold 8? no), it's no longer hot.
        control.decay_hotness();
        assert!(control
            .replication_plan(&ReplicationConfig::default())
            .is_empty());
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
    fn conductor_routing_picks_the_engine_with_the_cached_prefix() {
        let mut control = ControlPlane::new(vec![
            engine(),
            EngineEndpoint {
                id: "vllm-b".to_string(),
                base_url: "http://127.0.0.1:8002".to_string(),
                locality_domain: "local".to_string(),
                ..engine()
            },
        ])
        .with_conductor_routing(true);
        // vllm-b reports (via a KV event) that it cached prefix block "h0" — this
        // feeds the Conductor's prefix table.
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
        // The Conductor (prefix table fed by the KV event) routes to vllm-b.
        let decision = control.route(&request).unwrap();
        assert_eq!(decision.worker_id, "vllm-b");
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
        use crate::MockDataPlane;
        let control = ControlPlane::new(vec![engine()]);
        // By default there is no tensor data plane (infer from events).
        assert_eq!(control.data_plane().name(), "none");
        // A data plane (LMCache/KVBM/FlexKV adapter) plugs in at this seam.
        let control = control.with_data_plane(Box::new(MockDataPlane::new()));
        assert_eq!(control.data_plane().name(), "mock");
    }

    #[test]
    fn planner_emits_disaggregated_prefill_decode_plan() {
        let prefill = EngineEndpoint {
            id: "prefill-a".to_string(),
            role: EngineRole::Prefill,
            ..engine()
        };
        let decode = EngineEndpoint {
            id: "decode-a".to_string(),
            role: EngineRole::Decode,
            ..engine()
        };
        let control = ControlPlane::new(vec![prefill, decode]);
        let request = RequestShape {
            id: "req-pd".to_string(),
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
            adapter_id: None,
            tenant_id: "tenant-a".to_string(),
            session_id: None,
            blocks: vec![KvBlockKey::new(
                "Qwen/Qwen3-0.6B",
                "Qwen/Qwen3-0.6B",
                "tenant-a",
                "root",
                "cold",
                0,
                64,
            )],
            estimated_decode_tokens: 16,
            slo: SloTarget::default(),
        };

        let plan = control.plan(&request).unwrap();
        assert_eq!(plan.mode, ServingMode::Disaggregated);
        assert_eq!(plan.prefill_worker_id.as_deref(), Some("prefill-a"));
        assert_eq!(plan.decode_worker_id, "decode-a");
        assert!(plan
            .actions
            .iter()
            .any(|action| action.kind == PlanActionKind::RunPrefill));
        assert!(plan
            .actions
            .iter()
            .any(|action| action.kind == PlanActionKind::Decode));

        // The disaggregated plan implies a concrete prefill→decode KV handoff:
        // the cold block prefill-a computes is what decode-a must receive.
        let handoff = plan.kv_handoff().expect("disaggregated plan has a handoff");
        assert_eq!(handoff.request_id, "req-pd");
        assert_eq!(handoff.prefill_worker_id, "prefill-a");
        assert_eq!(handoff.decode_worker_id, "decode-a");
        assert_eq!(handoff.blocks.len(), 1);
    }

    #[test]
    fn aggregated_plan_has_no_kv_handoff() {
        // A single aggregated worker prefills and decodes in place — no KV
        // crosses the wire, so there is no prefill→decode handoff.
        let control = ControlPlane::new(vec![engine()]);
        let request = RequestShape {
            id: "req-agg".to_string(),
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
            adapter_id: None,
            tenant_id: "tenant-a".to_string(),
            session_id: None,
            blocks: vec![KvBlockKey::new(
                "Qwen/Qwen3-0.6B",
                "Qwen/Qwen3-0.6B",
                "tenant-a",
                "root",
                "cold",
                0,
                64,
            )],
            estimated_decode_tokens: 16,
            slo: SloTarget::default(),
        };
        let plan = control.plan(&request).unwrap();
        assert_eq!(plan.mode, ServingMode::Aggregated);
        assert!(plan.kv_handoff().is_none());
    }

    #[test]
    fn overload_admission_rejects_when_slo_cannot_be_met() {
        let cold = RequestShape {
            id: "r".to_string(),
            model_id: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
            adapter_id: None,
            tenant_id: "tenant-a".to_string(),
            session_id: None,
            blocks: vec![KvBlockKey::new(
                "Qwen/Qwen3-0.6B",
                "Qwen/Qwen3-0.6B",
                "tenant-a",
                "root",
                "cold",
                0,
                64,
            )],
            estimated_decode_tokens: 16,
            // A zero TTFT/TPOT budget can't be met by any prefill.
            slo: SloTarget {
                ttft_ms: 0,
                tpot_ms: 0,
            },
        };

        // With a zero violation budget, the request is rejected early (overload).
        let strict = ControlPlane::new(vec![engine()]).with_admission_slo_limit(0);
        match strict.admit(&cold).unwrap() {
            AdmissionDecision::Reject {
                best_slo_violation_us,
                ..
            } => assert!(best_slo_violation_us > 0),
            other => panic!("expected Reject, got {other:?}"),
        }

        // Default (no admission limit) admits the same request.
        let lenient = ControlPlane::new(vec![engine()]);
        assert!(matches!(
            lenient.admit(&cold).unwrap(),
            AdmissionDecision::Admit(_)
        ));
    }
}
