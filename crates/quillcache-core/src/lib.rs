//! `quillcache-core` — the shared data model and the cache-aware routing
//! **Conductor**, organized to mirror the reference designs (Mooncake / Dynamo):
//!
//! - **identity + cost** — `KvBlockKey`, `IdentityScope` / `ReuseViolation` (the
//!   identity guard), `CostModel`.
//! - **Conductor** (Mooncake's OSS KV-cache indexer) — the `conductor` module:
//!   `ModelContext` + `PrefixCacheTable` + `KVEventHandler`, queried by the
//!   cache-aware `router` (`DynamoCostRouter` = the Dynamo KV-router cost fn).
//! - **control plane** — `control` (`ControlPlane`) drives a request through the
//!   router, the residency index, and the `DataPlane` seam.
//! - **residency index** — the `IndexBackend` trait + `MemoryIndex`, with the
//!   persistent ART / LSM backends as the feature-gated `index_holt` /
//!   `index_rocksdb` modules (the ART-vs-LSM `bench` study).

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::str::FromStr;

pub mod bench;
pub mod conductor;
pub mod control;
pub mod router;

// Optional residency-index backends (the ART-vs-LSM study) live here as
// feature-gated *modules*, not separate crates — `holt` is pure Rust, `rocksdb`
// pulls C++/libclang, both off by default so the core build stays light.
#[cfg(feature = "holt")]
pub mod index_holt;
#[cfg(feature = "rocksdb")]
pub mod index_rocksdb;

pub use conductor::{Conductor, KVEventHandler, KvCacheEvent, ModelContext, PrefixCacheTable};
pub use control::*;
pub use router::*;

#[cfg(feature = "holt")]
pub use index_holt::HoltIndex;
#[cfg(feature = "rocksdb")]
pub use index_rocksdb::RocksIndex;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvBlockKey {
    pub model_id: String,
    pub tokenizer_id: String,
    pub adapter_id: Option<String>,
    pub tenant_id: String,
    pub prefix_hash: String,
    pub block_hash: String,
    pub block_index: u32,
    pub token_count: u32,
}

impl KvBlockKey {
    pub fn new(
        model_id: impl Into<String>,
        tokenizer_id: impl Into<String>,
        tenant_id: impl Into<String>,
        prefix_hash: impl Into<String>,
        block_hash: impl Into<String>,
        block_index: u32,
        token_count: u32,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            tokenizer_id: tokenizer_id.into(),
            adapter_id: None,
            tenant_id: tenant_id.into(),
            prefix_hash: prefix_hash.into(),
            block_hash: block_hash.into(),
            block_index,
            token_count,
        }
    }

    pub fn external_hash(parts: ExternalKvBlockKey) -> Self {
        Self {
            model_id: parts.model_id,
            tokenizer_id: parts.tokenizer_id,
            adapter_id: parts.adapter_id,
            tenant_id: parts.tenant_id,
            prefix_hash: parts.prefix_hash,
            block_hash: parts.block_hash,
            block_index: parts.block_index,
            token_count: parts.token_count,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalKvBlockKey {
    pub model_id: String,
    pub tokenizer_id: String,
    pub adapter_id: Option<String>,
    pub tenant_id: String,
    pub prefix_hash: String,
    pub block_hash: String,
    pub block_index: u32,
    pub token_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheTier {
    Hbm,
    RemoteHbm,
    CpuDram,
    LocalSsd,
    ObjectStore,
}

impl fmt::Display for CacheTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheTier::Hbm => f.write_str("hbm"),
            CacheTier::RemoteHbm => f.write_str("remote_hbm"),
            CacheTier::CpuDram => f.write_str("cpu_dram"),
            CacheTier::LocalSsd => f.write_str("local_ssd"),
            CacheTier::ObjectStore => f.write_str("object_store"),
        }
    }
}

impl FromStr for CacheTier {
    type Err = CacheTierParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "hbm" | "gpu" | "gpu_memory" | "vram" => Ok(Self::Hbm),
            "remote_hbm" | "remote_gpu" | "remote_gpu_memory" => Ok(Self::RemoteHbm),
            "cpu" | "dram" | "cpu_dram" | "host" | "host_memory" => Ok(Self::CpuDram),
            "ssd" | "local_ssd" | "nvme" | "disk" => Ok(Self::LocalSsd),
            "object" | "object_store" | "s3" | "blob" => Ok(Self::ObjectStore),
            _ => Err(CacheTierParseError {
                value: value.to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown cache tier or medium: {value}")]
pub struct CacheTierParseError {
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheResidency {
    pub key: KvBlockKey,
    pub worker_id: String,
    pub tier: CacheTier,
    pub bytes: u64,
    pub last_access_ms: u64,
    pub ref_count: u32,
    pub pinned: bool,
}

impl CacheResidency {
    pub fn hbm(worker_id: impl Into<String>, key: KvBlockKey, bytes: u64) -> Self {
        Self {
            key,
            worker_id: worker_id.into(),
            tier: CacheTier::Hbm,
            bytes,
            last_access_ms: 0,
            ref_count: 0,
            pinned: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerState {
    pub id: String,
    #[serde(default)]
    pub role: EngineRole,
    pub locality_domain: String,
    pub hbm_capacity_bytes: u64,
    pub hbm_used_bytes: u64,
    pub cpu_capacity_bytes: u64,
    pub cpu_used_bytes: u64,
    pub running_decodes: u32,
    pub queued_prefill_tokens: u32,
}

impl WorkerState {
    pub fn new(id: impl Into<String>, locality_domain: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role: EngineRole::Aggregated,
            locality_domain: locality_domain.into(),
            hbm_capacity_bytes: 80 * 1024 * 1024 * 1024,
            hbm_used_bytes: 0,
            cpu_capacity_bytes: 512 * 1024 * 1024 * 1024,
            cpu_used_bytes: 0,
            running_decodes: 0,
            queued_prefill_tokens: 0,
        }
    }

    pub fn with_load(mut self, queued_prefill_tokens: u32, running_decodes: u32) -> Self {
        self.queued_prefill_tokens = queued_prefill_tokens;
        self.running_decodes = running_decodes;
        self
    }

    pub fn with_role(mut self, role: EngineRole) -> Self {
        self.role = role;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineKind {
    Vllm,
    Sglang,
    Lmcache,
    Mock,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineRole {
    #[default]
    Aggregated,
    Prefill,
    Decode,
}

impl EngineRole {
    pub fn can_prefill(self) -> bool {
        matches!(self, Self::Aggregated | Self::Prefill)
    }

    pub fn can_decode(self) -> bool {
        matches!(self, Self::Aggregated | Self::Decode)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineEndpoint {
    pub id: String,
    pub kind: EngineKind,
    #[serde(default)]
    pub role: EngineRole,
    pub base_url: String,
    pub model_id: String,
    pub tokenizer_id: String,
    pub tenant_id: String,
    pub locality_domain: String,
}

impl EngineEndpoint {
    pub fn worker_state(&self) -> WorkerState {
        WorkerState::new(self.id.clone(), self.locality_domain.clone()).with_role(self.role)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockHint {
    pub block_hash: String,
    pub token_count: u32,
    pub bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestKvHints {
    pub request_id: Option<String>,
    pub model_id: Option<String>,
    pub tokenizer_id: Option<String>,
    pub adapter_id: Option<String>,
    pub tenant_id: Option<String>,
    pub session_id: Option<String>,
    pub block_hashes: Vec<String>,
    pub block_tokens: Option<u32>,
    pub estimated_decode_tokens: Option<u32>,
    pub block_bytes: Option<u64>,
}

impl RequestKvHints {
    pub fn to_blocks(
        &self,
        fallback_model_id: &str,
        fallback_tokenizer_id: &str,
        fallback_tenant_id: &str,
    ) -> Vec<KvBlockKey> {
        let model_id = self.model_id.as_deref().unwrap_or(fallback_model_id);
        let tokenizer_id = self
            .tokenizer_id
            .as_deref()
            .unwrap_or(fallback_tokenizer_id);
        let tenant_id = self.tenant_id.as_deref().unwrap_or(fallback_tenant_id);
        let token_count = self.block_tokens.unwrap_or(64);
        let mut parent = self.session_id.as_deref().unwrap_or("root").to_string();

        self.block_hashes
            .iter()
            .enumerate()
            .map(|(idx, block_hash)| {
                let key = KvBlockKey::external_hash(ExternalKvBlockKey {
                    model_id: model_id.to_string(),
                    tokenizer_id: tokenizer_id.to_string(),
                    adapter_id: self.adapter_id.clone(),
                    tenant_id: tenant_id.to_string(),
                    prefix_hash: parent.clone(),
                    block_hash: block_hash.clone(),
                    block_index: idx as u32,
                    token_count,
                });
                parent = block_hash.clone();
                key
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvEventBatch {
    pub engine_id: String,
    pub ts_ms: Option<u64>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub tokenizer_id: Option<String>,
    #[serde(default)]
    pub adapter_id: Option<String>,
    #[serde(default)]
    pub tenant_id: Option<String>,
    #[serde(default)]
    pub bytes_per_block: Option<u64>,
    pub events: Vec<KvEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KvEvent {
    BlockStored(BlockStoredEvent),
    BlockRemoved(BlockRemovedEvent),
    AllBlocksCleared,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockStoredEvent {
    pub block_hashes: Vec<String>,
    #[serde(default)]
    pub parent_block_hash: Option<String>,
    #[serde(default)]
    pub token_ids: Vec<u32>,
    pub block_size: u32,
    #[serde(default)]
    pub medium: Option<String>,
    #[serde(default)]
    pub lora_name: Option<String>,
    #[serde(default)]
    pub group_idx: Option<u32>,
    #[serde(default)]
    pub bytes_per_block: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockRemovedEvent {
    pub block_hashes: Vec<String>,
    #[serde(default)]
    pub medium: Option<String>,
    #[serde(default)]
    pub group_idx: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SloTarget {
    pub ttft_ms: u64,
    pub tpot_ms: u64,
}

impl Default for SloTarget {
    fn default() -> Self {
        Self {
            ttft_ms: 800,
            tpot_ms: 80,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestShape {
    pub id: String,
    pub model_id: String,
    pub tokenizer_id: String,
    pub adapter_id: Option<String>,
    pub tenant_id: String,
    /// Conversation/agent session this request belongs to, when known. Multi-turn
    /// and agentic workloads reuse a growing prefix, so a session-aware policy
    /// keeps a session's turns on the engine accumulating its KV.
    pub session_id: Option<String>,
    pub blocks: Vec<KvBlockKey>,
    pub estimated_decode_tokens: u32,
    pub slo: SloTarget,
}

impl RequestShape {
    pub fn input_tokens(&self) -> u32 {
        self.blocks.iter().map(|block| block.token_count).sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostModel {
    pub prefill_us_per_token: u64,
    pub decode_us_per_token: u64,
    pub queue_us_per_prefill_token: u64,
    pub running_decode_penalty_us: u64,
    pub hbm_hit_us: u64,
    pub remote_hbm_us_per_mb: u64,
    pub cpu_dram_us_per_mb: u64,
    pub local_ssd_us_per_mb: u64,
    pub object_store_us_per_mb: u64,
    pub cross_domain_penalty_us: u64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            prefill_us_per_token: 45,
            decode_us_per_token: 80,
            queue_us_per_prefill_token: 4,
            running_decode_penalty_us: 1_500,
            hbm_hit_us: 5,
            remote_hbm_us_per_mb: 20,
            cpu_dram_us_per_mb: 55,
            local_ssd_us_per_mb: 280,
            object_store_us_per_mb: 1_800,
            cross_domain_penalty_us: 350,
        }
    }
}

impl CostModel {
    pub fn prefill_cost_us(&self, tokens: u32) -> u64 {
        self.prefill_us_per_token * u64::from(tokens)
    }

    pub fn decode_cost_us(&self, tokens: u32, running_decodes: u32) -> u64 {
        self.decode_us_per_token * u64::from(tokens)
            + self.running_decode_penalty_us * u64::from(running_decodes)
    }

    pub fn queue_cost_us(&self, worker: &WorkerState) -> u64 {
        self.queue_us_per_prefill_token * u64::from(worker.queued_prefill_tokens)
    }

    pub fn transfer_cost_us(
        &self,
        tier: CacheTier,
        bytes: u64,
        same_worker: bool,
        same_locality_domain: bool,
    ) -> u64 {
        if same_worker && tier == CacheTier::Hbm {
            return self.hbm_hit_us;
        }

        let mb = bytes.div_ceil(1024 * 1024).max(1);
        let base = match tier {
            CacheTier::Hbm | CacheTier::RemoteHbm => self.remote_hbm_us_per_mb,
            CacheTier::CpuDram => self.cpu_dram_us_per_mb,
            CacheTier::LocalSsd => self.local_ssd_us_per_mb,
            CacheTier::ObjectStore => self.object_store_us_per_mb,
        } * mb;

        if same_worker || same_locality_domain {
            base
        } else {
            base + self.cross_domain_penalty_us
        }
    }
}

/// Identity scope for safe KV reuse.
///
/// Two blocks may share a `block_hash` yet be unsafe to reuse across each other
/// unless their model, tokenizer, adapter, and tenant agree. Backends carry this
/// scope so reuse stays identity-aware instead of matching on content hash
/// alone. This is the seam where unsafe reuse (wrong model/tenant/adapter) is
/// rejected before it ever reaches a routing decision.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IdentityScope {
    pub model_id: String,
    pub tokenizer_id: String,
    pub adapter_id: Option<String>,
    pub tenant_id: String,
}

impl IdentityScope {
    pub fn from_key(key: &KvBlockKey) -> Self {
        Self {
            model_id: key.model_id.clone(),
            tokenizer_id: key.tokenizer_id.clone(),
            adapter_id: key.adapter_id.clone(),
            tenant_id: key.tenant_id.clone(),
        }
    }

    /// Whether `key` belongs to this identity scope. Adapter identity must match
    /// exactly (including absence): a LoRA-adapted block is not reusable by a
    /// base-model request, and vice versa.
    pub fn matches(&self, key: &KvBlockKey) -> bool {
        self.reuse_violation(key).is_none()
    }

    /// Classify why a *content-matching* block (same `block_hash`) is unsafe to
    /// reuse for a request in this scope, or `None` when identity agrees and
    /// reuse is safe. Content hashes collide across identities — the same tokens
    /// produce the same `block_hash` — but the KV tensors do not, so a control
    /// plane that reuses on content alone serves wrong or leaked state. Checked
    /// in priority order (a block can violate on several axes at once).
    pub fn reuse_violation(&self, key: &KvBlockKey) -> Option<ReuseViolation> {
        if self.model_id != key.model_id {
            Some(ReuseViolation::Model)
        } else if self.tokenizer_id != key.tokenizer_id {
            Some(ReuseViolation::Tokenizer)
        } else if self.adapter_id != key.adapter_id {
            Some(ReuseViolation::Adapter)
        } else if self.tenant_id != key.tenant_id {
            Some(ReuseViolation::Tenant)
        } else {
            None
        }
    }

    /// Like [`Self::reuse_violation`] but comparing two scopes directly — the
    /// identity that wrote a cached object vs. a requester — for the metadata /
    /// master layer, where objects are keyed by content hash so the same key can
    /// be requested under a different, unsafe identity.
    pub fn reuse_violation_against(&self, other: &IdentityScope) -> Option<ReuseViolation> {
        if self.model_id != other.model_id {
            Some(ReuseViolation::Model)
        } else if self.tokenizer_id != other.tokenizer_id {
            Some(ReuseViolation::Tokenizer)
        } else if self.adapter_id != other.adapter_id {
            Some(ReuseViolation::Adapter)
        } else if self.tenant_id != other.tenant_id {
            Some(ReuseViolation::Tenant)
        } else {
            None
        }
    }
}

/// Why a content-hash-matching KV block is unsafe to reuse across an identity
/// boundary. The first three are **correctness** violations (the cached KV is
/// numerically wrong for the request); `Tenant` is a **privacy** violation (the
/// KV is valid but belongs to another tenant, so serving it leaks their state).
/// This is the contract data-plane caches (FlexKV / LMCache / KVBM) leave
/// implicit; QuillCache makes it explicit and measurable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReuseViolation {
    /// Different model weights — attention output, and thus KV, differs.
    Model,
    /// Different tokenizer — the same text can tokenize differently, so the
    /// "same" prefix is not actually the same token sequence.
    Tokenizer,
    /// Different LoRA adapter — adapted attention changes the KV tensors even
    /// for identical tokens and base weights.
    Adapter,
    /// Different tenant — the KV is numerically valid but reusing it serves one
    /// tenant's cached state to another (a prefix-cache privacy leak).
    Tenant,
}

impl ReuseViolation {
    /// A correctness violation yields *wrong* output; a privacy violation yields
    /// *leaked* output. They are mitigated and measured differently.
    pub fn is_correctness(self) -> bool {
        matches!(self, Self::Model | Self::Tokenizer | Self::Adapter)
    }

    pub fn is_privacy(self) -> bool {
        matches!(self, Self::Tenant)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Model => "cross_model",
            Self::Tokenizer => "cross_tokenizer",
            Self::Adapter => "cross_adapter",
            Self::Tenant => "cross_tenant",
        }
    }
}

/// Comparable metrics every index backend can report.
///
/// Fields that do not apply to a backend (for example `bytes_written` for a pure
/// in-memory map) are reported as zero. These are the numbers Experiment mode
/// compares across backends: in-memory vs Holt (persistent ART) vs RocksDB (LSM
/// baseline) vs filesystem.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexMetrics {
    pub resident_blocks: u64,
    pub resident_bytes: u64,
    pub puts: u64,
    pub removes: u64,
    pub prefix_scans: u64,
    /// Bytes physically written to the backing store, for write-amplification
    /// studies. In-memory backends report 0.
    pub bytes_written: u64,
}

/// A pluggable residency-index backend.
///
/// The index maps KV-block *identity* (`KvBlockKey`) to *residency* (which
/// worker and tier currently hold the block). It is the seam that lets
/// QuillCache compare interchangeable backends — in-memory, Holt (persistent
/// ART), RocksDB (LSM baseline), filesystem — on the same traces and policies.
///
/// Backends store and serve residency *metadata* only. They do not move or hold
/// KV tensors; that is the data plane (LMCache / KVBM / the engine itself).
/// Event translation (vLLM/SGLang KV events -> `KvBlockKey`) lives in the
/// control plane and is backend-agnostic, so every backend sees the same
/// `put` / `remove_block` / `clear_worker` calls.
///
/// `Send + Sync + Debug` so a backend can be held as `Box<dyn IndexBackend>`
/// inside an async control plane behind a lock and swapped at runtime.
pub trait IndexBackend: std::fmt::Debug + Send + Sync {
    /// Stable backend name for reports (for example "memory", "holt", "rocksdb").
    fn name(&self) -> &str;

    /// Insert or update a residency record for a block on a worker/tier.
    fn put(&mut self, residency: CacheResidency);

    /// Every residency for an exact block identity. A block may be resident on
    /// several workers or tiers at once.
    fn locate(&self, key: &KvBlockKey) -> Vec<CacheResidency>;

    /// Identity-aware prefix scan: residencies whose block belongs to `scope`
    /// and whose `prefix_hash` equals `prefix_hash`. This is the lookup where
    /// radix/ART backends are expected to beat flat maps and LSM stores.
    fn prefix_scan(&self, scope: &IdentityScope, prefix_hash: &str) -> Vec<CacheResidency>;

    /// Remove a single block (by content hash, within an identity scope) from a
    /// worker. Returns the number of residency records removed.
    fn remove_block(&mut self, scope: &IdentityScope, worker_id: &str, block_hash: &str) -> usize;

    /// Drop everything resident on one worker/engine (for `AllBlocksCleared` or
    /// worker loss).
    fn clear_worker(&mut self, worker_id: &str);

    /// Drop the entire index.
    fn clear(&mut self);

    /// Persist buffered state to disk (checkpoint the WAL / flush memtables). The
    /// in-memory reference backend is a no-op; persistent backends override this.
    /// The control plane calls it periodically and on graceful shutdown so a
    /// persistent residency index survives a restart.
    fn flush(&self) {}

    /// Full residency snapshot, for debugging and for routers that consume a
    /// slice of residency.
    fn snapshot(&self) -> Vec<CacheResidency>;

    /// Every residency whose block carries this content `block_hash`, across
    /// *all* identity scopes. The inline reuse guard uses it to find
    /// content-matching blocks that may belong to a **different** identity
    /// (the safety check). The default scans a snapshot — correct but O(N); the
    /// in-memory backend overrides it with a content-hash reverse index so the
    /// online guard is an O(matches) seek rather than a full index dump per
    /// request. Persistent backends can override similarly via their secondary
    /// namespace.
    fn residency_by_content_hash(&self, block_hash: &str) -> Vec<CacheResidency> {
        self.snapshot()
            .into_iter()
            .filter(|residency| residency.key.block_hash == block_hash)
            .collect()
    }

    /// Number of residency records currently held.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Comparable backend metrics (see [`IndexMetrics`]). The default derives
    /// `resident_blocks`/`resident_bytes` from a snapshot; persistent backends
    /// should override to also report `bytes_written` and counters.
    fn metrics(&self) -> IndexMetrics {
        let snapshot = self.snapshot();
        IndexMetrics {
            resident_blocks: snapshot.len() as u64,
            resident_bytes: snapshot.iter().map(|entry| entry.bytes).sum(),
            ..IndexMetrics::default()
        }
    }

    /// Whether this backend persists residency across process restarts. The
    /// in-memory reference backend returns `false`; Holt (ART), RocksDB (LSM),
    /// and filesystem backends return `true`. Drives recovery experiments and is
    /// surfaced in reports.
    fn persistent(&self) -> bool {
        false
    }
}

/// Canonical in-memory [`IndexBackend`]: a flat map from block identity to
/// residency. This is the reference backend and the baseline that persistent
/// backends (Holt/ART, RocksDB/LSM, filesystem) are compared against in
/// Experiment mode. It reports `bytes_written = 0` because nothing is persisted.
#[derive(Debug, Default)]
pub struct MemoryIndex {
    entries: HashMap<KvBlockKey, Vec<CacheResidency>>,
    /// Secondary reverse index: `(identity scope, block_hash) -> the set of full
    /// keys carrying that content hash`. `remove_block` is given a block hash but
    /// not its prefix, so without this it would scan the whole map; the reverse
    /// index turns eviction into an O(matches) lookup. This mirrors the
    /// secondary-namespace reverse index the persistent backends keep on disk.
    by_hash: HashMap<(IdentityScope, String), HashSet<KvBlockKey>>,
    /// Content-hash reverse index: `block_hash -> the set of full keys carrying
    /// that content hash, across *all* identities`. The inline reuse guard
    /// (`residency_by_content_hash`) needs cross-identity content matches; this
    /// makes that an O(matches) seek instead of a full snapshot scan per request.
    by_content_hash: HashMap<String, HashSet<KvBlockKey>>,
    puts: u64,
    removes: u64,
}

impl MemoryIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop a fully-evicted key from the content-hash reverse index.
    fn forget_content_hash(&mut self, key: &KvBlockKey) {
        if let Some(set) = self.by_content_hash.get_mut(&key.block_hash) {
            set.remove(key);
            if set.is_empty() {
                self.by_content_hash.remove(&key.block_hash);
            }
        }
    }
}

impl IndexBackend for MemoryIndex {
    fn name(&self) -> &str {
        "memory"
    }

    fn put(&mut self, residency: CacheResidency) {
        let key = residency.key.clone();
        let entries = self.entries.entry(key.clone()).or_default();
        entries.retain(|entry| {
            !(entry.worker_id == residency.worker_id && entry.tier == residency.tier)
        });
        entries.push(residency);
        self.by_hash
            .entry((IdentityScope::from_key(&key), key.block_hash.clone()))
            .or_default()
            .insert(key.clone());
        self.by_content_hash
            .entry(key.block_hash.clone())
            .or_default()
            .insert(key);
        self.puts += 1;
    }

    fn locate(&self, key: &KvBlockKey) -> Vec<CacheResidency> {
        self.entries.get(key).cloned().unwrap_or_default()
    }

    fn prefix_scan(&self, scope: &IdentityScope, prefix_hash: &str) -> Vec<CacheResidency> {
        self.entries
            .iter()
            .filter(|(key, _)| scope.matches(key) && key.prefix_hash == prefix_hash)
            .flat_map(|(_, entries)| entries.iter().cloned())
            .collect()
    }

    fn remove_block(&mut self, scope: &IdentityScope, worker_id: &str, block_hash: &str) -> usize {
        let map_key = (scope.clone(), block_hash.to_string());
        let Some(keys) = self.by_hash.get(&map_key) else {
            return 0;
        };
        // Snapshot the candidate keys so we can mutate `entries` while iterating.
        let candidates: Vec<KvBlockKey> = keys.iter().cloned().collect();
        let mut removed = 0;
        let mut emptied: Vec<KvBlockKey> = Vec::new();
        for key in candidates {
            if let Some(entries) = self.entries.get_mut(&key) {
                let before = entries.len();
                entries.retain(|entry| entry.worker_id != worker_id);
                removed += before - entries.len();
                if entries.is_empty() {
                    self.entries.remove(&key);
                    emptied.push(key);
                }
            }
        }
        if !emptied.is_empty() {
            if let Some(set) = self.by_hash.get_mut(&map_key) {
                for key in &emptied {
                    set.remove(key);
                }
                if set.is_empty() {
                    self.by_hash.remove(&map_key);
                }
            }
            for key in &emptied {
                self.forget_content_hash(key);
            }
        }
        self.removes += removed as u64;
        removed
    }

    fn clear_worker(&mut self, worker_id: &str) {
        let mut emptied: Vec<KvBlockKey> = Vec::new();
        self.entries.retain(|key, entries| {
            entries.retain(|entry| entry.worker_id != worker_id);
            if entries.is_empty() {
                emptied.push(key.clone());
                false
            } else {
                true
            }
        });
        for key in emptied {
            let map_key = (IdentityScope::from_key(&key), key.block_hash.clone());
            if let Some(set) = self.by_hash.get_mut(&map_key) {
                set.remove(&key);
                if set.is_empty() {
                    self.by_hash.remove(&map_key);
                }
            }
            self.forget_content_hash(&key);
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.by_hash.clear();
        self.by_content_hash.clear();
    }

    fn snapshot(&self) -> Vec<CacheResidency> {
        self.entries
            .values()
            .flat_map(|entries| entries.iter().cloned())
            .collect()
    }

    fn residency_by_content_hash(&self, block_hash: &str) -> Vec<CacheResidency> {
        // O(matches): seek the keys carrying this content hash, then their
        // residency — no full-index scan on the online reuse-guard path.
        let Some(keys) = self.by_content_hash.get(block_hash) else {
            return Vec::new();
        };
        keys.iter()
            .filter_map(|key| self.entries.get(key))
            .flat_map(|entries| entries.iter().cloned())
            .collect()
    }

    fn len(&self) -> usize {
        self.entries.values().map(Vec::len).sum()
    }

    fn metrics(&self) -> IndexMetrics {
        IndexMetrics {
            resident_blocks: self.len() as u64,
            resident_bytes: self
                .entries
                .values()
                .flatten()
                .map(|entry| entry.bytes)
                .sum(),
            puts: self.puts,
            removes: self.removes,
            prefix_scans: 0,
            bytes_written: 0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataPlaneMetrics {
    pub resident_blocks: u64,
    pub resident_bytes: u64,
    pub hbm_bytes: u64,
    pub cpu_dram_bytes: u64,
    pub local_ssd_bytes: u64,
    pub admits: u64,
    pub hits: u64,
    pub promotions: u64,
    pub demotions: u64,
    pub evictions: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataPlaneActionKind {
    Admit,
    Hit,
    Promote,
    Demote,
    Evict,
    Remove,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataPlaneAction {
    pub kind: DataPlaneActionKind,
    pub worker_id: String,
    pub key: KvBlockKey,
    pub from_tier: Option<CacheTier>,
    pub to_tier: Option<CacheTier>,
    pub bytes: u64,
    pub estimated_us: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataPlaneUpdate {
    pub actions: Vec<DataPlaneAction>,
    pub resident: Vec<CacheResidency>,
    pub removed: Vec<CacheResidency>,
}

impl DataPlaneUpdate {
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty() && self.resident.is_empty() && self.removed.is_empty()
    }
}

/// A KV tensor data plane (LMCache / NVIDIA Dynamo KVBM / Tencent FlexKV) that
/// QuillCache's control plane sits above. QuillCache owns metadata and
/// decisions; a data plane stores and moves KV tensor bytes across tiers (HBM /
/// DRAM / SSD / remote). This trait is the seam where such a system plugs in.
pub trait DataPlane: std::fmt::Debug + Send + Sync {
    /// Stable name for reports (for example "none", "mock", "lmcache").
    fn name(&self) -> &str;

    /// The tier a block's KV bytes currently occupy on `worker_id`, if resident.
    fn tier_of(&self, worker_id: &str, key: &KvBlockKey) -> Option<CacheTier>;

    /// Estimated microseconds to make the block available on `worker_id` (fetch
    /// / transfer / load from its tier). `None` if the data plane can't say.
    fn fetch_cost_us(&self, worker_id: &str, key: &KvBlockKey) -> Option<u64>;

    /// Observe that a worker has produced or touched a KV block. Implementations
    /// may admit it, promote it, demote colder blocks, or evict victims. The
    /// returned update is what the control plane mirrors into the residency
    /// index.
    fn place(&mut self, residency: CacheResidency) -> DataPlaneUpdate;

    /// Remove a block from a worker/tier namespace by identity scope and content
    /// hash. This is used when real KV events correct inferred placement.
    fn remove_block(
        &mut self,
        scope: &IdentityScope,
        worker_id: &str,
        block_hash: &str,
    ) -> DataPlaneUpdate;

    fn clear_worker(&mut self, worker_id: &str) -> DataPlaneUpdate;

    fn snapshot(&self) -> Vec<CacheResidency>;

    fn metrics(&self) -> DataPlaneMetrics {
        let snapshot = self.snapshot();
        DataPlaneMetrics {
            resident_blocks: snapshot.len() as u64,
            resident_bytes: snapshot.iter().map(|entry| entry.bytes).sum(),
            hbm_bytes: snapshot
                .iter()
                .filter(|entry| entry.tier == CacheTier::Hbm)
                .map(|entry| entry.bytes)
                .sum(),
            cpu_dram_bytes: snapshot
                .iter()
                .filter(|entry| entry.tier == CacheTier::CpuDram)
                .map(|entry| entry.bytes)
                .sum(),
            local_ssd_bytes: snapshot
                .iter()
                .filter(|entry| entry.tier == CacheTier::LocalSsd)
                .map(|entry| entry.bytes)
                .sum(),
            ..DataPlaneMetrics::default()
        }
    }
}

/// No external data plane: QuillCache infers residency from routing decisions
/// and KV events alone (the default — there is no separate tensor store to ask).
#[derive(Debug, Default)]
pub struct NoDataPlane;

impl DataPlane for NoDataPlane {
    fn name(&self) -> &str {
        "none"
    }

    fn tier_of(&self, _worker_id: &str, _key: &KvBlockKey) -> Option<CacheTier> {
        None
    }

    fn fetch_cost_us(&self, _worker_id: &str, _key: &KvBlockKey) -> Option<u64> {
        None
    }

    fn place(&mut self, _residency: CacheResidency) -> DataPlaneUpdate {
        DataPlaneUpdate::default()
    }

    fn remove_block(
        &mut self,
        _scope: &IdentityScope,
        _worker_id: &str,
        _block_hash: &str,
    ) -> DataPlaneUpdate {
        DataPlaneUpdate::default()
    }

    fn clear_worker(&mut self, _worker_id: &str) -> DataPlaneUpdate {
        DataPlaneUpdate::default()
    }

    fn snapshot(&self) -> Vec<CacheResidency> {
        Vec::new()
    }
}

/// In-memory reference data plane for tests and demos: records which tier each
/// `(worker, block)` lives in and reports a tier-derived fetch cost. A real
/// LMCache / KVBM / FlexKV adapter implements [`DataPlane`] the same way over a
/// live tensor store.
#[derive(Debug, Default)]
pub struct MockDataPlane {
    tiers: HashMap<(String, KvBlockKey), CacheTier>,
    cost: CostModel,
}

impl MockDataPlane {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that a block's KV bytes occupy `tier` on `worker_id`.
    pub fn place(&mut self, worker_id: impl Into<String>, key: KvBlockKey, tier: CacheTier) {
        self.tiers.insert((worker_id.into(), key), tier);
    }
}

impl DataPlane for MockDataPlane {
    fn name(&self) -> &str {
        "mock"
    }

    fn tier_of(&self, worker_id: &str, key: &KvBlockKey) -> Option<CacheTier> {
        self.tiers
            .get(&(worker_id.to_string(), key.clone()))
            .copied()
    }

    fn fetch_cost_us(&self, worker_id: &str, key: &KvBlockKey) -> Option<u64> {
        let tier = self.tier_of(worker_id, key)?;
        let bytes = u64::from(key.token_count) * 128 * 1024;
        Some(self.cost.transfer_cost_us(tier, bytes, true, true))
    }

    fn place(&mut self, residency: CacheResidency) -> DataPlaneUpdate {
        self.place(
            residency.worker_id.clone(),
            residency.key.clone(),
            residency.tier,
        );
        DataPlaneUpdate {
            actions: vec![DataPlaneAction {
                kind: DataPlaneActionKind::Admit,
                worker_id: residency.worker_id.clone(),
                key: residency.key.clone(),
                from_tier: None,
                to_tier: Some(residency.tier),
                bytes: residency.bytes,
                estimated_us: 0,
            }],
            resident: vec![residency],
            removed: Vec::new(),
        }
    }

    fn remove_block(
        &mut self,
        scope: &IdentityScope,
        worker_id: &str,
        block_hash: &str,
    ) -> DataPlaneUpdate {
        let mut removed = Vec::new();
        self.tiers.retain(|(worker, key), tier| {
            let should_remove =
                worker == worker_id && key.block_hash == block_hash && scope.matches(key);
            if should_remove {
                removed.push(CacheResidency {
                    key: key.clone(),
                    worker_id: worker.clone(),
                    tier: *tier,
                    bytes: u64::from(key.token_count) * 128 * 1024,
                    last_access_ms: 0,
                    ref_count: 0,
                    pinned: false,
                });
            }
            !should_remove
        });
        DataPlaneUpdate {
            actions: removed
                .iter()
                .map(|entry| DataPlaneAction {
                    kind: DataPlaneActionKind::Remove,
                    worker_id: entry.worker_id.clone(),
                    key: entry.key.clone(),
                    from_tier: Some(entry.tier),
                    to_tier: None,
                    bytes: entry.bytes,
                    estimated_us: 0,
                })
                .collect(),
            resident: Vec::new(),
            removed,
        }
    }

    fn clear_worker(&mut self, worker_id: &str) -> DataPlaneUpdate {
        let mut removed = Vec::new();
        self.tiers.retain(|(worker, key), tier| {
            let should_remove = worker == worker_id;
            if should_remove {
                removed.push(CacheResidency {
                    key: key.clone(),
                    worker_id: worker.clone(),
                    tier: *tier,
                    bytes: u64::from(key.token_count) * 128 * 1024,
                    last_access_ms: 0,
                    ref_count: 0,
                    pinned: false,
                });
            }
            !should_remove
        });
        DataPlaneUpdate {
            actions: removed
                .iter()
                .map(|entry| DataPlaneAction {
                    kind: DataPlaneActionKind::Remove,
                    worker_id: entry.worker_id.clone(),
                    key: entry.key.clone(),
                    from_tier: Some(entry.tier),
                    to_tier: None,
                    bytes: entry.bytes,
                    estimated_us: 0,
                })
                .collect(),
            resident: Vec::new(),
            removed,
        }
    }

    fn snapshot(&self) -> Vec<CacheResidency> {
        self.tiers
            .iter()
            .map(|((worker_id, key), tier)| CacheResidency {
                key: key.clone(),
                worker_id: worker_id.clone(),
                tier: *tier,
                bytes: u64::from(key.token_count) * 128 * 1024,
                last_access_ms: 0,
                ref_count: 0,
                pinned: false,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hbm_hit_is_cheaper_than_recompute_for_a_block() {
        let cost = CostModel::default();
        assert!(
            cost.transfer_cost_us(CacheTier::Hbm, 4 * 1024 * 1024, true, true)
                < cost.prefill_cost_us(64)
        );
    }

    #[test]
    fn data_plane_reports_tier_and_fetch_cost() {
        let key = KvBlockKey::new("m", "t", "ten", "p", "b", 0, 64);
        // No data plane: knows nothing (QuillCache infers from events instead).
        assert_eq!(NoDataPlane.name(), "none");
        assert_eq!(NoDataPlane.tier_of("w0", &key), None);
        assert_eq!(NoDataPlane.fetch_cost_us("w0", &key), None);

        // Mock data plane: records tiers and derives a fetch cost.
        let mut dp = MockDataPlane::new();
        assert_eq!(dp.name(), "mock");
        assert_eq!(dp.tier_of("w0", &key), None);
        dp.place("w0", key.clone(), CacheTier::LocalSsd);
        assert_eq!(dp.tier_of("w0", &key), Some(CacheTier::LocalSsd));
        assert!(dp.fetch_cost_us("w0", &key).unwrap() > 0);
        // A block placed nowhere is still unknown.
        assert_eq!(dp.tier_of("w1", &key), None);
    }

    fn mem_scope() -> IdentityScope {
        IdentityScope {
            model_id: "m".to_string(),
            tokenizer_id: "t".to_string(),
            adapter_id: None,
            tenant_id: "ten".to_string(),
        }
    }

    fn mem_block(prefix: &str, hash: &str, idx: u32) -> KvBlockKey {
        KvBlockKey::new("m", "t", "ten", prefix, hash, idx, 64)
    }

    #[test]
    fn reuse_violation_classifies_identity_mismatch() {
        let scope = IdentityScope {
            model_id: "m".into(),
            tokenizer_id: "t".into(),
            adapter_id: Some("lora-a".into()),
            tenant_id: "tenant-1".into(),
        };
        let same = KvBlockKey {
            adapter_id: Some("lora-a".into()),
            ..KvBlockKey::new("m", "t", "tenant-1", "p", "b", 0, 64)
        };
        assert_eq!(scope.reuse_violation(&same), None); // identical identity: safe
        assert!(scope.matches(&same));

        // Same content (block hash) but a different tenant: a privacy leak.
        let other_tenant = KvBlockKey {
            adapter_id: Some("lora-a".into()),
            ..KvBlockKey::new("m", "t", "tenant-2", "p", "b", 0, 64)
        };
        assert_eq!(
            scope.reuse_violation(&other_tenant),
            Some(ReuseViolation::Tenant)
        );
        assert!(scope.reuse_violation(&other_tenant).unwrap().is_privacy());

        // Different adapter: a correctness error (priority over tenant).
        let other_adapter = KvBlockKey::new("m", "t", "tenant-2", "p", "b", 0, 64); // adapter None
        assert_eq!(
            scope.reuse_violation(&other_adapter),
            Some(ReuseViolation::Adapter)
        );
        assert!(scope
            .reuse_violation(&other_adapter)
            .unwrap()
            .is_correctness());

        // Different model outranks everything else.
        let other_model = KvBlockKey::new("m2", "t", "tenant-1", "p", "b", 0, 64);
        assert_eq!(
            scope.reuse_violation(&other_model),
            Some(ReuseViolation::Model)
        );
    }

    #[test]
    fn remove_block_uses_the_reverse_index_and_stays_consistent() {
        let mut idx = MemoryIndex::new();
        idx.put(CacheResidency::hbm("w0", mem_block("root", "b0", 0), 1024));
        idx.put(CacheResidency::hbm("w0", mem_block("b0", "b1", 1), 1024));
        // Same block hash resident on a second worker: remove("w0", b0) must drop
        // only w0's copy and keep w1's, and the reverse-index entry must survive.
        idx.put(CacheResidency::hbm("w1", mem_block("root", "b0", 0), 1024));
        assert_eq!(idx.len(), 3);

        assert_eq!(idx.remove_block(&mem_scope(), "w0", "b0"), 1);
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.prefix_scan(&mem_scope(), "root").len(), 1); // w1 still there
        assert_eq!(idx.locate(&mem_block("root", "b0", 0)).len(), 1);

        // Removing the last copy drops the reverse-index bucket; a re-put rebuilds
        // it and the block is findable again (the eviction -> recompute cycle).
        assert_eq!(idx.remove_block(&mem_scope(), "w1", "b0"), 1);
        assert!(idx.prefix_scan(&mem_scope(), "root").is_empty());
        assert_eq!(idx.remove_block(&mem_scope(), "w1", "b0"), 0); // idempotent
        idx.put(CacheResidency::hbm("w0", mem_block("root", "b0", 0), 1024));
        assert_eq!(idx.prefix_scan(&mem_scope(), "root").len(), 1);

        // clear_worker must also prune the reverse index (no stale buckets).
        idx.clear_worker("w0");
        assert!(idx.prefix_scan(&mem_scope(), "root").is_empty());
        assert_eq!(idx.remove_block(&mem_scope(), "w0", "b0"), 0);
    }

    #[test]
    fn residency_by_content_hash_finds_cross_identity_and_stays_consistent() {
        let mut idx = MemoryIndex::new();
        // Same content hash "b0" under two different keys (different prefix
        // position) plus a copy on a second worker -> three residency records,
        // all of which the content-hash seek must surface (this is what the
        // online reuse guard relies on to spot cross-identity matches).
        idx.put(CacheResidency::hbm("w0", mem_block("root", "b0", 0), 1024));
        idx.put(CacheResidency::hbm("w1", mem_block("root", "b0", 0), 1024));
        idx.put(CacheResidency::hbm("w0", mem_block("other", "b0", 5), 1024));
        idx.put(CacheResidency::hbm("w0", mem_block("root", "b1", 1), 1024));

        assert_eq!(idx.residency_by_content_hash("b0").len(), 3);
        assert_eq!(idx.residency_by_content_hash("b1").len(), 1);
        assert!(idx.residency_by_content_hash("absent").is_empty());

        // Removing one worker's copy of one key leaves the others findable.
        assert_eq!(idx.remove_block(&mem_scope(), "w0", "b0"), 2); // both "b0" keys on w0
        assert_eq!(idx.residency_by_content_hash("b0").len(), 1); // w1's copy remains
        idx.clear_worker("w1");
        assert!(idx.residency_by_content_hash("b0").is_empty()); // bucket pruned
                                                                 // The override agrees with the default snapshot-scan implementation.
        let by_scan: Vec<_> = idx
            .snapshot()
            .into_iter()
            .filter(|r| r.key.block_hash == "b1")
            .collect();
        assert_eq!(idx.residency_by_content_hash("b1").len(), by_scan.len());
    }
}
