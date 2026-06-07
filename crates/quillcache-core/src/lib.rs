use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineKind {
    Vllm,
    Sglang,
    Lmcache,
    Mock,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineEndpoint {
    pub id: String,
    pub kind: EngineKind,
    pub base_url: String,
    pub model_id: String,
    pub tokenizer_id: String,
    pub tenant_id: String,
    pub locality_domain: String,
}

impl EngineEndpoint {
    pub fn worker_state(&self) -> WorkerState {
        WorkerState::new(self.id.clone(), self.locality_domain.clone())
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
        self.model_id == key.model_id
            && self.tokenizer_id == key.tokenizer_id
            && self.adapter_id == key.adapter_id
            && self.tenant_id == key.tenant_id
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
pub trait IndexBackend {
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

    /// Full residency snapshot, for debugging and for routers that consume a
    /// slice of residency.
    fn snapshot(&self) -> Vec<CacheResidency>;

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
}
