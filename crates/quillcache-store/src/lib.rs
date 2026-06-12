//! The QuillCache KV block store — a Mooncake-Store-style data plane, in Rust.
//!
//! Mapping to the reference designs we are replicating:
//! - **Mooncake Store** = the pooled DRAM/SSD KV cache. Here that is
//!   [`LocalKvStore`] (the real byte pool on one node) plus [`PooledStore`]
//!   (local-first, else fetch a block from a peer node over the transfer engine).
//! - **Mooncake Transfer Engine** = the data path. We depend on
//!   `quillcache_transfer::Transfer` (TCP today, RDMA reserved) and serve our
//!   blocks to peers via [`StoreBlockSource`].
//! - **NVIDIA Dynamo KVBM** = the tiered block manager (G1 HBM / G2 host / G3
//!   disk). Here that is [`StoreDataPlane`]: it implements the control plane's
//!   `DataPlane` seam (HBM ↔ DRAM ↔ SSD admission / demotion / eviction) and,
//!   now fused with [`LocalKvStore`], its `place()` moves **real bytes** between
//!   the DRAM and SSD tiers, not just metadata.
//!
//! What QuillCache adds over both: the **identity guard** — a block is served
//! only when the requester's model · tokenizer · adapter · tenant matches.
//! Content-hash-keyed pools (Mooncake / LMCache / KVBM) leave that implicit.

use bytes::Bytes;
use quillcache_core::{
    CacheResidency, CacheTier, CostModel, DataPlane, DataPlaneAction, DataPlaneActionKind,
    DataPlaneMetrics, DataPlaneUpdate, IdentityScope, KvBlockKey, ReuseViolation,
};
use quillcache_transfer::{BlockSource, Transfer};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// =====================================================================
// LocalKvStore — the real KV byte pool (DRAM + SSD), identity-guarded.
// =====================================================================

/// Error from the real KV byte store.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("block not resident")]
    NotFound,
    /// The content is resident, but only under a different identity. Serving it
    /// would be a cross-tenant leak or a cross-adapter/model correctness error.
    #[error("unsafe cross-identity reuse refused ({0:?})")]
    Unsafe(ReuseViolation),
    #[error("io: {0}")]
    Io(String),
}

#[derive(Debug, Clone, Copy)]
struct BlockMeta {
    tier: CacheTier,
    bytes: u64,
    last_access: u64,
}

/// Where a `put` landed and what it pushed out.
#[derive(Debug)]
pub struct PutOutcome {
    pub tier: CacheTier,
    pub evicted: Vec<KvBlockKey>,
}

/// A durable manifest record for the SSD tier. The WAL is append-only: a block
/// becomes recoverable only once its `Commit` is fsynced — the atomic publish
/// point (NoKV's object-first write → single metadata commit). `Remove`
/// tombstones an evicted block. On reopen the WAL is replayed and each surviving
/// commit is verified against its on-disk file (length + CRC) before it
/// re-enters the index, so a half-written or corrupted block is never served and
/// a missing file never becomes a dangling pointer.
#[derive(Debug, Serialize, Deserialize)]
enum WalRecord {
    Commit {
        key: KvBlockKey,
        file_id: u64,
        len: u64,
        crc: u32,
    },
    Remove {
        key: KvBlockKey,
    },
}

/// A real single-node KV block byte pool with a DRAM tier (in memory) and an SSD
/// tier (files on disk). Holds actual bytes — not a simulation. With
/// [`Self::put`] it demotes the coldest block to SSD when DRAM is full and
/// evicts when disk is full (the standalone single-node store). When driven by
/// [`StoreDataPlane`] it exposes the tier moves as explicit primitives
/// ([`Self::put_dram`], [`Self::demote_to_ssd`], [`Self::drop_block`]) so the
/// data plane's tier policy moves the real bytes.
#[derive(Debug)]
pub struct LocalKvStore {
    dram: HashMap<KvBlockKey, Bytes>,
    ssd: HashMap<KvBlockKey, PathBuf>,
    meta: HashMap<KvBlockKey, BlockMeta>,
    /// content `block_hash` -> the full keys carrying it, for the identity guard.
    by_hash: HashMap<String, HashSet<KvBlockKey>>,
    ssd_dir: PathBuf,
    dram_capacity: u64,
    ssd_capacity: u64,
    dram_bytes: u64,
    ssd_bytes: u64,
    clock: u64,
    next_file: u64,
    admits: u64,
    hits: u64,
    demotions: u64,
    evictions: u64,
    wal: File,
}

impl LocalKvStore {
    /// Open a store backed by `ssd_dir` for the disk tier. Capacities are byte
    /// budgets for the DRAM and SSD tiers (use `u64::MAX` to disable the
    /// autonomous tiering and let an external policy drive the tier moves).
    pub fn new(
        ssd_dir: impl Into<PathBuf>,
        dram_capacity: u64,
        ssd_capacity: u64,
    ) -> std::io::Result<Self> {
        let ssd_dir = ssd_dir.into();
        std::fs::create_dir_all(&ssd_dir)?;
        // Fresh store: a clean WAL (leftover block files become ignorable orphans).
        let wal = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(ssd_dir.join("manifest.wal"))?;
        Ok(Self {
            dram: HashMap::new(),
            ssd: HashMap::new(),
            meta: HashMap::new(),
            by_hash: HashMap::new(),
            ssd_dir,
            dram_capacity,
            ssd_capacity,
            dram_bytes: 0,
            ssd_bytes: 0,
            clock: 0,
            next_file: 0,
            admits: 0,
            hits: 0,
            demotions: 0,
            evictions: 0,
            wal,
        })
    }

    /// Reopen a store from `ssd_dir`, recovering the durable SSD tier from its
    /// WAL. Replays the manifest, then admits only the blocks whose committed
    /// file is present and matches its recorded length + CRC. Half-written,
    /// uncommitted, or corrupted blocks are dropped (never served); their files
    /// are left as orphans for GC. The DRAM tier is volatile and starts empty.
    pub fn recover(
        ssd_dir: impl Into<PathBuf>,
        dram_capacity: u64,
        ssd_capacity: u64,
    ) -> std::io::Result<Self> {
        let ssd_dir = ssd_dir.into();
        std::fs::create_dir_all(&ssd_dir)?;
        let wal_path = ssd_dir.join("manifest.wal");

        // Replay the WAL into the live commit set (last write per key wins).
        let mut live: HashMap<KvBlockKey, (u64, u64, u32)> = HashMap::new();
        if let Ok(mut file) = File::open(&wal_path) {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            let mut offset = 0usize;
            while offset + 8 <= buf.len() {
                let len = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
                let crc = u32::from_le_bytes(buf[offset + 4..offset + 8].try_into().unwrap());
                let (start, end) = (offset + 8, offset + 8 + len);
                if end > buf.len() || crc32fast::hash(&buf[start..end]) != crc {
                    break; // torn or corrupt tail at the crash point: stop replay.
                }
                if let Ok(record) = serde_json::from_slice::<WalRecord>(&buf[start..end]) {
                    match record {
                        WalRecord::Commit {
                            key,
                            file_id,
                            len,
                            crc,
                        } => {
                            live.insert(key, (file_id, len, crc));
                        }
                        WalRecord::Remove { key } => {
                            live.remove(&key);
                        }
                    }
                }
                offset = end;
            }
        }

        // Verify each surviving commit against its on-disk file before trusting it.
        let mut ssd = HashMap::new();
        let mut meta = HashMap::new();
        let mut by_hash: HashMap<String, HashSet<KvBlockKey>> = HashMap::new();
        let mut ssd_bytes = 0u64;
        let mut next_file = 0u64;
        for (key, (file_id, len, crc)) in live {
            let path = ssd_dir.join(format!("blk-{file_id}.kv"));
            let Ok(data) = std::fs::read(&path) else {
                continue; // missing file -> not recoverable, no dangling pointer.
            };
            if data.len() as u64 != len || crc32fast::hash(&data) != crc {
                continue; // truncated / corrupted -> never served.
            }
            by_hash
                .entry(key.block_hash.clone())
                .or_default()
                .insert(key.clone());
            meta.insert(
                key.clone(),
                BlockMeta {
                    tier: CacheTier::LocalSsd,
                    bytes: len,
                    last_access: 0,
                },
            );
            ssd.insert(key, path);
            ssd_bytes += len;
            next_file = next_file.max(file_id + 1);
        }

        let wal = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)?;
        Ok(Self {
            dram: HashMap::new(),
            ssd,
            meta,
            by_hash,
            ssd_dir,
            dram_capacity,
            ssd_capacity,
            dram_bytes: 0,
            ssd_bytes,
            clock: 0,
            next_file,
            admits: 0,
            hits: 0,
            demotions: 0,
            evictions: 0,
            wal,
        })
    }

    /// Append a record to the WAL and fsync it — the durable commit point.
    fn wal_append(&mut self, record: &WalRecord) -> std::io::Result<()> {
        let payload = serde_json::to_vec(record).map_err(std::io::Error::other)?;
        let crc = crc32fast::hash(&payload);
        let mut framed = Vec::with_capacity(8 + payload.len());
        framed.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        framed.extend_from_slice(&crc.to_le_bytes());
        framed.extend_from_slice(&payload);
        self.wal.write_all(&framed)?;
        self.wal.sync_all()
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Admit a block's bytes (standalone path): land in DRAM, then capacity
    /// enforcement may demote it (or colder blocks) to SSD and evict from SSD.
    pub fn put(&mut self, key: KvBlockKey, data: Bytes) -> std::io::Result<PutOutcome> {
        self.put_dram(key.clone(), data);
        let mut evicted = Vec::new();
        self.enforce_capacity(&mut evicted)?;
        let tier = self
            .meta
            .get(&key)
            .map(|m| m.tier)
            .unwrap_or(CacheTier::CpuDram);
        Ok(PutOutcome { tier, evicted })
    }

    /// Insert a block's bytes into DRAM without autonomous tiering. Used when an
    /// external tier policy ([`StoreDataPlane`]) decides demotion/eviction.
    pub fn put_dram(&mut self, key: KvBlockKey, data: Bytes) {
        let now = self.tick();
        let len = data.len() as u64;
        let _ = self.remove_internal(&key);
        self.dram.insert(key.clone(), data);
        self.dram_bytes += len;
        self.meta.insert(
            key.clone(),
            BlockMeta {
                tier: CacheTier::CpuDram,
                bytes: len,
                last_access: now,
            },
        );
        self.by_hash
            .entry(key.block_hash.clone())
            .or_default()
            .insert(key);
        self.admits += 1;
    }

    /// Move a block's real bytes from DRAM to an SSD file. Returns `false` if the
    /// block has no bytes in DRAM (e.g. it was only ever inferred metadata).
    pub fn demote_to_ssd(&mut self, key: &KvBlockKey) -> std::io::Result<bool> {
        let Some(data) = self.dram.remove(key) else {
            return Ok(false);
        };
        let len = data.len() as u64;
        self.dram_bytes = self.dram_bytes.saturating_sub(len);
        self.next_file += 1;
        let file_id = self.next_file;
        let path = self.ssd_dir.join(format!("blk-{file_id}.kv"));
        // Object-first: write the block file and fsync it durable...
        {
            let mut file = File::create(&path)?;
            file.write_all(&data)?;
            file.sync_all()?;
        }
        let crc = crc32fast::hash(&data);
        // ...then a single atomic commit to the WAL — the publish point. A crash
        // before this leaves an orphan file with no commit (never served).
        self.wal_append(&WalRecord::Commit {
            key: key.clone(),
            file_id,
            len,
            crc,
        })?;
        self.ssd.insert(key.clone(), path);
        self.ssd_bytes += len;
        if let Some(m) = self.meta.get_mut(key) {
            m.tier = CacheTier::LocalSsd;
        }
        self.demotions += 1;
        Ok(true)
    }

    /// Drop a block from whichever tier holds it (deletes the SSD file if any).
    pub fn drop_block(&mut self, key: &KvBlockKey) -> bool {
        let Some(meta) = self.meta.remove(key) else {
            return false;
        };
        match meta.tier {
            CacheTier::LocalSsd => {
                // Tombstone first (durable), then delete the file: a crash in
                // between leaves a tombstoned orphan for GC, never a live pointer
                // to a deleted file.
                let _ = self.wal_append(&WalRecord::Remove { key: key.clone() });
                if let Some(path) = self.ssd.remove(key) {
                    let _ = std::fs::remove_file(path);
                }
                self.ssd_bytes = self.ssd_bytes.saturating_sub(meta.bytes);
            }
            _ => {
                self.dram.remove(key);
                self.dram_bytes = self.dram_bytes.saturating_sub(meta.bytes);
            }
        }
        if let Some(set) = self.by_hash.get_mut(&key.block_hash) {
            set.remove(key);
            if set.is_empty() {
                self.by_hash.remove(&key.block_hash);
            }
        }
        self.evictions += 1;
        true
    }

    /// Serve a block's bytes — but only if the requester's identity matches.
    /// Same content under a different identity returns [`StoreError::Unsafe`].
    pub fn get(&mut self, key: &KvBlockKey) -> Result<Bytes, StoreError> {
        if let Some(meta) = self.meta.get(key).copied() {
            let now = self.tick();
            if let Some(m) = self.meta.get_mut(key) {
                m.last_access = now;
            }
            self.hits += 1;
            return match meta.tier {
                CacheTier::LocalSsd => {
                    let path = self.ssd.get(key).ok_or(StoreError::NotFound)?;
                    std::fs::read(path)
                        .map(Bytes::from)
                        .map_err(|e| StoreError::Io(e.to_string()))
                }
                _ => self.dram.get(key).cloned().ok_or(StoreError::NotFound),
            };
        }
        if let Some(candidates) = self.by_hash.get(&key.block_hash) {
            let scope = IdentityScope::from_key(key);
            for candidate in candidates {
                if let Some(violation) = scope.reuse_violation(candidate) {
                    return Err(StoreError::Unsafe(violation));
                }
            }
        }
        Err(StoreError::NotFound)
    }

    /// Raw byte fetch by exact key — no identity guard (the requester's key
    /// already encodes identity). Used when serving blocks to a peer node.
    pub fn get_raw(&self, key: &KvBlockKey) -> Option<Bytes> {
        let meta = self.meta.get(key)?;
        match meta.tier {
            CacheTier::LocalSsd => {
                let path = self.ssd.get(key)?;
                std::fs::read(path).ok().map(Bytes::from)
            }
            _ => self.dram.get(key).cloned(),
        }
    }

    fn remove_internal(&mut self, key: &KvBlockKey) -> std::io::Result<()> {
        if let Some(meta) = self.meta.remove(key) {
            match meta.tier {
                CacheTier::LocalSsd => {
                    if let Some(path) = self.ssd.remove(key) {
                        let _ = std::fs::remove_file(path);
                    }
                    self.ssd_bytes = self.ssd_bytes.saturating_sub(meta.bytes);
                }
                _ => {
                    self.dram.remove(key);
                    self.dram_bytes = self.dram_bytes.saturating_sub(meta.bytes);
                }
            }
            if let Some(set) = self.by_hash.get_mut(&key.block_hash) {
                set.remove(key);
                if set.is_empty() {
                    self.by_hash.remove(&key.block_hash);
                }
            }
        }
        Ok(())
    }

    fn enforce_capacity(&mut self, evicted: &mut Vec<KvBlockKey>) -> std::io::Result<()> {
        while self.dram_bytes > self.dram_capacity {
            let Some(victim) = self.coldest_in(CacheTier::CpuDram) else {
                break;
            };
            if !self.demote_to_ssd(&victim)? {
                break;
            }
        }
        while self.ssd_bytes > self.ssd_capacity {
            let Some(victim) = self.coldest_in(CacheTier::LocalSsd) else {
                break;
            };
            if self.drop_block(&victim) {
                evicted.push(victim);
            } else {
                break;
            }
        }
        Ok(())
    }

    fn coldest_in(&self, tier: CacheTier) -> Option<KvBlockKey> {
        self.meta
            .iter()
            .filter(|(_, m)| m.tier == tier)
            .min_by_key(|(_, m)| m.last_access)
            .map(|(k, _)| k.clone())
    }

    pub fn len(&self) -> usize {
        self.meta.len()
    }

    pub fn is_empty(&self) -> bool {
        self.meta.is_empty()
    }

    pub fn tier_of(&self, key: &KvBlockKey) -> Option<CacheTier> {
        self.meta.get(key).map(|m| m.tier)
    }

    pub fn dram_bytes(&self) -> u64 {
        self.dram_bytes
    }

    pub fn ssd_bytes(&self) -> u64 {
        self.ssd_bytes
    }

    pub fn demotions(&self) -> u64 {
        self.demotions
    }

    pub fn evictions(&self) -> u64 {
        self.evictions
    }
}

// =====================================================================
// Cross-node fetch — the transfer-engine read path of the pooled store.
// =====================================================================

/// Serves a [`LocalKvStore`]'s blocks to peer nodes over the transfer engine.
/// This is the node's side of Mooncake's pooled store: any node can read a block
/// resident here.
#[derive(Clone)]
pub struct StoreBlockSource {
    inner: Arc<Mutex<LocalKvStore>>,
}

impl StoreBlockSource {
    pub fn new(inner: Arc<Mutex<LocalKvStore>>) -> Self {
        Self { inner }
    }
}

impl BlockSource for StoreBlockSource {
    fn get(&self, key: &KvBlockKey) -> Option<Bytes> {
        self.inner.lock().unwrap().get_raw(key)
    }

    fn put(&self, key: KvBlockKey, data: Bytes) {
        self.inner.lock().unwrap().put_dram(key, data);
    }
}

/// Resolves a node id to its transfer-engine address. This is Dynamo's
/// service-discovery role (etcd in production); [`StaticRegistry`] is the
/// in-memory dev backend, and an etcd-backed registry plugs in behind this trait
/// without touching the pooled read path.
pub trait NodeRegistry: Send + Sync + std::fmt::Debug {
    /// Transfer-engine address (`host:port`) for a node, if known.
    fn addr_of(&self, node_id: &str) -> Option<String>;
    /// The local node's id, so the pool can skip fetching from itself.
    fn this_node(&self) -> &str;
}

/// In-memory [`NodeRegistry`] — the etcd analogue for single-process / dev runs.
#[derive(Debug, Clone, Default)]
pub struct StaticRegistry {
    this_node: String,
    addrs: HashMap<String, String>,
}

impl StaticRegistry {
    pub fn new(this_node: impl Into<String>) -> Self {
        Self {
            this_node: this_node.into(),
            addrs: HashMap::new(),
        }
    }

    pub fn with_node(mut self, node_id: impl Into<String>, addr: impl Into<String>) -> Self {
        self.addrs.insert(node_id.into(), addr.into());
        self
    }
}

impl NodeRegistry for StaticRegistry {
    fn addr_of(&self, node_id: &str) -> Option<String> {
        self.addrs.get(node_id).cloned()
    }

    fn this_node(&self) -> &str {
        &self.this_node
    }
}

/// A node's view of the distributed KV cache pool — Mooncake Store's local shard
/// plus the cluster read path. Ties together this node's [`LocalKvStore`] (the
/// byte pool), the transfer engine (the data path), and a [`NodeRegistry`]
/// (node → address). [`Self::get_pooled`] is the pooled-cache read: serve
/// locally, else fetch from a peer the residency index located, admit it
/// locally, and return it — Conductor → metadata → Transfer Engine.
#[derive(Debug)]
pub struct PooledStore {
    local: LocalKvStore,
    transfer: Arc<dyn Transfer>,
    registry: Arc<dyn NodeRegistry>,
}

impl PooledStore {
    pub fn new(
        local: LocalKvStore,
        transfer: Arc<dyn Transfer>,
        registry: Arc<dyn NodeRegistry>,
    ) -> Self {
        Self {
            local,
            transfer,
            registry,
        }
    }

    pub fn local_mut(&mut self) -> &mut LocalKvStore {
        &mut self.local
    }

    /// Pooled read. `located_nodes` are the node ids the residency index says
    /// hold this block (`index.locate(key)` → worker ids). Serve locally first;
    /// otherwise try each located peer (skipping ourselves), resolve its address
    /// via the registry, fetch over the transfer engine, admit locally, and
    /// return. A local cross-identity match is refused before any fetch — the
    /// identity guard wins over a network round trip.
    pub async fn get_pooled(
        &mut self,
        key: &KvBlockKey,
        located_nodes: &[String],
    ) -> Result<Bytes, StoreError> {
        match self.local.get(key) {
            Ok(bytes) => return Ok(bytes),
            Err(StoreError::NotFound) => {}
            Err(other) => return Err(other),
        }
        for node in located_nodes {
            if node == self.registry.this_node() {
                continue;
            }
            let Some(addr) = self.registry.addr_of(node) else {
                continue;
            };
            // A peer that doesn't have it (stale index entry) or is unreachable
            // just falls through to the next located node.
            if let Ok(bytes) = self.transfer.read(&addr, key).await {
                // Keyed by the exact identity-bearing key, so safe to admit.
                let _ = self.local.put(key.clone(), bytes.clone());
                return Ok(bytes);
            }
        }
        Err(StoreError::NotFound)
    }
}

// =====================================================================
// StoreDataPlane — tiered block manager (DataPlane seam), fused with bytes.
// =====================================================================

/// Per-tier byte budgets for [`StoreDataPlane`].
#[derive(Debug, Clone, Copy)]
pub struct StoreTierConfig {
    pub hbm_capacity_bytes: u64,
    pub cpu_dram_capacity_bytes: u64,
    pub local_ssd_capacity_bytes: u64,
}

impl Default for StoreTierConfig {
    fn default() -> Self {
        Self {
            hbm_capacity_bytes: 2 * 1024 * 1024 * 1024,
            cpu_dram_capacity_bytes: 16 * 1024 * 1024 * 1024,
            local_ssd_capacity_bytes: 128 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone)]
struct TierEntry {
    residency: CacheResidency,
    last_access: u64,
}

#[derive(Debug, Default)]
struct WorkerCache {
    entries: HashMap<KvBlockKey, TierEntry>,
    clock: u64,
}

/// Tiered KV block manager (Dynamo KVBM analogue) implementing the control
/// plane's [`DataPlane`] seam: HBM ↔ DRAM ↔ SSD admission, promotion, demotion,
/// eviction across workers. Fused with per-worker [`LocalKvStore`] byte pools, so
/// when a block carrying real bytes is demoted DRAM→SSD or evicted, the bytes
/// actually move / are dropped — `place()` is a real data-path operation, not
/// just metadata. Blocks placed without bytes (inferred residency from routing)
/// are tracked as metadata only.
#[derive(Debug)]
pub struct StoreDataPlane {
    config: StoreTierConfig,
    workers: HashMap<String, WorkerCache>,
    byte_pools: HashMap<String, LocalKvStore>,
    base_dir: Option<PathBuf>,
    cost: CostModel,
    admits: u64,
    hits: u64,
    promotions: u64,
    demotions: u64,
    evictions: u64,
}

impl StoreDataPlane {
    pub fn new(config: StoreTierConfig) -> Self {
        Self {
            config,
            workers: HashMap::new(),
            byte_pools: HashMap::new(),
            base_dir: None,
            cost: CostModel::default(),
            admits: 0,
            hits: 0,
            promotions: 0,
            demotions: 0,
            evictions: 0,
        }
    }

    /// Root directory for the per-worker SSD byte pools (defaults to a temp dir).
    pub fn with_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.base_dir = Some(dir.into());
        self
    }

    pub fn config(&self) -> StoreTierConfig {
        self.config
    }

    fn byte_pool_mut(&mut self, worker_id: &str) -> std::io::Result<&mut LocalKvStore> {
        if !self.byte_pools.contains_key(worker_id) {
            let base = self.base_dir.clone().unwrap_or_else(|| {
                std::env::temp_dir().join(format!("qc-store-{}", std::process::id()))
            });
            // Byte pools hold all bytes; this data plane's tier machine drives the
            // demotion/eviction (u64::MAX disables the pool's autonomous tiering).
            let pool = LocalKvStore::new(base.join(worker_id), u64::MAX, u64::MAX)?;
            self.byte_pools.insert(worker_id.to_string(), pool);
        }
        Ok(self.byte_pools.get_mut(worker_id).unwrap())
    }

    /// Offload a block's real bytes into the store on `worker_id`. Lands in the
    /// DRAM tier; the tier machine may then demote it (or colder blocks) to SSD.
    pub fn put(&mut self, worker_id: &str, key: KvBlockKey, data: Bytes) -> std::io::Result<()> {
        let len = data.len() as u64;
        self.byte_pool_mut(worker_id)?.put_dram(key.clone(), data);
        let residency = CacheResidency {
            key,
            worker_id: worker_id.to_string(),
            tier: CacheTier::CpuDram,
            bytes: len,
            last_access_ms: 0,
            ref_count: 0,
            pinned: false,
        };
        self.place(residency);
        Ok(())
    }

    /// Serve a block's real bytes from `worker_id`, identity-guarded.
    pub fn get(&mut self, worker_id: &str, key: &KvBlockKey) -> Result<Bytes, StoreError> {
        match self.byte_pools.get_mut(worker_id) {
            Some(pool) => pool.get(key),
            None => Err(StoreError::NotFound),
        }
    }

    fn capacity(&self, tier: CacheTier) -> u64 {
        match tier {
            CacheTier::Hbm => self.config.hbm_capacity_bytes,
            CacheTier::CpuDram => self.config.cpu_dram_capacity_bytes,
            CacheTier::LocalSsd => self.config.local_ssd_capacity_bytes,
            CacheTier::RemoteHbm | CacheTier::ObjectStore => 0,
        }
    }

    fn lower_tier(tier: CacheTier) -> Option<CacheTier> {
        match tier {
            CacheTier::Hbm => Some(CacheTier::CpuDram),
            CacheTier::CpuDram => Some(CacheTier::LocalSsd),
            CacheTier::LocalSsd => None,
            CacheTier::RemoteHbm | CacheTier::ObjectStore => None,
        }
    }

    fn tier_rank(tier: CacheTier) -> u8 {
        match tier {
            CacheTier::Hbm => 0,
            CacheTier::RemoteHbm => 1,
            CacheTier::CpuDram => 2,
            CacheTier::LocalSsd => 3,
            CacheTier::ObjectStore => 4,
        }
    }

    fn tier_bytes(worker: &WorkerCache, tier: CacheTier) -> u64 {
        worker
            .entries
            .values()
            .filter(|entry| entry.residency.tier == tier)
            .map(|entry| entry.residency.bytes)
            .sum()
    }

    fn coldest_in_tier(worker: &WorkerCache, tier: CacheTier) -> Option<KvBlockKey> {
        worker
            .entries
            .iter()
            .filter(|(_, entry)| entry.residency.tier == tier)
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(key, _)| key.clone())
    }

    fn action(
        &self,
        kind: DataPlaneActionKind,
        residency: &CacheResidency,
        from_tier: Option<CacheTier>,
        to_tier: Option<CacheTier>,
    ) -> DataPlaneAction {
        let estimated_us = to_tier
            .or(from_tier)
            .map(|tier| {
                self.cost
                    .transfer_cost_us(tier, residency.bytes, true, true)
            })
            .unwrap_or(0);
        DataPlaneAction {
            kind,
            worker_id: residency.worker_id.clone(),
            key: residency.key.clone(),
            from_tier,
            to_tier,
            bytes: residency.bytes,
            estimated_us,
        }
    }

    fn enforce_capacity(
        &mut self,
        worker_id: &str,
        worker: &mut WorkerCache,
        update: &mut DataPlaneUpdate,
        affected: &mut HashSet<KvBlockKey>,
    ) {
        for tier in [CacheTier::Hbm, CacheTier::CpuDram, CacheTier::LocalSsd] {
            while Self::tier_bytes(worker, tier) > self.capacity(tier) {
                let Some(victim_key) = Self::coldest_in_tier(worker, tier) else {
                    break;
                };
                let Some(mut victim) = worker.entries.remove(&victim_key) else {
                    break;
                };
                affected.insert(victim_key.clone());

                if let Some(next_tier) = Self::lower_tier(tier) {
                    let old = victim.residency.clone();
                    victim.residency.tier = next_tier;
                    victim.residency.worker_id = worker_id.to_string();
                    victim.last_access = worker.clock;
                    worker.entries.insert(victim_key.clone(), victim.clone());
                    update.actions.push(self.action(
                        DataPlaneActionKind::Demote,
                        &victim.residency,
                        Some(old.tier),
                        Some(next_tier),
                    ));
                    self.demotions += 1;
                    // Fusion: a DRAM→SSD demotion moves the block's real bytes.
                    if next_tier == CacheTier::LocalSsd {
                        if let Some(pool) = self.byte_pools.get_mut(worker_id) {
                            let _ = pool.demote_to_ssd(&victim_key);
                        }
                    }
                } else {
                    update.actions.push(self.action(
                        DataPlaneActionKind::Evict,
                        &victim.residency,
                        Some(victim.residency.tier),
                        None,
                    ));
                    update.removed.push(victim.residency);
                    self.evictions += 1;
                    // Fusion: eviction drops the real bytes.
                    if let Some(pool) = self.byte_pools.get_mut(worker_id) {
                        pool.drop_block(&victim_key);
                    }
                }
            }
        }
    }
}

impl Default for StoreDataPlane {
    fn default() -> Self {
        Self::new(StoreTierConfig::default())
    }
}

impl DataPlane for StoreDataPlane {
    fn name(&self) -> &str {
        "store"
    }

    fn tier_of(&self, worker_id: &str, key: &KvBlockKey) -> Option<CacheTier> {
        self.workers
            .get(worker_id)?
            .entries
            .get(key)
            .map(|entry| entry.residency.tier)
    }

    fn fetch_cost_us(&self, worker_id: &str, key: &KvBlockKey) -> Option<u64> {
        let worker = self.workers.get(worker_id)?;
        let entry = worker.entries.get(key)?;
        Some(
            self.cost
                .transfer_cost_us(entry.residency.tier, entry.residency.bytes, true, true),
        )
    }

    fn place(&mut self, mut residency: CacheResidency) -> DataPlaneUpdate {
        let worker_id = residency.worker_id.clone();
        let mut update = DataPlaneUpdate::default();
        let mut affected = HashSet::new();
        let action_seed = self.cost;
        let mut worker = self.workers.remove(&worker_id).unwrap_or_default();
        worker.clock += 1;
        let target_tier = residency.tier;
        residency.last_access_ms = worker.clock;
        affected.insert(residency.key.clone());

        let existing = worker.entries.remove(&residency.key);
        match existing {
            Some(existing) if existing.residency.tier == target_tier => {
                self.hits += 1;
                update.actions.push(DataPlaneAction {
                    kind: DataPlaneActionKind::Hit,
                    worker_id: worker_id.clone(),
                    key: residency.key.clone(),
                    from_tier: Some(target_tier),
                    to_tier: Some(target_tier),
                    bytes: residency.bytes,
                    estimated_us: action_seed.transfer_cost_us(
                        target_tier,
                        residency.bytes,
                        true,
                        true,
                    ),
                });
            }
            Some(existing) => {
                let kind =
                    if Self::tier_rank(target_tier) < Self::tier_rank(existing.residency.tier) {
                        self.promotions += 1;
                        DataPlaneActionKind::Promote
                    } else {
                        self.demotions += 1;
                        DataPlaneActionKind::Demote
                    };
                update.actions.push(DataPlaneAction {
                    kind,
                    worker_id: worker_id.clone(),
                    key: residency.key.clone(),
                    from_tier: Some(existing.residency.tier),
                    to_tier: Some(target_tier),
                    bytes: residency.bytes,
                    estimated_us: action_seed.transfer_cost_us(
                        existing.residency.tier,
                        residency.bytes,
                        true,
                        true,
                    ),
                });
            }
            None => {
                self.admits += 1;
                update.actions.push(DataPlaneAction {
                    kind: DataPlaneActionKind::Admit,
                    worker_id: worker_id.clone(),
                    key: residency.key.clone(),
                    from_tier: None,
                    to_tier: Some(target_tier),
                    bytes: residency.bytes,
                    estimated_us: 0,
                });
            }
        }
        worker.entries.insert(
            residency.key.clone(),
            TierEntry {
                residency,
                last_access: worker.clock,
            },
        );
        self.enforce_capacity(&worker_id, &mut worker, &mut update, &mut affected);

        update.resident = affected
            .into_iter()
            .filter_map(|key| {
                worker
                    .entries
                    .get(&key)
                    .map(|entry| entry.residency.clone())
            })
            .collect();
        self.workers.insert(worker_id, worker);
        update
    }

    fn remove_block(
        &mut self,
        scope: &IdentityScope,
        worker_id: &str,
        block_hash: &str,
    ) -> DataPlaneUpdate {
        let mut update = DataPlaneUpdate::default();
        let Some(worker) = self.workers.get_mut(worker_id) else {
            return update;
        };
        let keys = worker
            .entries
            .keys()
            .filter(|key| key.block_hash == block_hash && scope.matches(key))
            .cloned()
            .collect::<Vec<_>>();
        for key in keys {
            if let Some(entry) = worker.entries.remove(&key) {
                update.actions.push(DataPlaneAction {
                    kind: DataPlaneActionKind::Remove,
                    worker_id: worker_id.to_string(),
                    key: entry.residency.key.clone(),
                    from_tier: Some(entry.residency.tier),
                    to_tier: None,
                    bytes: entry.residency.bytes,
                    estimated_us: 0,
                });
                update.removed.push(entry.residency);
            }
            if let Some(pool) = self.byte_pools.get_mut(worker_id) {
                pool.drop_block(&key);
            }
        }
        update
    }

    fn clear_worker(&mut self, worker_id: &str) -> DataPlaneUpdate {
        self.byte_pools.remove(worker_id);
        let Some(worker) = self.workers.remove(worker_id) else {
            return DataPlaneUpdate::default();
        };
        let removed = worker
            .entries
            .into_values()
            .map(|entry| entry.residency)
            .collect::<Vec<_>>();
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
        self.workers
            .values()
            .flat_map(|worker| {
                worker
                    .entries
                    .values()
                    .map(|entry| entry.residency.clone())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

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
            admits: self.admits,
            hits: self.hits,
            promotions: self.promotions,
            demotions: self.demotions,
            evictions: self.evictions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quillcache_transfer::{serve_listener, TcpTransfer};
    use tokio::net::TcpListener;

    fn tmp(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!("qc-store-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn key(tenant: &str, hash: &str) -> KvBlockKey {
        KvBlockKey::new("m", "t", tenant, "p", hash, 0, 64)
    }

    #[test]
    fn put_get_roundtrips_real_bytes() {
        let dir = tmp("rt");
        let mut store = LocalKvStore::new(&dir, 1 << 20, 1 << 20).unwrap();
        let k = key("ten-a", "h1");
        store
            .put(k.clone(), Bytes::from_static(b"real-kv-bytes"))
            .unwrap();
        assert_eq!(&store.get(&k).unwrap()[..], b"real-kv-bytes");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn identity_guard_refuses_cross_tenant() {
        let dir = tmp("guard");
        let mut store = LocalKvStore::new(&dir, 1 << 20, 1 << 20).unwrap();
        store
            .put(key("ten-a", "shared"), Bytes::from_static(b"secret-a"))
            .unwrap();
        let err = store.get(&key("ten-b", "shared")).unwrap_err();
        assert!(matches!(err, StoreError::Unsafe(ReuseViolation::Tenant)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn demotes_to_ssd_and_reads_back_from_disk() {
        let dir = tmp("demote");
        let mut store = LocalKvStore::new(&dir, 8, 1 << 20).unwrap();
        let k1 = key("ten-a", "b1");
        let k2 = key("ten-a", "b2");
        store
            .put(k1.clone(), Bytes::from_static(b"01234567"))
            .unwrap();
        store
            .put(k2.clone(), Bytes::from_static(b"89abcdef"))
            .unwrap();
        assert_eq!(store.tier_of(&k1), Some(CacheTier::LocalSsd));
        assert!(store.demotions() >= 1);
        assert_eq!(&store.get(&k1).unwrap()[..], b"01234567");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn evicts_when_over_ssd_capacity() {
        let dir = tmp("evict");
        let mut store = LocalKvStore::new(&dir, 4, 4).unwrap();
        store
            .put(key("ten-a", "e1"), Bytes::from_static(b"aaaa"))
            .unwrap();
        store
            .put(key("ten-a", "e2"), Bytes::from_static(b"bbbb"))
            .unwrap();
        store
            .put(key("ten-a", "e3"), Bytes::from_static(b"cccc"))
            .unwrap();
        assert!(store.evictions() >= 1);
        assert!(matches!(
            store.get(&key("ten-a", "e1")),
            Err(StoreError::NotFound)
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_data_plane_demotes_and_evicts_under_capacity() {
        let mut dp = StoreDataPlane::new(StoreTierConfig {
            hbm_capacity_bytes: 100,
            cpu_dram_capacity_bytes: 100,
            local_ssd_capacity_bytes: 100,
        });
        let mk = |hash: &str| KvBlockKey::new("m", "t", "ten", "root", hash, 0, 64);

        let a = dp.place(CacheResidency::hbm("w0", mk("a"), 100));
        assert_eq!(a.actions[0].kind, DataPlaneActionKind::Admit);
        assert_eq!(dp.tier_of("w0", &mk("a")), Some(CacheTier::Hbm));

        let b = dp.place(CacheResidency::hbm("w0", mk("b"), 100));
        assert_eq!(dp.tier_of("w0", &mk("b")), Some(CacheTier::Hbm));
        assert_eq!(dp.tier_of("w0", &mk("a")), Some(CacheTier::CpuDram));
        assert!(b
            .actions
            .iter()
            .any(|action| action.kind == DataPlaneActionKind::Demote));

        let _ = dp.place(CacheResidency::hbm("w0", mk("c"), 100));
        assert_eq!(dp.tier_of("w0", &mk("b")), Some(CacheTier::CpuDram));
        assert_eq!(dp.tier_of("w0", &mk("a")), Some(CacheTier::LocalSsd));

        let d = dp.place(CacheResidency::hbm("w0", mk("d"), 100));
        assert!(d
            .actions
            .iter()
            .any(|action| action.kind == DataPlaneActionKind::Evict));
        assert_eq!(dp.tier_of("w0", &mk("a")), None);
        assert_eq!(dp.metrics().resident_blocks, 3);
    }

    #[test]
    fn fused_place_moves_real_bytes_to_disk_and_back() {
        let dir = tmp("fuse");
        // DRAM tier holds ~8 bytes; the second offload pushes the first's REAL
        // bytes out to the SSD tier via the tier machine.
        let mut dp = StoreDataPlane::new(StoreTierConfig {
            hbm_capacity_bytes: u64::MAX,
            cpu_dram_capacity_bytes: 8,
            local_ssd_capacity_bytes: u64::MAX,
        })
        .with_dir(&dir);
        let f1 = key("ten-a", "f1");
        let f2 = key("ten-a", "f2");
        dp.put("w0", f1.clone(), Bytes::from_static(b"01234567"))
            .unwrap();
        dp.put("w0", f2.clone(), Bytes::from_static(b"89abcdef"))
            .unwrap();
        // Metadata says f1 demoted to SSD...
        assert_eq!(dp.tier_of("w0", &f1), Some(CacheTier::LocalSsd));
        // ...and the real bytes are on disk and read back, identity-guarded.
        assert_eq!(&dp.get("w0", &f1).unwrap()[..], b"01234567");
        assert_eq!(&dp.get("w0", &f2).unwrap()[..], b"89abcdef");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ssd_tier_survives_crash_and_rejects_half_written_or_corrupt_blocks() {
        let dir = tmp("crash");
        // Commit two blocks durably to the SSD tier (DRAM cap 0 forces demotion).
        {
            let mut store = LocalKvStore::new(&dir, 0, 1 << 20).unwrap();
            store
                .put(key("ten-a", "good1"), Bytes::from_static(b"durable-1"))
                .unwrap();
            store
                .put(key("ten-a", "good2"), Bytes::from_static(b"durable-2"))
                .unwrap();
            assert_eq!(
                store.tier_of(&key("ten-a", "good1")),
                Some(CacheTier::LocalSsd)
            );
            // A half-written block: a file with NO commit record (the crash hit
            // between the file write and the WAL commit).
            std::fs::write(dir.join("blk-999.kv"), b"uncommitted-garbage").unwrap();
            // `store` drops here = process death: in-memory state gone, disk remains.
        }

        // Recover from disk: only durably-committed blocks come back.
        let mut store = LocalKvStore::recover(&dir, 0, 1 << 20).unwrap();
        assert_eq!(
            &store.get(&key("ten-a", "good1")).unwrap()[..],
            b"durable-1"
        );
        assert_eq!(
            &store.get(&key("ten-a", "good2")).unwrap()[..],
            b"durable-2"
        );
        assert_eq!(store.len(), 2); // exactly the two committed blocks, no orphans/dangling.

        // Corruption: truncate good1's file, recover again -> it is dropped, good2 survives.
        let good1_path = store.ssd.get(&key("ten-a", "good1")).cloned().unwrap();
        drop(store);
        std::fs::write(&good1_path, b"x").unwrap(); // wrong length + CRC
        let mut store = LocalKvStore::recover(&dir, 0, 1 << 20).unwrap();
        assert!(matches!(
            store.get(&key("ten-a", "good1")),
            Err(StoreError::NotFound)
        ));
        assert_eq!(
            &store.get(&key("ten-a", "good2")).unwrap()[..],
            b"durable-2"
        );
        assert_eq!(store.len(), 1); // only the intact block survived.

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn pooled_store_fetches_from_a_located_peer_over_tcp() {
        // Node B holds a block and serves it over the transfer engine.
        let dir_b = tmp("pool-b");
        let store_b = Arc::new(Mutex::new(
            LocalKvStore::new(&dir_b, 1 << 20, 1 << 20).unwrap(),
        ));
        store_b
            .lock()
            .unwrap()
            .put(key("ten-a", "x1"), Bytes::from_static(b"remote-bytes"))
            .unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr_b = listener.local_addr().unwrap().to_string();
        tokio::spawn(serve_listener(
            listener,
            Arc::new(StoreBlockSource::new(store_b.clone())),
        ));

        // Node A: an empty local pool + a registry mapping node-b -> its address.
        let dir_a = tmp("pool-a");
        let local_a = LocalKvStore::new(&dir_a, 1 << 20, 1 << 20).unwrap();
        let registry = Arc::new(StaticRegistry::new("node-a").with_node("node-b", addr_b.clone()));
        let mut node_a = PooledStore::new(local_a, Arc::new(TcpTransfer), registry);

        // The residency index located the block on node-a (us — skipped) and
        // node-b; node-a resolves node-b's address and fetches it over TCP.
        let located = vec!["node-a".to_string(), "node-b".to_string()];
        let got = node_a
            .get_pooled(&key("ten-a", "x1"), &located)
            .await
            .unwrap();
        assert_eq!(&got[..], b"remote-bytes");
        // Now resident locally on A: a second read serves from the local pool.
        let cached = node_a.get_pooled(&key("ten-a", "x1"), &[]).await.unwrap();
        assert_eq!(&cached[..], b"remote-bytes");
        // A block located only on an unknown node is a clean miss.
        assert!(matches!(
            node_a
                .get_pooled(&key("ten-a", "absent"), &["node-c".to_string()])
                .await,
            Err(StoreError::NotFound)
        ));

        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }
}
