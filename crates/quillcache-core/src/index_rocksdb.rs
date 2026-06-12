//! RocksDB (LSM) residency-index backend for QuillCache — the LSM baseline in
//! the ART-vs-LSM study.
//!
//! Keys are encoded so that an identity-scoped `prefix_scan` becomes a RocksDB
//! range scan:
//!
//! ```text
//! PRIMARY \0 model \0 tokenizer \0 adapter \0 tenant \0 prefix_hash \0
//!   block_hash \0 block_index(BE) \0 worker \0 tier   ->   serialized CacheResidency
//! ```
//!
//! `put` writes the record above plus a `SECONDARY`-tagged reverse-index entry
//! so eviction can seek straight to a block instead of scanning the scope:
//!
//! ```text
//! SECONDARY \0 model \0 tokenizer \0 adapter \0 tenant \0 block_hash \0
//!   worker \0 tier \0 prefix_hash \0 block_index(BE)   ->   the primary key
//! ```
//!
//! Both are single writes (no read-modify-write), so the store still behaves
//! like a real LSM under ingest. The value of a primary key is one
//! [`CacheResidency`]; a block resident on several workers/tiers maps to several
//! keys sharing a block prefix. `remove_block` is given a block hash but not its
//! prefix, so without the reverse index it would scan + deserialize the whole
//! scope (the churn bottleneck the index benchmark surfaced); with it, eviction
//! is an O(matches) range seek, at the cost of a second key per residency.

use crate::{CacheResidency, IdentityScope, IndexBackend, IndexMetrics, KvBlockKey};
use rocksdb::{Direction, IteratorMode, Options, DB};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const SEP: u8 = 0x00;
/// Namespace tag for primary residency records (prefix-ordered).
const PRIMARY: u8 = 0x01;
/// Namespace tag for the reverse `block_hash -> primary key` index.
const SECONDARY: u8 = 0x02;

/// An owned RocksDB key or value, as yielded by the iterator.
type OwnedKey = Box<[u8]>;

/// A persistent residency index backed by RocksDB (an LSM-tree store).
pub struct RocksIndex {
    db: DB,
    path: PathBuf,
    /// The `Options` used to open the DB, kept so its shared statistics handle can
    /// be read back for write-amplification. Behind a `Mutex` so `RocksIndex`
    /// stays `Sync`.
    opts: Mutex<Options>,
}

impl std::fmt::Debug for RocksIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocksIndex")
            .field("path", &self.path)
            .finish()
    }
}

impl RocksIndex {
    /// Open (creating if missing) a RocksDB-backed index at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, rocksdb::Error> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        // Track flush/compaction bytes so we can report real LSM write amplification.
        opts.enable_statistics();
        // A small memtable so even a modest residency index flushes and compacts
        // across levels — otherwise everything stays in one memtable and the LSM
        // write-amplification it pays at scale never shows up.
        opts.set_write_buffer_size(256 * 1024);
        opts.set_max_bytes_for_level_base(1024 * 1024);
        let db = DB::open(&opts, path.as_ref())?;
        Ok(Self {
            db,
            path: path.as_ref().to_path_buf(),
            opts: Mutex::new(opts),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Real LSM write amplification: (bytes flushed to L0 + bytes rewritten by
    /// compaction) / user bytes written, from RocksDB's own statistics. This is
    /// the cost an LSM pays — rewriting data during compaction — that an
    /// append-only / ART store does not. Returns `(physical_bytes, write_amp)`.
    pub fn write_amplification(&self) -> (u64, f64) {
        let stats = self
            .opts
            .lock()
            .ok()
            .and_then(|opts| opts.get_statistics())
            .unwrap_or_default();
        // Lines look like: "rocksdb.flush.write.bytes COUNT : 12345".
        let ticker = |name: &str| -> u64 {
            stats
                .lines()
                .find(|line| line.trim_start().starts_with(name))
                .and_then(|line| line.rsplit(':').next())
                .and_then(|tail| tail.trim().parse().ok())
                .unwrap_or(0)
        };
        let flush = ticker("rocksdb.flush.write.bytes");
        let compaction = ticker("rocksdb.compact.write.bytes");
        let physical = flush + compaction;
        // Relative to the live data finally retained on disk: how many times each
        // byte of stored data was physically written (flush once, then rewritten
        // by each compaction). ~1× with no compaction, higher as data cascades
        // through levels.
        let live = self.on_disk_bytes().max(1);
        let amp = physical as f64 / live as f64;
        (physical, amp)
    }

    /// Force a full compaction so on-disk size reflects the merged state.
    pub fn compact(&self) {
        self.db.compact_range(None::<&[u8]>, None::<&[u8]>);
    }

    /// Total size of the on-disk SST files, in bytes.
    pub fn on_disk_bytes(&self) -> u64 {
        self.db
            .property_int_value("rocksdb.total-sst-files-size")
            .ok()
            .flatten()
            .unwrap_or(0)
    }

    fn enc_scope(
        buf: &mut Vec<u8>,
        model: &str,
        tokenizer: &str,
        adapter: Option<&str>,
        tenant: &str,
    ) {
        buf.extend_from_slice(model.as_bytes());
        buf.push(SEP);
        buf.extend_from_slice(tokenizer.as_bytes());
        buf.push(SEP);
        buf.extend_from_slice(adapter.unwrap_or("").as_bytes());
        buf.push(SEP);
        buf.extend_from_slice(tenant.as_bytes());
        buf.push(SEP);
    }

    /// Primary scope prefix (`PRIMARY \0 scope`), the seek point for an
    /// identity-scoped `prefix_scan`.
    fn scope_prefix(scope: &IdentityScope) -> Vec<u8> {
        let mut buf = vec![PRIMARY];
        Self::enc_scope(
            &mut buf,
            &scope.model_id,
            &scope.tokenizer_id,
            scope.adapter_id.as_deref(),
            &scope.tenant_id,
        );
        buf
    }

    fn primary_key_for(residency: &CacheResidency) -> Vec<u8> {
        let k = &residency.key;
        let mut buf = vec![PRIMARY];
        Self::enc_scope(
            &mut buf,
            &k.model_id,
            &k.tokenizer_id,
            k.adapter_id.as_deref(),
            &k.tenant_id,
        );
        buf.extend_from_slice(k.prefix_hash.as_bytes());
        buf.push(SEP);
        buf.extend_from_slice(k.block_hash.as_bytes());
        buf.push(SEP);
        buf.extend_from_slice(&k.block_index.to_be_bytes());
        buf.push(SEP);
        buf.extend_from_slice(residency.worker_id.as_bytes());
        buf.push(SEP);
        buf.extend_from_slice(residency.tier.to_string().as_bytes());
        buf
    }

    /// Reverse-index key: `SECONDARY \0 scope \0 block_hash \0 worker \0 tier \0
    /// prefix_hash \0 block_index`. Ordering block_hash + worker first is what
    /// makes `remove_block` a bounded range seek. The value is the primary key.
    fn secondary_key_for(residency: &CacheResidency) -> Vec<u8> {
        let k = &residency.key;
        let mut buf = vec![SECONDARY];
        Self::enc_scope(
            &mut buf,
            &k.model_id,
            &k.tokenizer_id,
            k.adapter_id.as_deref(),
            &k.tenant_id,
        );
        buf.extend_from_slice(k.block_hash.as_bytes());
        buf.push(SEP);
        buf.extend_from_slice(residency.worker_id.as_bytes());
        buf.push(SEP);
        buf.extend_from_slice(residency.tier.to_string().as_bytes());
        buf.push(SEP);
        buf.extend_from_slice(k.prefix_hash.as_bytes());
        buf.push(SEP);
        buf.extend_from_slice(&k.block_index.to_be_bytes());
        buf
    }

    /// Reverse-index seek prefix for evicting `(scope, worker, block_hash)`
    /// across every tier / prefix / block index it is resident under.
    fn remove_scan_prefix(scope: &IdentityScope, worker_id: &str, block_hash: &str) -> Vec<u8> {
        let mut buf = vec![SECONDARY];
        Self::enc_scope(
            &mut buf,
            &scope.model_id,
            &scope.tokenizer_id,
            scope.adapter_id.as_deref(),
            &scope.tenant_id,
        );
        buf.extend_from_slice(block_hash.as_bytes());
        buf.push(SEP);
        buf.extend_from_slice(worker_id.as_bytes());
        buf.push(SEP);
        buf
    }

    fn scan_prefix(&self, prefix: &[u8]) -> Vec<CacheResidency> {
        let mut out = Vec::new();
        for item in self
            .db
            .iterator(IteratorMode::From(prefix, Direction::Forward))
        {
            let (k, v) = match item {
                Ok(kv) => kv,
                Err(_) => break,
            };
            if !k.starts_with(prefix) {
                break;
            }
            if let Ok(residency) = serde_json::from_slice::<CacheResidency>(&v) {
                out.push(residency);
            }
        }
        out
    }
}

impl IndexBackend for RocksIndex {
    fn name(&self) -> &str {
        "rocksdb"
    }

    fn put(&mut self, residency: CacheResidency) {
        let primary = Self::primary_key_for(&residency);
        if let Ok(value) = serde_json::to_vec(&residency) {
            let _ = self.db.put(&primary, value);
            // Reverse index: block_hash -> primary key, for O(matches) eviction.
            let secondary = Self::secondary_key_for(&residency);
            let _ = self.db.put(&secondary, &primary);
        }
    }

    fn locate(&self, key: &KvBlockKey) -> Vec<CacheResidency> {
        let mut prefix = Self::scope_prefix(&IdentityScope::from_key(key));
        prefix.extend_from_slice(key.prefix_hash.as_bytes());
        prefix.push(SEP);
        prefix.extend_from_slice(key.block_hash.as_bytes());
        prefix.push(SEP);
        prefix.extend_from_slice(&key.block_index.to_be_bytes());
        prefix.push(SEP);
        self.scan_prefix(&prefix)
    }

    fn prefix_scan(&self, scope: &IdentityScope, prefix_hash: &str) -> Vec<CacheResidency> {
        let mut prefix = Self::scope_prefix(scope);
        prefix.extend_from_slice(prefix_hash.as_bytes());
        prefix.push(SEP);
        self.scan_prefix(&prefix)
    }

    fn remove_block(&mut self, scope: &IdentityScope, worker_id: &str, block_hash: &str) -> usize {
        // Seek the reverse index straight to this (scope, block_hash, worker)
        // instead of scanning the whole identity scope. Each hit's value is the
        // primary key to drop; we drop the reverse entry alongside it.
        let prefix = Self::remove_scan_prefix(scope, worker_id, block_hash);
        let mut pairs: Vec<(OwnedKey, OwnedKey)> = Vec::new();
        for item in self
            .db
            .iterator(IteratorMode::From(prefix.as_slice(), Direction::Forward))
        {
            let (sec_key, prim_key) = match item {
                Ok(kv) => kv,
                Err(_) => break,
            };
            if !sec_key.starts_with(prefix.as_slice()) {
                break;
            }
            pairs.push((sec_key, prim_key));
        }
        let mut removed = 0;
        for (sec_key, prim_key) in pairs {
            if self.db.delete(&prim_key).is_ok() {
                removed += 1;
            }
            let _ = self.db.delete(&sec_key);
        }
        removed
    }

    fn clear_worker(&mut self, worker_id: &str) {
        let prefix = [PRIMARY];
        let mut to_delete: Vec<(OwnedKey, Vec<u8>)> = Vec::new();
        for item in self
            .db
            .iterator(IteratorMode::From(&prefix, Direction::Forward))
        {
            let (k, v) = match item {
                Ok(kv) => kv,
                Err(_) => break,
            };
            if !k.starts_with(&prefix) {
                break;
            }
            if let Ok(residency) = serde_json::from_slice::<CacheResidency>(&v) {
                if residency.worker_id == worker_id {
                    to_delete.push((k, Self::secondary_key_for(&residency)));
                }
            }
        }
        for (primary, secondary) in to_delete {
            let _ = self.db.delete(&primary);
            let _ = self.db.delete(&secondary);
        }
    }

    fn clear(&mut self) {
        let keys: Vec<_> = self
            .db
            .iterator(IteratorMode::Start)
            .filter_map(Result::ok)
            .map(|(k, _)| k)
            .collect();
        for k in keys {
            let _ = self.db.delete(&k);
        }
    }

    fn snapshot(&self) -> Vec<CacheResidency> {
        let prefix = [PRIMARY];
        let mut out = Vec::new();
        for item in self
            .db
            .iterator(IteratorMode::From(&prefix, Direction::Forward))
        {
            let (k, v) = match item {
                Ok(kv) => kv,
                Err(_) => break,
            };
            if !k.starts_with(&prefix) {
                break;
            }
            if let Ok(residency) = serde_json::from_slice::<CacheResidency>(&v) {
                out.push(residency);
            }
        }
        out
    }

    fn len(&self) -> usize {
        // Count only primary residency records, not reverse-index entries.
        let prefix = [PRIMARY];
        let mut n = 0;
        for item in self
            .db
            .iterator(IteratorMode::From(&prefix, Direction::Forward))
        {
            let Ok((k, _)) = item else { break };
            if !k.starts_with(&prefix) {
                break;
            }
            n += 1;
        }
        n
    }

    fn flush(&self) {
        let _ = self.db.flush_wal(true);
    }

    fn persistent(&self) -> bool {
        true
    }

    fn metrics(&self) -> IndexMetrics {
        let snapshot = self.snapshot();
        IndexMetrics {
            resident_blocks: snapshot.len() as u64,
            resident_bytes: snapshot.iter().map(|residency| residency.bytes).sum(),
            bytes_written: self.on_disk_bytes(),
            ..IndexMetrics::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvBlockKey;

    fn temp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("quillcache-rocks-{}-{}", tag, std::process::id()))
    }

    fn scope() -> IdentityScope {
        IdentityScope {
            model_id: "m".to_string(),
            tokenizer_id: "t".to_string(),
            adapter_id: None,
            tenant_id: "tenant-a".to_string(),
        }
    }

    fn block(prefix: &str, hash: &str, idx: u32) -> KvBlockKey {
        KvBlockKey::new("m", "t", "tenant-a", prefix, hash, idx, 64)
    }

    #[test]
    fn put_scan_remove_and_persist_across_reopen() {
        let dir = temp_dir("roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        {
            let mut idx = RocksIndex::open(&dir).unwrap();
            idx.put(CacheResidency::hbm("w0", block("root", "b0", 0), 1024));
            idx.put(CacheResidency::hbm("w0", block("b0", "b1", 1), 1024));
            // prefix_scan is identity-scoped and prefix-addressed.
            assert_eq!(idx.prefix_scan(&scope(), "root").len(), 1);
            assert_eq!(idx.prefix_scan(&scope(), "b0").len(), 1);
            // a different tenant with the same prefix must not match.
            let other = IdentityScope {
                tenant_id: "tenant-b".to_string(),
                ..scope()
            };
            assert!(idx.prefix_scan(&other, "root").is_empty());
            assert_eq!(idx.len(), 2);
            // remove one block.
            assert_eq!(idx.remove_block(&scope(), "w0", "b0"), 1);
            assert_eq!(idx.len(), 1);
        }
        // reopen: state survives (persistence).
        {
            let idx = RocksIndex::open(&dir).unwrap();
            assert!(idx.persistent());
            assert_eq!(idx.len(), 1);
            assert_eq!(idx.prefix_scan(&scope(), "b0").len(), 1);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reverse_index_remove_is_scoped_and_survives_reopen() {
        let dir = temp_dir("reverse");
        let _ = std::fs::remove_dir_all(&dir);
        {
            let mut idx = RocksIndex::open(&dir).unwrap();
            // Same block hash resident on two workers under the same prefix.
            idx.put(CacheResidency::hbm("w0", block("root", "b0", 0), 1024));
            idx.put(CacheResidency::hbm("w1", block("root", "b0", 0), 1024));
            idx.put(CacheResidency::hbm("w0", block("b0", "b1", 1), 1024));
            assert_eq!(idx.len(), 3);

            // Evict only w0's copy of b0; w1's copy and b1 remain.
            assert_eq!(idx.remove_block(&scope(), "w0", "b0"), 1);
            assert_eq!(idx.len(), 2);
            assert_eq!(idx.prefix_scan(&scope(), "root").len(), 1);
            assert_eq!(idx.remove_block(&scope(), "w0", "b0"), 0); // idempotent
        }
        {
            // Reverse index persisted: eviction still seeks correctly after reopen.
            let mut idx = RocksIndex::open(&dir).unwrap();
            assert_eq!(idx.len(), 2);
            assert_eq!(idx.remove_block(&scope(), "w1", "b0"), 1);
            assert!(idx.prefix_scan(&scope(), "root").is_empty());
            // Re-put rebuilds both namespaces (eviction -> recompute).
            idx.put(CacheResidency::hbm("w0", block("root", "b0", 0), 1024));
            assert_eq!(idx.prefix_scan(&scope(), "root").len(), 1);
            assert_eq!(idx.remove_block(&scope(), "w0", "b0"), 1);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
