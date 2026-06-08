//! RocksDB (LSM) residency-index backend for QuillCache — the LSM baseline in
//! the ART-vs-LSM study.
//!
//! Keys are encoded so that an identity-scoped `prefix_scan` becomes a RocksDB
//! range scan:
//!
//! ```text
//! model \0 tokenizer \0 adapter \0 tenant \0 prefix_hash \0 block_hash \0
//!   block_index(BE) \0 worker \0 tier   ->   serialized CacheResidency
//! ```
//!
//! `put` is a single write (no read-modify-write), so the store behaves like a
//! real LSM under ingest. The value is one [`CacheResidency`]; a block resident
//! on several workers/tiers maps to several keys sharing a block prefix.

use quillcache_core::{CacheResidency, IdentityScope, IndexBackend, IndexMetrics, KvBlockKey};
use rocksdb::{Direction, IteratorMode, Options, DB};
use std::path::{Path, PathBuf};

const SEP: u8 = 0x00;

/// A persistent residency index backed by RocksDB (an LSM-tree store).
pub struct RocksIndex {
    db: DB,
    path: PathBuf,
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
        let db = DB::open(&opts, path.as_ref())?;
        Ok(Self {
            db,
            path: path.as_ref().to_path_buf(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
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

    fn scope_prefix(scope: &IdentityScope) -> Vec<u8> {
        let mut buf = Vec::new();
        Self::enc_scope(
            &mut buf,
            &scope.model_id,
            &scope.tokenizer_id,
            scope.adapter_id.as_deref(),
            &scope.tenant_id,
        );
        buf
    }

    fn key_for(residency: &CacheResidency) -> Vec<u8> {
        let k = &residency.key;
        let mut buf = Vec::new();
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
        let key = Self::key_for(&residency);
        if let Ok(value) = serde_json::to_vec(&residency) {
            let _ = self.db.put(key, value);
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
        let prefix = Self::scope_prefix(scope);
        let mut to_delete = Vec::new();
        for item in self
            .db
            .iterator(IteratorMode::From(prefix.as_slice(), Direction::Forward))
        {
            let (k, v) = match item {
                Ok(kv) => kv,
                Err(_) => break,
            };
            if !k.starts_with(prefix.as_slice()) {
                break;
            }
            if let Ok(residency) = serde_json::from_slice::<CacheResidency>(&v) {
                if residency.key.block_hash == block_hash && residency.worker_id == worker_id {
                    to_delete.push(k);
                }
            }
        }
        let mut removed = 0;
        for k in to_delete {
            if self.db.delete(&k).is_ok() {
                removed += 1;
            }
        }
        removed
    }

    fn clear_worker(&mut self, worker_id: &str) {
        let mut to_delete = Vec::new();
        for item in self.db.iterator(IteratorMode::Start) {
            let (k, v) = match item {
                Ok(kv) => kv,
                Err(_) => break,
            };
            if let Ok(residency) = serde_json::from_slice::<CacheResidency>(&v) {
                if residency.worker_id == worker_id {
                    to_delete.push(k);
                }
            }
        }
        for k in to_delete {
            let _ = self.db.delete(&k);
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
        let mut out = Vec::new();
        for item in self.db.iterator(IteratorMode::Start) {
            let (_, v) = match item {
                Ok(kv) => kv,
                Err(_) => break,
            };
            if let Ok(residency) = serde_json::from_slice::<CacheResidency>(&v) {
                out.push(residency);
            }
        }
        out
    }

    fn len(&self) -> usize {
        self.db
            .iterator(IteratorMode::Start)
            .filter_map(Result::ok)
            .count()
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
    use quillcache_core::KvBlockKey;

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
}
