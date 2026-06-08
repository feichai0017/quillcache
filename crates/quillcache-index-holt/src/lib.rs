//! Holt (persistent ART) residency-index backend — the ART arm of the
//! ART-vs-LSM study.
//!
//! Holt is an adaptive-radix-tree store for **path-shaped keys** with crash-safe
//! (WAL) persistence, so an identity-scoped `prefix_scan` maps directly onto
//! `Tree::scan(prefix)` and is prefix-native (no full-table scan, no LSM
//! compaction). Keys use the same path-shaped encoding as the RocksDB backend:
//!
//! ```text
//! model \0 tokenizer \0 adapter \0 tenant \0 prefix_hash \0 block_hash \0
//!   block_index(BE) \0 worker \0 tier   ->   serialized CacheResidency
//! ```

use holt::{Durability, RangeEntry, Tree, TreeBuilder};
use quillcache_core::{CacheResidency, IdentityScope, IndexBackend, IndexMetrics, KvBlockKey};
use std::path::{Path, PathBuf};

const SEP: u8 = 0x00;

/// A persistent residency index backed by Holt (an adaptive radix tree).
pub struct HoltIndex {
    tree: Tree,
    dir: PathBuf,
}

impl std::fmt::Debug for HoltIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HoltIndex").field("dir", &self.dir).finish()
    }
}

impl HoltIndex {
    /// Open (creating if missing) a Holt-backed index under directory `dir`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, holt::Error> {
        let dir = dir.as_ref().to_path_buf();
        let _ = std::fs::create_dir_all(&dir);
        let tree = TreeBuilder::new(dir.join("index.holt"))
            .durability(Durability::Wal { sync: false })
            .open()?;
        Ok(Self { tree, dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Checkpoint the WAL so on-disk state reflects all writes.
    pub fn flush(&self) {
        let _ = self.tree.checkpoint();
    }

    /// On-disk footprint (sum of files under the index directory), in bytes.
    pub fn on_disk_bytes(&self) -> u64 {
        fn dir_size(path: &Path) -> u64 {
            let mut total = 0;
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    match entry.metadata() {
                        Ok(meta) if meta.is_dir() => total += dir_size(&entry.path()),
                        Ok(meta) => total += meta.len(),
                        Err(_) => {}
                    }
                }
            }
            total
        }
        dir_size(&self.dir)
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
        for entry in self.tree.scan(prefix) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => break,
            };
            if let RangeEntry::Key { value, .. } = entry {
                if let Ok(residency) = serde_json::from_slice::<CacheResidency>(&value) {
                    out.push(residency);
                }
            }
        }
        out
    }
}

impl IndexBackend for HoltIndex {
    fn name(&self) -> &str {
        "holt"
    }

    fn put(&mut self, residency: CacheResidency) {
        let key = Self::key_for(&residency);
        if let Ok(value) = serde_json::to_vec(&residency) {
            let _ = self.tree.put(&key, &value);
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
        let mut to_delete: Vec<Vec<u8>> = Vec::new();
        for entry in self.tree.scan(&prefix) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => break,
            };
            if let RangeEntry::Key { key, value, .. } = entry {
                if let Ok(residency) = serde_json::from_slice::<CacheResidency>(&value) {
                    if residency.key.block_hash == block_hash && residency.worker_id == worker_id {
                        to_delete.push(key.to_vec());
                    }
                }
            }
        }
        let mut removed = 0;
        for key in to_delete {
            if self.tree.delete(key.as_slice()).unwrap_or(false) {
                removed += 1;
            }
        }
        removed
    }

    fn clear_worker(&mut self, worker_id: &str) {
        let mut to_delete: Vec<Vec<u8>> = Vec::new();
        for entry in self.tree.range() {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => break,
            };
            if let RangeEntry::Key { key, value, .. } = entry {
                if let Ok(residency) = serde_json::from_slice::<CacheResidency>(&value) {
                    if residency.worker_id == worker_id {
                        to_delete.push(key.to_vec());
                    }
                }
            }
        }
        for key in to_delete {
            let _ = self.tree.delete(key.as_slice());
        }
    }

    fn clear(&mut self) {
        let mut to_delete: Vec<Vec<u8>> = Vec::new();
        for entry in self.tree.range() {
            if let Ok(RangeEntry::Key { key, .. }) = entry {
                to_delete.push(key.to_vec());
            }
        }
        for key in to_delete {
            let _ = self.tree.delete(key.as_slice());
        }
    }

    fn snapshot(&self) -> Vec<CacheResidency> {
        let mut out = Vec::new();
        for entry in self.tree.range() {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => break,
            };
            if let RangeEntry::Key { value, .. } = entry {
                if let Ok(residency) = serde_json::from_slice::<CacheResidency>(&value) {
                    out.push(residency);
                }
            }
        }
        out
    }

    fn len(&self) -> usize {
        self.tree.range().into_iter().filter_map(Result::ok).count()
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
        std::env::temp_dir().join(format!("quillcache-holt-{}-{}", tag, std::process::id()))
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
            let mut idx = HoltIndex::open(&dir).unwrap();
            idx.put(CacheResidency::hbm("w0", block("root", "b0", 0), 1024));
            idx.put(CacheResidency::hbm("w0", block("b0", "b1", 1), 1024));
            assert_eq!(idx.prefix_scan(&scope(), "root").len(), 1);
            assert_eq!(idx.prefix_scan(&scope(), "b0").len(), 1);
            let other = IdentityScope {
                tenant_id: "tenant-b".to_string(),
                ..scope()
            };
            assert!(idx.prefix_scan(&other, "root").is_empty());
            assert_eq!(idx.len(), 2);
            assert_eq!(idx.remove_block(&scope(), "w0", "b0"), 1);
            assert_eq!(idx.len(), 1);
            idx.flush();
        }
        {
            let idx = HoltIndex::open(&dir).unwrap();
            assert!(idx.persistent());
            assert_eq!(idx.len(), 1);
            assert_eq!(idx.prefix_scan(&scope(), "b0").len(), 1);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
