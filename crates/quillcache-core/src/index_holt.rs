//! Holt (persistent ART) residency-index backend — the ART arm of the
//! ART-vs-LSM study.
//!
//! Holt is an adaptive-radix-tree store for **path-shaped keys** with crash-safe
//! (WAL) persistence, so an identity-scoped `prefix_scan` maps directly onto
//! `Tree::scan(prefix)` and is prefix-native (no full-table scan, no LSM
//! compaction). Keys use the same path-shaped encoding as the RocksDB backend:
//!
//! ```text
//! PRIMARY \0 model \0 tokenizer \0 adapter \0 tenant \0 prefix_hash \0
//!   block_hash \0 block_index(BE) \0 worker \0 tier   ->   serialized CacheResidency
//! ```
//!
//! A second, `SECONDARY`-tagged namespace is a reverse index that lets eviction
//! seek straight to a block instead of scanning the whole identity scope:
//!
//! ```text
//! SECONDARY \0 model \0 tokenizer \0 adapter \0 tenant \0 block_hash \0
//!   worker \0 tier \0 prefix_hash \0 block_index(BE)   ->   the primary key
//! ```
//!
//! `remove_block` is given a block hash but not its prefix, so against the
//! primary order alone it would scan + deserialize the whole scope (the churn
//! bottleneck the index benchmark surfaced). The reverse index makes it an
//! O(matches) seek, at the cost of a second key per residency on disk.

use crate::{CacheResidency, IdentityScope, IndexBackend, IndexMetrics, KvBlockKey};
use holt::{Durability, RangeEntry, Tree, TreeBuilder};
use std::path::{Path, PathBuf};

const SEP: u8 = 0x00;
/// Namespace tag for primary residency records (prefix-ordered).
const PRIMARY: u8 = 0x01;
/// Namespace tag for the reverse `block_hash -> primary key` index.
const SECONDARY: u8 = 0x02;

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
    /// makes `remove_block` a bounded seek. The value is the primary key.
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
        let primary = Self::primary_key_for(&residency);
        if let Ok(value) = serde_json::to_vec(&residency) {
            let _ = self.tree.put(&primary, &value);
            // Reverse index: block_hash -> primary key, for O(matches) eviction.
            let secondary = Self::secondary_key_for(&residency);
            let _ = self.tree.put(&secondary, &primary);
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
        let mut pairs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for entry in self.tree.scan(&prefix) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => break,
            };
            if let RangeEntry::Key { key, value, .. } = entry {
                pairs.push((key.to_vec(), value.to_vec()));
            }
        }
        let mut removed = 0;
        for (secondary, primary) in pairs {
            if self.tree.delete(primary.as_slice()).unwrap_or(false) {
                removed += 1;
            }
            let _ = self.tree.delete(secondary.as_slice());
        }
        removed
    }

    fn clear_worker(&mut self, worker_id: &str) {
        let mut to_delete: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for entry in self.tree.scan(&[PRIMARY]) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => break,
            };
            if let RangeEntry::Key { key, value, .. } = entry {
                if let Ok(residency) = serde_json::from_slice::<CacheResidency>(&value) {
                    if residency.worker_id == worker_id {
                        to_delete.push((key.to_vec(), Self::secondary_key_for(&residency)));
                    }
                }
            }
        }
        for (primary, secondary) in to_delete {
            let _ = self.tree.delete(primary.as_slice());
            let _ = self.tree.delete(secondary.as_slice());
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
        for entry in self.tree.scan(&[PRIMARY]) {
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
        // Count only primary residency records, not reverse-index entries.
        self.tree
            .scan(&[PRIMARY])
            .into_iter()
            .filter_map(Result::ok)
            .count()
    }

    fn flush(&self) {
        let _ = self.tree.checkpoint();
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

    #[test]
    fn reverse_index_remove_is_scoped_and_survives_reopen() {
        let dir = temp_dir("reverse");
        let _ = std::fs::remove_dir_all(&dir);
        {
            let mut idx = HoltIndex::open(&dir).unwrap();
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
            idx.flush();
        }
        {
            // Reverse index persisted: eviction still seeks correctly after reopen.
            let mut idx = HoltIndex::open(&dir).unwrap();
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
