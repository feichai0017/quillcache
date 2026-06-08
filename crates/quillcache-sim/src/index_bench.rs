//! Index-backend micro-benchmark — the rig behind the ART-vs-LSM study.
//!
//! Drives any [`IndexBackend`] with a KV-cache-shaped workload and measures the
//! three operations that dominate a residency / prefix index:
//! - **ingest** (`put`, on KV `BlockStored` events),
//! - **churn** (`remove_block` + `put`, simulating eviction + recompute under
//!   cache pressure — what a real KV-cache index sees once HBM fills),
//! - **prefix lookup** (`prefix_scan`, on every request).
//!
//! Backend-agnostic, so memory / Holt (ART) / RocksDB (LSM) run the exact same
//! workload for an apples-to-apples comparison.

use quillcache_core::{CacheResidency, IdentityScope, IndexBackend, IndexMetrics, KvBlockKey};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexBenchConfig {
    /// Number of distinct requests / sessions ingested.
    pub requests: u32,
    /// Shared prefix blocks reused by every request (system prompt + RAG docs).
    pub shared_prefix_blocks: u32,
    /// Unique suffix blocks per request.
    pub unique_blocks_per_request: u32,
    /// Tokens per block.
    pub block_tokens: u32,
    /// Bytes per block (residency records the size only; no tensor is stored).
    pub block_bytes: u64,
    /// Number of `prefix_scan` queries in the read phase.
    pub scan_queries: u32,
    /// Eviction-churn cycles after ingest: each cycle evicts one request's unique
    /// blocks (`remove_block`) and re-inserts them (`put`), exercising the
    /// insert+evict path a real KV-cache index sees under pressure. 0 = no churn.
    pub churn_ops: u32,
}

impl Default for IndexBenchConfig {
    fn default() -> Self {
        Self {
            requests: 2_000,
            shared_prefix_blocks: 16,
            unique_blocks_per_request: 4,
            block_tokens: 64,
            block_bytes: 4 * 1024 * 1024,
            scan_queries: 20_000,
            churn_ops: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexBenchReport {
    pub backend: String,
    pub persistent: bool,
    /// Total residency records written (`put` calls) during ingest.
    pub blocks_ingested: u64,
    pub ingest_secs: f64,
    pub ingest_puts_per_sec: f64,
    /// Eviction-churn cycles run (0 if `--churn-ops` was 0).
    pub churn_ops: u64,
    pub churn_secs: f64,
    /// remove_block + put throughput during the churn phase (ops/sec).
    pub churn_ops_per_sec: f64,
    pub scan_queries: u64,
    pub scan_mean_us: f64,
    pub scan_p50_us: f64,
    pub scan_p99_us: f64,
    pub scan_p999_us: f64,
    /// Reopen-from-disk time for persistent backends; `None` for in-memory.
    pub recovery_ms: Option<f64>,
    pub metrics: IndexMetrics,
}

fn bench_scope() -> IdentityScope {
    IdentityScope {
        model_id: "bench-model".to_string(),
        tokenizer_id: "bench-tok".to_string(),
        adapter_id: None,
        tenant_id: "bench-tenant".to_string(),
    }
}

/// Deterministic prefix hash for shared-prefix block `i`.
fn shared_prefix_hash(i: u32) -> String {
    format!("sp-{i}")
}

fn shared_block(i: u32, tokens: u32) -> KvBlockKey {
    KvBlockKey::new(
        "bench-model",
        "bench-tok",
        "bench-tenant",
        shared_prefix_hash(i),
        format!("shared-{i}"),
        i,
        tokens,
    )
}

fn unique_block_hash(req: u32, j: u32) -> String {
    format!("uniq-{req}-{j}")
}

fn unique_block(req: u32, j: u32, idx: u32, tokens: u32) -> KvBlockKey {
    KvBlockKey::new(
        "bench-model",
        "bench-tok",
        "bench-tenant",
        format!("uq-{req}-{j}"),
        unique_block_hash(req, j),
        idx,
        tokens,
    )
}

/// Run the index micro-benchmark against `backend`.
pub fn bench_index<B: IndexBackend + ?Sized>(
    backend: &mut B,
    config: IndexBenchConfig,
) -> IndexBenchReport {
    let worker = "worker-0";
    let scope = bench_scope();
    let shared = config.shared_prefix_blocks.max(1);

    // ---- ingest phase: replay BlockStored events ----
    let mut blocks_ingested = 0u64;
    let ingest_start = Instant::now();
    for req in 0..config.requests {
        // Shared prefix: identical keys every request (a reused system prompt);
        // `put` overwrites, so the index keeps one residency per shared block.
        for i in 0..config.shared_prefix_blocks {
            backend.put(CacheResidency::hbm(
                worker,
                shared_block(i, config.block_tokens),
                config.block_bytes,
            ));
            blocks_ingested += 1;
        }
        // Unique suffix: distinct keys, so the index grows.
        for j in 0..config.unique_blocks_per_request {
            let idx = config.shared_prefix_blocks + j;
            backend.put(CacheResidency::hbm(
                worker,
                unique_block(req, j, idx, config.block_tokens),
                config.block_bytes,
            ));
            blocks_ingested += 1;
        }
    }
    let ingest_secs = ingest_start.elapsed().as_secs_f64();

    // ---- churn phase: evict + recompute one request's unique blocks per cycle ----
    let requests = config.requests.max(1);
    let churn_start = Instant::now();
    for c in 0..config.churn_ops {
        let req = c % requests;
        for j in 0..config.unique_blocks_per_request {
            backend.remove_block(&scope, worker, &unique_block_hash(req, j));
        }
        for j in 0..config.unique_blocks_per_request {
            let idx = config.shared_prefix_blocks + j;
            backend.put(CacheResidency::hbm(
                worker,
                unique_block(req, j, idx, config.block_tokens),
                config.block_bytes,
            ));
        }
    }
    let churn_secs = churn_start.elapsed().as_secs_f64();
    let churn_block_ops =
        u64::from(config.churn_ops) * u64::from(config.unique_blocks_per_request) * 2;

    // ---- query phase: prefix_scan against the populated index ----
    let mut samples_ns: Vec<u64> = Vec::with_capacity(config.scan_queries as usize);
    for q in 0..config.scan_queries {
        let prefix = shared_prefix_hash(q % shared);
        let t = Instant::now();
        let hits = backend.prefix_scan(&scope, &prefix);
        let elapsed = t.elapsed().as_nanos() as u64;
        std::hint::black_box(hits.len());
        samples_ns.push(elapsed);
    }
    let (scan_mean_us, scan_p50_us, scan_p99_us, scan_p999_us) = summarize_us(&mut samples_ns);

    IndexBenchReport {
        backend: backend.name().to_string(),
        persistent: backend.persistent(),
        blocks_ingested,
        ingest_secs,
        ingest_puts_per_sec: if ingest_secs > 0.0 {
            blocks_ingested as f64 / ingest_secs
        } else {
            0.0
        },
        churn_ops: u64::from(config.churn_ops),
        churn_secs,
        churn_ops_per_sec: if churn_secs > 0.0 {
            churn_block_ops as f64 / churn_secs
        } else {
            0.0
        },
        scan_queries: u64::from(config.scan_queries),
        scan_mean_us,
        scan_p50_us,
        scan_p99_us,
        scan_p999_us,
        recovery_ms: None,
        metrics: backend.metrics(),
    }
}

fn summarize_us(samples_ns: &mut [u64]) -> (f64, f64, f64, f64) {
    if samples_ns.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mean_us =
        samples_ns.iter().map(|&n| n as f64).sum::<f64>() / samples_ns.len() as f64 / 1_000.0;
    samples_ns.sort_unstable();
    let pick = |p: f64| -> f64 {
        let idx = ((samples_ns.len() as f64 - 1.0) * p).round() as usize;
        samples_ns[idx] as f64 / 1_000.0
    };
    (mean_us, pick(0.50), pick(0.99), pick(0.999))
}

#[cfg(test)]
mod tests {
    use super::*;
    use quillcache_core::MemoryIndex;

    #[test]
    fn bench_runs_against_memory_backend() {
        let mut idx = MemoryIndex::new();
        let report = bench_index(
            &mut idx,
            IndexBenchConfig {
                requests: 50,
                shared_prefix_blocks: 4,
                unique_blocks_per_request: 2,
                block_tokens: 32,
                block_bytes: 1024,
                scan_queries: 200,
                churn_ops: 0,
            },
        );

        assert_eq!(report.backend, "memory");
        assert!(!report.persistent);
        // 50 requests * (4 shared + 2 unique) = 300 put calls.
        assert_eq!(report.blocks_ingested, 300);
        // Distinct residencies: 4 shared + 50*2 unique = 104.
        assert_eq!(report.metrics.resident_blocks, 104);
        assert_eq!(report.scan_queries, 200);
        assert_eq!(report.churn_ops, 0);
        assert!(report.scan_p999_us >= report.scan_p99_us);
        assert!(report.scan_p99_us >= report.scan_p50_us);
    }

    #[test]
    fn churn_keeps_the_resident_set_bounded() {
        let mut idx = MemoryIndex::new();
        let report = bench_index(
            &mut idx,
            IndexBenchConfig {
                requests: 20,
                shared_prefix_blocks: 2,
                unique_blocks_per_request: 2,
                block_tokens: 32,
                block_bytes: 1024,
                scan_queries: 50,
                churn_ops: 100,
            },
        );

        // Churn evicts then re-inserts the *same* keys, so the resident set does
        // not grow: 2 shared + 20*2 unique = 42.
        assert_eq!(report.metrics.resident_blocks, 42);
        assert_eq!(report.churn_ops, 100);
        assert!(report.churn_ops_per_sec > 0.0);
    }
}
