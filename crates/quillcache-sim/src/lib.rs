use quillcache_core::{
    CacheResidency, IndexBackend, IndexMetrics, KvBlockKey, MemoryIndex, RequestShape, SloTarget,
    WorkerState,
};
use quillcache_router::{GreedyStatePlaneRouter, RouterError, RoutingPolicy};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod index_bench;
pub use index_bench::{bench_index, IndexBenchConfig, IndexBenchReport};

pub mod safe_reuse;
pub use safe_reuse::{run_safe_reuse, SafeReuseConfig, SafeReuseReport};

pub mod tiered;
pub use tiered::{run_tiered, TieredConfig, TieredReport};

#[derive(Debug, Error)]
pub enum SimError {
    #[error(transparent)]
    Router(#[from] RouterError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntheticWorkloadConfig {
    pub requests: u32,
    pub workers: u32,
    pub shared_prefix_blocks: u32,
    pub unique_blocks_per_request: u32,
    pub block_tokens: u32,
    pub block_bytes: u64,
}

impl Default for SyntheticWorkloadConfig {
    fn default() -> Self {
        Self {
            requests: 32,
            workers: 4,
            shared_prefix_blocks: 8,
            unique_blocks_per_request: 2,
            block_tokens: 64,
            block_bytes: 4 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionSummary {
    pub request_id: String,
    pub worker_id: String,
    pub reusable_blocks: usize,
    pub recompute_blocks: usize,
    pub transfer_blocks: usize,
    pub estimated_ttft_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationReport {
    pub policy: String,
    pub index_backend: String,
    pub total_requests: u32,
    pub workers: u32,
    pub cache_reusable_blocks: u64,
    pub recompute_blocks: u64,
    pub transfer_blocks: u64,
    pub avg_estimated_ttft_ms: f64,
    pub index_metrics: IndexMetrics,
    pub decisions: Vec<DecisionSummary>,
}

/// Run the default Experiment-mode workload: cache-aware greedy routing over the
/// in-memory reference backend.
pub fn run_synthetic(config: SyntheticWorkloadConfig) -> Result<SimulationReport, SimError> {
    run_synthetic_with(
        &GreedyStatePlaneRouter::default(),
        &mut MemoryIndex::new(),
        config,
    )
}

/// Experiment-mode core: replay a synthetic workload through any [`RoutingPolicy`]
/// and any [`IndexBackend`], so policies (load-only vs cache-aware vs future
/// SLO-/session-aware) and index backends (memory vs Holt/ART vs RocksDB/LSM vs
/// filesystem) can be compared on the *same* trace. The returned report carries
/// the policy name, the backend name, and the backend metrics — including
/// `bytes_written` for write-amplification studies on persistent backends.
pub fn run_synthetic_with<P, B>(
    policy: &P,
    index: &mut B,
    config: SyntheticWorkloadConfig,
) -> Result<SimulationReport, SimError>
where
    P: RoutingPolicy,
    B: IndexBackend,
{
    let workers: Vec<_> = (0..config.workers)
        .map(|idx| WorkerState::new(format!("worker-{idx}"), format!("rack-{}", idx % 2)))
        .collect();
    let mut decisions = Vec::new();
    let mut cache_reusable_blocks = 0;
    let mut recompute_blocks = 0;
    let mut transfer_blocks = 0;
    let mut total_ttft_ms = 0.0;

    for request_idx in 0..config.requests {
        let request = synthetic_request(request_idx, config);
        let residency = index.snapshot();
        let decision = policy.route(&request, &workers, &residency)?;

        cache_reusable_blocks += decision.reusable_blocks() as u64;
        recompute_blocks += decision.recomputes.len() as u64;
        transfer_blocks += decision.transfers.len() as u64;
        let ttft_ms = decision.estimated_ttft_us as f64 / 1_000.0;
        total_ttft_ms += ttft_ms;

        decisions.push(DecisionSummary {
            request_id: decision.request_id.clone(),
            worker_id: decision.worker_id.clone(),
            reusable_blocks: decision.reusable_blocks(),
            recompute_blocks: decision.recomputes.len(),
            transfer_blocks: decision.transfers.len(),
            estimated_ttft_ms: ttft_ms,
        });

        for block in request.blocks {
            index.put(CacheResidency::hbm(
                decision.worker_id.clone(),
                block,
                config.block_bytes,
            ));
        }
    }

    Ok(SimulationReport {
        policy: policy.name().to_string(),
        index_backend: index.name().to_string(),
        total_requests: config.requests,
        workers: config.workers,
        cache_reusable_blocks,
        recompute_blocks,
        transfer_blocks,
        avg_estimated_ttft_ms: if config.requests == 0 {
            0.0
        } else {
            total_ttft_ms / f64::from(config.requests)
        },
        index_metrics: index.metrics(),
        decisions,
    })
}

fn synthetic_request(request_idx: u32, config: SyntheticWorkloadConfig) -> RequestShape {
    let mut blocks = Vec::new();

    for block_idx in 0..config.shared_prefix_blocks {
        blocks.push(KvBlockKey::new(
            "llama-3.1-70b",
            "llama-tokenizer",
            "tenant-a",
            format!("prefix-{block_idx}"),
            format!("shared-{block_idx}"),
            block_idx,
            config.block_tokens,
        ));
    }

    for unique_idx in 0..config.unique_blocks_per_request {
        let block_index = config.shared_prefix_blocks + unique_idx;
        blocks.push(KvBlockKey::new(
            "llama-3.1-70b",
            "llama-tokenizer",
            "tenant-a",
            format!("req-{request_idx}-prefix-{unique_idx}"),
            format!("req-{request_idx}-unique-{unique_idx}"),
            block_index,
            config.block_tokens,
        ));
    }

    RequestShape {
        id: format!("req-{request_idx}"),
        model_id: "llama-3.1-70b".to_string(),
        tokenizer_id: "llama-tokenizer".to_string(),
        adapter_id: None,
        tenant_id: "tenant-a".to_string(),
        session_id: None,
        blocks,
        estimated_decode_tokens: 128,
        slo: SloTarget::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quillcache_router::LeastLoadedRouter;

    #[test]
    fn synthetic_workload_reuses_shared_prefix_after_first_request() {
        let report = run_synthetic(SyntheticWorkloadConfig {
            requests: 4,
            workers: 2,
            shared_prefix_blocks: 3,
            unique_blocks_per_request: 1,
            block_tokens: 64,
            block_bytes: 2 * 1024 * 1024,
        })
        .unwrap();

        assert_eq!(report.total_requests, 4);
        assert_eq!(report.policy, "greedy-state-plane");
        assert_eq!(report.index_backend, "memory");
        assert!(report.cache_reusable_blocks >= 3);
        assert_eq!(report.decisions.len(), 4);
    }

    #[test]
    fn experiment_mode_compares_policies_on_the_same_backend() {
        let config = SyntheticWorkloadConfig::default();
        let greedy = run_synthetic_with(
            &GreedyStatePlaneRouter::default(),
            &mut MemoryIndex::new(),
            config,
        )
        .unwrap();
        let least = run_synthetic_with(
            &LeastLoadedRouter::default(),
            &mut MemoryIndex::new(),
            config,
        )
        .unwrap();

        assert_eq!(greedy.policy, "greedy-state-plane");
        assert_eq!(least.policy, "least-loaded");
        assert_eq!(greedy.index_backend, "memory");
        assert_eq!(least.index_backend, "memory");
        // Both policies replay the full workload, so the backend sees the same
        // number of residency writes regardless of which worker was chosen.
        assert_eq!(greedy.index_metrics.puts, least.index_metrics.puts);
        assert_eq!(greedy.total_requests, least.total_requests);
    }
}
