use quillcache_core::{CacheResidency, CacheTier, CostModel, KvBlockKey, RequestShape, WorkerState};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RouterError {
    #[error("no workers are available")]
    NoWorkers,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferPlan {
    pub key: KvBlockKey,
    pub from_worker_id: String,
    pub tier: CacheTier,
    pub bytes: u64,
    pub estimated_us: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecomputePlan {
    pub key: KvBlockKey,
    pub estimated_us: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteDecision {
    pub request_id: String,
    pub worker_id: String,
    pub local_hits: Vec<KvBlockKey>,
    pub transfers: Vec<TransferPlan>,
    pub recomputes: Vec<RecomputePlan>,
    pub estimated_ttft_us: u64,
    pub estimated_tpot_us: u64,
    pub slo_violation_us: u64,
}

impl RouteDecision {
    pub fn reusable_blocks(&self) -> usize {
        self.local_hits.len() + self.transfers.len()
    }
}

/// A routing policy turns a request plus the current fleet and residency state
/// into a [`RouteDecision`]. It is the seam that lets Experiment mode compare
/// strategies — load-only baseline, cache-aware greedy, and future SLO- or
/// session-aware policies — on the same trace and the same index backend.
///
/// Policies differ only in *which* worker they select; per-worker cost
/// accounting is shared via [`plan_for_worker`], so comparisons stay apples to
/// apples.
pub trait RoutingPolicy {
    /// Stable policy name for reports (for example "greedy-state-plane").
    fn name(&self) -> &str;

    fn route(
        &self,
        request: &RequestShape,
        workers: &[WorkerState],
        residency: &[CacheResidency],
    ) -> Result<RouteDecision, RouterError>;
}

/// Plan a request against a single target worker: decide, per block, whether to
/// take a local HBM hit, transfer from another worker/tier, or recompute via
/// prefill, then estimate the resulting TTFT/TPOT and SLO violation. Shared by
/// every [`RoutingPolicy`] so policies differ only in which worker they pick.
pub fn plan_for_worker(
    cost_model: &CostModel,
    request: &RequestShape,
    target: &WorkerState,
    worker_by_id: &HashMap<&str, &WorkerState>,
    residency: &[CacheResidency],
) -> RouteDecision {
    let mut local_hits = Vec::new();
    let mut transfers = Vec::new();
    let mut recomputes = Vec::new();
    let mut prefill_or_transfer_us = 0;

    for block in &request.blocks {
        let recompute_us = cost_model.prefill_cost_us(block.token_count);
        let best_residency = residency
            .iter()
            .filter(|entry| entry.key == *block)
            .map(|entry| {
                let source = worker_by_id.get(entry.worker_id.as_str()).copied();
                let same_worker = entry.worker_id == target.id;
                let same_domain = source
                    .map(|source| source.locality_domain == target.locality_domain)
                    .unwrap_or(false);
                let transfer_us =
                    cost_model.transfer_cost_us(entry.tier, entry.bytes, same_worker, same_domain);
                (entry, transfer_us)
            })
            .min_by_key(|(_, transfer_us)| *transfer_us);

        match best_residency {
            Some((entry, transfer_us)) if transfer_us < recompute_us => {
                prefill_or_transfer_us += transfer_us;
                if entry.worker_id == target.id && entry.tier == CacheTier::Hbm {
                    local_hits.push(block.clone());
                } else {
                    transfers.push(TransferPlan {
                        key: block.clone(),
                        from_worker_id: entry.worker_id.clone(),
                        tier: entry.tier,
                        bytes: entry.bytes,
                        estimated_us: transfer_us,
                    });
                }
            }
            _ => {
                prefill_or_transfer_us += recompute_us;
                recomputes.push(RecomputePlan {
                    key: block.clone(),
                    estimated_us: recompute_us,
                });
            }
        }
    }

    let estimated_ttft_us = cost_model.queue_cost_us(target) + prefill_or_transfer_us;
    let estimated_tpot_us =
        cost_model.decode_cost_us(request.estimated_decode_tokens, target.running_decodes);
    let ttft_budget_us = request.slo.ttft_ms * 1_000;
    let tpot_budget_us = request.slo.tpot_ms * 1_000;
    let slo_violation_us = estimated_ttft_us.saturating_sub(ttft_budget_us)
        + estimated_tpot_us.saturating_sub(tpot_budget_us);

    RouteDecision {
        request_id: request.id.clone(),
        worker_id: target.id.clone(),
        local_hits,
        transfers,
        recomputes,
        estimated_ttft_us,
        estimated_tpot_us,
        slo_violation_us,
    }
}

/// Cache-aware policy: score every worker by estimated TTFT + TPOT + SLO
/// violation + decode pressure, and pick the best. This is the default online
/// policy and the main comparison target; it is not a final research claim, it
/// exists to make baselines and traces executable.
#[derive(Debug, Clone, Default)]
pub struct GreedyStatePlaneRouter {
    cost_model: CostModel,
}

impl GreedyStatePlaneRouter {
    pub fn new(cost_model: CostModel) -> Self {
        Self { cost_model }
    }

    pub fn route(
        &self,
        request: &RequestShape,
        workers: &[WorkerState],
        residency: &[CacheResidency],
    ) -> Result<RouteDecision, RouterError> {
        if workers.is_empty() {
            return Err(RouterError::NoWorkers);
        }

        let worker_by_id: HashMap<&str, &WorkerState> = workers
            .iter()
            .map(|worker| (worker.id.as_str(), worker))
            .collect();

        let mut best: Option<(u64, RouteDecision)> = None;
        for worker in workers {
            let decision =
                plan_for_worker(&self.cost_model, request, worker, &worker_by_id, residency);
            let score = decision.estimated_ttft_us
                + decision.estimated_tpot_us
                + decision.slo_violation_us * 4
                + u64::from(worker.running_decodes) * 1_000;

            if best
                .as_ref()
                .is_none_or(|(best_score, _)| score < *best_score)
            {
                best = Some((score, decision));
            }
        }

        Ok(best.expect("workers is not empty").1)
    }
}

impl RoutingPolicy for GreedyStatePlaneRouter {
    fn name(&self) -> &str {
        "greedy-state-plane"
    }

    fn route(
        &self,
        request: &RequestShape,
        workers: &[WorkerState],
        residency: &[CacheResidency],
    ) -> Result<RouteDecision, RouterError> {
        GreedyStatePlaneRouter::route(self, request, workers, residency)
    }
}

/// Load-only baseline: pick the worker with the least queue + decode pressure,
/// ignoring KV residency for the *choice*. It still reports honest
/// hit/transfer/recompute accounting for the chosen worker (via
/// [`plan_for_worker`]), so it is a fair "no cache awareness in routing"
/// baseline against [`GreedyStatePlaneRouter`].
#[derive(Debug, Clone, Default)]
pub struct LeastLoadedRouter {
    cost_model: CostModel,
}

impl LeastLoadedRouter {
    pub fn new(cost_model: CostModel) -> Self {
        Self { cost_model }
    }
}

impl RoutingPolicy for LeastLoadedRouter {
    fn name(&self) -> &str {
        "least-loaded"
    }

    fn route(
        &self,
        request: &RequestShape,
        workers: &[WorkerState],
        residency: &[CacheResidency],
    ) -> Result<RouteDecision, RouterError> {
        let target = workers
            .iter()
            .min_by_key(|worker| {
                u64::from(worker.queued_prefill_tokens) + u64::from(worker.running_decodes) * 1_000
            })
            .ok_or(RouterError::NoWorkers)?;
        let worker_by_id: HashMap<&str, &WorkerState> = workers
            .iter()
            .map(|worker| (worker.id.as_str(), worker))
            .collect();
        Ok(plan_for_worker(
            &self.cost_model,
            request,
            target,
            &worker_by_id,
            residency,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quillcache_core::{CacheResidency, KvBlockKey, SloTarget};

    fn request_with_shared_block() -> RequestShape {
        let block = KvBlockKey::new("llama", "tok", "tenant-a", "p0", "shared", 0, 128);
        RequestShape {
            id: "req-1".to_string(),
            model_id: "llama".to_string(),
            tokenizer_id: "tok".to_string(),
            adapter_id: None,
            tenant_id: "tenant-a".to_string(),
            blocks: vec![block],
            estimated_decode_tokens: 32,
            slo: SloTarget::default(),
        }
    }

    #[test]
    fn routes_to_worker_with_local_cache_hit() {
        let request = request_with_shared_block();
        let workers = vec![
            WorkerState::new("w0", "rack-a"),
            WorkerState::new("w1", "rack-a"),
        ];
        let residency = vec![CacheResidency::hbm(
            "w1",
            request.blocks[0].clone(),
            4 * 1024 * 1024,
        )];
        let decision = GreedyStatePlaneRouter::default()
            .route(&request, &workers, &residency)
            .unwrap();

        assert_eq!(decision.worker_id, "w1");
        assert_eq!(decision.local_hits.len(), 1);
        assert!(decision.recomputes.is_empty());
    }

    #[test]
    fn recomputes_when_cache_transfer_is_more_expensive() {
        let mut request = request_with_shared_block();
        request.blocks[0].token_count = 1;
        let workers = vec![WorkerState::new("w0", "rack-a")];
        let residency = vec![CacheResidency {
            key: request.blocks[0].clone(),
            worker_id: "cold-store".to_string(),
            tier: CacheTier::ObjectStore,
            bytes: 64 * 1024 * 1024,
            last_access_ms: 0,
            ref_count: 0,
            pinned: false,
        }];

        let decision = GreedyStatePlaneRouter::default()
            .route(&request, &workers, &residency)
            .unwrap();
        assert_eq!(decision.recomputes.len(), 1);
        assert_eq!(decision.transfers.len(), 0);
    }

    #[test]
    fn routing_policy_trait_is_object_safe_across_policies() {
        let request = request_with_shared_block();
        let workers = vec![
            WorkerState::new("w0", "rack-a"),
            WorkerState::new("w1", "rack-a"),
        ];
        let residency = vec![CacheResidency::hbm(
            "w1",
            request.blocks[0].clone(),
            4 * 1024 * 1024,
        )];

        let policies: Vec<Box<dyn RoutingPolicy>> = vec![
            Box::new(GreedyStatePlaneRouter::default()),
            Box::new(LeastLoadedRouter::default()),
        ];
        for policy in &policies {
            let decision = policy.route(&request, &workers, &residency).unwrap();
            assert!(!decision.worker_id.is_empty(), "{} produced no worker", policy.name());
        }

        // Cache-aware greedy must follow residency to w1; the load-only baseline
        // ignores residency and (with equal load) falls back to the first worker.
        assert_eq!(
            policies[0].route(&request, &workers, &residency).unwrap().worker_id,
            "w1"
        );
    }
}
