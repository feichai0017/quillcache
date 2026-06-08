use quillcache_core::{
    CacheResidency, CacheTier, CostModel, KvBlockKey, RequestShape, WorkerState,
};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
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
pub trait RoutingPolicy: std::fmt::Debug + Send + Sync {
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

/// Cache-affine policy ("approximate" prefix-aware routing): hash the request's
/// shared prefix to a worker, so every request carrying the same prefix lands on
/// the same engine and reuses its prefix cache — no KV events required. This is
/// the routing the cache-aware story rests on for a real multi-engine fleet.
#[derive(Debug, Clone, Default)]
pub struct PrefixAffinityRouter {
    cost_model: CostModel,
}

impl PrefixAffinityRouter {
    pub fn new(cost_model: CostModel) -> Self {
        Self { cost_model }
    }
}

impl RoutingPolicy for PrefixAffinityRouter {
    fn name(&self) -> &str {
        "prefix-affinity"
    }

    fn route(
        &self,
        request: &RequestShape,
        workers: &[WorkerState],
        residency: &[CacheResidency],
    ) -> Result<RouteDecision, RouterError> {
        if workers.is_empty() {
            return Err(RouterError::NoWorkers);
        }
        // Hash the longest shared prefix (the first block's prefix_hash) so that
        // requests sharing a system prompt / session map to the same worker.
        let affinity_key = request
            .blocks
            .first()
            .map(|block| block.prefix_hash.as_str())
            .unwrap_or(request.id.as_str());
        let mut hasher = DefaultHasher::new();
        affinity_key.hash(&mut hasher);
        let idx = (hasher.finish() % workers.len() as u64) as usize;
        let target = &workers[idx];
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

/// Spread baseline: round-robin across workers, ignoring prefix and cache — a
/// fair "no affinity" comparison for [`PrefixAffinityRouter`]. The same prefix is
/// scattered across the fleet, so each engine recomputes it.
#[derive(Debug, Default)]
pub struct RoundRobinRouter {
    cost_model: CostModel,
    next: AtomicUsize,
}

impl RoundRobinRouter {
    pub fn new(cost_model: CostModel) -> Self {
        Self {
            cost_model,
            next: AtomicUsize::new(0),
        }
    }
}

impl RoutingPolicy for RoundRobinRouter {
    fn name(&self) -> &str {
        "round-robin"
    }

    fn route(
        &self,
        request: &RequestShape,
        workers: &[WorkerState],
        residency: &[CacheResidency],
    ) -> Result<RouteDecision, RouterError> {
        if workers.is_empty() {
            return Err(RouterError::NoWorkers);
        }
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % workers.len();
        let target = &workers[idx];
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

/// SLO-aware, cache/session-affine policy: treat the TTFT/TPOT SLO as a
/// near-hard constraint. Among workers that **meet** the SLO budget, pick the one
/// with the most **local** cache hits — KV already resident in that engine's HBM
/// (true session affinity, no transfer), which with the closed residency loop is
/// the engine that already served this session — tie-breaking on latency. Only
/// when **no** worker can meet the SLO does it fall back to the least-violating.
///
/// This differs from [`GreedyStatePlaneRouter`], which blends latency and reuse
/// into one additive score and will pull a session's KV across engines (a cheap
/// intra-domain transfer) to chase marginally lower latency, even while the
/// engine holding it locally was meeting the SLO. SLO-aware keeps the session on
/// its warm engine — no KV movement, higher fleet-wide local hit rate — until
/// load actually threatens the SLO, then spills to protect tail latency.
#[derive(Debug, Clone, Default)]
pub struct SloAwareRouter {
    cost_model: CostModel,
}

impl SloAwareRouter {
    pub fn new(cost_model: CostModel) -> Self {
        Self { cost_model }
    }
}

impl RoutingPolicy for SloAwareRouter {
    fn name(&self) -> &str {
        "slo-aware"
    }

    fn route(
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
        let plans: Vec<RouteDecision> = workers
            .iter()
            .map(|worker| {
                plan_for_worker(&self.cost_model, request, worker, &worker_by_id, residency)
            })
            .collect();

        // Among SLO-feasible workers, maximize *local* cache hits — KV already in
        // that engine's HBM (true session affinity, zero transfer), not blocks it
        // would pull over the network — tie-breaking on lowest TTFT.
        let feasible = plans
            .iter()
            .filter(|decision| decision.slo_violation_us == 0)
            .max_by_key(|decision| {
                (
                    decision.local_hits.len(),
                    Reverse(decision.estimated_ttft_us),
                )
            });

        // If none can meet the SLO, take the least-violating (tie-break: most local hits).
        let chosen = feasible.or_else(|| {
            plans.iter().min_by_key(|decision| {
                (
                    decision.slo_violation_us,
                    Reverse(decision.local_hits.len()),
                )
            })
        });

        Ok(chosen.expect("plans is non-empty").clone())
    }
}

/// Session/DAG-affine policy for multi-turn and agentic workloads, where a
/// session reuses a *growing* prefix and rebuilding its context is expensive.
/// It prioritizes **session locality above load**: follow the session's KV to
/// whichever engine already holds the most of it (most local hits); on a cold
/// session, pin it to a deterministic home engine by hashing the `session_id`
/// (or the conversation root prefix). With the closed residency loop, turn 1
/// pins the session and writes its blocks home, and every later turn finds them
/// resident there and sticks — the whole session's KV accumulates on one engine.
///
/// Unlike [`SloAwareRouter`] (which spills a session off its engine under SLO
/// pressure) or [`GreedyStatePlaneRouter`] (which scatters by latency), this
/// keeps a session pinned, trading per-request load balance for maximal context
/// reuse — the right call when recomputing a long agent history dominates.
#[derive(Debug, Clone, Default)]
pub struct SessionAffinityRouter {
    cost_model: CostModel,
}

impl SessionAffinityRouter {
    pub fn new(cost_model: CostModel) -> Self {
        Self { cost_model }
    }
}

impl RoutingPolicy for SessionAffinityRouter {
    fn name(&self) -> &str {
        "session-affinity"
    }

    fn route(
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

        // Warm session: follow its KV to the engine holding the most of it.
        let warmest = workers
            .iter()
            .map(|worker| {
                plan_for_worker(&self.cost_model, request, worker, &worker_by_id, residency)
            })
            .max_by_key(|decision| decision.local_hits.len());
        if let Some(decision) = warmest {
            if !decision.local_hits.is_empty() {
                return Ok(decision);
            }
        }

        // Cold session: pin to a deterministic home by session id (or the
        // conversation root prefix), so all of its turns land together.
        let key = request
            .session_id
            .as_deref()
            .or_else(|| {
                request
                    .blocks
                    .first()
                    .map(|block| block.prefix_hash.as_str())
            })
            .unwrap_or(request.id.as_str());
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let idx = (hasher.finish() % workers.len() as u64) as usize;
        Ok(plan_for_worker(
            &self.cost_model,
            request,
            &workers[idx],
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
            session_id: None,
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
            assert!(
                !decision.worker_id.is_empty(),
                "{} produced no worker",
                policy.name()
            );
        }

        // Cache-aware greedy must follow residency to w1; the load-only baseline
        // ignores residency and (with equal load) falls back to the first worker.
        assert_eq!(
            policies[0]
                .route(&request, &workers, &residency)
                .unwrap()
                .worker_id,
            "w1"
        );
    }

    #[test]
    fn slo_aware_keeps_affinity_while_greedy_chases_latency() {
        // w0 is cold but fast; w1 holds the session's block but carries modest
        // queue load, making it marginally slower — yet still within the SLO.
        let request = request_with_shared_block();
        let workers = vec![
            WorkerState::new("w0", "rack-a"),
            WorkerState::new("w1", "rack-a").with_load(2_000, 0),
        ];
        let residency = vec![CacheResidency::hbm(
            "w1",
            request.blocks[0].clone(),
            4 * 1024 * 1024,
        )];

        // SLO-aware keeps the session local on the warm engine (both meet the
        // SLO; w1 has the KV in HBM, zero transfer)...
        let slo = SloAwareRouter::default()
            .route(&request, &workers, &residency)
            .unwrap();
        assert_eq!(slo.worker_id, "w1");
        assert_eq!(slo.local_hits.len(), 1);

        // ...while greedy pulls the KV to the faster cold engine (cheap
        // intra-domain transfer), moving the session off its warm engine.
        let greedy = GreedyStatePlaneRouter::default()
            .route(&request, &workers, &residency)
            .unwrap();
        assert_eq!(greedy.worker_id, "w0");
        assert_eq!(greedy.transfers.len(), 1);
    }

    #[test]
    fn slo_aware_spills_off_a_warm_engine_that_would_violate_slo() {
        // w1 holds the block but is badly overloaded — its queue blows the TTFT
        // SLO — so the guard must spill to the cold-but-feasible w0.
        let request = request_with_shared_block();
        let workers = vec![
            WorkerState::new("w0", "rack-a"),
            WorkerState::new("w1", "rack-a").with_load(250_000, 0),
        ];
        let residency = vec![CacheResidency::hbm(
            "w1",
            request.blocks[0].clone(),
            4 * 1024 * 1024,
        )];

        let decision = SloAwareRouter::default()
            .route(&request, &workers, &residency)
            .unwrap();
        assert_eq!(decision.worker_id, "w0");
        assert!(decision.recomputes.len() == 1 || !decision.transfers.is_empty());
    }

    #[test]
    fn session_affinity_pins_cold_then_follows_warm() {
        let workers = vec![
            WorkerState::new("w0", "rack-a"),
            WorkerState::new("w1", "rack-a"),
        ];
        let router = SessionAffinityRouter::default();

        // Cold session: pinned to a deterministic home by session id, and every
        // turn of the same session lands there (no residency yet).
        let mut turn1 = request_with_shared_block();
        turn1.session_id = Some("session-42".to_string());
        let home = router.route(&turn1, &workers, &[]).unwrap().worker_id;
        let mut turn1b = turn1.clone();
        turn1b.id = "req-1b".to_string();
        assert_eq!(
            router.route(&turn1b, &workers, &[]).unwrap().worker_id,
            home
        );

        // Warm: the session's KV is now resident on the *other* engine — session
        // affinity follows the KV there, even against the cold hash home.
        let other = if home == "w0" { "w1" } else { "w0" };
        let residency = vec![CacheResidency::hbm(
            other,
            turn1.blocks[0].clone(),
            4 * 1024 * 1024,
        )];
        let decision = router.route(&turn1, &workers, &residency).unwrap();
        assert_eq!(decision.worker_id, other);
        assert_eq!(decision.local_hits.len(), 1);
    }
}
