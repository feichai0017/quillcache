//! Hot-prefix replication scheduling — the global cache-balancing piece the
//! Mooncake Conductor does that a per-request cost router alone cannot.
//!
//! The cost router sends each request to the lowest-cost worker (cache locality
//! vs load). But when one prefix becomes **hot** — a shared system prompt, a
//! viral few-shot preamble — every request for it is drawn to the *single* worker
//! that holds its KV. Cache affinity turns that worker into a hotspot while peers
//! idle. The fix is to **replicate the hot prefix's KV to more workers** so the
//! load spreads while every request still hits cache.
//!
//! This module decides *which* hot prefixes to replicate to *which* workers; the
//! actual byte copy is the Transfer Engine's job (and, intra-node, the zero-copy
//! peer path in `quillcache-transfer-engine`). The planner is a pure function of
//! (hot prefixes + where they live) × (worker load), so it is deterministic and
//! unit-testable in isolation.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::{IdentityScope, RequestShape};

/// How aggressively to replicate hot prefixes.
#[derive(Debug, Clone)]
pub struct ReplicationConfig {
    /// Replicate a hot prefix until at least this many workers hold it.
    pub target_replicas: usize,
    /// A prefix counts as hot once its access count reaches this.
    pub hotness_threshold: u32,
    /// Cap on actions emitted per planning pass (backpressure on the copies).
    pub max_actions: usize,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            target_replicas: 2,
            hotness_threshold: 8,
            max_actions: 32,
        }
    }
}

/// A hot prefix and the workers currently holding its KV (in HBM).
#[derive(Debug, Clone)]
pub struct PrefixResidency {
    pub scope: IdentityScope,
    pub prefix_hash: String,
    pub holders: Vec<String>,
    pub accesses: u32,
}

/// A worker's current load — lower is a better replication target.
#[derive(Debug, Clone)]
pub struct WorkerLoad {
    pub id: String,
    pub load: u32,
}

/// "Copy this prefix's KV from a worker that holds it to one that doesn't, to
/// spread the load on a hotspot." The execution worker is the Transfer Engine's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationAction {
    pub scope: IdentityScope,
    pub prefix_hash: String,
    pub from_worker: String,
    pub to_worker: String,
    pub accesses: u32,
}

/// Decide replications: for each hot, under-replicated prefix (hottest first),
/// copy it from its least-loaded current holder to the least-loaded workers that
/// don't already hold it, until it reaches `target_replicas` or workers run out.
/// Deterministic: ties break by worker id / prefix hash.
pub fn plan_replications(
    prefixes: &[PrefixResidency],
    workers: &[WorkerLoad],
    cfg: &ReplicationConfig,
) -> Vec<ReplicationAction> {
    // Workers ascending by load (least-loaded first); stable by id on ties.
    let mut by_load: Vec<&WorkerLoad> = workers.iter().collect();
    by_load.sort_by(|a, b| a.load.cmp(&b.load).then_with(|| a.id.cmp(&b.id)));

    // Hot + under-replicated + actually resident somewhere; hottest first.
    let mut hot: Vec<&PrefixResidency> = prefixes
        .iter()
        .filter(|p| {
            p.accesses >= cfg.hotness_threshold
                && !p.holders.is_empty()
                && p.holders.len() < cfg.target_replicas
        })
        .collect();
    hot.sort_by(|a, b| {
        b.accesses
            .cmp(&a.accesses)
            .then_with(|| a.prefix_hash.cmp(&b.prefix_hash))
    });

    let mut actions = Vec::new();
    for p in hot {
        if actions.len() >= cfg.max_actions {
            break;
        }
        let holders: HashSet<&str> = p.holders.iter().map(String::as_str).collect();
        // Source: the least-loaded worker that already holds it.
        let from = by_load
            .iter()
            .find(|w| holders.contains(w.id.as_str()))
            .map(|w| w.id.clone())
            .unwrap_or_else(|| p.holders[0].clone());

        let need = cfg.target_replicas - p.holders.len();
        let mut added = 0;
        for w in &by_load {
            if added >= need || actions.len() >= cfg.max_actions {
                break;
            }
            if holders.contains(w.id.as_str()) {
                continue; // already has it
            }
            actions.push(ReplicationAction {
                scope: p.scope.clone(),
                prefix_hash: p.prefix_hash.clone(),
                from_worker: from.clone(),
                to_worker: w.id.clone(),
                accesses: p.accesses,
            });
            added += 1;
        }
    }
    actions
}

/// Counts how often each (identity, prefix) is requested, so the planner can spot
/// hot prefixes. Interior-mutable (the request path holds `&ControlPlane`); a
/// periodic [`Self::decay`] keeps the signal recent rather than all-time.
#[derive(Debug, Default)]
pub struct HotnessTracker {
    counts: Mutex<HashMap<(IdentityScope, String), u32>>,
}

impl HotnessTracker {
    /// Count one access per distinct (identity, prefix) the request touches.
    pub fn record_request(&self, request: &RequestShape) {
        if request.blocks.is_empty() {
            return;
        }
        let mut counts = self.counts.lock().unwrap();
        let mut seen = HashSet::new();
        for block in &request.blocks {
            let key = (IdentityScope::from_key(block), block.prefix_hash.clone());
            if seen.insert(key.clone()) {
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    /// Prefixes at or above `threshold` accesses, as (identity, prefix, accesses).
    pub fn hot(&self, threshold: u32) -> Vec<(IdentityScope, String, u32)> {
        let counts = self.counts.lock().unwrap();
        counts
            .iter()
            .filter(|(_, &n)| n >= threshold)
            .map(|((scope, prefix), &n)| (scope.clone(), prefix.clone(), n))
            .collect()
    }

    /// Halve every count (call periodically) so recently-hot prefixes dominate;
    /// drops anything that decays to zero.
    pub fn decay(&self) {
        let mut counts = self.counts.lock().unwrap();
        counts.retain(|_, n| {
            *n /= 2;
            *n > 0
        });
    }

    /// Number of distinct (identity, prefix) currently tracked.
    pub fn tracked(&self) -> usize {
        self.counts.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> IdentityScope {
        IdentityScope {
            model_id: "m".into(),
            tokenizer_id: "t".into(),
            adapter_id: None,
            tenant_id: "default".into(),
        }
    }

    fn worker(id: &str, load: u32) -> WorkerLoad {
        WorkerLoad {
            id: id.into(),
            load,
        }
    }

    fn prefix(hash: &str, holders: &[&str], accesses: u32) -> PrefixResidency {
        PrefixResidency {
            scope: scope(),
            prefix_hash: hash.into(),
            holders: holders.iter().map(|s| s.to_string()).collect(),
            accesses,
        }
    }

    #[test]
    fn replicates_hot_singly_held_prefix_to_least_loaded_peer() {
        let cfg = ReplicationConfig {
            target_replicas: 2,
            hotness_threshold: 8,
            max_actions: 32,
        };
        let prefixes = [prefix("p1", &["w1"], 20)];
        let workers = [worker("w1", 9), worker("w2", 5), worker("w3", 2)];
        let actions = plan_replications(&prefixes, &workers, &cfg);

        // One copy needed (1 holder -> target 2), to the least-loaded non-holder (w3).
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].from_worker, "w1");
        assert_eq!(actions[0].to_worker, "w3");
        assert_eq!(actions[0].prefix_hash, "p1");
    }

    #[test]
    fn cold_prefix_is_not_replicated() {
        let cfg = ReplicationConfig::default();
        let prefixes = [prefix("cold", &["w1"], 3)]; // below threshold 8
        let workers = [worker("w1", 1), worker("w2", 0)];
        assert!(plan_replications(&prefixes, &workers, &cfg).is_empty());
    }

    #[test]
    fn already_replicated_prefix_is_left_alone() {
        let cfg = ReplicationConfig::default();
        let prefixes = [prefix("hot", &["w1", "w2"], 50)]; // already 2 holders == target
        let workers = [worker("w1", 9), worker("w2", 9), worker("w3", 0)];
        assert!(plan_replications(&prefixes, &workers, &cfg).is_empty());
    }

    #[test]
    fn no_spare_worker_means_no_action() {
        let cfg = ReplicationConfig::default();
        let prefixes = [prefix("hot", &["w1"], 50)];
        let workers = [worker("w1", 0)]; // the only worker already holds it
        assert!(plan_replications(&prefixes, &workers, &cfg).is_empty());
    }

    #[test]
    fn higher_replication_factor_emits_multiple_copies_least_loaded_first() {
        let cfg = ReplicationConfig {
            target_replicas: 3,
            hotness_threshold: 8,
            max_actions: 32,
        };
        let prefixes = [prefix("p", &["w1"], 40)];
        let workers = [
            worker("w1", 9),
            worker("w2", 7),
            worker("w3", 1),
            worker("w4", 4),
        ];
        let actions = plan_replications(&prefixes, &workers, &cfg);
        // Need 2 more holders; least-loaded non-holders are w3(1) then w4(4).
        let targets: Vec<&str> = actions.iter().map(|a| a.to_worker.as_str()).collect();
        assert_eq!(targets, vec!["w3", "w4"]);
    }

    #[test]
    fn max_actions_caps_the_pass() {
        let cfg = ReplicationConfig {
            target_replicas: 4,
            hotness_threshold: 8,
            max_actions: 1,
        };
        let prefixes = [prefix("a", &["w1"], 99), prefix("b", &["w1"], 98)];
        let workers = [worker("w1", 0), worker("w2", 0), worker("w3", 0)];
        assert_eq!(plan_replications(&prefixes, &workers, &cfg).len(), 1);
    }

    #[test]
    fn hotness_tracker_counts_distinct_prefixes_and_decays() {
        use crate::{KvBlockKey, SloTarget};
        let block = |prefix: &str| KvBlockKey {
            model_id: "m".into(),
            tokenizer_id: "t".into(),
            adapter_id: None,
            tenant_id: "default".into(),
            prefix_hash: prefix.into(),
            block_hash: format!("{prefix}-blk"),
            block_index: 0,
            token_count: 16,
        };
        let req = |prefix: &str| RequestShape {
            id: "r".into(),
            model_id: "m".into(),
            tokenizer_id: "t".into(),
            adapter_id: None,
            tenant_id: "default".into(),
            session_id: None,
            blocks: vec![block(prefix)],
            estimated_decode_tokens: 1,
            slo: SloTarget::default(),
        };
        let t = HotnessTracker::default();
        for _ in 0..10 {
            t.record_request(&req("hot"));
        }
        t.record_request(&req("warm"));
        assert_eq!(t.tracked(), 2);
        let hot = t.hot(8);
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0].1, "hot");
        assert_eq!(hot[0].2, 10);
        t.decay(); // 10 -> 5, 1 -> 0 (dropped)
        assert_eq!(t.tracked(), 1);
        assert!(t.hot(8).is_empty());
    }
}
