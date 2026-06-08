//! Tiered KV block management — a KVBM-style multi-tier cache.
//!
//! Mirrors the tiered block managers of production stacks (NVIDIA Dynamo KVBM's
//! G1–G4, Tencent FlexKV's StorageEngine): KV blocks live in HBM → DRAM → SSD.
//! Under capacity pressure a tier demotes its coldest block to the next tier
//! down (and the bottom tier evicts); on reuse a block is promoted back toward
//! HBM. The persistent ART index is the natural cross-tier catalog.
//!
//! The experiment replays a skewed access trace through the tiered cache and an
//! HBM-only baseline on the *same* trace, to quantify the win: how many
//! recomputes tiering turns into cheap tier-hits, and the prefill cost saved.

use quillcache_core::{CacheTier, CostModel};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TieredConfig {
    /// Distinct blocks in the working set.
    pub blocks: u32,
    /// Total accesses replayed.
    pub accesses: u32,
    /// HBM capacity, in blocks (the hot tier).
    pub hbm_blocks: u32,
    /// DRAM capacity, in blocks.
    pub dram_blocks: u32,
    /// SSD capacity, in blocks.
    pub ssd_blocks: u32,
    /// Percent of blocks that are "hot" (receive ~80% of accesses).
    pub hot_percent: u32,
    pub block_tokens: u32,
    pub block_bytes: u64,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            blocks: 4_000,
            accesses: 40_000,
            hbm_blocks: 200,
            dram_blocks: 800,
            ssd_blocks: 4_000,
            hot_percent: 10,
            block_tokens: 256,
            block_bytes: 2 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TieredReport {
    pub accesses: u64,
    pub hbm_hits: u64,
    pub dram_hits: u64,
    pub ssd_hits: u64,
    /// Misses = blocks not resident in any tier (a full prefill recompute).
    pub misses: u64,
    pub promotions: u64,
    pub demotions: u64,
    pub evictions: u64,
    /// Total cache capacity across tiers — the "effective cache size".
    pub effective_cache_blocks: u64,
    /// Total access cost (HBM hit / tier transfer / recompute), in ms.
    pub total_cost_ms: f64,
    /// Counterfactual: the same trace through an HBM-only cache of the same HBM
    /// size — everything beyond HBM is recomputed.
    pub hbm_only_cost_ms: f64,
    pub hbm_only_misses: u64,
    /// Cost reduction tiering buys over HBM-only.
    pub cost_saved_pct: f64,
    /// Recomputes tiering avoided vs HBM-only (turned into cheap tier-hits).
    pub recomputes_avoided: u64,
}

/// A multi-tier LRU cache: tier 0 is HBM, then DRAM, then SSD. A block lives in
/// exactly one tier; promotion moves it to HBM, overflow cascades demotions down
/// and evicts off the bottom.
struct TieredCache {
    caps: Vec<usize>,
    lru: Vec<VecDeque<u32>>,
    tier_of: HashMap<u32, usize>,
    promotions: u64,
    demotions: u64,
    evictions: u64,
}

impl TieredCache {
    fn new(caps: Vec<usize>) -> Self {
        let lru = caps.iter().map(|_| VecDeque::new()).collect();
        Self {
            caps,
            lru,
            tier_of: HashMap::new(),
            promotions: 0,
            demotions: 0,
            evictions: 0,
        }
    }

    /// Access `block`. Returns the tier it was found in (`None` = miss). Either
    /// way the block ends up at HBM (tier 0) MRU, with overflow cascaded down.
    fn access(&mut self, block: u32) -> Option<usize> {
        let found = self.tier_of.remove(&block);
        if let Some(tier) = found {
            if let Some(pos) = self.lru[tier].iter().position(|&b| b == block) {
                self.lru[tier].remove(pos);
            }
            if tier > 0 {
                self.promotions += 1;
            }
        }
        // (Re)insert at HBM MRU.
        self.lru[0].push_back(block);
        self.tier_of.insert(block, 0);
        self.cascade();
        found
    }

    /// Push overflow down the tiers; evict off the bottom.
    fn cascade(&mut self) {
        for tier in 0..self.caps.len() {
            while self.lru[tier].len() > self.caps[tier] {
                let Some(victim) = self.lru[tier].pop_front() else {
                    break;
                };
                if tier + 1 < self.caps.len() {
                    self.lru[tier + 1].push_back(victim);
                    self.tier_of.insert(victim, tier + 1);
                    self.demotions += 1;
                } else {
                    self.tier_of.remove(&victim);
                    self.evictions += 1;
                }
            }
        }
    }
}

/// Deterministic skewed access trace with temporal locality (the realistic case
/// LRU/tiering exploits): ~80% of accesses fall in the hot region with a
/// Zipf-like concentration on the hottest blocks, ~20% sweep the larger cold
/// region. The hot region is bigger than HBM, so the hottest blocks stay in HBM
/// while the warm-hot ones live in DRAM and the cold tail in SSD. A fixed seed
/// keeps it reproducible. (`a % hot` would be a sequential scan — LRU's
/// worst case — so we sample instead.)
fn trace(config: &TieredConfig) -> Vec<u32> {
    let blocks = config.blocks.max(1);
    let hot = (blocks * config.hot_percent / 100).clamp(1, blocks);
    let cold = (blocks - hot).max(1);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rng = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..config.accesses)
        .map(|_| {
            if rng() % 100 < 80 {
                // Hot region, Zipf-like: block = hot^u, u in [0,1) — concentrates
                // accesses on the lowest (hottest) indices, giving LRU locality.
                let u = (rng() >> 11) as f64 / (1u64 << 53) as f64;
                (f64::from(hot).powf(u) as u32)
                    .saturating_sub(1)
                    .min(hot - 1)
            } else {
                // Cold region: a uniform sweep over the long tail.
                hot + (rng() % u64::from(cold)) as u32
            }
        })
        .collect()
}

/// Run the tiered-vs-HBM-only experiment.
pub fn run_tiered(config: TieredConfig) -> TieredReport {
    let cost = CostModel::default();
    let recompute_us = cost.prefill_cost_us(config.block_tokens) as f64;
    let tier_cost =
        |tier: CacheTier| cost.transfer_cost_us(tier, config.block_bytes, true, true) as f64;
    let hbm_us = cost.transfer_cost_us(CacheTier::Hbm, config.block_bytes, true, true) as f64;
    let dram_us = tier_cost(CacheTier::CpuDram);
    let ssd_us = tier_cost(CacheTier::LocalSsd);

    let accesses = trace(&config);

    // Tiered cache: HBM -> DRAM -> SSD.
    let mut tiered = TieredCache::new(vec![
        config.hbm_blocks as usize,
        config.dram_blocks as usize,
        config.ssd_blocks as usize,
    ]);
    let (mut hbm_hits, mut dram_hits, mut ssd_hits, mut misses) = (0u64, 0u64, 0u64, 0u64);
    let mut total_us = 0.0;
    for &block in &accesses {
        match tiered.access(block) {
            Some(0) => {
                hbm_hits += 1;
                total_us += hbm_us;
            }
            Some(1) => {
                dram_hits += 1;
                total_us += dram_us;
            }
            Some(_) => {
                ssd_hits += 1;
                total_us += ssd_us;
            }
            None => {
                misses += 1;
                total_us += recompute_us;
            }
        }
    }

    // HBM-only baseline on the same trace.
    let mut hbm_only = TieredCache::new(vec![config.hbm_blocks as usize]);
    let (mut hbm_only_misses, mut hbm_only_us) = (0u64, 0.0);
    for &block in &accesses {
        match hbm_only.access(block) {
            Some(_) => hbm_only_us += hbm_us,
            None => {
                hbm_only_misses += 1;
                hbm_only_us += recompute_us;
            }
        }
    }

    let cost_saved_pct = if hbm_only_us > 0.0 {
        100.0 * (hbm_only_us - total_us) / hbm_only_us
    } else {
        0.0
    };

    TieredReport {
        accesses: accesses.len() as u64,
        hbm_hits,
        dram_hits,
        ssd_hits,
        misses,
        promotions: tiered.promotions,
        demotions: tiered.demotions,
        evictions: tiered.evictions,
        effective_cache_blocks: u64::from(config.hbm_blocks)
            + u64::from(config.dram_blocks)
            + u64::from(config.ssd_blocks),
        total_cost_ms: total_us / 1_000.0,
        hbm_only_cost_ms: hbm_only_us / 1_000.0,
        hbm_only_misses,
        cost_saved_pct,
        recomputes_avoided: hbm_only_misses.saturating_sub(misses),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiering_turns_recomputes_into_tier_hits() {
        let report = run_tiered(TieredConfig::default());

        assert_eq!(report.accesses, 40_000);
        // The lower tiers catch reuse the HBM-only cache would recompute.
        assert!(report.dram_hits + report.ssd_hits > 0);
        assert!(
            report.misses < report.hbm_only_misses,
            "tiering should recompute less: {} vs {}",
            report.misses,
            report.hbm_only_misses
        );
        assert!(report.recomputes_avoided > 0);
        // A recompute is far costlier than a tier transfer, so tiering wins.
        assert!(
            report.cost_saved_pct > 0.0,
            "expected positive saving, got {}%",
            report.cost_saved_pct
        );
        // Effective cache spans all tiers.
        assert_eq!(report.effective_cache_blocks, 200 + 800 + 4_000);
    }

    #[test]
    fn promotions_and_demotions_happen_under_pressure() {
        let report = run_tiered(TieredConfig {
            blocks: 1_000,
            accesses: 5_000,
            hbm_blocks: 20,
            dram_blocks: 80,
            ssd_blocks: 1_000,
            hot_percent: 10,
            block_tokens: 64,
            block_bytes: 1024 * 1024,
        });
        // Cold blocks accessed from lower tiers get promoted; HBM overflow demotes.
        assert!(report.promotions > 0);
        assert!(report.demotions > 0);
    }
}
