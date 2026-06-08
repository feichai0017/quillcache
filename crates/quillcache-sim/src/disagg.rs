//! Prefill/decode (PD) disaggregation — the headline topology of NVIDIA Dynamo
//! and llm-d.
//!
//! In *aggregated* serving each engine does both prefill and decode, so an
//! engine tied up in a long (autoregressive) decode can't start the prefill of a
//! newly arrived request — time-to-first-token (TTFT) waits behind decode work.
//! *Disaggregated* serving splits the fleet into a prefill pool and a decode pool
//! (KV is transferred between them), so the prefill pool only ever does short
//! prefills: same utilization, much shorter service time, so far shorter waits.
//!
//! Steady-state *throughput* is similar either way (both saturate the GPUs); the
//! win is **TTFT under load**. This is a discrete-event simulation: requests
//! arrive over time at a chosen fraction of fleet throughput and are assigned to
//! the engine that frees soonest; we measure TTFT (arrival → first token) under
//! both topologies on the same arrival stream.

use quillcache_core::CostModel;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisaggConfig {
    /// Requests in the arrival stream.
    pub requests: u32,
    /// Total engines (GPUs) in the fleet.
    pub engines: u32,
    /// Prompt tokens to prefill per request.
    pub prefill_tokens: u32,
    /// Tokens to decode per request (chat-style defaults are decode-dominant).
    pub decode_tokens: u32,
    /// Prefill engines for the split; `0` auto-balances P:D to the work ratio.
    pub prefill_engines: u32,
    /// Offered load as a percent of fleet throughput (queueing pressure).
    pub load_percent: u32,
}

impl Default for DisaggConfig {
    fn default() -> Self {
        Self {
            requests: 2_000,
            engines: 8,
            prefill_tokens: 512,
            decode_tokens: 1_024,
            prefill_engines: 0,
            load_percent: 75,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisaggReport {
    pub requests: u64,
    pub engines: u64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub load_percent: u64,
    pub agg_ttft_p50_ms: f64,
    pub agg_ttft_p99_ms: f64,
    pub disagg_prefill_engines: u64,
    pub disagg_decode_engines: u64,
    pub disagg_ttft_p50_ms: f64,
    pub disagg_ttft_p99_ms: f64,
    /// p99 TTFT reduction disaggregation buys.
    pub ttft_p99_reduction_pct: f64,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (((sorted.len() - 1) as f64) * p).round() as usize;
    sorted[idx]
}

fn argmin(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v < values[best] {
            best = i;
        }
    }
    best
}

/// Simulate a pool of FIFO servers over an arrival stream, assigning each
/// arrival to the engine that frees soonest. An engine is *occupied* for
/// `occupancy_us` per job (prefill+decode when aggregated; prefill only for a
/// prefill pool), but the first token — what TTFT measures — lands after
/// `ttft_us`. Returns each job's TTFT (`start + ttft_us - arrival`); the wait
/// differs because a request can be stuck behind an engine busy with decode.
fn simulate_ttft(arrivals: &[f64], pool: usize, occupancy_us: f64, ttft_us: f64) -> Vec<f64> {
    let mut free = vec![0.0f64; pool.max(1)];
    arrivals
        .iter()
        .map(|&arrival| {
            let e = argmin(&free);
            let start = arrival.max(free[e]);
            free[e] = start + occupancy_us;
            start + ttft_us - arrival
        })
        .collect()
}

/// Run the aggregated-vs-disaggregated TTFT experiment.
pub fn run_disagg(config: DisaggConfig) -> DisaggReport {
    let cost = CostModel::default();
    let prefill_us = cost.prefill_cost_us(config.prefill_tokens) as f64;
    let decode_us = cost.decode_cost_us(config.decode_tokens, 0) as f64;
    let engines = config.engines.max(1) as usize;
    let requests = config.requests.max(1) as usize;

    // Fleet throughput (req/us) and the arrival rate at the offered load.
    let throughput = engines as f64 / (prefill_us + decode_us);
    let lambda = (config.load_percent.max(1) as f64 / 100.0) * throughput;
    // Poisson arrivals (exponential inter-arrival) — real traffic is bursty, and
    // bursts are what build queues. A fixed seed keeps it reproducible.
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next_u = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 53) as f64).max(1e-12)
    };
    let mut t = 0.0;
    let arrivals: Vec<f64> = (0..requests)
        .map(|_| {
            t += -next_u().ln() / lambda;
            t
        })
        .collect();

    // Aggregated: each engine is occupied prefill+decode per request, but the
    // first token lands after prefill. TTFT waits behind whatever (including long
    // decodes) is occupying the engine.
    let agg = simulate_ttft(&arrivals, engines, prefill_us + decode_us, prefill_us);

    // Disaggregated: a prefill pool sized to the work ratio (decode-dominant ->
    // fewer prefill engines), serving only short prefills.
    let p = if config.prefill_engines > 0 {
        (config.prefill_engines as usize).clamp(1, engines.saturating_sub(1).max(1))
    } else {
        let ratio = prefill_us / (prefill_us + decode_us);
        ((engines as f64 * ratio).round() as usize).clamp(1, engines - 1)
    };
    let d = (engines - p).max(1);
    // The prefill pool is occupied only by short prefills, so it frees fast.
    let dis = simulate_ttft(&arrivals, p, prefill_us, prefill_us);

    let mut agg_s = agg.clone();
    let mut dis_s = dis.clone();
    agg_s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dis_s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ms = |us: f64| us / 1_000.0;
    let agg_p99 = percentile(&agg_s, 0.99);
    let dis_p99 = percentile(&dis_s, 0.99);

    DisaggReport {
        requests: requests as u64,
        engines: engines as u64,
        prefill_ms: ms(prefill_us),
        decode_ms: ms(decode_us),
        load_percent: u64::from(config.load_percent),
        agg_ttft_p50_ms: ms(percentile(&agg_s, 0.50)),
        agg_ttft_p99_ms: ms(agg_p99),
        disagg_prefill_engines: p as u64,
        disagg_decode_engines: d as u64,
        disagg_ttft_p50_ms: ms(percentile(&dis_s, 0.50)),
        disagg_ttft_p99_ms: ms(dis_p99),
        ttft_p99_reduction_pct: if agg_p99 > 0.0 {
            100.0 * (agg_p99 - dis_p99) / agg_p99
        } else {
            0.0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disaggregation_cuts_ttft_when_decode_dominates() {
        let report = run_disagg(DisaggConfig::default());
        assert_eq!(report.requests, 2_000);
        assert_eq!(
            report.disagg_prefill_engines + report.disagg_decode_engines,
            report.engines
        );
        // Decode-dominant: the prefill pool is the smaller one.
        assert!(report.disagg_prefill_engines < report.disagg_decode_engines);
        // The headline: disaggregation lowers p99 TTFT under load, because the
        // prefill pool never waits behind a long decode.
        assert!(
            report.disagg_ttft_p99_ms < report.agg_ttft_p99_ms,
            "disagg p99 {} should beat agg p99 {}",
            report.disagg_ttft_p99_ms,
            report.agg_ttft_p99_ms
        );
        assert!(report.ttft_p99_reduction_pct > 10.0);
    }

    #[test]
    fn auto_balance_respects_an_explicit_split() {
        let report = run_disagg(DisaggConfig {
            prefill_engines: 3,
            ..DisaggConfig::default()
        });
        assert_eq!(report.disagg_prefill_engines, 3);
        assert_eq!(report.disagg_decode_engines, report.engines - 3);
    }
}
