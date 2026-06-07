# Experiment Plan

The benchmark discipline is to measure routing and state management effects
separately from model-kernel improvements.

## Workload Axes

- short chat with repeated system prompts
- long-context chat with rolling conversation history
- RAG with repeated documents and changing query suffixes
- agent workflows with repeated tool schemas and multi-turn state
- mixed tenants with different cache sharing policies

## System Axes

- colocated prefill/decode
- disaggregated prefill/decode
- local HBM-only prefix cache
- external KV cache with CPU DRAM and SSD tiers
- remote KV pool with network-aware routing

## Metrics

- TTFT and P99 TTFT
- TPOT and P99 TPOT
- SLO goodput
- GPU idle time
- reusable token ratio
- reusable byte ratio
- bytes transferred per request
- recomputed prefill tokens
- cache pollution and eviction churn

## Baselines

- round-robin routing
- load-aware routing
- local prefix-cache-aware routing
- LMCache-style external cache
- Mooncake-style distributed KV pool
- memory residency index
- Holt ART residency index
- RocksDB/LSM residency index

## First Falsifiable Claim

For workloads with repeated prefixes or agent session state, a router that sees
KV residency can reduce estimated TTFT versus round-robin without increasing
estimated TPOT beyond the configured SLO.

## First Index Experiment

Compare Holt ART and RocksDB/LSM as residency-index backends using the same
`IndexBackend` contract:

- ingest-only lifecycle event stream
- mixed store/remove stream under HBM pressure
- prefix scan for repeated system prompts and RAG documents
- crash and restart with recovery timing
- index-size and write-amplification accounting

The expected output is a small table and two plots: p99 lookup/scan latency and
write amplification versus event rate.
