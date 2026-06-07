# Research Agenda

## Problem

LLM serving is becoming stateful. KV cache is no longer an internal tensor
buffer owned by one engine replica; it is a large, expensive, reusable inference
state object that affects routing, placement, transfer, recompute, and SLOs.

## Claim Budget

Guaranteed properties:

- KV reuse decisions must include model, tokenizer, adapter, and tenant identity.
- Unsupported or unsafe reuse must fall back to recompute.

Measured effects:

- TTFT improvement from cache-aware routing.
- SLO goodput improvement from tiered placement.
- Bytes moved versus tokens recomputed under different network and storage tiers.

Design hypotheses:

- Agent workflows need session/DAG-aware KV policies, not only prefix LRU.
- Network topology belongs in decode-worker selection once prefill/decode is
  disaggregated.
- Adaptive compression should be chosen by service context, not as a static
  deployment flag.

Non-goals:

- custom attention kernels
- model weight runtime
- ANN indexing
- SQL execution

## Build Order

1. Define KV block and residency model.
2. Build a trace simulator with synthetic shared-prefix workloads.
3. Implement round-robin and greedy cache-aware baselines.
4. Add SLO-aware scoring.
5. Add network-aware transfer costs.
6. Add tiered admission and eviction.
7. Add vLLM or SGLang connector.
8. Run real traces and compare against engine-local prefix caching.
