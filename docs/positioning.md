# Positioning

QuillCache is an AI infrastructure research platform for LLM KV cache control
planes.

The project is not an inference engine, a KV tensor data plane, or a production
Kubernetes operator. It is a control-plane prototype and evaluation platform
that makes KV cache state observable, routable, explainable, and comparable
across real inference engines.

## One-Line Claim

QuillCache is a research platform and control-plane prototype for
identity-aware, persistent, and policy-driven KV cache reuse in LLM serving.

## Layer Boundaries

| Layer | System | Role |
| --- | --- | --- |
| Inference runtime | vLLM, SGLang | Execute model prefill/decode and own live GPU KV tensors. |
| KV tensor data plane | LMCache, Dynamo KVBM, HiCache, NIXL, Mooncake-style stores | Move, offload, and materialize KV tensors across GPU/DRAM/SSD/remote tiers. |
| Control plane and research platform | QuillCache | Ingest KV metadata, maintain residency indexes, choose routing and policy decisions, and run controlled experiments. |
| Persistent ART index backend | Holt | Store identity-aware residency metadata and prefix/session indexes. |
| LSM baseline | RocksDB or SGLANG-LSM-style backend | Compare ART against an LSM design for KV-cache metadata/index workloads. |

## What QuillCache Does

- Proxies OpenAI-compatible requests to real vLLM/SGLang workers.
- Ingests KV lifecycle events from real runtimes or connector bridges.
- Maintains a global residency index across workers and tiers.
- Evaluates route/reuse/recompute/transfer decisions under a cost model.
- Exposes decision traces through response headers and state endpoints.
- Provides a common test harness for policy and index-backend experiments.

## What QuillCache Does Not Do

- It does not implement transformer attention kernels.
- It does not replace vLLM, SGLang, LMCache, or Dynamo KVBM.
- It does not move KV tensor bytes in v0.1.
- It does not claim production-grade multi-tenant isolation yet.

## Precisely: what "interface with real KV cache" means

QuillCache taps the **state** of real engines' KV cache (which block is cached
where, via their KV events), runs **different routing / reuse policies** on it,
and is **engine-neutral** (vLLM / SGLang). It does not touch the KV tensor
**bytes**.

- **Reads (today):** ingest engine KV events → real, live residency state.
- **Writes (later):** instruct the engine / LMCache to reuse or load a block
  (e.g. via vLLM `kv_transfer`); QuillCache *orchestrates* the data plane, it
  does not move tensors itself.

Three pluggable axes: **engines** (vLLM / SGLang), **routing policies**
(`RoutingPolicy`), and **index storage engines** (`IndexBackend`:
memory / Holt-ART / RocksDB-LSM).

## Research Bet

The research bet is that performance-oriented KV-cache systems and
security-oriented cache-isolation work are still too separate. QuillCache makes
object identity explicit in the control plane:

- model identity
- tokenizer identity
- adapter identity
- tenant and cache-sharing policy
- session and workflow identity
- block version or lease epoch
- cache salt or safety boundary

The platform should be able to measure the cost of safe reuse instead of
treating safety as a vague policy statement.

## Engineering Bet

The engineering bet is that useful AI infra work does not require rewriting the
serving engine. A small gateway, a real event ingest path, and a pluggable
residency index are enough to reproduce many of the routing, placement, and
debugging questions that production inference platforms face.

The first version proves this by wiring QuillCache to real OpenAI-compatible
workers and a vLLM KV-events bridge.
