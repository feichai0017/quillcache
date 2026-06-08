# Residency Index Backends

QuillCache treats the residency index as a replaceable control-plane backend.
The index stores metadata about where KV blocks live; it does not own the KV
tensor bytes.

## Interface

The Rust boundary is the single trait `IndexBackend` in `quillcache-core`:

- `put` / `remove_block` / `clear_worker` — identity-scoped residency updates
- `locate` — all residencies for an exact block identity
- `prefix_scan` — identity-aware prefix lookup (the ART/radix strength)
- `snapshot` — resident blocks for routing
- `metrics` — comparable `IndexMetrics` (incl. `bytes_written`)
- `persistent` — whether residency survives restart

KV-event translation is **not** part of the trait: `quillcache-control` resolves
block identity once in the backend-agnostic `ingest_batch(&mut dyn IndexBackend, …)`,
so backends never re-implement vLLM/SGLang event parsing.

The first implementation is `MemoryIndex` (in `quillcache-core`). It is
deliberately small: it proves the gateway, event ingest, routing, and state API,
and is the baseline against which persistent backends are compared.

## Stored Object

Each index entry maps a strict KV block identity to one or more residencies:

```text
(model_id, tokenizer_id, adapter_id, tenant_id, prefix_hash, block_hash, block_index)
  -> [{ worker_id, tier, bytes, last_access_ms, ref_count, pinned }]
```

The identity includes model, tokenizer, adapter, and tenant because a raw block
hash is not enough to authorize KV reuse. Later versions should add explicit
cache-sharing policy, block version, and lease epoch when the gateway begins to
make stronger safety claims.

## Backend Plan

| Backend | Status | Purpose |
| --- | --- | --- |
| Memory | v0.1 implemented | Fast smoke tests, routing experiments, and local gateway demos. |
| Holt ART | implemented | Persistent prefix/residency index with prefix-native lookups and crash recovery (`quillcache-index-holt`). |
| RocksDB/LSM | implemented | LSM baseline for write amplification, recovery, and prefix-scan vs ART (`quillcache-index-rocksdb`). |
| Filesystem catalog | planned baseline | Emulate simple block catalog designs used by file-backed KV offload systems. |

## Holt Integration Shape

Holt should be used as the persistent metadata index, not as the live GPU tensor
manager. The first key layout should be prefix-friendly:

```text
tenant/model/tokenizer/adapter/session/prefix_hash/block_hash/block_index
```

The value should be a compact serialized `CacheResidency` plus metadata needed
for recovery:

- worker id
- cache tier
- byte size
- last seen timestamp
- lease or version epoch
- optional data-plane object handle

This keeps Holt in the control plane. A data-plane backend such as LMCache,
Dynamo KVBM, HiCache, or a vLLM/SGLang connector can still own tensor movement
and materialization.

## Measurement Plan

The first backend experiment should compare ART and LSM under KV-cache metadata
workloads:

- event ingest throughput
- p50/p99 point lookup latency
- prefix scan latency
- write amplification
- restart recovery time
- index size on disk
- stale-entry cleanup cost

The workload should include repeated system prompts, RAG documents, agent tool
schemas, multi-turn sessions, and block removals from HBM pressure.

## v0.1 Boundary

v0.1 ships only the memory backend. The important architectural decision is that
the gateway, control plane, and router depend on the `IndexBackend` trait, so
Holt and RocksDB can be introduced without changing request proxying or route
scoring.
