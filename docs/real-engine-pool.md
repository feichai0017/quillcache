# Real engine ↔ KV pool (the data plane)

This is the **other half** of [m3-real-vllm.md](m3-real-vllm.md). That doc wires a
real vLLM to the gateway for **routing** (the control plane: the gateway picks an
engine, KV *events* report residency). This doc wires a real engine to the
**distributed KV pool** so it moves real KV **bytes** in and out — the Mooncake
*Store* path:

- **master** — `quillcache master`: the shared residency index + node registry
  over HTTP. "Who holds block X, and at what transfer address?"
- **node** — `quillcache node`: a crash-consistent KV byte store (DRAM + SSD) and
  a transfer server, registered with the master. The thing an engine offloads to.
- **client + connector** — `bridge/quillcache_client.py` (the master HTTP + the
  transfer TCP wire) and `bridge/vllm_quillcache_connector.py` (offload on
  prefill, load on a prefix-cache hit, with local-first + cross-node fetch).

```
        ┌─────────────────────────── master (HTTP) ───────────────────────────┐
        │  register · placed · locate · nodes · state   (residency index)      │
        └───────▲───────────────────────▲──────────────────────────▲──────────┘
                │ register/report        │ locate                   │
   ┌────────────┴───────┐      ┌─────────┴──────────┐    vLLM worker + connector
   │  quillcache node A │◀────▶│  quillcache node B │◀───────────────┘
   │  byte pool + xfer  │ TCP  │  byte pool + xfer  │   offload bytes ▲ │ load bytes
   └────────────────────┘bytes └────────────────────┘                 (transfer wire)
```

## Local dry run — no GPU, all real (verified)

Three terminals. This moves **real bytes** through the **real** Rust pool; only
the "KV" payload is fake (a GPU would supply the actual tensor bytes).

```bash
# 1) master — the residency index + node registry
cargo run -- master --addr 127.0.0.1:7777

# 2) a pool node — a byte store + transfer server, registers with the master
cargo run -- node --addr 127.0.0.1:7001 --master http://127.0.0.1:7777 \
    --id node-demo --data-dir /tmp/qc-node-demo

# 3) the connector demo — offload a block, then load it back through the pool
python3 bridge/vllm_quillcache_connector.py http://127.0.0.1:7777 127.0.0.1:7001
#   -> loaded back: b'fake-kv-bytes'
python3 bridge/quillcache_client.py http://127.0.0.1:7777
#   -> state: {... 'node_count': 1, 'resident_blocks': 1}
```

Add a second node (`--addr 127.0.0.1:7002 --id node-2`) and the connector's
`load_block` exercises the **cross-node fetch + local re-cache** path: a miss on
the local pool → `locate` on the master → fetch from the peer's transfer server →
re-cache locally and re-report. Same logic as `quillcache cluster`, but
out-of-process and driven from Python.

## On a GPU (Modal) — wire the connector to vLLM

The connector's pool logic (offload / locate / cross-node fetch / re-cache) is
real and tested above. What needs a real GPU + a pinned vLLM version is the
**KV-tensor (de)serialization** and the **`KVConnectorBase_V1` hook signatures** —
left as documented TODOs in `bridge/vllm_quillcache_connector.py`.

1. Run the **master** somewhere both the laptop and the GPU box can reach (a
   public host or a tunnel; a laptop-local master is not reachable from Modal).
2. Run a **node** co-located with each vLLM worker (same box → `localhost`
   transfer, no network hop for the warm path).
3. In your vLLM, register `QuillCacheConnector` as the KV connector and wire its
   hooks to your installed vLLM's
   `vllm/distributed/kv_transfer/kv_connector/v1/base.py`:
   - `save_kv_layer` → `offload_block(...)` (serialize the freshly computed block;
     quantize to FP8 on offload to match `quillcache-cuda`'s device tier).
   - `start_load_kv` → `load_block(...)` (copy pool bytes into vLLM's paged-KV
     slots).
   - `get_num_new_matched_tokens` → sum `token_count` over blocks the master can
     `locate`, so the scheduler skips recomputing that prefix.
4. Measure: prefix-cache **hit rate** and **TTFT** with the connector on vs off,
   under a shared-prefix workload (`bench/run_trace.py`).

## Status — what's real vs what needs a GPU

| Piece | Status |
| --- | --- |
| `quillcache master` (HTTP residency index + registry) | **real, tested** (`cargo test`) |
| `quillcache node` (byte pool + transfer server) | **real, tested** (local e2e above) |
| `bridge/quillcache_client.py` (master HTTP + transfer TCP) | **real, tested** (local e2e above) |
| connector offload / locate / cross-node fetch / re-cache | **real, tested** (local e2e above) |
| vLLM `KVConnectorBase_V1` hooks + KV-tensor (de)serialization | **skeleton** — wire to your vLLM version + GPU |
| RDMA / GPUDirect transfer (vs the TCP wire) | **reserved** — `quillcache-cuda`, needs NVIDIA hardware |

This mirrors the project's honesty rule: the control plane and the byte-moving
data plane are real and verified on a laptop; the GPU-resident tensor plumbing is
a clearly-marked seam you complete on your own hardware.
