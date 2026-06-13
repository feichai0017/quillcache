# Real engine ↔ the KV store (the data plane)

The faithful Mooncake **Store** data path: a real engine offloads/loads KV bytes
to/from the distributed pool via the store `MasterService` + the Transfer Engine.
(For the **control plane** — the gateway routing requests cache-aware — see
[m3-real-vllm.md](m3-real-vllm.md).)

The pieces (all in this repo):

| piece | what it is |
| --- | --- |
| `quillcache store-master` | the `MasterService` over HTTP — two-phase Put, identity-guarded Get, Mount (`src/store_master_http.rs`) |
| `quillcache transfer-node` | a Transfer Engine storage node serving one named RAM segment over the `(segment, offset)` wire (`src/transfer_node.rs`) |
| `bridge/quillcache_store_client.py` | stdlib client: the store master (HTTP) + the transfer wire (TCP) |
| `bridge/vllm_quillcache_connector.py` | a vLLM KV-connector skeleton on the above |

The flow is Mooncake's: **put** = `put_start` (master allocates replica buffers)
→ WRITE the bytes to each `(segment, offset)` over the transfer engine →
`put_end`; **get** = `get_replica_list` (identity-guarded — refused *before* any
byte moves) → READ a replica over the transfer engine. No object bytes ever flow
through the master.

## Local dry run — no GPU, all real (verified)

Three terminals. This moves **real bytes** through the **real** store; only the
"KV" payload is fake (a GPU supplies the real tensor bytes).

```bash
# 1) a storage node — a Transfer Engine segment served over TCP
cargo run -- transfer-node --addr 127.0.0.1:8100 --segment seg-0

# 2) the store master — metadata, two-phase Put, the identity guard
cargo run -- store-master --addr 127.0.0.1:7777

# 3) the connector demo — offload a block, load it back through the store
python3 bridge/vllm_quillcache_connector.py     # run from the bridge/ dir
#   -> loaded back: b'fake-kv-bytes-over-the-faithful-store'
curl -s http://127.0.0.1:7777/v1/state
#   -> {"objects":1,"segments":1,"capacity":...,"allocated":37}
```

Add more storage nodes (`--addr ... --segment seg-1`, register them in the
connector's `segment_endpoints` + `--replica-num 2`) and a Put replicates across
distinct segments. A Get under a different `tenant_id` is refused with HTTP 403
(the identity guard, over the wire).

## On a GPU (Modal) — wire the connector to vLLM

The connector's store logic (put_start → transfer WRITE → put_end; get_replica_list
→ transfer READ; identity guard) is real and tested above. What needs a real
GPU + a pinned vLLM version is the **KV-tensor (de)serialization** and the
`KVConnectorBase_V1` **hook signatures** — documented TODOs in
`bridge/vllm_quillcache_connector.py`:

1. Run `store-master` somewhere reachable, and a `transfer-node` co-located with
   each vLLM worker (same box → `localhost`, no network hop for the warm path).
2. Register `QuillCacheConnector` as the vLLM KV connector; wire its hooks:
   - `save_kv_layer` → `offload(block_key, kv_bytes)` (quantize to FP8 to match
     `quillcache-cuda`'s device tier);
   - `start_load_kv` → `load(block_key)` (copy pool bytes into vLLM's paged-KV);
   - `get_num_new_matched_tokens` → how many prefix tokens the store can serve, so
     the scheduler skips recomputing that prefix.
3. Measure prefix-cache **hit rate** + **TTFT** with the connector on vs off, under
   a shared-prefix workload (`bench/run_trace.py`).

## Status — what's real vs what needs hardware

| piece | status |
| --- | --- |
| `store-master` (MasterService over HTTP) | **real, tested** (`cargo test`) |
| `transfer-node` (Transfer Engine segment server) | **real, tested** (the engine's TCP round-trip) |
| `quillcache_store_client.py` (master HTTP + transfer wire) | **real, tested** (the local e2e above) |
| connector offload / load (two-phase Put, identity-guarded Get) | **real, tested** (the local e2e above) |
| vLLM `KVConnectorBase_V1` hooks + KV-tensor (de)serialization | **skeleton** — wire to your vLLM + GPU |
| RDMA / GPUDirect transfer (vs the TCP wire) | **reserved** — `--features rdma/nvlink`, needs a NIC/GPU |

This mirrors the project's honesty rule: the store + the byte-moving transfer
engine + the connector path are real and verified on a laptop; the GPU-resident
tensor plumbing is a clearly-marked seam you complete on your own hardware.
