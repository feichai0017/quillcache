---
title: Mooncake / Dynamo mapping
description: Every Mooncake and NVIDIA Dynamo component, mapped to a QuillCache crate.
---

QuillCache mirrors Mooncake's decomposition piece by piece, then adds its
differentiation on top. The concepts line up one-to-one — built small enough to
read end-to-end and measure.

| Mooncake / Dynamo | QuillCache | Status |
| --- | --- | --- |
| Transfer Engine (`TransferEngine` + `Transport`) | `quillcache-transfer-engine` (`engine` + `transport::{tcp,rdma,nvlink}`) | ✅ TCP · ✅ **one-sided RDMA verbs** (`ibverbs`, verified over SoftRoCE) / ◑ cross-node RDMA Transport wiring remaining |
| Store `Client` (`PutStart`/`PutEnd`/`Get`) | `DummyClient` / `RealClient` | ✅ end-to-end over the transfer engine |
| Store `MasterService` (two-phase Put, eviction, **HA**, batch) | `MasterService` | ✅ replica alloc · lease eviction · **HA** (snapshot + heartbeat + etcd election) · **batch Put/Get** |
| `BufferAllocator` + `AllocationStrategy` | `OffsetBufferAllocator` + **`BinnedBufferAllocator` (O(1) size-binned)** + `Random`/`FreeRatioFirst` | ✅ |
| Overload-oriented early rejection | `ControlPlane::admit` (SLO-violation budget) | ✅ |
| `TransferMetadata` (etcd/redis/http/p2p) | `MetadataBackend`: `InMemoryMetadata` / `EtcdMetadata` (feature `etcd`) | ✅ in-memory · ✅ etcd (verified vs real etcd) · ⊙ redis/http/p2p |
| Dynamo KV-router cost function | `DynamoCostRouter` | ✅ reproduces the worked example |
| Dynamo KVBM tiers (G1 HBM / G2 host / G3 disk) | `StoreDataPlane` (DRAM/SSD) + `quillcache-cuda` (HBM G1 + FP8 quantize) | ✅ DRAM/SSD · ✅ HBM (L4-verified) |
| Mooncake GPU data path (GPUDirect-RDMA · NVLink · GDS) | `DeviceSegment` zero-copy **peer copy** (`cuMemcpyPeer`) + one-sided `rdma` verbs | ✅ **intra-node HBM↔HBM** (2×L4-verified, 4.4× vs host-staged) · ✅ **one-sided RDMA verified over SoftRoCE** · ⊙ GPUDirect-RDMA needs a NIC |
| Mooncake Conductor / Dynamo KV-Cache Indexer | `conductor` (`PrefixCacheTable` + `ModelContext`) + residency index (Holt ART) + **`replication` (hot-prefix balancing)** | ✅ longest-prefix overlap · persistent · **hot-prefix replication** |
| vLLM/SGLang engine integration + disaggregated P/D | `QuillCacheV1Connector` (`KVConnectorBase_V1`) + `pd-proxy` router | ✅ **true vLLM-native P/D** (kv_producer/consumer, 2×L4-verified) |
| Metrics / observability | gateway `/metrics` (Prometheus) | ✅ cache hits · transfers · refused reuse · SLO goodput |
| — *(neither does this)* | **identity guard + crash-consistent `DiskTier`** | 🎯 differentiation |

> `quillcache-cuda` is the one piece that is **not** a 1:1 Mooncake component:
> Mooncake puts GPU in the Transfer Engine (the `rdma` / `nvlink` transports
> above); the HBM-tier + FP8-quantize crate mirrors NVIDIA Dynamo's KVBM, not
> Mooncake.

## The Dynamo cost function

`DynamoCostRouter` reproduces the cost function NVIDIA Dynamo's KV router runs.
For each worker:

```text
overlap_credit   = 1.0·device + 0.75·host + 0.25·disk   (HBM / DRAM / SSD hits)
adjusted_prefill = max(0, raw_prefill_blocks − overlap_credit)
cost             = prefill_load_scale · adjusted_prefill + decode_blocks
```

It routes to the lowest-cost worker. A GPU-resident prefix hit is worth 4× an SSD
one — cache locality vs load, as a single number — and a unit test reproduces
Dynamo's own published worked example (costs 18 / 10 / 11 → pick worker 2).

## The distributed read path

The store's read mirrors Mooncake's `Client` → `MasterService` → Transfer Engine
flow:

1. the `Client` asks the `MasterService` for the block's replicas
   (`get_replica_list`), which is **identity-guarded** — a cross-identity request
   is refused with `ReuseViolation` *before* any bytes move;
2. the reply names the holding **segment** and **offset** in registered memory;
3. the **Transfer Engine** moves those bytes one-sidedly by `(segment, offset)`
   (TCP today, RDMA / NVLink reserved) — it transfers by location, never by key.
