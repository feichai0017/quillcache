# Transfer line: efficient KV transfer — SOTA, selection, and design

**Goal.** Make KV-cache movement (prefill→decode handoff, cross-tier fetch, store
Put/Get) match the state of the art on the wire, and — our contribution —
**co-schedule transfer with compute so it is never on the critical path**. The
scheduling is backend-agnostic and sits on top of whatever wire we use.

This is the transfer layer under the cluster-level
[KV-cache-centric co-scheduler](./co-scheduler-design.md). The co-scheduler decides
when to fetch, replicate, demote, or recompute; this document defines how the
chosen transfer should run without stalling decode.

## 1. SOTA survey (as of mid-2026)

| Engine | Architecture | GPUDirect | SM-free? | Multi-NIC / topology | Notes |
| --- | --- | --- | --- | --- | --- |
| **NIXL** (NVIDIA, Dynamo) | point-to-point, modular backends (RDMA/IB, RoCE via UCX, TCP, NVMe-oF, S3) | yes | **yes** (UCX, no SM) | yes | current reference; top throughput |
| **Mooncake TE** | read/write API, segment+offset | yes | yes | **NIC-GPU PCIe affinity + multi-NIC aggregation** | 25% lower TTFT vs TCP; feature-rich, but a public bench says it fails to saturate a 50 GB/s link |
| **NCCL / RCCL** | collective comm | via kernels | **no** (burns SM) | — | 30–50% slower for KV; SM contention with inference |
| **UCCL P2P** | read/write + collective, SM-free | yes | **yes** | auto topology | comparable to NIXL; academic/open |

**Key techniques the SOTA converges on:**
- **GPUDirect RDMA** — move GPU↔GPU/NIC without bouncing through host DRAM, no CPU copy.
- **SM-freeness** — the transfer must **not** consume GPU SMs (NCCL's flaw): those
  SMs are needed for inference. NIXL/UCCL/Mooncake all use GDR-without-SM (UCX-style).
- **IBGDA** (InfiniBand GPUDirect Async) — the GPU kernel *initiates* the RDMA, CPU
  fully out of the loop → lowest latency. (Mooncake `device` transport = P2P + IBGDA.)
- **Multi-NIC aggregation + topology-aware NIC↔GPU affinity** (from PCIe topology).
- **Layer-wise / chunked overlap** — a 3-stage pipeline (produce → transfer →
  consume) that hides transfer latency behind compute.

**Honest nuances (must not be glossed):**
- **GPUDirect is not always faster.** Mooncake issue #1459 reports GDR ~15 GB/s vs
  CPU-staged RDMA ~47 GB/s on some setups — PCIe topology, GPU BAR, and NIC↔GPU
  affinity dominate. "Turn on GDR" is *not* automatically a win; **topology affinity
  matters more.**
- NCCL P2P is the wrong tool for KV transfer (SM contention with the forward pass).

## 2. Design principles

1. **SM-free** — never burn GPU compute on byte movement.
2. **Zero-copy GPUDirect where topology supports it; CPU-staged fallback when
   NIC↔GPU affinity is poor** (decide per path, not globally).
3. **Topology-aware** — NIC↔GPU PCIe affinity + multi-NIC aggregation.
4. **Off the critical path** — layer-wise overlap with compute.

## 3. Selection — what QuillCache uses (don't reinvent the wire)

- **GPU wire**: adopt **NIXL / UCX** as a transfer backend (SM-free GDR, multi-backend,
  the Dynamo standard) — **do not hand-roll IBGDA**. Wrap via FFI or a thin sidecar.
- **CPU tiers (DRAM/SSD) + the SoftRoCE-testable path**: keep our existing **ibverbs
  RDMA + TCP** transports.
- **Topology**: extend `topology::select_devices` to PCIe NIC↔GPU affinity (align to
  Mooncake's automatic affinity detection).
- **Our novelty**: the **layer-wise overlap scheduler** on top — backend-agnostic.

> Rationale: the wire is a solved, fast-moving problem (NIXL/UCX). Reinventing IBGDA
> is years of work for no novelty. **Innovate on the *schedule*, adopt the best wire.**
> This is both job-relevant (NIXL/Dynamo) and research-relevant (co-scheduling).

## 4. First contribution: layer-wise overlapped transfer

KV layout is `[num_layers][2 (K,V)][tokens × heads × dim]`. Today we move it as one
blob (`slice_pool::run_slices` = fixed-size **byte** slices, bounded in-flight).

Layer-wise upgrade:
- **Chunk by layer** (semantic boundary), not just bytes.
- **Per-layer completion signal** (Mooncake's `submitTransferWithNotify`): the
  consumer (decode) starts layer 0 the instant it lands, while the producer is still
  sending later layers → the 3-stage pipeline overlaps.
- Builds directly on `slice_pool`: add `layer_slices()` + a per-layer notify channel.

**Result:** consumer-start latency ≈ **time-to-first-layer**, not time-to-all-layers
→ a real TTFT drop, independent of the wire.

## 5. Phases

- **P0 (hardware-free, CI-testable):** layer-wise chunk + per-layer notify in the
  transfer engine over **TCP**; benchmark consumer-start latency (overlap) vs the
  monolithic blob. Pure scheduling win; runs on Mac/CI.
- **P1:** wire into the connector — per-layer `save_kv_layer` / `start_load_kv` (the
  engine↔cache co-design touchpoint).
- **P1.5:** expose layer-aware store metadata (`LayerManifest`) so the scheduler can
  reason about time-to-first-layer and not only whole-object latency.
- **P2 (hardware-gated):** NIXL/UCX GPU backend + GDR + topology affinity; measure on
  a real NIC/GPU (Modal multi-node). *Landed (Rust, tested):* the topology-affinity
  primitive `topology::{PcieAffinity, set_affinity, affine_nics, prefers_gpudirect,
  rebuild_matrix_from_affinity}` — ranks NICs by PCIe proximity to the GPU and gates
  GPUDirect on affinity (the #1459 lesson). *Remaining (needs hardware):* the NIXL/UCX
  backend behind the `Transport` trait + real-NIC/GPU bandwidth + SM-utilization numbers.

## 6. Evaluation

- **Metrics:** time-to-first-layer, full-transfer time, end-to-end TTFT, **GPU SM
  utilization during transfer**, bandwidth (GB/s).
- **Baselines:** monolithic blob transfer; NCCL P2P (show SM contention); TCP.
- **Ablations:** in-flight depth, layer-chunk granularity, topology affinity on/off,
  GDR vs CPU-staged (reproduce the #1459 crossover).

## 7. Honest constraints

- Real GDR / NIXL needs a proper NIC + GPU; CI/Mac only exercises TCP (+ SoftRoCE).
  **P0 is hardware-free; P2 is hardware-gated.**
- NIXL integration is an FFI/sidecar dependency — scope it **after** P0/P1 prove the
  scheduling win, so the contribution stands on its own first.

## 8. Sources

- NVIDIA NIXL / disaggregated inference (Spheron), Mooncake Transfer Engine design
  doc, UCCL "KV Cache Transfer Engine" comparison, Mooncake issue #1459 (GDR vs CPU
  RDMA bandwidth), layer-wise overlap (LLM inference handbook / recent disagg papers).

## 9. P0 result (measured — TCP wall-clock model)

`layer_wise_overlap_hides_transfer_behind_compute` (in `slice_pool.rs`), N=6 layers,
20 ms transfer + 20 ms compute per layer, in-flight depth 2:

| metric | monolithic (barrier) | layer-wise overlap | win |
| --- | --- | --- | --- |
| consumer-start latency | ~65 ms (wait for all layers) | ~22 ms (wait for layer 0) | **~3× earlier** |
| total wall-clock | ~203 ms (transfer **then** compute) | ~145 ms (transfer **under** compute) | **~29% lower** |

The overlap reclaims `min(transfer, compute)` minus one layer: transfer stops being a
barrier in front of serving. The gap widens as the layer count grows or as compute
dominates (the common case) — transfer is then **fully** hidden. Backend-agnostic:
the same scheduler over RDMA/GPUDirect (P2) only changes the per-layer cost, not the
overlap structure.
