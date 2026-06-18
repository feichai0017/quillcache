# Design references

The papers, systems, and algorithms QuillCache is built on — grouped by area, each
with what QuillCache borrows from it. This is the citation base for the design docs
([architecture](./transfer-line-design.md) and the site docs).

## Disaggregated LLM serving & the KV cache

- **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** — Qin et
  al., USENIX **FAST '25** (also ACM ToS 2025); arXiv [2407.00079](https://arxiv.org/abs/2407.00079);
  FAST talk: <https://www.usenix.org/conference/fast25/presentation/qin>.
  → The primary reference. QuillCache mirrors its component decomposition: two-phase-Put
  `Client`, sharded `MasterService`, replica/segment model, lease eviction, the
  pooled-DRAM/SSD KVCache store, the Transfer Engine, and the OpLog/snapshot HA model.
- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving**
  — Zhong et al., **OSDI '24**; arXiv [2401.09670](https://arxiv.org/abs/2401.09670);
  <https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin>.
  → The prefill/decode disaggregation rationale (eliminate P/D interference; co-optimize
  per-phase GPU allocation for goodput) behind our `EngineRole` + P/D derivation.
- **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** — Patel et
  al., **ISCA '24**. → Corroborating P/D split design; phase-aware resource use.
- **MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert
  Parallelism** — Zhu et al., ByteDance Seed / PKU, **SIGCOMM '25**; arXiv
  [2504.02263](https://arxiv.org/abs/2504.02263).
  → Cluster-level disaggregation beyond P/D: attention/FFN separation, ping-pong
  pipeline parallelism, module-specific parallelism, and high-performance M2N
  communication to raise per-GPU throughput. This motivates QuillCache's
  cluster-level co-scheduler rather than request-local routing only.
- **Efficient Memory Management for LLM Serving with PagedAttention (vLLM)** — Kwon et
  al., **SOSP '23**; arXiv [2309.06180](https://arxiv.org/abs/2309.06180).
  → The engine-side paged KV cache QuillCache integrates with via the V1 connector;
  the "engine owns its HBM working set" boundary.

## KV router / dynamic resource allocation

- **NVIDIA Dynamo** — low-latency distributed inference framework (Smart Router,
  Planner, KV Block Manager / KVBM, NIXL); <https://github.com/ai-dynamo/dynamo>,
  [intro blog](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/).
  → Our `DynamoCostRouter` reproduces its KV-router cost function (overlap credit
  1.0/0.75/0.25 by tier); KVBM's G1/G2/G3 tiers map to our `CacheTier` + `quillcache-cuda`.
- **Flux: Fast Software-based Communication Overlap on GPUs through Kernel Fusion** —
  Chang et al., ByteDance / PKU; arXiv [2406.06858](https://arxiv.org/abs/2406.06858).
  → Fine-grained communication/computation overlap for training and inference. It
  supports the design choice that hiding communication behind compute is a scheduling
  problem, not only a faster-network problem.
- **Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts** —
  Zhang et al., ByteDance Seed / SJTU, **MLSys '25**; arXiv
  [2502.19811](https://arxiv.org/abs/2502.19811).
  → Runtime data-dependency analysis and task rescheduling for MoE overlap. This is
  the closest AML-style analog for QuillCache's layer-wise KV transfer overlap:
  preserve compute efficiency while reducing non-overlapped communication.

## Data transfer engines

- **NIXL (NVIDIA Inference Xfer Library)** — point-to-point KV transfer, SM-free
  GPUDirect RDMA via UCX, backends RDMA/IB·RoCE·TCP·NVMe-oF·S3; <https://github.com/ai-dynamo/nixl>;
  overview: <https://www.spheron.network/blog/nvidia-nixl-disaggregated-inference-guide/>.
  → **Selected GPU-wire backend** (P2). We adopt it rather than hand-rolling IBGDA.
- **Mooncake Transfer Engine** — <https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html>.
  → Our `quillcache-transfer-engine`: `(segment, offset)` one-sided transfer, batched
  slices (worker_pool), topology-aware NIC selection, pooled QPs.
- **UCCL — KV Cache Transfer Engine comparison** — <https://uccl-project.github.io/posts/kv-transfer-engine/>.
  → The SM-free design principle and NIXL/Mooncake/NCCL throughput comparison.
- **Mooncake issue #1459** (GPUDirect ~15 GB/s vs CPU-staged RDMA ~47 GB/s) —
  <https://github.com/kvcache-ai/Mooncake/issues/1459>.
  → The honest nuance: NIC↔GPU PCIe affinity matters more than blindly enabling GDR.
- **NIXL GPUDirect benchmarking** — Muradli et al., 2025;
  <http://cs.iit.edu/~scs/assets/files/muradli2025gpudirect.pdf>.
  → Empirical reminder that GDS/NIXL benefits depend on I/O size, topology, and
  CPU utilization; QuillCache should measure backend choices instead of assuming
  GPUDirect always wins.

## Prefix cache, index structures, hot-key tracking

- **SGLang / RadixAttention** — Zheng et al., "SGLang: Efficient Execution of
  Structured Language Model Programs", **NeurIPS '24**; arXiv [2312.07104](https://arxiv.org/abs/2312.07104).
  → Prefix-cache / radix-prefix sharing; the HiCache backend we target (#22).
- **The Adaptive Radix Tree (ART): ARTful Indexing for Main-Memory Databases** — Leis,
  Kemper, Neumann, **ICDE '13**. → The `HoltIndex` residency backend + the persistent-ART
  master substrate design (ordered prefix scan, instant-reopen recovery).
- **An Improved Data Stream Summary: The Count-Min Sketch and its Applications** —
  Cormode, Muthukrishnan, J. Algorithms 2005. → `CountMinSketch` hot-key/hot-prefix tracking.

## Storage, allocation, durability, HA

- **Exploring CXL-based KV Cache Storage for LLM Serving** — Tang et al., Yale/UIUC/
  ByteDance, NeurIPS ML for Systems Workshop 2024.
  → CXL as a KV-cache storage tier under TTFT SLO. This expands QuillCache's tier
  model beyond HBM/DRAM/SSD and strengthens the "remote fetch versus recompute"
  decision in the co-scheduler.
- **Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving** —
  Qiu et al.; arXiv [2605.03375](https://arxiv.org/abs/2605.03375).
  → SSD-backed KV cache needs GPU-centric object I/O, bulk transfers, and slack-aware
  scheduling to avoid GPU stalls. This informs QuillCache's future DiskTier/GDS path.
- **InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM
  Inference** — Pan et al.; arXiv [2409.04992](https://arxiv.org/abs/2409.04992).
  → Moves attention-side work near storage to avoid PCIe-bound KV transfers. This is
  a stronger future baseline for long-context/offline inference than simple SSD fetch.
- **LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference** —
  Liu et al.; arXiv [2510.09665](https://arxiv.org/abs/2510.09665).
  → A production-oriented KV cache layer for vLLM/SGLang, with batched movement,
  pipelining, and control APIs. It is the most direct external baseline for
  QuillCache's connector/store path.
- **I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM
  Serving** — Wu et al., ByteDance / SUSTech, **NDSS '25**.
  → Shows cross-tenant prompt leakage risks from KV-cache sharing. This supports
  QuillCache's full identity guard as a safety/correctness requirement, not a
  cosmetic metadata field.
- **The CacheLib Caching Engine: Design and Experiences at Scale** — Berg et al.,
  **OSDI '20**. → The slab + offset-allocator lineage behind our buffer allocators.
- **Sebastian Aaltonen — OffsetAllocator** — <https://github.com/sebbbi/OffsetAllocator>
  (O(1) hard-real-time offset allocator, 256-bin 2-level bitmap; TLSF-family).
  → Faithfully ported as `OffsetBufferAllocator` (the store's default allocator).
- **DeepSeek 3FS (Fire-Flyer File System)** — <https://github.com/deepseek-ai/3FS>.
  → Mooncake's distributed-SSD backend (`hf3fs`); a gap we have not yet matched.
- **In Search of an Understandable Consensus Algorithm (Raft)** — Ongaro, Ousterhout,
  **USENIX ATC '14**. → etcd's internals; we use etcd only for leadership/epoch/watermarks
  (the rare strongly-consistent decisions), not for per-op metadata.
- **LMCache** — <https://github.com/LMCache/LMCache>. → A content-hash KV reuse baseline
  (no full-identity guard) we position against.

## How these compose in QuillCache

The store + transfer + router faithfully track Mooncake/Dynamo (the rows above). Our
two differentiators sit on top: **identity-governed safe reuse** (full model·tokenizer·
adapter·tenant scope, vs the content-hash/tenant scope of the reference designs) and a
**crash-consistent persistent tier** (per-block WAL+CRC on recovery, vs trust-by-size).
The active research line — **non-blocking KV transfer / KV-cache-centric co-scheduling**
— builds on the disaggregation (DistServe/Mooncake) + transfer (NIXL/Mooncake TE) +
layer-wise-overlap work; see [transfer-line-design.md](./transfer-line-design.md).
