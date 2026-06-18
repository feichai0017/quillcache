# Interview prep — dynamic planning · utilization/throughput/latency · transfer & comms

Maps the three interview focus areas to **concrete, real things in this project** (what
to say, grounded in code), then the transfer/comms deep dive (RDMA / NVMe / SPDK / GDS).
The recurring senior move: **the fast path is conditional — measure, don't assume.**

## 1. The three question areas → what to say

### A. Dynamic balancing & planning
- **It means:** dynamically (re)allocate **compute** (prefill/decode ratio), **memory**
  (HBM working-set vs reusable cache), and **bandwidth** to hold high utilization while
  meeting SLO under shifting, bursty, skewed load.
- **What we have (talk-able):** cost-based cache-aware routing (`CostModel` =
  prefill/decode/queue/tier costs; `GreedyStatePlaneRouter` argmin; `DynamoCostRouter`
  reproduces Dynamo's published worked example); P/D mode derivation; admission control
  (SLO-violation budget); hot-prefix replication (CountMinSketch); in-flight load
  feedback into routing.
- **Gap / what I'd build:** a *planner* — a per-interval feedback loop that flexes the
  P/D ratio and the HBM working-vs-cache split from live queue depth + SLO headroom
  (Dynamo's Planner). Today roles/splits are static.
- **One-liner:** *"Routing is the per-request decision; planning is the per-interval
  decision. I co-optimize P/D ratio + HBM split + cache capacity against one
  SLO-goodput objective, with the cost model as the shared currency."*

### B. Utilization / throughput / latency
- **The triangle:** TTFT (prefill-bound), TPOT (decode-bound), throughput (batch). P/D
  disaggregation removes interference; cache reuse cuts prefill recompute (throughput);
  layer-wise overlap hides transfer (latency); batching raises utilization.
- **What we have:** live **SLO-goodput** metering (TTFT vs budget); cache-aware routing
  (fewer recomputes); **layer-wise overlap** (our bench: consumer-start 3× earlier, wall
  29% lower); the cost model exposing the TTFT/TPOT/SLO trade explicitly.
- **One-liner:** *"I optimize goodput-under-SLO, not raw throughput — a reply that's
  fast but past its SLO is wasted work. The router scores TTFT+TPOT+SLO-violation, and
  overlap keeps transfer off the TTFT critical path."*

### C. Transfer / communication
- **What we have:** Mooncake-style Transfer Engine — one-sided transfer by
  `(segment, offset)`, **pooled QPs** (10× from reuse, SoftRoCE-verified), **topology-aware
  NIC↔GPU PCIe affinity + GDR gating** (`PcieAffinity`/`prefers_gpudirect`), layer-wise
  overlap; **NIXL/UCX** selected for the GPU wire.
- **One-liner:** *"The wire is solved by NIXL / Mooncake-TE — SM-free GPUDirect RDMA.
  The interesting decisions are topology affinity (a far NIC makes GDR slower than
  staging) and keeping transfer off the critical path."*

## 2. Transfer/comms deep dive

### RDMA
- **One-sided** (READ/WRITE — remote CPU *uninvolved*) vs **two-sided** (SEND/RECV). KV
  transfer uses one-sided: the initiator reads/writes registered remote memory by
  `(rkey, addr)` → low latency, no remote scheduling. Mooncake/ours move "by location".
- **QP** (queue pair) + **CQ** (completion queue); **MR** registration pins memory and
  yields the `rkey`. **Pooling QPs** (our `EndpointStore`, FIFO/SIEVE) amortizes the
  expensive setup → the 10× we measured.
- **GPUDirect-RDMA**: NIC DMAs straight to/from GPU HBM (no host bounce). **IBGDA**: the
  GPU *kernel* posts the RDMA, CPU fully out of the loop → lowest latency.
- **Nuance (#1459):** GDR is **not** always faster — a NIC far from the GPU
  (cross-NUMA/switch) can lose to CPU-staged RDMA. → affinity gates GDR.

### NVMe & io_uring
- **NVMe**: deep, parallel SQ/CQ queues; built for high queue-depth random I/O.
- **io_uring**: async submission/completion rings; interrupt-driven *or* poll mode;
  zero-copy with registered buffers.
- **Nuance:** for inference KV offload, **interrupt-driven io_uring often gives the best
  balance** of tail latency / throughput / CPU; **polling wins only when small-random-I/O
  dominates and CPU is ample.**

### SPDK (Storage Performance Development Kit)
- **Userspace, polled-mode, lockless, zero-copy** NVMe driver — bypasses the kernel (no
  syscall / context-switch / interrupt), polls completions, message-passing instead of
  locks. ~120M IOPS class on a 2U box.
- **bdev** pluggable block layer; **NVMe-oF target** (serve NVMe over RDMA/TCP).
- **Wins when:** small-random-I/O bound + spare cores to burn on polling. **Loses when:**
  CPU-constrained or large-sequential — interrupt io_uring is the better balance.
- **Costs:** dedicates cores to busy-polling; hugepages; userspace-driver complexity.

### GDS (GPUDirect Storage)
- Direct GPU↔NVMe path (PCIe P2P local; NVMe-oF / NFS-RDMA remote) — no CPU bounce.
- **Challenge:** the fragmented paged-KV layout → **many tiny random I/Os** → the CPU
  still bottlenecks per-I/O initiation even with GDS → **coalescing/batching KV blocks**
  matters as much as the direct path.

### The synthesis (the impressive part)
Kernel-bypass + zero-copy + poll-when-CPU-ample is the recipe — but **none of GDR / SPDK
/ GDS is universally faster.** The senior move is to choose by regime: **topology
affinity** for GDR, **I/O size + CPU budget** for SPDK-vs-io_uring, **coalescing** for GDS
fragmentation. The fast path is conditional, everywhere in the stack.

## 3. The dynamic-planning answer (deepest topic)
- **Control loop:** observe (GPU util, prefill queue, decode batch occupancy, SLO
  headroom, cache hit rate) → decide → actuate, on a short interval.
- **Knobs:** P/D worker ratio (repurpose a decode GPU to prefill when the prefill queue
  grows); HBM split (working-set vs reusable cache); cache-tier capacities; admission
  threshold; hot-prefix replication.
- **Objective:** maximize Σ goodput s.t. SLO, under the GPU/HBM budget.
- **Why not static:** load is bursty + phase-skewed (prefill-heavy vs decode-heavy);
  static leaves GPUs idle *or* violates SLO. The cost model is the shared currency; the
  planner is the per-interval optimizer on top of the per-request router.

## 4. What to build for credibility (recommended)
- **Co-scheduling controller (A/B):** a feedback loop that flexes P/D ratio + HBM split
  to hold SLO under a synthetic bursty load; report utilization/goodput vs static.
  Local-simulatable — real numbers without hardware.
- **io_uring DiskTier path (C):** swap the `DiskTier` POSIX I/O for io_uring (`io-uring`
  / `tokio-uring`); bench vs POSIX. Real high-perf storage-I/O experience, with **SPDK as
  the documented next step** (needs hugepages + dedicated cores; justified only in the
  small-random-I/O + CPU-ample regime).

## Sources
- SPDK: <https://spdk.io/doc/about.html>, <https://spdk.io/doc/userspace.html>,
  120M IOPS <https://spdk.io/news/2023/02/01/nvme-120m-iops/>.
- io_uring vs SPDK / GDS for inference: KV-offload + GDS (NetApp), Tutti
  (arXiv 2605.03375), InstInfer (arXiv 2409.04992).
- RDMA / GDR affinity nuance: Mooncake issue #1459. NIXL: <https://github.com/ai-dynamo/nixl>.
