# M3 — Connect QuillCache to a real vLLM

Goal: run the QuillCache gateway in front of a **real** vLLM on a cloud GPU,
drive it with a shared-prefix workload, and measure **real TTFT** plus the
routing decision headers — and, with the bridge, real KV residency state.

> This doc is the **control plane** (the gateway routes; KV *events* report
> residency). For the **data plane** — an engine moving real KV *bytes* in and out
> of the distributed pool via `quillcache master` + `quillcache node` + a vLLM KV
> connector — see [real-engine-pool.md](real-engine-pool.md).

What QuillCache provides (this repo) vs what you run:

| Piece | Where it runs | Status |
| --- | --- | --- |
| `deploy/modal_vllm.py` | your Modal account (cloud GPU) | recipe — `modal deploy` |
| `bridge/vllm_kv_bridge.py` | co-located with vLLM | run when you want precise KV state |
| `bench/run_trace.py` | your laptop | stdlib, runs anywhere |
| QuillCache gateway | your laptop or the cloud box | `cargo run -- gateway --config ...` |

> You run the cloud GPU with your own account — it can't be provisioned for you.
> Modal/vLLM APIs evolve; pin versions to match your setup.

## Tier 1 — real engine, real TTFT (easiest)

Just proxy a real vLLM through the gateway. No KV-events wiring yet; routing uses
request hints / prefix hashing, but TTFT and responses are real.

1. **Deploy vLLM** (cloud GPU):
   ```bash
   pip install modal && modal token new
   modal deploy deploy/modal_vllm.py     # prints https://<...>.modal.run
   ```
   (Or on a RunPod/Lambda box over SSH: `vllm serve Qwen/Qwen2.5-0.5B-Instruct
   --host 0.0.0.0 --port 8000 --enable-prefix-caching`.)

2. **Point the gateway at it** — set the engine `base_url` in
   `examples/quillcache-gateway.yaml` to the vLLM URL, then:
   ```bash
   cargo run -- gateway --config examples/quillcache-gateway.yaml
   ```

3. **Run the trace** and read TTFT + decision headers:
   ```bash
   python bench/run_trace.py --base-url http://127.0.0.1:8080 \
       --model Qwen/Qwen2.5-0.5B-Instruct --requests 64 --concurrency 8
   ```

To show the **cache-aware routing TTFT benefit** you need **≥2 vLLM instances**
and a shared-prefix workload — see [examples/quillcache-modal.yaml](../examples/quillcache-modal.yaml),
a ready 2-engine fleet on Modal L4. Compare the gateway `policy:` knob:
`dynamo-cost` (cache-aware, the KV-router cost function) vs `round-robin`
(cache-blind baseline). Warm both engines first so the measured TTFT is
steady-state, not Modal scale-to-zero cold start:

```bash
python bench/run_trace.py --base-url http://127.0.0.1:8080 \
    --model Qwen/Qwen2.5-0.5B-Instruct --requests 64 --concurrency 8 \
    --warmup-urls https://<engine-a>.modal.run,https://<engine-b>.modal.run
```

## Tier 2 — precise KV residency (add the bridge)

Run the bridge co-located with vLLM so QuillCache sees real BlockStored /
BlockRemoved events:

```bash
pip install pyzmq msgpack requests
python bridge/vllm_kv_bridge.py \
    --zmq tcp://127.0.0.1:5557 --topic "" \
    --gateway http://<gateway-host>:8080 --engine-id vllm-a
```

The gateway must be reachable from where the bridge runs. Easiest topology: run
vLLM + bridge + gateway on the **same** cloud box (the bridge posts to
`localhost:8080`); expose only the gateway port. `--debug` prints raw vLLM
payloads so you can adjust `translate()` to your vLLM version's field names.

## Local dry run (no GPU)

Validate the whole client path without a GPU: point `run_trace.py` at any
OpenAI-compatible mock, or at the gateway in front of a mock upstream. This
checks request shaping, streaming TTFT measurement, and decision-header capture;
swapping in the real vLLM URL is then a one-line change.

## What to capture

- TTFT p50/p99 (from `run_trace.py`), `policy: dynamo-cost` vs `policy: round-robin`.
- `x-quillcache-*` headers: selected engine, mode, prefill/decode engine ids,
  local hits, transfer/recompute blocks, planner actions, cache actions.
- `/v1/state`: resident KV blocks per engine (with the bridge running).

## Troubleshooting

- **`RuntimeError: Could not find nvcc ... cuda_home='/usr/local/cuda' doesn't exist`**
  on vLLM startup: flashinfer JIT-compiles its sampler kernel and needs a CUDA
  toolkit the slim image lacks. `deploy/modal_vllm.py` sets
  `VLLM_USE_FLASHINFER_SAMPLER=0` (native sampler, no JIT) to avoid it; a CUDA
  *devel* base image (with `nvcc`) is the alternative.
- **First request is slow / returns a 303 redirect with `__modal_function_call_id`:**
  the container is cold-starting (GPU boot + model load, ~1–2 min). Warm it with
  `curl -L <url>/v1/models` and retry.
- **TTFT through the gateway looks wrong:** the gateway now streams the upstream
  body instead of buffering it, so TTFT should reflect the real engine response.
  Check that the upstream vLLM endpoint itself is streaming and that your client
  measures first byte / first SSE chunk instead of full response latency.
- **Cost:** the Modal app scales to zero on idle; `modal app stop quillcache-vllm`
  forces it down.
