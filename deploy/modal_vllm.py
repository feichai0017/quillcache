"""Serve vLLM (OpenAI-compatible, with KV cache events) on a cloud GPU via Modal.

This is a deploy recipe you run with YOUR Modal account — it cannot be deployed
on your behalf.

    pip install modal
    modal token new                      # one-time auth
    modal deploy deploy/modal_vllm.py    # prints a public https URL

Point a QuillCache gateway engine `base_url` at that URL (see
docs/m3-real-vllm.md). Modal's API and vLLM flags evolve; pin/adjust versions to
match your setup.

KV events: vLLM publishes them over ZMQ *inside* this container. To capture them
precisely, run bridge/vllm_kv_bridge.py as a sidecar in this container (see the
runbook). For a first run you can skip events and just proxy requests to get real
TTFT from bench/run_trace.py.
"""
import os
import subprocess

import modal

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm", "huggingface_hub")
app = modal.App("quillcache-vllm")


@app.function(gpu="L4", image=image, timeout=60 * 60, max_containers=1)
@modal.web_server(8000, startup_timeout=600)
def serve():
    # flashinfer JIT-compiles its sampler kernel at runtime and needs nvcc,
    # which the slim image lacks. Use vLLM's native sampler (no JIT, no nvcc).
    # The model forward (attention) uses a prebuilt backend and is unaffected.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        "4096",
        "--enable-prefix-caching",
        # Tier 2 (precise KV residency): add the two lines below and run
        # bridge/vllm_kv_bridge.py as a sidecar — see docs/m3-real-vllm.md.
        #   "--kv-events-config",
        #   '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}',
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
