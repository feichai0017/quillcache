# MVP Runbook

This runbook starts a QuillCache gateway in front of two vLLM OpenAI-compatible
servers and feeds KV cache events into the residency index.

## 1. Start vLLM Workers

Use a small model for local smoke tests. The important part is that each worker
has a different HTTP port and that KV events are enabled.

```bash
vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8001 \
  --enable-prefix-caching \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557", "topic": "kv-events"}'
```

```bash
vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8002 \
  --enable-prefix-caching \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5558", "topic": "kv-events"}'
```

## 2. Start QuillCache

```bash
cargo run -- gateway --config examples/quillcache-gateway.yaml
```

## 3. Bridge vLLM KV Events

Run one bridge per vLLM worker in the Python environment where vLLM is installed.

```bash
scripts/vllm_kv_events_bridge.py \
  --engine-id vllm-a \
  --model-id Qwen/Qwen3-0.6B \
  --tokenizer-id Qwen/Qwen3-0.6B \
  --zmq-endpoint tcp://127.0.0.1:5557
```

```bash
scripts/vllm_kv_events_bridge.py \
  --engine-id vllm-b \
  --model-id Qwen/Qwen3-0.6B \
  --tokenizer-id Qwen/Qwen3-0.6B \
  --zmq-endpoint tcp://127.0.0.1:5558
```

## 4. Send a Request Through QuillCache

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Write a short explanation of KV cache."}
    ],
    "max_tokens": 64
  }'
```

The response is proxied from the selected vLLM worker. QuillCache adds headers
such as:

- `x-quillcache-engine-id`
- `x-quillcache-request-id`
- `x-quillcache-local-hits`
- `x-quillcache-transfer-blocks`
- `x-quillcache-recompute-blocks`
- `x-quillcache-estimated-ttft-us`

## 5. Inspect Residency

```bash
curl http://127.0.0.1:8080/v1/state | jq
```

The response includes `index.backend`, `index.resident_blocks`, and
`index.persistent`. v0.1 should report the memory backend.

## 6. Synthetic Event Smoke Test

When vLLM is not available, the gateway can still be tested with a synthetic KV
event:

```bash
curl http://127.0.0.1:8080/v1/kv-events \
  -H 'content-type: application/json' \
  -d '{
    "engine_id": "vllm-a",
    "model_id": "Qwen/Qwen3-0.6B",
    "tokenizer_id": "Qwen/Qwen3-0.6B",
    "tenant_id": "default",
    "bytes_per_block": 4194304,
    "events": [
      {
        "type": "block_stored",
        "block_hashes": ["demo-block-0"],
        "parent_block_hash": null,
        "token_ids": [1, 2, 3, 4],
        "block_size": 4,
        "medium": "gpu"
      }
    ]
  }'
```

Then send a request with a QuillCache hint:

```bash
curl -i http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 16,
    "quillcache": {
      "request_id": "demo-req",
      "tokenizer_id": "Qwen/Qwen3-0.6B",
      "tenant_id": "default",
      "block_hashes": ["demo-block-0"],
      "block_tokens": 4
    }
  }'
```

The gateway strips the `quillcache` object before forwarding the request to the
engine.

## Current Limitations

- Exact request-to-vLLM block matching requires client-provided block hints or a
  future tokenizer/block-hash sidecar.
- The gateway consumes KV metadata only; it does not move KV tensors yet.
- The vLLM bridge depends on vLLM's Python event classes and should run in the
  same virtual environment as vLLM.
- SGLang support is planned as a separate event adapter.
