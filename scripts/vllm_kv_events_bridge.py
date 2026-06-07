#!/usr/bin/env python3
"""Bridge vLLM KV-cache events into the QuillCache HTTP event API.

Run this script in the same Python environment as vLLM. It subscribes to the
vLLM ZMQ KV event publisher, decodes vLLM's msgpack event batches, and posts a
vendor-neutral JSON batch to QuillCache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from typing import Any


def import_vllm_event_types():
    try:
        import zmq
        from msgspec.msgpack import Decoder
        from vllm.distributed.kv_events import (
            AllBlocksCleared,
            BlockRemoved,
            BlockStored,
            KVEventBatch,
        )
    except Exception as exc:  # pragma: no cover - environment dependent.
        print(
            "failed to import vLLM KV event dependencies. "
            "Run this inside an environment with vLLM, pyzmq, and msgspec.",
            file=sys.stderr,
        )
        raise exc

    return zmq, Decoder, KVEventBatch, BlockStored, BlockRemoved, AllBlocksCleared


def normalize_hash(value: Any) -> str:
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def event_to_json(event: Any, types: tuple[Any, Any, Any]) -> dict[str, Any]:
    block_stored, block_removed, all_blocks_cleared = types
    if isinstance(event, block_stored):
        return {
            "type": "block_stored",
            "block_hashes": [normalize_hash(item) for item in event.block_hashes],
            "parent_block_hash": (
                normalize_hash(event.parent_block_hash)
                if event.parent_block_hash is not None
                else None
            ),
            "token_ids": list(event.token_ids),
            "block_size": int(event.block_size),
            "medium": event.medium or "gpu",
            "lora_name": event.lora_name,
            "group_idx": event.group_idx,
        }
    if isinstance(event, block_removed):
        return {
            "type": "block_removed",
            "block_hashes": [normalize_hash(item) for item in event.block_hashes],
            "medium": event.medium or "gpu",
            "group_idx": event.group_idx,
        }
    if isinstance(event, all_blocks_cleared):
        return {"type": "all_blocks_cleared"}

    raise TypeError(f"unsupported vLLM KV event type: {type(event)!r}")


def post_json(url: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zmq-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--topic", default="kv-events")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8080")
    parser.add_argument("--engine-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--bytes-per-block", type=int, default=4 * 1024 * 1024)
    args = parser.parse_args()

    zmq, decoder_cls, batch_type, block_stored, block_removed, all_blocks_cleared = (
        import_vllm_event_types()
    )
    decoder = decoder_cls(type=batch_type)
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.connect(args.zmq_endpoint)
    socket.setsockopt_string(zmq.SUBSCRIBE, args.topic)

    event_url = args.gateway_url.rstrip("/") + "/v1/kv-events"
    print(
        f"bridging vLLM KV events {args.zmq_endpoint} topic={args.topic!r} "
        f"to {event_url}",
        file=sys.stderr,
    )

    while True:
        parts = socket.recv_multipart()
        payload = parts[-1]
        batch = decoder.decode(payload)
        events = [
            event_to_json(event, (block_stored, block_removed, all_blocks_cleared))
            for event in batch.events
        ]
        out = {
            "engine_id": args.engine_id,
            "ts_ms": int(getattr(batch, "ts", time.time()) * 1000),
            "model_id": args.model_id,
            "tokenizer_id": args.tokenizer_id,
            "tenant_id": args.tenant_id,
            "bytes_per_block": args.bytes_per_block,
            "events": events,
        }
        post_json(event_url, out)


if __name__ == "__main__":
    raise SystemExit(main())
