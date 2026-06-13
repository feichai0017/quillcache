"""A vLLM KV connector on the Mooncake-faithful QuillCache store. SKELETON.

- offload (prefill done): `master.put_start` → WRITE each replica's bytes to its
  `(segment, offset)` over the transfer engine → `master.put_end`.
- load (prefix hit): `master.get_replica_list` — **identity-guarded**, refused
  before any byte moves — → READ a replica over the transfer engine.

The store master + the transfer wire are REAL: run `quillcache store-master` +
`quillcache transfer-node` (see docs/real-engine-pool.md) and the offload/load
round-trip works with no GPU. What needs a GPU + a pinned vLLM version is the
`KVConnectorBase_V1` hooks + the KV-tensor (de)serialization — left as documented
TODOs at the bottom.
"""

from quillcache_store_client import StoreMasterClient, TransferEngineClient, identity


class QuillCacheConnector:
    """QuillCache-backed external KV store for one vLLM worker."""

    def __init__(self, master_url, segment_endpoints, model_id, tokenizer_id, tenant_id="default"):
        self.master = StoreMasterClient(master_url)
        self.transfer = TransferEngineClient()
        # {segment_name: "host:port"} — the transfer node serving each segment.
        self.segment_endpoints = dict(segment_endpoints)
        self.identity = identity(model_id, tokenizer_id, tenant_id)
        # Mount each storage segment on the master (so put_start can allocate on it).
        for name in self.segment_endpoints:
            self.master.mount(name, 1 << 30)

    def offload(self, key, data, replica_num=1):
        """Two-phase Put: allocate replicas, WRITE the bytes to each, commit."""
        buffers = self.master.put_start(key, self.identity, len(data), replica_num)
        for buffer in buffers:
            endpoint = self.segment_endpoints[buffer["segment_name"]]
            self.transfer.write(endpoint, buffer["offset"], data)
        self.master.put_end(key)

    def load(self, key):
        """Identity-guarded Get: locate a replica, READ its bytes. None on miss.

        Raises urllib HTTPError 403 if the request's identity doesn't match the
        writer's (the guard refuses before any byte moves)."""
        replicas = self.master.get_replica_list(key, self.identity)
        for replica in replicas:
            memory = replica["data"].get("Memory")
            if memory:
                endpoint = self.segment_endpoints[memory["segment_name"]]
                return self.transfer.read(endpoint, memory["offset"], memory["size"])
        return None

    # ---- vLLM KVConnectorBase_V1 hooks — wire to YOUR vLLM version ----
    #
    # def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kw):
    #     serialize the freshly computed block → offload(block_key, kv_bytes, ...)
    #     (quantize to FP8 here to match quillcache-cuda's device tier).
    # def start_load_kv(self, forward_context, **kw):
    #     for each prefix block → load(block_key) → copy bytes into vLLM's paged-KV
    #     slots (the HBM<->bytes layout is the GPU-specific part).
    # def get_num_new_matched_tokens(self, request, num_computed_tokens):
    #     sum token_count over the prefix blocks master.get_replica_list resolves,
    #     so the scheduler skips recomputing that prefix.
    # See vllm/distributed/kv_transfer/kv_connector/v1/base.py (version-specific).


if __name__ == "__main__":
    # No-GPU round trip against a running store-master + transfer-node:
    #   quillcache transfer-node --addr 127.0.0.1:8100 --segment seg-0
    #   quillcache store-master  --addr 127.0.0.1:7777
    #   python bridge/vllm_quillcache_connector.py
    import sys

    master_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7777"
    node = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1:8100"
    conn = QuillCacheConnector(
        master_url,
        segment_endpoints={"seg-0": node},
        model_id="Qwen/Qwen2.5-0.5B",
        tokenizer_id="Qwen/Qwen2.5-0.5B",
        tenant_id="demo",
    )
    conn.offload("prefix-block-0", b"fake-kv-bytes-over-the-faithful-store")
    print("loaded back:", conn.load("prefix-block-0"))
