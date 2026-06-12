"""A vLLM KV connector that offloads / loads KV blocks to / from the QuillCache pool.

SKELETON. The vLLM KV-connector API (`vllm.distributed.kv_transfer.kv_connector`)
is version-specific — method names and signatures shift across releases. Wire the
hooks at the bottom to your installed vLLM's `KVConnectorBase_V1`
(`vllm/distributed/kv_transfer/kv_connector/v1/base.py`) and verify against your
version. Run this on the GPU node, alongside vLLM.

The idea (mirrors quillcache_store::PooledStore + EngineConnector, in Python):
  - on prefill done, serialize each computed KV block and `offload` it to the
    node-local QuillCache pool (transfer engine) + block-report to the master;
  - on a new request, ask the master which blocks of the prompt's prefix are
    cached and `load` them — locally first, else cross-node fetch + re-cache —
    so vLLM can skip recomputing that prefix.

The cross-node fetch + local re-cache logic IS implemented here against the
QuillCache wire; only the vLLM-side tensor (de)serialization + hook signatures
are left as TODOs, because those need a real GPU + a pinned vLLM version.
"""

from quillcache_client import (
    MasterClient,
    TransferClient,
    kv_block_key,
    residency,
)


class QuillCacheConnector:
    """QuillCache-backed external KV store for a single vLLM worker (one node)."""

    def __init__(
        self,
        master_url,
        node_id,
        node_transfer_addr,
        model_id,
        tokenizer_id,
        tenant_id="default",
    ):
        self.master = MasterClient(master_url)
        self.transfer = TransferClient()
        self.node_id = node_id
        self.node_addr = node_transfer_addr  # this worker's own transfer server
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.tenant_id = tenant_id
        # Join the pool so other nodes can locate + fetch our blocks.
        self.master.register(node_id, node_transfer_addr)

    def _key(self, prefix_hash, block_hash, block_index, token_count):
        return kv_block_key(
            self.model_id,
            self.tokenizer_id,
            self.tenant_id,
            prefix_hash,
            block_hash,
            block_index,
            token_count,
        )

    # ---- offload: prefill computed a block, push it to the pool ----
    def offload_block(
        self, prefix_hash, block_hash, block_index, token_count, kv_bytes
    ):
        """Write a computed KV block to this node's pool and report it."""
        key = self._key(prefix_hash, block_hash, block_index, token_count)
        self.transfer.write_block(self.node_addr, key, kv_bytes)
        self.master.placed([residency(key, self.node_id, len(kv_bytes))])

    # ---- load: prefix-cache hit, pull a block from the pool ----
    def load_block(self, prefix_hash, block_hash, block_index, token_count):
        """Fetch a KV block: local pool first, else cross-node + re-cache locally.

        Returns the raw KV bytes, or None on a pool-wide miss (vLLM recomputes).
        """
        key = self._key(prefix_hash, block_hash, block_index, token_count)

        # 1) Local pool — the common warm case, no network.
        local = self.transfer.read_block(self.node_addr, key)
        if local is not None:
            return local

        # 2) Ask the master who holds it; fetch from a peer over the transfer wire.
        located = self.master.locate(key).get("nodes", [])
        node_map = self.master.nodes()
        for peer in located:
            if peer == self.node_id:
                continue
            addr = node_map.get(peer)
            if not addr:
                continue
            data = self.transfer.read_block(addr, key)
            if data is not None:
                # Re-cache locally + report, so the next hit on this node is local.
                self.transfer.write_block(self.node_addr, key, data)
                self.master.placed([residency(key, self.node_id, len(data))])
                return data

        # 3) Pool-wide miss.
        return None

    # ---- vLLM KVConnectorBase_V1 hooks — wire to YOUR vLLM version ----
    #
    # The methods below are the integration points. Names/signatures differ by
    # vLLM release; consult vllm/distributed/kv_transfer/kv_connector/v1/base.py.
    #
    # def get_num_new_matched_tokens(self, request, num_computed_tokens):
    #     """How many of `request`'s prefix tokens the pool can serve, so the
    #     scheduler skips recomputing them. Sum token_count over located blocks:
    #         hit = 0
    #         for blk in self._blocks_of(request):
    #             if self.master.locate(self._key(*blk))["nodes"]:
    #                 hit += blk.token_count
    #         return hit, False  # (matched_tokens, load_async)
    #     """
    #
    # def start_load_kv(self, forward_context, **kw):
    #     """Before forward: for each prefix block, load_block(...) and copy the
    #     bytes into the paged-KV slots vLLM allocated. The HBM<->bytes layout is
    #     vLLM-version-specific — this is where a real GPU is required."""
    #
    # def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kw):
    #     """After prefill: serialize freshly computed blocks and offload_block(...).
    #     Quantize to FP8 on offload here to match quillcache-cuda's device tier."""
    #
    # def wait_for_save(self):
    #     """Block until outstanding offloads complete (we write synchronously, so
    #     this is a no-op unless you make offload_block async)."""


if __name__ == "__main__":
    # Illustrative round-trip against a running master + this node's transfer
    # server (no vLLM, no GPU): offload a block, then load it back.
    import sys

    master_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7777"
    node_addr = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1:7001"
    conn = QuillCacheConnector(
        master_url,
        node_id="node-demo",
        node_transfer_addr=node_addr,
        model_id="Qwen/Qwen2.5-0.5B",
        tokenizer_id="Qwen/Qwen2.5-0.5B",
        tenant_id="demo",
    )
    conn.offload_block("pfx", "blk0", 0, 64, b"fake-kv-bytes")
    got = conn.load_block("pfx", "blk0", 0, 64)
    print("loaded back:", got)
