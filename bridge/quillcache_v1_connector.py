"""A REAL vLLM v1 KV connector backed by the QuillCache store.

This subclasses vLLM's `KVConnectorBase_V1` (verified against the exact API of
the deployed vllm 0.22.1 — see deploy/modal_vllm_introspect.py) and is modelled
on vLLM's own reference `ExampleConnector`. It keeps vLLM's paged-KV slot-mapping
extract/inject logic *verbatim* (so the GPU tensor layout is correct for MLA /
Triton / default attention backends) and swaps the reference's
safetensors-on-local-disk for the **QuillCache distributed store**:

  - save  (offload): extract a layer's KV for the new prefix blocks → serialize
    (safetensors in-memory) → two-phase Put into the store (`put_start` → WRITE
    each replica over the transfer engine → `put_end`).
  - load  (prefix hit): **identity-guarded** `get_replica_list` (refused, HTTP
    403, *before any byte moves*, if the requester's identity doesn't match the
    writer's) → READ a replica over the transfer engine → deserialize → inject
    into vLLM's paged KV buffer.

The store master + transfer wire are the same real, tested QuillCache services
used by docs/real-engine-pool.md. The identity guard is QuillCache's
differentiator over a vanilla shared-storage connector: cross-tenant / cross-model
KV reuse is refused at the store, not merely by convention.

Two operating modes, selected by KVTransferConfig.kv_role:

  - `kv_both` (default): single-pool, **content-addressed** reuse. A prefix's KV
    is keyed by `qc/{identity+prompt-hash}`; any later request with the same
    prefix transparently loads it. This is the prefix-cache offload path.
  - `kv_producer` / `kv_consumer`: **true mid-request disaggregation** (vLLM-native
    P/D). A router mints a `transfer_id` per request and threads it through
    `kv_transfer_params`: the prefill instance (producer, `do_remote_decode`)
    offloads the request's KV under `qc-pd/{transfer_id}` and does no decode; the
    decode instance (consumer, `do_remote_prefill`) pulls that KV by id and skips
    prefill. The store is the rendezvous, so — unlike a direct RDMA-pull connector
    — the producer frees its blocks immediately (no delay-free) and the consumer
    loads synchronously. See src/pd_proxy.rs for the router that mints the id.

Run it (see deploy/modal_vllm_connector.py for the full Modal recipe):

    vllm serve <model> \
      --no-enable-prefix-caching \           # force prefix hits to come via the store
      --disable-hybrid-kv-cache-manager \    # this connector is not HMA-aware (like ExampleConnector)
      --kv-transfer-config '{
        "kv_connector": "QuillCacheV1Connector",
        "kv_connector_module_path": "quillcache_v1_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "master_url": "http://127.0.0.1:7777",
          "segment_endpoints": {"seg-0": "127.0.0.1:8100"},
          "tenant_id": "default",
          "replica_num": 1
        }
      }'
"""

import concurrent.futures
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import safetensors.torch
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.utils.hashing import safe_hash
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

# The QuillCache store clients (stdlib; the master over HTTP + the transfer wire over TCP).
from quillcache_store_client import StoreMasterClient, TransferEngineClient, identity

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    """Largest block-aligned token count strictly below `num_tokens` (vLLM's rule)."""
    return max(0, (num_tokens - 1) // block_size * block_size)


def _prefix_hash(token_ids: list[int], block_size: int, mm_hashes: list[str]) -> tuple[str, int]:
    """Canonical (hash, aligned_len) for the block-aligned prompt prefix.

    Computed identically scheduler-side (match check + meta build) so save and
    load address the same store keys. Mirrors ExampleConnector's foldername hash.
    """
    aligned = align_to_block_size(len(token_ids), block_size)
    token_bytes = torch.tensor(token_ids[:aligned], dtype=torch.long).numpy().tobytes()
    if mm_hashes:
        token_bytes += ("-".join(mm_hashes)).encode("utf-8")
    digest = safe_hash(token_bytes, usedforsecurity=False).hexdigest()
    return digest, aligned


@dataclass
class ReqMeta:
    # Block-aligned prompt-prefix tokens this op covers.
    token_ids: torch.Tensor
    # Paged-KV slot for each token (same length as token_ids).
    slot_mapping: torch.Tensor
    # True = save (offload) this prefix; False = load it from the store.
    is_store: bool
    # The store-key namespace for this op's layers + manifest. Either content-
    # addressed (`qc/{identity-prefix-hash}`, for transparent prefix reuse) or
    # disaggregation-handshake (`qc-pd/{transfer_id}`, for true mid-request P/D
    # where a router-minted transfer_id — not the prompt content — names the KV).
    key_prefix: str

    @staticmethod
    def make_meta(
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        key_prefix: str,
    ) -> "ReqMeta":
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            key_prefix=key_prefix,
        )


@dataclass
class QuillCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        key_prefix: str,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, is_store, key_prefix)
        )


class QuillCacheV1Connector(KVConnectorBase_V1):
    """vLLM v1 KV connector whose external store is the QuillCache pool."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size

        cfg = self._kv_transfer_config
        model = vllm_config.model_config.model
        tokenizer = getattr(vllm_config.model_config, "tokenizer", None) or model
        # Identity = QuillCache IdentityScope. The store enforces it on every Get.
        self._identity = identity(
            model_id=cfg.get_from_extra_config("model_id", model),
            tokenizer_id=cfg.get_from_extra_config("tokenizer_id", tokenizer),
            tenant_id=cfg.get_from_extra_config("tenant_id", "default"),
            adapter_id=cfg.get_from_extra_config("adapter_id", None),
        )
        self._master = StoreMasterClient(
            cfg.get_from_extra_config("master_url", "http://127.0.0.1:7777")
        )
        self._transfer = TransferEngineClient()
        # {segment_name: "host:port"} — the transfer node serving each segment.
        endpoints = cfg.get_from_extra_config("segment_endpoints", {"seg-0": "127.0.0.1:8100"})
        if isinstance(endpoints, str):
            endpoints = json.loads(endpoints)
        self._segment_endpoints = dict(endpoints)
        self._replica_num = int(cfg.get_from_extra_config("replica_num", 1))

        # Disaggregation role (vLLM KVTransferConfig.kv_role), using vLLM's own
        # semantics: a pure prefill instance is `kv_producer` (saves a request's
        # KV for a remote decode), a pure decode instance is `kv_consumer` (loads
        # it), and `kv_both` is BOTH. The disagg handshake only fires when the
        # router sets do_remote_* in kv_transfer_params; absent that, a kv_both
        # instance falls through to single-pool, content-addressed reuse.
        kv_role = getattr(cfg, "kv_role", None) or "kv_both"
        self._is_producer = bool(getattr(cfg, "is_kv_producer", False))
        self._is_consumer = bool(getattr(cfg, "is_kv_consumer", False))

        # Scheduler-side: requests for which a prefix hit was found and blocks
        # were allocated, so the worker must load them next forward pass.
        self._requests_need_load: dict[str, Request] = {}
        # Scheduler-side disagg handshake: req_id -> router-minted transfer_id.
        # `_disagg_save` = producer must offload this prefill's KV under the id;
        # `_disagg_load` = consumer must pull the KV named by the id (not content).
        self._disagg_save: dict[str, str] = {}
        self._disagg_load: dict[str, str] = {}
        # Worker-side: key-prefixes saved this step -> aligned token count (for manifest).
        self._saved_this_step: dict[str, int] = {}
        # Worker-side layer-wise-overlap load state: per-layer in-flight prefetches,
        # plus the forward context captured at start_load_kv and consumed per layer
        # in wait_for_layer_load. The executor is created on the worker only.
        self._pending_layer_loads: dict[str, list] = {}
        self._load_forward_context: "ForwardContext | None" = None
        self._load_attn_metadata: Any = None
        self._load_executor: "concurrent.futures.ThreadPoolExecutor | None" = None

        # The worker owns byte movement; it registers the storage segments on the
        # master so put_start can allocate on them. Idempotent-tolerant.
        if role == KVConnectorRole.WORKER:
            for name in self._segment_endpoints:
                try:
                    self._master.mount(name, 1 << 30)
                except Exception as e:  # already mounted, or master not up yet
                    logger.info("segment %s mount skipped: %s", name, e)
            # A single background fetch thread prefetches each layer's KV while the
            # forward pass computes the previous layer, so KV load hides behind
            # compute instead of blocking it. One worker keeps the store clients
            # single-threaded (no concurrent-access assumption); raise max_workers
            # only if StoreMasterClient/TransferEngineClient are thread-safe (then
            # layers also fetch in parallel for bandwidth).
            self._load_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="qc-kv-load"
            )

        logger.info(
            "QuillCacheV1Connector role=%s kv_role=%s(producer=%s consumer=%s) master=%s segments=%s identity=%s",
            role,
            kv_role,
            self._is_producer,
            self._is_consumer,
            self._master.base,
            list(self._segment_endpoints),
            self._identity,
        )

    # ==============================
    # Store primitives (the real QuillCache data path)
    # ==============================

    @staticmethod
    def _content_prefix(prefix_hash: str) -> str:
        """Key namespace for transparent, content-addressed prefix reuse."""
        return f"qc/{prefix_hash}"

    @staticmethod
    def _pd_prefix(transfer_id: str) -> str:
        """Key namespace for a disagg P/D handshake (router-minted transfer_id)."""
        return f"qc-pd/{transfer_id}"

    def _layer_key(self, key_prefix: str, layer_name: str) -> str:
        return f"{key_prefix}/{layer_name}"

    def _manifest_key(self, key_prefix: str) -> str:
        return f"{key_prefix}/__manifest__"

    def _put_bytes(self, key: str, data: bytes) -> None:
        """Two-phase Put: allocate replica buffers, WRITE each, commit."""
        buffers = self._master.put_start(key, self._identity, len(data), self._replica_num)
        for buffer in buffers:
            endpoint = self._segment_endpoints[buffer["segment_name"]]
            self._transfer.write(endpoint, buffer["offset"], data)
        self._master.put_end(key)

    def _get_bytes(self, key: str) -> bytes | None:
        """Identity-guarded Get: locate a replica, READ its bytes. None on miss.

        The store refuses (HTTP 403) before any byte moves if this connector's
        identity doesn't match the writer's — surfaced here as a logged miss."""
        try:
            replicas = self._master.get_replica_list(key, self._identity)
        except Exception as e:
            code = getattr(e, "code", None)
            if code == 403:
                logger.warning("QuillCache identity guard REFUSED reuse of %s", key)
            elif code not in (404,):
                logger.warning("get_replica_list(%s) failed: %s", key, e)
            return None
        for replica in replicas:
            memory = (replica.get("data") or {}).get("Memory")
            if memory:
                endpoint = self._segment_endpoints[memory["segment_name"]]
                return self._transfer.read(endpoint, memory["offset"], memory["size"])
        return None

    def _exists(self, key: str) -> bool:
        """Does this prefix's commit manifest exist & is it ours? (no byte move)."""
        try:
            return bool(self._master.get_replica_list(key, self._identity))
        except Exception as e:
            if getattr(e, "code", None) == 403:
                logger.warning("QuillCache identity guard REFUSED match on %s", key)
            return False

    @staticmethod
    def _serialize(t: torch.Tensor) -> bytes:
        return safetensors.torch.save({"kv": t.detach().cpu().contiguous()})

    @staticmethod
    def _deserialize(buf: bytes, device: str) -> torch.Tensor:
        return safetensors.torch.load(buf)["kv"].to(device)

    # ==============================
    # Worker-side methods
    # ==============================

    def _inject_kv_into_layer(
        self,
        dst_kv_cache_layer: torch.Tensor,
        src_kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> None:
        """Inject one layer's store-resident KV into vLLM's paged buffer. Verbatim
        from vLLM's ExampleConnector — layout-correct per attention backend."""
        dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages = dst_kv_cache_layer_shape[0]
            page_size = dst_kv_cache_layer_shape[1]
            dst_kv_cache_layer = dst_kv_cache_layer.reshape(num_pages * page_size, -1)
            dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
        elif isinstance(attn_metadata, TritonAttentionMetadata):
            block_idxs = slot_mapping // self._block_size
            offsets = slot_mapping % self._block_size
            dst_kv_cache_layer[block_idxs, :, offsets] = src_kv_cache
        else:
            num_pages = dst_kv_cache_layer_shape[1]
            page_size = dst_kv_cache_layer_shape[2]
            dst_kv_cache_layer = dst_kv_cache_layer.reshape(2, num_pages * page_size, -1)
            dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Kick off **layer-wise overlapped** loads: prefetch each layer's KV on a
        background thread so the forward pass can compute layer i while layer i+1's
        KV is still being fetched. Injection happens per layer in
        `wait_for_layer_load`, which blocks only on *that* layer — so KV load hides
        behind compute instead of being one synchronous barrier before the pass.
        Mirrors `slice_pool::run_layers_with_notify` on the Rust transfer side."""
        self._pending_layer_loads = {}
        self._load_forward_context = forward_context
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, QuillCacheConnectorMetadata)
        n_load = sum(1 for r in metadata.requests if not r.is_store)
        attn_metadata = forward_context.attn_metadata
        self._load_attn_metadata = attn_metadata
        if attn_metadata is None:
            if n_load:
                logger.warning(
                    "QC start_load_kv: attn_metadata is None but load_reqs=%d — load SKIPPED",
                    n_load,
                )
            return
        if self._load_executor is None:  # scheduler-side connector never loads bytes
            return

        # Submit one prefetch per (load request, attention layer). They run on the
        # background fetch thread while this forward pass proceeds; each is awaited
        # just-in-time in wait_for_layer_load.
        for request in metadata.requests:
            if request.is_store:
                continue
            logger.warning(
                "QC layer-wise loading %d tokens of KV from the store (%s)",
                len(request.slot_mapping),
                request.key_prefix,
            )
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                if getattr(layer, "kv_cache", None) is None:
                    continue  # skip non-attention layers (MLP/MoE)
                key = self._layer_key(request.key_prefix, layer_name)
                fut = self._load_executor.submit(self._get_bytes, key)
                self._pending_layer_loads.setdefault(layer_name, []).append((request, fut))

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until *this* layer's prefetch lands, then inject it — vLLM calls this
        right before the layer is computed, so earlier layers already overlapped their
        fetch with compute. Blocks on one layer, not the whole blob."""
        pending = self._pending_layer_loads.pop(layer_name, None)
        if not pending:
            return
        forward_context = self._load_forward_context
        attn_metadata = self._load_attn_metadata
        # Preserve the original behaviour: only the dict-metadata path injects.
        if forward_context is None or not isinstance(attn_metadata, dict):
            return
        layer = forward_context.no_compile_layers.get(layer_name)
        kv_cache_layer = getattr(layer, "kv_cache", None) if layer is not None else None
        if kv_cache_layer is None:
            return
        for request, fut in pending:
            buf = fut.result()  # blocks ONLY on this layer's fetch
            if buf is None:
                continue  # miss / evicted / refused — vLLM recomputes this layer
            kv_cache = self._deserialize(buf, str(kv_cache_layer.device))
            self._inject_kv_into_layer(
                kv_cache_layer, kv_cache, request.slot_mapping, attn_metadata[layer_name]
            )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Extract a layer's KV for each store request and offload it to the pool."""

        def extract_kv_from_layer(layer: torch.Tensor, slot_mapping: torch.Tensor) -> torch.Tensor:
            # Verbatim from vLLM's ExampleConnector — inverse of inject.
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
            elif isinstance(attn_metadata, TritonAttentionMetadata):
                block_idxs = slot_mapping // self._block_size
                offsets = slot_mapping % self._block_size
                return layer[block_idxs, :, offsets]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, QuillCacheConnectorMetadata)
        for request in metadata.requests:
            if not request.is_store:
                continue
            kv_cache = extract_kv_from_layer(kv_layer, request.slot_mapping)
            self._put_bytes(self._layer_key(request.key_prefix, layer_name), self._serialize(kv_cache))
            self._saved_this_step[request.key_prefix] = int(request.token_ids.numel())

    def wait_for_save(self) -> None:
        """Commit a manifest per saved prefix — the marker a later match check reads.

        For a disagg producer (`qc-pd/...`) the consumer pulls by transfer_id and
        ignores the manifest, but writing it is harmless and keeps the save path
        uniform (and records the token count for observability)."""
        for key_prefix, num_tokens in self._saved_this_step.items():
            manifest = json.dumps({"tokens": num_tokens, "block_size": self._block_size}).encode()
            self._put_bytes(self._manifest_key(key_prefix), manifest)
            logger.warning(
                "QC committed %d-token prefix to the store (%s)",
                num_tokens,
                key_prefix,
            )
        self._saved_this_step.clear()

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """How many *additional* prefix tokens the store can serve for this request."""
        token_ids = list(request.prompt_token_ids or [])

        # --- disagg consumer (decode side): the router told us this request's
        # prefill was done remotely (`do_remote_prefill`) and named the KV with a
        # `transfer_id`. Claim the whole block-aligned prefix by id — no content
        # match, no manifest lookup; the producer guarantees it's in the store. ---
        params = request.kv_transfer_params or {}
        if self._is_consumer and params.get("do_remote_prefill"):
            transfer_id = params.get("transfer_id")
            aligned = align_to_block_size(len(token_ids), self._block_size)
            count = aligned - num_computed_tokens
            if transfer_id and count > 0:
                self._disagg_load[request.request_id] = str(transfer_id)
                logger.warning(
                    "QC disagg consumer: pull %d tokens for transfer_id=%s (req=%s)",
                    count,
                    transfer_id,
                    getattr(request, "request_id", "?"),
                )
                return count, False
            return 0, False

        mm_hashes = [f.identifier for f in request.mm_features]
        prefix_hash, aligned = _prefix_hash(token_ids, self._block_size, mm_hashes)
        exists = self._exists(self._manifest_key(self._content_prefix(prefix_hash)))
        logger.warning(
            "QC match-check req=%s prefix=%s manifest=%s num_computed=%d aligned=%d block_size=%d ntok=%d",
            getattr(request, "request_id", "?"),
            prefix_hash[:12],
            exists,
            num_computed_tokens,
            aligned,
            self._block_size,
            len(token_ids),
        )
        if aligned <= num_computed_tokens or not exists:
            return 0, False
        logger.warning(
            "QC external cache HIT prefix=%s (+%d tokens)",
            prefix_hash[:12],
            aligned - num_computed_tokens,
        )
        # Synchronous load (we inject during the forward pass), so async=False.
        return aligned - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        # --- disagg producer (prefill side): the router asked us to make this
        # request's KV available for a remote decode (`do_remote_decode`) under a
        # `transfer_id`. Remember it so build_connector_meta emits a save op. ---
        params = request.kv_transfer_params or {}
        if self._is_producer and params.get("do_remote_decode"):
            transfer_id = params.get("transfer_id")
            if transfer_id:
                self._disagg_save[request.request_id] = str(transfer_id)
            else:
                logger.warning("QC disagg producer: do_remote_decode with no transfer_id")
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = QuillCacheConnectorMetadata()
        total_need_load = 0

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            req_id = new_req.req_id

            # --- disagg producer: offload this prefill's KV under transfer_id. ---
            if req_id in self._disagg_save:
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=True,
                    key_prefix=self._pd_prefix(self._disagg_save[req_id]),
                )
                continue

            # --- disagg consumer: pull the KV named by transfer_id, skip prefill. ---
            if req_id in self._disagg_load:
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                    key_prefix=self._pd_prefix(self._disagg_load[req_id]),
                )
                total_need_load += 1
                continue

            # --- content-addressed reuse (single-pool / kv_both default). ---
            mm_hashes = [f.identifier for f in new_req.mm_features]
            prefix_hash, _ = _prefix_hash(token_ids, self._block_size, mm_hashes)
            if req_id in self._requests_need_load:
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                    key_prefix=self._content_prefix(prefix_hash),
                )
                total_need_load += 1
            elif not self._exists(self._manifest_key(self._content_prefix(prefix_hash))):
                # Not already in the store -> save this prefix after the forward.
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=True,
                    key_prefix=self._content_prefix(prefix_hash),
                )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids
            if not resumed_from_preemption or req_id not in self._requests_need_load:
                continue
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            new_block_ids = cached_reqs.new_block_ids[i]
            request = self._requests_need_load[req_id]
            total_tokens = num_computed_tokens + num_new_tokens
            token_ids = request.all_token_ids[:total_tokens]
            assert new_block_ids is not None
            # A resumed disagg consumer keeps its transfer_id key; otherwise content.
            if req_id in self._disagg_load:
                key_prefix = self._pd_prefix(self._disagg_load[req_id])
            else:
                mm_hashes = [f.identifier for f in request.mm_features]
                prefix_hash, _ = _prefix_hash(list(token_ids), self._block_size, mm_hashes)
                key_prefix = self._content_prefix(prefix_hash)
            meta.add_request(
                token_ids=list(token_ids),
                block_ids=new_block_ids[0],
                block_size=self._block_size,
                is_store=False,
                key_prefix=key_prefix,
            )
            total_need_load += 1

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        self._disagg_save.clear()
        self._disagg_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Disagg producer: confirm the handshake back to the router.

        The prefill's KV is durably in the store (committed during wait_for_save),
        so we DON'T delay-free the prefill blocks (`False`) — unlike a direct
        RDMA-pull connector that must hold blocks until the decode side reads
        them. We return the handshake (`do_remote_prefill` + `transfer_id`) so a
        router that relays the response's kv_transfer_params can route the decode.
        """
        params = request.kv_transfer_params or {}
        if self._is_producer and params.get("do_remote_decode") and params.get("transfer_id"):
            return False, {
                "do_remote_prefill": True,
                "transfer_id": params["transfer_id"],
            }
        return False, None
