"""QuillCache Python client — talk to the master (HTTP) and to a node's transfer
engine (TCP), so a real engine (vLLM) can offload / fetch KV blocks to / from the
QuillCache distributed pool.

Stdlib only (socket, struct, json, urllib) so it runs inside a vLLM container.
The transfer wire protocol mirrors crates/quillcache-store/src/transfer.rs:

    read  : [u8 0][u32 keylen][key_json]                 -> [u8 status][u64 len][data]
    write : [u8 1][u32 keylen][key_json][u64 len][data]  -> [u8 status]

All integers are big-endian; status 0 = ok, 1 = not_found, 2 = error.
"""

import json
import socket
import struct
import urllib.request

OP_READ, OP_WRITE = 0, 1
ST_OK, ST_NOTFOUND = 0, 1


def kv_block_key(
    model_id,
    tokenizer_id,
    tenant_id,
    prefix_hash,
    block_hash,
    block_index=0,
    token_count=64,
    adapter_id=None,
):
    """Build a KvBlockKey dict matching quillcache_core::KvBlockKey's JSON."""
    return {
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "adapter_id": adapter_id,
        "tenant_id": tenant_id,
        "prefix_hash": prefix_hash,
        "block_hash": block_hash,
        "block_index": block_index,
        "token_count": token_count,
    }


def residency(key, node_id, num_bytes, tier="CpuDram"):
    """Build a CacheResidency dict for /v1/placed."""
    return {
        "key": key,
        "worker_id": node_id,
        "tier": tier,
        "bytes": num_bytes,
        "last_access_ms": 0,
        "ref_count": 0,
        "pinned": False,
    }


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("transfer peer closed the connection")
        buf += chunk
    return buf


class TransferClient:
    """The node-local transfer-engine wire (TCP): offload / fetch raw KV bytes."""

    def write_block(self, node_addr, key, data):
        host, port = node_addr.rsplit(":", 1)
        kb = json.dumps(key).encode()
        with socket.create_connection((host, int(port))) as sock:
            sock.sendall(
                struct.pack(">BI", OP_WRITE, len(kb))
                + kb
                + struct.pack(">Q", len(data))
                + data
            )
            return _recv_exact(sock, 1)[0] == ST_OK

    def read_block(self, node_addr, key):
        host, port = node_addr.rsplit(":", 1)
        kb = json.dumps(key).encode()
        with socket.create_connection((host, int(port))) as sock:
            sock.sendall(struct.pack(">BI", OP_READ, len(kb)) + kb)
            status = _recv_exact(sock, 1)[0]
            if status == ST_NOTFOUND:
                return None
            if status != ST_OK:
                raise IOError("transfer server reported an error")
            (length,) = struct.unpack(">Q", _recv_exact(sock, 8))
            return _recv_exact(sock, length)


class MasterClient:
    """The master metadata service (HTTP): register, block-report, locate."""

    def __init__(self, base_url):
        self.base = base_url.rstrip("/")

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read() or "null")

    def _get(self, path):
        with urllib.request.urlopen(self.base + path) as resp:
            return json.loads(resp.read())

    def register(self, node_id, transfer_addr):
        return self._post(
            "/v1/register", {"node_id": node_id, "transfer_addr": transfer_addr}
        )

    def placed(self, residencies):
        return self._post("/v1/placed", residencies)

    def locate(self, key):
        # -> {"nodes": [...], "residencies": [...]}
        return self._post("/v1/locate", {"key": key})

    def nodes(self):
        return self._get("/v1/nodes")

    def state(self):
        return self._get("/v1/state")


if __name__ == "__main__":
    # Smoke test against a running `quillcache master` + a node's transfer server.
    # (Start: `quillcache master` and a node, e.g. `quillcache cluster`-style.)
    import sys

    master = MasterClient(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7777")
    print("nodes:", master.nodes())
    print("state:", master.state())
