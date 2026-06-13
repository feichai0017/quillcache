"""QuillCache Python client for the Mooncake-faithful store — talk to the store
`MasterService` (HTTP) and to a Transfer Engine storage node (the `(segment,
offset)` TCP wire). Stdlib only, so it runs inside a vLLM container.

Two services back this (run them with the `quillcache` binary):
  - `quillcache store-master`   — the MasterService (two-phase Put, identity-guarded Get)
  - `quillcache transfer-node`  — a storage node serving one named RAM segment

The transfer wire mirrors quillcache-transfer-engine/src/transport/tcp.rs:
    read  : [u8 0][u64 offset][u64 len]            -> [u8 status][u64 len][data]
    write : [u8 1][u64 offset][u64 len][data]      -> [u8 status]
all integers big-endian; status 0 = ok, 2 = error.
"""

import json
import socket
import struct
import urllib.request

OP_READ, OP_WRITE = 0, 1
ST_OK = 0


def identity(model_id, tokenizer_id, tenant_id, adapter_id=None):
    """An IdentityScope dict matching quillcache_core::IdentityScope's JSON."""
    return {
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "adapter_id": adapter_id,
        "tenant_id": tenant_id,
    }


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("transfer node closed the connection")
        buf += chunk
    return buf


class TransferEngineClient:
    """Move bytes to / from a storage node's RAM segment by `(offset, length)`."""

    def read(self, endpoint, offset, length):
        host, port = endpoint.rsplit(":", 1)
        with socket.create_connection((host, int(port))) as sock:
            sock.sendall(struct.pack(">BQQ", OP_READ, offset, length))
            status = _recv_exact(sock, 1)[0]
            if status != ST_OK:
                raise IOError("transfer node read out of segment bounds")
            (n,) = struct.unpack(">Q", _recv_exact(sock, 8))
            return _recv_exact(sock, n)

    def write(self, endpoint, offset, data):
        host, port = endpoint.rsplit(":", 1)
        with socket.create_connection((host, int(port))) as sock:
            sock.sendall(struct.pack(">BQQ", OP_WRITE, offset, len(data)) + data)
            if _recv_exact(sock, 1)[0] != ST_OK:
                raise IOError("transfer node write failed")


class StoreMasterClient:
    """The store MasterService over HTTP (`quillcache store-master`)."""

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

    def mount(self, name, capacity):
        return self._post("/v1/mount", {"name": name, "capacity": capacity})

    def put_start(self, key, identity_scope, size, replica_num=1):
        # -> [{"segment_name", "offset", "size"}, ...] (the allocated replica buffers)
        return self._post(
            "/v1/put_start",
            {"key": key, "identity": identity_scope, "size": size, "replica_num": replica_num},
        )["buffers"]

    def put_end(self, key):
        return self._post("/v1/put_end", {"key": key})

    def put_revoke(self, key):
        return self._post("/v1/put_revoke", {"key": key})

    def get_replica_list(self, key, identity_scope):
        # -> [{"id","status","ref_count","data":{"Memory":{segment_name,offset,size}}}, ...]
        # Raises urllib HTTPError 403 if the identity guard refuses the request.
        return self._post(
            "/v1/get_replica_list", {"key": key, "identity": identity_scope}
        )["replicas"]

    def remove(self, key, force=False):
        return self._post("/v1/remove", {"key": key, "force": force})

    def state(self):
        with urllib.request.urlopen(self.base + "/v1/state") as resp:
            return json.loads(resp.read())
