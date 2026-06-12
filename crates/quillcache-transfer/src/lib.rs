//! Transfer engine seam — move KV blocks between nodes.
//!
//! Mooncake's transfer engine moves KV tensors over RDMA / TCP / NVLink with a
//! pooled, zero-copy, topology-aware data path. This crate is the same seam as a
//! Rust trait: the store depends only on `dyn Transfer`, so the wire backend is
//! swappable without touching the data path. [`LocalTransfer`] and
//! [`TcpTransfer`] work today on any machine; the RDMA backend (behind the
//! `rdma` feature) is the reserved interface, stubbed until a NIC is wired —
//! when it lands, nothing above this trait changes.

use async_trait::async_trait;
use bytes::Bytes;
use quillcache_core::{CacheTier, KvBlockKey};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// The physical link a transfer rides, so the control plane's cost model can
/// price the path (HBM hit < NVLink < RDMA < TCP < recompute).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkClass {
    /// Same machine, shared memory.
    LocalShm,
    /// Intra-node GPU↔GPU.
    Nvlink,
    /// Inter-node RDMA over InfiniBand.
    RdmaIb,
    /// Inter-node RDMA over Converged Ethernet.
    RdmaRoce,
    /// Inter-node TCP (the always-available fallback).
    Tcp,
}

impl LinkClass {
    /// The residency tier the cost model prices this link as.
    pub fn as_tier(self) -> CacheTier {
        match self {
            LinkClass::LocalShm | LinkClass::Nvlink => CacheTier::RemoteHbm,
            LinkClass::RdmaIb | LinkClass::RdmaRoce => CacheTier::RemoteHbm,
            LinkClass::Tcp => CacheTier::CpuDram,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            LinkClass::LocalShm => "local-shm",
            LinkClass::Nvlink => "nvlink",
            LinkClass::RdmaIb => "rdma-ib",
            LinkClass::RdmaRoce => "rdma-roce",
            LinkClass::Tcp => "tcp",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TransferError {
    #[error("block not found on remote")]
    NotFound,
    #[error("io: {0}")]
    Io(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
}

impl From<std::io::Error> for TransferError {
    fn from(err: std::io::Error) -> Self {
        TransferError::Io(err.to_string())
    }
}

/// A node address. `host:port` for TCP; an opaque id for the in-process backend.
pub type NodeAddr = String;

/// Move a KV block's bytes to / from a remote node. Backends differ only in the
/// wire (shared-memory, TCP, RDMA); the store sees one trait.
#[async_trait]
pub trait Transfer: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    fn link_class(&self) -> LinkClass;
    async fn read(&self, remote: &NodeAddr, key: &KvBlockKey) -> Result<Bytes, TransferError>;
    async fn write(
        &self,
        remote: &NodeAddr,
        key: &KvBlockKey,
        data: Bytes,
    ) -> Result<(), TransferError>;
}

/// In-process transfer over a shared map. The fast path for same-machine moves
/// and the deterministic backend for tests — no sockets, no serialization.
#[derive(Clone, Debug, Default)]
pub struct LocalTransfer {
    inner: Arc<Mutex<HashMap<(NodeAddr, KvBlockKey), Bytes>>>,
}

impl LocalTransfer {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Transfer for LocalTransfer {
    fn name(&self) -> &str {
        "local"
    }

    fn link_class(&self) -> LinkClass {
        LinkClass::LocalShm
    }

    async fn read(&self, remote: &NodeAddr, key: &KvBlockKey) -> Result<Bytes, TransferError> {
        self.inner
            .lock()
            .unwrap()
            .get(&(remote.clone(), key.clone()))
            .cloned()
            .ok_or(TransferError::NotFound)
    }

    async fn write(
        &self,
        remote: &NodeAddr,
        key: &KvBlockKey,
        data: Bytes,
    ) -> Result<(), TransferError> {
        self.inner
            .lock()
            .unwrap()
            .insert((remote.clone(), key.clone()), data);
        Ok(())
    }
}

// ---- TCP backend: a length-prefixed request/response over a plain socket ----

const OP_READ: u8 = 0;
const OP_WRITE: u8 = 1;
const ST_OK: u8 = 0;
const ST_NOTFOUND: u8 = 1;
const ST_ERR: u8 = 2;

/// TCP transfer client. Connects per operation to a node running
/// [`serve_listener`]. The always-available backend before RDMA hardware: real
/// bytes really cross a socket, measurable on any two machines / cloud VMs.
#[derive(Clone, Debug, Default)]
pub struct TcpTransfer;

#[async_trait]
impl Transfer for TcpTransfer {
    fn name(&self) -> &str {
        "tcp"
    }

    fn link_class(&self) -> LinkClass {
        LinkClass::Tcp
    }

    async fn read(&self, remote: &NodeAddr, key: &KvBlockKey) -> Result<Bytes, TransferError> {
        let mut sock = TcpStream::connect(remote).await?;
        let key_bytes =
            serde_json::to_vec(key).map_err(|e| TransferError::Protocol(e.to_string()))?;
        sock.write_u8(OP_READ).await?;
        sock.write_u32(key_bytes.len() as u32).await?;
        sock.write_all(&key_bytes).await?;
        sock.flush().await?;
        match sock.read_u8().await? {
            ST_OK => {
                let len = sock.read_u64().await? as usize;
                let mut buf = vec![0u8; len];
                sock.read_exact(&mut buf).await?;
                Ok(Bytes::from(buf))
            }
            ST_NOTFOUND => Err(TransferError::NotFound),
            _ => Err(TransferError::Protocol("remote reported an error".into())),
        }
    }

    async fn write(
        &self,
        remote: &NodeAddr,
        key: &KvBlockKey,
        data: Bytes,
    ) -> Result<(), TransferError> {
        let mut sock = TcpStream::connect(remote).await?;
        let key_bytes =
            serde_json::to_vec(key).map_err(|e| TransferError::Protocol(e.to_string()))?;
        sock.write_u8(OP_WRITE).await?;
        sock.write_u32(key_bytes.len() as u32).await?;
        sock.write_all(&key_bytes).await?;
        sock.write_u64(data.len() as u64).await?;
        sock.write_all(&data).await?;
        sock.flush().await?;
        match sock.read_u8().await? {
            ST_OK => Ok(()),
            _ => Err(TransferError::Protocol("remote reported an error".into())),
        }
    }
}

/// Serves KV blocks to [`TcpTransfer`] clients. Anything that can produce / store
/// block bytes (e.g. the store's local byte pool) plugs in here.
pub trait BlockSource: Send + Sync + 'static {
    fn get(&self, key: &KvBlockKey) -> Option<Bytes>;
    fn put(&self, key: KvBlockKey, data: Bytes);
}

/// A simple in-memory [`BlockSource`] for tests and the local node.
#[derive(Clone, Debug, Default)]
pub struct MemBlockSource {
    inner: Arc<Mutex<HashMap<KvBlockKey, Bytes>>>,
}

impl MemBlockSource {
    pub fn new() -> Self {
        Self::default()
    }
}

impl BlockSource for MemBlockSource {
    fn get(&self, key: &KvBlockKey) -> Option<Bytes> {
        self.inner.lock().unwrap().get(key).cloned()
    }

    fn put(&self, key: KvBlockKey, data: Bytes) {
        self.inner.lock().unwrap().insert(key, data);
    }
}

/// Accept TCP transfer connections on an already-bound listener and serve blocks
/// from `source`. Binding outside lets callers use port 0 and learn the address.
pub async fn serve_listener<S: BlockSource>(
    listener: TcpListener,
    source: Arc<S>,
) -> std::io::Result<()> {
    loop {
        let (sock, _) = listener.accept().await?;
        let source = source.clone();
        tokio::spawn(async move {
            let _ = handle_conn(sock, source).await;
        });
    }
}

async fn handle_conn<S: BlockSource>(
    mut sock: TcpStream,
    source: Arc<S>,
) -> Result<(), TransferError> {
    let op = sock.read_u8().await?;
    let key_len = sock.read_u32().await? as usize;
    let mut key_buf = vec![0u8; key_len];
    sock.read_exact(&mut key_buf).await?;
    let key: KvBlockKey =
        serde_json::from_slice(&key_buf).map_err(|e| TransferError::Protocol(e.to_string()))?;
    match op {
        OP_READ => match source.get(&key) {
            Some(data) => {
                sock.write_u8(ST_OK).await?;
                sock.write_u64(data.len() as u64).await?;
                sock.write_all(&data).await?;
            }
            None => sock.write_u8(ST_NOTFOUND).await?,
        },
        OP_WRITE => {
            let len = sock.read_u64().await? as usize;
            let mut data = vec![0u8; len];
            sock.read_exact(&mut data).await?;
            source.put(key, Bytes::from(data));
            sock.write_u8(ST_OK).await?;
        }
        _ => sock.write_u8(ST_ERR).await?,
    }
    sock.flush().await?;
    Ok(())
}

/// Reserved RDMA backend. The real implementation registers memory regions,
/// posts RDMA READ/WRITE work requests via ibverbs, polls the completion queue,
/// and (with GPUDirect) lands bytes straight in GPU HBM. Stubbed behind the
/// `rdma` feature until a NIC is available — the trait above does not change.
#[cfg(feature = "rdma")]
#[derive(Clone, Debug, Default)]
pub struct RdmaTransfer;

#[cfg(feature = "rdma")]
#[async_trait]
impl Transfer for RdmaTransfer {
    fn name(&self) -> &str {
        "rdma"
    }

    fn link_class(&self) -> LinkClass {
        LinkClass::RdmaRoce
    }

    async fn read(&self, _remote: &NodeAddr, _key: &KvBlockKey) -> Result<Bytes, TransferError> {
        Err(TransferError::Unsupported(
            "rdma backend not yet wired (needs an RDMA NIC + ibverbs)",
        ))
    }

    async fn write(
        &self,
        _remote: &NodeAddr,
        _key: &KvBlockKey,
        _data: Bytes,
    ) -> Result<(), TransferError> {
        Err(TransferError::Unsupported(
            "rdma backend not yet wired (needs an RDMA NIC + ibverbs)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> KvBlockKey {
        KvBlockKey::new("m", "t", "ten", "p", "blk", 0, 64)
    }

    #[tokio::test]
    async fn local_transfer_roundtrips_real_bytes() {
        let transfer = LocalTransfer::new();
        let node = "node-1".to_string();
        transfer
            .write(&node, &key(), Bytes::from_static(b"kv-bytes"))
            .await
            .unwrap();
        let got = transfer.read(&node, &key()).await.unwrap();
        assert_eq!(&got[..], b"kv-bytes");
        // A different node has nothing.
        assert!(matches!(
            transfer.read(&"node-2".to_string(), &key()).await,
            Err(TransferError::NotFound)
        ));
    }

    #[tokio::test]
    async fn tcp_transfer_roundtrips_over_a_socket() {
        let source = Arc::new(MemBlockSource::new());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        tokio::spawn(serve_listener(listener, source.clone()));

        let client = TcpTransfer;
        client
            .write(&addr, &key(), Bytes::from_static(b"hello-kv"))
            .await
            .unwrap();
        let got = client.read(&addr, &key()).await.unwrap();
        assert_eq!(&got[..], b"hello-kv");

        let missing = KvBlockKey::new("m", "t", "ten", "p", "absent", 1, 64);
        assert!(matches!(
            client.read(&addr, &missing).await,
            Err(TransferError::NotFound)
        ));
    }
}
