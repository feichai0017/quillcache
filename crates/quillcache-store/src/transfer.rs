//! Node block-serving endpoint — serves a [`BlockSource`]'s KV blocks to peers
//! over a small length-prefixed TCP protocol (read / write by key).
//!
//! This is the **server side** of the older key-oriented wire, kept only for the
//! real-engine connector's "path B" node (`src/node.rs`, which the Python bridge
//! in `bridge/` talks to). The faithful Mooncake transfer engine — byte movement
//! by `(segment, offset)` between registered memory — now lives in the
//! `quillcache-transfer-engine` crate; the client-side movers and the pooled
//! store that used to live here were retired with the Mooncake-faithful port.

use bytes::Bytes;
use quillcache_core::KvBlockKey;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Debug, thiserror::Error)]
pub enum TransferError {
    #[error("block not found on remote")]
    NotFound,
    #[error("io: {0}")]
    Io(String),
    #[error("protocol error: {0}")]
    Protocol(String),
}

impl From<std::io::Error> for TransferError {
    fn from(err: std::io::Error) -> Self {
        TransferError::Io(err.to_string())
    }
}

const OP_READ: u8 = 0;
const OP_WRITE: u8 = 1;
const ST_OK: u8 = 0;
const ST_NOTFOUND: u8 = 1;
const ST_ERR: u8 = 2;

/// Serves KV blocks to peers. Anything that can produce / store block bytes
/// (e.g. a node's [`crate::LocalKvStore`] via [`crate::StoreBlockSource`]) plugs
/// in here.
pub trait BlockSource: Send + Sync + 'static {
    fn get(&self, key: &KvBlockKey) -> Option<Bytes>;
    fn put(&self, key: KvBlockKey, data: Bytes);
}

/// A simple in-memory [`BlockSource`] for tests.
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

/// Accept connections on an already-bound listener and serve blocks from
/// `source`. Binding outside lets callers use port 0 and learn the address.
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
