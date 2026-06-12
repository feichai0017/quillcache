//! TCP backend (Mooncake's `TcpTransport`) — the portable, always-available
//! transport. Moves bytes one-sidedly against a peer's RAM segment by
//! `(offset, length)`: READ pulls a range, WRITE pushes a range. This is the
//! real path on any machine; the RDMA backend supersedes it where a NIC exists.

use super::{LinkClass, TransferError, Transport};
use async_trait::async_trait;
use bytes::Bytes;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const OP_READ: u8 = 0;
const OP_WRITE: u8 = 1;
const ST_OK: u8 = 0;
const ST_ERR: u8 = 2;

#[derive(Debug, Default)]
pub struct TcpTransport;

#[async_trait]
impl Transport for TcpTransport {
    fn name(&self) -> &str {
        "tcp"
    }

    fn link_class(&self) -> LinkClass {
        LinkClass::Tcp
    }

    async fn read_remote(
        &self,
        endpoint: &str,
        offset: u64,
        length: u64,
    ) -> Result<Bytes, TransferError> {
        let mut sock = TcpStream::connect(endpoint).await.map_err(io_err)?;
        sock.write_u8(OP_READ).await.map_err(io_err)?;
        sock.write_u64(offset).await.map_err(io_err)?;
        sock.write_u64(length).await.map_err(io_err)?;
        let status = sock.read_u8().await.map_err(io_err)?;
        if status != ST_OK {
            return Err(TransferError::Protocol(
                "remote read out of segment bounds".into(),
            ));
        }
        let len = sock.read_u64().await.map_err(io_err)?;
        let mut buf = vec![0u8; len as usize];
        sock.read_exact(&mut buf).await.map_err(io_err)?;
        Ok(Bytes::from(buf))
    }

    async fn write_remote(
        &self,
        endpoint: &str,
        offset: u64,
        data: Bytes,
    ) -> Result<(), TransferError> {
        let mut sock = TcpStream::connect(endpoint).await.map_err(io_err)?;
        sock.write_u8(OP_WRITE).await.map_err(io_err)?;
        sock.write_u64(offset).await.map_err(io_err)?;
        sock.write_u64(data.len() as u64).await.map_err(io_err)?;
        sock.write_all(&data).await.map_err(io_err)?;
        let status = sock.read_u8().await.map_err(io_err)?;
        if status != ST_OK {
            return Err(TransferError::Protocol("remote write failed".into()));
        }
        Ok(())
    }
}

fn io_err(e: std::io::Error) -> TransferError {
    TransferError::Io(e.to_string())
}

/// Serve a node's RAM segment to peers: handle READ / WRITE frames against the
/// shared byte arena. Each connection may issue many ops. A WRITE past the
/// current end grows the arena (zero-filled) — the segment owns its registered
/// space. Spawned by [`crate::engine::TransferEngine::init`].
pub async fn serve_segment(listener: TcpListener, segment: Arc<Mutex<Vec<u8>>>) {
    loop {
        let Ok((sock, _)) = listener.accept().await else {
            return;
        };
        let segment = segment.clone();
        tokio::spawn(async move {
            let _ = handle_conn(sock, segment).await;
        });
    }
}

async fn handle_conn(mut sock: TcpStream, segment: Arc<Mutex<Vec<u8>>>) -> std::io::Result<()> {
    loop {
        let op = match sock.read_u8().await {
            Ok(op) => op,
            Err(_) => return Ok(()), // peer closed
        };
        let offset = sock.read_u64().await? as usize;
        let len = sock.read_u64().await? as usize;
        match op {
            OP_READ => {
                let data = {
                    let seg = segment.lock().unwrap();
                    match offset.checked_add(len) {
                        Some(end) if end <= seg.len() => {
                            Some(Bytes::copy_from_slice(&seg[offset..end]))
                        }
                        _ => None,
                    }
                };
                match data {
                    Some(bytes) => {
                        sock.write_u8(ST_OK).await?;
                        sock.write_u64(bytes.len() as u64).await?;
                        sock.write_all(&bytes).await?;
                    }
                    None => sock.write_u8(ST_ERR).await?,
                }
            }
            OP_WRITE => {
                let mut buf = vec![0u8; len];
                sock.read_exact(&mut buf).await?;
                {
                    let mut seg = segment.lock().unwrap();
                    let end = offset + len;
                    if seg.len() < end {
                        seg.resize(end, 0);
                    }
                    seg[offset..end].copy_from_slice(&buf);
                }
                sock.write_u8(ST_OK).await?;
            }
            _ => return Ok(()),
        }
    }
}
