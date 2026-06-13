//! CUDA device segment — GPU HBM registered as a Transfer Engine segment.
//!
//! Mooncake registers memory *segments* (host OR device) and moves bytes
//! one-sidedly by `(offset, length)`. This is the **device** form: the segment's
//! bytes live in GPU HBM (`cudaMalloc`), and READ / WRITE stage through
//! `cudaMemcpy` (D2H / H2D). It speaks the exact same wire as the host TCP
//! segment ([`crate::transport::tcp`]), so an unmodified
//! [`crate::transport::tcp::TcpTransport`] peer can READ & WRITE a GPU-resident
//! segment with no changes above the [`crate::transport::Transport`] trait.
//!
//! The host hop (D2H to serve a read, H2D to apply a write) is exactly what
//! GPUDirect-RDMA / NVLink (the reserved `nvlink` feature) removes — NIC / GPU ↔
//! HBM with no staging. This `cuda` path is the real, single-GPU-verifiable core;
//! `nvlink` is the zero-copy network optimization layered on top.
//!
//! Built only with `--features cuda` (cudarc with `dynamic-loading`, so it
//! compiles without CUDA present and runs on a GPU box).

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

// Same wire as transport/tcp.rs.
const OP_READ: u8 = 0;
const OP_WRITE: u8 = 1;
const ST_OK: u8 = 0;
const ST_ERR: u8 = 2;

/// A GPU HBM byte arena registered as a transfer-engine segment. Capacity is
/// fixed (HBM is precious); a logical length grows on WRITE up to capacity,
/// mirroring the host segment's grow-on-write.
pub struct DeviceSegment {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    arena: Mutex<Arena>,
    capacity: usize,
}

struct Arena {
    hbm: CudaSlice<u8>,
    len: usize,
}

impl std::fmt::Debug for DeviceSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceSegment")
            .field("ordinal", &self.ctx.ordinal())
            .field("capacity", &self.capacity)
            .field("len", &self.len())
            .finish()
    }
}

impl DeviceSegment {
    /// Bind GPU `ordinal` and reserve `capacity` bytes of HBM for the segment.
    pub fn new(ordinal: usize, capacity: usize) -> Result<Arc<Self>, String> {
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("CudaContext::new: {e:?}"))?;
        let stream = ctx.default_stream();
        let hbm = stream
            .alloc_zeros::<u8>(capacity)
            .map_err(|e| format!("alloc HBM arena: {e:?}"))?;
        Ok(Arc::new(Self {
            ctx,
            stream,
            arena: Mutex::new(Arena { hbm, len: 0 }),
            capacity,
        }))
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.arena.lock().unwrap().len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Which CUDA device the segment's HBM lives on.
    pub fn ordinal(&self) -> usize {
        self.ctx.ordinal()
    }

    /// Register a buffer: H2D-copy it into the arena at the current end; returns
    /// its offset — the handle a peer targets.
    pub fn register(&self, data: &[u8]) -> Result<u64, String> {
        let mut a = self.arena.lock().unwrap();
        let offset = a.len;
        self.write_locked(&mut a, offset, data)?;
        Ok(offset as u64)
    }

    /// READ: D2H-copy `len` bytes from HBM at `offset`. Errors if out of bounds.
    pub fn read(&self, offset: usize, len: usize) -> Result<Vec<u8>, String> {
        let a = self.arena.lock().unwrap();
        let end = offset.checked_add(len).ok_or("offset+len overflow")?;
        if end > a.len {
            return Err("read out of segment bounds".into());
        }
        let mut host = vec![0u8; len];
        if len > 0 {
            let view = a.hbm.slice(offset..end);
            self.stream
                .memcpy_dtoh(&view, &mut host)
                .map_err(|e| format!("d2h: {e:?}"))?;
            self.stream.synchronize().map_err(|e| format!("{e:?}"))?;
        }
        Ok(host)
    }

    /// WRITE: H2D-copy `data` into HBM at `offset`, growing the logical length up
    /// to capacity. Errors if it would exceed capacity.
    pub fn write(&self, offset: usize, data: &[u8]) -> Result<(), String> {
        let mut a = self.arena.lock().unwrap();
        self.write_locked(&mut a, offset, data)
    }

    fn write_locked(&self, a: &mut Arena, offset: usize, data: &[u8]) -> Result<(), String> {
        let end = offset
            .checked_add(data.len())
            .ok_or("offset+len overflow")?;
        if end > self.capacity {
            return Err(format!(
                "write exceeds HBM segment capacity ({end} > {})",
                self.capacity
            ));
        }
        if !data.is_empty() {
            let mut view = a.hbm.slice_mut(offset..end);
            self.stream
                .memcpy_htod(data, &mut view)
                .map_err(|e| format!("h2d: {e:?}"))?;
            self.stream.synchronize().map_err(|e| format!("{e:?}"))?;
        }
        if end > a.len {
            a.len = end;
        }
        Ok(())
    }
}

/// Serve a GPU HBM [`DeviceSegment`] to peers over the same one-sided
/// `(offset, length)` wire as the host TCP segment, so an unmodified
/// [`crate::transport::tcp::TcpTransport`] peer READs / WRITEs GPU-resident
/// bytes. The bytes stage through `cudaMemcpy` here (GPUDirect would remove the
/// hop). The blocking copies run inline on the connection task — fine for the
/// reference path; a production server would `spawn_blocking` them.
pub async fn serve_device_segment(listener: TcpListener, segment: Arc<DeviceSegment>) {
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

async fn handle_conn(mut sock: TcpStream, segment: Arc<DeviceSegment>) -> std::io::Result<()> {
    loop {
        let op = match sock.read_u8().await {
            Ok(op) => op,
            Err(_) => return Ok(()), // peer closed
        };
        let offset = sock.read_u64().await? as usize;
        let len = sock.read_u64().await? as usize;
        match op {
            OP_READ => match segment.read(offset, len) {
                Ok(bytes) => {
                    sock.write_u8(ST_OK).await?;
                    sock.write_u64(bytes.len() as u64).await?;
                    sock.write_all(&bytes).await?;
                }
                Err(_) => sock.write_u8(ST_ERR).await?,
            },
            OP_WRITE => {
                let mut buf = vec![0u8; len];
                sock.read_exact(&mut buf).await?;
                match segment.write(offset, &buf) {
                    Ok(()) => sock.write_u8(ST_OK).await?,
                    Err(_) => sock.write_u8(ST_ERR).await?,
                }
            }
            _ => return Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::tcp::TcpTransport;
    use crate::transport::Transport;
    use bytes::Bytes;

    // Needs a real NVIDIA GPU; compiles everywhere (dynamic-loading) but only run
    // on a GPU box: `cargo test -p quillcache-transfer-engine --features cuda -- --ignored`.
    #[tokio::test]
    #[ignore = "requires an NVIDIA GPU"]
    async fn tcp_peer_reads_writes_a_gpu_hbm_segment() {
        // A GPU HBM segment served over the one-sided wire.
        let seg = DeviceSegment::new(0, 1 << 20).expect("alloc HBM segment");
        let off = seg.register(b"resident-in-HBM").expect("register"); // 15 bytes
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        tokio::spawn(serve_device_segment(listener, seg.clone()));

        // An UNMODIFIED TCP transport peer reads the GPU-resident bytes back.
        let tcp = TcpTransport;
        let got = tcp.read_remote(&endpoint, off, 15).await.expect("read HBM");
        assert_eq!(&got[..], b"resident-in-HBM");

        // ...and writes new bytes straight into HBM, which read back identically.
        tcp.write_remote(&endpoint, 100, Bytes::from_static(b"written-into-HBM"))
            .await
            .expect("write HBM");
        let got2 = tcp
            .read_remote(&endpoint, 100, 16)
            .await
            .expect("read HBM 2");
        assert_eq!(&got2[..], b"written-into-HBM");
    }
}
