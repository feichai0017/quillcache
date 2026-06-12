//! TransferEngine (Mooncake's `transfer_engine.h`) — the top-level facade. Owns
//! this node's RAM segment (a registered byte arena), the metadata backend, the
//! [`MultiTransport`], and the batch table; for TCP it binds a listener and
//! serves its segment to peers.
//!
//! The faithful flow:
//! 1. [`TransferEngine::init`] — bind, publish our [`SegmentDesc`], serve.
//! 2. [`TransferEngine::register_local_memory`] — register a buffer → offset.
//! 3. [`TransferEngine::open_segment`] — resolve a peer's segment → [`SegmentId`].
//! 4. [`TransferEngine::allocate_batch_id`] → [`TransferEngine::submit_transfer`]
//!    (READ / WRITE against `(target_id, target_offset)`) →
//!    [`TransferEngine::get_transfer_status`] → [`TransferEngine::free_batch_id`].

use crate::metadata::{MetadataBackend, SegmentDesc};
use crate::multi_transport::MultiTransport;
use crate::transport::tcp::{self, TcpTransport};
use crate::transport::{
    BatchId, OpCode, SegmentId, TransferError, TransferRequest, TransferStatus, TransferStatusEnum,
};
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

#[derive(Debug)]
pub struct TransferEngine {
    local_name: String,
    /// This node's RAM segment: a registered byte arena peers READ / WRITE by offset.
    segment: Arc<Mutex<Vec<u8>>>,
    metadata: Arc<dyn MetadataBackend>,
    multi: MultiTransport,
    next_batch: AtomicU64,
    batches: Mutex<HashMap<BatchId, TransferStatus>>,
    /// Opened remote segments: handle → (protocol, endpoint).
    segment_cache: Mutex<HashMap<SegmentId, (String, String)>>,
    next_seg: AtomicI64,
}

impl TransferEngine {
    /// Bind a TCP listener, publish this node's segment to the metadata backend,
    /// install the TCP transport, and start serving the segment to peers.
    pub async fn init(
        local_name: impl Into<String>,
        metadata: Arc<dyn MetadataBackend>,
        bind_addr: &str,
    ) -> std::io::Result<Arc<Self>> {
        let local_name = local_name.into();
        let segment = Arc::new(Mutex::new(Vec::new()));
        let listener = TcpListener::bind(bind_addr).await?;
        let endpoint = listener.local_addr()?.to_string();

        let mut multi = MultiTransport::new();
        multi.install("tcp", Arc::new(TcpTransport));

        metadata.put_segment(SegmentDesc {
            name: local_name.clone(),
            protocol: "tcp".into(),
            endpoint,
            buffers: Vec::new(),
        });

        let engine = Arc::new(Self {
            local_name,
            segment: segment.clone(),
            metadata,
            multi,
            next_batch: AtomicU64::new(1),
            batches: Mutex::new(HashMap::new()),
            segment_cache: Mutex::new(HashMap::new()),
            next_seg: AtomicI64::new(1),
        });
        tokio::spawn(tcp::serve_segment(listener, segment));
        Ok(engine)
    }

    pub fn local_name(&self) -> &str {
        &self.local_name
    }

    /// Register a local buffer (copy it into the RAM segment); returns its offset
    /// — the handle a peer targets, and the source for our own writes.
    pub fn register_local_memory(&self, data: &[u8]) -> u64 {
        let mut seg = self.segment.lock().unwrap();
        let offset = seg.len() as u64;
        seg.extend_from_slice(data);
        offset
    }

    /// Reserve `len` zeroed bytes in the RAM segment — a READ destination.
    pub fn register_zeroed(&self, len: usize) -> u64 {
        let mut seg = self.segment.lock().unwrap();
        let offset = seg.len() as u64;
        seg.resize(offset as usize + len, 0);
        offset
    }

    /// Read bytes back from our own RAM segment.
    pub fn read_local(&self, offset: u64, len: u64) -> Bytes {
        let seg = self.segment.lock().unwrap();
        Bytes::copy_from_slice(&seg[offset as usize..(offset + len) as usize])
    }

    /// Resolve a remote segment by name → an opaque [`SegmentId`] handle.
    pub fn open_segment(&self, name: &str) -> Result<SegmentId, TransferError> {
        let desc = self
            .metadata
            .get_segment(name)
            .ok_or(TransferError::NotFound)?;
        let id = self.next_seg.fetch_add(1, Ordering::Relaxed);
        self.segment_cache
            .lock()
            .unwrap()
            .insert(id, (desc.protocol, desc.endpoint));
        Ok(id)
    }

    /// Open a batch for `submit_transfer` (Mooncake's `allocateBatchID`).
    pub fn allocate_batch_id(&self, _batch_size: usize) -> BatchId {
        let id = self.next_batch.fetch_add(1, Ordering::Relaxed);
        self.batches.lock().unwrap().insert(
            id,
            TransferStatus {
                state: TransferStatusEnum::Waiting,
                transferred_bytes: 0,
            },
        );
        id
    }

    /// Execute a batch of one-sided transfers against opened remote segments.
    /// READ copies `target[target_offset..]` into `local[source_offset..]`;
    /// WRITE copies `local[source_offset..]` into `target[target_offset..]`.
    pub async fn submit_transfer(
        &self,
        batch: BatchId,
        requests: Vec<TransferRequest>,
    ) -> Result<(), TransferError> {
        let mut transferred = 0u64;
        for req in &requests {
            let (protocol, endpoint) = self
                .segment_cache
                .lock()
                .unwrap()
                .get(&req.target_id)
                .cloned()
                .ok_or(TransferError::BadSegment)?;
            let transport = self
                .multi
                .select(&protocol)
                .ok_or(TransferError::Unsupported("no transport for protocol"))?;
            match req.opcode {
                OpCode::Read => {
                    let bytes = transport
                        .read_remote(&endpoint, req.target_offset, req.length)
                        .await?;
                    let mut seg = self.segment.lock().unwrap();
                    let (start, end) = (
                        req.source_offset as usize,
                        (req.source_offset + req.length) as usize,
                    );
                    if end > seg.len() {
                        return Err(TransferError::Protocol(
                            "local source buffer too small for READ".into(),
                        ));
                    }
                    seg[start..end].copy_from_slice(&bytes);
                }
                OpCode::Write => {
                    let data = {
                        let seg = self.segment.lock().unwrap();
                        let (start, end) = (
                            req.source_offset as usize,
                            (req.source_offset + req.length) as usize,
                        );
                        if end > seg.len() {
                            return Err(TransferError::Protocol(
                                "local source buffer too small for WRITE".into(),
                            ));
                        }
                        Bytes::copy_from_slice(&seg[start..end])
                    };
                    transport
                        .write_remote(&endpoint, req.target_offset, data)
                        .await?;
                }
            }
            transferred += req.length;
        }
        self.batches.lock().unwrap().insert(
            batch,
            TransferStatus {
                state: TransferStatusEnum::Completed,
                transferred_bytes: transferred,
            },
        );
        Ok(())
    }

    pub fn get_transfer_status(&self, batch: BatchId) -> Option<TransferStatus> {
        self.batches.lock().unwrap().get(&batch).cloned()
    }

    pub fn free_batch_id(&self, batch: BatchId) {
        self.batches.lock().unwrap().remove(&batch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::InMemoryMetadata;

    #[tokio::test]
    async fn tcp_segment_read_and_write_over_registered_memory() {
        // Shared (in-memory / P2PHANDSHAKE) metadata so the two engines discover
        // each other with no external store.
        let md: Arc<dyn MetadataBackend> = Arc::new(InMemoryMetadata::new());

        // Engine A registers a buffer in its RAM segment.
        let a = TransferEngine::init("A", md.clone(), "127.0.0.1:0")
            .await
            .unwrap();
        let a_off = a.register_local_memory(b"hello-mooncake-kv"); // 17 bytes

        // Engine B opens A's segment and READs the bytes over TCP into local memory.
        let b = TransferEngine::init("B", md.clone(), "127.0.0.1:0")
            .await
            .unwrap();
        let seg_a = b.open_segment("A").unwrap();
        let dst = b.register_zeroed(17);
        let batch = b.allocate_batch_id(1);
        b.submit_transfer(
            batch,
            vec![TransferRequest {
                opcode: OpCode::Read,
                source_offset: dst,
                target_id: seg_a,
                target_offset: a_off,
                length: 17,
            }],
        )
        .await
        .unwrap();
        assert_eq!(
            b.get_transfer_status(batch).unwrap().state,
            TransferStatusEnum::Completed
        );
        assert_eq!(&b.read_local(dst, 17)[..], b"hello-mooncake-kv");
        b.free_batch_id(batch);

        // WRITE path: B pushes a local buffer into A's segment at a fresh offset;
        // A reads it back locally.
        let src = b.register_local_memory(b"written-by-B"); // 12 bytes
        let batch2 = b.allocate_batch_id(1);
        b.submit_transfer(
            batch2,
            vec![TransferRequest {
                opcode: OpCode::Write,
                source_offset: src,
                target_id: seg_a,
                target_offset: 1000,
                length: 12,
            }],
        )
        .await
        .unwrap();
        assert_eq!(&a.read_local(1000, 12)[..], b"written-by-B");

        // Opening a segment that was never published is a clean miss.
        assert!(matches!(
            b.open_segment("does-not-exist"),
            Err(TransferError::NotFound)
        ));
    }
}
