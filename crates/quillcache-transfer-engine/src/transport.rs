//! The `Transport` seam (Mooncake's `transport/transport.h`) plus its request /
//! status types. A backend moves bytes one-sidedly against a remote node's RAM
//! segment by `(offset, length)`; the engine selects a backend per request via
//! [`crate::multi_transport::MultiTransport`].

use async_trait::async_trait;
use bytes::Bytes;

pub mod nvlink;
pub mod rdma;
pub mod tcp;

/// An opaque handle to an opened remote segment (Mooncake's `SegmentID`).
pub type SegmentId = i64;
/// A submitted transfer batch (Mooncake's `BatchID`).
pub type BatchId = u64;

/// Transfer direction — read from / write to the remote segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCode {
    Read,
    Write,
}

/// Lifecycle of a submitted batch (Mooncake's `TransferStatusEnum`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStatusEnum {
    Waiting,
    Pending,
    Completed,
    Failed,
    Timeout,
}

#[derive(Debug, Clone)]
pub struct TransferStatus {
    pub state: TransferStatusEnum,
    pub transferred_bytes: u64,
}

/// The physical link a transfer rides — feeds topology-aware path selection and
/// the store's cost model (HBM < NVLink < RDMA < TCP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkClass {
    LocalShm,
    Nvlink,
    RdmaIb,
    RdmaRoce,
    Tcp,
}

impl LinkClass {
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

/// A one-sided transfer between local registered memory (`source_offset` in this
/// node's RAM segment) and a remote segment at `(target_id, target_offset)` —
/// mirrors Mooncake's
/// `TransferRequest{ opcode, source, target_id, target_offset, length }`.
#[derive(Debug, Clone)]
pub struct TransferRequest {
    pub opcode: OpCode,
    pub source_offset: u64,
    pub target_id: SegmentId,
    pub target_offset: u64,
    pub length: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum TransferError {
    #[error("segment or object not found")]
    NotFound,
    #[error("unknown target segment handle")]
    BadSegment,
    #[error("io: {0}")]
    Io(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
}

/// Move bytes to / from a remote node's RAM segment. Backends differ only in the
/// wire (TCP now; RDMA / NVMe-oF / GPU reserved). The store layer maps object
/// keys to `(segment, offset)`; this trait is key-agnostic, exactly like
/// Mooncake's transport.
#[async_trait]
pub trait Transport: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    fn link_class(&self) -> LinkClass;
    /// Read `length` bytes from `endpoint`'s segment starting at `offset`.
    async fn read_remote(
        &self,
        endpoint: &str,
        offset: u64,
        length: u64,
    ) -> Result<Bytes, TransferError>;
    /// Write `data` into `endpoint`'s segment starting at `offset`.
    async fn write_remote(
        &self,
        endpoint: &str,
        offset: u64,
        data: Bytes,
    ) -> Result<(), TransferError>;
}
