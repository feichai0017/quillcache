//! QuillCache Transfer Engine — a faithful Rust port of Mooncake's
//! `mooncake-transfer-engine` (the KVCache-centric data plane, FAST'25).
//!
//! **Component map** (Mooncake C++ → this crate):
//!
//! | Mooncake (`mooncake-transfer-engine/`) | here |
//! | --- | --- |
//! | `TransferEngine` (`transfer_engine.h`) | [`engine::TransferEngine`] facade |
//! | `MultiTransport` (`multi_transport.h`) | [`multi_transport::MultiTransport`] |
//! | `Transport` + subclasses (`transport/*`) | [`transport::Transport`] + backends |
//! | &nbsp;&nbsp;`TcpTransport` | [`transport::tcp::TcpTransport`] — **real** |
//! | &nbsp;&nbsp;`RdmaTransport` (ibverbs) | [`transport::rdma::RdmaTransport`] — reserved |
//! | `TransferMetadata` (`transfer_metadata.h`) | [`metadata`] (`SegmentDesc`, backend) |
//! | `Topology` (`topology.h`) | [`topology::Topology`] |
//!
//! **Design we mirror:** Mooncake's engine moves bytes *one-sidedly* between
//! *registered memory* regions addressed by `(segment, offset)` — NOT by key
//! (keys live in the store layer). Each engine owns one RAM segment (a
//! registered byte arena); a peer [`engine::TransferEngine::open_segment`]s it
//! and [`engine::TransferEngine::submit_transfer`]s READ/WRITE requests against
//! `(target_id, target_offset)`. TCP is the real, portable backend; RDMA /
//! GPUDirect / NVMe-oF are the reserved seams (need a NIC / GPU) — nothing above
//! the [`transport::Transport`] trait changes when they land.
//!
//! This is Phase 1 of the Mooncake-faithful restructure: the store
//! (`quillcache-store`) is rebuilt on this engine in a later phase.

pub mod engine;
pub mod metadata;
pub mod multi_transport;
pub mod topology;
pub mod transport;

pub use engine::TransferEngine;
pub use metadata::{BufferDesc, InMemoryMetadata, MetadataBackend, SegmentDesc};
pub use multi_transport::MultiTransport;
pub use topology::Topology;
pub use transport::{
    BatchId, LinkClass, OpCode, SegmentId, TransferError, TransferRequest, TransferStatus,
    TransferStatusEnum, Transport,
};
