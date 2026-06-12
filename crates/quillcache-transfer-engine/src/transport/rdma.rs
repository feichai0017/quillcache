//! RDMA backend (Mooncake's `RdmaTransport`, ibverbs / GPUDirect-RDMA) — the
//! reserved seam. RDMA is Mooncake's *primary* path (and its core IP: one-sided
//! RDMA READ/WRITE, multi-NIC striping, GPUDirect HBM↔NIC zero-copy). It needs
//! an RDMA NIC, so it is stubbed until hardware is wired — the real impl
//! registers memory regions (`lkey`/`rkey`), opens queue pairs, and posts
//! one-sided verbs. Nothing above the [`Transport`] trait changes when it lands;
//! build the real version behind `--features rdma`.

use super::{LinkClass, TransferError, Transport};
use async_trait::async_trait;
use bytes::Bytes;

#[derive(Debug, Default)]
pub struct RdmaTransport;

const NEEDS_NIC: &str = "RDMA transport requires an ibverbs NIC (reserved; build with --features rdma once a NIC is wired)";

#[async_trait]
impl Transport for RdmaTransport {
    fn name(&self) -> &str {
        "rdma"
    }

    fn link_class(&self) -> LinkClass {
        LinkClass::RdmaIb
    }

    async fn read_remote(
        &self,
        _endpoint: &str,
        _offset: u64,
        _length: u64,
    ) -> Result<Bytes, TransferError> {
        Err(TransferError::Unsupported(NEEDS_NIC))
    }

    async fn write_remote(
        &self,
        _endpoint: &str,
        _offset: u64,
        _data: Bytes,
    ) -> Result<(), TransferError> {
        Err(TransferError::Unsupported(NEEDS_NIC))
    }
}
