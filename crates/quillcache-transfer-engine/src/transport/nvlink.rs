//! NVLink / GPUDirect transport (Mooncake's `nvlink_transport` + `transport/device`)
//! — the GPU-resident data path: NVLink for intra-node GPU↔GPU, GPUDirect-RDMA
//! for HBM↔NIC zero-copy. **This is where "CUDA" lives in Mooncake's Transfer
//! Engine** — moving KV bytes to / from / between GPU HBM without staging through
//! host memory — not a separate quantize-on-offload tier. Reserved seam: needs an
//! NVIDIA GPU (CUDA + NVLink / IBGDA), stubbed until hardware is wired; the real
//! impl lands behind `--features nvlink` and nothing above the [`Transport`] trait
//! changes.
//!
//! (`quillcache-cuda` is a *different* concern — the Dynamo-KVBM HBM tier with
//! FP16→FP8 quantize-on-offload, an LMCache/KVBM-flavored idea Mooncake does not
//! have. GPUDirect-RDMA itself rides the `rdma` backend; cuFile / GDS would be an
//! `nvmeof` backend.)

use super::{LinkClass, TransferError, Transport};
use async_trait::async_trait;
use bytes::Bytes;

#[derive(Debug, Default)]
pub struct NvlinkTransport;

const NEEDS_GPU: &str = "NVLink/GPUDirect transport requires an NVIDIA GPU (reserved; build with --features nvlink once CUDA + NVLink/IBGDA is wired)";

#[async_trait]
impl Transport for NvlinkTransport {
    fn name(&self) -> &str {
        "nvlink"
    }

    fn link_class(&self) -> LinkClass {
        LinkClass::Nvlink
    }

    async fn read_remote(
        &self,
        _endpoint: &str,
        _offset: u64,
        _length: u64,
    ) -> Result<Bytes, TransferError> {
        Err(TransferError::Unsupported(NEEDS_GPU))
    }

    async fn write_remote(
        &self,
        _endpoint: &str,
        _offset: u64,
        _data: Bytes,
    ) -> Result<(), TransferError> {
        Err(TransferError::Unsupported(NEEDS_GPU))
    }
}
