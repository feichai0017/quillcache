//! Real CUDA device tier (built only with `--features cuda` on a GPU box).
//!
//! Copies KV blocks between GPU HBM and host memory via `cudarc`. The async,
//! multi-stream, GPUDirect-RDMA version is the production path; this is the
//! single-stream sync core that the store's HBM tier drives.
//!
//! Verify the `cudarc` API against your installed version — the 0.12 surface is
//! used here (`CudaDevice::new`, `htod_sync_copy`, `dtoh_sync_copy`).

use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// The GPU HBM tier of the KV store. Owns a CUDA device handle and moves block
/// bytes across the PCIe / NVLink boundary.
#[derive(Debug)]
pub struct DeviceTier {
    device: Arc<CudaDevice>,
}

impl DeviceTier {
    pub fn available() -> bool {
        true
    }

    /// Bind to CUDA device `ordinal` (0 = first GPU).
    pub fn new(ordinal: usize) -> Result<Self, String> {
        let device = CudaDevice::new(ordinal).map_err(|e| e.to_string())?;
        Ok(Self { device })
    }

    /// Reload a cooled block from host memory back into GPU HBM (H2D copy).
    pub fn reload_to_device(&self, host: &[u8]) -> Result<CudaSlice<u8>, String> {
        self.device.htod_sync_copy(host).map_err(|e| e.to_string())
    }

    /// Offload a block from GPU HBM to host memory (D2H copy) — the first hop of
    /// demotion to the DRAM / SSD tiers.
    pub fn offload_to_host(&self, device_buf: &CudaSlice<u8>) -> Result<Vec<u8>, String> {
        self.device
            .dtoh_sync_copy(device_buf)
            .map_err(|e| e.to_string())
    }
}
