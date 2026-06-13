//! Real CUDA device tier — built with `--features cuda` (cudarc 0.19, with the
//! `dynamic-loading` feature, so it compiles even where CUDA is absent and loads
//! libcuda at runtime on a GPU box).
//!
//! The GPU HBM tier of the KV store: it moves block bytes across the PCIe / NVLink
//! boundary (H2D reload, D2H offload) and runs the FP16 → FP8 (E4M3)
//! quantize-on-offload kernel (compiled at startup with NVRTC). The store's HBM
//! tier drives this; GPUDirect-RDMA / NVLink zero-copy is the Transfer Engine's
//! concern (see `quillcache-transfer-engine`'s device segment).

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::QUANTIZE_KERNEL_CU;

const QUANTIZE_FN: &str = "quantize_fp16_to_fp8";

/// The GPU HBM tier of the KV store. Owns a CUDA context + stream; moves block
/// bytes across the host/device boundary. The FP8 quantize kernel is compiled
/// lazily (NVRTC) on first use, so binding the device needs only the CUDA driver.
#[derive(Debug)]
pub struct DeviceTier {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    quantize: OnceLock<CudaFunction>,
}

impl DeviceTier {
    pub fn available() -> bool {
        true
    }

    /// Bind to CUDA device `ordinal` (0 = first GPU). Driver-only — no NVRTC yet.
    pub fn new(ordinal: usize) -> Result<Self, String> {
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("CudaContext::new: {e:?}"))?;
        let stream = ctx.default_stream();
        Ok(Self {
            ctx,
            stream,
            quantize: OnceLock::new(),
        })
    }

    /// Which CUDA device this tier is bound to.
    pub fn ordinal(&self) -> usize {
        self.ctx.ordinal()
    }

    /// Compile (once, via NVRTC) + cache the FP16→FP8 quantize kernel.
    fn quantizer(&self) -> Result<&CudaFunction, String> {
        if let Some(f) = self.quantize.get() {
            return Ok(f);
        }
        let ptx = compile_ptx(QUANTIZE_KERNEL_CU).map_err(|e| format!("nvrtc compile: {e:?}"))?;
        let module = self
            .ctx
            .load_module(ptx)
            .map_err(|e| format!("load_module: {e:?}"))?;
        let func = module
            .load_function(QUANTIZE_FN)
            .map_err(|e| format!("load_function {QUANTIZE_FN}: {e:?}"))?;
        let _ = self.quantize.set(func); // first writer wins on a race
        Ok(self.quantize.get().unwrap())
    }

    /// Reload a cooled block from host memory back into GPU HBM (H2D copy).
    pub fn reload_to_device(&self, host: &[u8]) -> Result<CudaSlice<u8>, String> {
        let dev = self
            .stream
            .clone_htod(host)
            .map_err(|e| format!("h2d: {e:?}"))?;
        self.stream.synchronize().map_err(|e| format!("{e:?}"))?;
        Ok(dev)
    }

    /// Offload a block from GPU HBM to host memory (D2H copy) — the first hop of
    /// demotion to the DRAM / SSD tiers.
    pub fn offload_to_host(&self, device_buf: &CudaSlice<u8>) -> Result<Vec<u8>, String> {
        let host = self
            .stream
            .clone_dtoh(device_buf)
            .map_err(|e| format!("d2h: {e:?}"))?;
        self.stream.synchronize().map_err(|e| format!("{e:?}"))?;
        Ok(host)
    }

    /// FP16 → FP8 (E4M3) quantize-on-offload, on the GPU. `fp16` is the raw
    /// little-endian FP16 bytes of a (cooled) KV block; returns the FP8 bytes,
    /// i.e. half the size. Halves a cooled block's host/DRAM footprint.
    pub fn quantize_fp16_to_fp8(&self, fp16: &[u8], scale: f32) -> Result<Vec<u8>, String> {
        if fp16.len() & 1 != 0 {
            return Err("fp16 byte length must be even (2 bytes per __half)".into());
        }
        let n = (fp16.len() / 2) as i32; // number of __half elements
        if n == 0 {
            return Ok(Vec::new());
        }
        // Upload the FP16 bytes (the kernel reinterprets the buffer as __half*),
        // allocate the FP8 output (1 byte per element), launch, copy back.
        let d_in: CudaSlice<u8> = self
            .stream
            .clone_htod(fp16)
            .map_err(|e| format!("h2d in: {e:?}"))?;
        let mut d_out: CudaSlice<u8> = self
            .stream
            .alloc_zeros(n as usize)
            .map_err(|e| format!("alloc out: {e:?}"))?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let func = self.quantizer()?;
        let mut builder = self.stream.launch_builder(func);
        builder.arg(&d_in).arg(&mut d_out).arg(&n).arg(&scale);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("launch: {e:?}"))?;
        let out = self
            .stream
            .clone_dtoh(&d_out)
            .map_err(|e| format!("d2h out: {e:?}"))?;
        self.stream.synchronize().map_err(|e| format!("{e:?}"))?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These need a real NVIDIA GPU; they compile everywhere (dynamic-loading) but
    // only run on a GPU box: `cargo test -p quillcache-cuda --features cuda -- --ignored`.

    // Driver-only: H2D/D2H round trip preserves the bytes. Needs just libcuda.
    #[test]
    #[ignore = "requires an NVIDIA GPU"]
    fn host_device_roundtrip() {
        let tier = DeviceTier::new(0).expect("bind GPU 0");
        let block: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let dev = tier.reload_to_device(&block).expect("h2d");
        let back = tier.offload_to_host(&dev).expect("d2h");
        assert_eq!(block, back, "H2D/D2H must round-trip exactly");
    }

    // NVRTC: FP16 -> FP8 halves the byte count and runs on the GPU. Needs libnvrtc.
    #[test]
    #[ignore = "requires an NVIDIA GPU + NVRTC"]
    fn quantize_kernel() {
        let tier = DeviceTier::new(0).expect("bind GPU 0");
        let fp16 = vec![0u8; 2048]; // 1024 __half elements
        let fp8 = tier.quantize_fp16_to_fp8(&fp16, 1.0).expect("quantize");
        assert_eq!(fp8.len(), 1024, "FP8 output is one byte per element");
    }
}
