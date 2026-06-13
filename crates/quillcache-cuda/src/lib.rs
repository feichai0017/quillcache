//! CUDA device tier for QuillCache — the GPU HBM tier of the KV store.
//!
//! **Mapping note:** this is NOT a 1:1 Mooncake component. Mooncake puts GPU in
//! its *Transfer Engine* (GPUDirect-RDMA HBM↔NIC, NVLink, cuFile/GDS — see the
//! `rdma` / `nvlink` reserved transports in `quillcache-transfer-engine`) and has
//! no quantize tier. This crate is the **NVIDIA-Dynamo-KVBM-style G1 (HBM) tier**:
//! it (a) copies KV blocks between GPU HBM and host memory (offload / reload) and
//! (b) quantizes FP16 → FP8 on offload (an LMCache / KVBM idea) to halve a cooled
//! block's footprint. The real path is behind the `cuda` feature, built on a GPU
//! box; a host-only stub keeps callers compiling without an NVIDIA GPU.
//!
//! Excluded from the workspace so the default build never resolves `cudarc`.
//! Build standalone: `cargo build --features cuda` on a GPU box.

/// The FP16 → FP8 (E4M3) quantize-on-offload kernel source. Compiled with `nvcc`
/// on a GPU box and loaded by the device tier; embedded here as the reference.
pub const QUANTIZE_KERNEL_CU: &str = include_str!("kernels.cu");

#[cfg(feature = "cuda")]
mod gpu;
#[cfg(feature = "cuda")]
pub use gpu::DeviceTier;

#[cfg(not(feature = "cuda"))]
mod gpu_stub;
#[cfg(not(feature = "cuda"))]
pub use gpu_stub::DeviceTier;
