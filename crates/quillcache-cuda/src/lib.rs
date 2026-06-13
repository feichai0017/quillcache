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
//! A workspace member, but the real path is behind the `cuda` feature: cudarc
//! 0.19 with `dynamic-loading` compiles with no CUDA present (the default build
//! is a host-only stub, so CI needs no toolkit) and runs on a GPU box. Build it
//! with `cargo build -p quillcache-cuda --features cuda`; the H2D/D2H + FP8
//! quantize round-trips are verified on a Modal NVIDIA L4 via
//! `deploy/modal_cuda_verify.py` (the GPU tests are `#[ignore]`).

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
