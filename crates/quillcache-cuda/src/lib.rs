//! CUDA device tier for QuillCache — the GPU side of the KV store.
//!
//! The HBM tier of a Mooncake-style store lives in GPU memory. This crate is the
//! seam that (a) copies KV blocks between GPU HBM and host memory (the offload /
//! reload path) and (b) quantizes FP16 → FP8 on offload to halve the DRAM
//! footprint of a cooled block. The real path is behind the `cuda` feature and
//! is built on a GPU box; without it, a host-only stub keeps callers compiling
//! on machines with no NVIDIA GPU.
//!
//! This crate is excluded from the workspace so the default build never resolves
//! `cudarc`. Build it standalone: `cargo build --features cuda` on a GPU box.

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
