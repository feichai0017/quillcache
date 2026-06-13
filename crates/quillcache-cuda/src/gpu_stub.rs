//! Host-only fallback for the CUDA device tier.
//!
//! Compiled when the `cuda` feature is off (e.g. on a machine with no NVIDIA
//! GPU), so callers and tests still build. Construction fails with a clear
//! message and the byte-based ops report the GPU path is unavailable; the real
//! tier lives in `gpu.rs` behind `--features cuda`. The `CudaSlice`-returning
//! ops (reload_to_device / offload_to_host) exist only on the real tier, since
//! their type comes from `cudarc`.

/// Placeholder for the GPU HBM tier. Without the `cuda` feature there is no GPU
/// to bind to, so construction fails with a clear message.
#[derive(Debug, Default)]
pub struct DeviceTier;

impl DeviceTier {
    pub fn available() -> bool {
        false
    }

    pub fn new(_ordinal: usize) -> Result<Self, String> {
        Err("built without the `cuda` feature; build with --features cuda on a GPU box".to_string())
    }

    pub fn ordinal(&self) -> usize {
        0
    }

    /// Mirror of the real tier's quantize op so non-CUDA callers type-check; it
    /// can never run because [`DeviceTier::new`] fails without the `cuda` feature.
    pub fn quantize_fp16_to_fp8(&self, _fp16: &[u8], _scale: f32) -> Result<Vec<u8>, String> {
        Err("CUDA quantize unavailable; build with --features cuda on a GPU box".to_string())
    }
}
