//! Host-only fallback for the CUDA device tier.
//!
//! Compiled when the `cuda` feature is off (e.g. on a machine with no NVIDIA
//! GPU), so callers and tests still build. Every operation reports that the GPU
//! path is unavailable; the real tier lives in `gpu.rs` behind `--features cuda`.

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
}
