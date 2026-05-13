//! `CudaBackend`'s `ComputeBackend`-family trait implementations.
//!
//! One file per sub-trait. The umbrella `ComputeBackend` impl
//! (`name`, `device_info`, `supports`) lives here.

mod decode;
mod matmul;
mod quant_matvec;

use super::CudaBackend;
use crate::backend::Capability;
use crate::backend::ComputeBackend;

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        "cuda (GPU)"
    }

    fn device_info(&self) -> String {
        let name = self.ctx.name().unwrap_or_else(|_| "Unknown NVIDIA GPU".to_string());
        format!("NVIDIA {} (flop threshold: {})", name, self.flop_threshold())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn supports(&self, cap: Capability) -> bool {
        matches!(
            cap,
            Capability::F32Gemv
                | Capability::QuantMatVec
                | Capability::Q4VecMat
                | Capability::DecodeToken
                | Capability::PrefillQ4
                | Capability::DecodeProfile
        )
    }
}