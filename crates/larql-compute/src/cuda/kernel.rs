//! `CudaKernel` — bundled kernel function + dispatch geometry.
//!
//! Mirrors `metal::kernel::KernelHandle`. Every tiled CUDA kernel
//! bundles its compiled function handle with the dispatch geometry
//! (rows per block, threads per block) so the dispatcher can compute
//! `gridDim = ceil(num_rows / rows_per_block)` without importing
//! constants from a separate shader module.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule};

/// A tiled CUDA kernel: compiled function + dispatch geometry.
///
/// Every dispatch site reads `func` for the kernel launch and
/// `rows_per_block` / `threads_per_block` for grid sizing.
/// Geometry travels with the kernel; bumping a kernel (e.g.
/// `q4k_matvec` → `q4k_matvec_8warp`) = swap the type parameter
/// at the binding site.
#[derive(Clone)]
pub struct CudaKernel {
    /// The compiled kernel function.
    pub func: CudaFunction,
    /// Output rows the kernel covers per block.
    pub rows_per_block: u64,
    /// Threads per block the kernel expects.
    pub threads_per_block: u64,
    /// Kernel function name (for diagnostics only).
    pub kernel_name: &'static str,
}

/// Marker trait for tiled kernels. Each shader module implements
/// this to expose its name and dispatch geometry as compile-time
/// constants.
pub trait TiledKernel {
    /// Kernel function name as it appears in `__global__ void <name>(…)`.
    const KERNEL_NAME: &'static str;
    /// Output rows the kernel covers per block.
    const ROWS_PER_BLOCK: u64;
    /// Threads per block the kernel expects.
    const THREADS_PER_BLOCK: u64;
}

impl CudaKernel {
    /// Build a kernel handle from a `TiledKernel` marker type.
    /// This is the preferred constructor — the shader module owns
    /// its own name + geometry.
    ///
    /// Returns `None` if the function isn't found in the module.
    pub fn from_tiled<K: TiledKernel>(module: &Arc<CudaModule>) -> Option<Self> {
        let func = module.load_function(K::KERNEL_NAME).ok()?;
        Some(Self {
            func,
            rows_per_block: K::ROWS_PER_BLOCK,
            threads_per_block: K::THREADS_PER_BLOCK,
            kernel_name: K::KERNEL_NAME,
        })
    }
}