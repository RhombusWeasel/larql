//! f32 gemv dispatch — GPU matrix-vector multiply for the LM head.
//!
//! Mirrors `metal::direct_ops::encode_f32_gemv` and `metal::f32_ops`.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

/// Compute `out[N] = W[N, K] · x[K]` on GPU.
///
/// Returns `None` if W or x have incompatible dimensions or the
/// dispatch fails. The caller should fall back to CPU matmul.
pub fn encode_f32_gemv(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    w: ndarray::ArrayView2<f32>,
    x: &[f32],
) -> Option<Vec<f32>> {
    let n = w.shape()[0];
    let k = w.shape()[1];
    if x.len() != k {
        return None;
    }

    // Flatten weights to contiguous f32 (row-major).
    // ndarray may not be contiguous in memory, so always copy.
    let w_flat: Vec<f32> = w.iter().copied().collect();

    // Copy weights and input to device
    let w_dev = stream.clone_htod(&w_flat).ok()?;
    let x_dev = stream.clone_htod(x).ok()?;

    // Allocate output
    let mut out_dev = stream.alloc_zeros::<f32>(n).ok()?;

    // Launch kernel
    let num_blocks = (n as u64).div_ceil(kernel.rows_per_block) as u32;
    let block_size = kernel.threads_per_block as u32;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&w_dev)
            .arg(&x_dev)
            .arg(&mut out_dev)
            .arg(&n_u32)
            .arg(&k_u32)
            .launch(cfg)
            .ok()?;
    }

    // Synchronize before reading back results
    stream.synchronize().ok()?;

    // Read back results
    stream.clone_dtoh(&out_dev).ok()
}