//! Q4_K matrix-matrix multiply (GEMM) dispatch for prefill.
//!
//! Computes `out[M, N] = W[N, K] @ X[M, K]^T` for `M > 1` input positions,
//! amortising Q4_K dequant cost across positions. Falls back to per-position
//! matvec when `M = 1` (decode path).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;
use crate::cuda::ops::q4k_matvec;

/// Dispatch Q4_K matrix-matrix multiply.
///
/// `W4K` has shape `[N, K]` in Q4_K packed format.
/// `X` has shape `[M * K]` in row-major f32.
/// Output has shape `[M * N]` in row-major f32.
///
/// When `M = 1`, falls back to `q4k_matvec` since the amortisation
/// benefit doesn't apply for a single position.
pub fn encode_q4k_matmul(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    w4k: &[u8],
    x: &[f32],
    n: usize,
    k: usize,
    m: usize,
) -> Option<Vec<f32>> {
    // Fall back to per-position matvec for M=1
    if m == 1 {
        return q4k_matvec::encode_q4k_matvec(ctx, stream, kernel, w4k, x, n, k);
    }

    if k == 0 || k % 256 != 0 || n == 0 {
        return None;
    }

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let m_u32 = m as u32;

    // Copy weight matrix to device
    let w_dev = stream.clone_htod(w4k).ok()?;

    // Copy X matrix to device
    let x_dev = stream.clone_htod(x).ok()?;

    // Allocate output buffer [M * N]
    let mut out_dev: CudaSlice<f32> = stream.alloc_zeros(m * n).ok()?;

    // Launch kernel
    let grid_x = (m as u64).div_ceil(crate::cuda::shaders::q4k_matmul::COLS_PER_BLOCK) as u32;
    let grid_y = (n as u64).div_ceil(crate::cuda::shaders::q4k_matmul::ROWS_PER_BLOCK) as u32;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (crate::cuda::shaders::q4k_matmul::THREADS_PER_BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&w_dev)
            .arg(&x_dev)
            .arg(&mut out_dev)
            .arg(&n_u32)
            .arg(&k_u32)
            .arg(&m_u32)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;

    let result = stream.clone_dtoh(&out_dev).ok()?;
    Some(result)
}