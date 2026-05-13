//! f32 matrix multiply (SGEMM) dispatch — CUDA.

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

/// Tiled f32 matrix multiply: `C[M,N] = A[M,K] × B[K,N]`
///
/// 32×32 tiles with shared memory. Grid: ((N+31)/32, (M+31)/32, 1).
pub fn encode_sgemm(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Option<Vec<f32>> {
    let a_dev = stream.clone_htod(a).ok()?;
    let b_dev = stream.clone_htod(b).ok()?;
    let mut c_dev = stream.alloc_zeros::<f32>(m * n).ok()?;

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let grid_x = (n as u32 + 31) / 32;
    let grid_y = (m as u32 + 31) / 32;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (32, 32, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut c_dev)
            .arg(&m_u32)
            .arg(&n_u32)
            .arg(&k_u32)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;
    stream.clone_dtoh(&c_dev).ok()
}

/// Tiled f32 matrix multiply transposed B: `C[M,N] = A[M,K] × B^T[N,K] = A[M,K] × B[N,K]^T`
///
/// 32×32 tiles with shared memory. Grid: ((N+31)/32, (M+31)/32, 1).
pub fn encode_sgemm_transb(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Option<Vec<f32>> {
    let a_dev = stream.clone_htod(a).ok()?;
    let b_dev = stream.clone_htod(b).ok()?;
    let mut c_dev = stream.alloc_zeros::<f32>(m * n).ok()?;

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let grid_x = (n as u32 + 31) / 32;
    let grid_y = (m as u32 + 31) / 32;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (32, 32, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut c_dev)
            .arg(&m_u32)
            .arg(&n_u32)
            .arg(&k_u32)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;
    stream.clone_dtoh(&c_dev).ok()
}