//! Q6_K matrix-vector multiply dispatch — CUDA.
//!
//! Two kernel variants:
//! - Primary (4 warp / 128 threads / 4 rows/block)
//! - 8-warp (8 rows/block / 256 threads)

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

/// Dispatch Q6_K matvec on GPU.
///
/// `q6k_data`: packed Q6_K weights `[N, K]` (row-major, `N * (K/256) * 210` bytes)
/// `x`: f32 input vector `[K]`
/// `num_rows`: N
/// `hidden`: K (must be multiple of 256)
pub fn encode_q6k_matvec(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    q6k_data: &[u8],
    x: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Option<Vec<f32>> {
    if hidden % 256 != 0 || x.len() < hidden {
        return None;
    }

    let w_dev = stream.clone_htod(q6k_data).ok()?;
    let x_dev = stream.clone_htod(x).ok()?;
    let mut out_dev = stream.alloc_zeros::<f32>(num_rows).ok()?;

    let n_u32 = num_rows as u32;
    let k_u32 = hidden as u32;
    let num_blocks = (num_rows as u64).div_ceil(kernel.rows_per_block) as u32;
    let block_size = kernel.threads_per_block as u32;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
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
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;
    stream.clone_dtoh(&out_dev).ok()
}

/// Dispatch Q6_K matvec on device buffers (no host readback).
pub fn encode_q6k_matvec_on_device(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    w_dev: &CudaSlice<u8>,
    x_dev: &CudaSlice<f32>,
    out_dev: &mut CudaSlice<f32>,
    num_rows: usize,
    hidden: usize,
) -> Option<()> {
    if hidden % 256 != 0 {
        return None;
    }

    let n_u32 = num_rows as u32;
    let k_u32 = hidden as u32;
    let num_blocks = (num_rows as u64).div_ceil(kernel.rows_per_block) as u32;
    let block_size = kernel.threads_per_block as u32;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(w_dev)
            .arg(x_dev)
            .arg(out_dev)
            .arg(&n_u32)
            .arg(&k_u32)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}