//! Q4_K matrix-vector multiply dispatch — CUDA.
//!
//! Mirrors the Metal `q4k_matvec` dispatch pattern. Three kernel variants:
//!
//! - **Primary** (4 warp / 128 threads / 4 rows/block): default for most matvec sites
//! - **8-warp** (8 rows/block / 256 threads): higher occupancy for bandwidth-bound kernels
//! - **Stride-32** (8 rows/block / 256 threads): stable reduction tree for LM head argmax

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

/// Dispatch Q4_K matvec on GPU.
///
/// `q4k_data`: packed Q4_K weights `[N, K]` (row-major, `N * (K/256) * 144` bytes)
/// `x`: f32 input vector `[K]`
/// `num_rows`: N (number of output rows)
/// `hidden`: K (input dimension, must be multiple of 256)
///
/// Returns `Some(Vec<f32>)` with `num_rows` output values, or `None` on dispatch failure.
pub fn encode_q4k_matvec(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    q4k_data: &[u8],
    x: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Option<Vec<f32>> {
    if hidden % 256 != 0 || x.len() < hidden {
        return None;
    }

    let w_dev = stream.clone_htod(q4k_data).ok()?;
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

/// Dispatch Q4_K matvec on device buffers (no host readback).
///
/// All buffers reside on the GPU — weights, input, and output.
/// The caller is responsible for synchronisation and buffer lifetime.
///
/// `w_dev`: packed Q4_K weights `[N, K]` already on device
/// `x_dev`: f32 input vector `[K]` already on device
/// `out_dev`: f32 output buffer `[N]` (pre-allocated, will be overwritten)
/// `num_rows`: N (number of output rows)
/// `hidden`: K (input dimension, must be multiple of 256)
pub fn encode_q4k_matvec_on_device(
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