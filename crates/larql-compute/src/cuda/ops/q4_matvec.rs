//! Q4_0 × Q8 matrix-vector dispatch — CUDA.

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

/// Q4_0 × Q8 matvec: `out[N] = Q4[N,K] · Q8_x[K]`
///
/// Q4_0 format: 18 bytes per block of 32 values (2B f16 scale + 16B nibbles).
/// Q8 format: int8 values + per-block float scales.
pub fn encode_q4_matvec(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    q4_data: &[u8],
    q8_data: &[i8],
    q8_scales: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Option<Vec<f32>> {
    if hidden % 32 != 0 {
        return None;
    }

    let q4_dev = stream.clone_htod(q4_data).ok()?;
    let q8_dev = stream.clone_htod(q8_data).ok()?;
    let q8s_dev = stream.clone_htod(q8_scales).ok()?;
    let mut out_dev = stream.alloc_zeros::<f32>(num_rows).ok()?;

    let num_rows_u32 = num_rows as u32;
    let hidden_u32 = hidden as u32;
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
            .arg(&q4_dev)
            .arg(&q8_dev)
            .arg(&q8s_dev)
            .arg(&mut out_dev)
            .arg(&num_rows_u32)
            .arg(&hidden_u32)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;
    stream.clone_dtoh(&out_dev).ok()
}

/// Q4_0 vector-matrix (scatter-accumulate): `out[K] = activation[N] @ Q4[N,K]`
///
/// One thread per output column (K elements).
pub fn encode_q4_vecmat(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    activation: &[f32],
    q4_data: &[u8],
    n: usize,
    k: usize,
) -> Option<Vec<f32>> {
    if k % 32 != 0 {
        return None;
    }

    let act_dev = stream.clone_htod(activation).ok()?;
    let q4_dev = stream.clone_htod(q4_data).ok()?;
    let mut out_dev = stream.alloc_zeros::<f32>(k).ok()?;

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let total_threads = k as u32;
    let threads = 256u32;
    let blocks = (total_threads as u64).div_ceil(threads as u64) as u32;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&act_dev)
            .arg(&q4_dev)
            .arg(&mut out_dev)
            .arg(&n_u32)
            .arg(&k_u32)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;
    stream.clone_dtoh(&out_dev).ok()
}

/// Q4_0 × f32 matvec: `out[N] = Q4[N,K] · x[K]`
///
/// Input x is f32 (not Q8). Each thread handles one output row.
pub fn encode_q4_f32_matvec(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    q4_data: &[u8],
    x: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Option<Vec<f32>> {
    if hidden % 32 != 0 {
        return None;
    }

    let q4_dev = stream.clone_htod(q4_data).ok()?;
    let x_dev = stream.clone_htod(x).ok()?;
    let mut out_dev = stream.alloc_zeros::<f32>(num_rows).ok()?;

    let n_u32 = num_rows as u32;
    let k_u32 = hidden as u32;
    let total_threads = num_rows as u32;
    let threads = 256u32;
    let blocks = (total_threads as u64).div_ceil(threads as u64) as u32;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&q4_dev)
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