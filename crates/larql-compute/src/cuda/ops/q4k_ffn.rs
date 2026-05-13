//! Fused Q4_K gate+up projection dispatch — CUDA.
//!
//! Computes gate and up projections in one kernel launch, sharing the
//! input vector X between both. Grid: `2 * ceil(N / 4)` blocks —
//! first half → gate, second → up.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

/// Fused Q4_K gate+up dispatch (host → device → host).
///
/// Returns `(gate_out, up_out)` each with `num_rows` f32 values, or `None` on failure.
pub fn encode_q4k_ffn_gate_up(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    wg_data: &[u8],
    wu_data: &[u8],
    x: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Option<(Vec<f32>, Vec<f32>)> {
    if hidden % 256 != 0 || x.len() < hidden {
        return None;
    }

    let wg_dev = stream.clone_htod(wg_data).ok()?;
    let wu_dev = stream.clone_htod(wu_data).ok()?;
    let x_dev = stream.clone_htod(x).ok()?;
    let mut gate_dev = stream.alloc_zeros::<f32>(num_rows).ok()?;
    let mut up_dev = stream.alloc_zeros::<f32>(num_rows).ok()?;

    let n_u32 = num_rows as u32;
    let k_u32 = hidden as u32;
    let tgs_per_mat = (num_rows as u64).div_ceil(kernel.rows_per_block) as u32;
    let total_blocks = tgs_per_mat * 2;
    let block_size = kernel.threads_per_block as u32;

    let cfg = LaunchConfig {
        grid_dim: (total_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&wg_dev)
            .arg(&wu_dev)
            .arg(&x_dev)
            .arg(&mut gate_dev)
            .arg(&mut up_dev)
            .arg(&n_u32)
            .arg(&k_u32)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;

    let gate_out = stream.clone_dtoh(&gate_dev).ok()?;
    let up_out = stream.clone_dtoh(&up_dev).ok()?;

    Some((gate_out, up_out))
}

/// Fused Q4_K gate+up dispatch on device buffers (no host readback).
///
/// Takes pre-uploaded device buffers for weights and input,
/// writes results to pre-allocated device output buffers.
pub fn encode_q4k_ffn_gate_up_on_device(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    wg_dev: &CudaSlice<u8>,
    wu_dev: &CudaSlice<u8>,
    x_dev: &CudaSlice<f32>,
    gate_dev: &mut CudaSlice<f32>,
    up_dev: &mut CudaSlice<f32>,
    num_rows: usize,
    hidden: usize,
) -> Option<()> {
    if hidden % 256 != 0 {
        return None;
    }

    let n_u32 = num_rows as u32;
    let k_u32 = hidden as u32;
    let tgs_per_mat = (num_rows as u64).div_ceil(kernel.rows_per_block) as u32;
    let total_blocks = tgs_per_mat * 2;
    let block_size = kernel.threads_per_block as u32;

    let cfg = LaunchConfig {
        grid_dim: (total_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(wg_dev)
            .arg(wu_dev)
            .arg(x_dev)
            .arg(gate_dev)
            .arg(up_dev)
            .arg(&n_u32)
            .arg(&k_u32)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}