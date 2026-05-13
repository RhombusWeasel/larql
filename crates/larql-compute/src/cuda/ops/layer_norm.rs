//! Layer norm dispatch — CUDA.
//!
//! `layer_norm`: out = (x - mean) / sqrt(var + eps) * (weight + offset) + bias
//! `layer_norm_no_bias`: out = (x - mean) / sqrt(var + eps) * (weight + offset)

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

/// Compute LayerNorm: `out[i] = (x[i] - mean) / sqrt(var + eps) * (weight[i] + offset) + bias[i]`.
pub fn encode_layer_norm(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    len: usize,
    eps: f32,
    offset: f32,
) -> Option<Vec<f32>> {
    if x.len() < len || weight.len() < len || bias.len() < len {
        return None;
    }

    let x_dev = stream.clone_htod(&x[..len]).ok()?;
    let w_dev = stream.clone_htod(&weight[..len]).ok()?;
    let b_dev = stream.clone_htod(&bias[..len]).ok()?;
    let mut out_dev = stream.alloc_zeros::<f32>(len).ok()?;

    let len_u32 = len as u32;
    let block_size = len.min(1024) as u32;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&x_dev)
            .arg(&w_dev)
            .arg(&b_dev)
            .arg(&mut out_dev)
            .arg(&len_u32)
            .arg(&eps)
            .arg(&offset)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;
    stream.clone_dtoh(&out_dev).ok()
}

/// Compute LayerNorm without bias: `out[i] = (x[i] - mean) / sqrt(var + eps) * (weight[i] + offset)`.
pub fn encode_layer_norm_no_bias(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    x: &[f32],
    weight: &[f32],
    len: usize,
    eps: f32,
    offset: f32,
) -> Option<Vec<f32>> {
    if x.len() < len || weight.len() < len {
        return None;
    }

    let x_dev = stream.clone_htod(&x[..len]).ok()?;
    let w_dev = stream.clone_htod(&weight[..len]).ok()?;
    let mut out_dev = stream.alloc_zeros::<f32>(len).ok()?;

    let len_u32 = len as u32;
    let block_size = len.min(1024) as u32;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&x_dev)
            .arg(&w_dev)
            .arg(&mut out_dev)
            .arg(&len_u32)
            .arg(&eps)
            .arg(&offset)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;
    stream.clone_dtoh(&out_dev).ok()
}