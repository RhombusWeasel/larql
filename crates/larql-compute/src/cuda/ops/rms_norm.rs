//! RMS norm, residual add, scale vector dispatch — port of `metal/shaders/residual_inject.rs`.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

/// Compute RMS norm: `out[i] = x[i] * (weight[i] + offset) / sqrt(mean(x²) + eps)`.
///
/// Dispatched as 1 block with up to 1024 threads. Hidden sizes > 1024
/// are handled by the stride loop inside the kernel.
#[allow(unused_variables)]
pub fn encode_rms_norm(
    ctx: &Arc<CudaContext>,
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
    // Use up to 1024 threads for large hidden sizes
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

/// Compute residual add: `out[i] = a[i] + b[i]`.
#[allow(unused_variables)]
pub fn encode_residual_add(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    a: &[f32],
    b: &[f32],
    len: usize,
) -> Option<Vec<f32>> {
    if a.len() < len || b.len() < len {
        return None;
    }

    let a_dev = stream.clone_htod(&a[..len]).ok()?;
    let b_dev = stream.clone_htod(&b[..len]).ok()?;
    let mut out_dev = stream.alloc_zeros::<f32>(len).ok()?;

    let len_u32 = len as u32;
    let threads = 256u32;
    let blocks = (len as u64).div_ceil(threads as u64) as u32;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut out_dev)
            .arg(&len_u32)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;
    stream.clone_dtoh(&out_dev).ok()
}

/// Compute scale vector: `out[i] = input[i] * scalar`.
#[allow(unused_variables)]
pub fn encode_scale_vector(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    input: &[f32],
    scalar: f32,
    len: usize,
) -> Option<Vec<f32>> {
    if input.len() < len {
        return None;
    }

    let in_dev = stream.clone_htod(&input[..len]).ok()?;
    let mut out_dev = stream.alloc_zeros::<f32>(len).ok()?;

    let len_u32 = len as u32;
    let threads = 256u32;
    let blocks = (len as u64).div_ceil(threads as u64) as u32;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&in_dev)
            .arg(&mut out_dev)
            .arg(&len_u32)
            .arg(&scalar)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;
    stream.clone_dtoh(&out_dev).ok()
}