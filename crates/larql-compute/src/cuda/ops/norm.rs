//! RMS norm, residual add, scale vector, and GEGLU+down dispatch — CUDA.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

// ── RMS norm ──

/// Compute RMS norm: `out[i] = x[i] * (weight[i] + offset) / sqrt(mean(x²) + eps)`.
///
/// Dispatched as 1 block with up to 1024 threads. Hidden sizes > 1024
/// are handled by the stride loop inside the kernel.
pub fn encode_rms_norm(
    _ctx: &Arc<CudaContext>,
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

// ── Residual add ──

/// Compute residual add: `out[i] = a[i] + b[i]`.
pub fn encode_residual_add(
    _ctx: &Arc<CudaContext>,
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

// ── Scale vector ──

/// Compute scale vector: `out[i] = input[i] * scalar`.
pub fn encode_scale_vector(
    _ctx: &Arc<CudaContext>,
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

// ── Fused GEGLU + Q4_K down projections ──

/// Fused SiLU activation + Q4_K down projection.
///
/// Computes `down_out[row] = Σ_i W_down[row,i] * (silu(gate[i]) * up[i])`
/// where `silu(x) = x / (1 + exp(-x))`.
///
/// Returns `Vec<f32>` with `N` (hidden) values, or `None` on failure.
pub fn encode_q4k_geglu_silu_down(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    w_down: &[u8],
    gate: &[f32],
    up: &[f32],
    n: usize,
    k: usize,
) -> Option<Vec<f32>> {
    if k % 256 != 0 || gate.len() < k || up.len() < k {
        return None;
    }

    let w_dev = stream.clone_htod(w_down).ok()?;
    let gate_dev = stream.clone_htod(gate).ok()?;
    let up_dev = stream.clone_htod(up).ok()?;
    let mut out_dev = stream.alloc_zeros::<f32>(n).ok()?;

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let num_blocks = (n as u64).div_ceil(kernel.rows_per_block) as u32;
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
            .arg(&gate_dev)
            .arg(&up_dev)
            .arg(&mut out_dev)
            .arg(&n_u32)
            .arg(&k_u32)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;
    stream.clone_dtoh(&out_dev).ok()
}

/// Fused GELU-tanh activation + Q4_K down projection.
///
/// Computes `down_out[row] = Σ_i W_down[row,i] * (gelu_tanh(gate[i]) * up[i])`
/// where `gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))`.
///
/// Returns `Vec<f32>` with `N` (hidden) values, or `None` on failure.
pub fn encode_q4k_geglu_gelu_tanh_down(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    w_down: &[u8],
    gate: &[f32],
    up: &[f32],
    n: usize,
    k: usize,
) -> Option<Vec<f32>> {
    if k % 256 != 0 || gate.len() < k || up.len() < k {
        return None;
    }

    let w_dev = stream.clone_htod(w_down).ok()?;
    let gate_dev = stream.clone_htod(gate).ok()?;
    let up_dev = stream.clone_htod(up).ok()?;
    let mut out_dev = stream.alloc_zeros::<f32>(n).ok()?;

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let num_blocks = (n as u64).div_ceil(kernel.rows_per_block) as u32;
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
            .arg(&gate_dev)
            .arg(&up_dev)
            .arg(&mut out_dev)
            .arg(&n_u32)
            .arg(&k_u32)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;
    stream.clone_dtoh(&out_dev).ok()
}

// ── On-device GEGLU + Q4_K down variants (device-to-device, no host readback) ──

/// Fused SiLU activation + Q4_K down projection on device buffers.
///
/// All buffers reside on the GPU. The caller is responsible for
/// synchronisation and buffer lifetime.
///
/// `w_down`: packed Q4_K down weights `[N, K]` already on device
/// `gate`: gate output `[K]` already on device
/// `up`: up output `[K]` already on device
/// `out_dev`: output buffer `[N]` (pre-allocated, will be overwritten)
/// `n`: N (hidden dimension — number of output rows)
/// `k`: K (intermediate dimension, must be multiple of 256)
pub fn encode_q4k_geglu_silu_down_on_device(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    w_down: &CudaSlice<u8>,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    out_dev: &mut CudaSlice<f32>,
    n: usize,
    k: usize,
) -> Option<()> {
    if k % 256 != 0 {
        return None;
    }

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let num_blocks = (n as u64).div_ceil(kernel.rows_per_block) as u32;
    let block_size = kernel.threads_per_block as u32;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(w_down)
            .arg(gate)
            .arg(up)
            .arg(out_dev)
            .arg(&n_u32)
            .arg(&k_u32)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// Fused GELU-tanh activation + Q4_K down projection on device buffers.
///
/// All buffers reside on the GPU. The caller is responsible for
/// synchronisation and buffer lifetime.
pub fn encode_q4k_geglu_gelu_tanh_down_on_device(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    w_down: &CudaSlice<u8>,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    out_dev: &mut CudaSlice<f32>,
    n: usize,
    k: usize,
) -> Option<()> {
    if k % 256 != 0 {
        return None;
    }

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let num_blocks = (n as u64).div_ceil(kernel.rows_per_block) as u32;
    let block_size = kernel.threads_per_block as u32;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(w_down)
            .arg(gate)
            .arg(up)
            .arg(out_dev)
            .arg(&n_u32)
            .arg(&k_u32)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}