//! f32 argmax and top-K partial dispatch — port of `metal::direct_ops` argmax path.
//!
//! Two-phase pattern:
//! 1. GPU: per-block partial reduction → `num_blocks` (val, idx) pairs
//! 2. CPU: merge partial results into the final top-1 or top-K

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;
use crate::cuda::shaders::f32_argmax::{K_TOPK, PARTIAL_BLOCK_SIZE};

/// Find argmax of a scores buffer on GPU using partial-block reduction.
///
/// Returns `(index, value)` of the maximum element. The GPU does per-block
/// partial argmax; the CPU merges the partial results.
pub fn encode_f32_argmax_partial(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    scores: &[f32],
) -> Option<(u32, f32)> {
    if scores.is_empty() {
        return None;
    }

    let n = scores.len();
    let num_blocks = (n as u64).div_ceil(PARTIAL_BLOCK_SIZE) as u32;

    // Copy scores to device
    let scores_dev = stream.clone_htod(scores).ok()?;

    // Allocate output buffers
    let mut out_val = stream.alloc_zeros::<f32>(num_blocks as usize).ok()?;
    let mut out_idx = stream.alloc_zeros::<u32>(num_blocks as usize).ok()?;

    // Launch kernel
    let n_u32 = n as u32;
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (PARTIAL_BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&scores_dev)
            .arg(&mut out_val)
            .arg(&mut out_idx)
            .arg(&n_u32)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;

    // Read back partial results
    let partial_vals = stream.clone_dtoh(&out_val).ok()?;
    let partial_idxs = stream.clone_dtoh(&out_idx).ok()?;

    // CPU: merge partial results
    let mut best_val = f32::NEG_INFINITY;
    let mut best_idx: u32 = 0;
    for i in 0..num_blocks as usize {
        let v = partial_vals[i];
        let idx = partial_idxs[i];
        if v > best_val || (v == best_val && idx < best_idx) {
            best_val = v;
            best_idx = idx;
        }
    }

    Some((best_idx, best_val))
}

/// Find top-K elements of a scores buffer on GPU using partial-block reduction.
///
/// Returns a Vec of `(index, value)` pairs sorted by score descending,
/// up to `top_k` entries. If `top_k > K_TOPK`, returns None.
pub fn encode_f32_topk_partial(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    scores: &[f32],
    top_k: usize,
) -> Option<Vec<(u32, f32)>> {
    if top_k == 0 || top_k > K_TOPK || scores.is_empty() {
        return None;
    }

    let n = scores.len();
    let num_blocks = (n as u64).div_ceil(PARTIAL_BLOCK_SIZE) as u32;

    // Copy scores to device
    let scores_dev = stream.clone_htod(scores).ok()?;

    // Allocate output buffers
    let out_len = num_blocks as usize * K_TOPK;
    let mut out_val = stream.alloc_zeros::<f32>(out_len).ok()?;
    let mut out_idx = stream.alloc_zeros::<u32>(out_len).ok()?;

    // Launch kernel
    let n_u32 = n as u32;
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (PARTIAL_BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&scores_dev)
            .arg(&mut out_val)
            .arg(&mut out_idx)
            .arg(&n_u32)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;

    // Read back partial results
    let partial_vals = stream.clone_dtoh(&out_val).ok()?;
    let partial_idxs = stream.clone_dtoh(&out_idx).ok()?;

    // CPU: merge partial top-K results
    let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(out_len);
    for i in 0..out_len {
        candidates.push((partial_idxs[i], partial_vals[i]));
    }

    // Sort descending by score (stable on index for ties)
    candidates.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    // Deduplicate by index
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    for (idx, val) in candidates {
        if seen.insert(idx) {
            result.push((idx, val));
            if result.len() >= top_k {
                break;
            }
        }
    }

    Some(result)
}

/// Convenience: f32 gemv + GPU argmax → single GPU dispatch + partial reduction.
/// Returns `(token_id, score)` for the top-1 element without materialising
/// the full scores vector on CPU.
pub fn encode_f32_gemv_topk1(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    gemv_kernel: &CudaKernel,
    argmax_kernel: &CudaKernel,
    w: ndarray::ArrayView2<f32>,
    x: &[f32],
) -> Option<(u32, f32)> {
    let n = w.shape()[0];
    let k = w.shape()[1];
    if x.len() != k {
        return None;
    }

    // Step 1: f32 gemv → scores on GPU
    let w_flat: Vec<f32> = w.iter().copied().collect();
    let w_dev = stream.clone_htod(&w_flat).ok()?;
    let x_dev = stream.clone_htod(x).ok()?;
    let mut scores_dev = stream.alloc_zeros::<f32>(n).ok()?;

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let num_blocks = (n as u64).div_ceil(gemv_kernel.rows_per_block) as u32;
    let block_size = gemv_kernel.threads_per_block as u32;

    let cfg_gemv = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&gemv_kernel.func)
            .arg(&w_dev)
            .arg(&x_dev)
            .arg(&mut scores_dev)
            .arg(&n_u32)
            .arg(&k_u32)
            .launch(cfg_gemv)
            .ok()?;
    }

    // Step 2: argmax partial reduction (same stream, no sync needed between)
    let argmax_num_blocks = (n as u64).div_ceil(PARTIAL_BLOCK_SIZE) as u32;
    let mut out_val = stream.alloc_zeros::<f32>(argmax_num_blocks as usize).ok()?;
    let mut out_idx = stream.alloc_zeros::<u32>(argmax_num_blocks as usize).ok()?;

    let cfg_argmax = LaunchConfig {
        grid_dim: (argmax_num_blocks, 1, 1),
        block_dim: (PARTIAL_BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&argmax_kernel.func)
            .arg(&scores_dev)
            .arg(&mut out_val)
            .arg(&mut out_idx)
            .arg(&n_u32)
            .launch(cfg_argmax)
            .ok()?;
    }

    stream.synchronize().ok()?;

    // Read back only the partial results (much smaller than the full scores)
    let partial_vals = stream.clone_dtoh(&out_val).ok()?;
    let partial_idxs = stream.clone_dtoh(&out_idx).ok()?;

    // CPU merge
    let mut best_val = f32::NEG_INFINITY;
    let mut best_idx: u32 = 0;
    for i in 0..argmax_num_blocks as usize {
        let v = partial_vals[i];
        let idx = partial_idxs[i];
        if v > best_val || (v == best_val && idx < best_idx) {
            best_val = v;
            best_idx = idx;
        }
    }

    Some((best_idx, best_val))
}