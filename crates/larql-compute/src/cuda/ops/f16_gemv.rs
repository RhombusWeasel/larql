//! f16 gemv dispatch — CUDA.

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

/// f16 weight × f32 vector → f32 output.
///
/// `out[N] = W[N,K] × X[K]` where W is stored as half-precision (u16 per element).
pub fn encode_f16_gemv(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    w_f16: &[u16],
    x: &[f32],
    n: usize,
    k: usize,
) -> Option<Vec<f32>> {
    let w_dev = stream.clone_htod(w_f16).ok()?;
    let x_dev = stream.clone_htod(x).ok()?;
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

/// f16 gemv + argmax: compute f16 matvec then find the index of the maximum.
///
/// Returns `(argmax_index, argmax_value, scores)` where scores is the full output vector.
pub fn encode_f16_gemv_topk1(
    stream: &Arc<CudaStream>,
    gemv_kernel: &CudaKernel,
    _argmax_kernel: &CudaKernel,
    w_f16: &[u16],
    x: &[f32],
    n: usize,
    k: usize,
) -> Option<(usize, f32, Vec<f32>)> {
    let scores = encode_f16_gemv(stream, gemv_kernel, w_f16, x, n, k)?;

    // Find argmax on CPU (Phase 4 will add GPU argmax)
    let mut best_idx = 0;
    let mut best_val = scores[0];
    for (i, &v) in scores.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Some((best_idx, best_val, scores))
}

/// f16 gemv + partial top-K: compute f16 matvec, return top-K indices.
///
/// Returns top-K (index, value) pairs sorted by value descending.
pub fn encode_f16_gemv_topk(
    stream: &Arc<CudaStream>,
    gemv_kernel: &CudaKernel,
    w_f16: &[u16],
    x: &[f32],
    n: usize,
    k: usize,
    topk: usize,
) -> Option<Vec<(usize, f32)>> {
    let scores = encode_f16_gemv(stream, gemv_kernel, w_f16, x, n, k)?;

    // Sort on CPU for now (Phase 4 will add GPU top-K)
    let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(topk);
    Some(indexed)
}