//! f32 gemv — matrix-vector multiply for the LM head.
//!
//! Port of `metal/shaders/f32_gemv.rs`. Computes `out[N] = W[N, K] · x[K]`
//! where W is row-major f32.
//!
//! One warp per output row. Each of the 32 lanes reads K/32 strided
//! elements, accumulates a partial dot product, then warp_reduce_sum
//! reduces into a single output value.
//!
//! Sized for Gemma 3/4 tied LM head: N ~ 262K, K = 2560–5120.

use crate::cuda::kernel::TiledKernel;

/// CUDA kernel source for f32 gemv.
pub const SHADER: &str = r#"
extern "C" __global__ void f32_gemv(
    const float* __restrict__ W,   // [N, K] row-major
    const float* __restrict__ X,   // [K]
    float* __restrict__ out,       // [N]
    const unsigned int N,
    const unsigned int K)
{
    // 8 warps per block, one row per warp = 8 rows per block
    // Mirrors Metal's F32GEMV_SG_PER_TG = 8
    const unsigned int ROWS_PER_BLOCK = 8;
    unsigned int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / 32);

    if (row >= N) return;

    const float* w_row = W + (unsigned long long)row * K;

    float acc = 0.0f;
    // Stride-32 over K; four unrolled per-lane accumulators
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    unsigned int lane = threadIdx.x % 32;
    unsigned int k = lane;
    for (; k + 3 * 32 < K; k += 4 * 32) {
        a0 = fma(w_row[k],        X[k],        a0);
        a1 = fma(w_row[k + 32],   X[k + 32],   a1);
        a2 = fma(w_row[k + 64],   X[k + 64],   a2);
        a3 = fma(w_row[k + 96],   X[k + 96],   a3);
    }
    acc = (a0 + a1) + (a2 + a3);
    for (; k < K; k += 32) {
        acc += w_row[k] * X[k];
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

/// Tiled kernel marker: 8 rows per block (8 warps × 32 threads = 256 threads/block).
pub struct F32GemvKernel;
impl TiledKernel for F32GemvKernel {
    const KERNEL_NAME: &'static str = "f32_gemv";
    const ROWS_PER_BLOCK: u64 = 8;
    const THREADS_PER_BLOCK: u64 = 256; // 8 warps × 32 threads
}