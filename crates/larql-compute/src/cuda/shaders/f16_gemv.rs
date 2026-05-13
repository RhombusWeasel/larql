//! f16 weight × f32 query → f32 output (LM head).
//!
//! 8 rows per block (8 warps), 32 lanes per warp.
//! f16 weights are decoded to f32 inline for accumulator precision.

use crate::cuda::kernel::TiledKernel;

pub struct F16GemvKernel;
impl TiledKernel for F16GemvKernel {
    const KERNEL_NAME: &'static str = "f16_gemv";
    const ROWS_PER_BLOCK: u64 = 8;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const SHADER: &str = r#"
// f16 weight × f32 vector → f32 output.
// 8 rows per block, each warp handles one row.
// 4-way unrolled accumulation with stride-32 access pattern.
extern "C" __global__ void f16_gemv(
    const uint16_local_t* __restrict__ W,     // [N, K] row-major, f16
    const float* __restrict__ X,              // [K] f32 input
    float* __restrict__ out,                  // [N] f32 output
    const unsigned int N,
    const unsigned int K)
{
    unsigned int sg_id = threadIdx.x / 32;
    unsigned int lane = threadIdx.x % 32;
    unsigned int row = blockIdx.x * 8 + sg_id;
    if (row >= N) return;

    const uint16_local_t* w_row = W + (unsigned long)row * K;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    unsigned int k = lane;
    for (; k + 3 * 32 < K; k += 4 * 32) {
        a0 = fma(decode_f16(w_row[k         ]), X[k         ], a0);
        a1 = fma(decode_f16(w_row[k + 32    ]), X[k + 32    ], a1);
        a2 = fma(decode_f16(w_row[k + 64     ]), X[k + 64    ], a2);
        a3 = fma(decode_f16(w_row[k + 96     ]), X[k + 96    ], a3);
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; k < K; k += 32) {
        acc = fma(decode_f16(w_row[k]), X[k], acc);
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;