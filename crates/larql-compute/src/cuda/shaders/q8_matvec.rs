//! Q8_0 matrix-vector multiply.
//!
//! scores[N] = W8[N, K] · Q8_x[K]
//! Q8_0 format: int8 weights + per-block float scales.
//! 8 rows per block (8 warps), 32 lanes per warp.
//! Shared memory for Q8 input to reduce bandwidth.

use crate::cuda::kernel::TiledKernel;

pub struct Q8MatvecKernel;
impl TiledKernel for Q8MatvecKernel {
    const KERNEL_NAME: &'static str = "q8_matvec";
    const ROWS_PER_BLOCK: u64 = 8;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const SHADER: &str = r#"
// Q8 × Q8 matvec: scores[N] = W8[N, K] · Q8_x[K]
// 8 rows per block, 8 warps of 32 threads.
// Shared memory loads Q8 input + scales once per threadgroup.
extern "C" __global__ void q8_matvec(
    const uint8_local_t* __restrict__ W8,     // [N * K] raw int8 weights
    const signed char* __restrict__ Q8,         // [K] Q8 input int8 values
    const float* __restrict__ W8s,            // [N * blocks] weight per-block scales
    const float* __restrict__ Q8s,            // [blocks] input per-block scales
    float* __restrict__ out,                   // [N]
    const unsigned int N,
    const unsigned int K)
{
    __shared__ signed char s_q8[8192];
    __shared__ float s_q8s[256];

    unsigned int blocks = K / 32;
    unsigned int sg_id = threadIdx.x / 32;
    unsigned int lane = threadIdx.x % 32;
    unsigned int row_idx = blockIdx.x * 8 + sg_id;
    unsigned int tid = threadIdx.x;

    // Cooperative load of Q8 input into shared memory
    for (unsigned int i = tid; i < K; i += 256) {
        s_q8[i] = Q8[i];
    }
    for (unsigned int i = tid; i < blocks; i += 256) {
        s_q8s[i] = Q8s[i];
    }
    __syncthreads();

    if (row_idx >= N) return;

    const signed char* row = (const signed char*)(W8 + row_idx * K);
    const float* row_scales = W8s + row_idx * blocks;

    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks; b += 32) {
        float combined_scale = row_scales[b] * s_q8s[b];
        const signed char* wb = row + b * 32;
        const signed char* xb = s_q8 + b * 32;

        int isum = 0;
        for (unsigned int j = 0; j < 32; j++) {
            isum += (int)wb[j] * (int)xb[j];
        }
        acc += (float)isum * combined_scale;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row_idx] = acc;
}
"#;