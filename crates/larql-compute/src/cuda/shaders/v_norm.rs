//! V-norm — parameter-free RMS normalization of V heads (Gemma 4).
//!
//! Each V head is normalised independently: for head h,
//! `v_out[h*hd..(h+1)*hd] = v_out[h*hd+i] / sqrt(mean(v²) + eps)`.
//! No learned weight — just RMSNorm with offset=0.

use crate::cuda::kernel::TiledKernel;

pub struct VNormKernel;

impl TiledKernel for VNormKernel {
    const KERNEL_NAME: &'static str = "v_norm";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const SHADER: &str = r#"
extern "C" __global__ void v_norm(
    float* __restrict__ v_out,        // [num_kv_heads * head_dim] in-place
    const unsigned int head_dim,
    const float eps,
    const unsigned int num_kv_heads)
{
    unsigned int head = blockIdx.x;
    if (head >= num_kv_heads) return;

    float* v_head = v_out + head * head_dim;
    unsigned int lane = threadIdx.x;

    // Cooperative sum-of-squares
    float partial = 0.0f;
    for (unsigned int i = lane; i < head_dim; i += blockDim.x) {
        partial += v_head[i] * v_head[i];
    }
    partial = warp_reduce_sum(partial);

    __shared__ float s_shared[32];
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) s_shared[threadIdx.x / 32] = partial;
    __syncthreads();
    // Thread 0 alone does the final merge
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float sum_sq = s_shared[0];
        for (int i = 1; i < num_warps; i++) sum_sq += s_shared[i];
        s_shared[0] = sum_sq;
    }
    __syncthreads();
    float sum_sq = s_shared[0];

    float rms = 1.0f / sqrtf(sum_sq / (float)head_dim + eps);
    for (unsigned int i = lane; i < head_dim; i += blockDim.x) {
        v_head[i] *= rms;
    }
}
"#;