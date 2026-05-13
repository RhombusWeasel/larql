//! QK-norm — per-head RMSNorm on Q and K heads. CUDA kernel.
//!
//! Normalises each head independently: for head h with offset,
//! `q_out[h*hd..(h+1)*hd] = (q_out[h*hd+i] + offset) / sqrt(mean(q²) + eps) * w[h*hd+i]`.
//! Used by Gemma 3/4 (QK-norm with learned weights).

use crate::cuda::kernel::TiledKernel;

/// Per-head RMSNorm on both Q and K in a single dispatch.
pub struct QkNormKernel;

impl TiledKernel for QkNormKernel {
    const KERNEL_NAME: &'static str = "qk_norm";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const SHADER: &str = r#"
// QK-norm: per-head RMS norm on Q and K heads.
// Q and K are in separate buffers (matching Metal's qk_norm_qk kernel).
// Each thread block handles one head.
extern "C" __global__ void qk_norm(
    float* __restrict__ q_out,         // [num_q_heads * head_dim]
    float* __restrict__ k_out,         // [num_kv_heads * head_dim]
    const float* __restrict__ q_norm_w, // [head_dim] Q norm weights
    const float* __restrict__ k_norm_w, // [head_dim] K norm weights
    const unsigned int head_dim,
    const unsigned int num_q_heads,
    const float eps,
    const float offset)
{
    // total_heads = num_q_heads + num_kv_heads
    // blockIdx.x selects the head (0..num_q-1 for Q, num_q..total-1 for K)
    unsigned int head = blockIdx.x;
    unsigned int lane = threadIdx.x;

    bool is_q = (head < num_q_heads);
    unsigned int local_head = is_q ? head : (head - num_q_heads);
    float* head_ptr = is_q
        ? (q_out + local_head * head_dim)
        : (k_out + local_head * head_dim);
    const float* norm_w = is_q ? q_norm_w : k_norm_w;

    // Cooperative sum-of-squares within this block
    float partial = 0.0f;
    for (unsigned int i = lane; i < head_dim; i += blockDim.x) {
        float v = head_ptr[i];
        partial += v * v;
    }
    partial = warp_reduce_sum(partial);
    __shared__ float s_shared[32];
    int lane_id = lane % 32;
    if (lane_id == 0) s_shared[lane / 32] = partial;
    __syncthreads();

    // Thread 0 does final merge, store in shared for all threads
    __shared__ float total_sum_sq;
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float t = s_shared[0];
        for (int i = 1; i < num_warps; i++) t += s_shared[i];
        total_sum_sq = t;
    }
    __syncthreads();
    float sum_sq = total_sum_sq;

    float rms = 1.0f / sqrtf(sum_sq / (float)head_dim + eps);
    for (unsigned int i = lane; i < head_dim; i += blockDim.x) {
        head_ptr[i] = head_ptr[i] * rms * (norm_w[i] + offset);
    }
}
"#;