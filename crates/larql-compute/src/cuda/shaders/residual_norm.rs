//! Residual + norm fused operations — CUDA kernels.
//!
//! `residual_norm`: out = rmsnorm(h + attn_out, weight, eps, offset)
//! `residual_norm_store`: same, but also writes the raw residual (h + attn_out)
//! `residual_norm_q8`: residual + rmsnorm + Q8 quantize in one pass

use crate::cuda::kernel::TiledKernel;

pub struct ResidualNormKernel;
impl TiledKernel for ResidualNormKernel {
    const KERNEL_NAME: &'static str = "residual_norm";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub struct ResidualNormStoreKernel;
impl TiledKernel for ResidualNormStoreKernel {
    const KERNEL_NAME: &'static str = "residual_norm_store";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub struct PostAttnResidualNormStoreKernel;
impl TiledKernel for PostAttnResidualNormStoreKernel {
    const KERNEL_NAME: &'static str = "post_attn_residual_norm_store";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub struct PostFfnNormResidualAddKernel;
impl TiledKernel for PostFfnNormResidualAddKernel {
    const KERNEL_NAME: &'static str = "post_ffn_norm_residual_add";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const RESIDUAL_NORM_SHADER: &str = r#"
// residual_norm: norm_out = rmsnorm(h + residual, weight, eps, offset)
// Also used as the generic fused residual+norm path.
extern "C" __global__ void residual_norm(
    const float* __restrict__ h,          // [hidden] residual base
    const float* __restrict__ residual,    // [hidden] residual addend (e.g. attn_out)
    const float* __restrict__ weight,      // [hidden] norm weight
    float* __restrict__ norm_out,         // [hidden] output
    const unsigned int len,
    const float eps,
    const float offset)
{
    unsigned int lane = threadIdx.x;

    // Cooperative sum of (h+residual)²
    float sum_sq = 0.0f;
    for (unsigned int i = lane; i < len; i += blockDim.x) {
        float v = h[i] + residual[i];
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float s_shared[32];
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) s_shared[threadIdx.x / 32] = sum_sq;
    __syncthreads();
    // Thread 0 alone does the final merge, storing result in shared for all threads
    __shared__ float total_sum_sq;
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float t = s_shared[0];
        for (int i = 1; i < num_warps; i++) t += s_shared[i];
        total_sum_sq = t;
    }
    __syncthreads();
    sum_sq = total_sum_sq;

    float rms = 1.0f / sqrtf(sum_sq / (float)len + eps);
    for (unsigned int i = lane; i < len; i += blockDim.x) {
        norm_out[i] = (h[i] + residual[i]) * rms * (weight[i] + offset);
    }
}
"#;

pub const RESIDUAL_NORM_STORE_SHADER: &str = r#"
// residual_norm_store: Computes residual = h + attn_out, then:
//   h_post_attn = residual
//   ffn_norm_out = rmsnorm(residual, weight, eps, offset)
// Used for Q4_K family (no Q8 quantize needed).
extern "C" __global__ void residual_norm_store(
    const float* __restrict__ h,          // [hidden] residual base
    const float* __restrict__ attn_out,   // [hidden] attention output (or o_proj result)
    const float* __restrict__ weight,     // [hidden] post-attn norm weight
    float* __restrict__ ffn_norm_out,     // [hidden] normed output (FFN input)
    float* __restrict__ h_post_attn,      // [hidden] raw residual
    const unsigned int len,
    const float eps,
    const float offset)
{
    unsigned int lane = threadIdx.x;

    float sum_sq = 0.0f;
    for (unsigned int i = lane; i < len; i += blockDim.x) {
        float v = h[i] + attn_out[i];
        h_post_attn[i] = v;
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float s_shared[32];
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) s_shared[threadIdx.x / 32] = sum_sq;
    __syncthreads();
    // Thread 0 alone does the final merge, store in shared for all threads
    __shared__ float total_sum_sq;
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float t = s_shared[0];
        for (int i = 1; i < num_warps; i++) t += s_shared[i];
        total_sum_sq = t;
    }
    __syncthreads();
    sum_sq = total_sum_sq;

    float rms = 1.0f / sqrtf(sum_sq / (float)len + eps);
    for (unsigned int i = lane; i < len; i += blockDim.x) {
        float residual = h_post_attn[i];
        ffn_norm_out[i] = residual * rms * (weight[i] + offset);
    }
}
"#;

pub const POST_ATTN_RESIDUAL_NORM_STORE_SHADER: &str = r#"
// Triple-fused: post_attn_norm(o_out) + normed + pre_ffn_norm → ffn_norm_out,
// plus h + o_out → h_post_attn.
// h_post_attn = h + o_out
// ffn_norm_out = pre_ffn_norm(rmsnorm(h + o_out, post_attn_norm, eps, offset))
// When pre_ffn_norm == post_attn_norm (the common case), this simplifies to
// a single norm.
extern "C" __global__ void post_attn_residual_norm_store(
    const float* __restrict__ h,             // [hidden] layer input
    const float* __restrict__ o_out,         // [hidden] O projection output
    const float* __restrict__ post_attn_norm,// [hidden] post-attn norm weight
    const float* __restrict__ pre_ffn_norm,  // [hidden] pre-FFN norm weight (or same as post_attn)
    float* __restrict__ ffn_norm_out,       // [hidden] FFN input (normed)
    float* __restrict__ h_post_attn,        // [hidden] raw residual h + o_out
    const unsigned int len,
    const float eps,
    const float offset)
{
    unsigned int lane = threadIdx.x;

    float sum_sq = 0.0f;
    for (unsigned int i = lane; i < len; i += blockDim.x) {
        float v = h[i] + o_out[i];
        h_post_attn[i] = v;
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float s_shared[32];
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) s_shared[threadIdx.x / 32] = sum_sq;
    __syncthreads();
    __shared__ float total_sum_sq;
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float t = s_shared[0];
        for (int i = 1; i < num_warps; i++) t += s_shared[i];
        total_sum_sq = t;
    }
    __syncthreads();
    sum_sq = total_sum_sq;

    float rms = 1.0f / sqrtf(sum_sq / (float)len + eps);
    for (unsigned int i = lane; i < len; i += blockDim.x) {
        float residual = h_post_attn[i];
        float normed = residual * rms * (post_attn_norm[i] + offset);
        ffn_norm_out[i] = normed * (pre_ffn_norm[i] + offset);
    }
}
"#;

pub const POST_FFN_NORM_RESIDUAL_ADD_SHADER: &str = r#"
// Post-FFN norm + residual add:
// new_h = h_post_attn + rmsnorm(down_out, post_ffn_norm, eps, offset)
// This kernel also optionally applies a layer scalar.
extern "C" __global__ void post_ffn_norm_residual_add(
    const float* __restrict__ down_out,     // [hidden] FFN down projection output
    const float* __restrict__ h_post_attn,  // [hidden] residual base (h + attn_out)
    const float* __restrict__ norm_weight,   // [hidden] post-FFN norm weight
    float* __restrict__ new_h,              // [hidden] output
    const unsigned int len,
    const float eps,
    const float offset,
    const float layer_scalar)
{
    unsigned int lane = threadIdx.x;

    float sum_sq = 0.0f;
    for (unsigned int i = lane; i < len; i += blockDim.x) {
        float v = down_out[i];
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float s_shared[32];
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) s_shared[threadIdx.x / 32] = sum_sq;
    __syncthreads();
    __shared__ float total_sum_sq;
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float t = s_shared[0];
        for (int i = 1; i < num_warps; i++) t += s_shared[i];
        total_sum_sq = t;
    }
    __syncthreads();
    sum_sq = total_sum_sq;

    float rms = 1.0f / sqrtf(sum_sq / (float)len + eps);
    for (unsigned int i = lane; i < len; i += blockDim.x) {
        float normed = down_out[i] * rms * (norm_weight[i] + offset);
        new_h[i] = h_post_attn[i] + normed;
        if (layer_scalar != 0.0f && layer_scalar != 1.0f) {
            new_h[i] *= layer_scalar;
        }
    }
}
"#;

pub const RMS_NORM_Q8_SHADER: &str = r#"
// Fused RMS norm + Q8 quantize.
// norm_out = x * (weight + offset) / sqrt(mean(x²) + eps)
// Then quantize norm_out → q8_out + scales.
extern "C" __global__ void rms_norm_q8(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    signed char* __restrict__ q8_out,
    float* __restrict__ scales,
    const unsigned int len,
    const float eps,
    const float offset)
{
    unsigned int lane = threadIdx.x;

    float partial = 0.0f;
    for (unsigned int i = lane; i < len; i += blockDim.x) {
        partial += x[i] * x[i];
    }
    partial = block_reduce_sum(partial);
    float rms = 1.0f / sqrtf(partial / (float)len + eps);

    unsigned int num_blocks_q8 = len / 32;
    for (unsigned int i = lane; i < len; i += blockDim.x) {
        float normed = x[i] * (weight[i] + offset) * rms;

        unsigned int block = i / 32;
        // Find block max for quantization (warp-level)
        float my_abs = fabsf(normed);
        float block_max = my_abs;
        __shared__ float s_block_max[1024]; // max 1024 threads
        {
            block_max = warp_reduce_max(block_max);
            int warp_id = threadIdx.x / 32;
            int lane_id = threadIdx.x % 32;
            if (lane_id == 0) s_block_max[warp_id] = block_max;
            __syncthreads();
            int num_warps = (blockDim.x + 31) / 32;
            block_max = (lane_id < num_warps) ? s_block_max[lane_id] : 0.0f;
            block_max = warp_reduce_max(block_max);
        }
        // Note: only thread 0 of each warp will have the correct block_max
        // This is a simplified version; production should use per-block reduction
        float scale = block_max / 127.0f;
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        if ((i % 32) == 0) scales[block] = scale;
        int q = (int)roundf(normed * inv_scale);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        q8_out[i] = (signed char)q;
    }
}
"#;

pub const RESIDUAL_NORM_Q8_SHADER: &str = r#"
// Fused residual add + RMS norm + Q8 quantize.
// f32_out[i] = a[i] + b[i]
// norm_out = f32_out * (weight + offset) / sqrt(mean(f32_out²) + eps)
// q8_out = quantize(norm_out), scales = per-block max/127
extern "C" __global__ void residual_norm_q8(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ weight,
    signed char* __restrict__ q8_out,
    float* __restrict__ scales,
    float* __restrict__ f32_out,
    const unsigned int len,
    const float eps,
    const float offset)
{
    unsigned int lane = threadIdx.x;

    // Write f32 sum first
    for (unsigned int i = lane; i < len; i += blockDim.x) {
        f32_out[i] = a[i] + b[i];
    }

    // Cooperative sum-of-squares
    float partial = 0.0f;
    for (unsigned int i = lane; i < len; i += blockDim.x) {
        float v = f32_out[i];
        partial += v * v;
    }
    partial = block_reduce_sum(partial);
    float rms = 1.0f / sqrtf(partial / (float)len + eps);

    for (unsigned int i = lane; i < len; i += blockDim.x) {
        float normed = f32_out[i] * (weight[i] + offset) * rms;

        // Simple per-block Q8 quantization
        unsigned int block = i / 32;
        float my_abs = fabsf(normed);
        float block_max = my_abs;
        block_max = warp_reduce_max(block_max);
        __shared__ float s_block_max[32];
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        if (lane_id == 0) s_block_max[warp_id] = block_max;
        __syncthreads();
        int num_warps = (blockDim.x + 31) / 32;
        block_max = (lane_id < num_warps) ? s_block_max[lane_id] : 0.0f;
        block_max = warp_reduce_max(block_max);

        float scale = block_max / 127.0f;
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        if ((i % 32) == 0) scales[block] = scale;
        int q = (int)roundf(normed * inv_scale);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        q8_out[i] = (signed char)q;
    }
}
"#;