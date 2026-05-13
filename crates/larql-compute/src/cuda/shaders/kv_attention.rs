//! KV cache operations — append and attend. CUDA kernels.
//!
//! KV Append: writes new K/V row into the pre-allocated cache at position `pos`.
//! KV Attend: computes softmax(Q * K^T / sqrt(d)) * V for a single decode step.

use crate::cuda::kernel::TiledKernel;

/// KV cache append: copies K and V vectors into the cache at position `pos`.
/// Layout: k_cache[max_seq * num_kv_heads * head_dim], v_cache same.
pub struct KvAppendKernel;

impl TiledKernel for KvAppendKernel {
    const KERNEL_NAME: &'static str = "kv_append";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

/// KV cache attend: computes softmax attention for a single token.
/// Q shape: [num_q_heads, head_dim], K/V shape: [pos+1, num_kv_heads, head_dim].
/// Output: attn_out[num_q_heads * head_dim].
///
/// Short variant: uses shared memory for QK dot products, limited to
/// attention spans ≤ 1024 positions.
pub struct KvAttendKernel;

impl TiledKernel for KvAttendKernel {
    const KERNEL_NAME: &'static str = "kv_attend";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

/// KV cache attend: long-context variant for spans > 1024.
/// Uses global memory for partial scores instead of shared memory.
pub struct KvAttendLongKernel;

impl TiledKernel for KvAttendLongKernel {
    const KERNEL_NAME: &'static str = "kv_attend_long";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

/// Fused KV append + attend: copies K/V into cache AND computes attention
/// in a single kernel launch. More efficient for short sequences.
pub struct KvAppendAttendFusedKernel;

impl TiledKernel for KvAppendAttendFusedKernel {
    const KERNEL_NAME: &'static str = "kv_append_attend_fused";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

/// Short attention span threshold: above this, use the long-context kernel.
pub const SHORT_ATTENTION_SPAN: u32 = 1024;

pub const KV_APPEND_SHADER: &str = r#"
// KV cache append: copy new K and V into pre-allocated cache.
// k_cache layout: [max_seq, num_kv_heads, head_dim]
// v_cache layout: same
// new_k layout: [num_kv_heads * head_dim]
// new_v layout: same
// Position `pos` in the cache is at offset pos * num_kv_heads * head_dim.
extern "C" __global__ void kv_append(
    const float* __restrict__ new_k,     // [num_kv_heads * head_dim]
    const float* __restrict__ new_v,     // [num_kv_heads * head_dim]
    float* __restrict__ k_cache,        // [max_seq * num_kv_heads * head_dim]
    float* __restrict__ v_cache,        // [max_seq * num_kv_heads * head_dim]
    const unsigned int pos,
    const unsigned int num_kv_heads,
    const unsigned int head_dim)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_kv_heads * head_dim;
    unsigned int cache_offset = pos * total;

    for (unsigned int i = tid; i < total; i += blockDim.x * gridDim.x) {
        k_cache[cache_offset + i] = new_k[i];
        v_cache[cache_offset + i] = new_v[i];
    }
}
"#;

pub const KV_ATTEND_SHADER: &str = r#"
// KV cache attend: single-token attention against the full KV cache.
// Q layout: [num_q_heads, head_dim]
// K layout: [seq_len, num_kv_heads, head_dim]
// V layout: same as K
// Output: attn_out[num_q_heads, head_dim]
//
// Each thread block handles one Q head. In GQA (grouped query attention),
// multiple Q heads share a KV head; we compute the QK dot product for each.
//
// For short sequences (≤1024 positions), we use shared memory for the
// softmax accumulation.  The long variant (kv_attend_long) should be used
// for spans > 1024.
extern "C" __global__ void kv_attend(
    const float* __restrict__ q,         // [num_q_heads * head_dim]
    const float* __restrict__ k_cache,   // [max_seq * num_kv_heads * head_dim]
    const float* __restrict__ v_cache,   // [max_seq * num_kv_heads * head_dim]
    float* __restrict__ attn_out,        // [num_q_heads * head_dim]
    const unsigned int seq_len,          // number of positions (pos + 1)
    const unsigned int head_dim,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float scale,                   // 1/sqrt(head_dim) or 1.0 for QK-normed
    const unsigned int window_size)      // sliding window size, 0 = full attention
{
    // Block-level scratch for per-warp reductions.  Declared once at kernel
    // scope — do NOT redeclare extern __shared__ inside nested blocks.
    extern __shared__ float smem[];
    __shared__ float s_max;
    __shared__ float s_sum;

    // Each block handles one Q head
    unsigned int q_head = blockIdx.x;
    if (q_head >= num_q_heads) return;

    // GQA: map Q head to KV head.  Clamp for non-uniform groups.
    unsigned int kv_head = (q_head * num_kv_heads) / num_q_heads;

    const float* q_row = q + q_head * head_dim;
    float* out_row = attn_out + q_head * head_dim;

    // Sliding window: only attend to the last `window_size` positions
    unsigned int start_pos = 0;
    if (window_size > 0 && seq_len > window_size) {
        start_pos = seq_len - window_size;
    }
    unsigned int attend_len = seq_len - start_pos;

    int num_warps = (blockDim.x + 31) / 32;

    // Step 1: Compute QK dot products and find max for numerical stability.
    // Each thread handles a stripe of positions.
    float max_score = -INFINITY;
    for (unsigned int t = threadIdx.x; t < attend_len; t += blockDim.x) {
        unsigned int pos = start_pos + t;
        const float* k_row = k_cache + (pos * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_row[d];
        }
        float score = dot * scale;
        if (score > max_score) max_score = score;
    }

    // Block-level max reduction: warp reduce, then thread 0 merges.
    float val = warp_reduce_max(max_score == -INFINITY ? -INFINITY : max_score);
    if (threadIdx.x % 32 == 0) {
        smem[threadIdx.x / 32] = val;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = smem[0];
        for (int i = 1; i < num_warps; i++) {
            v = fmax(v, smem[i]);
        }
        s_max = v;
    }
    __syncthreads();

    // Step 2: Compute exp(score - max) and accumulate (recompute scores inline).
    float sum_exp = 0.0f;
    for (unsigned int t = threadIdx.x; t < attend_len; t += blockDim.x) {
        unsigned int pos = start_pos + t;
        const float* k_row = k_cache + (pos * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_row[d];
        }
        sum_exp += expf(dot * scale - s_max);
    }

    // Block-level sum reduction: warp reduce, then thread 0 merges.
    sum_exp = warp_reduce_sum(sum_exp);
    if (threadIdx.x % 32 == 0) {
        smem[threadIdx.x / 32] = sum_exp;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = smem[0];
        for (int i = 1; i < num_warps; i++) {
            v += smem[i];
        }
        s_sum = v;
    }
    __syncthreads();

    float inv_sum = 1.0f / (s_sum + 1e-10f);

    // Step 3: Weighted sum of V (recompute scores for each position).
    // Each thread computes one element of the output.
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int t = 0; t < attend_len; t++) {
            unsigned int pos = start_pos + t;
            const float* k_row = k_cache + (pos * num_kv_heads + kv_head) * head_dim;
            const float* v_row = v_cache + (pos * num_kv_heads + kv_head) * head_dim;

            float dot = 0.0f;
            for (unsigned int dd = 0; dd < head_dim; dd++) {
                dot += q_row[dd] * k_row[dd];
            }
            acc += expf(dot * scale - s_max) * inv_sum * v_row[d];
        }
        out_row[d] = acc;
    }
}
"#;

pub const KV_APPEND_ATTEND_FUSED_SHADER: &str = r#"
// Fused KV append + attend: copies K/V into cache AND computes attention
// in a single kernel. More efficient for short sequences where the K/V
// data is still in registers/shared memory.
//
// Similar to kv_attend but first writes new_k/new_v into the cache at
// position (seq_len - 1), then attends over all positions.
extern "C" __global__ void kv_append_attend_fused(
    const float* __restrict__ q,         // [num_q_heads * head_dim]
    float* __restrict__ k_cache,         // [max_seq * num_kv_heads * head_dim]
    float* __restrict__ v_cache,         // [max_seq * num_kv_heads * head_dim]
    float* __restrict__ attn_out,        // [num_q_heads * head_dim]
    const unsigned int seq_len,
    const unsigned int head_dim,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float scale,
    const unsigned int window_size,
    const float* __restrict__ new_k,      // [num_kv_heads * head_dim]
    const float* __restrict__ new_v)     // [num_kv_heads * head_dim]
{
    extern __shared__ float smem[];
    __shared__ float s_max;
    __shared__ float s_sum;

    // First: append new K/V to cache
    unsigned int pos = seq_len - 1;
    unsigned int total_kv = num_kv_heads * head_dim;
    unsigned int cache_off = pos * total_kv;
    for (unsigned int i = threadIdx.x; i < total_kv; i += blockDim.x) {
        k_cache[cache_off + i] = new_k[i];
        v_cache[cache_off + i] = new_v[i];
    }
    __syncthreads();

    // Then: attend (same logic as kv_attend)
    unsigned int q_head = blockIdx.x;
    if (q_head >= num_q_heads) return;
    unsigned int kv_head = (q_head * num_kv_heads) / num_q_heads;
    const float* q_row = q + q_head * head_dim;
    float* out_row = attn_out + q_head * head_dim;

    unsigned int start_pos = 0;
    if (window_size > 0 && seq_len > window_size) {
        start_pos = seq_len - window_size;
    }
    unsigned int attend_len = seq_len - start_pos;
    int num_warps = (blockDim.x + 31) / 32;

    // Find max score
    float max_score = -INFINITY;
    for (unsigned int t = threadIdx.x; t < attend_len; t += blockDim.x) {
        unsigned int p = start_pos + t;
        const float* k_row = k_cache + (p * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) dot += q_row[d] * k_row[d];
        float s = dot * scale;
        if (s > max_score) max_score = s;
    }
    max_score = warp_reduce_max(max_score == -INFINITY ? -INFINITY : max_score);
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = max_score;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = smem[0];
        for (int i = 1; i < num_warps; i++) v = fmax(v, smem[i]);
        s_max = v;
    }
    __syncthreads();

    // Compute softmax sum
    float sum_exp = 0.0f;
    for (unsigned int t = threadIdx.x; t < attend_len; t += blockDim.x) {
        unsigned int p = start_pos + t;
        const float* k_row = k_cache + (p * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) dot += q_row[d] * k_row[d];
        sum_exp += expf(dot * scale - s_max);
    }
    sum_exp = warp_reduce_sum(sum_exp);
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = sum_exp;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = smem[0];
        for (int i = 1; i < num_warps; i++) v += smem[i];
        s_sum = v;
    }
    __syncthreads();
    float inv_sum = 1.0f / (s_sum + 1e-10f);

    // Weighted V sum
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int t = 0; t < attend_len; t++) {
            unsigned int p = start_pos + t;
            const float* k_row = k_cache + (p * num_kv_heads + kv_head) * head_dim;
            const float* v_row = v_cache + (p * num_kv_heads + kv_head) * head_dim;
            float dot = 0.0f;
            for (unsigned int dd = 0; dd < head_dim; dd++) dot += q_row[dd] * k_row[dd];
            acc += expf(dot * scale - s_max) * inv_sum * v_row[d];
        }
        out_row[d] = acc;
    }
}
"#;