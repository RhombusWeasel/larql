//! Rotary position encoding (RoPE) — CUDA kernels.
//!
//! Five variants:
//! - `rope_at_pos`: single vector at a specific position
//! - `rope_apply`: batched positions (prefill)
//! - `rope_at_pos_batched`: all heads for a single position
//! - `rope_at_pos_batched_qk`: fused Q+K heads at a single position
//! - `rope_at_pos_q` / `rope_at_pos_k`: per-head-type variants

use crate::cuda::kernel::TiledKernel;

pub struct RopeAtPosKernel;
impl TiledKernel for RopeAtPosKernel {
    const KERNEL_NAME: &'static str = "rope_at_pos";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub struct RopeApplyKernel;
impl TiledKernel for RopeApplyKernel {
    const KERNEL_NAME: &'static str = "rope_apply";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub struct RopeBatchedKernel;
impl TiledKernel for RopeBatchedKernel {
    const KERNEL_NAME: &'static str = "rope_at_pos_batched";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub struct RopeBatchedQkKernel;
impl TiledKernel for RopeBatchedQkKernel {
    const KERNEL_NAME: &'static str = "rope_at_pos_batched_qk";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub struct RopeQKernel;
impl TiledKernel for RopeQKernel {
    const KERNEL_NAME: &'static str = "rope_at_pos_q";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub struct RopeKKernel;
impl TiledKernel for RopeKKernel {
    const KERNEL_NAME: &'static str = "rope_at_pos_k";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const ROPE_AT_POS_SHADER: &str = r#"
extern "C" __global__ void rope_at_pos(
    float* __restrict__ x,              // [dim] — modified in-place
    const unsigned int dim,
    const float base,
    const unsigned int pos,
    const unsigned int rotary_dim)
{
    unsigned int rdim = (rotary_dim == 0) ? dim : min(rotary_dim, dim);
    unsigned int hdim = rdim / 2;
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hdim) return;

    float freq = 1.0f / powf(base, (2.0f * d) / (float)rdim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    float re = x[d];
    float im = x[d + hdim];
    x[d]        = re * cos_val - im * sin_val;
    x[d + hdim] = re * sin_val + im * cos_val;
}
"#;

pub const ROPE_APPLY_SHADER: &str = r#"
// Batched RoPE: apply to [seq_len, dim] matrix in-place.
// Grid: (rotary_dim/2, seq_len), one thread per (pos, pair).
extern "C" __global__ void rope_apply(
    float* __restrict__ x,              // [seq_len * dim] — modified in-place
    const unsigned int dim,
    const float base,
    const unsigned int rotary_dim,
    const unsigned int seq_len)
{
    unsigned int rdim = (rotary_dim == 0) ? dim : min(rotary_dim, dim);
    unsigned int hdim = rdim / 2;
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pos = blockIdx.y;
    if (d >= hdim || pos >= seq_len) return;

    float freq = 1.0f / powf(base, (2.0f * d) / (float)rdim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    unsigned int idx_re = pos * dim + d;
    unsigned int idx_im = idx_re + hdim;

    float re = x[idx_re];
    float im = x[idx_im];
    x[idx_re] = re * cos_val - im * sin_val;
    x[idx_im] = re * sin_val + im * cos_val;
}
"#;

pub const ROPE_BATCHED_SHADER: &str = r#"
// Batched RoPE: all heads at a single position.
// x = [num_heads * head_dim] contiguous.
// Grid: (roty_dim/2, num_heads), one thread per (pair, head).
extern "C" __global__ void rope_at_pos_batched(
    float* __restrict__ x,              // [num_heads * head_dim] — in-place
    const unsigned int head_dim,
    const float base,
    const unsigned int pos,
    const unsigned int rotary_dim,
    const unsigned int num_heads)
{
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int h = blockIdx.y;
    unsigned int rdim = (rotary_dim == 0) ? head_dim : min(rotary_dim, head_dim);
    unsigned int hdim = rdim / 2;
    if (d >= hdim || h >= num_heads) return;

    float freq = 1.0f / powf(base, (2.0f * d) / (float)rdim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    unsigned int base_idx = h * head_dim;
    float re = x[base_idx + d];
    float im = x[base_idx + d + hdim];
    x[base_idx + d]        = re * cos_val - im * sin_val;
    x[base_idx + d + hdim] = re * sin_val + im * cos_val;
}
"#;

pub const ROPE_BATCHED_QK_SHADER: &str = r#"
// Fused Q+K batched RoPE: Q heads then K heads in one dispatch.
// Grid: (roty_dim/2, num_q + num_kv), one thread per (pair, head).
extern "C" __global__ void rope_at_pos_batched_qk(
    float* __restrict__ Q,              // [num_q_heads * head_dim]
    float* __restrict__ K,              // [num_kv_heads * head_dim]
    const unsigned int head_dim,
    const float rope_base,
    const unsigned int pos,
    const unsigned int rotary_dim,
    const unsigned int num_q)
{
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int h = blockIdx.y;

    unsigned int rdim = (rotary_dim == 0) ? head_dim : min(rotary_dim, head_dim);
    unsigned int hdim = rdim / 2;
    if (d >= hdim) return;

    bool is_q = (h < num_q);
    unsigned int local_h = is_q ? h : (h - num_q);
    float* x = is_q ? Q : K;
    unsigned int base_idx = local_h * head_dim;

    float freq = 1.0f / powf(rope_base, (2.0f * d) / (float)rdim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    float re = x[base_idx + d];
    float im = x[base_idx + d + hdim];
    x[base_idx + d]        = re * cos_val - im * sin_val;
    x[base_idx + d + hdim] = re * sin_val + im * cos_val;
}
"#;

/// RoPE at a specific position — applied to Q heads only.
pub const ROPE_Q_SHADER: &str = r#"
// Rotary position encoding for Q heads at position `pos`.
// q_out layout: [num_q_heads * head_dim], row-major (head, dim).
// Each thread handles one pair (dim_idx, dim_idx+1) within one head.
extern "C" __global__ void rope_at_pos_q(
    float* __restrict__ q_out,        // [num_q_heads * head_dim]
    const unsigned int head_dim,
    const float rope_base,
    const unsigned int pos,
    const unsigned int rotary_dim,
    const unsigned int num_q_heads)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int rope_pairs = rotary_dim / 2;
    unsigned int total_pairs = num_q_heads * rope_pairs;

    if (tid >= total_pairs) return;

    unsigned int head = tid / rope_pairs;
    unsigned int pair = tid % rope_pairs;

    // Compute frequency
    float freq = 1.0f / powf(rope_base, (2.0f * pair) / (float)rotary_dim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    unsigned int d0 = head * head_dim + pair;
    unsigned int d1 = d0 + rope_pairs;

    float x0 = q_out[d0];
    float x1 = q_out[d1];
    q_out[d0] = x0 * cos_val - x1 * sin_val;
    q_out[d1] = x0 * sin_val + x1 * cos_val;
}
"#;

/// RoPE at a specific position — applied to K heads only.
pub const ROPE_K_SHADER: &str = r#"
extern "C" __global__ void rope_at_pos_k(
    float* __restrict__ k_out,        // [num_kv_heads * head_dim]
    const unsigned int head_dim,
    const float rope_base,
    const unsigned int pos,
    const unsigned int rotary_dim,
    const unsigned int num_kv_heads)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int rope_pairs = rotary_dim / 2;
    unsigned int total_pairs = num_kv_heads * rope_pairs;

    if (tid >= total_pairs) return;

    unsigned int head = tid / rope_pairs;
    unsigned int pair = tid % rope_pairs;

    float freq = 1.0f / powf(rope_base, (2.0f * pair) / (float)rotary_dim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    unsigned int d0 = head * head_dim + pair;
    unsigned int d1 = d0 + rope_pairs;

    float x0 = k_out[d0];
    float x1 = k_out[d1];
    k_out[d0] = x0 * cos_val - x1 * sin_val;
    k_out[d1] = x0 * sin_val + x1 * cos_val;
}
"#;