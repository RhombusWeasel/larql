//! Causal self-attention: QK^T softmax V for small seq_len.
//! One thread per (head_dim, query_position). Designed for seq ≤ 64.

use crate::cuda::kernel::TiledKernel;

pub struct CausalAttnKernel;
impl TiledKernel for CausalAttnKernel {
    const KERNEL_NAME: &'static str = "causal_attention";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const SHADER: &str = r#"
// Causal self-attention: out[q, d] = Σ_k softmax(QK^T/sqrt(d))[q,k] * V[k, d]
// Grid: (head_dim, seq_len) — one thread per (d, q) pair.
extern "C" __global__ void causal_attention(
    const float* __restrict__ Q,        // [seq_len, head_dim]
    const float* __restrict__ K,        // [seq_len, head_dim]
    const float* __restrict__ V,        // [seq_len, head_dim]
    float* __restrict__ out,            // [seq_len, head_dim]
    const unsigned int seq_len,
    const unsigned int head_dim,
    const float scale)
{
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (d >= head_dim || q >= seq_len) return;

    // Find max score for numerical stability
    float max_score = -1e30f;
    for (unsigned int k = 0; k <= q; k++) {
        float score = 0.0f;
        for (unsigned int i = 0; i < head_dim; i++) {
            score += Q[q * head_dim + i] * K[k * head_dim + i];
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    // Softmax numerator + weighted V sum
    float sum_exp = 0.0f;
    float weighted_v = 0.0f;
    for (unsigned int k = 0; k <= q; k++) {
        float score = 0.0f;
        for (unsigned int i = 0; i < head_dim; i++) {
            score += Q[q * head_dim + i] * K[k * head_dim + i];
        }
        score *= scale;
        float w = expf(score - max_score);
        sum_exp += w;
        weighted_v += w * V[k * head_dim + d];
    }
    out[q * head_dim + d] = weighted_v / sum_exp;
}
"#;