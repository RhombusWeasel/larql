//! Fused QK-norm + RoPE — single dispatch for Gemma 3/4 attention.
//!
//! Replaces consecutive `qk_norm` + `rope_at_pos_batched_qk` dispatches.
//! Each thread block handles one (Q or K) head.
//! Phase 1: cooperative RMS reduction, Phase 2: normalize, Phase 3: RoPE.

use crate::cuda::kernel::TiledKernel;

pub struct QkNormRopeFusedKernel;
impl TiledKernel for QkNormRopeFusedKernel {
    const KERNEL_NAME: &'static str = "qk_norm_rope_fused";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const SHADER: &str = r#"
// Fused QK-norm + RoPE: Normalizes Q/K heads then applies RoPE in-place.
// Grid: (num_q + num_kv) blocks, one per head.
// Each block: Phase 1 = cooperative RMS reduction, Phase 2 = normalize,
//             Phase 3 = in-place RoPE rotation.
extern "C" __global__ void qk_norm_rope_fused(
    float* __restrict__ Q,             // [num_q * head_dim] — in-place
    float* __restrict__ K,             // [num_kv * head_dim] — in-place
    const float* __restrict__ q_weight, // [head_dim]
    const float* __restrict__ k_weight, // [head_dim]
    const unsigned int head_dim,
    const unsigned int num_q,
    const float eps,
    const float offset,
    const float rope_base,
    const unsigned int pos,
    const unsigned int rotary_dim)
{
    unsigned int h_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int block_sz = blockDim.x;

    bool is_q = (h_idx < num_q);
    unsigned int local_head = is_q ? h_idx : (h_idx - num_q);
    float* buf = is_q ? Q : K;
    const float* weight = is_q ? q_weight : k_weight;
    unsigned int base = local_head * head_dim;

    // Phase 1: cooperative RMS reduction
    float partial = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += block_sz) {
        float v = buf[base + i];
        partial += v * v;
    }
    partial = block_reduce_sum(partial);
    float rms = sqrtf(partial / (float)head_dim + eps);
    float inv_rms = 1.0f / rms;

    // Phase 2: normalize
    for (unsigned int i = tid; i < head_dim; i += block_sz) {
        buf[base + i] = (buf[base + i] * inv_rms) * (offset + weight[i]);
    }
    __syncthreads();

    // Phase 3: in-place RoPE rotation
    unsigned int rdim = (rotary_dim == 0) ? head_dim : min(rotary_dim, head_dim);
    unsigned int hdim = rdim / 2;
    for (unsigned int d = tid; d < hdim; d += block_sz) {
        float freq = 1.0f / powf(rope_base, (float)(2 * d) / (float)rdim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float re = buf[base + d];
        float im = buf[base + d + hdim];
        buf[base + d]        = re * cos_a - im * sin_a;
        buf[base + d + hdim] = re * sin_a + im * cos_a;
    }
}
"#;