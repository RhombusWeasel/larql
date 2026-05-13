//! Fused GEGLU activation + Q4_K down projection — CUDA port of
//! `metal/shaders/q4k_geglu_down.rs`.
//!
//! Two variants:
//! - **SiLU**: `silu(gate) * up` (Llama, Mistral, Qwen)
//! - **GELU-tanh**: `0.5 * gate * (1 + tanh(sqrt(2/π) * (gate + 0.044715*gate³))) * up`
//!   (Gemma, GPT-2, Phi)
//!
//! Each warp processes one output row. The warp walks all super-blocks
//! of the down-projection weight, reading gate/up activation values
//! on-the-fly and applying the activation function per element.

use crate::cuda::kernel::TiledKernel;

// ── SiLU + Q4_K down ──

pub const GEGLU_SILU_SHADER: &str = r#"
extern "C" __global__ void q4k_geglu_silu_down(
    const uint8_local_t* __restrict__ W_down,   // down weights [N, K] Q4_K
    const float* __restrict__ gate,              // gate output [K]
    const float* __restrict__ up,               // up output [K]
    float* __restrict__ out,                     // output [N]
    const unsigned int N,
    const unsigned int K)
{
    const unsigned int ROWS_PER_BLOCK = 8;
    const unsigned int Q4K_BLOCK_SIZE = 144;
    unsigned int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / 32);
    if (row >= N) return;

    unsigned int superblocks = K / 256;
    unsigned long long bytes_per_row = (unsigned long long)superblocks * Q4K_BLOCK_SIZE;
    const uint8_local_t* row_bytes = W_down + (unsigned long long)row * bytes_per_row;
    unsigned int lane = threadIdx.x % 32;
    float acc = 0.0f;

    for (unsigned int sb = lane; sb < superblocks; sb += 32) {
        const uint8_local_t* block = row_bytes + (unsigned long long)sb * Q4K_BLOCK_SIZE;

        uint16_local_t d_bits    = (uint16_local_t)block[0] | ((uint16_local_t)block[1] << 8);
        uint16_local_t dmin_bits = (uint16_local_t)block[2] | ((uint16_local_t)block[3] << 8);
        float d    = decode_f16(d_bits);
        float dmin = decode_f16(dmin_bits);

        const uint8_local_t* sb_bytes = block + 4;
        unsigned int scales[8];
        unsigned int mins[8];
        for (int jj = 0; jj < 4; jj++) {
            scales[jj] = (unsigned int)(sb_bytes[jj])   & 0x3Fu;
            mins[jj]   = (unsigned int)(sb_bytes[jj+4]) & 0x3Fu;
        }
        for (int jj = 4; jj < 8; jj++) {
            scales[jj] = ((unsigned int)(sb_bytes[jj+4]) & 0x0Fu) | (((unsigned int)(sb_bytes[jj-4]) >> 6) << 4);
            mins[jj]   = ((unsigned int)(sb_bytes[jj+4]) >> 4)    | (((unsigned int)(sb_bytes[jj])   >> 6) << 4);
        }

        const uint8_local_t* qs = block + 16;
        unsigned int x_base = sb * 256;
        float sb_acc = 0.0f;
        for (unsigned int g = 0; g < 4; g++) {
            unsigned int sub_lo = 2 * g;
            unsigned int sub_hi = 2 * g + 1;
            float sc_lo = d * (float)scales[sub_lo];
            float sc_hi = d * (float)scales[sub_hi];
            float mn_lo = dmin * (float)mins[sub_lo];
            float mn_hi = dmin * (float)mins[sub_hi];
            float dot_lo = 0.0f, sum_lo = 0.0f;
            float dot_hi = 0.0f, sum_hi = 0.0f;
            for (unsigned int l = 0; l < 32; l++) {
                uint8_local_t byte_val = qs[g * 32 + l];
                float nib_lo = (float)(byte_val & 0x0Fu);
                float nib_hi = (float)((byte_val >> 4u) & 0x0Fu);
                unsigned int idx_lo = x_base + sub_lo * 32 + l;
                unsigned int idx_hi = x_base + sub_hi * 32 + l;
                float g_lo = gate[idx_lo];
                float act_lo = (g_lo / (1.0f + expf(-g_lo))) * up[idx_lo];
                float g_hi = gate[idx_hi];
                float act_hi = (g_hi / (1.0f + expf(-g_hi))) * up[idx_hi];
                dot_lo = fmaf(nib_lo, act_lo, dot_lo);
                sum_lo += act_lo;
                dot_hi = fmaf(nib_hi, act_hi, dot_hi);
                sum_hi += act_hi;
            }
            sb_acc += sc_lo * dot_lo - mn_lo * sum_lo;
            sb_acc += sc_hi * dot_hi - mn_hi * sum_hi;
        }
        acc += sb_acc;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

// ── GELU-tanh + Q4_K down ──

pub const GEGLU_GELU_TANH_SHADER: &str = r#"
extern "C" __global__ void q4k_geglu_gelu_tanh_down(
    const uint8_local_t* __restrict__ W_down,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const unsigned int N,
    const unsigned int K)
{
    const unsigned int ROWS_PER_BLOCK = 8;
    const unsigned int Q4K_BLOCK_SIZE = 144;
    unsigned int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / 32);
    if (row >= N) return;

    unsigned int superblocks = K / 256;
    unsigned long long bytes_per_row = (unsigned long long)superblocks * Q4K_BLOCK_SIZE;
    const uint8_local_t* row_bytes = W_down + (unsigned long long)row * bytes_per_row;
    unsigned int lane = threadIdx.x % 32;
    float acc = 0.0f;

    float c = 0.7978845608f; // sqrt(2/pi)
    for (unsigned int sb = lane; sb < superblocks; sb += 32) {
        const uint8_local_t* block = row_bytes + (unsigned long long)sb * Q4K_BLOCK_SIZE;

        uint16_local_t d_bits    = (uint16_local_t)block[0] | ((uint16_local_t)block[1] << 8);
        uint16_local_t dmin_bits = (uint16_local_t)block[2] | ((uint16_local_t)block[3] << 8);
        float d    = decode_f16(d_bits);
        float dmin = decode_f16(dmin_bits);

        const uint8_local_t* sb_bytes = block + 4;
        unsigned int scales[8];
        unsigned int mins[8];
        for (int jj = 0; jj < 4; jj++) {
            scales[jj] = (unsigned int)(sb_bytes[jj])   & 0x3Fu;
            mins[jj]   = (unsigned int)(sb_bytes[jj+4]) & 0x3Fu;
        }
        for (int jj = 4; jj < 8; jj++) {
            scales[jj] = ((unsigned int)(sb_bytes[jj+4]) & 0x0Fu) | (((unsigned int)(sb_bytes[jj-4]) >> 6) << 4);
            mins[jj]   = ((unsigned int)(sb_bytes[jj+4]) >> 4)    | (((unsigned int)(sb_bytes[jj])   >> 6) << 4);
        }

        const uint8_local_t* qs = block + 16;
        unsigned int x_base = sb * 256;
        float sb_acc = 0.0f;
        for (unsigned int g = 0; g < 4; g++) {
            unsigned int sub_lo = 2 * g;
            unsigned int sub_hi = 2 * g + 1;
            float sc_lo = d * (float)scales[sub_lo];
            float sc_hi = d * (float)scales[sub_hi];
            float mn_lo = dmin * (float)mins[sub_lo];
            float mn_hi = dmin * (float)mins[sub_hi];
            float dot_lo = 0.0f, sum_lo = 0.0f;
            float dot_hi = 0.0f, sum_hi = 0.0f;
            for (unsigned int l = 0; l < 32; l++) {
                uint8_local_t byte_val = qs[g * 32 + l];
                float nib_lo = (float)(byte_val & 0x0Fu);
                float nib_hi = (float)((byte_val >> 4u) & 0x0Fu);
                unsigned int idx_lo = x_base + sub_lo * 32 + l;
                unsigned int idx_hi = x_base + sub_hi * 32 + l;
                float g_lo = gate[idx_lo];
                float t_lo = tanhf(c * (g_lo + 0.044715f * g_lo * g_lo * g_lo));
                float act_lo = (0.5f * g_lo * (1.0f + t_lo)) * up[idx_lo];
                float g_hi = gate[idx_hi];
                float t_hi = tanhf(c * (g_hi + 0.044715f * g_hi * g_hi * g_hi));
                float act_hi = (0.5f * g_hi * (1.0f + t_hi)) * up[idx_hi];
                dot_lo = fmaf(nib_lo, act_lo, dot_lo);
                sum_lo += act_lo;
                dot_hi = fmaf(nib_hi, act_hi, dot_hi);
                sum_hi += act_hi;
            }
            sb_acc += sc_lo * dot_lo - mn_lo * sum_lo;
            sb_acc += sc_hi * dot_hi - mn_hi * sum_hi;
        }
        acc += sb_acc;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

pub struct Q4KGegeluSiluDownKernel;
impl TiledKernel for Q4KGegeluSiluDownKernel {
    const KERNEL_NAME: &'static str = "q4k_geglu_silu_down";
    const ROWS_PER_BLOCK: u64 = 8;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub struct Q4KGegeluGeluTanhDownKernel;
impl TiledKernel for Q4KGegeluGeluTanhDownKernel {
    const KERNEL_NAME: &'static str = "q4k_geglu_gelu_tanh_down";
    const ROWS_PER_BLOCK: u64 = 8;
    const THREADS_PER_BLOCK: u64 = 256;
}