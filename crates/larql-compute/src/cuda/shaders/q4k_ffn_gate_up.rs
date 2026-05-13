//! Fused Q4_K gate+up projection — CUDA port of `metal/shaders/q4k_ffn_gate_up.rs`.
//!
//! Two matvecs sharing the same input vector in one kernel launch.
//! Grid: `2 * ceil(N / ROWS_PER_BLOCK)` blocks — first half for gate,
//! second half for up.

use crate::cuda::kernel::TiledKernel;

pub const SHADER: &str = r#"
extern "C" __global__ void q4k_ffn_gate_up(
    const uint8_local_t* __restrict__ Wg,    // gate weights [N, K] Q4_K
    const uint8_local_t* __restrict__ Wu,    // up weights [N, K] Q4_K
    const float* __restrict__ X,              // input [K]
    float* __restrict__ G_out,               // gate output [N]
    float* __restrict__ U_out,               // up output [N]
    const unsigned int N,
    const unsigned int K)
{
    const unsigned int ROWS_PER_BLOCK = 4;
    const unsigned int Q4K_GU_BLOCK_SIZE = 144;

    unsigned int tgs_per_mat = (N + ROWS_PER_BLOCK - 1u) / ROWS_PER_BLOCK;
    bool is_up  = (blockIdx.x >= tgs_per_mat);
    unsigned int mat_tg = is_up ? (blockIdx.x - tgs_per_mat) : blockIdx.x;

    unsigned int row_idx = mat_tg * ROWS_PER_BLOCK + (threadIdx.x / 32);
    if (row_idx >= N) return;

    const uint8_local_t* W      = is_up ? Wu : Wg;
    float* out_buf = is_up ? U_out : G_out;

    const unsigned int superblocks = K / 256u;
    const unsigned long long bytes_per_row = (unsigned long long)superblocks * Q4K_GU_BLOCK_SIZE;
    const uint8_local_t* row_w = W + (unsigned long long)row_idx * bytes_per_row;

    const unsigned int lane = threadIdx.x % 32;
    const unsigned int ix  = lane & 1u;
    const unsigned int tid = lane >> 1u;
    const unsigned int j   = tid >> 1u;
    const unsigned int sh  = tid & 1u;
    const bool hi    = (j & 1u) != 0;
    const unsigned int group = j >> 1u;

    float acc = 0.0f;

    for (unsigned int sb = ix; sb < superblocks; sb += 2u) {
        const uint8_local_t* block = row_w + (unsigned long long)sb * Q4K_GU_BLOCK_SIZE;
        uint16_local_t d_bits    = (uint16_local_t)block[0] | ((uint16_local_t)block[1] << 8u);
        uint16_local_t dmin_bits = (uint16_local_t)block[2] | ((uint16_local_t)block[3] << 8u);
        float d    = decode_f16(d_bits);
        float dmin = decode_f16(dmin_bits);

        const uint8_local_t* sb_bytes = block + 4u;
        unsigned int sc, mn;
        if (j < 4u) {
            sc = (unsigned int)(sb_bytes[j])      & 0x3Fu;
            mn = (unsigned int)(sb_bytes[j + 4u]) & 0x3Fu;
        } else {
            sc = ((unsigned int)(sb_bytes[j + 4u]) & 0x0Fu) | (((unsigned int)(sb_bytes[j - 4u]) >> 6u) << 4u);
            mn = ((unsigned int)(sb_bytes[j + 4u]) >> 4u)    | (((unsigned int)(sb_bytes[j])      >> 6u) << 4u);
        }
        float scale = d * (float)sc;
        float mmin  = dmin * (float)mn;

        const unsigned int x_base = sb * 256u + j * 32u + sh * 16u;
        float xl[16];
        #pragma unroll
        for (unsigned int l = 0u; l < 16u; l++) { xl[l] = X[x_base + l]; }

        const uint8_local_t* qs = block + 16u + group * 32u + sh * 16u;

        float sumy = 0.0f;
        #pragma unroll
        for (unsigned int l = 0u; l < 16u; l++) { sumy += xl[l]; }

        float dot_acc = 0.0f;
        #pragma unroll
        for (unsigned int l = 0u; l < 16u; l++) {
            uint8_local_t byte_val = qs[l];
            float nib = hi ? (float)((byte_val >> 4u) & 0x0Fu) : (float)(byte_val & 0x0Fu);
            dot_acc = fmaf(nib, xl[l], dot_acc);
        }
        acc += scale * dot_acc - mmin * sumy;
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0u) out_buf[row_idx] = acc;
}
"#;

pub struct Q4KFfnGateUpKernel;
impl TiledKernel for Q4KFfnGateUpKernel {
    const KERNEL_NAME: &'static str = "q4k_ffn_gate_up";
    const ROWS_PER_BLOCK: u64 = 4;
    const THREADS_PER_BLOCK: u64 = 128;
}