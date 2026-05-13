//! Q4_K matrix-vector multiply — CUDA port of `metal/shaders/q4k_matvec.rs`.
//!
//! GGUF 144-byte block layout. Three variants:
//!
//! - **Primary** (4 warp, 128 threads): default for most matvec sites.
//! - **8-warp** (256 threads): higher occupancy for bandwidth-bound kernels.
//! - **Stride-32** (256 threads): reduction tree matches f16_gemv, used for
//!   the LM head when the production kernel's block-adaptive lane split
//!   drifts enough vs CPU to flip top-1 on close-call tokens.
//!
//! All variants use 2-way inter-superblock interleaving (`ix = lane & 1`).

use crate::cuda::kernel::TiledKernel;

// ── Primary variant (4 warp, 128 threads, 4 rows/block) ──

pub const SHADER: &str = r#"
extern "C" __global__ void q4k_matvec(
    const uint8_local_t* __restrict__ W4K,   // [N * superblocks * 144]
    const float* __restrict__ X,              // [K]
    float* __restrict__ out,                  // [N]
    const unsigned int N,
    const unsigned int K)
{
    const unsigned int ROWS_PER_BLOCK = 4;
    const unsigned int Q4K_BLOCK_SIZE = 144;
    unsigned int row_idx = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / 32);
    if (row_idx >= N) return;

    const unsigned int superblocks = K / 256;
    const unsigned long long bytes_per_row = (unsigned long long)superblocks * Q4K_BLOCK_SIZE;
    const uint8_local_t* row_w = W4K + (unsigned long long)row_idx * bytes_per_row;

    // 2-way inter-superblock interleaving
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int ix  = lane & 1u;
    const unsigned int tid = lane >> 1u;
    const unsigned int j   = tid >> 1u;
    const unsigned int sh  = tid & 1u;
    const bool hi    = (j & 1u) != 0;
    const unsigned int group = j >> 1u;

    float acc = 0.0f;

    for (unsigned int sb = ix; sb < superblocks; sb += 2u) {
        const uint8_local_t* block = row_w + (unsigned long long)sb * Q4K_BLOCK_SIZE;
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
    if (lane == 0u) out[row_idx] = acc;
}
"#;

pub struct Q4KMatvecKernel;
impl TiledKernel for Q4KMatvecKernel {
    const KERNEL_NAME: &'static str = "q4k_matvec";
    const ROWS_PER_BLOCK: u64 = 4;
    const THREADS_PER_BLOCK: u64 = 128;
}

// ── 8-warp variant (8 rows/block, 256 threads) ──

pub const SHADER_8SG: &str = r#"
extern "C" __global__ void q4k_matvec_8sg(
    const uint8_local_t* __restrict__ W4K,
    const float* __restrict__ X,
    float* __restrict__ out,
    const unsigned int N,
    const unsigned int K)
{
    const unsigned int ROWS_PER_BLOCK = 8;
    const unsigned int Q4K_BLOCK_SIZE = 144;
    unsigned int row_idx = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / 32);
    if (row_idx >= N) return;

    const unsigned int superblocks = K / 256;
    const unsigned long long bytes_per_row = (unsigned long long)superblocks * Q4K_BLOCK_SIZE;
    const uint8_local_t* row_w = W4K + (unsigned long long)row_idx * bytes_per_row;

    const unsigned int lane = threadIdx.x % 32;
    const unsigned int ix  = lane & 1u;
    const unsigned int tid = lane >> 1u;
    const unsigned int j   = tid >> 1u;
    const unsigned int sh  = tid & 1u;
    const bool hi    = (j & 1u) != 0;
    const unsigned int group = j >> 1u;

    float acc = 0.0f;

    for (unsigned int sb = ix; sb < superblocks; sb += 2u) {
        const uint8_local_t* block = row_w + (unsigned long long)sb * Q4K_BLOCK_SIZE;
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
    if (lane == 0u) out[row_idx] = acc;
}
"#;

pub struct Q4KMatvec8sgKernel;
impl TiledKernel for Q4KMatvec8sgKernel {
    const KERNEL_NAME: &'static str = "q4k_matvec_8sg";
    const ROWS_PER_BLOCK: u64 = 8;
    const THREADS_PER_BLOCK: u64 = 256;
}

// ── Stride-32 variant (8 rows/block, 256 threads) ──
// Per-element decomposition: lane k accumulates Σ_{i : i%32==k} dequant(W,i)*X[i]
// Same reduction tree as f16_gemv => stable for LM head argmax.

pub const SHADER_S32: &str = r#"
extern "C" __global__ void q4k_matvec_stride32(
    const uint8_local_t* __restrict__ W4K,
    const float* __restrict__ X,
    float* __restrict__ out,
    const unsigned int N,
    const unsigned int K)
{
    const unsigned int ROWS_PER_BLOCK = 8;
    const unsigned int Q4K_BLOCK_SIZE = 144;
    unsigned int row_idx = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / 32);
    if (row_idx >= N) return;

    const unsigned int superblocks = K / 256;
    const unsigned long long bytes_per_row = (unsigned long long)superblocks * Q4K_BLOCK_SIZE;
    const uint8_local_t* row_w = W4K + (unsigned long long)row_idx * bytes_per_row;

    unsigned int lane = threadIdx.x % 32;
    float acc = 0.0f;

    const unsigned int sh    = lane >> 4u;
    const unsigned int inner = lane & 15u;

    for (unsigned int sb = 0u; sb < superblocks; sb++) {
        const uint8_local_t* block = row_w + (unsigned long long)sb * Q4K_BLOCK_SIZE;

        uint16_local_t d_bits    = (uint16_local_t)block[0] | ((uint16_local_t)block[1] << 8u);
        uint16_local_t dmin_bits = (uint16_local_t)block[2] | ((uint16_local_t)block[3] << 8u);
        float d    = decode_f16(d_bits);
        float dmin = decode_f16(dmin_bits);
        const uint8_local_t* sb_bytes = block + 4u;

        for (unsigned int sub = 0u; sub < 8u; sub++) {
            unsigned int sc, mn;
            if (sub < 4u) {
                sc = (unsigned int)(sb_bytes[sub])      & 0x3Fu;
                mn = (unsigned int)(sb_bytes[sub + 4u]) & 0x3Fu;
            } else {
                sc = ((unsigned int)(sb_bytes[sub + 4u]) & 0x0Fu) | (((unsigned int)(sb_bytes[sub - 4u]) >> 6u) << 4u);
                mn = ((unsigned int)(sb_bytes[sub + 4u]) >> 4u)    | (((unsigned int)(sb_bytes[sub])      >> 6u) << 4u);
            }
            float scale = d * (float)sc;
            float mmin  = dmin * (float)mn;

            unsigned int group_idx = sub >> 1u;
            bool hi = (sub & 1u) != 0u;
            uint8_local_t byte_val = block[16u + group_idx * 32u + sh * 16u + inner];
            float nib = hi ? (float)((byte_val >> 4u) & 0x0Fu) : (float)(byte_val & 0x0Fu);

            unsigned int x_idx = sb * 256u + sub * 32u + lane;
            float w = scale * nib - mmin;
            acc += w * X[x_idx];
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0u) out[row_idx] = acc;
}
"#;

pub struct Q4KMatvecStride32Kernel;
impl TiledKernel for Q4KMatvecStride32Kernel {
    const KERNEL_NAME: &'static str = "q4k_matvec_stride32";
    const ROWS_PER_BLOCK: u64 = 8;
    const THREADS_PER_BLOCK: u64 = 256;
}