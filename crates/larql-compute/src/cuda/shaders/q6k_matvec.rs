//! Q6_K matrix-vector multiply — CUDA port of `metal/shaders/q6k_matvec.rs`.
//!
//! Q6_K super-block: 256 values in 210 bytes.
//! Two variants:
//! - **Primary** (4 warp, 128 threads, 4 rows/block)
//! - **8-warp** (8 rows/block, 256 threads)
//!
//! Uses deferred scaling (4× fewer scale multiplications), X preloading,
//! and 2-way inter-superblock interleaving.

use crate::cuda::kernel::TiledKernel;

// ── Primary variant (4 warp, 128 threads) ──

pub const SHADER: &str = r#"
extern "C" __global__ void q6k_matvec(
    const uint8_local_t* __restrict__ W6K,
    const float* __restrict__ X,
    float* __restrict__ out,
    const unsigned int N,
    const unsigned int K)
{
    const unsigned int ROWS_PER_BLOCK = 4;
    const unsigned int Q6K_BLOCK_SIZE = 210;
    unsigned int row_idx = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / 32);
    if (row_idx >= N) return;

    const unsigned int superblocks = K / 256u;
    const unsigned long long bytes_per_row = (unsigned long long)superblocks * Q6K_BLOCK_SIZE;
    const uint8_local_t* row = W6K + (unsigned long long)row_idx * bytes_per_row;

    const unsigned int lane = threadIdx.x % 32;
    const unsigned int ix  = lane & 1u;
    const unsigned int tid = lane >> 1u;
    const unsigned int base    = tid << 2u;
    const unsigned int sc_base = tid >> 2u;

    float acc = 0.0f;

    for (unsigned int i = ix; i < superblocks; i += 2u) {
        const uint8_local_t* block = row + (unsigned long long)i * Q6K_BLOCK_SIZE;
        const uint8_local_t* ql   = block;
        const uint8_local_t* qh   = block + 128u;
        const int8_local_t* sc = (const int8_local_t*)(block + 192u);
        uint16_local_t d_bits = (uint16_local_t)block[208] | ((uint16_local_t)block[209] << 8u);
        float d = decode_f16(d_bits);

        const unsigned int xb = i * 256u + base;
        float xl[16];
        xl[ 0] = X[xb      ]; xl[ 1] = X[xb +  1u];
        xl[ 2] = X[xb +  2u]; xl[ 3] = X[xb +  3u];
        xl[ 4] = X[xb + 64u]; xl[ 5] = X[xb + 65u];
        xl[ 6] = X[xb + 66u]; xl[ 7] = X[xb + 67u];
        xl[ 8] = X[xb +128u]; xl[ 9] = X[xb +129u];
        xl[10] = X[xb +130u]; xl[11] = X[xb +131u];
        xl[12] = X[xb +192u]; xl[13] = X[xb +193u];
        xl[14] = X[xb +194u]; xl[15] = X[xb +195u];

        // Pass 0: elements base+0..3 (scale group sc_base+0)
        {
            const unsigned int b = base;
            uint8_local_t la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * (float)sc[sc_base + 0u];
            acc += _sc * (
                (float)((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 0] +
                (float)((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 1] +
                (float)((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[ 2] +
                (float)((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 3]);
        }

        // Pass 1: elements base+64..67 (scale group sc_base+4)
        {
            const unsigned int b = base + 64u;
            uint8_local_t la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * (float)sc[sc_base + 4u];
            acc += _sc * (
                (float)((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 4] +
                (float)((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 5] +
                (float)((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[ 6] +
                (float)((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 7]);
        }

        // Pass 2: elements base+128..131 (scale group sc_base+8)
        {
            const unsigned int b = base + 128u;
            uint8_local_t la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * (float)sc[sc_base + 8u];
            acc += _sc * (
                (float)((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 8] +
                (float)((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 9] +
                (float)((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[10] +
                (float)((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[11]);
        }

        // Pass 3: elements base+192..195 (scale group sc_base+12)
        {
            const unsigned int b = base + 192u;
            uint8_local_t la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * (float)sc[sc_base + 12u];
            acc += _sc * (
                (float)((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[12] +
                (float)((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[13] +
                (float)((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[14] +
                (float)((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[15]);
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0u) out[row_idx] = acc;
}
"#;

pub struct Q6KMatvecKernel;
impl TiledKernel for Q6KMatvecKernel {
    const KERNEL_NAME: &'static str = "q6k_matvec";
    const ROWS_PER_BLOCK: u64 = 4;
    const THREADS_PER_BLOCK: u64 = 128;
}

// ── 8-warp variant (8 rows/block, 256 threads) ──

pub const SHADER_8SG: &str = r#"
extern "C" __global__ void q6k_matvec_8sg(
    const uint8_local_t* __restrict__ W6K,
    const float* __restrict__ X,
    float* __restrict__ out,
    const unsigned int N,
    const unsigned int K)
{
    const unsigned int ROWS_PER_BLOCK = 8;
    const unsigned int Q6K_BLOCK_SIZE = 210;
    unsigned int row_idx = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / 32);
    if (row_idx >= N) return;

    const unsigned int superblocks = K / 256u;
    const unsigned long long bytes_per_row = (unsigned long long)superblocks * Q6K_BLOCK_SIZE;
    const uint8_local_t* row = W6K + (unsigned long long)row_idx * bytes_per_row;

    const unsigned int lane = threadIdx.x % 32;
    const unsigned int ix  = lane & 1u;
    const unsigned int tid = lane >> 1u;
    const unsigned int base    = tid << 2u;
    const unsigned int sc_base = tid >> 2u;

    float acc = 0.0f;

    for (unsigned int i = ix; i < superblocks; i += 2u) {
        const uint8_local_t* block = row + (unsigned long long)i * Q6K_BLOCK_SIZE;
        const uint8_local_t* ql   = block;
        const uint8_local_t* qh   = block + 128u;
        const int8_local_t* sc = (const int8_local_t*)(block + 192u);
        uint16_local_t d_bits = (uint16_local_t)block[208] | ((uint16_local_t)block[209] << 8u);
        float d = decode_f16(d_bits);

        const unsigned int xb = i * 256u + base;
        float xl[16];
        xl[ 0] = X[xb      ]; xl[ 1] = X[xb +  1u];
        xl[ 2] = X[xb +  2u]; xl[ 3] = X[xb +  3u];
        xl[ 4] = X[xb + 64u]; xl[ 5] = X[xb + 65u];
        xl[ 6] = X[xb + 66u]; xl[ 7] = X[xb + 67u];
        xl[ 8] = X[xb +128u]; xl[ 9] = X[xb +129u];
        xl[10] = X[xb +130u]; xl[11] = X[xb +131u];
        xl[12] = X[xb +192u]; xl[13] = X[xb +193u];
        xl[14] = X[xb +194u]; xl[15] = X[xb +195u];

        {
            const unsigned int b = base;
            uint8_local_t la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * (float)sc[sc_base + 0u];
            acc += _sc * (
                (float)((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 0] +
                (float)((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 1] +
                (float)((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[ 2] +
                (float)((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 3]);
        }
        {
            const unsigned int b = base + 64u;
            uint8_local_t la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * (float)sc[sc_base + 4u];
            acc += _sc * (
                (float)((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 4] +
                (float)((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 5] +
                (float)((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[ 6] +
                (float)((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 7]);
        }
        {
            const unsigned int b = base + 128u;
            uint8_local_t la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * (float)sc[sc_base + 8u];
            acc += _sc * (
                (float)((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 8] +
                (float)((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 9] +
                (float)((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[10] +
                (float)((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[11]);
        }
        {
            const unsigned int b = base + 192u;
            uint8_local_t la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * (float)sc[sc_base + 12u];
            acc += _sc * (
                (float)((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[12] +
                (float)((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[13] +
                (float)((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[14] +
                (float)((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[15]);
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0u) out[row_idx] = acc;
}
"#;

pub struct Q6KMatvec8sgKernel;
impl TiledKernel for Q6KMatvec8sgKernel {
    const KERNEL_NAME: &'static str = "q6k_matvec_8sg";
    const ROWS_PER_BLOCK: u64 = 8;
    const THREADS_PER_BLOCK: u64 = 256;
}