//! Q4_K matrix-matrix multiply (GEMM) for prefill.
//!
//! Port of `metal/shaders/q4k_matmul.rs`: processes `M` input positions
//! in a single dispatch, amortising the Q4_K dequant cost across M.
//!
//! Layout matches the Metal kernel:
//!   - W: `[N, K]` Q4_K row-major (144 bytes per 256 cols)
//!   - X: `[M, K]` f32 row-major
//!   - out: `[M, N]` f32 row-major
//!
//! Dispatch: `tg_id.y` covers N in chunks of ROWS_PER_BLOCK,
//! `tg_id.x` covers M in chunks of COLS_PER_BLOCK=4.

pub const SHADER: &str = r#"
extern "C" __global__ void q4k_matmul(
    const uchar*  W4K,    // Q4_K weight matrix [N, K/256 * 144]
    const float*  X,      // Input matrix [M, K]
    float*        out,    // Output matrix [M, N]
    unsigned int  N,      // Output rows (W rows)
    unsigned int  K,      // Hidden / inner dim
    unsigned int  M       // Input positions (seq_len for prefill)
) {
    // ── Constants ──
    const uint ROWS_PER_BLOCK = 4u;
    const uint COLS_PER_BLOCK = 4u;
    const uint Q4K_BLOCK_SIZE = 144u;
    const uint Q4K_BLOCK_ELEMS = 256u;

    uint row_idx = blockIdx.y * ROWS_PER_BLOCK + (threadIdx.x / 32u);
    if (row_idx >= N) return;

    uint m_base = blockIdx.x * COLS_PER_BLOCK;
    if (m_base >= M) return;
    uint cols_in_block = min(COLS_PER_BLOCK, M - m_base);

    uint lane = threadIdx.x % 32u;
    uint sg_id = threadIdx.x / 32u;

    // ── Q4_K dequant setup (mirrors q4k_matvec) ──
    const uint superblocks = K / Q4K_BLOCK_ELEMS;
    const uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE;
    const uchar* row_w = W4K + row_idx * bytes_per_row;

    // 2-way inter-superblock interleaving
    const uint ix  = lane & 1u;
    const uint tid = lane >> 1u;
    const uint j   = tid >> 1u;
    const uint sh  = tid & 1u;
    const bool  hi = (j & 1u) != 0u;
    const uint  group = j >> 1u;

    // Per-position accumulators
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (uint sb = ix; sb < superblocks; sb += 2u) {
        const uchar* block = row_w + sb * Q4K_BLOCK_SIZE;

        // Read super-block scale/min
        ushort d_bits    = (ushort)block[0] | ((ushort)block[1] << 8u);
        ushort dmin_bits = (ushort)block[2] | ((ushort)block[3] << 8u);
        float d    = __half2float(__ushort_as_half(d_bits));
        float dmin = __half2float(__ushort_as_half(dmin_bits));

        // Unpack 6-bit scale/min
        const uchar* sb_bytes = block + 4u;
        uint sc, mn;
        if (j < 4u) {
            sc = (uint)(sb_bytes[j]) & 0x3Fu;
            mn = (uint)(sb_bytes[j + 4u]) & 0x3Fu;
        } else {
            sc = ((uint)(sb_bytes[j + 4u]) & 0x0Fu) | (((uint)(sb_bytes[j - 4u]) >> 6u) << 4u);
            mn = ((uint)(sb_bytes[j + 4u]) >> 4u) | (((uint)(sb_bytes[j]) >> 6u) << 4u);
        }
        float scale = d * (float)sc;
        float mmin  = dmin * (float)mn;

        // Dequantise 16 nibbles once per super-block (amortised across M positions)
        const uchar* qs = block + 16u + group * 32u + sh * 16u;
        float nibs[16];
        #pragma unroll
        for (uint l = 0u; l < 16u; l++) {
            uchar byte_val = qs[l];
            nibs[l] = hi ? (float)((byte_val >> 4u) & 0x0Fu) : (float)(byte_val & 0x0Fu);
        }

        const uint x_sb_off = sb * 256u + j * 32u + sh * 16u;

        // Process up to COLS_PER_BLOCK positions
        #pragma unroll
        for (uint m = 0u; m < COLS_PER_BLOCK; m++) {
            uint pos = (m < cols_in_block) ? (m_base + m) : m_base;
            uint x_off = pos * K + x_sb_off;

            float sumy = 0.0f;
            float dot_acc = 0.0f;
            #pragma unroll
            for (uint l = 0u; l < 16u; l++) {
                float xl = X[x_off + l];
                sumy += xl;
                dot_acc += nibs[l] * xl;
            }
            float contrib = scale * dot_acc - mmin * sumy;
            switch(m) {
                case 0u: acc0 += contrib; break;
                case 1u: acc1 += contrib; break;
                case 2u: acc2 += contrib; break;
                case 3u: acc3 += contrib; break;
            }
        }
    }

    // Warp reduction for each accumulator
    acc0 = warp_reduce_sum(acc0);
    acc1 = warp_reduce_sum(acc1);
    acc2 = warp_reduce_sum(acc2);
    acc3 = warp_reduce_sum(acc3);

    if (lane == 0u) {
        if (0u < cols_in_block) out[(m_base + 0u) * N + row_idx] = acc0;
        if (1u < cols_in_block) out[(m_base + 1u) * N + row_idx] = acc1;
        if (2u < cols_in_block) out[(m_base + 2u) * N + row_idx] = acc2;
        if (3u < cols_in_block) out[(m_base + 3u) * N + row_idx] = acc3;
    }
}
"#;

pub const ROWS_PER_BLOCK: u64 = 4;
pub const COLS_PER_BLOCK: u64 = 4;
pub const THREADS_PER_BLOCK: u64 = 128;

use crate::cuda::kernel::TiledKernel;

pub struct Kernel;
impl TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4k_matmul";
    const ROWS_PER_BLOCK: u64 = ROWS_PER_BLOCK;
    const THREADS_PER_BLOCK: u64 = THREADS_PER_BLOCK;
}