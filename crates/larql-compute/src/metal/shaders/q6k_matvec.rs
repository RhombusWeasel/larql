//! Q6_K matrix-vector multiply — used by Ollama for V projection and FFN down.
//!
//! Q6_K super-block layout (256 values = 210 bytes):
//!   [0..127]    128 bytes: lo4 — lower 4 bits of each value (2 per byte)
//!   [128..191]   64 bytes: hi2 — upper 2 bits (4 per byte)
//!   [192..207]   16 bytes: int8 scales (one per 16-value sub-block)
//!   [208..209]    2 bytes: f16 super-block scale d
//!
//! Dequantize element i: d * scales[i/16] * ((lo4[i] | (hi2[i] << 4)) - 32)
//!
//! **Parallelism strategy (all-lanes-per-superblock):**
//!
//! All 32 lanes cooperate on EVERY superblock. Each lane handles 8 elements
//! per superblock (256/32 = 8), iterating over 8 passes with stride 32.
//! No shared memory: K=10240 (40 KB f32) fits in GPU L2 cache; X reads are
//! effectively free once cached on the first TG read.
//!
//! ROWS_PER_TG = 4 (one row per simdgroup, 4 simdgroups per TG).
//! Down proj has only 2560 rows: at 8 rows/TG that's 320 TGs — too few to
//! saturate the memory bus (gate+up has 2560 TGs). Halving to 4 rows/TG
//! doubles TG count to 640, increasing concurrent memory pressure.

pub const SHADER: &str = r#"
constant uint Q6K_ROWS_PER_TG = 8;
constant uint Q6K_BLOCK_SIZE  = 210;

kernel void q6k_matvec(
    device const uchar*  W6K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id * Q6K_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    uint superblocks   = K / 256u;
    uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE;
    device const uchar* row = W6K + row_idx * bytes_per_row;

    float acc = 0.0f;

    for (uint sb = 0u; sb < superblocks; sb++) {
        device const uchar* block = row + sb * Q6K_BLOCK_SIZE;
        device const uchar* ql   = block;
        device const uchar* qh   = block + 128u;
        ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
        float d = decode_f16_metal(d_bits);

        // Preload 16 scaled int8 scales into registers — eliminates one
        // device read per element in the inner loops below.
        device const char* sc_dev = (device const char*)(block + 192u);
        float sc_f[16];
        for (uint s = 0u; s < 16u; s++) { sc_f[s] = d * float(sc_dev[s]); }

        uint x_base = sb * 256u;

        // 4-element batching: each lane processes 4 consecutive elements
        // per pass so that hi2 shifts are compile-time constants (0,2,4,6)
        // instead of the runtime `(i & 3) << 1` from the scalar loop.
        // 2 passes × 32 lanes × 4 elements = 256 elements/superblock.
        // Each group of 4 shares one hi2 byte and one scale entry, so
        // byte-read count drops from 4 per 4 elements to 3 (2 lo4 + 1 hi2).
        // All 4 elements also share the same scale (base is aligned to 4,
        // so floor(base/16) == floor((base+3)/16) always holds).
        for (uint pass = 0u; pass < 2u; pass++) {
            uint base = pass * 128u + lane * 4u;

            float sc = sc_f[base >> 4u];

            // hi2: one byte → 4 values via compile-time-constant shifts.
            uchar hi = qh[base >> 2u];
            uint hi2_0 =  hi        & 0x03u;
            uint hi2_1 = (hi >> 2u) & 0x03u;
            uint hi2_2 = (hi >> 4u) & 0x03u;
            uint hi2_3 = (hi >> 6u) & 0x03u;

            // lo4: two bytes → 4 nibbles.
            uint lo_idx = base >> 1u;
            uchar lo_a = ql[lo_idx];
            uchar lo_b = ql[lo_idx + 1u];
            uint lo4_0 =  lo_a        & 0x0Fu;
            uint lo4_1 = (lo_a >> 4u) & 0x0Fu;
            uint lo4_2 =  lo_b        & 0x0Fu;
            uint lo4_3 = (lo_b >> 4u) & 0x0Fu;

            acc = fma(sc * float(int(lo4_0 | (hi2_0 << 4u)) - 32), X[x_base + base    ], acc);
            acc = fma(sc * float(int(lo4_1 | (hi2_1 << 4u)) - 32), X[x_base + base + 1u], acc);
            acc = fma(sc * float(int(lo4_2 | (hi2_2 << 4u)) - 32), X[x_base + base + 2u], acc);
            acc = fma(sc * float(int(lo4_3 | (hi2_3 << 4u)) - 32), X[x_base + base + 3u], acc);
        }
    }

    acc = simd_sum(acc);
    if (lane == 0u) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::metal::kernel::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q6k_matvec";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
