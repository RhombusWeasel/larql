//! Fused mixed-quant QKV projection — Q4_K for Q/K rows, Q6_K for V rows.
//!
//! **Q/K branch: 2-way inter-superblock interleaving (same as q4k_matvec).**
//!
//! The previous Q/K branch used `for (sb = lane; sb < superblocks; sb += 32)` —
//! for K=2560 (10 superblocks) only lanes 0..9 were active (31% utilisation).
//! New: `ix = lane & 1` ensures all 32 lanes are busy and adjacent lanes read
//! from different 144-byte superblock regions simultaneously.
//!
//! Lane decomposition for Q/K branch:
//!   ix  = lane & 1      — 0/1: even/odd superblock group
//!   tid = lane >> 1     — 0..15
//!   j   = tid >> 1      — 0..7: sub-block index
//!   sh  = tid & 1       — 0/1: first/last 16 elements
//!   X preloaded into xl[16] before weight reads.
//!
//! **V branch: original scalar loop (known correct, Q6_K all-lanes-per-superblock).**
//! The Q6_K inter-superblock optimisation is tracked separately — the ix/tid
//! decomposition for Q6_K (which uses ip/il to split upper/lower 128 elements)
//! conflicts with the Q4_K decomposition (j/sh) in the same kernel scope.

pub const SHADER: &str = r#"
constant uint Q4K_Q6K_ROWS_PER_TG  = 4;
constant uint Q4K_BLOCK_SIZE_MIXED  = 144;
constant uint Q6K_BLOCK_SIZE_MIXED  = 210;

kernel void q4k_q6k_qkv_proj(
    device const uchar*  Wq     [[buffer(0)]],
    device const uchar*  Wk     [[buffer(1)]],
    device const uchar*  Wv     [[buffer(2)]],
    device const float*  X      [[buffer(3)]],
    device float*        Q_out  [[buffer(4)]],
    device float*        K_out  [[buffer(5)]],
    device float*        V_out  [[buffer(6)]],
    constant uint&       q_rows [[buffer(7)]],
    constant uint&       k_rows [[buffer(8)]],
    constant uint&       v_rows [[buffer(9)]],
    constant uint&       K      [[buffer(10)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    uint total_rows = q_rows + k_rows + v_rows;
    uint global_row = tg_id * Q4K_Q6K_ROWS_PER_TG + sg_id;
    if (global_row >= total_rows) return;

    const uint superblocks = K / 256u;
    float acc = 0.0f;

    if (global_row < q_rows + k_rows) {
        // ── Q/K rows: Q4_K — 2-way inter-superblock interleaving ──
        uint local_row;
        device const uchar* W;
        device float* out_buf;
        if (global_row < q_rows) {
            W = Wq; out_buf = Q_out; local_row = global_row;
        } else {
            W = Wk; out_buf = K_out; local_row = global_row - q_rows;
        }

        const uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE_MIXED;
        device const uchar* row = W + local_row * bytes_per_row;

        const uint ix  = lane & 1u;
        const uint tid = lane >> 1u;
        const uint j   = tid >> 1u;
        const uint sh  = tid & 1u;
        const bool hi    = (j & 1u) != 0u;
        const uint group = j >> 1u;

        for (uint sb = ix; sb < superblocks; sb += 2u) {
            device const uchar* block = row + sb * Q4K_BLOCK_SIZE_MIXED;
            ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8u);
            ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8u);
            float d    = decode_f16_metal(d_bits);
            float dmin = decode_f16_metal(dmin_bits);

            device const uchar* sb_bytes = block + 4u;
            uint sc, mn;
            if (j < 4u) {
                sc = uint(sb_bytes[j])      & 0x3Fu;
                mn = uint(sb_bytes[j + 4u]) & 0x3Fu;
            } else {
                sc = (uint(sb_bytes[j + 4u]) & 0x0Fu) | ((uint(sb_bytes[j - 4u]) >> 6u) << 4u);
                mn = (uint(sb_bytes[j + 4u]) >> 4u)    | ((uint(sb_bytes[j])      >> 6u) << 4u);
            }
            float scale = d * float(sc);
            float mmin  = dmin * float(mn);

            const uint x_base = sb * 256u + j * 32u + sh * 16u;
            float xl[16];
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 16u; l++) { xl[l] = X[x_base + l]; }

            device const uchar* qs = block + 16u + group * 32u + sh * 16u;
            float dot_acc = 0.0f, sum_acc = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 16u; l++) {
                uchar byte = qs[l];
                float nib = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);
                dot_acc = fma(nib, xl[l], dot_acc);
                sum_acc += xl[l];
            }
            acc += scale * dot_acc - mmin * sum_acc;
        }

        acc = simd_sum(acc);
        if (lane == 0u) out_buf[local_row] = acc;

    } else {
        // ── V rows: Q6_K — scalar all-lanes-per-superblock (original, correct) ──
        // TODO: apply inter-superblock treatment once the ix/tid clash with the
        // Q4_K branch above is resolved (the Q6_K branch needs ip/il which spans
        // elements 0..127 and 128..255 separately, incompatible with j/sh here).
        uint local_row = global_row - q_rows - k_rows;
        const uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE_MIXED;
        device const uchar* row = Wv + local_row * bytes_per_row;

        for (uint sb = 0u; sb < superblocks; sb++) {
            device const uchar* block = row + sb * Q6K_BLOCK_SIZE_MIXED;
            device const uchar* ql    = block;
            device const uchar* qh    = block + 128u;
            device const char*  sc    = (device const char*)(block + 192u);
            ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
            float d = decode_f16_metal(d_bits);

            const uint x_base = sb * 256u;
            for (uint pass = 0u; pass < 8u; pass++) {
                uint i = pass * 32u + lane;
                uchar lo_byte = ql[i >> 1u];
                uint lo4 = (i & 1u) ? ((lo_byte >> 4u) & 0x0Fu) : (lo_byte & 0x0Fu);
                uchar hi_byte = qh[i >> 2u];
                uint hi2 = (hi_byte >> ((i & 3u) << 1u)) & 0x03u;
                int raw = int(lo4 | (hi2 << 4u)) - 32;
                float val = d * float(sc[i >> 4u]) * float(raw);
                acc = fma(val, X[x_base + i], acc);
            }
        }

        acc = simd_sum(acc);
        if (lane == 0u) V_out[local_row] = acc;
    }
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;

/// MSL source for the fused RMS-norm + QKV projection variant.
/// Takes raw `H` (un-normalised hidden state) + `norm_weight` instead of
/// pre-normalised `X`, computing the norm cooperatively within each TG.
/// Eliminates the separate `rms_norm` dispatch (saves 34 dispatches/token).
pub const NORMED_SHADER: &str = r#"

kernel void q4k_q6k_qkv_proj_normed(
    device const uchar*  Wq      [[buffer(0)]],
    device const uchar*  Wk      [[buffer(1)]],
    device const uchar*  Wv      [[buffer(2)]],
    device const float*  H       [[buffer(3)]],   // raw hidden (un-normed)
    device const float*  norm_w  [[buffer(4)]],   // RMS norm weight
    device float*        Q_out   [[buffer(5)]],
    device float*        K_out   [[buffer(6)]],
    device float*        V_out   [[buffer(7)]],
    constant uint&       q_rows  [[buffer(8)]],
    constant uint&       k_rows  [[buffer(9)]],
    constant uint&       v_rows  [[buffer(10)]],
    constant uint&       K       [[buffer(11)]],
    constant float&      eps     [[buffer(12)]],
    constant float&      offset  [[buffer(13)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]],
    uint tid    [[thread_index_in_threadgroup]])
{
    // ── Phase 1: cooperative RMS norm (all 128 threads in TG) ──
    // All threads participate regardless of row validity so barriers are uniform.
    const uint tg_sz = Q4K_Q6K_ROWS_PER_TG * 32u;  // = 128
    float partial = 0.0f;
    for (uint i = tid; i < K; i += tg_sz) {
        float h = H[i];
        partial += h * h;
    }
    float sg_sum = simd_sum(partial);
    threadgroup float tg_p[4];
    if (lane == 0u) tg_p[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_sq = tg_p[0] + tg_p[1] + tg_p[2] + tg_p[3];
    float rms = 1.0f / sqrt(sum_sq / float(K) + eps);

    // ── Phase 2: same Q4_K / Q6_K matvec as q4k_q6k_qkv_proj ──
    // X[i] replaced by H[i] * rms * (offset + norm_w[i]).
    // H and norm_w are 10 KB each — L1-cached after first few TG reads.
    uint total_rows = q_rows + k_rows + v_rows;
    uint global_row = tg_id * Q4K_Q6K_ROWS_PER_TG + sg_id;
    if (global_row >= total_rows) return;

    const uint superblocks = K / 256u;
    float acc = 0.0f;

    if (global_row < q_rows + k_rows) {
        uint local_row;
        device const uchar* W;
        device float* out_buf;
        if (global_row < q_rows) {
            W = Wq; out_buf = Q_out; local_row = global_row;
        } else {
            W = Wk; out_buf = K_out; local_row = global_row - q_rows;
        }
        const uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE_MIXED;
        device const uchar* row = W + local_row * bytes_per_row;

        const uint ix  = lane & 1u;
        const uint ptid = lane >> 1u;
        const uint j   = ptid >> 1u;
        const uint sh  = ptid & 1u;
        const bool hi    = (j & 1u) != 0u;
        const uint group = j >> 1u;

        for (uint sb = ix; sb < superblocks; sb += 2u) {
            device const uchar* block = row + sb * Q4K_BLOCK_SIZE_MIXED;
            ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8u);
            ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8u);
            float d    = decode_f16_metal(d_bits);
            float dmin = decode_f16_metal(dmin_bits);

            device const uchar* sb_bytes = block + 4u;
            uint sc, mn;
            if (j < 4u) {
                sc = uint(sb_bytes[j])      & 0x3Fu;
                mn = uint(sb_bytes[j + 4u]) & 0x3Fu;
            } else {
                sc = (uint(sb_bytes[j + 4u]) & 0x0Fu) | ((uint(sb_bytes[j - 4u]) >> 6u) << 4u);
                mn = (uint(sb_bytes[j + 4u]) >> 4u)    | ((uint(sb_bytes[j])      >> 6u) << 4u);
            }
            float scale = d * float(sc);
            float mmin  = dmin * float(mn);

            const uint x_base = sb * 256u + j * 32u + sh * 16u;
            float xl[16];
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 16u; l++) {
                float h = H[x_base + l];
                xl[l] = h * rms * (offset + norm_w[x_base + l]);
            }

            device const uchar* qs = block + 16u + group * 32u + sh * 16u;
            float dot_acc = 0.0f, sum_acc = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (uint l = 0u; l < 16u; l++) {
                uchar byte = qs[l];
                float nib = hi ? float((byte >> 4u) & 0x0Fu) : float(byte & 0x0Fu);
                dot_acc = fma(nib, xl[l], dot_acc);
                sum_acc += xl[l];
            }
            acc += scale * dot_acc - mmin * sum_acc;
        }

        acc = simd_sum(acc);
        if (lane == 0u) out_buf[local_row] = acc;

    } else {
        uint local_row = global_row - q_rows - k_rows;
        const uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE_MIXED;
        device const uchar* row = Wv + local_row * bytes_per_row;

        for (uint sb = 0u; sb < superblocks; sb++) {
            device const uchar* block = row + sb * Q6K_BLOCK_SIZE_MIXED;
            device const uchar* ql    = block;
            device const uchar* qh    = block + 128u;
            device const char*  sc    = (device const char*)(block + 192u);
            ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
            float d = decode_f16_metal(d_bits);

            const uint x_base = sb * 256u;
            for (uint pass = 0u; pass < 8u; pass++) {
                uint i = pass * 32u + lane;
                uchar lo_byte = ql[i >> 1u];
                uint lo4 = (i & 1u) ? ((lo_byte >> 4u) & 0x0Fu) : (lo_byte & 0x0Fu);
                uchar hi_byte = qh[i >> 2u];
                uint hi2 = (hi_byte >> ((i & 3u) << 1u)) & 0x03u;
                int raw = int(lo4 | (hi2 << 4u)) - 32;
                float val = d * float(sc[i >> 4u]) * float(raw);
                // Inline normalization: H[i] * rms * (offset + norm_w[i])
                float xi = H[x_base + i] * rms * (offset + norm_w[x_base + i]);
                acc = fma(val, xi, acc);
            }
        }

        acc = simd_sum(acc);
        if (lane == 0u) V_out[local_row] = acc;
    }
}
"#;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::metal::kernel::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4k_q6k_qkv_proj";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}

/// Marker for the fused-norm variant (takes raw H + norm_weight).
pub struct NormedKernel;
impl crate::metal::kernel::TiledKernel for NormedKernel {
    const KERNEL_NAME: &'static str = "q4k_q6k_qkv_proj_normed";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
