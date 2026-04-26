//! f32 gemv — matrix-vector multiply for the LM head.
//!
//! Computes `out[N] = W[N, K] · x[K]` where `W` is row-major f32.
//!
//! One simdgroup per row. Each of the 32 lanes reads `K/32` strided
//! elements, accumulates a partial dot product, then `simd_sum` reduces
//! into a single output.
//!
//! Sized for the Gemma 3/4 tied LM head: N ~ 262 K, K = 2560–5120. The
//! simdgroup-per-row pattern gets ~4× over the 32×32 tiled sgemm at M=1
//! (which wastes 31/32 of its threads and leaves accumulation precision
//! different enough to shift argmax on noisy logits).

pub const SHADER: &str = r#"
constant uint F32GEMV_SG_PER_TG = 8;   // simdgroups per threadgroup
constant uint F32GEMV_ROWS_PER_TG = F32GEMV_SG_PER_TG; // one row per simdgroup

kernel void f32_gemv(
    device const float* W   [[buffer(0)]],   // [N, K] row-major
    device const float* X   [[buffer(1)]],   // [K]
    device float*       out [[buffer(2)]],   // [N]
    constant uint&      N   [[buffer(3)]],
    constant uint&      K   [[buffer(4)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * F32GEMV_ROWS_PER_TG + sg_id;
    if (row >= N) return;

    device const float* w_row = W + row * K;

    float acc = 0.0f;
    // Stride-32 over K; four unrolled per-lane accumulators avoid
    // serialising on a single latency-bound chain.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint k = lane;
    for (; k + 3 * 32 < K; k += 4 * 32) {
        a0 = fma(w_row[k         ], X[k         ], a0);
        a1 = fma(w_row[k + 32    ], X[k + 32    ], a1);
        a2 = fma(w_row[k + 64    ], X[k + 64    ], a2);
        a3 = fma(w_row[k + 96    ], X[k + 96    ], a3);
    }
    acc = (a0 + a1) + (a2 + a3);
    for (; k < K; k += 32) acc = fma(w_row[k], X[k], acc);

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256; // 8 simdgroups × 32 lanes

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::metal::kernel::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "f32_gemv";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}

/// Metal source for the two-phase f32 argmax shader.
/// Phase 1 (`f32_argmax_partial`): each TG of 256 threads finds its
/// local max → writes (val, idx) to a partial result array.
/// The caller reduces the partial results on CPU (1024 candidates).
/// Phase 2 is CPU-side (1024 × 8 bytes = 8 KB, ~1 µs).
pub const ARGMAX_SHADER: &str = r#"
// Phase 1: per-TG argmax. Grid: ceil(N/256) TGs × 256 threads.
// Writes one (float, uint) pair per TG to out_val / out_idx.
kernel void f32_argmax_partial(
    device const float* scores   [[buffer(0)]],
    device float*       out_val  [[buffer(1)]],
    device uint*        out_idx  [[buffer(2)]],
    constant uint&      N        [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{
    uint i = tg_id * tg_sz + tid;
    float local_val = (i < N) ? scores[i] : -1e38f;
    uint  local_idx = (i < N) ? i : 0u;

    // Simd reduction: find max value in simdgroup, then find index.
    float sg_max = simd_max(local_val);
    // Among lanes holding the max, take the smallest index (stable argmax).
    uint sg_idx = (local_val >= sg_max) ? local_idx : ~0u;
    sg_idx = simd_min(sg_idx);

    // Threadgroup reduction across simdgroups.
    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];
    if (lane == 0u) { tg_v[sg_id] = sg_max; tg_i[sg_id] = sg_idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint n_sg = (tg_sz + 31u) / 32u;
        float best_val = tg_v[0]; uint best_idx = tg_i[0];
        for (uint s = 1u; s < n_sg; s++) {
            if (tg_v[s] > best_val || (tg_v[s] == best_val && tg_i[s] < best_idx)) {
                best_val = tg_v[s]; best_idx = tg_i[s];
            }
        }
        out_val[tg_id] = best_val;
        out_idx[tg_id] = best_idx;
    }
}
"#;

pub struct ArgmaxKernel;
impl crate::metal::kernel::ShaderKernel for ArgmaxKernel {
    const KERNEL_NAME: &'static str = "f32_argmax_partial";
}
