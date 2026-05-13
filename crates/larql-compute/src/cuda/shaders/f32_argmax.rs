//! f32 argmax and top-K partial reduction — port of `metal/shaders/f32_gemv.rs`.
//!
//! Two-phase argmax/top-K:
//! 1. GPU: per-block partial top-K (each block emits K_TOPK=8 candidates)
//! 2. CPU: merge partial results into final answer
//!
//! **f32_argmax_partial**: Each block finds its local max and writes one
//! (val, idx) pair. CPU merges `num_blocks` candidates.
//!
//! **f32_topk_partial**: Each block finds its local top-K_TOPK and writes
//! K_TOPK (val, idx) pairs. CPU merges `num_blocks × K_TOPK` candidates.

use crate::cuda::kernel::TiledKernel;

/// Block size for argmax/topk partial reductions. Mirrors Metal's
/// `PARTIAL_TG_SZ = 256`.
pub const PARTIAL_BLOCK_SIZE: u64 = 256;

/// Number of top-K candidates per block. Mirrors Metal's `K_TOPK = 8`.
pub const K_TOPK: usize = 8;

/// Max warps per block (256 / 32 = 8).
pub const MAX_WARPS_PER_BLOCK: usize = PARTIAL_BLOCK_SIZE as usize / 32;

// ── f32_argmax_partial ──

pub const ARGMAX_SHADER: &str = r#"
extern "C" __global__ void f32_argmax_partial(
    const float* __restrict__ scores,    // [N]
    float* __restrict__ out_val,          // [num_blocks]
    unsigned int* __restrict__ out_idx,   // [num_blocks]
    const unsigned int N)
{
    // Thread-level max
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local_val = (i < N) ? scores[i] : -1e38f;
    unsigned int local_idx = (i < N) ? i : 0xFFFFFFFFu;

    // Warp-level reduction: find max, broadcast to all lanes
    float warp_max = local_val;
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max = fmax(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
    }
    // Broadcast warp_max from lane 0 to all lanes
    warp_max = __shfl_sync(0xFFFFFFFF, warp_max, 0);

    // Among lanes holding the max, take the smallest index (stable argmax)
    unsigned int warp_idx = (local_val >= warp_max) ? local_idx : 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_idx = min(warp_idx, __shfl_down_sync(0xFFFFFFFF, warp_idx, offset));
    }
    // Broadcast warp_idx from lane 0 to all lanes
    warp_idx = __shfl_sync(0xFFFFFFFF, warp_idx, 0);

    // Block-level reduction via shared memory
    __shared__ float shared_val[8];
    __shared__ unsigned int shared_idx[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_val[warp_id] = warp_max;
        shared_idx[warp_id] = warp_idx;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float best_val = shared_val[0];
        unsigned int best_idx = shared_idx[0];
        for (int s = 1; s < num_warps; s++) {
            if (shared_val[s] > best_val || (shared_val[s] == best_val && shared_idx[s] < best_idx)) {
                best_val = shared_val[s];
                best_idx = shared_idx[s];
            }
        }
        out_val[blockIdx.x] = best_val;
        out_idx[blockIdx.x] = best_idx;
    }
}
"#;

// ── f32_topk_partial ──

/// Build the topk shader source, substituting Rust constants for the
/// NVRTC-incompatible `constant uint` declarations.
pub fn topk_shader_source() -> String {
    TOPK_SHADER_BODY
        .replace("K_TOPK_PLACEHOLDER", &K_TOPK.to_string())
        .replace("PARTIAL_BLOCK_SIZE_PLACEHOLDER", &PARTIAL_BLOCK_SIZE.to_string())
}

const TOPK_SHADER_BODY: &str = r#"
extern "C" __global__ void f32_topk_partial(
    const float* __restrict__ scores,    // [N]
    float* __restrict__ out_val,         // [num_blocks * K_TOPK]
    unsigned int* __restrict__ out_idx,  // [num_blocks * K_TOPK]
    const unsigned int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float tg_v[PARTIAL_BLOCK_SIZE_PLACEHOLDER];
    __shared__ unsigned int tg_i[PARTIAL_BLOCK_SIZE_PLACEHOLDER];
    tg_v[threadIdx.x] = (i < N) ? scores[i] : -1e38f;
    tg_i[threadIdx.x] = (i < N) ? i : 0xFFFFFFFFu;
    __syncthreads();

    float winner_v;
    unsigned int winner_i;

    for (unsigned int k = 0; k < K_TOPK_PLACEHOLDER; k++) {
        // Warp-level max
        float v = tg_v[threadIdx.x];
        float warp_max = v;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_max = fmax(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
        }
        warp_max = __shfl_sync(0xFFFFFFFF, warp_max, 0);

        unsigned int cand = (v >= warp_max) ? tg_i[threadIdx.x] : 0xFFFFFFFFu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            cand = min(cand, __shfl_down_sync(0xFFFFFFFF, cand, offset));
        }
        cand = __shfl_sync(0xFFFFFFFF, cand, 0);

        int lane = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;
        __shared__ float sg_v[8];
        __shared__ unsigned int sg_i[8];
        if (lane == 0) { sg_v[warp_id] = warp_max; sg_i[warp_id] = cand; }
        __syncthreads();

        if (threadIdx.x == 0) {
            int num_warps = (blockDim.x + 31) / 32;
            float best_v = sg_v[0];
            unsigned int best_i = sg_i[0];
            for (int s = 1; s < num_warps; s++) {
                if (sg_v[s] > best_v || (sg_v[s] == best_v && sg_i[s] < best_i)) {
                    best_v = sg_v[s];
                    best_i = sg_i[s];
                }
            }
            out_val[blockIdx.x * K_TOPK_PLACEHOLDER + k] = best_v;
            out_idx[blockIdx.x * K_TOPK_PLACEHOLDER + k] = best_i;
            winner_v = best_v;
            winner_i = best_i;
        }
        __syncthreads();

        // Mask the winning thread's value
        if (tg_i[threadIdx.x] == winner_i) {
            tg_v[threadIdx.x] = -1e38f;
        }
        __syncthreads();
    }
}
"#;

/// Argmax kernel marker. Not a tiled kernel in the traditional sense — uses
/// fixed block size and variable grid.
pub struct ArgmaxKernel;
impl TiledKernel for ArgmaxKernel {
    const KERNEL_NAME: &'static str = "f32_argmax_partial";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = PARTIAL_BLOCK_SIZE;
}

/// TopK kernel marker. Same dispatch geometry as ArgmaxKernel.
pub struct TopKKernel;
impl TiledKernel for TopKKernel {
    const KERNEL_NAME: &'static str = "f32_topk_partial";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = PARTIAL_BLOCK_SIZE;
}