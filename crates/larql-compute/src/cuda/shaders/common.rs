//! Shared CUDA kernel utilities — f16 decode, constants, warp reductions.
//!
//! This header is included in every kernel compilation unit.
//! Mirrors `metal/shaders/common.rs`.

// ── Common header: includes and struct definitions ──

/// Preamble included in every CUDA kernel compilation.
/// Defines the Q4_K / Q4_KF / Q6_K block structs and f16 decode.
///
/// NVRTC doesn't support C++ headers like <cstdint> or C11's
/// <stdint.h>. We define our own typedefs and use CUDA built-in
/// __half instead of C++ types.
pub const HEADER: &str = r#"
#include <cuda_fp16.h>

// NVRTC-safe type aliases (no <cstdint> available)
typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;
typedef unsigned char  uint8_local_t;
typedef unsigned short uint16_local_t;
typedef unsigned int   uint32_local_t;
typedef signed char    int8_local_t;

// NVRTC doesn't include <cmath> or <math.h> by default
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fc00000))
#endif

// ── Q4_K super-block: 256 values in 144 bytes (GGUF / llama.cpp layout) ──
//
// Same layout as Metal's `block_q4_K`:
//   [0..2]    half d        (super-block scale)
//   [2..4]    half dmin     (super-block min scale)
//   [4..16]   12 bytes of packed 6-bit scales + 6-bit mins
//   [16..144] 128 bytes of 4-bit nibbles (256 values, 8 sub-blocks)
struct block_q4_K {
    half d;           // super-block scale (2 bytes)
    half dmin;        // super-block min scale (2 bytes)
    uint8_local_t scales[12]; // 8 scales + 8 mins packed 6 bits each
    uint8_local_t qs[128];     // 256 × 4-bit values (128 bytes)
};                     // Total: 144 bytes

// ── Q4_KF super-block: 256 values in 160 bytes ──
// Pre-baked scales: d*scale_j and dmin*min_j pre-computed as half.
struct block_q4_kf {
    half scales[8];   // pre-computed d * scale_j (16 bytes)
    half mins[8];     // pre-computed dmin * min_j (16 bytes)
    uint8_local_t qs[128]; // 256 × 4-bit values (128 bytes)
};                    // Total: 160 bytes

// ── Q6_K super-block: 256 values in 210 bytes (GGUF layout) ──
struct block_q6_K {
    half d;            // super-block scale
    half dmin;         // super-block min scale
    uint8_local_t scales[16]; // packed scales and mins
    uint8_local_t ql[128];    // packed 6-bit values (low bits)
    uint8_local_t qh[64];     // packed 6-bit values (high bits, 2 bits each)
};                      // Total: 210 bytes

// ── f16 decode — mirrors Metal's decode_f16_metal ──
__device__ __forceinline__ float decode_f16(uint16_local_t bits) {
    return __half2float(__ushort_as_half(bits));
}

// ── Q4_K scale/min unpack ──
// Mirrors `unpack_q4k_scales_mins` from the plan.
__device__ __forceinline__ void unpack_q4k_scales_mins(
    const uint8_local_t* sb_bytes, int j,
    float d, float dmin,
    float* scale, float* mmin)
{
    uint8_local_t sc, mn;
    if (j < 4) {
        sc = sb_bytes[j] & 0x3F;
        mn = sb_bytes[j + 4] & 0x3F;
    } else {
        sc = (sb_bytes[j + 4] & 0x0F) | ((sb_bytes[j - 4] >> 6) << 4);
        mn = (sb_bytes[j + 4] >> 4) | ((sb_bytes[j] >> 6) << 4);
    }
    *scale = d * (float)sc;
    *mmin  = dmin * (float)mn;
}
"#;

// ── Warp reduction functions ──

/// Warp-level and block-level reduction helpers.
/// These mirror Metal's `simd_sum` / `simd_max` / `simd_min`.
pub const WARP_REDUCTIONS: &str = r#"
// ── Warp (32-thread) reductions — mirrors Metal's simd_sum/simd_max/simd_min ──

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ unsigned int warp_reduce_min_uint(unsigned int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// ── Block-level reductions (for >1 warp per block) ──
//
// Thread 0 does the final merge — avoids the UB that happens when
// warp-reduce intrinsics are called from a divergent subset of warp
// threads (e.g. threads 0-7 of warp 0 with 0xFFFFFFFF mask while
// threads 8-31 are inactive).

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    val = warp_reduce_sum(val);
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        val = shared[0];
        for (int i = 1; i < num_warps; i++) val += shared[i];
        shared[0] = val;
    }
    __syncthreads();
    val = shared[0];
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    val = warp_reduce_max(val);
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        val = shared[0];
        for (int i = 1; i < num_warps; i++) val = fmax(val, shared[i]);
        shared[0] = val;
    }
    __syncthreads();
    val = shared[0];
    return val;
}
"#;