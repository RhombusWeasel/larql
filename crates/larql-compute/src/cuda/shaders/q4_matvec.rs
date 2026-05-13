//! Q4_0 × Q8 matrix-vector multiply.
//!
//! Q4_0 block: 18 bytes per 32 values (2-byte f16 scale + 16 bytes of 4-bit
//! nibbles). Input is Q8 format (int8 values + per-block f32 scales).
//! 4 warps per block, 8 rows per block, matching the Metal kernel geometry.

use crate::cuda::kernel::TiledKernel;

pub struct Q4MatvecKernel;
impl TiledKernel for Q4MatvecKernel {
    const KERNEL_NAME: &'static str = "q4_matvec_v4";
    const ROWS_PER_BLOCK: u64 = 8;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const SHADER: &str = r#"
// Q4_0 × Q8 matvec: scores[N] = Q4[N,K] · Q8_x[K]
// Q4 format: 18 bytes per 32 values (2B f16 scale + 16B nibbles)
// Q8 format: int8 values + per-block float scales
// 8 rows per block, 32 threads per warp, 8 warps = 256 threads
extern "C" __global__ void q4_matvec_v4(
    const uint8_local_t* __restrict__ Q4,      // [N, blocks_per_row * 18]
    const signed char* __restrict__ Q8,        // [K] int8 values
    const float* __restrict__ Q8s,             // [K / 32] per-block scales
    float* __restrict__ out,                   // [N]
    const unsigned int N,
    const unsigned int K)
{
    unsigned int blocks = K / 32;
    unsigned int bytes_per_row = blocks * 18;

    // Each warp handles one row; sg_id selects the row within this block
    unsigned int sg_id = threadIdx.x / 32;
    unsigned int lane = threadIdx.x % 32;
    unsigned int row_idx = blockIdx.x * 8 + sg_id;
    if (row_idx >= N) return;

    const uint8_local_t* row = Q4 + row_idx * bytes_per_row;

    // Q8 data is loaded from global memory directly (no shared memory)
    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks; b += 32) {
        // Q4 block: 2 bytes scale (f16) + 16 bytes nibbles
        const uint8_local_t* block = row + b * 18;

        // Scale: first 2 bytes are f16 combined scale
        uint16_local_t scale_bits = (uint16_local_t)(block[0]) | ((uint16_local_t)(block[1]) << 8);
        float combined_scale = decode_f16(scale_bits) * Q8s[b];

        // Decode 16 nibble bytes → 32 values, dot with Q8
        const uint8_local_t* qs = block + 2;
        const signed char* q8_block = Q8 + b * 32;

        int isum = 0;
        for (unsigned int j = 0; j < 16; j++) {
            uint8_local_t byte = qs[j];
            int lo = (int)(byte & 0x0Fu) - 8;
            int hi = (int)((byte >> 4) & 0x0Fu) - 8;
            isum += lo * (int)q8_block[j * 2];
            isum += hi * (int)q8_block[j * 2 + 1];
        }
        acc += (float)isum * combined_scale;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row_idx] = acc;
}
"#;