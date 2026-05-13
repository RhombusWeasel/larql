//! Q4_0 vector-matrix (scatter-accumulate) — down projection transposed.
//!
//! out[K] = activation[N] @ Q4[N, K]
//! One thread per output element (column), reading across all N rows.

use crate::cuda::kernel::TiledKernel;

pub struct Q4VecMatKernel;
impl TiledKernel for Q4VecMatKernel {
    const KERNEL_NAME: &'static str = "q4_vecmat";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const SHADER: &str = r#"
// Q4_0 vector-matrix multiply (scatter-accumulate).
// out[K] = activation[N] @ Q4[N, K]
// One thread per output element, each thread reads across all rows.
// Q4 format: 18 bytes per block of 32 values (2B f16 scale + 16B nibbles).
extern "C" __global__ void q4_vecmat(
    const float* __restrict__ activation, // [N]
    const uint8_local_t* __restrict__ Q4,  // [N, blocks_per_row * 18]
    float* __restrict__ out,              // [K]
    const unsigned int N,
    const unsigned int K)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= K) return;

    unsigned int blocks_per_row = K / 32;
    unsigned int bytes_per_row = blocks_per_row * 18;
    unsigned int block_idx = tid / 32;
    unsigned int elem_in_block = tid % 32;
    unsigned int nibble_idx = elem_in_block / 2;
    bool is_high = (elem_in_block & 1) != 0;

    float acc = 0.0f;
    for (unsigned int row = 0; row < N; row++) {
        float act = activation[row];
        if (act < 1e-10f && act > -1e-10f) continue;

        const uint8_local_t* block = Q4 + row * bytes_per_row + block_idx * 18;
        uint16_local_t scale_bits = (uint16_local_t)(block[0]) | ((uint16_local_t)(block[1]) << 8);
        float q4_scale = decode_f16(scale_bits);

        uint8_local_t byte = block[2 + nibble_idx];
        int q_val = is_high ? ((int)(byte >> 4) - 8) : ((int)(byte & 0x0F) - 8);
        acc += (float)q_val * q4_scale * act;
    }
    out[tid] = acc;
}
"#;