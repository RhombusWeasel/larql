//! Q4_0 × f32 matrix-vector multiply.
//!
//! scores[N] = Q4[N, K] · x[K]
//! Input x is f32 (not Q8). Each thread handles one output row.

use crate::cuda::kernel::TiledKernel;

pub struct Q4F32MatvecKernel;
impl TiledKernel for Q4F32MatvecKernel {
    const KERNEL_NAME: &'static str = "q4_f32_matvec";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const SHADER: &str = r#"
// Q4 × f32 matvec: scores[N] = Q4[N, K] · x[K]
// Each thread handles one output row (flat grid).
// Q4 format: 18 bytes per block of 32 (2B f16 scale + 16B nibbles).
extern "C" __global__ void q4_f32_matvec(
    const uint8_local_t* __restrict__ Q4,     // [N, blocks_per_row * 18]
    const float* __restrict__ x,              // [K]
    float* __restrict__ out,                  // [N]
    const unsigned int N,
    const unsigned int K)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    unsigned int blocks = K / 32;
    unsigned int bytes_per_row = blocks * 18;
    const uint8_local_t* row = Q4 + tid * bytes_per_row;

    float acc = 0.0f;
    for (unsigned int b = 0; b < blocks; b++) {
        const uint8_local_t* block = row + b * 18;
        uint16_local_t scale_bits = (uint16_local_t)(block[0]) | ((uint16_local_t)(block[1]) << 8);
        float q4_scale = decode_f16(scale_bits);

        const uint8_local_t* quants = block + 2;
        const float* xb = x + b * 32;
        float block_sum = 0.0f;
        for (unsigned int j = 0; j < 16; j++) {
            uint8_local_t byte = quants[j];
            float lo = (float)((int)(byte & 0x0F) - 8);
            float hi = (float)((int)((byte >> 4) & 0x0F) - 8);
            block_sum += lo * xb[j * 2] + hi * xb[j * 2 + 1];
        }
        acc += block_sum * q4_scale;
    }
    out[tid] = acc;
}
"#;