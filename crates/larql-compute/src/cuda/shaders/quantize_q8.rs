//! Quantize f32 → Q8_0: per-block scale + int8 quantization.
//!
//! Q8_0 format: each block of 32 floats → 1 f32 scale + 32 signed int8 values.
//! Output: `q8_out[K]` (int8) + `scales[K/32]` (f32).
//! One thread per block of 32 elements.

use crate::cuda::kernel::TiledKernel;

pub struct QuantizeQ8Kernel;
impl TiledKernel for QuantizeQ8Kernel {
    const KERNEL_NAME: &'static str = "quantize_q8";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const SHADER: &str = r#"
// Quantize f32 → Q8_0: per-block (32 elements) int8 with scale.
// One thread per block.
// q8_out is treated as raw bytes: output is (int8 values, 32 per block)
// scales is a float array: one scale per block.
extern "C" __global__ void quantize_q8(
    const float* __restrict__ input,
    signed char* __restrict__ q8_out,
    float* __restrict__ scales,
    const unsigned int K)
{
    unsigned int block = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num_blocks = K / 32;
    if (block >= num_blocks) return;

    unsigned int off = block * 32;
    float amax = 0.0f;
    for (unsigned int j = 0; j < 32; j++) {
        float v = fabsf(input[off + j]);
        if (v > amax) amax = v;
    }
    float scale = amax / 127.0f;
    float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    scales[block] = scale;
    for (unsigned int j = 0; j < 32; j++) {
        float v = input[off + j] * inv;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        q8_out[off + j] = (signed char)roundf(v);
    }
}
"#;