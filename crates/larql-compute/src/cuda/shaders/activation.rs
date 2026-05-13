//! Standalone SiLU and GELU-tanh elementwise activation kernels.
//!
//! These are single-input kernels (not GEGLU which takes gate+up).
//! SiLU(x) = x * sigmoid(x)
//! GELU-tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))

use crate::cuda::kernel::TiledKernel;

pub struct SiluKernel;
impl TiledKernel for SiluKernel {
    const KERNEL_NAME: &'static str = "silu";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub struct GeluTanhKernel;
impl TiledKernel for GeluTanhKernel {
    const KERNEL_NAME: &'static str = "gelu_tanh";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const SILU_SHADER: &str = r#"
extern "C" __global__ void silu(
    const float* __restrict__ input,
    float* __restrict__ output,
    const unsigned int n)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}
"#;

pub const GELU_TANH_SHADER: &str = r#"
extern "C" __global__ void gelu_tanh(
    const float* __restrict__ input,
    float* __restrict__ output,
    const unsigned int n)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float x = input[i];
        float inner = SQRT_2_OVER_PI * (x + 0.044715f * x * x * x);
        output[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}
"#;