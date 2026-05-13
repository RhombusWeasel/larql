//! GEGLU activation — SiLU and GELU-tanh elementwise kernels.
//!
//! GEGLU(gate, up) = activation(gate) * up
//! SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//! GELU-tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))

use crate::cuda::kernel::TiledKernel;

pub struct GegeluSiluKernel;
impl TiledKernel for GegeluSiluKernel {
    const KERNEL_NAME: &'static str = "geglu_silu";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub struct GegeluGeluTanhKernel;
impl TiledKernel for GegeluGeluTanhKernel {
    const KERNEL_NAME: &'static str = "geglu_gelu_tanh";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const GEGELU_SILU_SHADER: &str = r#"
// GEGLU SiLU: out[i] = silu(gate[i]) * up[i]
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
extern "C" __global__ void gegelu_silu(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const unsigned int n)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}
"#;

pub const GEGELU_GELU_TANH_SHADER: &str = r#"
// GEGLU GELU-tanh: out[i] = gelu_tanh(gate[i]) * up[i]
// gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
extern "C" __global__ void gegelu_gelu_tanh(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const unsigned int n)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const float SQRT_2_OVER_PI = 0.7978845608028654f; // sqrt(2/pi)
    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float g = gate[i];
        float inner = SQRT_2_OVER_PI * (g + 0.044715f * g * g * g);
        float gelu = 0.5f * g * (1.0f + tanhf(inner));
        out[i] = gelu * up[i];
    }
}
"#;