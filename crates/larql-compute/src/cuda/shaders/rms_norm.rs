//! RMS norm — port of `metal/shaders/residual_inject.rs` (RmsNormKernel).
//!
//! Computes `out[i] = x[i] * (weight[i] + offset) * rms` where
//! `rms = 1 / sqrt(mean(x²) + eps)`.
//!
//! Dispatched as 1 block with `len` threads. Each thread sums a stripe
//! of elements, then warp+block reduce for the sum-of-squares.
//! All threads then write their output elements.

use crate::cuda::kernel::TiledKernel;

pub const SHADER: &str = r#"
extern "C" __global__ void rms_norm(
    const float* __restrict__ x,         // [len]
    const float* __restrict__ weight,    // [len]
    float* __restrict__ out,             // [len]
    const unsigned int len,
    const float eps,
    const float offset)
{
    // Cooperative sum_sq: each thread sums a stripe, then warp+block reduce
    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < len; i += blockDim.x) {
        partial += x[i] * x[i];
    }

    // Warp-level reduce
    partial = warp_reduce_sum(partial);

    // Block-level reduce
    __shared__ float shared[32]; // max 32 warps per block
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared[warp_id] = partial;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    float sum_sq = (lane < num_warps) ? shared[lane] : 0.0f;
    sum_sq = warp_reduce_sum(sum_sq);

    // Broadcast: store total sum to shared, all threads read it
    __shared__ float total_sum;
    if (threadIdx.x == 0) {
        total_sum = sum_sq;
    }
    __syncthreads();
    sum_sq = total_sum;

    float rms_val = 1.0f / sqrtf(sum_sq / (float)len + eps);

    // Write all output elements (loop for len > blockDim.x)
    for (unsigned int i = threadIdx.x; i < len; i += blockDim.x) {
        out[i] = x[i] * (weight[i] + offset) * rms_val;
    }
}
"#;

/// RmsNormKernel dispatch geometry.
/// Not a tiled kernel in the traditional sense — it's dispatched as
/// 1 block with up to 1024 threads. The block size is set at dispatch
/// time based on hidden size.
pub struct RmsNormKernel;
impl TiledKernel for RmsNormKernel {
    const KERNEL_NAME: &'static str = "rms_norm";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256; // Default, overridden at dispatch
}

/// Residual add: out[i] = a[i] + b[i]. Elementwise, 1 thread per element.
pub const RESIDUAL_ADD_SHADER: &str = r#"
extern "C" __global__ void residual_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    const unsigned int len)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) out[i] = a[i] + b[i];
}
"#;

/// Residual add kernel marker.
pub struct ResidualAddKernel;
impl TiledKernel for ResidualAddKernel {
    const KERNEL_NAME: &'static str = "residual_add";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

/// Scale vector: out[i] = input[i] * scalar. Elementwise, 1 thread per element.
pub const SCALE_VECTOR_SHADER: &str = r#"
extern "C" __global__ void scale_vector(
    const float* __restrict__ input,
    float* __restrict__ out,
    const unsigned int len,
    const float scalar)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) out[i] = input[i] * scalar;
}
"#;

/// Scale vector kernel marker.
pub struct ScaleVectorKernel;
impl TiledKernel for ScaleVectorKernel {
    const KERNEL_NAME: &'static str = "scale_vector";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}