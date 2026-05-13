//! LayerNorm — standard layer normalization (mean-subtraction + variance normalization).
//!
//! `layer_norm`: out = (x - mean) / sqrt(var + eps) * (weight + offset) + bias
//! `layer_norm_no_bias`: out = (x - mean) / sqrt(var + eps) * (weight + offset)
//!
//! Each thread handles one element, but reads all elements for mean/var.
//! Grid: (len, 1, 1).

use crate::cuda::kernel::TiledKernel;

pub struct LayerNormKernel;
impl TiledKernel for LayerNormKernel {
    const KERNEL_NAME: &'static str = "layer_norm";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub struct LayerNormNoBiasKernel;
impl TiledKernel for LayerNormNoBiasKernel {
    const KERNEL_NAME: &'static str = "layer_norm_no_bias";
    const ROWS_PER_BLOCK: u64 = 1;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const SHADER: &str = r#"
// LayerNorm: out = (x - mean) / sqrt(var + eps) * (weight + offset) + bias
// Each thread handles one output element; all threads read the full vector for mean/var.
extern "C" __global__ void layer_norm(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const unsigned int len,
    const float eps,
    const float offset)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len) return;

    // Cooperative sum for mean
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < len; i += blockDim.x) {
        sum += x[i];
    }
    sum = block_reduce_sum(sum);
    float mean = sum / (float)len;

    // Cooperative variance
    float var_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < len; i += blockDim.x) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum);
    float inv_std = 1.0f / sqrtf(var_sum / (float)len + eps);

    // Write all output elements
    for (unsigned int i = threadIdx.x; i < len; i += blockDim.x) {
        out[i] = (x[i] - mean) * inv_std * (weight[i] + offset) + bias[i];
    }
}

// LayerNorm without bias: out = (x - mean) / sqrt(var + eps) * (weight + offset)
extern "C" __global__ void layer_norm_no_bias(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    const unsigned int len,
    const float eps,
    const float offset)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len && threadIdx.x != 0) {
        // All threads must participate in the reduction even if they don't write
    }

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < len; i += blockDim.x) {
        sum += x[i];
    }
    sum = block_reduce_sum(sum);
    float mean = sum / (float)len;

    float var_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < len; i += blockDim.x) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum);
    float inv_std = 1.0f / sqrtf(var_sum / (float)len + eps);

    for (unsigned int i = threadIdx.x; i < len; i += blockDim.x) {
        out[i] = (x[i] - mean) * inv_std * (weight[i] + offset);
    }
}
"#;