//! Tiled f32 matrix multiply: C = A × B and C = A × B^T.
//!
//! 32×32 tiles, shared memory tiling.

use crate::cuda::kernel::TiledKernel;

pub struct SgemmKernel;
impl TiledKernel for SgemmKernel {
    const KERNEL_NAME: &'static str = "sgemm";
    const ROWS_PER_BLOCK: u64 = 32;
    const THREADS_PER_BLOCK: u64 = 1024;
}

pub struct SgemmTransBKernel;
impl TiledKernel for SgemmTransBKernel {
    const KERNEL_NAME: &'static str = "sgemm_transb";
    const ROWS_PER_BLOCK: u64 = 32;
    const THREADS_PER_BLOCK: u64 = 1024;
}

pub const SGEMM_SHADER: &str = r#"
// Tiled f32 matrix multiply: C = A × B
// 32×32 tiles with shared memory.
// Grid: ((N + 31) / 32, (M + 31) / 32, 1)
// Each thread block computes a 32×32 tile of C.
extern "C" __global__ void sgemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const unsigned int M,
    const unsigned int N,
    const unsigned int K)
{
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    unsigned int row = blockIdx.y * 32 + threadIdx.y;
    unsigned int col = blockIdx.x * 32 + threadIdx.x;
    float acc = 0.0f;
    unsigned int tiles = (K + 31) / 32;

    for (unsigned int t = 0; t < tiles; t++) {
        unsigned int ac = t * 32 + threadIdx.x;
        unsigned int br = t * 32 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ac < K) ? A[row * K + ac] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (br < K && col < N) ? B[br * N + col] : 0.0f;
        __syncthreads();
        for (unsigned int i = 0; i < 32; i++) {
            acc = fma(As[threadIdx.y][i], Bs[i][threadIdx.x], acc);
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
"#;

pub const SGEMM_TRANSB_SHADER: &str = r#"
// Tiled f32 matrix multiply transposed: C = A × B^T
// 32×32 tiles with shared memory.
// Grid: ((N + 31) / 32, (M + 31) / 32, 1)
extern "C" __global__ void sgemm_transb(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const unsigned int M,
    const unsigned int N,
    const unsigned int K)
{
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    unsigned int row = blockIdx.y * 32 + threadIdx.y;
    unsigned int col = blockIdx.x * 32 + threadIdx.x;
    float acc = 0.0f;
    unsigned int tiles = (K + 31) / 32;

    for (unsigned int t = 0; t < tiles; t++) {
        unsigned int ac = t * 32 + threadIdx.x;
        unsigned int bk = t * 32 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ac < K) ? A[row * K + ac] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && bk < K) ? B[col * K + bk] : 0.0f;
        __syncthreads();
        for (unsigned int i = 0; i < 32; i++) {
            acc = fma(As[threadIdx.y][i], Bs[i][threadIdx.x], acc);
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
"#;