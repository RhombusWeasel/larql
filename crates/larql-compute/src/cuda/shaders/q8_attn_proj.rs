//! Fused Q8 QKV projection — all 3 attention projections in one dispatch.
//!
//! `q8_qkv_proj`: fused Q+K+V projection.
//! `q8_proj_rope`: single projection for O.

use crate::cuda::kernel::TiledKernel;

pub struct Q8QkvProjKernel;
impl TiledKernel for Q8QkvProjKernel {
    const KERNEL_NAME: &'static str = "q8_qkv_proj";
    const ROWS_PER_BLOCK: u64 = 8;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub struct Q8ProjRopeKernel;
impl TiledKernel for Q8ProjRopeKernel {
    const KERNEL_NAME: &'static str = "q8_proj_rope";
    const ROWS_PER_BLOCK: u64 = 8;
    const THREADS_PER_BLOCK: u64 = 256;
}

pub const SHADER: &str = r#"
extern "C" __global__ void q8_qkv_proj(
    const uint8_local_t* __restrict__ Wq,
    const uint8_local_t* __restrict__ Wk,
    const uint8_local_t* __restrict__ Wv,
    const signed char* __restrict__ X8,
    const float* __restrict__ Wqs,
    const float* __restrict__ Wks,
    const float* __restrict__ Wvs,
    const float* __restrict__ X8s,
    float* __restrict__ Q_out,
    float* __restrict__ K_out,
    float* __restrict__ V_out,
    const unsigned int q_rows,
    const unsigned int k_rows,
    const unsigned int v_rows,
    const unsigned int K)
{
    unsigned int total_rows = q_rows + k_rows + v_rows;
    unsigned int sg_id = threadIdx.x / 32;
    unsigned int lane = threadIdx.x % 32;
    unsigned int global_row = blockIdx.x * 8 + sg_id;
    if (global_row >= total_rows) return;

    unsigned int blocks = K / 32;

    __shared__ signed char s_x8[8192];
    __shared__ float s_xs[256];
    unsigned int tid = threadIdx.x;
    for (unsigned int i = tid; i < K; i += 256) {
        s_x8[i] = X8[i];
    }
    for (unsigned int i = tid; i < blocks; i += 256) {
        s_xs[i] = X8s[i];
    }
    __syncthreads();

    const uint8_local_t* W;
    const float* Ws;
    float* out_buf;
    unsigned int local_row;
    if (global_row < q_rows) {
        W = Wq; Ws = Wqs; out_buf = Q_out; local_row = global_row;
    } else if (global_row < q_rows + k_rows) {
        W = Wk; Ws = Wks; out_buf = K_out; local_row = global_row - q_rows;
    } else {
        W = Wv; Ws = Wvs; out_buf = V_out; local_row = global_row - q_rows - k_rows;
    }

    const signed char* row_data = (const signed char*)(W + local_row * K);
    const float* row_scales = Ws + local_row * blocks;

    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks; b += 32) {
        float combined_scale = row_scales[b] * s_xs[b];
        const signed char* wb = row_data + b * 32;
        const signed char* xb = s_x8 + b * 32;

        int isum = 0;
        for (unsigned int j = 0; j < 32; j++) {
            isum += (int)wb[j] * (int)xb[j];
        }
        acc += (float)isum * combined_scale;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) out_buf[local_row] = acc;
}
"#;

pub const PROJ_ROPE_SHADER: &str = r#"
extern "C" __global__ void q8_proj_rope(
    const uint8_local_t* __restrict__ W8,
    const signed char* __restrict__ X8,
    const float* __restrict__ W8s,
    const float* __restrict__ X8s,
    float* __restrict__ out,
    const unsigned int num_rows,
    const unsigned int K)
{
    unsigned int sg_id = threadIdx.x / 32;
    unsigned int lane = threadIdx.x % 32;
    unsigned int row = blockIdx.x * 8 + sg_id;
    if (row >= num_rows) return;

    unsigned int blocks = K / 32;

    __shared__ signed char s_x8[8192];
    __shared__ float s_xs[256];
    unsigned int tid = threadIdx.x;
    for (unsigned int i = tid; i < K; i += 256) {
        s_x8[i] = X8[i];
    }
    for (unsigned int i = tid; i < blocks; i += 256) {
        s_xs[i] = X8s[i];
    }
    __syncthreads();

    const signed char* row_data = (const signed char*)(W8 + row * K);
    const float* row_scales = W8s + row * blocks;

    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks; b += 32) {
        float combined_scale = row_scales[b] * s_xs[b];
        const signed char* wb = row_data + b * 32;
        const signed char* xb = s_x8 + b * 32;

        int isum = 0;
        for (unsigned int j = 0; j < 32; j++) {
            isum += (int)wb[j] * (int)xb[j];
        }
        acc += (float)isum * combined_scale;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;