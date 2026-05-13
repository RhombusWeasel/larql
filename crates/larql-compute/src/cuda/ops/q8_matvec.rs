//! Q8_0 matrix-vector dispatch — CUDA.

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

/// Q8_0 matvec: `out[N] = W8[N,K] · Q8_x[K]`
///
/// 8 rows per block, shared memory for Q8 input.
pub fn encode_q8_matvec(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    w8_data: &[u8],
    q8_data: &[i8],
    w8_scales: &[f32],
    q8_scales: &[f32],
    num_rows: usize,
    hidden: usize,
) -> Option<Vec<f32>> {
    if hidden % 32 != 0 {
        return None;
    }

    let w8_dev = stream.clone_htod(w8_data).ok()?;
    let q8_dev = stream.clone_htod(q8_data).ok()?;
    let w8s_dev = stream.clone_htod(w8_scales).ok()?;
    let q8s_dev = stream.clone_htod(q8_scales).ok()?;
    let mut out_dev = stream.alloc_zeros::<f32>(num_rows).ok()?;

    let n_u32 = num_rows as u32;
    let k_u32 = hidden as u32;
    let num_blocks = (num_rows as u64).div_ceil(kernel.rows_per_block) as u32;
    let block_size = kernel.threads_per_block as u32;

    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&w8_dev)
            .arg(&q8_dev)
            .arg(&w8s_dev)
            .arg(&q8s_dev)
            .arg(&mut out_dev)
            .arg(&n_u32)
            .arg(&k_u32)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;
    stream.clone_dtoh(&out_dev).ok()
}