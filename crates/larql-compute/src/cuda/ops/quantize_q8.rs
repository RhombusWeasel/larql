//! Quantize f32 → Q8_0 dispatch — CUDA.
//!
//! Q8_0 format: each block of 32 floats → 1 f32 scale + 32 signed int8 values.
//! Output: `q8_out[K]` (int8) + `scales[K/32]` (f32).

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

/// Quantize f32 vector to Q8_0 format.
///
/// Returns `(q8_bytes, scales)` where:
/// - `q8_bytes` is `K` signed i8 values
/// - `scales` is `K/32` f32 values (one per block of 32)
pub fn encode_quantize_q8(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    input: &[f32],
    k: usize,
) -> Option<(Vec<i8>, Vec<f32>)> {
    if k % 32 != 0 || input.len() < k {
        return None;
    }

    let num_blocks = k / 32;
    let input_dev = stream.clone_htod(&input[..k]).ok()?;
    let mut q8_dev = stream.alloc_zeros::<i8>(k).ok()?;
    let mut scales_dev = stream.alloc_zeros::<f32>(num_blocks).ok()?;

    let k_u32 = k as u32;
    let threads = 256u32;
    let blocks = (num_blocks as u64).div_ceil(threads as u64) as u32;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    // Note: i8 (signed char) in the kernel maps to CudaSlice<i8> here.
    // The kernel uses `signed char*` for q8_out which matches i8 on the Rust side.
    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&input_dev)
            .arg(&mut q8_dev)
            .arg(&mut scales_dev)
            .arg(&k_u32)
            .launch(cfg)
            .ok()?;
    }

    stream.synchronize().ok()?;
    let q8_out = stream.clone_dtoh(&q8_dev).ok()?;
    let scales_out = stream.clone_dtoh(&scales_dev).ok()?;
    Some((q8_out, scales_out))
}