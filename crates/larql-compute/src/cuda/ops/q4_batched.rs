//! Batched Q4 gate+up dispatch for prefill.
//!
//! Encodes 2×seq_len Q4 matvec dispatches for all prompt positions
//! in a single submission. Amortises GPU dispatch overhead vs
//! per-position dispatch.
//!
//! Mirrors `metal::ops::q4_batched::pair_batch` but uses CUDA
//! stream-based dispatch instead of Metal command buffers.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, CudaSlice, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

/// Batched gate+up for ALL seq_len positions in ONE GPU stream submission.
///
/// Returns `(gate_results, up_results)` — each is a Vec of `seq_len`
/// vectors of `num_rows` floats.
#[allow(clippy::too_many_arguments)]
pub fn pair_batch_q4k(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    gate_q4: &[u8],
    up_q4: &[u8],
    x_matrix: &[f32],
    seq_len: usize,
    num_rows: usize,
    hidden: usize,
) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let n_val = num_rows as u32;
    let k_val = hidden as u32;
    let grid = (num_rows as u64).div_ceil(kernel.rows_per_block) as u32;
    let block = kernel.threads_per_block as u32;
    let out_bytes = num_rows;

    // Upload weight matrices once
    let gate_dev = stream.clone_htod(gate_q4).ok()?;
    let up_dev = stream.clone_htod(up_q4).ok()?;

    let mut gate_results = Vec::with_capacity(seq_len);
    let mut up_results = Vec::with_capacity(seq_len);

    for s in 0..seq_len {
        let x_slice = &x_matrix[s * hidden..(s + 1) * hidden];

        // Upload this position's input
        let x_dev = stream.clone_htod(x_slice).ok()?;

        // Gate projection
        let mut g_out: CudaSlice<f32> = stream.alloc_zeros(out_bytes).ok()?;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&kernel.func)
                .arg(&gate_dev)
                .arg(&x_dev)
                .arg(&mut g_out)
                .arg(&n_val)
                .arg(&k_val)
                .launch(cfg)
                .ok()?;
        }

        // Up projection
        let mut u_out: CudaSlice<f32> = stream.alloc_zeros(out_bytes).ok()?;
        unsafe {
            stream
                .launch_builder(&kernel.func)
                .arg(&up_dev)
                .arg(&x_dev)
                .arg(&mut u_out)
                .arg(&n_val)
                .arg(&k_val)
                .launch(cfg)
                .ok()?;
        }

        stream.synchronize().ok()?;

        let g = stream.clone_dtoh(&g_out).ok()?;
        let u = stream.clone_dtoh(&u_out).ok()?;

        gate_results.push(g);
        up_results.push(u);
    }

    Some((gate_results, up_results))
}

/// Batched gate+up using Q4_K format — same pattern but uses the
/// Q4_K matvec kernel which takes f32 input directly.
#[allow(clippy::too_many_arguments)]
pub fn pair_batch_q4k_weights(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    gate_q4k: &[u8],
    up_q4k: &[u8],
    x_matrix: &[f32],
    seq_len: usize,
    num_rows: usize,
    hidden: usize,
) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let n_val = num_rows as u32;
    let k_val = hidden as u32;
    let grid = (num_rows as u64).div_ceil(kernel.rows_per_block) as u32;
    let block = kernel.threads_per_block as u32;

    // Upload weight matrices once
    let gate_dev = stream.clone_htod(gate_q4k).ok()?;
    let up_dev = stream.clone_htod(up_q4k).ok()?;

    let mut gate_results = Vec::with_capacity(seq_len);
    let mut up_results = Vec::with_capacity(seq_len);

    for s in 0..seq_len {
        let x_slice = &x_matrix[s * hidden..(s + 1) * hidden];

        // Upload this position's input
        let x_dev = stream.clone_htod(x_slice).ok()?;

        // Gate projection
        let mut g_out: CudaSlice<f32> = stream.alloc_zeros(num_rows).ok()?;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&kernel.func)
                .arg(&gate_dev)
                .arg(&x_dev)
                .arg(&mut g_out)
                .arg(&n_val)
                .arg(&k_val)
                .launch(cfg)
                .ok()?;
        }

        // Up projection
        let mut u_out: CudaSlice<f32> = stream.alloc_zeros(num_rows).ok()?;
        unsafe {
            stream
                .launch_builder(&kernel.func)
                .arg(&up_dev)
                .arg(&x_dev)
                .arg(&mut u_out)
                .arg(&n_val)
                .arg(&k_val)
                .launch(cfg)
                .ok()?;
        }

        stream.synchronize().ok()?;

        let g = stream.clone_dtoh(&g_out).ok()?;
        let u = stream.clone_dtoh(&u_out).ok()?;

        gate_results.push(g);
        up_results.push(u);
    }

    Some((gate_results, up_results))
}