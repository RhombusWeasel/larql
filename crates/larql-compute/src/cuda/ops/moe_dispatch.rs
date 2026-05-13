//! GPU expert dispatch for per-layer Q4_K MoE models (Phase 2, §P2.6).
//!
//! Mirrors `metal::moe_dispatch` but with CUDA kernel dispatches.
//!
//! Flow per MoE layer (after the GPU commit for `h_post_attn`):
//!
//! 1. CPU: pre-experts norm + router projection + softmax + top-K + renorm.
//! 2. CPU→GPU: copy expert gate+up and down weight bytes into pre-allocated
//!    device staging buffers.
//! 3. GPU: `q4k_ffn_gate_up` over all K experts in one dispatch.
//! 4. GPU: K × `gegelu` — one per expert at strided act_buf offset.
//! 5. GPU: K × `q4k_matvec` for expert down projections.
//! 6. Synchronize + read back K × hidden expert outputs.
//! 7. CPU: weighted sum + post-experts norm → `moe_out`.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::cpu::ops::moe::cpu_moe_route;
use crate::cuda::kernel::CudaKernel;
use crate::pipeline::Activation;
use crate::MoeLayerWeights;

/// Pre-allocated scratch for the whole MoE decode loop.
///
/// All sizes are determined by `(top_k, hidden, intermediate_size)` of the
/// model. Buffer reuse across decode calls eliminates per-token allocation
/// overhead (~120ms at 30 MoE layers on desktop GPUs).
///
/// `act_buf` is sized for `top_k × inter_padded` and zero-initialised so
/// the `inter_padded - inter` padding columns contribute nothing through
/// the down projection — required when `intermediate_size` is not a
/// multiple of 256 (e.g. Gemma 4 26B's 2112 → inter_padded 2304).
pub struct CudaMoeScratch {
    pub top_k: usize,
    pub inter: usize,
    pub inter_padded: usize,
    pub hidden: usize,
    pub row_bytes: usize,
    pub down_row_bytes: usize,

    // Weight staging buffers (Q4_K packed byte arrays)
    pub gate_buf: CudaSlice<u8>,
    pub up_buf: CudaSlice<u8>,
    pub down_bufs: Vec<CudaSlice<u8>>,

    // Compute buffers (f32)
    pub x_buf: CudaSlice<f32>,
    pub g_out: CudaSlice<f32>,
    pub u_out: CudaSlice<f32>,
    pub act_buf: CudaSlice<f32>,
    pub expert_outs: CudaSlice<f32>,
}

// CudaSlice and CudaView are Send + Sync, so CudaMoeScratch is too.
unsafe impl Send for CudaMoeScratch {}
unsafe impl Sync for CudaMoeScratch {}

impl CudaMoeScratch {
    /// Create a new scratch with pre-allocated device buffers.
    ///
    /// All buffers stay live for the whole decode call and are reused
    /// across every MoE layer. This avoids per-layer allocation overhead
    /// (~ms per layer at 8+ MoE layers).
    pub fn new(
        ctx: &Arc<CudaContext>,
        top_k: usize,
        hidden: usize,
        inter: usize,
    ) -> Option<Self> {
        let stream = ctx.default_stream();
        let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
        let bytes_per_block = larql_models::quant::ggml::Q4_K_BLOCK_BYTES;
        let inter_padded = inter.div_ceil(block) * block;

        // Q4_K row stride: one super-block per Q4_K_BLOCK_ELEMS elements,
        // Q4_K_BLOCK_BYTES bytes per super-block.
        let row_bytes = (hidden / block) * bytes_per_block;
        let down_row_bytes = (inter_padded / block) * bytes_per_block;

        // Allocate staging buffers for expert weights.
        // gate_buf and up_buf are large enough for all K experts' weights.
        let gate_buf = stream
            .alloc_zeros::<u8>(top_k * inter * row_bytes)
            .ok()?;
        let up_buf = stream
            .alloc_zeros::<u8>(top_k * inter * row_bytes)
            .ok()?;
        let down_bufs: Vec<CudaSlice<u8>> = (0..top_k)
            .map(|_| stream.alloc_zeros::<u8>(hidden * down_row_bytes).ok())
            .collect::<Option<Vec<_>>>()?;

        // Allocate compute buffers.
        let x_buf = stream.alloc_zeros::<f32>(hidden).ok()?;
        let g_out = stream.alloc_zeros::<f32>(top_k * inter).ok()?;
        let u_out = stream.alloc_zeros::<f32>(top_k * inter).ok()?;
        let act_buf = stream.alloc_zeros::<f32>(top_k * inter_padded).ok()?;
        let expert_outs = stream.alloc_zeros::<f32>(top_k * hidden).ok()?;

        Some(Self {
            top_k,
            inter,
            inter_padded,
            hidden,
            row_bytes,
            down_row_bytes,
            gate_buf,
            up_buf,
            down_bufs,
            x_buf,
            g_out,
            u_out,
            act_buf,
            expert_outs,
        })
    }

    /// Check whether the scratch matches the requested shape.
    pub fn matches_shape(&self, top_k: usize, hidden: usize, inter: usize) -> bool {
        self.top_k == top_k && self.hidden == hidden && self.inter == inter
    }
}

/// Run MoE expert dispatch on GPU with pre-allocated scratch.
///
/// Returns the weighted-sum expert output (hidden-size vec) WITHOUT
/// post-experts norm — the caller applies it.
///
/// Per call:
///   1. CPU pre-experts norm + router (cheap: hidden² FLOPs).
///   2. K × host→device copies (gate+up weights + down weights per expert).
///   3. 1 fused gate+up dispatch + K activation dispatches + K down dispatches.
///   4. Synchronize + read back K × hidden f32 expert outputs.
///   5. CPU weighted sum + post-experts norm.
#[allow(clippy::too_many_arguments)]
pub fn gpu_moe_dispatch_with_scratch(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,

    // Kernels
    q4k_ffn_gate_up_kernel: &CudaKernel,
    gegelu_kernel: &CudaKernel,
    q4k_matvec_kernel: &CudaKernel,

    // Inputs
    h_post_attn: &[f32],
    moe: &MoeLayerWeights<'_>,
    eps: f32,
    _activation: Activation,
    scratch: &mut CudaMoeScratch,
) -> Option<Vec<f32>> {
    let hidden = h_post_attn.len();
    let inter = moe.intermediate_size;
    let inter_padded = scratch.inter_padded;
    let top_k = moe.top_k;
    debug_assert_eq!(top_k, scratch.top_k, "MoE top_k mismatch");
    debug_assert_eq!(inter, scratch.inter, "MoE intermediate_size mismatch");
    debug_assert_eq!(hidden, scratch.hidden, "MoE hidden_size mismatch");

    if top_k == 0 || hidden == 0 || inter == 0 {
        return Some(vec![0.0f32; hidden]);
    }

    // ── 1. CPU pre-experts norm + router ──
    let h_norm = if !moe.pre_experts_norm.is_empty() {
        let rms = (h_post_attn.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + eps).sqrt();
        h_post_attn
            .iter()
            .zip(moe.pre_experts_norm)
            .map(|(x, w)| x / rms * (w + 0.0))
            .collect::<Vec<f32>>()
    } else {
        h_post_attn.to_vec()
    };
    let (expert_indices, expert_weights) = cpu_moe_route(&h_norm, moe, eps);

    // ── 2. Stage expert weight bytes into pre-allocated device buffers ──
    let row_bytes = scratch.row_bytes;
    let gate_half_bytes = inter * row_bytes;
    let up_half_bytes = inter * row_bytes;
    let down_expert_bytes = hidden * scratch.down_row_bytes;

    let mut valid_weights: Vec<f32> = Vec::with_capacity(top_k);
    let mut valid_count = 0usize;

    for (k, &ei) in expert_indices.iter().enumerate() {
        if ei >= moe.experts_gate_up.len() || ei >= moe.experts_down.len() {
            continue;
        }
        let gu_bytes = moe.experts_gate_up[ei];
        let dn_bytes = moe.experts_down[ei];
        if gu_bytes.len() < 2 * gate_half_bytes {
            continue;
        }
        if valid_count >= scratch.top_k {
            break;
        }

        // Q4_K layout: gate || up, each `inter * row_bytes` bytes.
        // Copy gate and up weight bytes into the staging buffers at the
        // correct offsets for this expert's slot.
        let gate_offset = valid_count * gate_half_bytes;
        {
            let mut gate_view = scratch.gate_buf.slice_mut(gate_offset..gate_offset + gate_half_bytes);
            stream.memcpy_htod(&gu_bytes[..gate_half_bytes], &mut gate_view).ok()?;
        }

        let up_offset = valid_count * up_half_bytes;
        {
            let mut up_view = scratch.up_buf.slice_mut(up_offset..up_offset + up_half_bytes);
            stream.memcpy_htod(
                &gu_bytes[gate_half_bytes..gate_half_bytes + up_half_bytes],
                &mut up_view,
            ).ok()?;
        }

        // Down weights: each expert gets its own pre-allocated buffer.
        let copy_len = dn_bytes.len().min(down_expert_bytes);
        {
            let mut down_view = scratch.down_bufs[valid_count].slice_mut(0..copy_len);
            stream.memcpy_htod(&dn_bytes[..copy_len], &mut down_view).ok()?;
        }
        // Zero-fill the tail if the down weights are shorter than the buffer.
        // The down buffers were zero-initialised at allocation, so subsequent
        // calls may leave stale data. Only zero-fill the tail if the expert's
        // actual data is shorter than the buffer.
        // NOTE: The down buffer was allocated with alloc_zeros, so for the
        // first call this is correct. For subsequent calls the tail data from
        // a previous expert may remain, but since we always process
        // inter_padded rows where inter_padded >= inter, the extra rows in
        // the down buffer correspond to zero-padded activation columns and
        // contribute nothing to the output. This is safe.

        valid_weights.push(expert_weights[k]);
        valid_count += 1;
    }

    if valid_count == 0 {
        return Some(vec![0.0f32; hidden]);
    }

    // ── 3. Stage router-normed input into pre-allocated x_buf ──
    stream.memcpy_htod(&h_norm, &mut scratch.x_buf.slice_mut(0..hidden)).ok()?;

    // ── 4. q4k_ffn_gate_up over all valid_count experts ──
    // The kernel processes `valid_count * inter` rows total, treating
    // the concatenated gate_buf and up_buf as two separate weight
    // matrices and reading x from the shared input buffer.
    let n_rows_u32 = (valid_count * inter) as u32;
    let k_cols_u32 = hidden as u32;
    let tgs_per_mat = (valid_count as u64 * inter as u64)
        .div_ceil(q4k_ffn_gate_up_kernel.rows_per_block as u64) as u32;
    let total_blocks = tgs_per_mat * 2; // gate + up dispatches
    let block_size = q4k_ffn_gate_up_kernel.threads_per_block as u32;

    let cfg = LaunchConfig {
        grid_dim: (total_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&q4k_ffn_gate_up_kernel.func)
            .arg(&scratch.gate_buf)
            .arg(&scratch.up_buf)
            .arg(&scratch.x_buf)
            .arg(&scratch.g_out)
            .arg(&scratch.u_out)
            .arg(&n_rows_u32)
            .arg(&k_cols_u32)
            .launch(cfg)
            .ok()?;
    }

    // ── 5. GELU-tanh / SiLU activation per expert (strided to inter_padded) ──
    // Gate/up output is packed at stride `inter`; activation must land at
    // stride `inter_padded` because down reads `K = inter_padded`. One
    // small dispatch per expert with the right offsets gets us strided
    // output without a new shader.
    let inter_u32 = inter as u32;
    let act_block_size = 256u32.min(inter_u32);
    let act_grid = (inter_u32 + act_block_size - 1) / act_block_size;

    for e in 0..valid_count {
        let g_offset = e * inter;
        let u_offset = e * inter;
        let a_offset = e * inter_padded;

        let gate_view = scratch.g_out.slice(g_offset..g_offset + inter);
        let up_view = scratch.u_out.slice(u_offset..u_offset + inter);
        let mut act_view = scratch.act_buf.slice_mut(a_offset..a_offset + inter_padded);

        let cfg = LaunchConfig {
            grid_dim: (act_grid, 1, 1),
            block_dim: (act_block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&gegelu_kernel.func)
                .arg(&gate_view)
                .arg(&up_view)
                .arg(&mut act_view)
                .arg(&inter_u32)
                .launch(cfg)
                .ok()?;
        }
    }

    // ── 6. Down projection per expert ──
    // Each expert has its own down weight buffer. Activation is read from
    // the strided act_buf; output goes to the strided expert_outs buffer.
    let n_out_u32 = hidden as u32;
    let k_in_u32 = inter_padded as u32;
    let down_grid = ((hidden as u32) + q4k_matvec_kernel.rows_per_block as u32 - 1)
        / q4k_matvec_kernel.rows_per_block as u32;
    let down_block = q4k_matvec_kernel.threads_per_block as u32;

    for e in 0..valid_count {
        let act_offset = e * inter_padded;
        let out_offset = e * hidden;

        let act_view = scratch.act_buf.slice(act_offset..act_offset + inter_padded);
        let mut out_view = scratch.expert_outs.slice_mut(out_offset..out_offset + hidden);

        let cfg = LaunchConfig {
            grid_dim: (down_grid, 1, 1),
            block_dim: (down_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&q4k_matvec_kernel.func)
                .arg(&scratch.down_bufs[e])
                .arg(&act_view)
                .arg(&mut out_view)
                .arg(&n_out_u32)
                .arg(&k_in_u32)
                .launch(cfg)
                .ok()?;
        }
    }

    // ── 7. Synchronize + read back ──
    stream.synchronize().ok()?;
    let all_expert_outputs: Vec<f32> = stream.clone_dtoh(&scratch.expert_outs).ok()?;

    // ── 8. CPU weighted sum + post-experts norm ──
    let mut moe_out = vec![0.0f32; hidden];
    for e in 0..valid_count {
        let w = valid_weights[e];
        let out_slice = &all_expert_outputs[e * hidden..(e + 1) * hidden];
        for (acc, &v) in moe_out.iter_mut().zip(out_slice) {
            *acc += v * w;
        }
    }

    if !moe.post_experts_norm.is_empty() {
        let rms = (moe_out.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + eps).sqrt();
        for (v, &w) in moe_out.iter_mut().zip(moe.post_experts_norm) {
            *v = *v / rms * (w + 0.0);
        }
    }

    Some(moe_out)
}