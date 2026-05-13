//! Attention operations — RoPE, QK-norm, V-norm, GEGLU, residual+norm dispatch.
//!
//! These are the per-stage GPU operations for the decode pipeline's
//! attention block (steps 1.5–5 in the Metal nomenclature).
//!
//! All functions take device buffers (`CudaSlice<f32>`) and work without
//! host readback. This is the on-device variant used by the decode pipeline.

use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;

use cudarc::driver::CudaSlice;

/// Apply RoPE (rotary position encoding) to Q heads in-place.
///
/// `q_out` is modified in-place. Each head has `head_dim` elements;
/// the first `rotary_dim` dimensions get the rotation applied.
pub fn encode_rope_q(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    q_out: &mut CudaSlice<f32>,
    head_dim: u32,
    rope_base: f32,
    pos: u32,
    rotary_dim: u32,
    num_q_heads: u32,
) -> Option<()> {
    let rope_pairs = rotary_dim / 2;
    let total_threads = num_q_heads * rope_pairs;
    let threads = 256u32.min(total_threads);
    let blocks = ((total_threads as u32) + threads - 1) / threads;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(q_out)
            .arg(&head_dim)
            .arg(&rope_base)
            .arg(&pos)
            .arg(&rotary_dim)
            .arg(&num_q_heads)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// Apply RoPE to K heads in-place.
pub fn encode_rope_k(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    k_out: &mut CudaSlice<f32>,
    head_dim: u32,
    rope_base: f32,
    pos: u32,
    rotary_dim: u32,
    num_kv_heads: u32,
) -> Option<()> {
    let rope_pairs = rotary_dim / 2;
    let total_threads = num_kv_heads * rope_pairs;
    let threads = 256u32.min(total_threads);
    let blocks = ((total_threads as u32) + threads - 1) / threads;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(k_out)
            .arg(&head_dim)
            .arg(&rope_base)
            .arg(&pos)
            .arg(&rotary_dim)
            .arg(&num_kv_heads)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// Per-head RMSNorm on Q and K heads (two separate buffers, matching Metal).
///
/// `q_out` and `k_out` are separate buffers — the kernel selects the right
/// buffer based on head index.
pub fn encode_qk_norm(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    q_out: &mut CudaSlice<f32>,
    k_out: &mut CudaSlice<f32>,
    q_norm_w: &CudaSlice<f32>,
    k_norm_w: &CudaSlice<f32>,
    head_dim: u32,
    num_q_heads: u32,
    eps: f32,
    offset: f32,
    total_heads: u32,
) -> Option<()> {
    let cfg = LaunchConfig {
        grid_dim: (total_heads, 1, 1),
        block_dim: (256.min(head_dim), 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(q_out)
            .arg(k_out)
            .arg(q_norm_w)
            .arg(k_norm_w)
            .arg(&head_dim)
            .arg(&num_q_heads)
            .arg(&eps)
            .arg(&offset)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// Parameter-free V-norm: normalize each KV head by its RMS.
pub fn encode_v_norm(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    v_out: &mut CudaSlice<f32>,
    head_dim: u32,
    eps: f32,
    num_kv_heads: u32,
) -> Option<()> {
    let threads = next_power_of_2(head_dim).min(256);
    let cfg = LaunchConfig {
        grid_dim: (num_kv_heads, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(v_out)
            .arg(&head_dim)
            .arg(&eps)
            .arg(&num_kv_heads)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// GEGLU SiLU activation: `out[i] = silu(gate[i]) * up[i]`.
pub fn encode_geglu_silu(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n: u32,
) -> Option<()> {
    let threads = 256u32;
    let blocks = ((n as u32) + threads - 1) / threads;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(gate)
            .arg(up)
            .arg(out)
            .arg(&n)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// GEGLU GELU-tanh activation: `out[i] = gelu_tanh(gate[i]) * up[i]`.
#[allow(dead_code)]
pub fn encode_geglu_gelu_tanh(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n: u32,
) -> Option<()> {
    let threads = 256u32;
    let blocks = ((n as u32) + threads - 1) / threads;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(gate)
            .arg(up)
            .arg(out)
            .arg(&n)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// Fused residual + RMSNorm + store: computes h_post_attn = h + attn_out,
/// then ffn_norm_out = rmsnorm(h_post_attn, weight, eps, offset).
pub fn encode_residual_norm_store(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    h: &CudaSlice<f32>,
    attn_out: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    ffn_norm_out: &mut CudaSlice<f32>,
    h_post_attn: &mut CudaSlice<f32>,
    len: u32,
    eps: f32,
    offset: f32,
) -> Option<()> {
    let threads = 256u32.min(len);
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(h)
            .arg(attn_out)
            .arg(weight)
            .arg(ffn_norm_out)
            .arg(h_post_attn)
            .arg(&len)
            .arg(&eps)
            .arg(&offset)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// Triple-fused post-attn: h_post_attn = h + o_out,
/// ffn_norm_out = pre_ffn_norm(rmsnorm(h+o_out, post_attn_norm, eps, offset)).
#[allow(dead_code)]
pub fn encode_post_attn_residual_norm_store(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    h: &CudaSlice<f32>,
    o_out: &CudaSlice<f32>,
    post_attn_norm: &CudaSlice<f32>,
    pre_ffn_norm: &CudaSlice<f32>,
    ffn_norm_out: &mut CudaSlice<f32>,
    h_post_attn: &mut CudaSlice<f32>,
    len: u32,
    eps: f32,
    offset: f32,
) -> Option<()> {
    let threads = 256u32.min(len);
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(h)
            .arg(o_out)
            .arg(post_attn_norm)
            .arg(pre_ffn_norm)
            .arg(ffn_norm_out)
            .arg(h_post_attn)
            .arg(&len)
            .arg(&eps)
            .arg(&offset)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// Post-FFN norm + residual add:
/// new_h = h_post_attn + rmsnorm(down_out, norm_weight, eps, offset)
/// Optionally applies layer_scalar.
pub fn encode_post_ffn_norm_residual_add(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    down_out: &CudaSlice<f32>,
    h_post_attn: &CudaSlice<f32>,
    norm_weight: &CudaSlice<f32>,
    new_h: &mut CudaSlice<f32>,
    len: u32,
    eps: f32,
    offset: f32,
    layer_scalar: f32,
) -> Option<()> {
    let threads = 256u32.min(len);
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(down_out)
            .arg(h_post_attn)
            .arg(norm_weight)
            .arg(new_h)
            .arg(&len)
            .arg(&eps)
            .arg(&offset)
            .arg(&layer_scalar)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// On-device residual add: `out[i] = a[i] + b[i]`.
/// This variant takes device buffers instead of host slices.
pub fn encode_residual_add_on_device(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    len: u32,
) -> Option<()> {
    let threads = 256u32;
    let blocks = (len + threads - 1) / threads;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(a)
            .arg(b)
            .arg(out)
            .arg(&len)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// On-device scale vector: `out[i] = input[i] * scalar`.
/// When `input` and `out` are the same buffer, the operation is in-place
/// (safe because each thread reads before writing, with no cross-thread deps).
pub fn encode_scale_vector_on_device(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    input: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    scalar: f32,
    len: u32,
) -> Option<()> {
    let threads = 256u32;
    let blocks = (len + threads - 1) / threads;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(input)
            .arg(out)
            .arg(&len)
            .arg(&scalar)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// In-place scale vector: `buf[i] *= scalar`.
///
/// Safe because the scale_vector kernel performs element-wise reads
/// before writes with no cross-thread dependencies.
pub fn encode_scale_vector_in_place(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    buf: &mut CudaSlice<f32>,
    scalar: f32,
    len: u32,
) -> Option<()> {
    let threads = 256u32;
    let blocks = (len + threads - 1) / threads;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    // In-place: pass the same buffer as both input and output.
    // The kernel does element-wise `out[i] = in[i] * scalar`,
    // which is safe in-place since each GPU thread reads before writing.
    // We use a raw pointer to create both & and &mut references
    // from the same buffer, which is sound because CudaSlice is a GPU
    // handle (not Rust-owned data) and the GPU kernel semantics guarantee
    // in-place safety.
    let buf_raw = buf as *mut CudaSlice<f32>;
    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(&*buf_raw)           // input: reborrow as immutable
            .arg(&mut *buf_raw)        // out: reborrow as mutable
            .arg(&len)
            .arg(&scalar)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// On-device RMS norm (no host readback, operates on device buffers).
pub fn encode_rms_norm_on_device(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    x: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    len: u32,
    eps: f32,
    offset: f32,
) -> Option<()> {
    let block_size = len.min(1024);

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(x)
            .arg(weight)
            .arg(out)
            .arg(&len)
            .arg(&eps)
            .arg(&offset)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

fn next_power_of_2(n: u32) -> u32 {
    let mut v = n;
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v += 1;
    v
}