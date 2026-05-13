//! KV cache GPU memory management + attention dispatch for CUDA decode pipeline.
//!
//! Mirrors `metal::ops::kv_cache`. Manages per-layer K/V device buffers
//! for autoregressive decode, provides append and attend operations.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::cuda::kernel::CudaKernel;
use crate::cuda::ops::kv_attention::SHORT_ATTENTION_SPAN;

/// Per-layer KV cache geometry.
#[derive(Clone)]
pub struct KVLayerShape {
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// Per-layer KV cache data on the GPU.
pub struct LayerKVCache {
    pub k_cache: CudaSlice<f32>,
    pub v_cache: CudaSlice<f32>,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub max_seq_len: usize,
}

impl LayerKVCache {
    fn new(ctx: &Arc<CudaContext>, max_seq: usize, num_kv_heads: usize, head_dim: usize) -> Option<Self> {
        let stream = ctx.default_stream();
        let size = max_seq * num_kv_heads * head_dim;
        Some(Self {
            k_cache: stream.alloc_zeros::<f32>(size).ok()?,
            v_cache: stream.alloc_zeros::<f32>(size).ok()?,
            num_kv_heads,
            head_dim,
            seq_len: 0,
            max_seq_len: max_seq,
        })
    }
}

/// GPU-hosted KV cache for autoregressive decode.
///
/// Stores K and V tensors in device memory, indexed by
/// `(layer, position, head, dim)`. Pre-allocated up to `max_seq_len`
/// positions to avoid per-token allocation.
pub struct CudaKVCache {
    /// Per-layer KV cache data.
    pub layers: Vec<LayerKVCache>,
    /// Current global sequence length (all layers share this).
    pub seq_len: usize,
    /// Maximum pre-allocated sequence length.
    pub max_seq_len: usize,
}

impl CudaKVCache {
    /// Create a new KV cache with the given per-layer shapes and max length.
    pub fn new(
        ctx: &Arc<CudaContext>,
        shapes: Vec<KVLayerShape>,
        max_seq_len: usize,
    ) -> Option<Self> {
        let layers = shapes
            .iter()
            .map(|shape| LayerKVCache::new(ctx, max_seq_len, shape.num_kv_heads, shape.head_dim))
            .collect::<Option<Vec<_>>>()?;

        Some(Self {
            layers,
            seq_len: 0,
            max_seq_len: max_seq_len,
        })
    }

    /// Create a uniform KV cache (all layers same shape).
    pub fn new_uniform(
        ctx: &Arc<CudaContext>,
        num_layers: usize,
        max_seq: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Option<Self> {
        let shapes = (0..num_layers)
            .map(|_| KVLayerShape { num_kv_heads, head_dim })
            .collect();
        Self::new(ctx, shapes, max_seq)
    }

    /// Current sequence length.
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Truncate the cache to a previous length.
    pub fn truncate(&mut self, len: usize) {
        if len <= self.seq_len {
            self.seq_len = len;
            for layer in &mut self.layers {
                layer.seq_len = len;
            }
        }
    }

    /// Reset for a new prompt.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for layer in &mut self.layers {
            layer.seq_len = 0;
        }
    }

    /// Check if shapes match expected per-layer geometry.
    pub fn has_shape_mismatch(&self, shapes: &[(usize, usize)]) -> bool {
        if self.layers.len() < shapes.len() {
            return true;
        }
        self.layers.iter().zip(shapes.iter()).any(
            |(layer, &(expected_kv, expected_hd))| {
                layer.num_kv_heads != expected_kv || layer.head_dim != expected_hd
            },
        )
    }

    /// Grow the cache to cover additional shapes.
    pub fn grow_to_shapes(
        &mut self,
        ctx: &Arc<CudaContext>,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) -> Option<()> {
        while self.layers.len() < shapes.len() {
            let (num_kv, hd) = shapes[self.layers.len()];
            self.layers.push(LayerKVCache::new(
                ctx,
                max_seq,
                num_kv,
                hd,
            )?);
        }
        self.max_seq_len = self.max_seq_len.max(max_seq);
        Some(())
    }
}

impl KVLayerShape {
    /// Create a LayerKVCache from this shape, allocating GPU buffers.
    pub fn into_layer(&self, ctx: &Arc<CudaContext>, max_seq: usize) -> Option<LayerKVCache> {
        LayerKVCache::new(ctx, max_seq, self.num_kv_heads, self.head_dim)
    }
}

/// Compute the attention span for sliding window attention.
pub fn attention_span(t: u32, window_size: u32) -> u32 {
    if window_size > 0 && t > window_size {
        window_size
    } else {
        t
    }
}

/// KV Append: copy new K and V data into the cache at the current position.
pub fn encode_kv_append(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    new_k: &CudaSlice<f32>,
    new_v: &CudaSlice<f32>,
    k_cache: &mut CudaSlice<f32>,
    v_cache: &mut CudaSlice<f32>,
    pos: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Option<()> {
    let total = num_kv_heads * head_dim;
    let threads = 256u32;
    let blocks = (total + threads - 1) / threads;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(new_k)
            .arg(new_v)
            .arg(k_cache)
            .arg(v_cache)
            .arg(&pos)
            .arg(&num_kv_heads)
            .arg(&head_dim)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// KV Attend: compute softmax(Q * K^T / sqrt(d)) * V for a single decode step.
pub fn encode_kv_attend(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    long_kernel: Option<&CudaKernel>,
    q: &CudaSlice<f32>,
    k_cache: &CudaSlice<f32>,
    v_cache: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    seq_len: u32,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    scale: f32,
    window_size: u32,
) -> Option<()> {
    let effective_kernel = if attention_span(seq_len, window_size) > SHORT_ATTENTION_SPAN {
        long_kernel.unwrap_or(kernel)
    } else {
        kernel
    };

    let threads = 256u32;
    let blocks = num_q_heads;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 128,
    };

    unsafe {
        stream
            .launch_builder(&effective_kernel.func)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(out)
            .arg(&seq_len)
            .arg(&head_dim)
            .arg(&num_q_heads)
            .arg(&num_kv_heads)
            .arg(&scale)
            .arg(&window_size)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

/// Fused KV append + attend.
#[allow(dead_code)]
pub fn encode_kv_append_attend_fused(
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    q: &CudaSlice<f32>,
    k_cache: &mut CudaSlice<f32>,
    v_cache: &mut CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    new_k: &CudaSlice<f32>,
    new_v: &CudaSlice<f32>,
    seq_len: u32,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    scale: f32,
    window_size: u32,
) -> Option<()> {
    let threads = 256u32;
    let blocks = num_q_heads;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 128,
    };

    unsafe {
        stream
            .launch_builder(&kernel.func)
            .arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(out)
            .arg(&seq_len)
            .arg(&head_dim)
            .arg(&num_q_heads)
            .arg(&num_kv_heads)
            .arg(&scale)
            .arg(&window_size)
            .arg(new_k)
            .arg(new_v)
            .launch(cfg)
            .ok()?;
    }

    Some(())
}

#[cfg(test)]
mod tests {
    const SHAPE_SMALL: (usize, usize) = (2, 64);
    const SHAPE_LARGE: (usize, usize) = (4, 128);

    #[test]
    fn shape_mismatch_detects_conflicting_existing_layer() {
        assert!(!super::shape_pairs_have_mismatch(
            &[SHAPE_SMALL],
            &[SHAPE_SMALL, SHAPE_LARGE]
        ));
        assert!(super::shape_pairs_have_mismatch(
            &[SHAPE_SMALL],
            &[SHAPE_LARGE]
        ));
    }
}

#[allow(dead_code)]
fn shape_pairs_have_mismatch(existing: &[(usize, usize)], expected: &[(usize, usize)]) -> bool {
    existing.iter().zip(expected.iter()).any(
        |(&(actual_num_kv, actual_head_dim), &(expected_num_kv, expected_head_dim))| {
            actual_num_kv != expected_num_kv || actual_head_dim != expected_head_dim
        },
    )
}