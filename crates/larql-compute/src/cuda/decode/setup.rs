//! Per-decode-token scratch and weight-buffer pre-allocation.
//!
//! [`DecodeScratch`] is built once at the top of `decode_token` and
//! threaded through the per-layer loop. It owns per-layer weight data
//! on GPU and per-stage scratch buffers reused across all layers.
//!
//! Weights are uploaded as raw packed Q4_K/Q6_K bytes — no CPU dequant.
//! The quantized CUDA kernels operate directly on packed data, matching
//! the Metal backend's approach. Peak packed memory per decode_token:
//! ~200 MB for 34 layers (vs ~12.8 GB f32).

use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use crate::pipeline::FullPipelineLayer;

pub(super) struct DecodeScratch {
    // ── Hidden-state ping-pong + layer-0 input ──
    pub h_init: CudaSlice<f32>,
    pub h_a: CudaSlice<f32>,
    pub h_b: CudaSlice<f32>,

    // ── Per-layer packed weight data (raw Q4_K/Q6_K bytes) ──
    pub wq_q4k_bufs: Vec<Option<CudaSlice<u8>>>,
    pub wk_q4k_bufs: Vec<Option<CudaSlice<u8>>>,
    pub wv_q4k_bufs: Vec<Option<CudaSlice<u8>>>,
    pub wo_q4k_bufs: Vec<Option<CudaSlice<u8>>>,
    pub gate_q4k_bufs: Vec<Option<CudaSlice<u8>>>,
    pub up_q4k_bufs: Vec<Option<CudaSlice<u8>>>,
    pub down_q4k_bufs: Vec<Option<CudaSlice<u8>>>,

    // ── Per-stage scratch (one buffer, reused every layer) ──
    pub norm_out: CudaSlice<f32>,
    pub q_out: CudaSlice<f32>,
    pub k_out: CudaSlice<f32>,
    pub v_out: CudaSlice<f32>,
    pub attn_out_buf: CudaSlice<f32>,
    pub o_out_buf: CudaSlice<f32>,
    pub h_post_attn: CudaSlice<f32>,
    pub ffn_norm_out: CudaSlice<f32>,
    pub gate_out_scratch: CudaSlice<f32>,
    pub up_out: CudaSlice<f32>,
    pub act_buf: CudaSlice<f32>,
    pub down_out: CudaSlice<f32>,
    pub normed_scratch: CudaSlice<f32>,

    // ── Constants derived from `layers` ──
    pub inter_padded: usize,
    pub num_layers: usize,
    pub has_moe: bool,
}

/// Upload raw packed Q4_K/Q6_K weight bytes to GPU (no dequantization).
fn upload_packed<F>(
    stream: &Arc<CudaStream>,
    layers: &[FullPipelineLayer<'_>],
    get_data: F,
) -> Vec<Option<CudaSlice<u8>>>
where
    F: for<'a> Fn(&'a FullPipelineLayer<'_>) -> (&'a [u8], usize, usize),
{
    layers
        .iter()
        .enumerate()
        .map(|(li, l)| {
            let (qdata, rows, cols) = get_data(l);
            if qdata.is_empty() {
                None
            } else {
                stream.clone_htod(qdata).ok()
            }
        })
        .collect()
}

impl DecodeScratch {
    pub(super) fn new(
        ctx: &Arc<CudaContext>,
        layers: &[FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Option<Self> {
        let stream = ctx.default_stream();
        let num_layers = layers.len();
        let inter_padded = inter.div_ceil(larql_models::quant::ggml::Q4_K_BLOCK_ELEMS)
            * larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;

        // Max dimension across all layers for heterogeneous attention (Gemma 4)
        let max_q_dim = layers
            .iter()
            .map(|l| l.num_q_heads * l.head_dim)
            .max()
            .unwrap_or(q_dim);
        let max_kv_dim = layers
            .iter()
            .map(|l| l.num_kv_heads * l.head_dim)
            .max()
            .unwrap_or(kv_dim);

        // Upload raw packed weights to GPU — no CPU dequant
        let wq_q4k_bufs = upload_packed(&stream, layers, |l| {
            (l.wq.data, l.num_q_heads * l.head_dim, hidden)
        });
        let wk_q4k_bufs = upload_packed(&stream, layers, |l| {
            (l.wk.data, l.num_kv_heads * l.head_dim, hidden)
        });
        let wv_q4k_bufs = upload_packed(&stream, layers, |l| {
            (l.wv.data, l.num_kv_heads * l.head_dim, hidden)
        });
        let wo_q4k_bufs = upload_packed(&stream, layers, |l| {
            (l.wo.data, hidden, l.num_q_heads * l.head_dim)
        });
        let gate_q4k_bufs = upload_packed(&stream, layers, |l| {
            (l.gate.data, inter, hidden)
        });
        let up_q4k_bufs = upload_packed(&stream, layers, |l| {
            (l.up.data, inter, hidden)
        });
        let down_q4k_bufs = upload_packed(&stream, layers, |l| {
            (l.down.data, hidden, inter_padded)
        });

        // Upload input to GPU
        let h_init = stream.clone_htod(x).ok()?;

        // Allocate scratch buffers
        let h_a = stream.alloc_zeros::<f32>(hidden).ok()?;
        let h_b = stream.alloc_zeros::<f32>(hidden).ok()?;
        let norm_out = stream.alloc_zeros::<f32>(hidden).ok()?;
        let q_out = stream.alloc_zeros::<f32>(max_q_dim).ok()?;
        let k_out = stream.alloc_zeros::<f32>(max_kv_dim).ok()?;
        let v_out = stream.alloc_zeros::<f32>(max_kv_dim).ok()?;
        let attn_out_buf = stream.alloc_zeros::<f32>(max_q_dim).ok()?;
        let o_out_buf = stream.alloc_zeros::<f32>(hidden).ok()?;
        let h_post_attn = stream.alloc_zeros::<f32>(hidden).ok()?;
        let ffn_norm_out = stream.alloc_zeros::<f32>(hidden).ok()?;
        let gate_out_scratch = stream.alloc_zeros::<f32>(inter).ok()?;
        let up_out = stream.alloc_zeros::<f32>(inter).ok()?;
        let act_buf = stream.alloc_zeros::<f32>(inter_padded).ok()?;
        let down_out = stream.alloc_zeros::<f32>(hidden).ok()?;
        let normed_scratch = stream.alloc_zeros::<f32>(hidden).ok()?;

        let has_moe = layers.iter().any(|l| l.moe.is_some() || l.ffn_is_remote);

        Some(Self {
            h_init, h_a, h_b,
            wq_q4k_bufs, wk_q4k_bufs, wv_q4k_bufs, wo_q4k_bufs,
            gate_q4k_bufs, up_q4k_bufs, down_q4k_bufs,
            norm_out, q_out, k_out, v_out,
            attn_out_buf, o_out_buf,
            h_post_attn, ffn_norm_out,
            gate_out_scratch, up_out, act_buf, down_out,
            normed_scratch,
            inter_padded, num_layers, has_moe,
        })
    }
}
