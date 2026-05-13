//! Per-layer attention block — Steps 1.5 through 5 of the decode loop.
//!
//! Uses quantized kernels for O-projection. RoPE, QK-norm, V-norm,
//! KV-append/attend all operate on device buffers.

use cudarc::driver::CudaSlice;
use crate::cuda::CudaBackend;
use crate::cuda::ops::{attention, kv_cache, q4k_matvec};
use crate::cuda::ops::kv_cache::CudaKVCache;
use crate::pipeline::FullPipelineLayer;

pub(super) struct AttnBufs<'a> {
    pub h_buf: &'a CudaSlice<f32>,
    pub q_out: &'a mut CudaSlice<f32>,
    pub k_out: &'a mut CudaSlice<f32>,
    pub v_out: &'a mut CudaSlice<f32>,
    pub attn_out_buf: &'a mut CudaSlice<f32>,
    pub o_out_buf: &'a mut CudaSlice<f32>,
    pub ffn_norm_out: &'a mut CudaSlice<f32>,
    pub h_post_attn: &'a mut CudaSlice<f32>,
    pub normed_scratch: &'a mut CudaSlice<f32>,
    pub norm_out: &'a mut CudaSlice<f32>,
}

pub(super) struct AttnDims {
    pub hidden: usize,
    pub layer_q_dim: usize,
}

impl CudaBackend {
    /// Encode the per-layer attention block (Steps 1.5–5).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_attention_block(
        &self,
        layer: &FullPipelineLayer<'_>,
        kv_cache: &mut CudaKVCache,
        layer_idx: usize,
        wo_q4k_buf: Option<&CudaSlice<u8>>,
        bufs: AttnBufs<'_>,
        dims: AttnDims,
    ) -> Option<()> {
        let stream = self.stream();
        let AttnDims { hidden, layer_q_dim } = dims;
        let hidden_val = hidden as u32;
        let norm_offset = layer.norm_offset;
        let eps = layer.eps;
        let scale = layer.attn_scale;
        let layer_head_dim = layer.head_dim;
        let layer_num_q_heads = layer.num_q_heads;
        let layer_num_kv_heads = layer.num_kv_heads;
        let layer_rotary_dim = if layer.rotary_dim > 0 { layer.rotary_dim } else { layer_head_dim };
        let window_size = layer.sliding_window as u32;

        let pos = kv_cache.layers[layer_idx].seq_len as u32;
        let t_val = pos + 1;

        // ── Step 1.5: QK-norm (optional, Gemma 3/4) ──
        if let (Some(q_w), Some(k_w)) = (layer.q_norm_weight, layer.k_norm_weight) {
            let q_w_dev = stream.clone_htod(q_w).ok()?;
            let k_w_dev = stream.clone_htod(k_w).ok()?;
            let total_heads = (layer_num_q_heads + layer_num_kv_heads) as u32;
            attention::encode_qk_norm(
                stream, &self.qk_norm,
                bufs.q_out, bufs.k_out,
                &q_w_dev, &k_w_dev,
                layer_head_dim as u32,
                layer_num_q_heads as u32,
                eps, layer.qk_norm_offset,
                total_heads,
            )?;
        }

        // ── Step 2: RoPE on Q and K heads ──
        attention::encode_rope_q(
            stream, &self.rope_at_pos_q,
            bufs.q_out, layer_head_dim as u32,
            layer.rope_base, pos,
            layer_rotary_dim as u32,
            layer_num_q_heads as u32,
        )?;
        attention::encode_rope_k(
            stream, &self.rope_at_pos_k,
            bufs.k_out, layer_head_dim as u32,
            layer.rope_base, pos,
            layer_rotary_dim as u32,
            layer_num_kv_heads as u32,
        )?;

        // ── Step 3: V-norm (optional, Gemma 4) ──
        if layer.has_v_norm {
            attention::encode_v_norm(
                stream, &self.v_norm,
                bufs.v_out, layer_head_dim as u32,
                eps, layer_num_kv_heads as u32,
            )?;
        }

        // ── Step 4: KV-append + KV-attend ──
        let layer_cache = &mut kv_cache.layers[layer_idx];
        kv_cache::encode_kv_append(
            stream, &self.kv_append,
            bufs.k_out, bufs.v_out,
            &mut layer_cache.k_cache, &mut layer_cache.v_cache,
            pos,
            layer_num_kv_heads as u32,
            layer_head_dim as u32,
        )?;
        kv_cache::encode_kv_attend(
            self.ctx(), stream,
            &self.kv_attend, self.kv_attend_long.as_ref(),
            bufs.q_out,
            &layer_cache.k_cache, &layer_cache.v_cache,
            bufs.attn_out_buf,
            t_val,
            layer_head_dim as u32,
            layer_num_q_heads as u32,
            layer_num_kv_heads as u32,
            scale, window_size,
        )?;
        layer_cache.seq_len += 1;

        // ── Step 5a: O projection (quantized) ──
        q4k_matvec::encode_q4k_matvec_on_device(
            stream, &self.q4k_matvec,
            wo_q4k_buf?, bufs.attn_out_buf, bufs.o_out_buf,
            hidden, layer_q_dim,
        )?;

        // ── Step 5b: Residual + post-attn norm + ffn-input norm ──
        if layer.has_post_norms {
            let post_attn_norm_dev = stream.clone_htod(layer.post_attn_norm).ok()?;
            let pre_ffn_norm_dev = stream.clone_htod(
                layer.pre_ffn_norm.unwrap_or(layer.post_attn_norm)).ok()?;
            attention::encode_rms_norm_on_device(
                stream, &self.rms_norm,
                bufs.o_out_buf, &post_attn_norm_dev,
                bufs.normed_scratch, hidden_val, eps, norm_offset,
            )?;
            attention::encode_residual_add_on_device(
                stream, &self.residual_add,
                bufs.h_buf, bufs.normed_scratch,
                bufs.h_post_attn, hidden_val,
            )?;
            attention::encode_rms_norm_on_device(
                stream, &self.rms_norm,
                bufs.h_post_attn, &pre_ffn_norm_dev,
                bufs.ffn_norm_out, hidden_val, eps, norm_offset,
            )?;
        } else {
            let post_attn_norm_dev = stream.clone_htod(layer.post_attn_norm).ok()?;
            attention::encode_residual_norm_store(
                stream, &self.residual_norm_store,
                bufs.h_buf, bufs.o_out_buf,
                &post_attn_norm_dev,
                bufs.ffn_norm_out, bufs.h_post_attn,
                hidden_val, eps, norm_offset,
            )?;
        }

        Some(())
    }
}
