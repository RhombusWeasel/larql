//! Post-FFN norm + residual add dispatch for CUDA decode pipeline.
//!
//! Mirrors `metal::decode::encode_post_ffn`.

use cudarc::driver::{CudaSlice};

use crate::cuda::CudaBackend;
use crate::cuda::ops::attention;
use crate::pipeline::FullPipelineLayer;

pub(super) struct PostFfnBufs<'a> {
    pub down_out: &'a CudaSlice<f32>,
    pub h_post_attn: &'a CudaSlice<f32>,
    pub new_h: &'a mut CudaSlice<f32>,
    pub normed_scratch: &'a mut CudaSlice<f32>,
}

impl CudaBackend {
    /// Encode post-FFN residual + optional norm.
    ///
    /// Computes: new_h = h_post_attn + norm(down_out)
    /// Where norm is the post-FFN norm (if has_post_norms) or plain
    /// residual add (if not).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_post_ffn_residual(
        &self,
        layer: &FullPipelineLayer<'_>,
        bufs: PostFfnBufs<'_>,
        hidden: usize,
        use_fused: bool,
    ) -> Option<()> {
        let stream = self.stream();
        let hidden_u32 = hidden as u32;
        let eps = layer.eps;
        let norm_offset = layer.norm_offset;

        // Post-norm (Gemma): norm down_out with post_ffn_norm, then add
        // Pre-norm (Llama): plain h_post_attn + down_out
        if layer.has_post_norms {
            let post_ffn_norm = layer.post_ffn_norm.unwrap_or(layer.post_attn_norm);
            let post_ffn_dev = stream.clone_htod(post_ffn_norm).ok()?;

            if use_fused {
                attention::encode_post_ffn_norm_residual_add(
                    stream, &self.post_ffn_norm_residual_add,
                    bufs.down_out, bufs.h_post_attn, &post_ffn_dev,
                    bufs.new_h, hidden_u32, eps, norm_offset, layer.layer_scalar,
                )?;
            } else {
                attention::encode_rms_norm_on_device(
                    stream, &self.rms_norm,
                    bufs.down_out, &post_ffn_dev, bufs.normed_scratch,
                    hidden_u32, eps, norm_offset,
                )?;
                attention::encode_residual_add_on_device(
                    stream, &self.residual_add,
                    bufs.h_post_attn, bufs.normed_scratch, bufs.new_h, hidden_u32,
                )?;
                if layer.layer_scalar != 0.0 && layer.layer_scalar != 1.0 {
                    attention::encode_scale_vector_in_place(
                        stream, &self.scale_vector, bufs.new_h, layer.layer_scalar, hidden_u32,
                    )?;
                }
            }
        } else {
            // Pre-norm: plain residual add
            attention::encode_residual_add_on_device(
                stream, &self.residual_add,
                bufs.h_post_attn, bufs.down_out, bufs.new_h, hidden_u32,
            )?;
        }

        Some(())
    }
}