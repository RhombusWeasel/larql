//! CUDA decode pipeline — single-token autoregressive generation.
//!
//! Mirrors `metal::decode/mod.rs` but with CUDA kernel dispatches.
//! All GPU operations stay on the device between layers; only the final
//! result is read back to host after all layers complete.
//!
//! Phase 2: MoE layers cause a GPU sync + CPU router + GPU expert dispatch
//! + CPU weighted-sum round-trip. This matches the Metal backend's MoE
//! interleave pattern (commit → wait → expert → restart).

mod encode_attn;
mod encode_ffn;
mod encode_post_ffn;
mod encode_qkv;
pub mod profile;
pub mod setup;

use cudarc::driver::CudaSlice;

use crate::cpu::ops::outer_combine::{apply_layer_scalar_in_place, outer_post_norm_residual};
use crate::cuda::CudaBackend;
use crate::cuda::ops::kv_cache::{CudaKVCache, KVLayerShape};
use crate::cuda::ops::moe_dispatch::{gpu_moe_dispatch_with_scratch, CudaMoeScratch};
use crate::pipeline::FullPipelineLayer;

pub(crate) const DEFAULT_KV_CACHE_MAX_SEQ: usize = 4096;

impl CudaBackend {
    /// Create a KV cache for decode mode with uniform per-layer dims.
    pub fn create_kv_cache(
        &self,
        num_layers: usize,
        max_seq: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Option<CudaKVCache> {
        CudaKVCache::new_uniform(&self.ctx, num_layers, max_seq, num_kv_heads, head_dim)
    }

    /// Create a KV cache with per-layer shapes (Gemma 4 alternating attention).
    pub fn create_kv_cache_per_layer(
        &self,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) -> Option<CudaKVCache> {
        let layer_shapes: Vec<KVLayerShape> = shapes
            .iter()
            .map(|&(num_kv, hd)| KVLayerShape { num_kv_heads: num_kv, head_dim: hd })
            .collect();
        CudaKVCache::new(&self.ctx, layer_shapes, max_seq)
    }

    fn kv_shapes_for_layers(
        layers: &[FullPipelineLayer<'_>],
    ) -> Vec<(usize, usize)> {
        layers
            .iter()
            .map(|layer| (layer.num_kv_heads, layer.head_dim))
            .collect()
    }

    pub fn ensure_kv_cache_for_layers<'a>(
        &'a self,
        cache: &'a mut Option<CudaKVCache>,
        layers: &[FullPipelineLayer<'_>],
        max_seq: usize,
    ) -> &'a mut CudaKVCache {
        let shapes = Self::kv_shapes_for_layers(layers);
        let needs_rebuild = cache
            .as_ref()
            .is_none_or(|kv| kv.has_shape_mismatch(&shapes));

        if needs_rebuild {
            *cache = self.create_kv_cache_per_layer(&shapes, max_seq);
        }

        let kv = cache.as_mut().expect("KV cache initialized above");
        kv.grow_to_shapes(&self.ctx, &shapes, max_seq);
        kv
    }

    /// Decode one token through all layers with KV cache.
    ///
    /// Single-stream pipeline: all layers execute on the same CUDA stream.
    /// MoE layers cause a sync + readback + GPU expert dispatch cycle.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token_pipeline(
        &self,
        kv_cache: &mut CudaKVCache,
        layers: &[FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
    ) -> Option<Vec<f32>> {
        let stream = self.stream();

        // Allocate scratch buffers on GPU
        let mut scratch = setup::DecodeScratch::new(
            &self.ctx, layers, x, hidden, inter, q_dim, kv_dim,
        )?;

        let num_layers = scratch.num_layers;
        let has_moe = scratch.has_moe;

        // Warm up MoE scratch if any layer has MoE
        if has_moe {
            let mut moe_guard = self.moe_scratch.lock().unwrap();
            if let Some(shape) = layers
                .iter()
                .find_map(|l| l.moe.as_ref())
                .map(|m| (m.top_k, hidden, m.intermediate_size))
            {
                let needs_alloc = match moe_guard.as_ref() {
                    Some(s) => !s.matches_shape(shape.0, shape.1, shape.2),
                    None => true,
                };
                if needs_alloc {
                    *moe_guard = CudaMoeScratch::new(&self.ctx, shape.0, shape.1, shape.2);
                }
            }
        }

        for l in 0..num_layers {
            let layer = &layers[l];
            let layer_q_dim = layer.num_q_heads * layer.head_dim;
            let layer_kv_dim = layer.num_kv_heads * layer.head_dim;

            // ── Step 1: Input norm + QKV projection ──
            self.encode_input_norm_and_qkv(
                scratch.wq_q4k_bufs[l].as_ref(),
                scratch.wk_q4k_bufs[l].as_ref(),
                scratch.wv_q4k_bufs[l].as_ref(),
                encode_qkv::QkvBufs {
                    h_in: if l == 0 {
                        &scratch.h_init
                    } else if l % 2 == 1 {
                        &scratch.h_a
                    } else {
                        &scratch.h_b
                    },
                    input_norm: layer.input_norm,
                    norm_out: &mut scratch.norm_out,
                    q_out: &mut scratch.q_out,
                    k_out: &mut scratch.k_out,
                    v_out: &mut scratch.v_out,
                },
                encode_qkv::QkvDims {
                    hidden,
                    layer_q_dim,
                    layer_kv_dim,
                    eps: layer.eps,
                    norm_offset: layer.norm_offset,
                },
            )?;

            // ── Steps 1.5–5: Attention block ──
            self.encode_attention_block(
                layer,
                kv_cache,
                l,
                scratch.wo_q4k_bufs[l].as_ref(),
                encode_attn::AttnBufs {
                    h_buf: if l == 0 {
                        &scratch.h_init
                    } else if l % 2 == 1 {
                        &scratch.h_a
                    } else {
                        &scratch.h_b
                    },
                    q_out: &mut scratch.q_out,
                    k_out: &mut scratch.k_out,
                    v_out: &mut scratch.v_out,
                    attn_out_buf: &mut scratch.attn_out_buf,
                    o_out_buf: &mut scratch.o_out_buf,
                    ffn_norm_out: &mut scratch.ffn_norm_out,
                    h_post_attn: &mut scratch.h_post_attn,
                    normed_scratch: &mut scratch.normed_scratch,
                    norm_out: &mut scratch.norm_out,
                },
                encode_attn::AttnDims {
                    hidden,
                    layer_q_dim,
                },
            )?;

            // ── Check for MoE or remote-FFN layer ──
            if layer.moe.is_some() || layer.ffn_is_remote {
                // MoE interleave: synchronize GPU, read back h_post_attn,
                // run CPU router + GPU expert dispatch + CPU weighted sum,
                // then write result back to GPU.
                self.handle_moe_layer(
                    layer,
                    l,
                    &mut scratch,
                    hidden,
                    inter,
                )?;
                continue;
            }

            // ── Steps 6–7: FFN + post-FFN residual (non-MoE path) ──
            self.encode_ffn_step(
                layer,
                scratch.gate_q4k_bufs[l].as_ref(),
                scratch.up_q4k_bufs[l].as_ref(),
                scratch.down_q4k_bufs[l].as_ref(),
                encode_ffn::FfnBufs {
                    ffn_norm_out: &scratch.ffn_norm_out,
                    gate_out_scratch: &mut scratch.gate_out_scratch,
                    up_out: &mut scratch.up_out,
                    act_buf: &mut scratch.act_buf,
                    down_out: &mut scratch.down_out,
                },
                encode_ffn::FfnDims {
                    hidden,
                    inter,
                    inter_padded: scratch.inter_padded,
                },
            )?;

            let use_fused_post_ffn = !matches!(
                std::env::var("LARQL_FUSED_POST_FFN_NORM").as_deref(),
                Ok("0") | Ok("false") | Ok("off") | Ok("no")
            );

            // Determine the target ping-pong buffer for this layer
            {
                let new_h: &mut CudaSlice<f32> = if l % 2 == 0 {
                    &mut scratch.h_a
                } else {
                    &mut scratch.h_b
                };

                self.encode_post_ffn_residual(
                    layer,
                    encode_post_ffn::PostFfnBufs {
                        down_out: &scratch.down_out,
                        h_post_attn: &scratch.h_post_attn,
                        new_h,
                        normed_scratch: &mut scratch.normed_scratch,
                    },
                    hidden,
                    use_fused_post_ffn,
                )?;
            }

            // ── Step 8: Optional layer scalar (non-MoE layers) ──
            if layer.layer_scalar != 0.0 && layer.layer_scalar != 1.0 && !has_moe {
                let new_h: &mut CudaSlice<f32> = if l % 2 == 0 {
                    &mut scratch.h_a
                } else {
                    &mut scratch.h_b
                };
                crate::cuda::ops::attention::encode_scale_vector_in_place(
                    stream,
                    &self.scale_vector,
                    new_h,
                    layer.layer_scalar,
                    hidden as u32,
                )?;
            }

            // Dump per-layer hidden state (with pos to avoid overwrites)
            if let Ok(dir) = std::env::var("LARQL_CUDA_DUMP_LAYERS") {
                stream.synchronize().ok();
                let new_h: &CudaSlice<f32> = if l % 2 == 0 { &scratch.h_a } else { &scratch.h_b };
                if let Ok(data) = stream.clone_dtoh(new_h) {
                    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let pos = kv_cache.layers.first().map(|l| l.seq_len).unwrap_or(0);
                    let path = format!("{dir}/cuda_layer_{l:02}_pos{pos}.f32");
                    let _ = std::fs::write(&path, &bytes);
                }
            }
        }

        // Synchronize and read back the final result
        stream.synchronize().ok()?;

        // Dump final hidden state to file for comparison with CPU
        if let Ok(dir) = std::env::var("LARQL_CUDA_DUMP_DIR") {
            let h_dev: &CudaSlice<f32> = if num_layers % 2 == 0 { &scratch.h_b } else { &scratch.h_a };
            if let Ok(data) = stream.clone_dtoh(h_dev) {
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                let path = format!("{dir}/cuda_decode_out.f32");
                let _ = std::fs::write(&path, &bytes);
            }
        }

        let h_buf: &CudaSlice<f32> = if num_layers % 2 == 0 {
            &scratch.h_b
        } else {
            &scratch.h_a
        };
        let result = stream.clone_dtoh(h_buf).ok()?;
        Some(result)
    }

    /// Diagnostic dump: write both f32 and quantized projection outputs
    /// to disk for element-by-element comparison.
    ///
    /// Triggered by `LARQL_CUDA_DUMP_DIR=/some/path`. When a Q4_K env var
    /// is set, the corresponding projection's f32 and quantized outputs
    /// are written as `{dump_dir}/cuda_{proj}_{format}_l{layer}_p{pos}.f32`.
    ///
    /// Call after `stream.synchronize()` to ensure buffers are ready.
    #[allow(dead_code)]
    fn dump_projection_diag(
        &self,
        tag: &str,
        f32_out: &CudaSlice<f32>,
        q4k_out: Option<&CudaSlice<f32>>,
        layer_idx: usize,
        kv_cache: &CudaKVCache,
    ) {
        if let Ok(dir) = std::env::var("LARQL_CUDA_DUMP_DIR") {
            let stream = self.stream();
            let pos = kv_cache.layers.first().map(|l| l.seq_len).unwrap_or(0);

            // Dump f32 reference
            if let Ok(data) = stream.clone_dtoh(f32_out) {
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                let path = format!("{dir}/cuda_{tag}_f32_l{layer_idx:02}_p{pos}.f32");
                let _ = std::fs::write(&path, &bytes);
            }

            // Dump quantized output if available
            if let Some(q4k) = q4k_out {
                if let Ok(data) = stream.clone_dtoh(q4k) {
                    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let path = format!("{dir}/cuda_{tag}_q4k_l{layer_idx:02}_p{pos}.f32");
                    let _ = std::fs::write(&path, &bytes);
                }
            }
        }
    }

    /// Handle a MoE or remote-FFN layer.
    ///
    /// This mirrors Metal's `handle_moe_interleave` pattern:
    /// 1. Synchronize GPU (ensure h_post_attn is ready)
    /// 2. Read h_post_attn back to host
    /// 3. Run dense FFN on GPU (if not remote)
    /// 4. Run CPU router + GPU expert dispatch
    /// 5. CPU weighted sum + outer combine
    /// 6. Write combined result back to GPU
    #[allow(clippy::too_many_arguments)]
    fn handle_moe_layer(
        &self,
        layer: &FullPipelineLayer<'_>,
        layer_idx: usize,
        scratch: &mut setup::DecodeScratch,
        hidden: usize,
        inter: usize,
    ) -> Option<()> {
        let stream = self.stream();

        // Synchronize to ensure h_post_attn is ready
        stream.synchronize().ok()?;

        // Read h_post_attn back to host
        let h_post_attn: Vec<f32> = stream.clone_dtoh(&scratch.h_post_attn).ok()?;

        // Compute new_h depending on whether this is hybrid MoE or remote-only
        let new_h_vec: Vec<f32> = if layer.ffn_is_remote {
            // Remote-FFN: the entire FFN is provided externally.
            // For now, just use h_post_attn (the remote FFN callback
            // is handled at a higher level in the trait impl).
            h_post_attn
        } else {
            // Hybrid MoE (dense FFN + MoE experts):
            // Run dense FFN on GPU, then add MoE output on top

            // Run dense FFN on GPU
            self.encode_ffn_step(
                layer,
                scratch.gate_q4k_bufs[layer_idx].as_ref(),
                scratch.up_q4k_bufs[layer_idx].as_ref(),
                scratch.down_q4k_bufs[layer_idx].as_ref(),
                encode_ffn::FfnBufs {
                    ffn_norm_out: &scratch.ffn_norm_out,
                    gate_out_scratch: &mut scratch.gate_out_scratch,
                    up_out: &mut scratch.up_out,
                    act_buf: &mut scratch.act_buf,
                    down_out: &mut scratch.down_out,
                },
                encode_ffn::FfnDims {
                    hidden,
                    inter,
                    inter_padded: scratch.inter_padded,
                },
            )?;

            // Post-FFN residual (unfused for MoE layers)
            {
                let new_h: &mut CudaSlice<f32> = if layer_idx % 2 == 0 {
                    &mut scratch.h_a
                } else {
                    &mut scratch.h_b
                };

                self.encode_post_ffn_residual(
                    layer,
                    encode_post_ffn::PostFfnBufs {
                        down_out: &scratch.down_out,
                        h_post_attn: &scratch.h_post_attn,
                        new_h,
                        normed_scratch: &mut scratch.normed_scratch,
                    },
                    hidden,
                    false,
                )?;
            }

            // Synchronize and read back the dense-FFN result
            stream.synchronize().ok()?;
            let dense_result = if layer_idx % 2 == 0 {
                stream.clone_dtoh(&scratch.h_a).ok()?
            } else {
                stream.clone_dtoh(&scratch.h_b).ok()?
            };

            // Run MoE expert dispatch if the layer has MoE weights
            if let Some(ref moe) = layer.moe {
                let moe_out = {
                    let mut moe_guard = self.moe_scratch.lock().unwrap();
                    let moe_scratch = moe_guard.as_mut().expect("MoE scratch should be allocated");

                    gpu_moe_dispatch_with_scratch(
                        &self.ctx,
                        stream,
                        &self.q4k_ffn_gate_up,
                        &self.gegelu_gelu_tanh,
                        &self.q4k_matvec,
                        &h_post_attn,
                        moe,
                        layer.eps,
                        layer.activation,
                        moe_scratch,
                    )?
                };

                // Dense + MoE: new_h already has (dense_ffn + h_post_attn),
                // add MoE output on top.
                let mut combined = dense_result;
                for (i, v) in moe_out.iter().enumerate() {
                    combined[i] += v;
                }

                // Apply outer combine (post-FFN norm for MoE layers)
                if layer.moe_combined_output_norm {
                    let h1_plus_h2: Vec<f32> = combined
                        .iter()
                        .zip(h_post_attn.iter())
                        .map(|(&c, &ha)| c - ha)
                        .collect();
                    outer_post_norm_residual(
                        &h_post_attn,
                        &h1_plus_h2,
                        layer.moe_outer_post_norm.or(layer.post_ffn_norm),
                        layer.norm_offset,
                        layer.eps,
                    )
                } else {
                    combined
                }
            } else {
                // No MoE — shouldn't reach this branch since outer if
                // checks `layer.moe.is_some() || layer.ffn_is_remote`
                dense_result
            }
        };

        // Apply whole-layer scalar (Gemma 4)
        let mut new_h_final = new_h_vec;
        apply_layer_scalar_in_place(&mut new_h_final, layer.layer_scalar);

        // Write the result back to GPU
        {
            let new_h_dev: &mut CudaSlice<f32> = if layer_idx % 2 == 0 {
                &mut scratch.h_a
            } else {
                &mut scratch.h_b
            };
            stream.memcpy_htod(&new_h_final, new_h_dev).ok()?;
        }

        Some(())
    }
}