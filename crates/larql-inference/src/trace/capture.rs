//! Trace capture — decomposed forward pass recording attn and FFN deltas.

use ndarray::Array2;

use crate::attention::AttentionWeights;
use crate::ffn::{FfnBackend, WeightFfn};
use crate::forward::{dot_proj, add_bias};
use crate::model::ModelWeights;
use crate::residual::{rms_norm, rms_norm_heads};
use larql_models::NormType;

use super::types::*;

/// Which positions to capture.
pub enum TracePositions {
    Last,
    All,
    Positions(Vec<usize>),
}

/// Capture a complete residual stream trace.
pub fn trace_residuals(
    weights: &ModelWeights,
    token_ids: &[u32],
    positions: TracePositions,
    capture_attention: bool,
    ffn: &dyn FfnBackend,
) -> ResidualTrace {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;

    let pos_list: Vec<usize> = match positions {
        TracePositions::Last => vec![seq_len - 1],
        TracePositions::All => (0..seq_len).collect(),
        TracePositions::Positions(ref ps) => ps.clone(),
    };

    let mut h = embed_tokens_raw(weights, token_ids);
    let mut nodes = Vec::new();
    let mut attention_captures = Vec::new();
    let zero = vec![0.0f32; hidden];

    // Embedding layer (-1)
    for &p in &pos_list {
        nodes.push(TraceNode {
            layer: -1, position: p,
            residual: h.row(p).to_vec(),
            attn_delta: zero.clone(),
            ffn_delta: zero.clone(),
        });
    }

    // Transformer layers
    for layer in 0..num_layers {
        let pre = h.clone();

        let (h_post_attn, attn_weights) = match run_attention_decomposed(
            weights, &h, layer, capture_attention,
        ) {
            Some(r) => r,
            None => continue,
        };

        let h_post_ffn = run_ffn_decomposed(weights, &h_post_attn, layer, ffn);

        for &p in &pos_list {
            let attn_delta: Vec<f32> = h_post_attn.row(p).iter()
                .zip(pre.row(p).iter())
                .map(|(&a, &b)| a - b)
                .collect();
            let ffn_delta: Vec<f32> = h_post_ffn.row(p).iter()
                .zip(h_post_attn.row(p).iter())
                .map(|(&a, &b)| a - b)
                .collect();

            nodes.push(TraceNode {
                layer: layer as i32, position: p,
                residual: h_post_ffn.row(p).to_vec(),
                attn_delta, ffn_delta,
            });
        }

        if let Some(w) = attn_weights {
            attention_captures.push((layer, w));
        }
        h = h_post_ffn;
    }

    let tokens: Vec<String> = token_ids.iter()
        .map(|&id| format!("t{}", id))
        .collect();

    ResidualTrace {
        prompt: String::new(), tokens, token_ids: token_ids.to_vec(),
        n_layers: num_layers, hidden_size: hidden,
        nodes, attention: attention_captures,
    }
}

/// Convenience: trace with default WeightFfn.
pub fn trace(
    weights: &ModelWeights, token_ids: &[u32], positions: TracePositions,
) -> ResidualTrace {
    let ffn = WeightFfn { weights };
    trace_residuals(weights, token_ids, positions, false, &ffn)
}

// ── Internal: decomposed layer execution ──

fn embed_tokens_raw(weights: &ModelWeights, token_ids: &[u32]) -> Array2<f32> {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let scale = weights.arch.embed_scale();
    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok_id) in token_ids.iter().enumerate() {
        let row = weights.embed.row(tok_id as usize);
        for j in 0..hidden { h[[i, j]] = row[j] * scale; }
    }
    h
}

fn apply_norm(
    weights: &ModelWeights, x: &Array2<f32>, weight_key: &str, norm_offset: f32,
) -> Array2<f32> {
    match weights.arch.norm_type() {
        NormType::LayerNorm => {
            let bias_key = weight_key.replace(".weight", ".bias");
            crate::residual::layer_norm(x, weights.vectors.get(weight_key), weights.vectors.get(&bias_key))
        }
        _ => rms_norm(x, weights.vectors.get(weight_key), norm_offset),
    }
}

fn run_attention_decomposed(
    weights: &ModelWeights, h: &Array2<f32>, layer: usize, capture_attention: bool,
) -> Option<(Array2<f32>, Option<AttentionWeights>)> {
    let arch = &*weights.arch;
    let head_dim = weights.head_dim;
    let num_q = weights.num_q_heads;
    let num_kv = weights.num_kv_heads;
    let reps = num_q / num_kv;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else { arch.attention_scale() };
    let seq_len = h.shape()[0];
    let norm_offset = arch.norm_weight_offset();

    let h_norm = apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_offset);

    let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
    let w_k = weights.tensors.get(&arch.attn_k_key(layer))?;
    let w_v = weights.tensors.get(&arch.attn_v_key(layer))?;
    let w_o = weights.tensors.get(&arch.attn_o_key(layer))?;

    let mut q_full = dot_proj(&h_norm, w_q);
    let mut k_full = dot_proj(&h_norm, w_k);
    let mut v_full = dot_proj(&h_norm, w_v);

    if let Some(bias) = arch.attn_q_bias_key(layer).and_then(|k| weights.vectors.get(&k)) { add_bias(&mut q_full, bias); }
    if let Some(bias) = arch.attn_k_bias_key(layer).and_then(|k| weights.vectors.get(&k)) { add_bias(&mut k_full, bias); }
    if let Some(bias) = arch.attn_v_bias_key(layer).and_then(|k| weights.vectors.get(&k)) { add_bias(&mut v_full, bias); }

    let qk_offset = weights.arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 { qk_offset } else { norm_offset };
    let q_normed = match arch.attn_q_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    let k_normed = match arch.attn_k_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(norm_w) => rms_norm_heads(&k_full, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full,
    };

    let layer_rope_base = arch.rope_base_for_layer(layer);
    let q_rope = crate::attention::apply_rope(&q_normed, num_q, head_dim, layer_rope_base);
    let k_rope = crate::attention::apply_rope(&k_normed, num_kv, head_dim, layer_rope_base);

    let softcap = arch.attn_logit_softcapping();
    let (attn_out, attn_weights) = crate::attention::gqa_attention_with_weights(
        &q_rope, &k_rope, &v_full, num_q, head_dim, reps, scale, seq_len,
        capture_attention, softcap,
    );
    let mut attn_projected = dot_proj(&attn_out, w_o);
    if let Some(bias) = arch.attn_o_bias_key(layer).and_then(|k| weights.vectors.get(&k)) { add_bias(&mut attn_projected, bias); }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = apply_norm(weights, &attn_projected, &arch.post_attention_layernorm_key(layer), norm_offset);
        if res_mult != 1.0 { h + &(&normed * res_mult) } else { h + &normed }
    } else if res_mult != 1.0 {
        h + &(&attn_projected * res_mult)
    } else {
        h + &attn_projected
    };

    Some((h_post_attn, attn_weights))
}

fn run_ffn_decomposed(
    weights: &ModelWeights, h_post_attn: &Array2<f32>, layer: usize, ffn: &dyn FfnBackend,
) -> Array2<f32> {
    let norm_offset = weights.arch.norm_weight_offset();
    let arch = &*weights.arch;

    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_ffn = match pre_ffn_key {
        Some(key) => apply_norm(weights, h_post_attn, &key, norm_offset),
        None => rms_norm(h_post_attn, None, norm_offset),
    };

    let ffn_out = ffn.forward(layer, &h_ffn);

    let res_mult = arch.residual_multiplier();
    if arch.has_post_norms() {
        let normed = match arch.post_feedforward_layernorm_key(layer) {
            Some(key) => apply_norm(weights, &ffn_out, &key, norm_offset),
            None => rms_norm(&ffn_out, None, norm_offset),
        };
        if res_mult != 1.0 { h_post_attn + &(&normed * res_mult) } else { h_post_attn + &normed }
    } else if res_mult != 1.0 {
        h_post_attn + &(&ffn_out * res_mult)
    } else {
        h_post_attn + &ffn_out
    }
}
