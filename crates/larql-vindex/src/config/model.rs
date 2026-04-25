//! Model-architecture config carried in `index.json` so the
//! architecture can be reconstructed without the original
//! `config.json`.
//!
//! Carved out of the monolithic `config/types.rs` in the 2026-04-25
//! round-2 cleanup.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct VindexModelConfig {
    pub model_type: String,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// MoE configuration (None for dense models).
    #[serde(default)]
    pub moe: Option<MoeConfig>,

    // ── Gemma 4 per-layer attention geometry ──
    // All optional for backward compatibility with existing vindexes.

    /// Head dimension for global (full) attention layers. If None, all layers use head_dim.
    /// Gemma 4: 512 for global layers, head_dim (256) for sliding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_head_dim: Option<usize>,
    /// Number of KV heads for global attention layers. If None, all layers use num_kv_heads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_global_kv_heads: Option<usize>,
    /// Fraction of head_dim to apply RoPE to (0.0–1.0). If None, full rotation.
    /// Gemma 4 global layers: 0.25.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f64>,
    /// Sliding window pattern: every Nth layer is full attention.
    /// Gemma 4: 6 (layers 5, 11, 17, ... are full).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window_pattern: Option<usize>,
    /// Explicit per-layer type array (e.g., ["sliding_attention", "full_attention", ...]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_types: Option<Vec<String>>,
    /// Whether value projection shares key projection (K=V).
    #[serde(default)]
    pub attention_k_eq_v: bool,
    /// Number of layers at the end that share KV from earlier layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_kv_shared_layers: Option<usize>,
    /// Per-layer embedding dimension (PLE). 0 or None = no PLE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_layer_embed_dim: Option<usize>,
    /// RoPE base for local/sliding window layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_local_base: Option<f64>,
    /// Query pre-attention scalar (overrides 1/sqrt(head_dim)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_pre_attn_scalar: Option<f64>,
    /// Final-logit tanh softcap (Gemma 2/3/4: 30.0). Applied to logits
    /// immediately before softmax in `logits_to_predictions`. Omitting it
    /// leaves logits uncapped — on E2B this peaked the softmax on the
    /// wrong token (observed: "Paris" → "hyperparameters").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcapping: Option<f64>,
}

/// MoE (Mixture of Experts) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Number of experts per layer.
    pub num_experts: usize,
    /// Number of experts selected per token (top-K routing).
    pub top_k: usize,
    /// Whether there's a shared expert always active (DeepSeek V2/V3).
    #[serde(default)]
    pub shared_expert: bool,
    /// Router type (e.g., "top_k_softmax", "gemma4_top_k_softmax").
    #[serde(default = "default_router_type")]
    pub router_type: String,
    /// Per-expert intermediate (hidden) dimension.
    /// Differs from the dense FFN intermediate_size in hybrid models (Gemma 4 A4B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_intermediate_size: Option<usize>,
    /// Hybrid MoE: dense MLP and expert block coexist in each layer, outputs summed.
    /// True for Gemma 4 A4B. False for pure MoE (Mixtral, DeepSeek).
    #[serde(default)]
    pub hybrid: bool,
}

fn default_router_type() -> String {
    "top_k_softmax".to_string()
}

