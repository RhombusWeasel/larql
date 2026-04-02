extern crate blas_src;

pub mod attention;
pub mod capture;
pub mod error;
pub mod ffn;
pub mod forward;
pub mod graph_ffn;
pub mod model;
pub mod route_ffn;
pub mod residual;
pub mod tokenizer;
pub mod trace;
pub mod vindex;
pub mod walker;

// Re-export dependencies for downstream crates.
pub use larql_models;
pub use larql_vindex;
pub use ndarray;
pub use safetensors;
pub use tokenizers;

// Re-export essentials at crate root.
pub use capture::{
    CaptureCallbacks, CaptureConfig, InferenceModel, TopKEntry, VectorFileHeader, VectorRecord,
};
pub use error::InferenceError;
pub use ffn::{FfnBackend, HighwayFfn, LayerFfnRouter, SparseFfn, WeightFfn};
pub use attention::AttentionWeights;
pub use forward::{
    calibrate_scalar_gains, capture_residuals, forward_to_layer, predict, predict_from_hidden,
    predict_from_hidden_with_ffn, predict_with_ffn, predict_with_router, predict_with_strategy,
    trace_forward, trace_forward_full, trace_forward_with_ffn, LayerAttentionCapture, LayerMode,
    PredictResult, TraceResult,
};
pub use graph_ffn::{GateIndex, IndexBuildCallbacks, SilentIndexCallbacks};
pub use ffn::experimental::cached::CachedFfn;
pub use ffn::experimental::clustered::{ClusteredFfn, ClusteredGateIndex};
pub use ffn::experimental::down_clustered::{DownClusteredFfn, DownClusteredIndex};
pub use ffn::experimental::entity_routed::EntityRoutedFfn;
pub use ffn::experimental::feature_list::FeatureListFfn;
pub use ffn::experimental::graph::GraphFfn;
pub use route_ffn::{RouteFfn, RouteGuidedFfn, RouteTable};
pub use trace::{
    trace_residuals, trace as trace_decomposed, AnswerWaypoint, LayerSummary,
    ResidualTrace, TraceNode, TracePositions, TraceStore, TraceWriter,
    BoundaryStore, BoundaryWriter,
    ContextStore, ContextWriter, ContextTier,
};
pub use vindex::WalkFfn;
pub use model::{load_model_dir, resolve_model_path, ModelWeights};
pub use tokenizer::{decode_token, load_tokenizer};

// Walker re-exports.
pub use walker::attention_walker::{AttentionLayerResult, AttentionWalker};
pub use walker::vector_extractor::{
    ExtractCallbacks, ExtractConfig, ExtractSummary, VectorExtractor,
};
pub use walker::weight_walker::{
    walk_model, LayerResult, LayerStats, WalkCallbacks, WalkConfig, WeightWalker,
};
