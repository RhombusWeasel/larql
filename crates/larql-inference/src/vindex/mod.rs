//! Local vector index — the full model as an in-memory KNN engine.
//!
//! Loads gate vectors and down-projection token metadata from extracted NDJSON
//! files (produced by `vector-extract`). Provides:
//!
//! 1. Gate KNN via BLAS matmul: residual × gate_vectors^T → top-K features
//! 2. Down token lookup: instant array access to precomputed output tokens
//!
//! This is the local equivalent of the SurrealDB walk — same vectors, same KNN,
//! same answer. No HTTP, no JSON serialisation, no round-trip. Array access.
//!
//! Memory: 34 layers × 10240 features × 2560 dim × 4 bytes = ~3.4GB for gate vectors.
//! Down metadata is lightweight (top_k token strings per feature).

mod build;
mod build_from_vectors;
pub mod config;
pub mod index;
mod load;
mod walk_ffn;
mod weights;

// ── Re-exports ──
// Everything that was public from the old vector_index.rs is re-exported here
// so that `use crate::vindex::*` (and the lib.rs `pub use vindex::...`) still work.

// Config types
pub use config::{VindexConfig, VindexLayerInfo, VindexModelConfig};

// Index types and traits
pub use index::{
    FeatureMeta, IndexLoadCallbacks, SilentLoadCallbacks, VectorIndex, WalkHit, WalkTrace,
};

// Build types and traits
pub use build::{IndexBuildCallbacks, SilentBuildCallbacks};

// Load functions
pub use load::{
    load_feature_labels, load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer,
};

// Walk FFN
pub use walk_ffn::WalkFfn;

// Weight functions
pub use weights::{find_tokenizer_path, load_model_weights_from_vindex, write_model_weights};
