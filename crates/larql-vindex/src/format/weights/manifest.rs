//! Shared manifest entry shape used by `write_q4k` to emit
//! `attn_weights_q4k_manifest.json`, `interleaved_q4k_manifest.json`,
//! and `down_features_q4k_manifest.json`. Pulled out so the loaders in
//! `index/storage/ffn_store.rs` can deserialise into a typed struct
//! instead of poking `serde_json::Value` with string keys — silently
//! `unwrap_or(0)`'ing missing fields was a real footgun (a renamed
//! field would silently produce zero-byte slices).
//!
//! One entry describes one tensor's slice within its `.bin` file:
//! - `offset` / `length` — byte range within the file
//! - `format` — quant tag, must round-trip via `quant::registry::lookup`
//! - `shape` — `[rows, padded_cols]` after `pad_rows_to_256`
//! - `key` — original tensor name (for human inspection / round-trip)
//!
//! The fields are deliberately laid out so the JSON shape matches what
//! the previous (string-keyed) loaders expected — switching loaders to
//! typed deserialisation is a no-op on existing on-disk manifests.

use serde::{Deserialize, Serialize};

use super::write_q4k::QuantBlockFormat;

/// One manifest entry describing one Q4_K/Q6_K-encoded tensor slice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Q4kManifestEntry {
    pub key: String,
    pub shape: Vec<usize>,
    pub format: QuantBlockFormat,
    pub offset: u64,
    pub length: u64,
}

impl Q4kManifestEntry {
    /// Padded row stride in elements (second dim of `shape`). Returns
    /// `None` when the manifest entry has fewer than 2 dimensions —
    /// caller decides whether to error or fall back to `hidden_size`.
    pub fn padded_width(&self) -> Option<usize> {
        self.shape.get(1).copied()
    }

    /// Format tag as the on-disk string (`"Q4_K"` / `"Q6_K"`).
    /// `quant::registry::lookup` consumes this directly.
    pub fn format_tag(&self) -> &'static str {
        match self.format {
            QuantBlockFormat::Q4K => "Q4_K",
            QuantBlockFormat::Q6K => "Q6_K",
        }
    }
}
