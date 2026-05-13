//! GPU dispatch — one file per operation.
//!
//! Each submodule contains Rust functions that allocate device buffers,
//! encode kernel launches, and read results back. Mirrors `metal::ops`.

pub mod attention;
pub mod f16_gemv;
pub mod f32_argmax;
pub mod f32_gemv;
pub mod f32_matmul;
pub mod kv_cache;
pub mod kv_attention;
pub mod layer_norm;
pub mod moe_dispatch;
pub mod norm;
pub mod prefill;
pub mod q4_batched;
pub mod q4_matvec;
pub mod q4k_ffn;
pub mod q4k_matmul;
pub mod q4k_matvec;
pub mod q6k_matvec;
pub mod q8_matvec;
pub mod quantize_q8;
pub mod rms_norm;

// Future submodules (Phase 4+):
// pub mod full_pipeline;