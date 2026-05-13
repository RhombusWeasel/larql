//! KV attention kernel type constants.
//!
//! Re-exported from the shaders module for use in dispatch functions.

pub use crate::cuda::shaders::kv_attention::{
    KvAppendAttendFusedKernel, KvAppendKernel, KvAttendKernel, KvAttendLongKernel,
    SHORT_ATTENTION_SPAN,
};