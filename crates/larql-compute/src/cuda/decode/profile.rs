//! Per-stage decode timing — CUDA event-based profiling.
//!
//! Mirrors `metal::decode::profile`. When `LARQL_PROFILE_SPLIT=1` is set,
//! the decode pipeline inserts `cudaStreamSynchronize()` boundaries between
//! attention and FFN stages. The resulting per-stage wall times land in a
//! thread-local cell so `decode_token_split_profile` can read them back.
//!
//! Granularity: attention vs full FFN block (gate+up → activation → down).
//! `down_ms` is reserved for the next-finer split.

/// Per-stage wall-clock decode timings in milliseconds.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProfileTimings {
    /// Wall time for attention: input norm → QKV → QK-norm → RoPE →
    /// KV-attend → O proj.
    pub attn_ms: f64,
    /// Wall time for FFN gate + up + activation.
    pub gate_up_ms: f64,
    /// Wall time for FFN down projection + post-FFN residual + scalar.
    /// Zero today; reserved for the next split.
    pub down_ms: f64,
}

/// True iff `LARQL_PROFILE_SPLIT=1` (or the legacy alias
/// `LARQL_DECODE_STAGE_TIMING=1`) is set in the environment.
pub fn split_profile_requested() -> bool {
    std::env::var("LARQL_PROFILE_SPLIT").is_ok()
        || std::env::var("LARQL_DECODE_STAGE_TIMING").is_ok()
}

thread_local! {
    /// Most recent per-stage timing recorded when `LARQL_PROFILE_SPLIT=1`.
    static LAST_SPLIT_TIMINGS: std::cell::Cell<Option<ProfileTimings>> =
        const { std::cell::Cell::new(None) };
}

/// Store the latest per-stage timing for the current thread.
#[allow(dead_code)]
pub(crate) fn store_last_split_timings(t: ProfileTimings) {
    LAST_SPLIT_TIMINGS.with(|cell| cell.set(Some(t)));
}

/// Take and clear the most recent per-stage timing recorded on the
/// current thread. Returns `None` if no profiling was done.
pub fn take_last_split_timings() -> Option<ProfileTimings> {
    LAST_SPLIT_TIMINGS.with(|cell| cell.take())
}

impl ProfileTimings {
    /// Sum across the three buckets — the whole-token cost.
    pub fn total_ms(&self) -> f64 {
        self.attn_ms + self.gate_up_ms + self.down_ms
    }

    /// Format a `[profile-split] …` line.
    pub fn format_summary(&self, num_layers: usize) -> String {
        let total = self.total_ms();
        let pct = |v: f64| if total > 0.0 { v / total * 100.0 } else { 0.0 };
        let per_layer = if num_layers > 0 { total / num_layers as f64 } else { 0.0 };
        format!(
            "[profile-split] {num_layers} layers — \
             attn={:.2}ms ({:.0}%)  gate+up={:.2}ms ({:.0}%)  \
             down={:.2}ms ({:.0}%)  total={:.2}ms ({per_layer:.3}ms/layer)",
            self.attn_ms,
            pct(self.attn_ms),
            self.gate_up_ms,
            pct(self.gate_up_ms),
            self.down_ms,
            pct(self.down_ms),
            total,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_ms_sums_buckets() {
        let p = ProfileTimings {
            attn_ms: 1.5,
            gate_up_ms: 2.5,
            down_ms: 1.0,
        };
        assert!((p.total_ms() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn format_summary_handles_zero_total() {
        let p = ProfileTimings::default();
        let s = p.format_summary(34);
        assert!(s.contains("total=0.00ms"));
        assert!(s.contains("34 layers"));
    }
}