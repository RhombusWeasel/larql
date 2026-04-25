//! `QuantMatVec` — quantised matrix × vector operations.
//!
//! Two entry points by intent:
//!
//! - [`Self::quant_matvec`] — **the convenience API.** Takes f32
//!   input, dispatches on [`crate::QuantFormat`], internally
//!   quantises to Q8 for Q4_0 / Q8_0. New callers should reach for
//!   this.
//! - [`Self::q4_matvec`] / [`Self::q4k_matvec`] / [`Self::q6k_matvec`]
//!   — **the pre-quantised-input fast path.** Hot decode paths
//!   pre-quantise the layer's input once and reuse it across many
//!   matvecs in that layer (gate, up, LM head, …). They take
//!   already-Q8 inputs and skip the per-call quantisation.
//!
//! Adding a new quant format = `QuantFormat` variant + match arm in
//! `quant_matvec` + per-format helper for the fast path.

use crate::QuantFormat;

/// Quantised matvec primitives.
pub trait QuantMatVec {
    /// Format-dispatched matvec.
    ///
    /// `out[N] = W[N, K] · x[K]`. Q4_K / Q4_KF / Q6_K consume f32 input
    /// directly; Q4_0 / Q8_0 internally re-quantise `x` to Q8 (per-32
    /// f32-scaled int8) before dispatching the kernel.
    ///
    /// Returns `None` if the backend doesn't implement the format.
    fn quant_matvec(
        &self,
        format: QuantFormat,
        weights: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        match format {
            QuantFormat::Q4_K | QuantFormat::Q4_KF => {
                self.q4k_matvec(weights, x, num_rows, hidden)
            }
            QuantFormat::Q6_K => self.q6k_matvec(weights, x, num_rows, hidden),
            QuantFormat::Q4_0 | QuantFormat::Q8_0 => {
                let (q8_x, q8_scales) =
                    crate::cpu::ops::q4_common::quantize_to_q8(x);
                self.q4_matvec(weights, &q8_x, &q8_scales, num_rows, hidden)
            }
        }
    }

    // ── Pre-quantised fast path ──
    //
    // These exist because the hot decode path pre-quantises its input
    // once and reuses it across many matvecs in a layer; the unified
    // `quant_matvec` re-quantises every call. Use these when the
    // caller already has Q8-quantised input on hand; reach for
    // `quant_matvec` otherwise.

    /// Q4_0 × Q8 matvec. `Some` if the backend supports Q4_0.
    fn q4_matvec(
        &self,
        _q4_data: &[u8], _q8_x: &[i8], _q8_scales: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Q4 vector-matrix: `out[K] = activation[N] @ Q4[N, K]`.
    fn q4_vecmat(
        &self,
        _activation: &[f32], _q4_data: &[u8],
        _intermediate: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Batched gate+up Q4 matvec for ALL seq positions in one submission.
    #[allow(clippy::type_complexity)]
    fn q4_matvec_pair_batch(
        &self,
        _gate_q4: &[u8], _up_q4: &[u8],
        _x_matrix: &[f32], _seq_len: usize,
        _num_rows: usize, _hidden: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> { None }

    /// Q4_K matvec: `scores[N] = Q4_K[N, K] @ f32_x[K]`.
    fn q4k_matvec(
        &self,
        _q4k_data: &[u8], _x: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Q6_K matvec: `scores[N] = Q6_K[N, K] @ f32_x[K]`.
    fn q6k_matvec(
        &self,
        _q6k_data: &[u8], _x: &[f32],
        _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> { None }

    /// Whether this backend implements any Q4 fused operation.
    fn has_q4(&self) -> bool { false }
}
