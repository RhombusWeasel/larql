//! `QuantMatVec` impl for CudaBackend.
//!
//! Phase 4: adds q4k_matmul for prefill, q4_matvec_pair_batch for
//! prefill, and full format-aware routing.

use crate::backend::QuantMatVec;
use crate::cuda::CudaBackend;

impl QuantMatVec for CudaBackend {
    fn q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        self.encode_q4k_matvec(q4k_data, x, num_rows, hidden)
    }

    fn q4k_matvec_stride32(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        self.encode_q4k_matvec_stride32(q4k_data, x, num_rows, hidden)
    }

    fn q4k_matmul(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Option<Vec<f32>> {
        crate::cuda::ops::q4k_matmul::encode_q4k_matmul(
            &self.ctx, &self.stream, &self.q4k_matmul,
            q4k_data, x, num_rows, hidden, seq_len,
        )
    }

    fn q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        self.encode_q6k_matvec(q6k_data, x, num_rows, hidden)
    }

    fn q4_matvec_pair_batch(
        &self,
        gate_q4: &[u8],
        up_q4: &[u8],
        x_matrix: &[f32],
        seq_len: usize,
        num_rows: usize,
        hidden: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        crate::cuda::ops::q4_batched::pair_batch_q4k(
            &self.ctx, &self.stream, &self.q4k_matvec,
            gate_q4, up_q4, x_matrix, seq_len, num_rows, hidden,
        )
    }

    fn has_q4(&self) -> bool {
        // Q4_K kernels are available (q4k_matvec, q4k_ffn_gate_up, q4k_geglu_down)
        true
    }
}