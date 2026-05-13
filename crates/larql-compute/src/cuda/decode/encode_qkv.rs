//! Step 1 of the CUDA decode pipeline: input norm + Q/K/V projection.
//!
//! Uses quantized kernels (`q4k_matvec`) operating directly on packed
//! weight bytes on the GPU. No host dequantization needed.

use cudarc::driver::CudaSlice;
use crate::cuda::CudaBackend;
use crate::cuda::ops::{attention, q4k_matvec, q6k_matvec};

pub(super) struct QkvBufs<'a> {
    pub h_in: &'a CudaSlice<f32>,
    pub input_norm: &'a [f32],
    pub norm_out: &'a mut CudaSlice<f32>,
    pub q_out: &'a mut CudaSlice<f32>,
    pub k_out: &'a mut CudaSlice<f32>,
    pub v_out: &'a mut CudaSlice<f32>,
}

pub(super) struct QkvDims {
    pub hidden: usize,
    pub layer_q_dim: usize,
    pub layer_kv_dim: usize,
    pub eps: f32,
    pub norm_offset: f32,
}

impl CudaBackend {
    /// Encode input norm + QKV projection using quantized kernels on device.
    pub(super) fn encode_input_norm_and_qkv(
        &self,
        wq_q4k_buf: Option<&CudaSlice<u8>>,
        wk_q4k_buf: Option<&CudaSlice<u8>>,
        wv_q4k_buf: Option<&CudaSlice<u8>>,
        bufs: QkvBufs<'_>,
        dims: QkvDims,
    ) -> Option<()> {
        let stream = self.stream();
        let QkvDims { hidden, layer_q_dim, layer_kv_dim, eps, norm_offset } = dims;

        // Step 1: Input RMS norm
        let norm_weight_dev = stream.clone_htod(bufs.input_norm).ok()?;
        attention::encode_rms_norm_on_device(
            stream, &self.rms_norm,
            bufs.h_in, &norm_weight_dev, bufs.norm_out,
            hidden as u32, eps, norm_offset,
        )?;

        // Step 2: QKV projection — quantized kernels
        q4k_matvec::encode_q4k_matvec_on_device(
            stream, &self.q4k_matvec,
            wq_q4k_buf?, bufs.norm_out, bufs.q_out, layer_q_dim, hidden,
        )?;
        q4k_matvec::encode_q4k_matvec_on_device(
            stream, &self.q4k_matvec,
            wk_q4k_buf?, bufs.norm_out, bufs.k_out, layer_kv_dim, hidden,
        )?;
        q6k_matvec::encode_q6k_matvec_on_device(
            stream, &self.q6k_matvec,
            wv_q4k_buf?, bufs.norm_out, bufs.v_out, layer_kv_dim, hidden,
        )?;

        Some(())
    }
}
