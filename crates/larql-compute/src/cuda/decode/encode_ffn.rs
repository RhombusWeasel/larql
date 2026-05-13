//! Step 6 of the decode pipeline: FFN dispatch using quantized kernels.
//!
//! Gate+up: fused `q4k_ffn_gate_up` kernel (Q4_K weights).
//! Down: GEGLU on device, then `q6k_matvec` (Q6_K weights, Gemma 3/4 convention).
//! For Q4_K down weights, a fused GEGLU+down kernel is used if available.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use crate::pipeline::{Activation, FullPipelineLayer};
use crate::cuda::CudaBackend;
use crate::cuda::ops::{attention, q4k_ffn, q6k_matvec, norm};

pub(super) struct FfnBufs<'a> {
    pub ffn_norm_out: &'a CudaSlice<f32>,
    pub gate_out_scratch: &'a mut CudaSlice<f32>,
    pub up_out: &'a mut CudaSlice<f32>,
    pub act_buf: &'a mut CudaSlice<f32>,
    pub down_out: &'a mut CudaSlice<f32>,
}

#[derive(Copy, Clone)]
pub(super) struct FfnDims {
    pub hidden: usize,
    pub inter: usize,
    pub inter_padded: usize,
}

impl CudaBackend {
    /// Encode FFN step using quantized kernels on device.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_ffn_step(
        &self,
        layer: &FullPipelineLayer<'_>,
        gate_q4k_buf: Option<&CudaSlice<u8>>,
        up_q4k_buf: Option<&CudaSlice<u8>>,
        down_q4k_buf: Option<&CudaSlice<u8>>,
        bufs: FfnBufs<'_>,
        dims: FfnDims,
    ) -> Option<()> {
        let FfnDims { hidden, inter, inter_padded } = dims;
        let inter_u32 = inter as u32;
        let stream = self.stream();

        if layer.is_gated() {
            // ── Fused gate+up (Q4_K) ──
            q4k_ffn::encode_q4k_ffn_gate_up_on_device(
                stream, &self.q4k_ffn_gate_up,
                gate_q4k_buf?, up_q4k_buf?,
                bufs.ffn_norm_out,
                bufs.gate_out_scratch, bufs.up_out,
                inter, hidden,
            )?;

            // ── Down projection: format-aware ──
            let down_is_q4k = layer.down.format == crate::pipeline::QuantFormat::Q4_K;
            if down_is_q4k {
                // Fused GEGLU+down kernel (Q4_K)
                match layer.activation {
                    Activation::GeluTanh => {
                        norm::encode_q4k_geglu_gelu_tanh_down_on_device(
                            stream, &self.q4k_geglu_gelu_tanh_down,
                            down_q4k_buf?,
                            bufs.gate_out_scratch, bufs.up_out, bufs.down_out,
                            hidden, inter_padded,
                        )?;
                    }
                    _ => {
                        norm::encode_q4k_geglu_silu_down_on_device(
                            stream, &self.q4k_geglu_silu_down,
                            down_q4k_buf?,
                            bufs.gate_out_scratch, bufs.up_out, bufs.down_out,
                            hidden, inter_padded,
                        )?;
                    }
                }
            } else {
                // Q6_K down (Gemma 3/4 convention): GEGLU then Q6_K matvec
                self.encode_geglu_on_device(
                    bufs.gate_out_scratch, bufs.up_out, bufs.act_buf,
                    inter_u32, layer.activation,
                )?;
                q6k_matvec::encode_q6k_matvec_on_device(
                    stream, &self.q6k_matvec,
                    down_q4k_buf?, bufs.act_buf, bufs.down_out,
                    hidden, inter_padded,
                )?;
            }
        } else {
            // Standard FFN: Q4_K matvec for up, activation, Q6_K matvec for down
            // (StarCoder2 / GPT-2 paths — less common)
            use crate::cuda::ops::q4k_matvec;
            q4k_matvec::encode_q4k_matvec_on_device(
                stream, &self.q4k_matvec,
                up_q4k_buf?, bufs.ffn_norm_out, bufs.up_out,
                inter, hidden,
            )?;
            self.encode_activation_on_device(
                bufs.up_out, bufs.act_buf, inter_u32, layer.activation,
            )?;
            q6k_matvec::encode_q6k_matvec_on_device(
                stream, &self.q6k_matvec,
                down_q4k_buf?, bufs.act_buf, bufs.down_out,
                hidden, inter_padded,
            )?;
        }

        Some(())
    }

    fn encode_geglu_on_device(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        act_buf: &mut CudaSlice<f32>,
        inter_u32: u32,
        activation: Activation,
    ) -> Option<()> {
        let stream = self.stream();
        let kernel = match activation {
            Activation::GeluTanh => &self.gegelu_gelu_tanh,
            _ => &self.gegelu_silu,
        };
        attention::encode_geglu_silu(stream, kernel, gate, up, act_buf, inter_u32)
    }

    fn encode_activation_on_device(
        &self,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        inter_u32: u32,
        activation: Activation,
    ) -> Option<()> {
        let stream = self.stream();
        let kernel = match activation {
            Activation::GeluTanh => &self.gelu_tanh,
            _ => &self.silu,
        };
        let threads = 256u32;
        let blocks = (inter_u32 + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&kernel.func)
                .arg(input)
                .arg(output)
                .arg(&inter_u32)
                .launch(cfg)
                .ok()?;
        }
        Some(())
    }
}
