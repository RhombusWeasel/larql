//! CUDA GPU compute backend — NVIDIA GPUs.
//!
//! All operations go through the [`ComputeBackend`] trait. CUDA-specific
//! optimisations: warp-shuffled reductions, shared memory tile caches,
//! NVRTC-compiled kernels (no nvcc build-time dependency).

pub mod buffers;
pub mod calibrate;
pub mod decode;
pub mod diag;
pub mod kernel;
pub mod ops;
pub mod shaders;
pub mod trait_impl;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use buffers::CudaBufferCache;
use cudarc::driver::{CudaContext, CudaModule, CudaStream};
use kernel::CudaKernel;
use ops::moe_dispatch::CudaMoeScratch;

/// CUDA GPU compute backend.
///
/// Kernels are added incrementally as they're implemented. Phase 0
/// provides f32_gemv only; Phase 1 adds argmax, rms_norm, and
/// quantized matmuls. Phase 2 adds the decode pipeline.
pub struct CudaBackend {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub bufs: CudaBufferCache,
    // ── Dense linear algebra (Phase 0) ──
    pub f32_gemv: CudaKernel,
    // ── Argmax / top-K (Phase 1) ──
    pub f32_argmax_partial: CudaKernel,
    #[allow(dead_code)] // Will be wired in Phase 4
    pub f32_topk_partial: CudaKernel,
    // ── Norm + activation (Phase 1) ──
    pub rms_norm: CudaKernel,
    pub residual_add: CudaKernel,
    pub scale_vector: CudaKernel,
    // ── Q4_K matvec variants (Phase 1) ──
    pub q4k_matvec: CudaKernel,
    pub q4k_matvec_8sg: CudaKernel,
    pub q4k_matvec_stride32: CudaKernel,
    // ── Q4_K FFN (Phase 1) ──
    pub q4k_ffn_gate_up: CudaKernel,
    pub q4k_geglu_silu_down: CudaKernel,
    pub q4k_geglu_gelu_tanh_down: CudaKernel,
    // ── Q6_K matvec variants (Phase 1) ──
    pub q6k_matvec: CudaKernel,
    pub q6k_matvec_8sg: CudaKernel,
    // ── Attention kernels (Phase 2) ──
    pub rope_at_pos_q: CudaKernel,
    pub rope_at_pos_k: CudaKernel,
    pub qk_norm: CudaKernel,
    pub kv_append: CudaKernel,
    pub kv_attend: CudaKernel,
    pub kv_attend_long: Option<CudaKernel>,
    pub kv_append_attend_fused: CudaKernel,
    pub v_norm: CudaKernel,
    // ── Activation kernels (Phase 2) ──
    pub gegelu_silu: CudaKernel,
    pub gegelu_gelu_tanh: CudaKernel,
    pub silu: CudaKernel,
    pub gelu_tanh: CudaKernel,
    // ── Residual+norm fused kernels (Phase 2) ──
    pub residual_norm_store: CudaKernel,
    pub post_attn_residual_norm_store: CudaKernel,
    pub post_ffn_norm_residual_add: CudaKernel,
    pub rms_norm_q8: CudaKernel,
    pub residual_norm_q8: CudaKernel,
    // ── Phase 3: Additional kernels ──
    pub layer_norm: CudaKernel,
    pub layer_norm_no_bias: CudaKernel,
    pub q4_matvec: CudaKernel,
    pub q4_vecmat: CudaKernel,
    pub q4_f32_matvec: CudaKernel,
    pub q8_matvec: CudaKernel,
    pub q8_qkv_proj: CudaKernel,
    pub q8_proj_rope: CudaKernel,
    pub rope_at_pos: CudaKernel,
    pub rope_apply: CudaKernel,
    pub rope_batched: CudaKernel,
    pub rope_batched_qk: CudaKernel,
    pub qk_norm_rope_fused: CudaKernel,
    pub causal_attention: CudaKernel,
    pub quantize_q8: CudaKernel,
    pub f16_gemv: CudaKernel,
    pub sgemm: CudaKernel,
    pub sgemm_transb: CudaKernel,
    // ── Prefill (Phase 4) ──
    pub q4k_matmul: CudaKernel,
    // ── Dynamic state ──
    pub kv_cache: std::sync::Mutex<Option<ops::kv_cache::CudaKVCache>>,
    pub moe_scratch: std::sync::Mutex<Option<CudaMoeScratch>>,
    pub flop_threshold: AtomicUsize,
}

/// Helper: extract a kernel from the module with a placeholder fallback.
/// Phase 2 kernels are compiled but may fail on some hardware; this
/// allows the backend to still initialise even if a kernel is missing,
/// though decode will fail at runtime if a required kernel isn't present.
fn kernel_or_placeholder(name: &'static str, module: &Arc<CudaModule>, fallback: &CudaKernel) -> CudaKernel {
    match module.load_function(name) {
        Ok(func) => CudaKernel { func, rows_per_block: 1, threads_per_block: 256, kernel_name: name },
        Err(_) => fallback.clone()
    }
}

impl CudaBackend {
    /// Create a CUDA backend. Returns `None` if no CUDA device is available
    /// or kernel compilation fails.
    pub fn new() -> Option<Self> {
        // 1. Initialise CUDA device (GPU 0)
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(_) => return None,
        };

        // 2. Get default stream
        let stream = ctx.default_stream();

        // 3. Compile all kernels from source via NVRTC
        let ptx = match shaders::compile_all_shaders() {
            Ok(p) => p,
            Err(_) => return None,
        };

        // 4. Load PTX module
        let module = match ctx.load_module(ptx) {
            Ok(m) => m,
            Err(_) => return None,
        };

        // 5. Create buffer cache
        let bufs = CudaBufferCache::new(&ctx);

        // 6. Extract kernel functions (Phase 0+1 required, Phase 2 optional)
        let f32_gemv = CudaKernel::from_tiled::<shaders::f32_gemv::F32GemvKernel>(&module)?;
        let f32_argmax_partial = CudaKernel::from_tiled::<shaders::f32_argmax::ArgmaxKernel>(&module)?;
        let f32_topk_partial = CudaKernel::from_tiled::<shaders::f32_argmax::TopKKernel>(&module)?;
        let rms_norm = CudaKernel::from_tiled::<shaders::rms_norm::RmsNormKernel>(&module)?;
        let residual_add = CudaKernel::from_tiled::<shaders::rms_norm::ResidualAddKernel>(&module)?;
        let scale_vector = CudaKernel::from_tiled::<shaders::rms_norm::ScaleVectorKernel>(&module)?;
        let q4k_matvec = CudaKernel::from_tiled::<shaders::q4k_matvec::Q4KMatvecKernel>(&module)?;
        let q4k_matvec_8sg = CudaKernel::from_tiled::<shaders::q4k_matvec::Q4KMatvec8sgKernel>(&module)?;
        let q4k_matvec_stride32 = CudaKernel::from_tiled::<shaders::q4k_matvec::Q4KMatvecStride32Kernel>(&module)?;
        let q4k_ffn_gate_up = CudaKernel::from_tiled::<shaders::q4k_ffn_gate_up::Q4KFfnGateUpKernel>(&module)?;
        let q4k_geglu_silu_down = CudaKernel::from_tiled::<shaders::q4k_geglu_down::Q4KGegeluSiluDownKernel>(&module)?;
        let q4k_geglu_gelu_tanh_down = CudaKernel::from_tiled::<shaders::q4k_geglu_down::Q4KGegeluGeluTanhDownKernel>(&module)?;
        let q6k_matvec = CudaKernel::from_tiled::<shaders::q6k_matvec::Q6KMatvecKernel>(&module)?;
        let q6k_matvec_8sg = CudaKernel::from_tiled::<shaders::q6k_matvec::Q6KMatvec8sgKernel>(&module)?;

        // Phase 2 kernels: non-fatal if missing (decode pipeline will fail gracefully)
        let rope_at_pos_q = kernel_or_placeholder("rope_at_pos_q", &module, &f32_gemv);
        let rope_at_pos_k = kernel_or_placeholder("rope_at_pos_k", &module, &f32_gemv);
        let qk_norm = kernel_or_placeholder("qk_norm", &module, &f32_gemv);
        let kv_append = kernel_or_placeholder("kv_append", &module, &f32_gemv);
        let kv_attend = kernel_or_placeholder("kv_attend", &module, &f32_gemv);
        let kv_attend_long = CudaKernel::from_tiled::<shaders::kv_attention::KvAttendLongKernel>(&module);
        let kv_append_attend_fused = kernel_or_placeholder("kv_append_attend_fused", &module, &f32_gemv);
        let v_norm = kernel_or_placeholder("v_norm", &module, &f32_gemv);
        let gegelu_silu = kernel_or_placeholder("gegelu_silu", &module, &f32_gemv);
        let gegelu_gelu_tanh = kernel_or_placeholder("gegelu_gelu_tanh", &module, &f32_gemv);
        let silu = kernel_or_placeholder("silu", &module, &f32_gemv);
        let gelu_tanh = kernel_or_placeholder("gelu_tanh", &module, &f32_gemv);
        let residual_norm_store = kernel_or_placeholder("residual_norm_store", &module, &f32_gemv);
        let post_attn_residual_norm_store = kernel_or_placeholder("post_attn_residual_norm_store", &module, &residual_norm_store);
        let post_ffn_norm_residual_add = kernel_or_placeholder("post_ffn_norm_residual_add", &module, &residual_norm_store);

        // Phase 3 kernels: norm/activation/attention/matmul parity
        let layer_norm = kernel_or_placeholder("layer_norm", &module, &rms_norm);
        let layer_norm_no_bias = kernel_or_placeholder("layer_norm_no_bias", &module, &rms_norm);
        let q4_matvec = kernel_or_placeholder("q4_matvec_v4", &module, &q4k_matvec);
        let q4_vecmat = kernel_or_placeholder("q4_vecmat", &module, &f32_gemv);
        let q4_f32_matvec = kernel_or_placeholder("q4_f32_matvec", &module, &f32_gemv);
        let q8_matvec = kernel_or_placeholder("q8_matvec", &module, &q4k_matvec);
        let q8_qkv_proj = kernel_or_placeholder("q8_qkv_proj", &module, &q4k_matvec);
        let q8_proj_rope = kernel_or_placeholder("q8_proj_rope", &module, &q4k_matvec);
        let rope_at_pos = kernel_or_placeholder("rope_at_pos", &module, &rope_at_pos_q);
        let rope_apply = kernel_or_placeholder("rope_apply", &module, &rope_at_pos_q);
        let rope_batched = kernel_or_placeholder("rope_at_pos_batched", &module, &rope_at_pos_q);
        let rope_batched_qk = CudaKernel::from_tiled::<shaders::rope::RopeBatchedQkKernel>(&module).unwrap_or_else(|| rope_at_pos_q.clone());
        let qk_norm_rope_fused = kernel_or_placeholder("qk_norm_rope_fused", &module, &qk_norm);
        let causal_attention = kernel_or_placeholder("causal_attention", &module, &f32_gemv);
        let quantize_q8 = kernel_or_placeholder("quantize_q8", &module, &rms_norm);
        let f16_gemv = kernel_or_placeholder("f16_gemv", &module, &f32_gemv);
        let sgemm = CudaKernel::from_tiled::<shaders::sgemm::SgemmKernel>(&module).unwrap_or_else(|| f32_gemv.clone());
        let sgemm_transb = CudaKernel::from_tiled::<shaders::sgemm::SgemmTransBKernel>(&module).unwrap_or_else(|| f32_gemv.clone());

        // Phase 4: Prefill kernels
        let q4k_matmul = CudaKernel::from_tiled::<shaders::q4k_matmul::Kernel>(&module).unwrap_or_else(|| q4k_matvec.clone());

        // These kernels may fail on some hardware; use placeholder fallback
        let rms_norm_q8 = kernel_or_placeholder("rms_norm_q8", &module, &residual_norm_store);
        let residual_norm_q8 = kernel_or_placeholder("residual_norm_q8", &module, &residual_norm_store);

        Some(Self {
            ctx,
            stream,
            bufs,
            f32_gemv,
            f32_argmax_partial,
            f32_topk_partial,
            rms_norm,
            residual_add,
            scale_vector,
            q4k_matvec,
            q4k_matvec_8sg,
            q4k_matvec_stride32,
            q4k_ffn_gate_up,
            q4k_geglu_silu_down,
            q4k_geglu_gelu_tanh_down,
            q6k_matvec,
            q6k_matvec_8sg,
            rope_at_pos_q,
            rope_at_pos_k,
            qk_norm,
            kv_append,
            kv_attend,
            kv_attend_long,
            kv_append_attend_fused,
            v_norm,
            gegelu_silu,
            gegelu_gelu_tanh,
            silu,
            gelu_tanh,
            residual_norm_store,
            post_attn_residual_norm_store,
            post_ffn_norm_residual_add,
            rms_norm_q8,
            residual_norm_q8,
            layer_norm,
            layer_norm_no_bias,
            q4_matvec,
            q4_vecmat,
            q4_f32_matvec,
            q8_matvec,
            q8_qkv_proj,
            q8_proj_rope,
            rope_at_pos,
            rope_apply,
            rope_batched,
            rope_batched_qk,
            qk_norm_rope_fused,
            causal_attention,
            quantize_q8,
            f16_gemv,
            sgemm,
            sgemm_transb,
            q4k_matmul,
            kv_cache: std::sync::Mutex::new(None),
            moe_scratch: std::sync::Mutex::new(None),
            flop_threshold: AtomicUsize::new(calibrate::DEFAULT_FLOP_THRESHOLD),
        })
    }

    /// Auto-calibrate CPU-vs-CUDA flop threshold.
    ///
    /// Runs small f32 gemv benchmarks on both CPU and GPU, finds the
    /// crossover point where GPU dispatch overhead is amortised, and
    /// stores it as the flop threshold. Falls back to
    /// `DEFAULT_FLOP_THRESHOLD` on any error (e.g. no CUDA device).
    pub fn calibrate(&self) {
        let threshold = calibrate::calibrate(&self.ctx, &self.stream, &self.f32_gemv);
        self.flop_threshold.store(threshold, Ordering::Relaxed);
    }

    pub fn flop_threshold(&self) -> usize {
        self.flop_threshold.load(Ordering::Relaxed)
    }

    pub fn set_flop_threshold(&self, t: usize) {
        self.flop_threshold
            .store(t.max(calibrate::MIN_FLOP_FLOOR), Ordering::Relaxed);
    }

    pub fn cache_size(&self) -> usize {
        self.bufs.len()
    }

    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn bufs(&self) -> &CudaBufferCache {
        &self.bufs
    }

    // ── Direct dispatch methods (for testing and trait wiring) ──

    /// Compute `out[N] = W[N, K] · x[K]` on GPU.
    pub fn encode_f32_gemv(&self, w: ndarray::ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        ops::f32_gemv::encode_f32_gemv(&self.ctx, &self.stream, &self.f32_gemv, w, x)
    }

    /// Compute RMS norm on GPU.
    pub fn encode_rms_norm(&self, x: &[f32], weight: &[f32], len: usize, eps: f32, offset: f32) -> Option<Vec<f32>> {
        ops::norm::encode_rms_norm(&self.ctx, &self.stream, &self.rms_norm, x, weight, len, eps, offset)
    }

    /// Compute residual add on GPU: `out = a + b`.
    pub fn encode_residual_add(&self, a: &[f32], b: &[f32], len: usize) -> Option<Vec<f32>> {
        ops::norm::encode_residual_add(&self.ctx, &self.stream, &self.residual_add, a, b, len)
    }

    /// Compute scale vector on GPU: `out = input * scalar`.
    pub fn encode_scale_vector(&self, input: &[f32], scalar: f32, len: usize) -> Option<Vec<f32>> {
        ops::norm::encode_scale_vector(&self.ctx, &self.stream, &self.scale_vector, input, scalar, len)
    }

    /// Q4_K matvec on GPU — default (4-warp) kernel.
    pub fn encode_q4k_matvec(&self, q4k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Option<Vec<f32>> {
        ops::q4k_matvec::encode_q4k_matvec(&self.ctx, &self.stream, &self.q4k_matvec, q4k_data, x, num_rows, hidden)
    }

    /// Q4_K matvec on GPU — 8-warp variant.
    pub fn encode_q4k_matvec_8sg(&self, q4k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Option<Vec<f32>> {
        ops::q4k_matvec::encode_q4k_matvec(&self.ctx, &self.stream, &self.q4k_matvec_8sg, q4k_data, x, num_rows, hidden)
    }

    /// Q4_K matvec on GPU — stride-32 variant (stable LM head argmax).
    pub fn encode_q4k_matvec_stride32(&self, q4k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Option<Vec<f32>> {
        ops::q4k_matvec::encode_q4k_matvec(&self.ctx, &self.stream, &self.q4k_matvec_stride32, q4k_data, x, num_rows, hidden)
    }

    /// Fused Q4_K gate+up on GPU.
    pub fn encode_q4k_ffn_gate_up(&self, wg_data: &[u8], wu_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        ops::q4k_ffn::encode_q4k_ffn_gate_up(&self.ctx, &self.stream, &self.q4k_ffn_gate_up, wg_data, wu_data, x, num_rows, hidden)
    }

    /// Fused SiLU activation + Q4_K down on GPU.
    pub fn encode_q4k_geglu_silu_down(&self, w_down: &[u8], gate: &[f32], up: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        ops::norm::encode_q4k_geglu_silu_down(&self.ctx, &self.stream, &self.q4k_geglu_silu_down, w_down, gate, up, n, k)
    }

    /// Fused GELU-tanh activation + Q4_K down on GPU.
    pub fn encode_q4k_geglu_gelu_tanh_down(&self, w_down: &[u8], gate: &[f32], up: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        ops::norm::encode_q4k_geglu_gelu_tanh_down(&self.ctx, &self.stream, &self.q4k_geglu_gelu_tanh_down, w_down, gate, up, n, k)
    }

    /// Q6_K matvec on GPU — default (4-warp) kernel.
    pub fn encode_q6k_matvec(&self, q6k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Option<Vec<f32>> {
        ops::q6k_matvec::encode_q6k_matvec(&self.ctx, &self.stream, &self.q6k_matvec, q6k_data, x, num_rows, hidden)
    }

    /// Q6_K matvec on GPU — 8-warp variant.
    pub fn encode_q6k_matvec_8sg(&self, q6k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Option<Vec<f32>> {
        ops::q6k_matvec::encode_q6k_matvec(&self.ctx, &self.stream, &self.q6k_matvec_8sg, q6k_data, x, num_rows, hidden)
    }
}