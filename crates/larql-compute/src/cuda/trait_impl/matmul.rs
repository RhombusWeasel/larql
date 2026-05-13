//! `MatMul` impl for CudaBackend.
//!
//! Dense matmul goes through CPU BLAS. The f32_gemv path dispatches
//! through the CUDA kernel for large matrices (above flop threshold).
//! Phase 4 adds GPU-sided top-K for LM head: f32_gemv_topk1, f16_gemv_topk1,
//! f16_gemv_topk, and Q4 matvec top-K variants.

use ndarray::ArrayView2;
use std::sync::atomic::Ordering;

use crate::backend::MatMul;
use crate::cuda::CudaBackend;
use crate::cuda::ops::f32_argmax;

impl MatMul for CudaBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> ndarray::Array2<f32> {
        crate::cpu::ops::f32_matmul::matmul(a, b)
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> ndarray::Array2<f32> {
        crate::cpu::ops::f32_matmul::matmul_transb(a, b)
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        if 2 * n * k < self.flop_threshold.load(Ordering::Relaxed) {
            return None;
        }
        self.encode_f32_gemv(w, x)
    }

    fn f32_gemv_force(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (_n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        self.encode_f32_gemv(w, x)
    }

    fn f32_gemv_topk1(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<(u32, f32)> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k || n == 0 {
            return None;
        }
        // Full pipelined path: gemv on GPU → argmax on GPU → read back 8 bytes
        f32_argmax::encode_f32_gemv_topk1(
            self.ctx(),
            self.stream(),
            &self.f32_gemv,
            &self.f32_argmax_partial,
            w,
            x,
        )
    }

    fn f16_gemv(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n * k * 2 || x.len() != k {
            return None;
        }
        if 2 * n * k < self.flop_threshold.load(Ordering::Relaxed) {
            return None;
        }
        self.encode_f16_gemv(w_f16, x, n, k)
    }

    fn f16_gemv_force(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n * k * 2 || x.len() != k {
            return None;
        }
        self.encode_f16_gemv(w_f16, x, n, k)
    }

    fn f16_gemv_topk1(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<(u32, f32)> {
        let scores = self.encode_f16_gemv(w_f16, x, n, k)?;
        f32_argmax::encode_f32_argmax_partial(
            self.ctx(),
            self.stream(),
            &self.f32_argmax_partial,
            &scores,
        )
    }

    fn f16_gemv_topk(
        &self,
        w_f16: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
        top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        let scores = self.encode_f16_gemv(w_f16, x, n, k)?;
        f32_argmax::encode_f32_topk_partial(
            self.ctx(),
            self.stream(),
            &self.f32_topk_partial,
            &scores,
            top_k,
        )
    }
}

impl CudaBackend {
    /// Encode f16 gemv on GPU.
    fn encode_f16_gemv(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        let stream = self.stream();
        let w_dev = stream.clone_htod(w_f16).ok()?;
        let x_dev = stream.clone_htod(x).ok()?;
        let mut out_dev = stream.alloc_zeros::<f32>(n).ok()?;

        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let grid = (n as u64).div_ceil(self.f16_gemv.rows_per_block) as u32;
        let block = self.f16_gemv.threads_per_block as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&self.f16_gemv.func)
                .arg(&w_dev)
                .arg(&x_dev)
                .arg(&mut out_dev)
                .arg(&n_u32)
                .arg(&k_u32)
                .launch(cfg)
                .ok()?;
        }

        stream.synchronize().ok()?;
        stream.clone_dtoh(&out_dev).ok()
    }
}