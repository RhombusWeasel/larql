//! CPU-vs-CUDA flop threshold auto-tuning.
//!
//! Mirrors `metal::calibrate`. Runs f32 gemv benchmarks on both CPU and
//! GPU to find the crossover point where GPU dispatch overhead is amortised.
//! Below this threshold, CPU BLAS is faster; above it, CUDA wins despite
//! the host↔device transfer latency.

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaStream, CudaContext};

use crate::cuda::kernel::CudaKernel;
use crate::cuda::ops::f32_gemv;

/// Default flop threshold — conservative starting point.
/// CUDA dispatch overhead (htod copy + launch + dtoh copy) is typically
/// 50-200µs on modern GPUs; the crossover for f32 gemv is around 2M FLOPs.
pub const DEFAULT_FLOP_THRESHOLD: usize = 2_000_000;
/// Minimum floor — don't go below this even if calibration suggests it.
/// Ensures we never dispatch trivially small ops to GPU.
pub const MIN_FLOP_FLOOR: usize = 500_000;

/// Run calibration and return the optimal FLOP threshold.
///
/// Benchmarks f32 matrix-vector multiply at several sizes on both CPU
/// (ndarray BLAS) and GPU (CUDA f32_gemv kernel), finds the smallest
/// FLOP count where GPU is faster, and returns that as the threshold.
/// Falls back to `DEFAULT_FLOP_THRESHOLD` on any error.
pub fn calibrate(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
) -> usize {
    let test_cases: &[(usize, usize)] = &[
        (64, 64),       // ~8K FLOPs — expect CPU to win
        (256, 256),     // ~131K FLOPs — expect CPU to win
        (1024, 2560),   // ~5.2M FLOPs — expect GPU to win
        (4096, 2560),   // ~21M FLOPs — expect GPU to win
        (16384, 2560),  // ~84M FLOPs — expect GPU to win decisively
    ];

    let mut best = DEFAULT_FLOP_THRESHOLD;

    for (_i, &(n, k)) in test_cases.iter().enumerate() {
        let flops = 2 * n * k;

        // GPU benchmark: f32 gemv via CUDA
        let gpu_us = bench_median(5, || {
            bench_gpu_gemv(ctx, stream, kernel, n, k)
        });

        // CPU benchmark: ndarray dot product
        let cpu_us = bench_median(5, || {
            bench_cpu_gemv(n, k)
        });

        // If GPU is faster at this size, update the threshold
        if gpu_us > 0 && cpu_us > 0 && gpu_us < cpu_us {
            best = best.min(flops);
        }
    }

    best.max(MIN_FLOP_FLOOR)
}

/// Benchmark a single f32 gemv on GPU.
///
/// Returns the wall-clock time in microseconds, or 0 on error.
fn bench_gpu_gemv(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &CudaKernel,
    n: usize,
    k: usize,
) -> u64 {
    // Create synthetic weight matrix (row-major, n×k)
    let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.001).sin()).collect();
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.001).cos()).collect();

    let w_view = ndarray::ArrayView2::from_shape((n, k), &w).unwrap();

    // Warm up
    let _ = f32_gemv::encode_f32_gemv(ctx, stream, kernel, w_view, &x);

    let start = Instant::now();
    let _ = f32_gemv::encode_f32_gemv(ctx, stream, kernel, w_view, &x);
    start.elapsed().as_micros() as u64
}

/// Benchmark a single f32 gemv on CPU (ndarray BLAS).
///
/// Returns the wall-clock time in microseconds.
fn bench_cpu_gemv(n: usize, k: usize) -> u64 {
    let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.001).sin()).collect();
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.001).cos()).collect();

    let w_arr = ndarray::Array2::from_shape_vec((n, k), w).unwrap();
    let x_arr = ndarray::Array1::from_vec(x.clone());

    // Warm up
    let _ = w_arr.dot(&x_arr);

    let start = Instant::now();
    let _ = w_arr.dot(&x_arr);
    start.elapsed().as_micros() as u64
}

/// Run `n` iterations of `f`, return the median time in microseconds.
fn bench_median<F: FnMut() -> u64>(n: usize, mut f: F) -> u64 {
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        times.push(f());
    }
    times.sort_unstable();
    times[n / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_threshold_is_above_floor() {
        assert!(
            DEFAULT_FLOP_THRESHOLD >= MIN_FLOP_FLOOR,
            "DEFAULT_FLOP_THRESHOLD ({}) must be >= MIN_FLOP_FLOOR ({})",
            DEFAULT_FLOP_THRESHOLD,
            MIN_FLOP_FLOOR,
        );
    }

    #[test]
    fn min_floor_is_sane() {
        // Floor should be at least 100K FLOPs — anything smaller is
        // almost certainly dominated by dispatch overhead.
        assert!(MIN_FLOP_FLOOR >= 100_000);
    }

    /// Calibration returns a value within the legal envelope.
    /// Only runs on machines with CUDA available.
    #[cfg(feature = "cuda")]
    #[test]
    fn calibrate_returns_threshold_in_legal_envelope() {
        let backend = match crate::cuda::CudaBackend::new() {
            Some(b) => b,
            None => return, // No CUDA device available
        };
        let threshold = calibrate(&backend.ctx, &backend.stream, &backend.f32_gemv);
        assert!(
            threshold >= MIN_FLOP_FLOOR,
            "calibrated threshold {threshold} below MIN_FLOP_FLOOR={MIN_FLOP_FLOOR}"
        );
        assert!(
            threshold <= DEFAULT_FLOP_THRESHOLD,
            "calibrated threshold {threshold} above DEFAULT_FLOP_THRESHOLD={DEFAULT_FLOP_THRESHOLD}"
        );
    }
}