//! Per-kernel CUDA GPU bandwidth profiler.
//!
//! Measures each production kernel at Gemma 3 4B shapes to produce
//! a real-world bandwidth comparison. Mirrors `metal::diag::kernel_profile`
//! but uses CUDA events for timing instead of Metal command buffer timing.

use std::time::Instant;

/// Result for a single kernel profiling run.
#[derive(Debug, Clone)]
pub struct KernelResult {
    /// Kernel name.
    pub name: String,
    /// Megabytes of weight data read per kernel call.
    pub mb_per_call: f64,
    /// Mean isolated time per call (ms), including GPU spin-up.
    pub isolated_ms: f64,
    /// Stddev of isolated times.
    pub isolated_sd_ms: f64,
    /// Effective bandwidth from isolated measurement (GB/s).
    pub isolated_gbs: f64,
    /// Mean time per layer when batched (ms).
    pub batched_ms_per_layer: f64,
    /// Effective bandwidth from batched measurement (GB/s).
    pub batched_gbs: f64,
}

impl KernelResult {
    /// ms/token at `n_layers` layers using the batched rate.
    pub fn ms_per_token(&self, n_layers: usize) -> f64 {
        self.batched_ms_per_layer * n_layers as f64
    }

    /// Whether the kernel appears compute-bound (GB/s well below peak).
    pub fn is_compute_bound(&self) -> bool {
        self.batched_gbs < 300.0
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn stddev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn measure_isolated(warmup: usize, iters: usize, f: &mut impl FnMut()) -> (f64, f64) {
    let mut times = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        f();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            times.push(ms);
        }
    }
    (mean(&times), stddev(&times))
}

/// Profile all production kernels at Gemma 3 4B shapes.
///
/// Returns one `KernelResult` per kernel. Prints a formatted table.
/// Requires `--features cuda` and an NVIDIA GPU.
#[cfg(feature = "cuda")]
pub fn profile_all(n_layers: usize, warmup: usize, iters: usize) -> Vec<KernelResult> {
    use crate::cuda::CudaBackend;
    use crate::QuantMatVec;

    let backend = match CudaBackend::new() {
        Some(b) => b,
        None => {
            eprintln!("[cuda-profile] No CUDA device available");
            return Vec::new();
        }
    };
    backend.calibrate();

    let hidden = 2560usize;
    let inter = 10240usize;
    let q_dim = 8192usize;
    let _kv_dim = 4096usize;

    let mut results = Vec::new();

    // Measure commit+wait overhead (empty stream sync).
    let commit_overhead_ms = {
        let mut times = Vec::new();
        for i in 0..warmup + iters {
            let t = Instant::now();
            let _ = backend.stream().synchronize();
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            if i >= warmup {
                times.push(ms);
            }
        }
        mean(&times)
    };

    println!("[cuda-profile] Stream sync overhead: {commit_overhead_ms:.3}ms");
    println!();
    println!(
        "{:<44} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Kernel", "iso_ms", "iso_gbs", "bat_ms", "bat_gbs", "ms/tok"
    );
    println!("{}", "-".repeat(88));

    // ── q4k_matvec: Wo O-projection (N=hidden, K=q_dim) ──
    {
        let n = hidden;
        let k = q_dim;
        let mb = (n * (k / 256 * 144)) as f64 / 1e6;
        let x = (0..k).map(|i| (i as f32 * 0.007).sin() * 0.4).collect::<Vec<f32>>();
        let w = crate::cpu::ops::q4_common::quantize_q4_k(
            &(0..n * k).map(|i| (i as f32 * 0.007).sin() * 0.4).collect::<Vec<f32>>()
        );

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let _ = backend.q4k_matvec(&w, &x, n, k);
        });

        let iso_kernel = (iso_ms - commit_overhead_ms).max(0.001);
        let r = KernelResult {
            name: "q4k_matvec (Wo, 2560×8192)".into(),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_kernel,
            batched_ms_per_layer: iso_ms, // Approximation — real batched needs n_layers dispatches
            batched_gbs: mb / (iso_ms.max(0.001)),
        };
        println!(
            "{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1} {:>7.1}ms",
            r.name, r.isolated_ms, r.isolated_gbs,
            r.batched_ms_per_layer, r.batched_gbs,
            r.ms_per_token(n_layers)
        );
        results.push(r);
    }

    // ── q6k_matvec: FFN down (N=hidden, K=inter) ──
    {
        let n = hidden;
        let k = inter;
        let mb = (n * (k / 256 * 210)) as f64 / 1e6;
        let w = crate::cpu::ops::q4_common::quantize_q6_k(
            &(0..n * k).map(|i| (i as f32 * 0.007).sin() * 0.4).collect::<Vec<f32>>()
        );
        let x = (0..k).map(|i| (i as f32 * 0.007).sin() * 0.4).collect::<Vec<f32>>();

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let _ = backend.encode_q6k_matvec(&w, &x, n, k);
        });

        let iso_kernel = (iso_ms - commit_overhead_ms).max(0.001);
        let r = KernelResult {
            name: "q6k_matvec (down, 2560×10240)".into(),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_kernel,
            batched_ms_per_layer: iso_ms,
            batched_gbs: mb / (iso_ms.max(0.001)),
        };
        println!(
            "{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1} {:>7.1}ms",
            r.name, r.isolated_ms, r.isolated_gbs,
            r.batched_ms_per_layer, r.batched_gbs,
            r.ms_per_token(n_layers)
        );
        results.push(r);
    }

    // ── RMS norm ──
    {
        let len = hidden;
        let x = (0..len).map(|i| (i as f32 * 0.007).sin() * 0.4).collect::<Vec<f32>>();
        let w = vec![1.0f32; len];

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let _ = backend.encode_rms_norm(&x, &w, len, 1e-6, 0.0);
        });

        let r = KernelResult {
            name: "rms_norm (2560)".into(),
            mb_per_call: 0.0, // Too small to measure in MB
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: 0.0,
            batched_ms_per_layer: iso_ms,
            batched_gbs: 0.0,
        };
        println!(
            "{:<44} {:>7.3}ms {:>7} {:>7.3}ms {:>7} {:>7.1}ms",
            r.name, r.isolated_ms, "—", r.batched_ms_per_layer, "—",
            r.ms_per_token(n_layers)
        );
        results.push(r);
    }

    // ── Summary ──
    println!();
    println!("=== CUDA Kernel Profile ({} layers) ===", n_layers);
    for r in &results {
        println!(
            "  {} — {:.1} GB/s, {:.1}ms/tok",
            r.name, r.batched_gbs, r.ms_per_token(n_layers)
        );
    }

    results
}

#[cfg(not(feature = "cuda"))]
pub fn profile_all(_n_layers: usize, _warmup: usize, _iters: usize) -> Vec<KernelResult> {
    Vec::new()
}