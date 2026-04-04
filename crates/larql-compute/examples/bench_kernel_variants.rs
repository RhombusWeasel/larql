//! Head-to-head Q4 matvec kernel comparison.
//!
//! v1: simdgroup reduction, threadgroup shared memory (current)
//! v2: 4 rows per thread, f32 input, no shared memory
//! v3: 8 rows per thread, fully unrolled
//!
//! Usage:
//!   cargo run --release -p larql-compute --features metal --example bench_kernel_variants

extern crate blas_src;

use std::ffi::c_void;
use std::time::Instant;

fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 32 == 0);
    let n = data.len() / 32;
    let mut out = Vec::with_capacity(n * 18);
    for i in 0..n {
        let blk = &data[i * 32..(i + 1) * 32];
        let amax = blk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 7.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let bits = scale.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;
        let f16 = if exp == 0 { sign as u16 }
            else if exp >= 31 + 127 - 15 { (sign | 0x7C00) as u16 }
            else if exp <= -15 + 127 { sign as u16 }
            else { (sign | (((exp - 127 + 15) as u32) << 10) | (mant >> 13)) as u16 };
        out.extend_from_slice(&f16.to_le_bytes());
        for j in 0..16 {
            let lo = ((blk[j * 2] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            let hi = ((blk[j * 2 + 1] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            out.push(lo | (hi << 4));
        }
    }
    out
}

fn main() {
    #[cfg(not(feature = "metal"))]
    { println!("Run with --features metal"); return; }

    #[cfg(feature = "metal")]
    {
        use metal::*;
        use larql_compute::cpu::q4;

        let hidden = 2560;
        let inter = 10240;
        let n_iters = 50;

        let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
        let q4_data = quantize_q4_0(&matrix);
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
        let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

        println!("=== Q4 Matvec Kernel Variants ===");
        println!("Matrix: [{inter}, {hidden}] = {:.1}MB Q4_0", q4_data.len() as f64 / 1e6);
        println!("Target: <0.2ms (llama.cpp implied ~0.08ms)\n");

        // Setup Metal
        let device = Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let src = larql_compute::metal::shaders::all_shaders();
        let opts = CompileOptions::new();
        let lib = device.new_library_with_source(&src, &opts).unwrap();

        let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
        let buf_q4 = bufs.get_bytes(&q4_data);
        let buf_x = bufs.transient_from_f32(&x);

        // CPU reference
        let cpu_result = q4::q4_matvec(&q4_data, &x, inter, hidden);

        // ── BLAS f32 baseline ──
        {
            let mat = ndarray::ArrayView2::from_shape((inter, hidden), &matrix).unwrap();
            let xv = ndarray::Array1::from_vec(x.clone());
            let _ = mat.dot(&xv);
            let t0 = Instant::now();
            for _ in 0..n_iters { let _ = mat.dot(&xv); }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
            println!("  BLAS f32 gemv:       {ms:>6.3}ms  (baseline)");
        }

        // ── CPU C kernel ──
        {
            let _ = q4::q4_matvec(&q4_data, &x, inter, hidden);
            let t0 = Instant::now();
            for _ in 0..n_iters { let _ = q4::q4_matvec(&q4_data, &x, inter, hidden); }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
            println!("  CPU C vdotq:         {ms:>6.3}ms");
        }

        // Helper to benchmark a Metal pipeline
        let bench_metal = |name: &str, pipeline: &ComputePipelineState, grid: MTLSize, tg: MTLSize,
                           setup_fn: &dyn Fn(&ComputeCommandEncoderRef, &Buffer)| {
            let buf_out = bufs.output((inter * 4) as u64);

            // Warmup
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(&buf_q4), 0);
            setup_fn(enc, &buf_out);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Benchmark
            let t0 = Instant::now();
            for _ in 0..n_iters {
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipeline);
                enc.set_buffer(0, Some(&buf_q4), 0);
                setup_fn(enc, &buf_out);
                enc.dispatch_thread_groups(grid, tg);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
            let gbps = q4_data.len() as f64 / ms / 1e6;

            // Check correctness
            let ptr = buf_out.contents() as *const f32;
            let result = unsafe { std::slice::from_raw_parts(ptr, inter) };
            let max_diff: f32 = cpu_result.iter().zip(result.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

            println!("  {name:22} {ms:>6.3}ms  ({gbps:>5.1} GB/s)  diff={max_diff:.4}");
        };

        // ── v1: simdgroup + threadgroup shared memory (current) ──
        {
            let pipeline = device.new_compute_pipeline_state_with_function(
                &lib.get_function("q4_matvec", None).unwrap()
            ).unwrap();
            let buf_q8 = bufs.transient_from_i8(&q8_x);
            let buf_sc = bufs.transient_from_f32(&q8_scales);
            let n_val = inter as u32;
            let k_val = hidden as u32;
            let rows_per_tg = 8u64;
            let num_tgs = ((inter as u64) + rows_per_tg - 1) / rows_per_tg;

            bench_metal("v1 (simdgroup+tg)", &pipeline,
                MTLSize::new(num_tgs, 1, 1), MTLSize::new(256, 1, 1),
                &|enc, buf_out| {
                    enc.set_buffer(1, Some(&buf_q8), 0);
                    enc.set_buffer(2, Some(&buf_sc), 0);
                    enc.set_buffer(3, Some(buf_out), 0);
                    enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);
                });
        }

        // ── v2: 4 rows per thread, f32 input ──
        {
            let pipeline = device.new_compute_pipeline_state_with_function(
                &lib.get_function("q4_matvec_v2", None).unwrap()
            ).unwrap();
            let n_val = inter as u32;
            let k_val = hidden as u32;
            let n_threads = ((inter + 3) / 4) as u64;

            bench_metal("v2 (4-row, f32 in)", &pipeline,
                MTLSize::new((n_threads + 255) / 256, 1, 1), MTLSize::new(256, 1, 1),
                &|enc, buf_out| {
                    enc.set_buffer(1, Some(&buf_x), 0);
                    enc.set_buffer(2, Some(buf_out), 0);
                    enc.set_bytes(3, 4, &n_val as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &k_val as *const u32 as *const c_void);
                });
        }

        // ── v3: 8 rows per thread, unrolled ──
        {
            let pipeline = device.new_compute_pipeline_state_with_function(
                &lib.get_function("q4_matvec_v3", None).unwrap()
            ).unwrap();
            let n_val = inter as u32;
            let k_val = hidden as u32;
            let n_threads = ((inter + 7) / 8) as u64;

            bench_metal("v3 (8-row, unrolled)", &pipeline,
                MTLSize::new((n_threads + 255) / 256, 1, 1), MTLSize::new(256, 1, 1),
                &|enc, buf_out| {
                    enc.set_buffer(1, Some(&buf_x), 0);
                    enc.set_buffer(2, Some(buf_out), 0);
                    enc.set_bytes(3, 4, &n_val as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &k_val as *const u32 as *const c_void);
                });
        }

        // ── v4: wide uint32 loads + simdgroup ──
        {
            let pipeline = device.new_compute_pipeline_state_with_function(
                &lib.get_function("q4_matvec_v4", None).unwrap()
            ).unwrap();
            let buf_q8 = bufs.transient_from_i8(&q8_x);
            let buf_sc = bufs.transient_from_f32(&q8_scales);
            let n_val = inter as u32;
            let k_val = hidden as u32;
            let rows_per_tg = 8u64;
            let num_tgs = ((inter as u64) + rows_per_tg - 1) / rows_per_tg;

            bench_metal("v4 (uint32+simdgrp)", &pipeline,
                MTLSize::new(num_tgs, 1, 1), MTLSize::new(256, 1, 1),
                &|enc, buf_out| {
                    enc.set_buffer(1, Some(&buf_q8), 0);
                    enc.set_buffer(2, Some(&buf_sc), 0);
                    enc.set_buffer(3, Some(buf_out), 0);
                    enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);
                });
        }

        // ── Attention-sized Q4 matrices ──
        println!("\n  --- Attention projections (v4 on smaller matrices) ---");
        {
            // Q/O projection: [2560, 2560]
            let wq_f32: Vec<f32> = (0..hidden * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
            let wq_q4 = quantize_q4_0(&wq_f32);
            let x1: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
            let (q8_1, sc_1) = q4::quantize_to_q8(&x1);

            let pipeline = device.new_compute_pipeline_state_with_function(
                &lib.get_function("q4_matvec_v4", None).unwrap()
            ).unwrap();
            let buf_wq = bufs.get_bytes(&wq_q4);
            let buf_q8_1 = bufs.transient_from_i8(&q8_1);
            let buf_sc_1 = bufs.transient_from_f32(&sc_1);
            let n_q = hidden as u32;
            let k_q = hidden as u32;
            let rows_per_tg = 8u64;
            let num_tgs_q = ((hidden as u64) + rows_per_tg - 1) / rows_per_tg;

            bench_metal("v4 Q proj [2560,2560]", &pipeline,
                MTLSize::new(num_tgs_q, 1, 1), MTLSize::new(256, 1, 1),
                &|enc, buf_out| {
                    enc.set_buffer(0, Some(&buf_wq), 0);
                    enc.set_buffer(1, Some(&buf_q8_1), 0);
                    enc.set_buffer(2, Some(&buf_sc_1), 0);
                    enc.set_buffer(3, Some(buf_out), 0);
                    enc.set_bytes(4, 4, &n_q as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &k_q as *const u32 as *const c_void);
                });

            // K/V projection: [512, 2560]
            let kv_dim = 512;
            let wk_f32: Vec<f32> = (0..kv_dim * hidden).map(|i| (i as f32 * 0.0002).sin()).collect();
            let wk_q4 = quantize_q4_0(&wk_f32);
            let buf_wk = bufs.get_bytes(&wk_q4);
            let n_k = kv_dim as u32;
            let num_tgs_k = ((kv_dim as u64) + rows_per_tg - 1) / rows_per_tg;

            // Need smaller output buffer
            let buf_out_k = bufs.output((kv_dim * 4) as u64);
            bench_metal("v4 K proj [512,2560]", &pipeline,
                MTLSize::new(num_tgs_k, 1, 1), MTLSize::new(256, 1, 1),
                &|enc, _buf_out| {
                    enc.set_buffer(0, Some(&buf_wk), 0);
                    enc.set_buffer(1, Some(&buf_q8_1), 0);
                    enc.set_buffer(2, Some(&buf_sc_1), 0);
                    enc.set_buffer(3, Some(&buf_out_k), 0);
                    enc.set_bytes(4, 4, &n_k as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &k_q as *const u32 as *const c_void);
                });

            // CPU BLAS f32 for comparison
            {
                let wq_arr = ndarray::Array2::from_shape_vec((hidden, hidden), wq_f32).unwrap();
                let x_arr = ndarray::Array2::from_shape_vec((1, hidden), x1.clone()).unwrap();
                let t0 = Instant::now();
                for _ in 0..n_iters { let _ = x_arr.dot(&wq_arr.t()); }
                let ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
                println!("  CPU BLAS Q proj [1,2560]@[2560,2560]^T:  {ms:.3}ms");
            }
        }

        println!("\n=== Done ===");
    }
}
