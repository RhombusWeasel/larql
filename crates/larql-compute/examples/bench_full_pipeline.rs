//! Full pipeline benchmark: 21 layers × (attention + FFN) in one Metal submission.
//!
//! Usage:
//!   cargo run --release -p larql-compute --features metal --example bench_full_pipeline

extern crate blas_src;

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
        use larql_compute::metal::MetalBackend;
        use larql_compute::metal::ops::full_pipeline::LayerWeights;

        let metal = MetalBackend::new().expect("Metal required");

        let hidden = 2560;
        let inter = 10240;
        let q_dim = 2560;  // num_q_heads * head_dim
        let kv_dim = 512;  // num_kv_heads * head_dim
        let num_layers = 21;
        let n = 10;

        println!("=== Full Pipeline Benchmark ===");
        println!("{num_layers} layers × (4 attn proj + 3 FFN ops), one Metal submission\n");

        // Build layer weights
        let mut layers_data: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::new();
        for l in 0..num_layers {
            let wq: Vec<f32> = (0..q_dim * hidden).map(|i| ((i + l * 1000) as f32 * 0.0001).cos()).collect();
            let wk: Vec<f32> = (0..kv_dim * hidden).map(|i| ((i + l * 2000) as f32 * 0.0002).sin()).collect();
            let wv: Vec<f32> = (0..kv_dim * hidden).map(|i| ((i + l * 3000) as f32 * 0.0003).cos()).collect();
            let wo: Vec<f32> = (0..hidden * q_dim).map(|i| ((i + l * 4000) as f32 * 0.0004).sin()).collect();
            let g: Vec<f32> = (0..inter * hidden).map(|i| ((i + l * 5000) as f32 * 0.0001).cos()).collect();
            let u: Vec<f32> = (0..inter * hidden).map(|i| ((i + l * 6000) as f32 * 0.0002).sin()).collect();
            let mut dt = vec![0.0f32; hidden * inter];
            for r in 0..inter { for c in 0..hidden { dt[c * inter + r] = ((r * hidden + c + l * 7000) as f32 * 0.0003).cos(); } }
            layers_data.push((wq, wk, wv, wo, quantize_q4_0(&g), quantize_q4_0(&u), quantize_q4_0(&dt)));
        }

        let layers: Vec<LayerWeights> = layers_data.iter().map(|(wq, wk, wv, wo, g, u, d)| {
            LayerWeights { w_q: wq, w_k: wk, w_v: wv, w_o: wo, gate_q4: g, up_q4: u, down_t_q4: d }
        }).collect();

        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();

        // Warmup
        let _ = metal.full_pipeline(&layers, &x, hidden, inter, q_dim, kv_dim);

        // Benchmark
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = metal.full_pipeline(&layers, &x, hidden, inter, q_dim, kv_dim);
        }
        let full_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let tps = 1000.0 / full_ms;

        // FFN-only for comparison
        let layers_q4: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = layers_data.iter()
            .map(|(_, _, _, _, g, u, d)| (g.clone(), u.clone(), d.clone())).collect();
        let _ = metal.multi_layer_q4_ffn(&layers_q4, &x, inter, hidden);
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = metal.multi_layer_q4_ffn(&layers_q4, &x, inter, hidden);
        }
        let ffn_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        // CPU baseline
        let cpu = larql_compute::cpu_backend();
        use larql_compute::ComputeBackend;
        let wq_arr = ndarray::Array2::from_shape_vec((q_dim, hidden), layers_data[0].0.clone()).unwrap();
        let x_arr = ndarray::Array2::from_shape_vec((1, hidden), x.clone()).unwrap();
        let t0 = Instant::now();
        for _ in 0..n {
            for _ in 0..num_layers {
                let _ = cpu.matmul_transb(x_arr.view(), wq_arr.view());
                let _ = cpu.matmul_transb(x_arr.view(), wq_arr.view());
                let _ = cpu.matmul_transb(x_arr.view(), wq_arr.view());
                let _ = cpu.matmul_transb(x_arr.view(), wq_arr.view());
            }
        }
        let cpu_attn_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        println!("  Metal full pipeline (attn+FFN, 1 cmd):  {full_ms:>6.1}ms  ({tps:.0} tok/s)");
        println!("  Metal FFN-only (1 cmd):                 {ffn_ms:>6.1}ms");
        println!("  CPU BLAS attn-only (4 proj × 21L):      {cpu_attn_ms:>6.1}ms");
        println!("  Attention overhead in pipeline:          {:.1}ms", full_ms - ffn_ms);
        println!();
        println!("  Projected with vindex logits + cache:");
        let projected = full_ms + 5.0; // + logits + other
        println!("    {projected:.0}ms → {:.0} tok/s", 1000.0 / projected);
        println!();
        println!("  Ollama reference: ~10ms → ~100 tok/s");

        println!("\n=== Done ===");
    }
}
