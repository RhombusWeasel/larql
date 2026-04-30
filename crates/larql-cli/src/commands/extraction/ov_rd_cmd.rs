use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, Subcommand};
use larql_inference::attention::run_attention_block_with_pre_o;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, WeightFfn};
use larql_vindex::{
    load_model_weights_q4k, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use ndarray::{s, Array2};
use serde::{Deserialize, Serialize};

#[derive(Args)]
pub struct OvRdArgs {
    #[command(subcommand)]
    command: OvRdCommand,
}

#[derive(Subcommand)]
enum OvRdCommand {
    /// Capture pre-W_O OV output statistics from a Q4K vindex.
    Capture(CaptureArgs),
}

#[derive(Args)]
struct CaptureArgs {
    /// Self-contained Q4K vindex directory.
    #[arg(long)]
    index: PathBuf,

    /// JSONL prompt file. Each line must include at least {"prompt": "..."}.
    #[arg(long)]
    prompts: PathBuf,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,

    /// Layers to capture. Comma-separated or range. Default: all.
    #[arg(long)]
    layers: Option<String>,

    /// Limit prompts for smoke runs.
    #[arg(long)]
    max_prompts: Option<usize>,

    /// Limit token positions per prompt for smoke runs.
    #[arg(long)]
    max_positions: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct PromptRecord {
    id: Option<String>,
    stratum: Option<String>,
    prompt: String,
}

#[derive(Debug)]
struct RunningHeadStats {
    count: u64,
    sum: Vec<f64>,
    sum_sq_norm: f64,
}

impl RunningHeadStats {
    fn new(head_dim: usize) -> Self {
        Self {
            count: 0,
            sum: vec![0.0; head_dim],
            sum_sq_norm: 0.0,
        }
    }

    fn add(&mut self, values: &[f32]) {
        self.count += 1;
        let mut sq = 0.0f64;
        for (dst, &v) in self.sum.iter_mut().zip(values.iter()) {
            let vf = v as f64;
            *dst += vf;
            sq += vf * vf;
        }
        self.sum_sq_norm += sq;
    }

    fn finish(&self) -> FinishedHeadStats {
        if self.count == 0 {
            return FinishedHeadStats {
                count: 0,
                mean_norm_sq: 0.0,
                second_moment: 0.0,
                variance: 0.0,
                rms_norm: 0.0,
            };
        }
        let n = self.count as f64;
        let mean_norm_sq = self
            .sum
            .iter()
            .map(|v| {
                let m = *v / n;
                m * m
            })
            .sum::<f64>();
        let second_moment = self.sum_sq_norm / n;
        let variance = (second_moment - mean_norm_sq).max(0.0);
        FinishedHeadStats {
            count: self.count,
            mean_norm_sq,
            second_moment,
            variance,
            rms_norm: second_moment.sqrt(),
        }
    }
}

#[derive(Debug, Serialize)]
struct FinishedHeadStats {
    count: u64,
    mean_norm_sq: f64,
    second_moment: f64,
    variance: f64,
    rms_norm: f64,
}

#[derive(Debug, Serialize)]
struct HeadReport {
    layer: usize,
    head: usize,
    head_dim: usize,
    stats: FinishedHeadStats,
}

#[derive(Debug, Serialize)]
struct CaptureReport {
    index: String,
    prompt_file: String,
    prompts_seen: usize,
    layers: Vec<usize>,
    max_positions: Option<usize>,
    heads: Vec<HeadReport>,
}

pub fn run(args: OvRdArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        OvRdCommand::Capture(capture) => run_capture(capture),
    }
}

fn run_capture(args: CaptureArgs) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&args.out)?;

    eprintln!("Loading vindex: {}", args.index.display());
    let start = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&args.index, &mut cb)?;
    index.load_attn_q4k(&args.index)?;
    index.load_interleaved_q4k(&args.index)?;
    let mut weights = load_model_weights_q4k(&args.index, &mut cb)?;
    let tokenizer = load_vindex_tokenizer(&args.index)?;
    eprintln!(
        "  {} layers, hidden_size={}, q_heads={}, head_dim={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        weights.num_q_heads,
        weights.head_dim,
        start.elapsed().as_secs_f64()
    );

    let layers: Vec<usize> = match &args.layers {
        Some(spec) => parse_layer_spec(spec)?,
        None => (0..weights.num_layers).collect(),
    };
    let capture_layer = |layer: usize| layers.contains(&layer);

    let prompts = load_prompts(&args.prompts, args.max_prompts)?;
    eprintln!("Prompts: {}", prompts.len());
    eprintln!("Layers: {:?}", layers);

    let mut stats: Vec<Vec<RunningHeadStats>> = (0..weights.num_layers)
        .map(|layer| {
            let heads = weights.arch.num_q_heads_for_layer(layer);
            let head_dim = weights.arch.head_dim_for_layer(layer);
            (0..heads)
                .map(|_| RunningHeadStats::new(head_dim))
                .collect()
        })
        .collect();

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  [{}/{}] {}", prompt_idx + 1, prompts.len(), label);

        let token_ids = encode_prompt(&tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }

        let mut h = embed_tokens_pub(&weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(&weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(&mut weights, &index, layer)?;

            if capture_layer(layer) {
                let (_, pre_o) = run_attention_block_with_pre_o(&weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                add_pre_o_stats(
                    &mut stats[layer],
                    &pre_o,
                    weights.arch.num_q_heads_for_layer(layer),
                    weights.arch.head_dim_for_layer(layer),
                    args.max_positions,
                );
            }

            {
                let ffn = WeightFfn { weights: &weights };
                if let Some((h_new, _, _)) = run_layer_with_ffn(
                    &weights,
                    &h,
                    layer,
                    &ffn,
                    false,
                    ple_inputs.get(layer),
                    None,
                ) {
                    h = h_new;
                }
            }

            remove_layer_tensors(&mut weights, inserted);
        }
    }

    let mut heads = Vec::new();
    for &layer in &layers {
        let head_dim = weights.arch.head_dim_for_layer(layer);
        for (head, stat) in stats[layer].iter().enumerate() {
            heads.push(HeadReport {
                layer,
                head,
                head_dim,
                stats: stat.finish(),
            });
        }
    }

    let report = CaptureReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen: prompts.len(),
        layers,
        max_positions: args.max_positions,
        heads,
    };

    let out_path = args.out.join("stage0_pre_o_stats.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

fn add_pre_o_stats(
    stats: &mut [RunningHeadStats],
    pre_o: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    max_positions: Option<usize>,
) {
    let positions = max_positions
        .map(|n| n.min(pre_o.nrows()))
        .unwrap_or_else(|| pre_o.nrows());
    for pos in 0..positions {
        for head in 0..num_heads {
            let start = head * head_dim;
            let end = start + head_dim;
            let row = pre_o.slice(s![pos, start..end]);
            if let Some(values) = row.as_slice() {
                stats[head].add(values);
            }
        }
    }
}

fn load_prompts(
    path: &PathBuf,
    max_prompts: Option<usize>,
) -> Result<Vec<PromptRecord>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut prompts = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        prompts.push(serde_json::from_str::<PromptRecord>(line)?);
        if max_prompts.is_some_and(|n| prompts.len() >= n) {
            break;
        }
    }
    Ok(prompts)
}

fn insert_q4k_layer_tensors(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    layer: usize,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let attn = index
        .attn_q4k_layer_data(layer)
        .ok_or_else(|| format!("attn Q4K slices missing for layer {layer}"))?;
    let ffn = index
        .interleaved_q4k_layer_data(layer)
        .ok_or_else(|| format!("ffn Q4K slices missing for layer {layer}"))?;

    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let head_dim = arch.head_dim_for_layer(layer);
    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;
    let intermediate = index.num_features(layer);

    let q_key = arch.attn_q_key(layer);
    let k_key = arch.attn_k_key(layer);
    let v_key = arch.attn_v_key(layer);
    let o_key = arch.attn_o_key(layer);
    let gate_key = arch.ffn_gate_key(layer);
    let up_key = arch.ffn_up_key(layer);
    let down_key = arch.ffn_down_key(layer);

    weights.tensors.insert(
        q_key.clone(),
        dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden).into_shared(),
    );
    weights.tensors.insert(
        k_key.clone(),
        dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden).into_shared(),
    );
    weights.tensors.insert(
        v_key.clone(),
        dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden).into_shared(),
    );
    weights.tensors.insert(
        o_key.clone(),
        dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim).into_shared(),
    );
    weights.tensors.insert(
        gate_key.clone(),
        dequantize_matrix(ffn[0].0, ffn[0].1, intermediate, hidden).into_shared(),
    );
    weights.tensors.insert(
        up_key.clone(),
        dequantize_matrix(ffn[1].0, ffn[1].1, intermediate, hidden).into_shared(),
    );

    let inter_padded = intermediate.div_ceil(larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS)
        * larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let w_down = if inter_padded != intermediate {
        let w = dequantize_matrix(ffn[2].0, ffn[2].1, hidden, inter_padded);
        w.slice(s![.., ..intermediate]).to_owned()
    } else {
        dequantize_matrix(ffn[2].0, ffn[2].1, hidden, intermediate)
    };
    weights
        .tensors
        .insert(down_key.clone(), w_down.into_shared());

    Ok(vec![q_key, k_key, v_key, o_key, gate_key, up_key, down_key])
}

fn remove_layer_tensors(weights: &mut larql_inference::ModelWeights, keys: Vec<String>) {
    for key in keys {
        weights.tensors.remove(&key);
    }
}

fn dequantize_matrix(bytes: &[u8], format: &str, rows: usize, cols: usize) -> Array2<f32> {
    let n = rows * cols;
    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let padded = n.div_ceil(block) * block;
    let info = larql_vindex::quant::registry::lookup(format)
        .unwrap_or_else(|| panic!("unsupported quant format in vindex: {format}"));
    let floats =
        (info.dequantize)(bytes, padded).unwrap_or_else(|e| panic!("{format} dequant failed: {e}"));
    let truncated = if floats.len() > n {
        floats[..n].to_vec()
    } else {
        floats
    };
    Array2::from_shape_vec((rows, cols), truncated).expect("shape mismatch dequantising matrix")
}

fn parse_layer_spec(spec: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let (a, b) = part
                .split_once('-')
                .ok_or_else(|| format!("invalid range: {part}"))?;
            let start: usize = a.parse()?;
            let end: usize = b.parse()?;
            layers.extend(start..=end);
        } else {
            layers.push(part.parse()?);
        }
    }
    Ok(layers)
}
