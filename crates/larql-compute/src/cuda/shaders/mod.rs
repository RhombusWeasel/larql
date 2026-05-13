//! CUDA kernel source strings and NVRTC compilation.
//!
//! Each kernel lives in its own submodule as a `pub const SHADER: &str`
//! containing the CUDA C source. `compile_all_shaders()` concatenates
//! them and compiles to PTX via NVRTC at `CudaBackend::new()` time.

pub mod activation;
pub mod causal_attention;
pub mod common;
pub mod q4k_matmul;
pub mod f16_gemv;
pub mod f32_argmax;
pub mod f32_gemv;
pub mod gegelu;
pub mod kv_attention;
pub mod layer_norm;
pub mod q4_matvec;
pub mod q4_f32_matvec;
pub mod q4_vecmat;
pub mod q4k_matvec;
pub mod q4k_ffn_gate_up;
pub mod q4k_geglu_down;
pub mod q4k_qkv_proj;
pub mod q6k_matvec;
pub mod q8_attn_proj;
pub mod q8_matvec;
pub mod qk_norm;
pub mod qk_norm_rope_fused;
pub mod quantize_q8;
pub mod residual_norm;
pub mod rms_norm;
pub mod rope;
pub mod sgemm;
pub mod v_norm;

use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};

/// Compile all CUDA kernel source into a PTX module.
///
/// Concatenates the common header with all kernel sources and
/// compiles via NVRTC. Returns the PTX data suitable for
/// `CudaContext::load_module()`.
pub fn compile_all_shaders() -> Result<Ptx, cudarc::nvrtc::CompileError> {
    let topk_src = f32_argmax::topk_shader_source();

    let all_src = format!(
        "{header}\n\
         {warp}\n\
         {gemv}\n\
         {argmax}\n\
         {topk}\n\
         {rms}\n\
         {residual}\n\
         {scale}\n\
         {q4k}\n\
         {q4k_8sg}\n\
         {q4k_s32}\n\
         {q4k_gate_up}\n\
         {q4k_geglu_silu}\n\
         {q4k_geglu_gelu}\n\
         {q6k}\n\
         {q6k_8sg}\n\
         {rope_scalar}\n\
         {rope_apply}\n\
         {rope_batched}\n\
         {rope_batched_qk}\n\
         {rope_q}\n\
         {rope_k}\n\
         {qk_norm}\n\
         {qk_norm_rope}\n\
         {kv_append}\n\
         {kv_attend}\n\
         {kv_fused}\n\
         {v_norm}\n\
         {gegelu_silu}\n\
         {gegelu_gelu}\n\
         {silu}\n\
         {gelu_tanh}\n\
         {res_norm}\n\
         {res_norm_store}\n\
         {post_attn_res_norm}\n\
         {post_ffn_res_add}\n\
         {rms_norm_q8}\n\
         {residual_norm_q8}\n\
         {q4k_q6k_qkv}\n\
         {q4k_q6k_qkv_normed}\n\
         {layer_norm}\n\
         {quantize_q8}\n\
         {q4_matvec}\n\
         {q4_vecmat}\n\
         {q4_f32_matvec}\n\
         {q8_matvec}\n\
         {q8_qkv_proj}\n\
         {q8_proj_rope}\n\
         {f16_gemv}\n\
         {sgemm}\n\
         {sgemm_transb}\n\
         {causal_attn}\n\
         {q4k_matmul}\n",
        header = common::HEADER,
        warp = common::WARP_REDUCTIONS,
        gemv = f32_gemv::SHADER,
        argmax = f32_argmax::ARGMAX_SHADER,
        topk = topk_src,
        rms = rms_norm::SHADER,
        residual = rms_norm::RESIDUAL_ADD_SHADER,
        scale = rms_norm::SCALE_VECTOR_SHADER,
        q4k = q4k_matvec::SHADER,
        q4k_8sg = q4k_matvec::SHADER_8SG,
        q4k_s32 = q4k_matvec::SHADER_S32,
        q4k_gate_up = q4k_ffn_gate_up::SHADER,
        q4k_geglu_silu = q4k_geglu_down::GEGLU_SILU_SHADER,
        q4k_geglu_gelu = q4k_geglu_down::GEGLU_GELU_TANH_SHADER,
        q6k = q6k_matvec::SHADER,
        q6k_8sg = q6k_matvec::SHADER_8SG,
        rope_scalar = rope::ROPE_AT_POS_SHADER,
        rope_apply = rope::ROPE_APPLY_SHADER,
        rope_batched = rope::ROPE_BATCHED_SHADER,
        rope_batched_qk = rope::ROPE_BATCHED_QK_SHADER,
        rope_q = rope::ROPE_Q_SHADER,
        rope_k = rope::ROPE_K_SHADER,
        qk_norm = qk_norm::SHADER,
        qk_norm_rope = qk_norm_rope_fused::SHADER,
        kv_append = kv_attention::KV_APPEND_SHADER,
        kv_attend = kv_attention::KV_ATTEND_SHADER,
        kv_fused = kv_attention::KV_APPEND_ATTEND_FUSED_SHADER,
        v_norm = v_norm::SHADER,
        gegelu_silu = gegelu::GEGELU_SILU_SHADER,
        gegelu_gelu = gegelu::GEGELU_GELU_TANH_SHADER,
        silu = activation::SILU_SHADER,
        gelu_tanh = activation::GELU_TANH_SHADER,
        res_norm = residual_norm::RESIDUAL_NORM_SHADER,
        res_norm_store = residual_norm::RESIDUAL_NORM_STORE_SHADER,
        post_attn_res_norm = residual_norm::POST_ATTN_RESIDUAL_NORM_STORE_SHADER,
        post_ffn_res_add = residual_norm::POST_FFN_NORM_RESIDUAL_ADD_SHADER,
        rms_norm_q8 = residual_norm::RMS_NORM_Q8_SHADER,
        residual_norm_q8 = residual_norm::RESIDUAL_NORM_Q8_SHADER,
        q4k_q6k_qkv = q4k_qkv_proj::SHADER,
        q4k_q6k_qkv_normed = q4k_qkv_proj::SHADER_NORMED,
        layer_norm = layer_norm::SHADER,
        quantize_q8 = quantize_q8::SHADER,
        q4_matvec = q4_matvec::SHADER,
        q4_vecmat = q4_vecmat::SHADER,
        q4_f32_matvec = q4_f32_matvec::SHADER,
        q8_matvec = q8_matvec::SHADER,
        q8_qkv_proj = q8_attn_proj::SHADER,
        q8_proj_rope = q8_attn_proj::PROJ_ROPE_SHADER,
        f16_gemv = f16_gemv::SHADER,
        sgemm = sgemm::SGEMM_SHADER,
        sgemm_transb = sgemm::SGEMM_TRANSB_SHADER,
        causal_attn = causal_attention::SHADER,
        q4k_matmul = q4k_matmul::SHADER,
    );

    // NVRTC needs the CUDA include path to find headers like <cuda_fp16.h>.
    let cuda_include = std::env::var("CUDA_PATH")
        .map(|p| format!("{}/include", p))
        .ok()
        .or_else(|| {
            let path = "/opt/cuda/include";
            std::path::Path::new(path).is_dir().then(|| path.to_string())
        })
        .or_else(|| {
            let path = "/usr/local/cuda/include";
            std::path::Path::new(path).is_dir().then(|| path.to_string())
        });

    let includes: Vec<String> = cuda_include.into_iter().collect();

    let opts = CompileOptions {
        use_fast_math: Some(false),
        include_paths: includes,
        ..Default::default()
    };
    compile_ptx_with_opts(all_src, opts)
}