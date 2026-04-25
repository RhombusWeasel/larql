# Roadmap — larql-vindex

## Current State

- 146 tests passing, 0 build warnings
- 3 storage formats: f32, Q8, Q4_K/Q6_K (Ollama-compatible)
- Mmap zero-copy with adaptive residency
- HNSW graph index for sub-linear KNN
- Patch system for editable knowledge

## P0: Decode-path performance

Items raised by the 2026-04-25 perf audit (see PERFORMANCE.md and the
`gpu_forward_gap` memo). Vindex-side only — Metal kernel work lives in
larql-compute's roadmap.

### Bound the Q4_K dequant cache (LRU like gate cache)
**Impact**: Caps CPU-fallback RAM at a configurable budget (worst-case
today: 10.7 GB on 4B / ~110 GB on 31B if all layers cache fully)
**Effort**: Low
**Status**: Not started

**Finding from 2026-04-25 audit**: the Metal hot path never populates
`q4k_ffn_cache` (`larql bench --backends metal -v` reports
`q4k_ffn_cache after larql-metal: 0 populated slots, 0.0 MB`). The
full-K Metal branch in `walk_ffn/sparse.rs:84-117` streams Q4_K bytes
through `q4k_matmul_transb` and bypasses `q4k_ffn_layer` entirely. The
dequant cache only fires in the CPU per-position fallback at
`walk_ffn/sparse.rs:145` (`hits.len() >= 512 && down_native.is_none()`)
— and there it's a 30× win because one 614 ms layer-dequant is
amortised across thousands of feature reads per token.

So the cache is correct, not pathological. What's missing is an upper
bound: a long-running CPU-only server can grow it to all 34 layers ×
105 MB on Gemma 3 4B (10.7 GB) or 60 layers × 1.85 GB on 31B (~110 GB).
Mirror the existing gate-cache pattern (`gate_cache_max_layers`,
`gate_cache_lru` in `index/core.rs` / `gate.rs:80`) for the Q4_K FFN
cache:

1. Add `q4k_ffn_cache_max_layers` (atomic) + `q4k_ffn_cache_lru`
   (Mutex<VecDeque<usize>>) to `VectorIndex`.
2. On insert in `q4k_ffn_layer`, push the layer to the LRU and evict
   from the front when the cap is exceeded; clear the evicted layer's
   slot triple.
3. Expose `set_q4k_ffn_cache_max_layers(n)` + a `--max-q4k-cache-layers
   N` flag on `larql serve` and any other long-running CLI.
4. Default cap = 0 (unbounded — keeps current behaviour). Recommend 8
   for a CPU-only Gemma 3 4B server (≈ 840 MB ceiling for the down
   leg; gate/up dequant aren't on the hot path).

### Q4_K interleaved madvise + per-layer prefetch
**Impact**: Free win on cold-page first-token latency; small steady-state
**Effort**: Low
**Status**: Not started

`load_interleaved_q4k` (`walk.rs:235`) opens with `mmap_demand_paged`
(MADV_RANDOM) but the decode loop reads every layer once per token in
order. The Q4_0 path already has `prefetch_interleaved_q4_layer`
(`walk.rs:649`) issuing MADV_WILLNEED for layer N+1 while N computes —
mirror it for Q4_K (`prefetch_interleaved_q4k_layer`) and call it from
the inference walk. Consider switching Q4_K's initial advise to
SEQUENTIAL since the access pattern is linear over layers within a
token.

### Audit `save_gate_vectors` 1.4 → 2.0 ms regression
**Impact**: 40% slip on a build-time hot path
**Effort**: Low
**Status**: Not started

`save_load/save_gate_vectors` was 1.4 ms in 2026-04-07's PERFORMANCE.md,
1.99 ms in 2026-04-25 criterion run on the same dimensions. Bisect via
`git log -p crates/larql-vindex/src/format/save.rs` since 2026-04-07.

### Lift gate KNN out of brute-force on the decode hot path
**Impact**: 64-expert MoE 230 → ~30 ms gate KNN/layer (HNSW table)
**Effort**: Medium
**Status**: Index built, not wired

`index/hnsw.rs` exists and the `q4k_vs_f32` bench already shows HNSW
beats brute force at 1024–28K features. Decode currently calls
`gate_walk` → `gate_knn` (full BLAS gemv). For dense 4B–8B the gemv
ceiling is fine; for high-expert MoE it dominates. Wire HNSW behind an
opt-in flag on `VectorIndex` and validate ranking parity vs brute on a
held-out feature set before defaulting on.

### Bench rig hygiene — fail fast under host contention
**Impact**: Makes regression detection meaningful again
**Effort**: Low
**Status**: Not started

`production_knn_per_layer` swung 4.56 → 8.58 ms run-to-run on
2026-04-25 because `larql-server` (6 GB RSS) and `larql-router` were
sharing cores. Add a precondition to `vindex_scaling`: refuse to run
if `pgrep -f 'larql-(server|router)'` returns non-empty, and surface a
warning if `pmset -g therm` reports throttling. Move scaling to its
own `make bench-scaling` target so it doesn't run back-to-back with
`vindex_ops` (which leaves the M3 Max thermal budget cooked).

## P0: Support Cached Layer Decode

### Store pre-computed residuals for template-fixed layers (L0-12)
**Impact**: Enables 155+ tok/s decode (skip 13 of 21 layers)  
**Effort**: Medium  
**Status**: Not started (infrastructure ready — CachedLayerGraph in larql-inference)

The vindex needs to store cached residuals per template. During extraction, run one forward pass per template through L0-12 and save the output residual. At decode time, look up the cached residual instead of computing 13 layers.

### Wire Q4_K FFN consumption (interleaved_q4k.bin) — DONE
**Impact**: Match Ollama's exact FFN quantization  
**Effort**: Medium  
**Status**: ✅ Complete (2026-04-07)

Added `load_interleaved_q4k()`, `has_interleaved_q4k()`, `interleaved_q4k_mmap_ref()` to vindex.
Inference `predict_honest` now prefers Q4_K FFN (`interleaved_q4k.bin`) over Q4_0.
Format tag (`ffn_format`) passed through `FullPipelineLayer` to compute for shader dispatch.

### GGUF Q4_K format option (144 bytes vs 148 bytes)
**Impact**: Direct compatibility with llama.cpp weight files  
**Effort**: Low  
**Status**: Quantizer ready in larql-compute (`quantize_q4_k_gguf`)

Add option to store attention weights in GGUF-canonical 144-byte Q4_K format (packed scales+mins in 12 bytes) instead of our 148-byte format.

## P1: Production Hardening

### HuggingFace resolution in Vindexfile
**Effort**: Medium  
**Status**: TODO in `vindexfile/mod.rs:162`

FROM directive in Vindexfile should resolve `hf://user/repo` paths.

### Streaming extraction checkpoints
**Effort**: Medium  
**Status**: Not started

Save extraction progress between layers so interrupted builds can resume.

### Q4_K FFN in vindex
**Effort**: Low  
**Status**: Not started (Q4_0 interleaved exists)

Currently FFN gate/up/down stored as Q4_0. Switch to Q4_K (matching Ollama) for better precision at similar size.

## P2: Research

### Multi-model vindex
Store features from multiple models in one vindex. Compare representations across architectures.

### Incremental extraction
Add new layers/features to an existing vindex without full rebuild.

## Completed

| Item | Date | Impact |
|------|------|--------|
| Core VectorIndex with mmap | 2026-03 | Foundation |
| Gate KNN (brute-force + BLAS) | 2026-03 | Walk engine |
| Walk FFN (per-feature down/up vectors) | 2026-03 | Sparse inference |
| Binary down_meta format | 2026-03 | 5x compression vs JSONL |
| F16 storage + decode cache | 2026-03 | 2x smaller gate vectors |
| Interleaved layout (gate\|up\|down packed) | 2026-04 | Reduced TLB thrash |
| Q4_0 gate vectors + interleaved | 2026-04 | 7x smaller gates |
| HNSW graph index | 2026-04 | Sub-linear KNN |
| Adaptive residency (pin/evict) | 2026-04 | Memory budget management |
| Patch system (PatchedVindex) | 2026-04 | Editable knowledge |
| MoE expert routing | 2026-04 | Mixtral/DeepSeek support |
| Q4_K/Q6_K attention weights | 2026-04 | Ollama-compatible |
| Q8 attention weights | 2026-04 | Higher precision option |
| Streaming extraction (mmap, per-layer) | 2026-04 | ~2 GB peak RAM |
| Safety doc for mmap_optimized | 2026-04-07 | Clippy compliance |
| VindexPatch::is_empty() | 2026-04-07 | API completeness |
| Q4_K FFN loader + wiring | 2026-04-07 | `interleaved_q4k.bin` end-to-end |
| Quantizer single source of truth | 2026-04-07 | Builder uses larql-compute (ADR-008) |
| Example cleanup (13→11) | 2026-04-07 | Removed Q4_0 attn + Q4_0 interleaved |
| 8 ADRs documented | 2026-04-07 | All major decisions recorded |
| PERFORMANCE.md + format alignment | 2026-04-07 | Fresh benchmarks, verified pipeline |
