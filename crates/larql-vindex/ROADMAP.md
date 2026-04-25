# Roadmap — larql-vindex

## Current state (as of 2026-04-25)

- **321 tests passing** on `larql-vindex` (173 unit + 148 integration);
  211 on `larql-models`. Workspace builds clean.
- **Folder layout decomposed**:
  - `index/{storage,compute,mutate}/` — substores, KNN dispatch, mutation
  - `format/{huggingface,weights,filenames,fp4_codec,…}/`
  - `engine/` (was `storage/`) — StorageEngine + epoch + MEMIT
  - No `.rs` file > 750 lines (down from 1366 monolith)
- **Quant dispatch via `quant::registry`** — adding the next K-quant is
  one table entry plus codec functions; ~3-file edit.
- **Filename literals centralised** in `format::filenames` (252+
  occurrences → one constant module).
- **`VectorIndex` god struct decomposed** into four typed substores
  (`GateStore`, `FfnStore`, `ProjectionStore`, `MetadataStore`). Adding
  a new field is one edit in the relevant store.
- **5 storage formats**: f32, f16, Q4_0, Q4_K/Q6_K (Ollama-compatible),
  Q8, FP4/FP8 (exp 26).
- Mmap zero-copy with adaptive residency.
- HNSW graph index wired into `gate_knn` (opt-in via `--hnsw`).
- Q4_K dequant cache LRU-bounded via `--max-q4k-cache-layers`.
- Patch system for editable knowledge (`PatchedVindex` overlay).
- `make coverage` + `make coverage-summary` (cargo-llvm-cov).
- Bench rig daemon-aware (`make bench-vindex-scaling` refuses if
  `larql-server` / `larql-router` are running on the host).

---

## P0: Active

Nothing in P0 is currently blocking — all known critical-path issues
have landed.

## P1: Active

### Split `config/types.rs` (628 L, 15 unrelated types)
**Impact**: Future quant / MoE / FP4 additions scoped to one file
**Effort**: Medium
**Status**: ⏸ Deferred from 2026-04-25 round-2 cleanup — needs careful
inter-type reference mapping. `VindexConfig` references `LayerBands`,
`Fp4Config`, `VindexModelConfig`, `VindexLayerInfo` across what would
become four files; safe split requires building the type-reference
graph first.

Proposed split:
- `config/index.rs` — `VindexConfig`, `VindexSource`, `ExtractLevel`,
  `VindexLayerInfo`, `DownMetaRecord`, `DownMetaTopK`
- `config/quantization.rs` — `QuantFormat`, `Precision`,
  `ProjectionFormat`, `Projections`, `Fp4Config`
- `config/model.rs` — `VindexModelConfig`, `MoeConfig`
- `config/compliance.rs` — `ComplianceGate`, `LayerBands`

`mod.rs` re-exports the previous flat surface for back-compat.

### Cached layer decode for template-fixed layers (L0–12) — parked
**Impact**: 155+ tok/s decode (skip 13 of 21 layers)
**Effort**: Medium
**Status**: ⏸ Parked — depends on upstream work that isn't ready yet.
Don't start until the prerequisite lands. Keep `CachedLayerGraph` in
`larql-inference` as the integration point.

### HuggingFace resolution in Vindexfile
**Effort**: Medium
**Status**: TODO in `vindexfile/mod.rs:162`

FROM directive in Vindexfile should resolve `hf://user/repo` paths.

### Streaming extraction checkpoints
**Effort**: Medium
**Status**: Not started

Save extraction progress between layers so interrupted builds can
resume.

### GGUF Q4_K format option (144 bytes vs 148 bytes)
**Impact**: Direct compatibility with llama.cpp weight files
**Effort**: Low
**Status**: Quantizer ready in `larql-compute` (`quantize_q4_k_gguf`)

Add option to store attention weights in GGUF-canonical 144-byte Q4_K
format (packed scales+mins in 12 bytes) instead of our 148-byte
format.

## P2: Forward-looking

### Parallelize gate KNN for batch inference
**Impact**: 2–4× prefill throughput on multi-token batches
**Effort**: Medium
**Status**: Forward-looking

`gate_matmul` already runs across all positions in one BLAS call but
the per-position top-K selection is sequential. Rayon-shard the
selection across rows (or fold into a single batched argpartial). Not
urgent — Metal kernel work (Q6_K dequant + 8-rows/TG) is the bigger
throughput lever.

### `VindexStorage` trait abstraction
**Impact**: Lets Redis / S3 / GPU-residency backends plug in
**Effort**: Medium
**Status**: Forward-looking

The substore extraction got most of the way there. Formalise a
sealed `VindexStorage` trait (mmap-agnostic row accessor) so Q4K row
reads can route through Redis-cached or S3-buffered backends without
walk-kernel changes.

### Expert-level sharding protocol
**Impact**: Unlocks > 256-expert MoE sharding-within-layer
**Effort**: Medium
**Status**: Forward-looking

Today `larql-router` shards by layer, not by expert ID within a
layer. For DeepSeek-V4-class models (1K+ experts) experts need to
shard across servers. Add an `ExpertRoute` message type to
`larql-router-protocol` and wire `GridState` dispatch.

### Q5_K / Q3_K / BF16 quant additions
**Effort**: Small per format (≈ 3 files thanks to the registry)
**Status**: Not yet needed — add when a target model demands it

Path: implement codec functions in `larql-models/src/quant/ggml/`,
add one entry to `QUANT_FORMATS` in `quant::registry`, add match arm
in `larql-compute::backend::quant_matvec`. Verified by the round-2
audit.

### Multi-model vindex
**Status**: Research

Store features from multiple models in one vindex. Compare
representations across architectures.

### Incremental extraction
**Status**: Research

Add new layers / features to an existing vindex without full rebuild.

---

## Won't fix

- **`detect.rs` (1391 L) split** in `larql-models` — cohesive single
  entry point dispatching to 12 architectures. Splitting fragments
  without modularity gain. Reconsider when a second detection system
  emerges (auto-discovery from model ID, multi-modal config).

---

## Completed

### 2026-04-25 — second audit + round-2 cleanup

| Item | Outcome |
|------|---------|
| Add 8 missing filename constants | `LM_HEAD_BIN` (10×), `GATE_VECTORS_FP4_BIN` (7×), `DOWN_FEATURES_FP8_BIN` (5×), `UP_FEATURES_FP4_BIN` (4×), 4× attn manifests |
| Migrate ~20 unmigrated `Q4_K`/`Q6_K` dispatch sites | Most in `larql-inference` (q4k_forward, walk_ffn, pipeline_layer); routed through `quant::registry::lookup` |
| Replace 2× `unwrap_or("Q4_K")` silent fallbacks | `attn.rs`, `ffn_store.rs` — now error on missing/unknown format tags |
| `storage/` → `engine/` rename | Top-level lifecycle dir; back-compat alias `pub use engine as storage;` |
| Duplicate `fp4_storage.rs` rename | `format/fp4_codec.rs` (codec) + `index/storage/fp4_store.rs` (runtime store) |
| Merge `ffn_data.rs` into `ffn_store.rs` | Struct + impls + Clone in one file |
| Inline `gate_trait.rs` (198 L) | Block moved into `index/core.rs` |
| `accessors.rs` → `gate_accessors.rs` | Disambiguates the gate-specific accessors |

### 2026-04-25 — first audit + round-1 cleanup

| Item | Outcome |
|------|---------|
| `quant::registry` — single dispatch table | Q5_K addition drops from 8 files to 3; deletes ~12 silent-fallback `_ => None` arms |
| `format::filenames` — 19 (then 27) constants | 244 filename literals consolidated |
| Folder split: `index/{storage,compute,mutate}/` | 11 files moved; backwards-compat aliases |
| `gate.rs` (992) split | → `compute/gate_knn.rs` (615) + `storage/gate_store.rs` (446) |
| `walk.rs` (862) split | → `storage/ffn_store.rs` (720) + `compute/q4k_dispatch.rs` (168) |
| `VectorIndex` god struct → 4 substores | `GateStore` / `FfnStore` / `ProjectionStore` / `MetadataStore` |
| `format/huggingface.rs` (1366) split | → `huggingface/{mod,download,publish,discovery}.rs` |
| `format/weights/write.rs` (1249) split | → `weights/{write_f32,write_q4k}.rs` |
| `larql-models/src/quant/ggml.rs` (1352) split | → `quant/ggml/{mod,legacy,q4_k,q6_k,quantize}.rs` |
| Naming pass `Q4k` → `Q4K` | 8 occurrences across 24 files; serialised tags unchanged |
| Coverage tooling | `make coverage` + `make coverage-summary` (cargo-llvm-cov) |
| GGML round-trip tests | Q4_0 / Q4_K / Q6_K with frozen tolerance bounds |
| Golden save/load test | Deterministic save, KNN bit-exact across save/load, mmap zero-copy invariant, HNSW post-reload |
| HNSW + Q4K cache benches | `benches/hnsw_decode.rs` + `benches/q4k_cache.rs` |
| README + PERFORMANCE.md refresh | Test counts, end-to-end Q4K decode timings |

### 2026-04-25 — perf audit fixes

| Item | Outcome |
|------|---------|
| Bound the Q4_K dequant cache (LRU) | `set_q4k_ffn_cache_max_layers` + `--max-q4k-cache-layers N` flag on `larql serve` |
| Q4_K interleaved madvise + per-layer prefetch | `prefetch_interleaved_q4k_layer` mirrors the Q4_0 path; wired into `walk_ffn/sparse.rs` |
| HNSW on the decode hot path | Zero-copy view for f32-mmap layers (was cloning ~100 MB / query); abs-magnitude ranking parity (oversample 4× + re-rank); `--hnsw` + `--hnsw-ef-search` flags |
| Bench rig hygiene | Refuses if `larql-(server\|router)` daemons are alive; `LARQL_BENCH_ALLOW_DAEMONS=1` override; `make bench-vindex` vs `bench-vindex-scaling` split |
| `save_gate_vectors` regression check | False alarm — criterion p=0.21, no statistically detectable change |

### 2026-04-07 — first iteration

| Item | Outcome |
|------|---------|
| Q4_K FFN loader + wiring | `interleaved_q4k.bin` end-to-end; inference `predict_honest` prefers Q4_K over Q4_0 |
| Quantizer single source of truth | Builder uses `larql-compute` (ADR-008) |
| Example cleanup (13 → 11) | Removed Q4_0 attn + Q4_0 interleaved |
| 8 ADRs documented | All major decisions recorded |
| PERFORMANCE.md + format alignment | Fresh benchmarks, verified pipeline |
| Safety doc for `mmap_optimized` | Clippy compliance |
| `VindexPatch::is_empty()` | API completeness |

### 2026-03 / 2026-04 — foundation

| Item | Date | Impact |
|------|------|--------|
| Core `VectorIndex` with mmap | 2026-03 | Foundation |
| Gate KNN (brute-force + BLAS) | 2026-03 | Walk engine |
| Walk FFN (per-feature down/up vectors) | 2026-03 | Sparse inference |
| Binary down_meta format | 2026-03 | 5× compression vs JSONL |
| F16 storage + decode cache | 2026-03 | 2× smaller gate vectors |
| Interleaved layout (gate\|up\|down packed) | 2026-04 | Reduced TLB thrash |
| Q4_0 gate vectors + interleaved | 2026-04 | 7× smaller gates |
| HNSW graph index | 2026-04 | Sub-linear KNN |
| Adaptive residency (pin/evict) | 2026-04 | Memory budget management |
| Patch system (`PatchedVindex`) | 2026-04 | Editable knowledge |
| MoE expert routing | 2026-04 | Mixtral/DeepSeek support |
| Q4_K/Q6_K attention weights | 2026-04 | Ollama-compatible |
| Q8 attention weights | 2026-04 | Higher precision option |
| Streaming extraction (mmap, per-layer) | 2026-04 | ~2 GB peak RAM |
