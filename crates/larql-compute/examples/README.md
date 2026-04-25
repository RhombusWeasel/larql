# larql-compute examples

Nine examples in three groups. Run any with:

```
cargo run --release --features metal -p larql-compute --example <name>
```

## Demos — show the API

| Example | What it does |
|---|---|
| `demo_basic` | Auto-detects the best backend, calls `matmul_transb` and a Q4 matvec. The 5-line "hello, world" of the crate. |
| `demo_architecture` | Guided tour of the major design points — `ComputeBackend` trait, `KernelHandle`, `quant_matvec`, `Capability`. Useful as a code-driven crate intro. |
| `demo_ridge_solve` | `ridge_decomposition_solve` — the closed-form ridge solve that underlies MEMIT-style weight edits. Linalg-side, no Metal needed. |

## Compares — full-pipeline benchmarks

These measure **end-to-end** decode/generation throughput. Different
surface from `benches/quant_matvec.rs` (which measures *kernel*-level
throughput). Run with `cargo run --release --features metal …`; they
print tok/s + per-stage breakdowns.

| Example | What it measures |
|---|---|
| `compare_decode` | Q4_K decode latency through `decode_token` with KV cache. The production decode path. |
| `compare_formats` | Q4_KF (pre-baked scales) vs Q4_K vs Q8 — quant-format tradeoff inside the same model geometry. |
| `compare_generation` | End-to-end token generation throughput — the headline tok/s figure. |
| `compare_ollama` | Head-to-head LARQL vs Ollama on the same machine, same model. The external benchmark. |
| `compare_pipeline` | Q4_K fused-QKV vs Q8 fused-QKV through `full_pipeline_q4`. |

For *kernel*-level throughput regressions (the bug class
`q4_matvec_v4` 75 %-row drop fell into), use the criterion bench
suite instead:

```
make bench           # run all kernel benches
make bench-save      # record baseline
make bench-check     # fail if any cell regressed
```

See `benches/quant_matvec.rs`.

## Debug — diagnostic tools

| Example | What it does |
|---|---|
| `debug_decode_pipeline` | Per-stage buffer reads in the decode pipeline — useful for bisecting CPU/Metal divergence at a specific layer/stage. Pair with `LARQL_METAL_DUMP_LAYERS=<dir>` and the residual-diff test in `larql-inference`. |

## Why so few?

This crate used to ship 25 examples, mostly ad-hoc `Instant::now()`
profilers (`profile_*.rs`, `best_*.rs`) that have been superseded by
the proper criterion bench suite under `benches/`. Examples here
should either *teach the API* (the demos) or *answer a measurement
question that's outside criterion's surface* (the compares + debug).
