# Roadmap — larql-server / larql-router

## Current state (as of 2026-04-26)

- 2-shard local grid validated end-to-end on Gemma 4 26B-A4B (30 layers,
  inclusive layer ranges 0-14 + 15-29).
- W2 feature-major down retrofittable in-place via
  `larql convert add-feature-major-down --input <vindex>` (1.12 s for
  30 layers, 152 MB output).
- Live W2 surface on `GET /v1/stats.q4k_ffn`:
  `{cache_slots, cache_bytes, feature_major_down}`.
- `--warmup-hnsw` flag eager-builds HNSW across owned layers at boot
  (~325 ms for 15-layer shards on Gemma 26B).
- Grid memory profile (per-shard, single-machine): **9.1 GB RSS**,
  6.7 GB MALLOC_LARGE (gate f32 cache), `down_features_q4k.bin`
  resident at 0 K (capability, not yet exercised on dense path).

## Live perf snapshot (M3 Max, 2-shard grid, 26B-A4B)

| Operation | Cold | Warm |
|---|---|---|
| `walk-ffn` 1 layer (router) | 12.8 ms | **0.2–0.3 ms** |
| `walk-ffn` 6 layers fanout | — | **1.3 ms** |
| `walk-ffn` 12 layers fanout | 64 ms | 2.6 ms |
| `walk-ffn` 24 layers fanout | 75 ms | 5.0 ms |
| `walk-ffn` 30 layers (full) | 30 ms | **5.9 ms** |
| `walk` (gate KNN, 30L) | — | 8.4 ms |
| 8-way concurrent × 15L fan-out | 112 ms wall | ~1070 layer-evals/sec |

P99 under 8-way contention: 24 ms.

---

## P0: Active

Nothing critical-path is blocking right now.

## P1: Active

### G1. Cold-start profile
**Impact**: The first walk-ffn fan-out at fresh layers costs 30–75 ms
(vs 1–6 ms warm) — that's ~50× tax on first-request SLA. Need to
attribute the cost: page-in vs initial dequant vs allocator heat-up
vs request-scoped one-shot bookkeeping.
**Plan**:
1. Pin a deterministic cold-start: kill + relaunch shard, hit
   `walk-ffn` once per layer, capture per-call latency + RSS delta.
2. Strace/dtrace the first call to attribute time across (a) mmap
   page faults, (b) `q4k_ffn_q4k_dequant` first-call branches,
   (c) malloc/free churn, (d) tokio handler setup.
3. Decide which subsystem owns the win.
**Bench**: extend `larql-server/tests/` with a cold-start harness
(spawn → request → measure → repeat across N layers).
**Status**: open.

### G2. `/v1/warmup` endpoint
**Impact**: Lets operators pre-touch mmap pages and prime the dequant
caches at boot — converts the 30 ms first-fan-out into the warm
5.9 ms baseline immediately. Pairs with the existing `--warmup-hnsw`
flag for HNSW shards.
**Plan**:
1. Add `POST /v1/warmup` route accepting `{layers: [..], components: ["gate","up","down"], warmup_q4k: bool}`.
2. Walk owned layers, page in interleaved_q4k slices, optionally
   trigger `q4k_ffn_layer` once per layer to fully prime if
   `warmup_q4k=true`.
3. Add a `larql-server --warmup-walk-ffn` CLI flag that calls the
   endpoint internally at boot (matching `--warmup-hnsw`).
4. Document in README `Recommended setup for larql-server`.
**Status**: open.

### G3. Dual-host gRPC self-assembling grid
**Impact**: Today both shards run on the same host, so per-shard
RSS reduction doesn't materialise (mmap pages share). Real benefit
shows on N hosts where shard K only mmaps its layer slice. The
`larql-router --grid-port` mechanism exists; need to validate it
across two real machines and document the production setup.
**Plan**:
1. Smoke-test on two physical hosts (same LAN): router on host A,
   shards on hosts A+B with `--join grpc://routerA:PORT --grid-key
   <secret>`.
2. Measure cross-host fan-out latency vs same-host (TCP RTT impact
   on per-layer cost).
3. README: replace single-host `--shards` recipe with a "production
   dual-host" section using `--grid-port` + `--join`.
4. Stress: kill one shard mid-request, verify the router fails
   gracefully and re-routes on next call.
**Status**: open. The gRPC layer + `--grid-port` flag already exist.

## P2: Forward-looking

### G4. mmap residency control endpoint
**Impact**: For long-running shards under memory pressure, expose
`POST /v1/mmap/advise {layers, advice: "willneed"|"dontneed"}` so
operators can trim RSS or pre-warm specific layer ranges without
restarting.

### G5. Per-shard expert routing
**Impact**: For DeepSeek-V3+/Kimi K-class models (1k+ experts), shard
by expert ID within a layer rather than by layer range. Needs an
`ExpertRoute` message type in `larql-router-protocol` and
GridState dispatch updates. Mentioned in larql-vindex P2.

### G6. Live router-shard topology change
**Impact**: Today shards are static (`--shards` flag at router boot).
For ops convenience, expose `POST /v1/router/shards` (admin-gated)
to add/remove a shard without restarting the router. Pair with
`--grid-port` health checks.

---

## Completed

### 2026-04-26 — W2 retrofit + grid validation

| Item | Outcome |
|---|---|
| `--warmup-hnsw` flag | Eager-builds HNSW across owned layers at boot via `warmup_hnsw_all_layers()`. Reports correct owned-layer count under `--layers`. |
| Boot log: W2 status | `Down features Q4K: loaded (W2 — per-feature decode skips q4k_ffn_layer cache)` when `down_features_q4k.bin` is present. |
| `/v1/stats.q4k_ffn` field | `{cache_slots, cache_bytes, feature_major_down}` — operators can verify W2 active + cache empty in steady state. |
| `larql convert add-feature-major-down` | New CLI subcommand. Retrofits an existing Q4K vindex without re-quantising the rest. 30 layers / 152 MB / 1.12 s on Gemma 26B. Idempotent. |
| Live grid validation | 2-shard layer-range split (0-14 + 15-29) on real 26B vindex, full fan-out via router, 8-way concurrent stress, 0.2 ms warm per-layer, 5.9 ms full-30-layer fan-out. |

### Pre-2026-04-26 — foundations (already in place)

- HTTP API: `/v1/walk`, `/v1/walk-ffn`, `/v1/stats`, `/v1/health`,
  `/v1/infer`, `/v1/insert`, `/v1/expert/{layer}/{id}`, etc.
- `--layers START-END` shard slicing (mmap pages outside range stay
  paged out, RSS proportional to shard size).
- `--max-q4k-cache-layers` LRU bound on the legacy Q4K dequant cache.
- `--ffn-only` / `--embed-only` mode flags.
- gRPC self-assembling grid (`--grid-port` / `--join` / `--grid-key`).
- Bench rig daemon-aware (`larql-vindex` benches refuse if a server
  shares the host; override with `LARQL_BENCH_ALLOW_DAEMONS=1`).
