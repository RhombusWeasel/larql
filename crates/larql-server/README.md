# larql-server

HTTP server for vindex knowledge queries and inference. Loads a vindex and serves it over the network. No GPU, no ML framework, no Python. One binary.

```bash
larql serve output/gemma3-4b.vindex --port 8080
# Serving google/gemma-3-4b-it (348K features, 1967 probe-confirmed)
# Listening: http://0.0.0.0:8080
```

```bash
curl "http://localhost:8080/v1/describe?entity=France"
# {"entity":"France","edges":[{"relation":"capital","target":"Paris","gate_score":1436.9,"layer":27,"source":"probe"}, ...]}
```

## Features

- **Browse endpoints** — DESCRIBE, WALK, SELECT, RELATIONS, STATS (no weights needed)
- **Inference** — full forward pass with WalkFfn (weights lazy-loaded on first request)
- **Relation labels** — probe-confirmed labels from `feature_labels.json` in DESCRIBE responses
- **Patch overlay** — apply knowledge patches via API without modifying base files
- **Multi-model serving** — serve multiple vindexes from a directory
- **HuggingFace support** — load vindexes directly from `hf://` paths
- **API key auth** — optional Bearer token authentication
- **TLS** — native HTTPS via rustls
- **Concurrency limit** — configurable max concurrent requests
- **CORS** — enable for browser-based clients
- **REPL integration** — `USE REMOTE "http://..."` in the LQL REPL

## Quickstart

```bash
# Build
cargo build --release -p larql-server

# Serve a local vindex
larql-server output/gemma3-4b-v2.vindex --port 8080

# Or via the CLI wrapper
larql serve output/gemma3-4b-v2.vindex --port 8080

# From HuggingFace
larql serve "hf://chrishayuk/gemma-3-4b-it-vindex" --port 8080

# Multi-model
larql serve --dir ./vindexes/ --port 8080

# With auth + TLS
larql serve output/gemma3-4b.vindex --api-key "sk-abc123" --tls-cert cert.pem --tls-key key.pem
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `<VINDEX_PATH>` | Path to .vindex directory or `hf://` URL | — |
| `--dir <DIR>` | Serve all .vindex directories in folder | — |
| `--port <PORT>` | Listen port | 8080 |
| `--host <HOST>` | Bind address | 0.0.0.0 |
| `--no-infer` | Disable inference (browse-only, saves memory) | false |
| `--cors` | Enable CORS headers | false |
| `--api-key <KEY>` | Require Bearer token auth (health exempt) | — |
| `--max-concurrent <N>` | Max concurrent requests | 100 |
| `--tls-cert <PATH>` | TLS certificate for HTTPS | — |
| `--tls-key <PATH>` | TLS private key for HTTPS | — |
| `--log-level <LEVEL>` | Logging level | info |

## API Endpoints

### Knowledge Endpoints (browse-only)

#### GET /v1/describe

Query all knowledge edges for an entity. Edges include probe-confirmed relation labels when `feature_labels.json` is present in the vindex.

```
GET /v1/describe?entity=France
GET /v1/describe?entity=France&band=all&verbose=true&limit=10&min_score=5.0
```

```json
{
  "entity": "France",
  "model": "google/gemma-3-4b-it",
  "edges": [
    {"relation": "capital", "target": "Paris", "gate_score": 1436.9, "layer": 27, "source": "probe", "also": ["Berlin", "Tokyo"]},
    {"target": "French", "gate_score": 35.2, "layer": 24},
    {"target": "Europe", "gate_score": 14.4, "layer": 25}
  ],
  "latency_ms": 12.3
}
```

| Param | Default | Description |
|-------|---------|-------------|
| `entity` | required | Entity name |
| `band` | `knowledge` | Layer band: syntax, knowledge, output, all |
| `verbose` | false | Include layer_min, layer_max, count per edge |
| `limit` | 20 | Max edges |
| `min_score` | 5.0 | Minimum gate score |

#### GET /v1/walk

Feature scan — which features fire for a prompt.

```
GET /v1/walk?prompt=The+capital+of+France+is&top=5
GET /v1/walk?prompt=Einstein&top=10&layers=24-33
```

```json
{
  "prompt": "The capital of France is",
  "hits": [
    {"layer": 27, "feature": 9515, "gate_score": 1436.9, "target": "Paris"},
    {"layer": 24, "feature": 4532, "gate_score": 26.1, "target": "French"}
  ],
  "latency_ms": 0.4
}
```

#### POST /v1/select

SQL-style edge query over down-projection metadata.

```json
POST /v1/select
{"entity": "France", "limit": 10, "order_by": "c_score", "order": "desc"}
```

```json
{
  "edges": [
    {"layer": 26, "feature": 8821, "target": "Paris", "c_score": 0.95}
  ],
  "total": 94,
  "latency_ms": 5.2
}
```

#### GET /v1/relations

List top tokens across knowledge layers.

```json
{
  "relations": [
    {"name": "Paris", "count": 94, "example": "Berlin"},
    {"name": "French", "count": 51, "example": "German"}
  ],
  "total": 512
}
```

#### GET /v1/stats

Model and index statistics.

```json
{
  "model": "google/gemma-3-4b-it",
  "family": "gemma3",
  "layers": 34,
  "features": 348160,
  "hidden_size": 2560,
  "layer_bands": {"syntax": [0, 13], "knowledge": [14, 27], "output": [28, 33]},
  "loaded": {"browse": true, "inference": true}
}
```

### Inference Endpoint

#### POST /v1/infer

Full forward pass with attention weights. Model weights are lazy-loaded on first request. Requires a vindex built with `--include-weights` (or extract level `inference`/`all`). Disabled when `--no-infer` is set.

```json
POST /v1/infer
{"prompt": "The capital of France is", "top": 5, "mode": "walk"}
```

| Field | Default | Description |
|-------|---------|-------------|
| `prompt` | required | Input text |
| `top` | 5 | Top-K predictions |
| `mode` | `walk` | `walk` (vindex FFN), `dense` (original FFN), `compare` (both) |

**Walk mode response:**

```json
{
  "prompt": "The capital of France is",
  "predictions": [
    {"token": "Paris", "probability": 0.9791},
    {"token": "the", "probability": 0.0042}
  ],
  "mode": "walk",
  "latency_ms": 210
}
```

**Compare mode response:**

```json
{
  "prompt": "The capital of France is",
  "walk": [{"token": "Paris", "probability": 0.9791}],
  "walk_ms": 210,
  "dense": [{"token": "Paris", "probability": 0.9801}],
  "dense_ms": 180,
  "latency_ms": 420
}
```

### Patch Endpoints

#### POST /v1/patches/apply

Apply a patch in-memory (does not modify base files).

```json
POST /v1/patches/apply
{"patch": {"version": 1, "base_model": "...", "operations": [...]}}
```

#### GET /v1/patches

List active patches.

#### DELETE /v1/patches/{name}

Remove a patch by description.

### Management Endpoints

#### GET /v1/health

Always accessible (exempt from API key auth).

```json
{"status": "ok", "uptime_seconds": 3600, "requests_served": 12450}
```

#### GET /v1/models

```json
{"models": [{"id": "gemma-3-4b-it", "path": "/v1", "features": 348160, "loaded": true}]}
```

## Authentication

When `--api-key` is set, all endpoints (except `/v1/health`) require a Bearer token:

```bash
larql serve output/gemma3-4b.vindex --api-key "sk-abc123"
```

```bash
curl -H "Authorization: Bearer sk-abc123" "http://localhost:8080/v1/describe?entity=France"
```

Requests without a valid token receive 401 Unauthorized.

## REPL Integration

The LQL REPL connects to a remote server transparently:

```sql
USE REMOTE "http://localhost:8080";
-- Connected: google/gemma-3-4b-it (34 layers, 348160 features)

DESCRIBE "France";
WALK "Einstein" TOP 10;
INFER "The capital of France is" TOP 5;
STATS;
SHOW RELATIONS;
```

## Multi-Model Serving

When using `--dir`, each vindex gets its own namespace:

```bash
larql serve --dir ./vindexes/ --port 8080
# /v1/gemma-3-4b-it/describe, /v1/llama-3-8b/describe, ...
```

```
GET /v1/gemma-3-4b-it/describe?entity=France
GET /v1/llama-3-8b/describe?entity=France
```

## Crate Structure

```
larql-server/
├── Cargo.toml
├── README.md
├── examples/
│   ├── server_demo.rs          Synthetic vindex API demo
│   └── server_bench.rs         Endpoint latency benchmarks
├── tests/
│   └── test_api.rs             Integration tests
└── src/
    ├── main.rs                 CLI parsing, vindex loading, server startup
    ├── state.rs                AppState: loaded models, probe labels, lazy weights
    ├── error.rs                ServerError → HTTP status codes
    ├── auth.rs                 API key Bearer token middleware
    └── routes/
        ├── mod.rs              Router setup (single + multi-model)
        ├── describe.rs         GET /v1/describe (with relation labels)
        ├── walk.rs             GET /v1/walk
        ├── select.rs           POST /v1/select
        ├── relations.rs        GET /v1/relations
        ├── stats.rs            GET /v1/stats
        ├── infer.rs            POST /v1/infer (walk/dense/compare)
        ├── patches.rs          POST/GET/DELETE /v1/patches
        ├── health.rs           GET /v1/health
        └── models.rs           GET /v1/models
```

## Dependencies

- `larql-vindex` — vector index loading, gate KNN, walk, patches
- `larql-inference` — forward pass, WalkFfn, dense FFN
- `axum` — HTTP framework
- `axum-server` — TLS support (rustls)
- `tokio` — async runtime
- `tower` — concurrency limit middleware
- `tower-http` — CORS, tracing middleware
- `clap` — CLI argument parsing

## Testing

```bash
# Unit/integration tests
cargo test -p larql-server

# Demo (synthetic data, no real vindex needed)
cargo run -p larql-server --example server_demo

# Benchmarks (synthetic data)
cargo run -p larql-server --example server_bench --release
```

## Deployment

### Docker

```dockerfile
FROM rust:1.82-slim AS builder
WORKDIR /build
COPY . .
RUN cargo build --release -p larql-server

FROM debian:bookworm-slim
COPY --from=builder /build/target/release/larql-server /usr/local/bin/
EXPOSE 8080
ENTRYPOINT ["larql-server"]
```

```bash
docker run -v ./vindexes:/data -p 8080:8080 larql-server /data/gemma3-4b.vindex
```

### Systemd

```ini
[Unit]
Description=LARQL Vindex Server
After=network.target

[Service]
ExecStart=/usr/local/bin/larql-server /data/gemma3-4b.vindex --port 8080
Restart=always
MemoryMax=8G

[Install]
WantedBy=multi-user.target
```

Browse-only (f16): ~3 GB RAM. No GPU needed.

## License

Apache-2.0
