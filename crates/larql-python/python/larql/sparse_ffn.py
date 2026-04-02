"""
MLX sparse FFN on Metal.

All on GPU — gate selection + sparse matmul. No CPU roundtrip.

Two modes:
  load()           — all-Metal, fast. Gate top-K + sparse matmul on GPU.
  load_guided()    — Rust vindex selects features (editable), MLX does matmul.
                     Slower (GPU→CPU sync per layer) but uses mutated vindex.

Usage:
    from larql.sparse_ffn import load
    import mlx_lm

    model, tokenizer = load("model.vindex", top_k=4096)
    response = mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=20)
"""

import json
from pathlib import Path
from typing import Tuple


def load(
    vindex_path: str,
    top_k: int = 4096,
) -> Tuple:
    """Load MLX model with sparse FFN — all on Metal GPU.

    Gate top-K selection and sparse matmul both run on Metal.
    No CPU roundtrip. Stays in MLX's lazy eval graph.

    Args:
        vindex_path: path to .vindex directory (requires --level all)
        top_k: features for sparse FFN (4096 = 40% of 10240)

    Returns:
        (model, tokenizer) — ready for mlx_lm.generate()
    """
    import mlx.core as mx
    import mlx_lm.utils as mlx_utils
    from larql.mlx import _build_config, _load_weights

    vpath = Path(vindex_path)

    with open(vpath / "index.json") as f:
        config = json.load(f)

    mlx_config = _build_config(str(vindex_path))
    model_class, model_args_class = mlx_utils._get_classes(config=mlx_config)
    model = model_class(model_args_class.from_dict(mlx_config))

    weights = _load_weights(str(vindex_path))

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    model.eval()
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    _patch_sparse_mlp(model, config, top_k)

    tokenizer = mlx_utils.load_tokenizer(vpath)

    intermediate = config.get("intermediate_size", 0)
    pct = (top_k / intermediate * 100) if intermediate > 0 else 0
    print(f"Sparse FFN: top_k={top_k}/{intermediate} ({pct:.0f}% of dense), all on Metal")

    return model, tokenizer


def _patch_sparse_mlp(model, config, top_k):
    """Replace each layer's MLP with sparse FFN — all on Metal."""
    import mlx.core as mx
    import mlx.nn as nn

    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    elif hasattr(model, "model"):
        layers = model.model.layers
    else:
        layers = model.layers

    family = config.get("family", "")
    model_type = config.get("model_config", {}).get("model_type", family)
    is_gated = any(t in model_type for t in ("gemma", "llama", "mistral", "qwen"))

    class SparseMLP(nn.Module):
        """All-Metal sparse FFN. Gate top-K + sparse matmul on GPU.

        No CPU roundtrip. Stays in MLX lazy eval graph.

        1. x @ gate.T → scores (on Metal)
        2. argpartition top-K (on Metal)
        3. mx.take to gather K rows from up/down (on Metal)
        4. Sparse matmul (on Metal)
        """
        def __init__(self, layer_idx, original_mlp):
            super().__init__()
            self._layer_idx = layer_idx
            self._top_k = top_k
            self._mlp = original_mlp

        def __call__(self, x):
            shape = x.shape
            hidden = shape[-1]
            if len(shape) == 3:
                x_2d = x.reshape(-1, hidden)
            else:
                x_2d = x

            if is_gated:
                out = self._gated_sparse(x_2d)
            else:
                out = self._ungated_sparse(x_2d)

            if len(shape) == 3:
                out = out.reshape(shape)
            return out

        def _gated_sparse(self, x):
            # Gate scores on Metal: (seq, intermediate)
            gate_scores = x @ self._mlp.gate_proj.weight.T

            # Top-K per position on Metal
            # Use last position's scores for feature selection (autoregressive)
            scores_last = mx.abs(gate_scores[-1:])  # (1, intermediate)
            # argpartition: O(n) selection, not O(n log n) sort
            neg_scores = -scores_last.squeeze(0)
            idx = mx.argpartition(neg_scores, kth=self._top_k - 1)[:self._top_k]

            # Gather selected rows — all on Metal
            gate_rows = mx.take(self._mlp.gate_proj.weight, idx, axis=0)  # (K, hidden)
            up_rows = mx.take(self._mlp.up_proj.weight, idx, axis=0)      # (K, hidden)
            down_cols = mx.take(self._mlp.down_proj.weight, idx, axis=1)  # (hidden, K)

            # Sparse matmul on Metal
            gated = nn.silu(x @ gate_rows.T) * (x @ up_rows.T)  # (seq, K)
            return gated @ down_cols.T                            # (seq, hidden)

        def _ungated_sparse(self, x):
            up_scores = x @ self._mlp.up_proj.weight.T
            scores_last = mx.abs(up_scores[-1:])
            neg_scores = -scores_last.squeeze(0)
            idx = mx.argpartition(neg_scores, kth=self._top_k - 1)[:self._top_k]

            up_rows = mx.take(self._mlp.up_proj.weight, idx, axis=0)
            down_cols = mx.take(self._mlp.down_proj.weight, idx, axis=1)

            activated = x @ up_rows.T
            if hasattr(self._mlp.up_proj, "bias") and self._mlp.up_proj.bias is not None:
                bias_sel = mx.take(self._mlp.up_proj.bias, idx, axis=0)
                activated = activated + bias_sel
            activated = mx.gelu(activated)
            out = activated @ down_cols.T
            if hasattr(self._mlp.down_proj, "bias") and self._mlp.down_proj.bias is not None:
                out = out + self._mlp.down_proj.bias
            return out

    for i in range(len(layers)):
        layers[i].mlp = SparseMLP(i, layers[i].mlp)


# ── Vindex-guided mode (editable knowledge layer) ──

def load_guided(
    vindex_path: str,
    top_k: int = 4096,
    gate_top_k: int = 8192,
) -> Tuple:
    """Load MLX model with vindex-guided sparse FFN.

    Rust vindex selects features (gate KNN on mmap'd index).
    MLX does the sparse matmul on Metal.

    Slower than load() due to GPU→CPU sync per layer, but uses the
    vindex for feature selection — so INSERT/DELETE mutations are
    reflected in inference.

    Args:
        vindex_path: path to .vindex directory
        top_k: features for sparse matmul
        gate_top_k: features for Rust gate KNN

    Returns:
        (model, tokenizer) — ready for mlx_lm.generate()
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm.utils as mlx_utils
    import larql
    from larql.mlx import _build_config, _load_weights

    vpath = Path(vindex_path)

    with open(vpath / "index.json") as f:
        config = json.load(f)

    walk_model = larql.WalkModel(str(vindex_path), top_k=gate_top_k)

    mlx_config = _build_config(str(vindex_path))
    model_class, model_args_class = mlx_utils._get_classes(config=mlx_config)
    model = model_class(model_args_class.from_dict(mlx_config))

    weights = _load_weights(str(vindex_path))

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    model.eval()
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    # Detect gated
    family = config.get("family", "")
    model_type = config.get("model_config", {}).get("model_type", family)
    is_gated = any(t in model_type for t in ("gemma", "llama", "mistral", "qwen"))

    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    elif hasattr(model, "model"):
        layers = model.model.layers
    else:
        layers = model.layers

    class GuidedSparseMLP(nn.Module):
        """Vindex-guided sparse FFN. Rust selects, MLX computes."""
        def __init__(self, layer_idx, original_mlp):
            super().__init__()
            self._layer_idx = layer_idx
            self._mlp = original_mlp

        def __call__(self, x):
            shape = x.shape
            hidden = shape[-1]
            if len(shape) == 3:
                x_2d = x.reshape(-1, hidden)
            else:
                x_2d = x
            seq_len = x_2d.shape[0]

            # GPU → CPU sync (unavoidable for Rust gate KNN)
            x_f32 = x_2d.astype(mx.float32)
            mx.eval(x_f32)
            x_bytes = bytes(x_f32)
            indices = walk_model.gate_select(
                layer=self._layer_idx, x_bytes=x_bytes,
                seq_len=seq_len, top_k=top_k,
            )

            if len(indices) == 0:
                return mx.zeros_like(x)

            idx = mx.array(indices)

            if is_gated:
                gate_rows = mx.take(self._mlp.gate_proj.weight, idx, axis=0)
                up_rows = mx.take(self._mlp.up_proj.weight, idx, axis=0)
                down_cols = mx.take(self._mlp.down_proj.weight, idx, axis=1)
                gated = nn.silu(x_2d @ gate_rows.T) * (x_2d @ up_rows.T)
                out = gated @ down_cols.T
            else:
                up_rows = mx.take(self._mlp.up_proj.weight, idx, axis=0)
                down_cols = mx.take(self._mlp.down_proj.weight, idx, axis=1)
                activated = x_2d @ up_rows.T
                if hasattr(self._mlp.up_proj, "bias") and self._mlp.up_proj.bias is not None:
                    activated = activated + mx.take(self._mlp.up_proj.bias, idx, axis=0)
                activated = mx.gelu(activated)
                out = activated @ down_cols.T
                if hasattr(self._mlp.down_proj, "bias") and self._mlp.down_proj.bias is not None:
                    out = out + self._mlp.down_proj.bias

            if len(shape) == 3:
                out = out.reshape(shape)
            return out

    for i in range(len(layers)):
        layers[i].mlp = GuidedSparseMLP(i, layers[i].mlp)

    tokenizer = mlx_lm.utils.load_tokenizer(vpath)

    intermediate = config.get("intermediate_size", 0)
    pct = (top_k / intermediate * 100) if intermediate > 0 else 0
    print(f"Guided sparse FFN: top_k={top_k}/{intermediate} ({pct:.0f}% of dense), "
          f"Rust gate KNN → Metal matmul")

    return model, tokenizer
