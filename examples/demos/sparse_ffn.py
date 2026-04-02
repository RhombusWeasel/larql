"""
Demo: Sparse FFN on Metal — top-K gate selection + sparse matmul, all on GPU.

No CPU roundtrip. Gate scores computed on Metal, argpartition for top-K,
gather selected rows, sparse matmul. 40% the compute of dense.

Usage:
    python examples/demos/sparse_ffn.py [path/to/model.vindex]
    SPARSE_TOP_K=2048 python examples/demos/sparse_ffn.py

Requires: mlx, mlx-lm, larql (built with maturin develop --release)
"""

import sys
import os
import time

VINDEX_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "..", "output", "gemma3-4b-v2.vindex"
)
TOP_K = int(os.environ.get("SPARSE_TOP_K", "4096"))


def main():
    try:
        import mlx.core as mx
        import mlx_lm
    except ImportError:
        print("Requires: pip install mlx mlx-lm")
        sys.exit(1)

    import larql

    print("=" * 60)
    print("  Sparse FFN on Metal")
    print("  Gate top-K + sparse matmul — all on GPU")
    print("=" * 60)

    # ── Load sparse model ──
    print(f"\nLoading sparse model (top_k={TOP_K})...")
    t0 = time.time()
    from larql.sparse_ffn import load
    model, tokenizer = load(VINDEX_PATH, top_k=TOP_K)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Generate ──
    prompts = [
        "The capital of France is",
        "Albert Einstein was a famous",
        "The programming language Python was created by",
        "The largest planet in our solar system is",
    ]

    print(f"\n{'─' * 60}")
    print("  Generation (sparse FFN)")
    print(f"{'─' * 60}")

    for prompt in prompts:
        t0 = time.time()
        response = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=30, verbose=False
        )
        elapsed = time.time() - t0
        print(f"\n  {prompt}")
        print(f"  → {response}")
        print(f"    {elapsed:.2f}s")

    # ── Dense vs Sparse ──
    print(f"\n{'─' * 60}")
    print("  Dense vs Sparse")
    print(f"{'─' * 60}")

    print("\n  Loading dense model...")
    t0 = time.time()
    dense_model, dense_tok = larql.mlx.load(VINDEX_PATH)
    print(f"  Dense loaded in {time.time() - t0:.1f}s")

    test_prompt = "The capital of France is"

    t0 = time.time()
    dense_resp = mlx_lm.generate(dense_model, dense_tok, prompt=test_prompt, max_tokens=20, verbose=False)
    dense_time = time.time() - t0

    t0 = time.time()
    sparse_resp = mlx_lm.generate(model, tokenizer, prompt=test_prompt, max_tokens=20, verbose=False)
    sparse_time = time.time() - t0

    print(f"\n  \"{test_prompt}\"")
    print(f"  Dense:  {dense_resp}  ({dense_time:.2f}s)")
    print(f"  Sparse: {sparse_resp}  ({sparse_time:.2f}s)")
    if sparse_time > 0:
        print(f"  Ratio:  {sparse_time / dense_time:.2f}x")

    # ── Top-K sweep ──
    print(f"\n{'─' * 60}")
    print("  Top-K sweep")
    print(f"{'─' * 60}")

    for k in [1024, 2048, 4096, 8192]:
        m, tok = load(VINDEX_PATH, top_k=k)
        t0 = time.time()
        resp = mlx_lm.generate(m, tok, prompt=test_prompt, max_tokens=20, verbose=False)
        elapsed = time.time() - t0
        print(f"  K={k:>5}: {resp}  ({elapsed:.2f}s)")

    print()


if __name__ == "__main__":
    main()
