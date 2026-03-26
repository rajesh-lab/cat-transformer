"""
Minimal PyTorch reference implementations of linear attention:
  1. naive_recurrent  — token-by-token loop (ground truth)
  2. chunk_linear_attn — two-pass chunked (parallel across chunks)
  3. fused_chunk_linear_attn — single-pass chunked (sequential across chunks)

All produce identical outputs (up to floating point).

CAT-masked variants (suffix _cat) restrict cross-chunk attention to only
the first token of each chunk, matching the mask from get_cat_mask:
  (within_block | divides_block) & causal

Shapes:
  q, k: [B, T, H, K]
  v:    [B, T, H, V]
  o:    [B, T, H, V]
  state (S / h): [B, H, K, V]
"""

import torch
from torch import Tensor, einsum
from einops import rearrange


# --------------------------------------------------------------------------- #
# 1. Naive recurrent (ground truth)
# --------------------------------------------------------------------------- #

def naive_recurrent(
    q: Tensor, k: Tensor, v: Tensor,
    scale: float | None = None,
) -> Tensor:
    """Token-by-token recurrence.  O(T·K·V) per head, but fully sequential."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K = q.shape
    V = v.shape[-1]

    S = q.new_zeros(B, H, K, V)
    o = torch.empty_like(v)

    for t in range(T):
        # accumulate outer product into state
        S = S + einsum('b h k, b h v -> b h k v', k[:, t], v[:, t])
        # query the state
        o[:, t] = einsum('b h k, b h k v -> b h v', q[:, t] * scale, S)

    return o


# --------------------------------------------------------------------------- #
# 2. Two-pass chunked  (matches fla's `chunk_linear_attn`)
# --------------------------------------------------------------------------- #

def chunk_linear_attn(
    q: Tensor, k: Tensor, v: Tensor,
    scale: float | None = None,
    chunk_size: int = 64,
) -> Tensor:
    """
    Two-pass algorithm:
      Pass 1 — scan over chunk boundaries to build per-chunk states h[i].
      Pass 2 — (parallelizable) combine inter-chunk + intra-chunk per chunk.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NC = T // C  # assume T divisible by C for simplicity

    # reshape into chunks: [B, NC, C, H, K/V]
    q_c = rearrange(q, 'b (nc c) h k -> b nc c h k', c=C) * scale
    k_c = rearrange(k, 'b (nc c) h k -> b nc c h k', c=C)
    v_c = rearrange(v, 'b (nc c) h v -> b nc c h v', c=C)

    # ------ Pass 1: compute inter-chunk states h[i] = Σ_{j<i} K_j^T @ V_j ------
    # kv per chunk: [B, NC, H, K, V]
    kv = einsum('b n c h k, b n c h v -> b n h k v', k_c, v_c)
    # cumulative sum, shifted right by 1 (state *before* each chunk)
    h = kv.cumsum(dim=1)
    h = torch.cat([q.new_zeros(B, 1, H, K, V), h[:, :-1]], dim=1)

    # ------ Pass 2: per-chunk output (all chunks are independent) ------
    # inter-chunk: each query token attends to accumulated past state
    inter = einsum('b n c h k, b n h k v -> b n c h v', q_c, h)

    # intra-chunk: causal Q @ K^T within the chunk, then @ V
    attn = einsum('b n i h k, b n j h k -> b n h i j', q_c, k_c)  # [B, NC, H, C, C]
    causal_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))
    attn = attn.masked_fill(~causal_mask, 0.0)
    intra = einsum('b n h i j, b n j h v -> b n i h v', attn, v_c)

    o = inter + intra
    return rearrange(o, 'b nc c h v -> b (nc c) h v')


# --------------------------------------------------------------------------- #
# 3. Single-pass fused chunked  (matches fla's `fused_chunk_linear_attn`)
# --------------------------------------------------------------------------- #

def fused_chunk_linear_attn(
    q: Tensor, k: Tensor, v: Tensor,
    scale: float | None = None,
    chunk_size: int = 64,
) -> Tensor:
    """
    Single-pass: walk through chunks sequentially, maintaining a running
    KV state in registers (here, a regular tensor). Each iteration computes
    the output for one chunk and updates the state.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NC = T // C

    q_c = rearrange(q, 'b (nc c) h k -> b nc c h k', c=C) * scale
    k_c = rearrange(k, 'b (nc c) h k -> b nc c h k', c=C)
    v_c = rearrange(v, 'b (nc c) h v -> b nc c h v', c=C)

    causal_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))

    S = q.new_zeros(B, H, K, V)  # running state — lives "in registers" in Triton
    o_chunks = []

    for i in range(NC):
        qi, ki, vi = q_c[:, i], k_c[:, i], v_c[:, i]  # [B, C, H, K/V]

        # inter-chunk: query the accumulated state from all prior chunks
        inter = einsum('b c h k, b h k v -> b c h v', qi, S)

        # intra-chunk: causal attention within this chunk
        attn = einsum('b i h k, b j h k -> b h i j', qi, ki)   # [B, H, C, C]
        attn = attn.masked_fill(~causal_mask, 0.0)
        intra = einsum('b h i j, b j h v -> b i h v', attn, vi)

        o_chunks.append(inter + intra)

        # update state: add this chunk's contribution
        S = S + einsum('b c h k, b c h v -> b h k v', ki, vi)

    return torch.cat(o_chunks, dim=1)


# =========================================================================== #
# CAT-masked variants
#
# The CAT mask (get_cat_mask) allows each token to attend to:
#   - all tokens in the SAME chunk (causally), AND
#   - the FIRST token of every PRECEDING chunk.
#
# For linear attention this means the inter-chunk KV state only accumulates
# from position-0 tokens, while intra-chunk attention is standard causal.
# =========================================================================== #

def reference_cat_attn(
    q: Tensor, k: Tensor, v: Tensor,
    scale: float | None = None,
    chunk_size: int = 8,
) -> Tensor:
    """Quadratic attention with the explicit CAT mask — used as ground truth."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K = q.shape

    attn = einsum('b i h k, b j h k -> b h i j', q * scale, k)

    idx = torch.arange(T, device=q.device)
    q_idx, kv_idx = idx.unsqueeze(1), idx.unsqueeze(0)
    within_block = (q_idx // chunk_size) == (kv_idx // chunk_size)
    divides_block = (kv_idx % chunk_size) == 0
    causal = q_idx >= kv_idx
    mask = (within_block | divides_block) & causal

    attn = attn.masked_fill(~mask, 0.0)
    return einsum('b h i j, b j h v -> b i h v', attn, v)


# --------------------------------------------------------------------------- #
# 4. Naive recurrent — CAT-masked
# --------------------------------------------------------------------------- #

def naive_recurrent_cat(
    q: Tensor, k: Tensor, v: Tensor,
    scale: float | None = None,
    chunk_size: int = 8,
) -> Tensor:
    """Token-by-token recurrence respecting the CAT mask.

    Maintains two states:
      S_inter — accumulated KV from first tokens of completed chunks.
      S_intra — accumulated KV within the current chunk (reset per chunk).
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NC = T // C

    S_inter = q.new_zeros(B, H, K, V)
    o = torch.empty_like(v)

    for c in range(NC):
        S_intra = q.new_zeros(B, H, K, V)
        for j in range(C):
            t = c * C + j
            S_intra = S_intra + einsum('b h k, b h v -> b h k v', k[:, t], v[:, t])
            o[:, t] = einsum('b h k, b h k v -> b h v', q[:, t] * scale, S_inter + S_intra)

        first = c * C
        S_inter = S_inter + einsum('b h k, b h v -> b h k v', k[:, first], v[:, first])

    return o


# --------------------------------------------------------------------------- #
# 5. Two-pass chunked — CAT-masked
# --------------------------------------------------------------------------- #

def chunk_linear_attn_cat(
    q: Tensor, k: Tensor, v: Tensor,
    scale: float | None = None,
    chunk_size: int = 8,
) -> Tensor:
    """
    Two-pass chunked linear attention under the CAT mask.

    Pass 1 builds inter-chunk states from the first token of each chunk only.
    Pass 2 combines inter-chunk + intra-chunk (all chunks parallelizable).
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NC = T // C

    q_c = rearrange(q, 'b (nc c) h k -> b nc c h k', c=C) * scale
    k_c = rearrange(k, 'b (nc c) h k -> b nc c h k', c=C)
    v_c = rearrange(v, 'b (nc c) h v -> b nc c h v', c=C)

    # ------ Pass 1: states from first token of each chunk only ------
    k_first = k_c[:, :, 0, :, :]                                        # (B, NC, H, K)
    v_first = v_c[:, :, 0, :, :]                                        # (B, NC, H, V)
    kv_first = einsum('b n h k, b n h v -> b n h k v', k_first, v_first)
    h = kv_first.cumsum(dim=1)
    h = torch.cat([q.new_zeros(B, 1, H, K, V), h[:, :-1]], dim=1)       # shifted right

    # ------ Pass 2: per-chunk (parallelizable) ------
    inter = einsum('b n c h k, b n h k v -> b n c h v', q_c, h)

    attn = einsum('b n i h k, b n j h k -> b n h i j', q_c, k_c)
    causal_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))
    attn = attn.masked_fill(~causal_mask, 0.0)
    intra = einsum('b n h i j, b n j h v -> b n i h v', attn, v_c)

    o = inter + intra
    return rearrange(o, 'b nc c h v -> b (nc c) h v')


# --------------------------------------------------------------------------- #
# Quick test
# --------------------------------------------------------------------------- #

def benchmark_fn(fn, *args, warmup=5, rep=20, **kwargs):
    """Time a function (fwd + bwd) in milliseconds."""
    # warmup
    for _ in range(warmup):
        o = fn(*args, **kwargs)
        o.sum().backward()

    torch.cuda.synchronize()
    timings = []
    for _ in range(rep):
        # fresh grads
        for a in args:
            if a.requires_grad:
                a.grad = None

        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)

        t0.record()
        o = fn(*args, **kwargs)
        o.sum().backward()
        t1.record()
        torch.cuda.synchronize()
        timings.append(t0.elapsed_time(t1))

    return timings


if __name__ == '__main__':
    torch.manual_seed(42)
    B, T, H, K, V = 2, 1024, 4, 64, 64
    chunk_size = 8
    device = 'cuda'

    q = torch.randn(B, T, H, K, device=device)
    k = torch.randn(B, T, H, K, device=device)
    v = torch.randn(B, T, H, V, device=device)

    # ---- correctness ----
    print("=" * 70)
    print(f"Correctness  (B={B}, T={T}, H={H}, K={K}, chunk_size={chunk_size})")
    print("=" * 70)

    o_naive = naive_recurrent(q, k, v)
    o_chunk = chunk_linear_attn(q, k, v, chunk_size=chunk_size)
    o_fused = fused_chunk_linear_attn(q, k, v, chunk_size=chunk_size)
    print("Standard linear attention:")
    print(f"  chunk  vs naive: max err = {(o_chunk - o_naive).abs().max().item():.2e}")
    print(f"  fused  vs naive: max err = {(o_fused - o_naive).abs().max().item():.2e}")

    o_ref       = reference_cat_attn(q, k, v, chunk_size=chunk_size)
    o_naive_cat = naive_recurrent_cat(q, k, v, chunk_size=chunk_size)
    o_chunk_cat = chunk_linear_attn_cat(q, k, v, chunk_size=chunk_size)
    print("CAT-masked linear attention:")
    print(f"  naive  vs ref:   max err = {(o_naive_cat - o_ref).abs().max().item():.2e}")
    print(f"  chunk  vs ref:   max err = {(o_chunk_cat - o_ref).abs().max().item():.2e}")

    # ---- throughput ----
    print()
    print("=" * 70)
    print(f"Throughput (fwd + bwd)  —  B={B}, T={T}, H={H}, K={K}, chunk={chunk_size}")
    print("=" * 70)

    q = torch.randn(B, T, H, K, device=device, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, requires_grad=True)

    benches = [
        ("naive_recurrent",       naive_recurrent,       {}),
        ("chunk_linear_attn",     chunk_linear_attn,     {"chunk_size": chunk_size}),
        ("fused_chunk_linear",    fused_chunk_linear_attn, {"chunk_size": chunk_size}),
        ("reference_cat (quad)",  reference_cat_attn,    {"chunk_size": chunk_size}),
        ("naive_recurrent_cat",   naive_recurrent_cat,   {"chunk_size": chunk_size}),
        ("chunk_linear_attn_cat", chunk_linear_attn_cat, {"chunk_size": chunk_size}),
    ]

    results = {}
    for name, fn, kw in benches:
        timings = benchmark_fn(fn, q, k, v, **kw)
        avg = sum(timings) / len(timings)
        mn, mx = min(timings), max(timings)
        results[name] = avg
        print(f"  {name:<25s}  {avg:8.2f} ms  (min={mn:.2f}, max={mx:.2f})")

    print()
    print("-" * 70)
    baseline = results["chunk_linear_attn"]
    for name, avg in results.items():
        ratio = avg / baseline
        print(f"  {name:<25s}  {ratio:5.2f}x  vs chunk_linear_attn")
    print("=" * 70)