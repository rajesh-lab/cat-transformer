"""
Minimal PyTorch reference implementations of Mamba2 / SSD (Structured State
Space Duality), including CAT-masked variants.

Standard Mamba2/SSD:
  1. naive_mamba2          — token-by-token recurrence (ground truth)
  2. mamba2_parallel       — SSD chunked parallel algorithm

CAT-masked Mamba2/SSD:
  3. naive_mamba2_cat      — token-by-token with CAT mask (ground truth)
  4. mamba2_parallel_cat   — SSD chunked parallel with CAT mask

Input convention (all pre-discretized):
  x:     [B, T, H, D]    input (x * dt)
  a:     [B, T, H]       log decay (A_continuous * dt, negative)
  b:     [B, T, H, N]    input-to-state projection
  c:     [B, T, H, N]    state-to-output projection
  state: [B, H, N, D]
  o:     [B, T, H, D]
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def segment_sum(log_decays: Tensor) -> Tensor:
    """Build a lower-triangular matrix of pairwise cumulative sums.

    Given log_decays of shape (..., L), returns a matrix M of shape (..., L, L)
    where M[i, j] = sum(log_decays[j+1 .. i]) for i >= j, and -inf otherwise.

    Used to construct the inter-chunk decay matrix:
        decay_matrix = exp(segment_sum(log_decays))
    so that decay_matrix[i, j] is the multiplicative decay from chunk j to chunk i.
    """
    L = log_decays.size(-1)
    # (..., L) -> (..., L, L), broadcast along last dim
    log_decays = log_decays.unsqueeze(-1).expand(*log_decays.shape, L)
    # Zero out upper triangle (including diagonal): only sum strictly lower entries
    mask_lower = torch.tril(torch.ones(L, L, device=log_decays.device, dtype=torch.bool), diagonal=-1)
    log_decays = log_decays.masked_fill(~mask_lower, 0.0)
    # Cumulative sum along rows gives pairwise sums
    cum = torch.cumsum(log_decays, dim=-2)
    # Keep only lower triangle (including diagonal); set upper to -inf so exp -> 0
    mask_causal = torch.tril(torch.ones(L, L, device=log_decays.device, dtype=torch.bool))
    return cum.masked_fill(~mask_causal, -torch.inf)


# --------------------------------------------------------------------------- #
# 1. Naive recurrent Mamba2 (ground truth)
# --------------------------------------------------------------------------- #

def naive_mamba2(
    x: Tensor, a: Tensor, b: Tensor, c: Tensor,
) -> Tensor:
    """Token-by-token SSM recurrence.  O(T·N·D) per head, fully sequential.

    s_t = exp(a_t) · s_{t-1}  +  b_t ⊗ x_t
    y_t = c_t · s_t            (contract over state dim N)
    """
    B, T, H, D = x.shape
    N = b.shape[-1]

    s = x.new_zeros(B, H, N, D)
    y = torch.empty_like(x)

    for t in range(T):
        alpha = a[:, t, :].exp()                                     # (B, H)
        s = (alpha[:, :, None, None] * s
             + b[:, t, :, :, None] * x[:, t, :, None, :])           # (B, H, N, D)
        y[:, t] = (c[:, t, :, :, None] * s).sum(dim=-2)             # (B, H, D)

    return y


# --------------------------------------------------------------------------- #
# 2. SSD parallel (chunked Mamba2)
# --------------------------------------------------------------------------- #

def mamba2_parallel(
    x: Tensor, a: Tensor, b: Tensor, c: Tensor,
    chunk_size: int = 64,
) -> Tensor:
    """Structured State Space Duality — chunked parallel algorithm.

    Splits the sequence into chunks and computes:
      Intra-chunk : attention-like causal computation  (all chunks parallel)
      Inter-chunk : state propagation across chunks    (parallel via decay matrix)
      State readout: inter-chunk state → output         (all chunks parallel)

    Following the fla Mamba2 reference, the entire SSD computation runs in
    float32 to avoid bf16 precision loss in cumsum/exp chains.
    """
    B, T, H, D = x.shape
    N = b.shape[-1]
    CS = chunk_size
    NC = T // CS
    orig_dtype = x.dtype

    x = x.float()
    a = a.float()
    b = b.float()
    c = c.float()

    # Reshape into chunks  (B, NC, CS, H, ...)  →  (B, NC, H, CS, ...)
    x_h = rearrange(x, 'b (nc cs) h d -> b nc h cs d', cs=CS)
    a_h = rearrange(a, 'b (nc cs) h   -> b nc h cs',   cs=CS)
    b_h = rearrange(b, 'b (nc cs) h n -> b nc h cs n', cs=CS)
    c_h = rearrange(c, 'b (nc cs) h n -> b nc h cs n', cs=CS)

    # --- Intra-chunk (diagonal blocks) ---
    A_cumsum = a_h.cumsum(dim=-1)                                    # (B, NC, H, CS)

    # Causal decay matrix  L[i,j] = exp(G_i − G_j)  for i ≥ j
    L = (A_cumsum.unsqueeze(-1) - A_cumsum.unsqueeze(-2)).exp()      # (B, NC, H, CS, CS)
    causal = torch.tril(torch.ones(CS, CS, device=x.device, dtype=torch.bool))
    L = L.masked_fill(~causal, 0.0)

    # Attention weights  G[i,j] = c_i^T b_j
    G = torch.matmul(c_h, b_h.transpose(-1, -2))                    # (B, NC, H, CS, CS)

    M = L * G
    Y_diag = torch.matmul(M, x_h)                                   # (B, NC, H, CS, D)

    # --- Per-chunk state contribution ---
    # Decay from each position to the end of its chunk
    decay_to_end = (A_cumsum[..., -1:] - A_cumsum).exp()             # (B, NC, H, CS)
    b_decay = b_h * decay_to_end.unsqueeze(-1)                       # (B, NC, H, CS, N)
    chunk_states = torch.matmul(b_decay.transpose(-1, -2), x_h)     # (B, NC, H, N, D)

    # --- Parallel inter-chunk state propagation (decay-matrix matmul) ---
    log_chunk_decay = A_cumsum[..., -1]

    padded = F.pad(log_chunk_decay, (0, 0, 1, 0))                   # (B, NC+1, H)
    padded_t = padded.permute(0, 2, 1)                               # (B, H, NC+1)
    decay_matrix = segment_sum(padded_t).exp()                       # (B, H, NC+1, NC+1)

    chunk_states_pad = torch.cat(
        [x.new_zeros(B, 1, H, N, D), chunk_states], dim=1,
    )
    cs_t = chunk_states_pad.permute(0, 2, 1, 3, 4)                  # (B, H, NC+1, N, D)
    all_states = torch.einsum('bhij,bhjnd->bhind', decay_matrix, cs_t)
    inter_states = all_states[:, :, :NC].permute(0, 2, 1, 3, 4)

    # --- State-to-output (off-diagonal blocks) ---
    decay_from_start = A_cumsum.exp()                                # (B, NC, H, CS)
    Y_off = torch.matmul(c_h, inter_states)                         # (B, NC, H, CS, D)
    Y_off = Y_off * decay_from_start.unsqueeze(-1)

    y = Y_diag + Y_off
    return rearrange(y, 'b nc h cs d -> b (nc cs) h d').to(orig_dtype)


# --------------------------------------------------------------------------- #
# 3. Naive recurrent Mamba2 — CAT-masked (ground truth)
# --------------------------------------------------------------------------- #

def naive_mamba2_cat(
    x: Tensor, a: Tensor, b: Tensor, c: Tensor,
    chunk_size: int = 8,
) -> Tensor:
    """Token-by-token SSM recurrence respecting the CAT mask.

    S_inter is accumulated via SSM update from first tokens only.
    Within each chunk the full SSM recurrence runs from S_inter.
    After each chunk, S_inter is updated using the first token applied
    to the *old* S_inter (before the chunk was processed).
    """
    B, T, H, D = x.shape
    N = b.shape[-1]
    CS = chunk_size
    NC = T // CS

    S_inter = x.new_zeros(B, H, N, D)
    y = torch.empty_like(x)

    for i in range(NC):
        S_inter_old = S_inter.clone()
        s = S_inter.clone()

        for j in range(CS):
            t = i * CS + j
            alpha = a[:, t, :].exp()
            s = (alpha[:, :, None, None] * s
                 + b[:, t, :, :, None] * x[:, t, :, None, :])
            y[:, t] = (c[:, t, :, :, None] * s).sum(dim=-2)

        # Update S_inter: first token of this chunk applied to OLD S_inter
        t0 = i * CS
        a0 = a[:, t0, :].exp()
        S_inter = (a0[:, :, None, None] * S_inter_old
                   + b[:, t0, :, :, None] * x[:, t0, :, None, :])

    return y


# --------------------------------------------------------------------------- #
# 4. SSD parallel — CAT-masked
# --------------------------------------------------------------------------- #

def mamba2_parallel_cat(
    x: Tensor, a: Tensor, b: Tensor, c: Tensor,
    chunk_size: int = 8,
) -> Tensor:
    """Chunked parallel SSD under the CAT mask.

    Pass 1: Compute inter-chunk states from first tokens only (parallel via
            decay-matrix matmul — same segment_sum trick as standard SSD).
    Pass 2: Standard SSD intra-chunk + readout from CAT inter-states (parallel).

    The entire SSD computation runs in float32 to avoid bf16 precision loss
    in cumsum/exp chains. Uses native view/permute (no einops) so that
    torch.compile can trace a clean graph without graph breaks.
    """
    B, T, H, D = x.shape
    N = b.shape[-1]
    CS = chunk_size
    NC = T // CS
    orig_dtype = x.dtype

    x = x.float()
    a = a.float()
    b = b.float()
    c = c.float()

    # Reshape into chunks — native view instead of einops
    x_c = x.view(B, NC, CS, H, D)
    a_c = a.view(B, NC, CS, H)
    b_c = b.view(B, NC, CS, H, N)
    c_c = c.view(B, NC, CS, H, N)

    # --- Pass 1: inter-chunk states from first tokens only (parallel) ---
    log_a0 = a_c[:, :, 0]                                           # (B, NC, H)
    b0 = b_c[:, :, 0]                                               # (B, NC, H, N)
    x0 = x_c[:, :, 0]                                               # (B, NC, H, D)
    bx0 = b0.unsqueeze(-1) * x0.unsqueeze(-2)                       # (B, NC, H, N, D)

    # Build decay matrix from first-token log-decays
    padded = F.pad(log_a0, (0, 0, 1, 0))                            # (B, NC+1, H)
    padded_t = padded.permute(0, 2, 1)                               # (B, H, NC+1)
    decay_matrix = segment_sum(padded_t).exp()                       # (B, H, NC+1, NC+1)

    # Prepend zero state, then parallel matmul
    bx0_pad = torch.cat([x.new_zeros(B, 1, H, N, D), bx0], dim=1)  # (B, NC+1, H, N, D)
    bx0_t = bx0_pad.permute(0, 2, 1, 3, 4)                         # (B, H, NC+1, N, D)
    all_states = torch.einsum('bhij,bhjnd->bhind', decay_matrix, bx0_t)
    inter_states = all_states[:, :, :NC].permute(0, 2, 1, 3, 4)     # (B, NC, H, N, D)

    # --- Pass 2: SSD parallel within each chunk ---
    x_h = x_c.permute(0, 1, 3, 2, 4)                                # (B, NC, H, CS, D)
    a_h = a_c.permute(0, 1, 3, 2)                                   # (B, NC, H, CS)
    b_h = b_c.permute(0, 1, 3, 2, 4)                                # (B, NC, H, CS, N)
    c_h = c_c.permute(0, 1, 3, 2, 4)                                # (B, NC, H, CS, N)

    A_cumsum = a_h.cumsum(dim=-1)

    L = (A_cumsum.unsqueeze(-1) - A_cumsum.unsqueeze(-2)).exp()
    causal = torch.tril(torch.ones(CS, CS, device=x.device, dtype=torch.bool))
    L = L.masked_fill(~causal, 0.0)

    G = torch.matmul(c_h, b_h.transpose(-1, -2))
    M = L * G
    Y_diag = torch.matmul(M, x_h)

    decay_from_start = A_cumsum.exp()
    Y_off = torch.matmul(c_h, inter_states)
    Y_off = Y_off * decay_from_start.unsqueeze(-1)

    y = Y_diag + Y_off                                              # (B, NC, H, CS, D)
    y = y.permute(0, 1, 3, 2, 4).contiguous().view(B, T, H, D)
    return y.to(orig_dtype)


# --------------------------------------------------------------------------- #
# Benchmark utilities
# --------------------------------------------------------------------------- #

def benchmark_fn(fn, *args, warmup=5, rep=20, **kwargs):
    """Time a function (fwd + bwd) in milliseconds."""
    for _ in range(warmup):
        o = fn(*args, **kwargs)
        o.sum().backward()

    torch.cuda.synchronize()
    timings = []
    for _ in range(rep):
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


# --------------------------------------------------------------------------- #
# Main: correctness checks + throughput benchmarks
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(42)

    B, T, H, D, N = 2, 1024, 4, 64, 16
    chunk_size = 16
    device = 'cuda'

    print("=" * 70)
    print(f"Correctness  (B={B}, T={T}, H={H}, D={D}, N={N}, chunk_size={chunk_size})")
    print("=" * 70)

    # --- inputs (no grad needed for correctness) ---
    x = torch.randn(B, T, H, D, device=device)
    a = torch.empty(B, T, H, device=device).uniform_(-1, 0)
    b_mat = torch.randn(B, T, H, N, device=device) * 0.1
    c_mat = torch.randn(B, T, H, N, device=device) * 0.1

    # standard SSD
    o_naive = naive_mamba2(x, a, b_mat, c_mat)
    o_par   = mamba2_parallel(x, a, b_mat, c_mat, chunk_size=chunk_size)
    print("Standard Mamba2/SSD:")
    print(f"  parallel vs naive: max err = {(o_par - o_naive).abs().max().item():.2e}")

    # CAT-masked SSD
    o_naive_cat = naive_mamba2_cat(x, a, b_mat, c_mat, chunk_size=chunk_size)
    o_par_cat   = mamba2_parallel_cat(x, a, b_mat, c_mat, chunk_size=chunk_size)
    print("CAT-masked Mamba2/SSD:")
    print(f"  parallel vs naive: max err = {(o_par_cat - o_naive_cat).abs().max().item():.2e}")

    # sanity: CAT vs standard should differ (CAT drops information)
    cat_vs_std = (o_naive_cat - o_naive).abs().max().item()
    print(f"  CAT naive vs standard naive: max diff = {cat_vs_std:.2e}  (expected: large)")

    # ---- throughput ----
    print()
    print("=" * 70)
    print(f"Throughput (fwd + bwd)  —  B={B}, T={T}, H={H}, D={D}, N={N}, chunk={chunk_size}")
    print("=" * 70)

    x = torch.randn(B, T, H, D, device=device, requires_grad=True)
    a = torch.empty(B, T, H, device=device).uniform_(-1, 0).requires_grad_(True)
    b_mat = (torch.randn(B, T, H, N, device=device) * 0.1).requires_grad_(True)
    c_mat = (torch.randn(B, T, H, N, device=device) * 0.1).requires_grad_(True)

    benches = [
        ("naive_mamba2",         naive_mamba2,         {}),
        ("mamba2_parallel",      mamba2_parallel,      {"chunk_size": chunk_size}),
        ("naive_mamba2_cat",     naive_mamba2_cat,     {"chunk_size": chunk_size}),
        ("mamba2_parallel_cat",  mamba2_parallel_cat,  {"chunk_size": chunk_size}),
    ]

    results = {}
    for name, fn, kw in benches:
        timings = benchmark_fn(fn, x, a, b_mat, c_mat, **kw)
        avg = sum(timings) / len(timings)
        mn, mx = min(timings), max(timings)
        results[name] = avg
        print(f"  {name:<25s}  {avg:8.2f} ms  (min={mn:.2f}, max={mx:.2f})")

    print()
    print("-" * 70)
    baseline = results.get("mamba2_parallel", 1.0)
    for name, avg in results.items():
        ratio = avg / baseline
        print(f"  {name:<25s}  {ratio:5.2f}x  vs mamba2_parallel")
    print("=" * 70)
