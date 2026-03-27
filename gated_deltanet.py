"""
Minimal PyTorch reference implementations of linear attention and the
Gated Delta Rule (https://arxiv.org/abs/2412.06464).

Linear attention:
  1. naive_recurrent  — token-by-token loop (ground truth)
  2. chunk_linear_attn — two-pass chunked (parallel across chunks)
  3. fused_chunk_linear_attn — single-pass chunked (sequential across chunks)

CAT-masked linear attention (suffix _cat):
  4. naive_recurrent_cat
  5. chunk_linear_attn_cat

Gated Delta Rule (suffix _gdn):
  6. naive_recurrent_gdn — token-by-token (ground truth)
  7. chunk_gdn — single-pass chunked

CAT-masked Gated Delta Rule (suffix _gdn_cat):
  8. naive_recurrent_gdn_cat — token-by-token (ground truth for CAT)
  9. chunk_gdn_cat — two-pass chunked (sequential intra-chunk)
 10. chunk_gdn_cat_parallel — two-pass chunked (WY-parallel intra-chunk)

Shapes:
  q, k:  [B, T, H, K]
  v:     [B, T, H, V]
  o:     [B, T, H, V]
  g:     [B, T, H]       (log decay, negative — GDN only)
  beta:  [B, T, H]       (write strength in (0,1) — GDN only)
  state: [B, H, K, V]
"""

import math

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


# =========================================================================== #
# Gated Delta Rule
#
# Recurrence (per token, per head):
#   S_t = exp(g_t) · S_{t-1} + β_t · k_t (v_t − k_t^⊤ S_{t-1})^⊤
#   o_t = q_t^⊤ · S_t
#
# exp(g_t) < 1 decays old memories;  β_t controls write strength;
# the delta term  v_t − k_t^⊤ S_{t-1}  corrects the prediction error.
# =========================================================================== #


# --------------------------------------------------------------------------- #
# 6. Naive recurrent — Gated DeltaNet (ground truth)
# --------------------------------------------------------------------------- #

def naive_recurrent_gdn(
    q: Tensor, k: Tensor, v: Tensor,
    g: Tensor, beta: Tensor,
    scale: float | None = None,
) -> Tensor:
    """Token-by-token gated delta rule.  O(T·K·V) per head, fully sequential."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K = q.shape
    V = v.shape[-1]

    S = q.new_zeros(B, H, K, V)
    o = torch.empty_like(v)

    for t in range(T):
        q_t = q[:, t] * scale          # (B, H, K)
        k_t = k[:, t]                  # (B, H, K)
        v_t = v[:, t]                  # (B, H, V)
        alpha_t = g[:, t].exp()         # (B, H)
        beta_t = beta[:, t]            # (B, H)

        # prediction error
        Sk = einsum('b h k v, b h k -> b h v', S, k_t)
        delta = v_t - Sk

        # gated state update
        S = (alpha_t[:, :, None, None] * S
             + beta_t[:, :, None, None] * einsum('b h k, b h v -> b h k v', k_t, delta))

        o[:, t] = einsum('b h k, b h k v -> b h v', q_t, S)

    return o


# --------------------------------------------------------------------------- #
# 7. Single-pass chunked — Gated DeltaNet
# --------------------------------------------------------------------------- #

def chunk_gdn(
    q: Tensor, k: Tensor, v: Tensor,
    g: Tensor, beta: Tensor,
    scale: float | None = None,
    chunk_size: int = 64,
) -> Tensor:
    """Single-pass chunked gated delta rule.

    Sequential across chunks with a C-step inner recurrence per chunk.
    Produces identical output to naive_recurrent_gdn.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NC = T // C

    q_c = rearrange(q, 'b (nc c) h k -> b nc c h k', c=C)
    k_c = rearrange(k, 'b (nc c) h k -> b nc c h k', c=C)
    v_c = rearrange(v, 'b (nc c) h v -> b nc c h v', c=C)
    g_c = rearrange(g, 'b (nc c) h -> b nc c h', c=C)
    beta_c = rearrange(beta, 'b (nc c) h -> b nc c h', c=C)

    S = q.new_zeros(B, H, K, V)
    o_chunks = []

    for i in range(NC):
        o_chunk = q.new_empty(B, C, H, V)
        for j in range(C):
            q_t = q_c[:, i, j] * scale
            k_t = k_c[:, i, j]
            v_t = v_c[:, i, j]
            alpha_t = g_c[:, i, j].exp()
            beta_t = beta_c[:, i, j]

            Sk = einsum('b h k v, b h k -> b h v', S, k_t)
            delta = v_t - Sk
            S = (alpha_t[:, :, None, None] * S
                 + beta_t[:, :, None, None] * einsum('b h k, b h v -> b h k v', k_t, delta))
            o_chunk[:, j] = einsum('b h k, b h k v -> b h v', q_t, S)

        o_chunks.append(o_chunk)

    return torch.cat(o_chunks, dim=1)


# --------------------------------------------------------------------------- #
# 7b. Two-pass chunked — Gated DeltaNet (WY-parallel intra-chunk)
# --------------------------------------------------------------------------- #

@torch.compile
def _parallel_scan_linear(Phi: Tensor, u: Tensor) -> Tensor:
    """Hillis-Steele parallel prefix scan for the matrix linear recurrence
        S_i = Phi_i @ S_{i-1} + u_i,   S_0 = 0.

    Args:
        Phi: (B, NC, H, K, K)  transition matrices per chunk
        u:   (B, NC, H, K, V)  input vectors per chunk

    Returns:
        states: (B, NC, H, K, V) — state BEFORE each chunk (exclusive prefix).
    """
    B, NC, H, K, V = u.shape

    Phi_cur = Phi.clone()
    u_cur = u.clone()

    n_steps = math.ceil(math.log2(NC)) if NC > 1 else 0
    for d in range(n_steps):
        stride = 1 << d
        if stride >= NC:
            break
        Phi_prev = Phi_cur[:, :NC - stride]
        u_prev = u_cur[:, :NC - stride]

        new_Phi = torch.matmul(Phi_cur[:, stride:], Phi_prev)
        new_u = torch.matmul(Phi_cur[:, stride:], u_prev) + u_cur[:, stride:]

        Phi_cur = torch.cat([Phi_cur[:, :stride], new_Phi], dim=1)
        u_cur = torch.cat([u_cur[:, :stride], new_u], dim=1)

    # u_cur is now the inclusive scan (S_after_chunk_i).
    # Shift right to get exclusive prefix (S_before_chunk_i).
    return torch.cat([u.new_zeros(B, 1, H, K, V), u_cur[:, :-1]], dim=1)


@torch.compile
def _gdn_pass2(q_h, k_h, v_h, states, W, U, exp_G_h):
    """Pass 2: compute all chunk outputs in parallel using WY (compiled)."""
    C = q_h.shape[-2]
    V_corr = U - torch.matmul(W, states)
    inter = torch.matmul(q_h, states)

    QK = torch.matmul(q_h, k_h.transpose(-1, -2))
    causal = torch.tril(torch.ones(C, C, device=q_h.device, dtype=torch.bool))
    QK = QK.masked_fill(~causal, 0.0)
    intra = torch.matmul(QK, V_corr)

    return exp_G_h.unsqueeze(-1) * (inter + intra)


def chunk_gdn_parallel(
    q: Tensor, k: Tensor, v: Tensor,
    g: Tensor, beta: Tensor,
    scale: float | None = None,
    chunk_size: int = 16,
) -> Tensor:
    """Two-pass chunked gated delta rule — fully parallel.

    Same result as chunk_gdn / naive_recurrent_gdn, but replaces the
    double sequential loop (NC × C) with:
      - WY/UT computation for all chunks in parallel  (intra-chunk)
      - Hillis-Steele parallel prefix scan             (inter-chunk)
      - Parallel output computation for all chunks

    The per-chunk transition is a K×K matrix linear recurrence:
        S_i = Φ_i · S_{i-1} + u_i
    where Φ_i = exp(G_last_i) · (I − kᵀ W_i)  and  u_i = exp(G_last_i) · ψ_i.
    This is solved in O(log NC) parallel steps via a Hillis-Steele scan
    with K×K batched matmuls.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K_dim = q.shape
    V_dim = v.shape[-1]
    C = chunk_size
    NC = T // C

    q_c = rearrange(q, 'b (nc c) h k -> b nc c h k', c=C)
    k_c = rearrange(k, 'b (nc c) h k -> b nc c h k', c=C)
    v_c = rearrange(v, 'b (nc c) h v -> b nc c h v', c=C)
    g_c = rearrange(g, 'b (nc c) h -> b nc c h', c=C)
    beta_c = rearrange(beta, 'b (nc c) h -> b nc c h', c=C)

    q_h = rearrange(q_c, 'b n c h k -> b n h c k') * scale
    k_h = rearrange(k_c, 'b n c h k -> b n h c k')
    v_h = rearrange(v_c, 'b n c h v -> b n h c v')

    # --- WY computation (all chunks in parallel) ---
    G_c = g_c.cumsum(dim=2)
    exp_G_h = G_c.exp().permute(0, 1, 3, 2)                         # (B, NC, H, C)
    alpha_c = g_c.exp()

    bt_h = (beta_c / alpha_c).permute(0, 1, 3, 2)                   # β̃ = β/exp(g)
    bh_h = (beta_c / G_c.exp()).permute(0, 1, 3, 2)                 # β̂ = β/exp(G)

    KK = torch.matmul(k_h, k_h.transpose(-1, -2))                   # (B, NC, H, C, C)
    A = -(bt_h.unsqueeze(-1) * KK)
    strict_lower = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool), diagonal=-1)
    A = A * strict_lower

    I_C = torch.eye(C, device=q.device, dtype=q.dtype).expand_as(A)
    T_mat = I_C
    for _ in range(C - 1):
        T_mat = I_C + torch.matmul(A, T_mat)

    W = torch.matmul(T_mat * bt_h.unsqueeze(-2), k_h)               # (B, NC, H, C, K)
    U = torch.matmul(T_mat * bh_h.unsqueeze(-2), v_h)               # (B, NC, H, C, V)

    # --- Build per-chunk transition (Φ_i, u_i) for the linear recurrence ---
    # Φ_i = exp(G_last_i) · (I_K − kᵀ W_i)              (B, NC, H, K, K)
    # u_i = exp(G_last_i) · ψ_i = exp(G_last_i) · kᵀ U  (B, NC, H, K, V)
    kW = torch.matmul(k_h.transpose(-1, -2), W)                     # (B, NC, H, K, K)
    exp_G_last = exp_G_h[..., -1]                                    # (B, NC, H)
    exp_G_last_mm = exp_G_last[..., None, None]                      # (B, NC, H, 1, 1)

    I_K = torch.eye(K_dim, device=q.device, dtype=q.dtype)
    Phi = exp_G_last_mm * (I_K - kW)                                 # (B, NC, H, K, K)

    psi = torch.matmul(k_h.transpose(-1, -2), U)                    # (B, NC, H, K, V)
    u = exp_G_last_mm * psi                                          # (B, NC, H, K, V)

    # --- Parallel inter-chunk state propagation (Hillis-Steele scan) ---
    states = _parallel_scan_linear(Phi, u)                           # (B, NC, H, K, V)

    # --- Parallel output computation ---
    o_h = _gdn_pass2(q_h, k_h, v_h, states, W, U, exp_G_h)
    return rearrange(o_h, 'b n h c v -> b (n c) h v')


# =========================================================================== #
# CAT-masked Gated Delta Rule
#
# Same two-state approach as the linear-attention CAT variants:
#   S_inter — built from first tokens of past chunks via the delta rule.
#   Intra-chunk — full delta rule recurrence starting from S_inter.
# =========================================================================== #


# --------------------------------------------------------------------------- #
# 8. Naive recurrent — CAT-masked Gated DeltaNet (ground truth for CAT)
# --------------------------------------------------------------------------- #

def naive_recurrent_gdn_cat(
    q: Tensor, k: Tensor, v: Tensor,
    g: Tensor, beta: Tensor,
    scale: float | None = None,
    chunk_size: int = 8,
) -> Tensor:
    """Token-by-token gated delta rule respecting the CAT mask.

    S_inter is accumulated via the delta rule from first tokens only.
    Within each chunk the full delta rule recurrence runs from S_inter.
    After each chunk, S_inter is updated using the first token applied
    to the *old* S_inter (before the chunk was processed).
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
        # snapshot S_inter before this chunk for the post-chunk update
        S_inter_old = S_inter.clone()
        S = S_inter.clone()

        for j in range(C):
            t = c * C + j
            q_t = q[:, t] * scale
            k_t = k[:, t]
            v_t = v[:, t]
            alpha_t = g[:, t].exp()
            beta_t = beta[:, t]

            Sk = einsum('b h k v, b h k -> b h v', S, k_t)
            delta = v_t - Sk
            S = (alpha_t[:, :, None, None] * S
                 + beta_t[:, :, None, None] * einsum('b h k, b h v -> b h k v', k_t, delta))
            o[:, t] = einsum('b h k, b h k v -> b h v', q_t, S)

        # update S_inter with first token (delta rule on the old S_inter)
        k0 = k[:, c * C]
        v0 = v[:, c * C]
        a0 = g[:, c * C].exp()
        b0 = beta[:, c * C]
        Sk0 = einsum('b h k v, b h k -> b h v', S_inter_old, k0)
        d0 = v0 - Sk0
        S_inter = (a0[:, :, None, None] * S_inter_old
                   + b0[:, :, None, None] * einsum('b h k, b h v -> b h k v', k0, d0))

    return o


# --------------------------------------------------------------------------- #
# 9. Two-pass chunked — CAT-masked Gated DeltaNet
# --------------------------------------------------------------------------- #

def chunk_gdn_cat(
    q: Tensor, k: Tensor, v: Tensor,
    g: Tensor, beta: Tensor,
    scale: float | None = None,
    chunk_size: int = 8,
) -> Tensor:
    """Two-pass chunked gated delta rule under the CAT mask.

    Pass 1 (sequential): build S_inter at each chunk boundary by running
            the delta rule on first tokens only.
    Pass 2 (independent per chunk): given S_inter, run the C-step
            intra-chunk delta rule recurrence.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NC = T // C

    q_c = rearrange(q, 'b (nc c) h k -> b nc c h k', c=C)
    k_c = rearrange(k, 'b (nc c) h k -> b nc c h k', c=C)
    v_c = rearrange(v, 'b (nc c) h v -> b nc c h v', c=C)
    g_c = rearrange(g, 'b (nc c) h -> b nc c h', c=C)
    beta_c = rearrange(beta, 'b (nc c) h -> b nc c h', c=C)

    # --- Pass 1: S_inter at each chunk boundary (first tokens only) ---
    states = []
    S_inter = q.new_zeros(B, H, K, V)
    for i in range(NC):
        states.append(S_inter)
        k0 = k_c[:, i, 0]
        v0 = v_c[:, i, 0]
        a0 = g_c[:, i, 0].exp()
        b0 = beta_c[:, i, 0]

        Sk0 = einsum('b h k v, b h k -> b h v', S_inter, k0)
        d0 = v0 - Sk0
        S_inter = (a0[:, :, None, None] * S_inter
                   + b0[:, :, None, None] * einsum('b h k, b h v -> b h k v', k0, d0))

    # --- Pass 2: per-chunk recurrence from S_inter (independent) ---
    o_chunks = []
    for i in range(NC):
        S = states[i].clone()
        o_chunk = q.new_empty(B, C, H, V)
        for j in range(C):
            q_t = q_c[:, i, j] * scale
            k_t = k_c[:, i, j]
            v_t = v_c[:, i, j]
            alpha_t = g_c[:, i, j].exp()
            beta_t = beta_c[:, i, j]

            Sk = einsum('b h k v, b h k -> b h v', S, k_t)
            delta = v_t - Sk
            S = (alpha_t[:, :, None, None] * S
                 + beta_t[:, :, None, None] * einsum('b h k, b h v -> b h k v', k_t, delta))
            o_chunk[:, j] = einsum('b h k, b h k v -> b h v', q_t, S)

        o_chunks.append(o_chunk)

    return torch.cat(o_chunks, dim=1)


# --------------------------------------------------------------------------- #
# 10. Two-pass chunked — CAT-masked Gated DeltaNet (WY-parallel intra-chunk)
# --------------------------------------------------------------------------- #

@torch.compile
def _gdn_cat_pass1_step(S, k0, a0, b0, kv0):
    """Single step of the inter-chunk delta rule recurrence (compiled)."""
    Sk0 = (S * k0.unsqueeze(-1)).sum(-2)
    return a0 * S + kv0 - b0 * k0.unsqueeze(-1) * Sk0.unsqueeze(-2)


@torch.compile
def _gdn_cat_pass2(q_h, k_h, v_h, states, g_c, beta_c):
    """Pass 2: WY + UT parallel computation (compiled)."""
    C = q_h.shape[-2]
    G_c = g_c.cumsum(dim=2)
    exp_G_h = G_c.exp().permute(0, 1, 3, 2)
    alpha_c = g_c.exp()

    bt_h = (beta_c / alpha_c).permute(0, 1, 3, 2)
    bh_h = (beta_c / G_c.exp()).permute(0, 1, 3, 2)

    KK = torch.matmul(k_h, k_h.transpose(-1, -2))
    A = -(bt_h.unsqueeze(-1) * KK)
    strict_lower = torch.tril(torch.ones(C, C, device=q_h.device, dtype=torch.bool), diagonal=-1)
    A = A * strict_lower

    I_C = torch.eye(C, device=q_h.device, dtype=q_h.dtype).expand_as(A)
    T = I_C
    for _ in range(C - 1):
        T = I_C + torch.matmul(A, T)

    W = torch.matmul(T * bt_h.unsqueeze(-2), k_h)
    U = torch.matmul(T * bh_h.unsqueeze(-2), v_h)

    V_corr = U - torch.matmul(W, states)
    inter = torch.matmul(q_h, states)

    QK = torch.matmul(q_h, k_h.transpose(-1, -2))
    causal = torch.tril(torch.ones(C, C, device=q_h.device, dtype=torch.bool))
    QK = QK.masked_fill(~causal, 0.0)
    intra = torch.matmul(QK, V_corr)

    return exp_G_h.unsqueeze(-1) * (inter + intra)


def chunk_gdn_cat_parallel(
    q: Tensor, k: Tensor, v: Tensor,
    g: Tensor, beta: Tensor,
    scale: float | None = None,
    chunk_size: int = 8,
) -> Tensor:
    """Two-pass chunked gated delta rule under the CAT mask — parallel intra-chunk.

    Same result as chunk_gdn_cat, but Pass 2 uses the WY representation and
    UT transform to replace the token-by-token loop with batched matmuls.
    Reference: https://sustcsonglin.github.io/blog/2024/deltanet-2/

    The gated decay is absorbed by defining a normalised state
        Ŝ_j = exp(-G_j) S_j,   G_j = cumsum(g)[j]
    whose transition matrix becomes (I − β̃ k kᵀ) with β̃ = β/exp(g),
    admitting a WY factorisation.  The input coefficient becomes β̂ = β/exp(G).

    All operations run in the input dtype (bfloat16-friendly).  The CxC
    triangular inverse uses a Neumann series (exact for nilpotent A)
    instead of solve_triangular, avoiding any float32 promotion.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K_dim = q.shape
    V_dim = v.shape[-1]
    C = chunk_size
    NC = T // C

    q_c = rearrange(q, 'b (nc c) h k -> b nc c h k', c=C)
    k_c = rearrange(k, 'b (nc c) h k -> b nc c h k', c=C)
    v_c = rearrange(v, 'b (nc c) h v -> b nc c h v', c=C)
    g_c = rearrange(g, 'b (nc c) h -> b nc c h', c=C)
    beta_c = rearrange(beta, 'b (nc c) h -> b nc c h', c=C)

    # --- Pass 1: S_inter at each chunk boundary (first tokens only) ---
    # Pre-extract all first-token quantities to avoid per-iteration indexing.
    k0_all = k_c[:, :, 0]                                         # (B, NC, H, K)
    a0_all = g_c[:, :, 0].exp()                                   # (B, NC, H)
    b0_all = beta_c[:, :, 0]                                      # (B, NC, H)
    # Pre-compute β·k⊗v for each chunk's first token
    kv0_all = (b0_all.unsqueeze(-1).unsqueeze(-1)
               * k0_all.unsqueeze(-1) * v_c[:, :, 0].unsqueeze(-2))  # (B, NC, H, K, V)

    # Pre-expand decay/beta dims for the compiled step function
    a0_exp = a0_all[:, :, :, None, None]                             # (B, NC, H, 1, 1)
    b0_exp = b0_all[:, :, :, None, None]                             # (B, NC, H, 1, 1)

    states = q.new_zeros(B, NC, H, K_dim, V_dim)
    S = q.new_zeros(B, H, K_dim, V_dim)
    for i in range(NC):
        states[:, i] = S
        S = _gdn_cat_pass1_step(
            S, k0_all[:, i], a0_exp[:, i], b0_exp[:, i], kv0_all[:, i],
        )

    # --- Pass 2: WY + UT parallel computation (compiled) ---
    q_h = rearrange(q_c, 'b n c h k -> b n h c k') * scale
    k_h = rearrange(k_c, 'b n c h k -> b n h c k')
    v_h = rearrange(v_c, 'b n c h v -> b n h c v')

    o_h = _gdn_cat_pass2(q_h, k_h, v_h, states, g_c, beta_c)
    return rearrange(o_h, 'b n h c v -> b (n c) h v')


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


def _l2_normalize(x: Tensor) -> Tensor:
    """L2-normalize along last dim (matches GDN's qk_norm='l2')."""
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(42)
    B, T, H, K, V = 2, 1024, 4, 64, 64
    chunk_size = 16
    device = 'cuda'

    q = torch.randn(B, T, H, K, device=device)
    k = torch.randn(B, T, H, K, device=device)
    v = torch.randn(B, T, H, V, device=device)

    # GDN inputs: L2-normalised q/k (required for stability) + decay/beta
    q_n = _l2_normalize(q)
    k_n = _l2_normalize(k)
    g = -torch.rand(B, T, H, device=device).abs()         # negative log-decay
    beta = torch.rand(B, T, H, device=device) * 0.5       # write strength ∈ (0, 0.5)

    # ---- correctness: linear attention ----
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

    # ---- correctness: gated delta rule (L2-normed q, k) ----
    print()
    o_gdn_naive = naive_recurrent_gdn(q_n, k_n, v, g, beta)
    o_gdn_chunk = chunk_gdn(q_n, k_n, v, g, beta, chunk_size=chunk_size)
    o_gdn_par   = chunk_gdn_parallel(q_n, k_n, v, g, beta, chunk_size=chunk_size)
    print("Gated DeltaNet (L2-normed q, k):")
    print(f"  chunk    vs naive: max err = {(o_gdn_chunk - o_gdn_naive).abs().max().item():.2e}")
    print(f"  parallel vs naive: max err = {(o_gdn_par - o_gdn_naive).abs().max().item():.2e}")

    o_gdn_naive_cat = naive_recurrent_gdn_cat(q_n, k_n, v, g, beta, chunk_size=chunk_size)
    o_gdn_chunk_cat = chunk_gdn_cat(q_n, k_n, v, g, beta, chunk_size=chunk_size)
    o_gdn_par_cat = chunk_gdn_cat_parallel(q_n, k_n, v, g, beta, chunk_size=chunk_size)
    print("CAT-masked Gated DeltaNet:")
    print(f"  chunk    vs naive: max err = {(o_gdn_chunk_cat - o_gdn_naive_cat).abs().max().item():.2e}")
    print(f"  parallel vs naive: max err = {(o_gdn_par_cat - o_gdn_naive_cat).abs().max().item():.2e}")

    # ---- throughput ----
    print()
    print("=" * 70)
    print(f"Throughput (fwd + bwd)  —  B={B}, T={T}, H={H}, K={K}, chunk={chunk_size}")
    print("=" * 70)

    q = torch.randn(B, T, H, K, device=device, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, requires_grad=True)

    # leaf tensors for GDN throughput (proper requires_grad)
    q_n = torch.randn(B, T, H, K, device=device)
    q_n = _l2_normalize(q_n).detach().requires_grad_(True)
    k_n = torch.randn(B, T, H, K, device=device)
    k_n = _l2_normalize(k_n).detach().requires_grad_(True)
    v2 = torch.randn(B, T, H, V, device=device, requires_grad=True)
    g = torch.empty(B, T, H, device=device).uniform_(-1, 0).requires_grad_(True)
    beta_t = torch.empty(B, T, H, device=device).uniform_(0, 0.5).requires_grad_(True)

    print("\nLinear attention:")
    la_benches = [
        ("naive_recurrent",       naive_recurrent,       {}),
        ("chunk_linear_attn",     chunk_linear_attn,     {"chunk_size": chunk_size}),
        # ("fused_chunk_linear",    fused_chunk_linear_attn, {"chunk_size": chunk_size}),
        # ("reference_cat (quad)",  reference_cat_attn,    {"chunk_size": chunk_size}),
        # ("naive_recurrent_cat",   naive_recurrent_cat,   {"chunk_size": chunk_size}),
        # ("chunk_linear_attn_cat", chunk_linear_attn_cat, {"chunk_size": chunk_size}),
    ]

    results = {}
    for name, fn, kw in la_benches:
        timings = benchmark_fn(fn, q, k, v, **kw)
        avg = sum(timings) / len(timings)
        mn, mx = min(timings), max(timings)
        results[name] = avg
        print(f"  {name:<25s}  {avg:8.2f} ms  (min={mn:.2f}, max={mx:.2f})")

    print("\nGated DeltaNet:")
    gdn_benches = [
        # ("naive_recurrent_gdn",     naive_recurrent_gdn,     {}),
        # ("chunk_gdn",               chunk_gdn,               {"chunk_size": chunk_size}),
        ("chunk_gdn_parallel",      chunk_gdn_parallel,      {"chunk_size": chunk_size}),
        # ("naive_recurrent_gdn_cat", naive_recurrent_gdn_cat, {"chunk_size": chunk_size}),
        # ("chunk_gdn_cat",           chunk_gdn_cat,           {"chunk_size": chunk_size}),
        ("chunk_gdn_cat_parallel",  chunk_gdn_cat_parallel,  {"chunk_size": chunk_size}),
    ]

    for name, fn, kw in gdn_benches:
        timings = benchmark_fn(fn, q_n, k_n, v2, g, beta_t, **kw)
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

    # ================================================================== #
    # Comparison with fla library (Triton kernels, requires bfloat16)
    # ================================================================== #
    print()
    print("=" * 70)
    print("Comparison with fla library  (bfloat16)")
    print("=" * 70)

    try:
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule as fla_chunk_gdn,
            fused_recurrent_gated_delta_rule as fla_recurrent_gdn,
        )
    except ImportError:
        print("  fla library not available — skipping comparison.")
        exit()

    torch.manual_seed(42)
    dtype = torch.bfloat16

    q_bf = _l2_normalize(torch.randn(B, T, H, K, device=device, dtype=dtype))
    k_bf = _l2_normalize(torch.randn(B, T, H, K, device=device, dtype=dtype))
    v_bf = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g_bf = -torch.rand(B, T, H, device=device, dtype=dtype).abs()
    beta_bf = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()

    # --- Correctness (ours vs fla) ---
    print("\nCorrectness (ours in bf16 vs fla in bf16):")

    o_ours_naive = naive_recurrent_gdn(q_bf, k_bf, v_bf, g_bf, beta_bf)
    o_ours_chunk = chunk_gdn(q_bf, k_bf, v_bf, g_bf, beta_bf, chunk_size=chunk_size)
    print(f"  ours: chunk vs naive       max err = {(o_ours_chunk - o_ours_naive).abs().max().item():.2e}")

    o_fla_chunk, _ = fla_chunk_gdn(q_bf, k_bf, v_bf, g_bf, beta_bf)
    o_fla_recur, _ = fla_recurrent_gdn(q_bf, k_bf, v_bf, g_bf, beta_bf)

    print(f"  fla chunk   vs ours naive  max err = {(o_fla_chunk - o_ours_naive).abs().max().item():.2e}")
    print(f"  fla recur   vs ours naive  max err = {(o_fla_recur - o_ours_naive).abs().max().item():.2e}")
    print(f"  fla chunk   vs fla recur   max err = {(o_fla_chunk - o_fla_recur).abs().max().item():.2e}")
    print(f"  fla chunk   vs ours chunk  max err = {(o_fla_chunk - o_ours_chunk).abs().max().item():.2e}")

    # --- Throughput (ours vs fla, bfloat16) ---
    print(f"\nThroughput  (fwd + bwd, bf16)  —  B={B}, T={T}, H={H}, K={K}, chunk={chunk_size}")

    q_bf = _l2_normalize(torch.randn(B, T, H, K, device=device, dtype=dtype)).detach().requires_grad_(True)
    k_bf = _l2_normalize(torch.randn(B, T, H, K, device=device, dtype=dtype)).detach().requires_grad_(True)
    v_bf = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g_bf = torch.empty(B, T, H, device=device, dtype=dtype).uniform_(-1, 0).requires_grad_(True)
    beta_bf = torch.empty(B, T, H, device=device, dtype=dtype).uniform_(0.1, 0.9).requires_grad_(True)

    def fla_chunk_wrapper(q, k, v, g, beta, **kw):
        o, _ = fla_chunk_gdn(q, k, v, g, beta)
        return o

    def fla_recur_wrapper(q, k, v, g, beta, **kw):
        o, _ = fla_recurrent_gdn(q, k, v, g, beta)
        return o

    fla_benches = [
        ("ours: naive_recurrent_gdn",  naive_recurrent_gdn,    {}),
        ("ours: chunk_gdn",            chunk_gdn,              {"chunk_size": chunk_size}),
        ("ours: chunk_gdn_cat_par",    chunk_gdn_cat_parallel, {"chunk_size": chunk_size}),
        ("fla:  chunk_gdn (Triton)",   fla_chunk_wrapper,      {}),
        # fused_recurrent_gdn omitted: backward not implemented in fla
    ]

    fla_results = {}
    for name, fn, kw in fla_benches:
        timings = benchmark_fn(fn, q_bf, k_bf, v_bf, g_bf, beta_bf, **kw)
        avg = sum(timings) / len(timings)
        mn, mx = min(timings), max(timings)
        fla_results[name] = avg
        print(f"  {name:<30s}  {avg:8.2f} ms  (min={mn:.2f}, max={mx:.2f})")

    print()
    print("-" * 70)
    fla_baseline = fla_results["fla:  chunk_gdn (Triton)"]
    for name, avg in fla_results.items():
        ratio = avg / fla_baseline
        print(f"  {name:<30s}  {ratio:5.2f}x  vs fla chunk (Triton)")
    print("=" * 70)