"""
Full-sequence Gated Delta Net — O(L²d) via WY representation, no chunking.

Math:
  1. A = tril(-diag(β̃) · KKᵀ, -1)          L×L strictly lower triangular
  2. T = (I - A)⁻¹                            triangular solve
  3. U = T · diag(β̂) · V                     L×V  (WY-transformed values)
  4. O = exp(G) ⊙ (QKᵀ ⊙ causal_mask) · U   L×V  output

where β̃ = β/exp(g),  β̂ = β/exp(G),  G = cumsum(g).
"""

import torch
from torch import Tensor


def full_sequence_gdn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: float | None = None,
) -> Tensor:
    """Full-sequence GDN — no chunking, O(L²d) compute, O(L²) memory.

    Args:
        q, k: (B, L, H, K)
        v:    (B, L, H, V)
        g:    (B, L, H)     per-step log-gates
        beta: (B, L, H)     per-step learning rates
        scale: optional query scaling (default: 1/√K)

    Returns:
        o: (B, L, H, V)
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    B, L, H, K = q.shape
    V_dim = v.shape[-1]

    # (B, L, H, d) -> (B, H, L, d) for matmul convenience
    q_h = q.permute(0, 2, 1, 3) * scale       # (B, H, L, K)
    k_h = k.permute(0, 2, 1, 3)               # (B, H, L, K)
    v_h = v.permute(0, 2, 1, 3)               # (B, H, L, V)
    g_h = g.permute(0, 2, 1)                   # (B, H, L)
    beta_h = beta.permute(0, 2, 1)             # (B, H, L)

    # ── Gating ──────────────────────────────────────────────
    G = g_h.cumsum(dim=-1)                      # (B, H, L)
    exp_G = G.exp()                             # (B, H, L)

    bt = beta_h / g_h.exp()                     # β̃ = β / exp(g)
    bh = beta_h / exp_G                         # β̂ = β / exp(G)

    # ── Step 1: adjacency matrix A ──────────────────────────
    # A[t, i] = -β̃_t · kₜᵀkᵢ  for i < t,  0 otherwise
    KK = torch.matmul(k_h, k_h.transpose(-1, -2))   # (B, H, L, L)
    A = -(bt.unsqueeze(-1) * KK)                      # scale rows by β̃

    mask = torch.tril(
        torch.ones(L, L, device=q.device, dtype=torch.bool),
        diagonal=-1,
    )
    A = A.masked_fill(~mask, 0.0)

    # ── Step 2: T = (I - A)⁻¹ via triangular solve ─────────
    I_L = torch.eye(L, device=q.device, dtype=q.dtype).expand(B, H, L, L)
    # Solve (I - A) @ T = I  →  T = (I - A)⁻¹
    T = torch.linalg.solve_triangular(I_L - A, I_L, upper=False)

    # ── Step 3: WY-transformed values ───────────────────────
    # U = T @ diag(β̂) @ V    →  (B, H, L, V)
    U = torch.matmul(T * bh.unsqueeze(-2), v_h)

    # ── Step 4: causal attention on transformed values ──────
    QK = torch.matmul(q_h, k_h.transpose(-1, -2))    # (B, H, L, L)
    causal = torch.tril(
        torch.ones(L, L, device=q.device, dtype=torch.bool),
    )
    QK = QK.masked_fill(~causal, 0.0)

    o_h = exp_G.unsqueeze(-1) * torch.matmul(QK, U)   # (B, H, L, V)

    return o_h.permute(0, 2, 1, 3)                     # (B, L, H, V)


# ── Test against naive recurrent implementation ──────────────

def naive_recurrent_gdn(q, k, v, g, beta, scale=None):
    """Step-by-step recurrent GDN for correctness reference."""
    if scale is None:
        scale = q.shape[-1] ** -0.5

    B, L, H, K = q.shape
    V_dim = v.shape[-1]
    q = q * scale

    o = torch.zeros_like(v)
    S = torch.zeros(B, H, K, V_dim, device=q.device, dtype=q.dtype)

    for t in range(L):
        k_t = k[:, t]                                  # (B, H, K)
        v_t = v[:, t]                                  # (B, H, V)
        q_t = q[:, t]                                  # (B, H, K)
        beta_t = beta[:, t]                             # (B, H)
        g_t = g[:, t]                                   # (B, H)

        # S = exp(g) * (S - β (S k - v) kᵀ)
        #   = exp(g) * (S (I - β k kᵀ) + β v kᵀ)
        Sk = torch.einsum('bhkv,bhk->bhv', S, k_t)     # (B, H, V)
        err = Sk - v_t                                   # (B, H, V)
        S = g_t.exp().unsqueeze(-1).unsqueeze(-1) * (
            S - beta_t.unsqueeze(-1).unsqueeze(-1)
            * torch.einsum('bhv,bhk->bhkv', err, k_t)
        )

        o[:, t] = torch.einsum('bhkv,bhk->bhv', S, q_t)

    return o


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64  # f64 for tight tolerance

    B, L, H, K, V = 2, 64, 4, 32, 16

    q = torch.randn(B, L, H, K, device=device, dtype=dtype)
    k = torch.nn.functional.normalize(
        torch.randn(B, L, H, K, device=device, dtype=dtype), dim=-1
    )
    v = torch.randn(B, L, H, V, device=device, dtype=dtype)
    g = torch.randn(B, L, H, device=device, dtype=dtype).abs() * 0.1
    beta = torch.rand(B, L, H, device=device, dtype=dtype) * 0.5

    o_ref = naive_recurrent_gdn(q, k, v, g, beta)
    o_wy = full_sequence_gdn(q, k, v, g, beta)

    err = (o_ref - o_wy).abs().max().item()
    print(f"Max absolute error: {err:.2e}")
    assert err < 1e-8, f"FAILED: error {err:.2e} too large"
    print("PASSED ✓")