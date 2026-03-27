"""
Profile the time breakdown of chunk_gdn_parallel vs mamba2_parallel.

For each function, we instrument the individual stages:
  - GDN:   (1) WY/UT intra-chunk  (2) inter-chunk parallel scan  (3) output pass2
  - Mamba2: (1) intra-chunk (diag) (2) inter-chunk state prop     (3) state readout

We time forward-only (no backward) for clean per-stage numbers, then also
time fwd+bwd end-to-end for the complete picture.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from mamba2_ssd import segment_sum

torch.set_float32_matmul_precision('high')


# =========================================================================== #
#  Instrumented chunk_gdn_parallel  (forward only, returns per-stage timings)
# =========================================================================== #

def profile_chunk_gdn_parallel(
    q: Tensor, k: Tensor, v: Tensor,
    g: Tensor, beta: Tensor,
    scale: float | None = None,
    chunk_size: int = 16,
    warmup: int = 5,
    rep: int = 20,
):
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

    # ---------- stage 1: WY / UT intra-chunk ----------
    def stage1_wy():
        G_c = g_c.cumsum(dim=2)
        exp_G_h = G_c.exp().permute(0, 1, 3, 2)
        alpha_c = g_c.exp()

        bt_h = (beta_c / alpha_c).permute(0, 1, 3, 2)
        bh_h = (beta_c / G_c.exp()).permute(0, 1, 3, 2)

        KK = torch.matmul(k_h, k_h.transpose(-1, -2))
        A = -(bt_h.unsqueeze(-1) * KK)
        strict_lower = torch.tril(torch.ones(C, C, device=q.device, dtype=q.dtype), diagonal=-1)
        A = A * strict_lower

        I_C = torch.eye(C, device=q.device, dtype=q.dtype).expand_as(A)
        T_mat = I_C
        for _ in range(C - 1):
            T_mat = I_C + torch.matmul(A, T_mat)

        W = torch.matmul(T_mat * bt_h.unsqueeze(-2), k_h)
        U = torch.matmul(T_mat * bh_h.unsqueeze(-2), v_h)
        return exp_G_h, W, U

    # ---------- stage 2: inter-chunk parallel scan ----------
    def stage2_scan(exp_G_h, W, U):
        kW = torch.matmul(k_h.transpose(-1, -2), W)
        exp_G_last = exp_G_h[..., -1]
        exp_G_last_mm = exp_G_last[..., None, None]

        I_K = torch.eye(K_dim, device=q.device, dtype=q.dtype)
        Phi = exp_G_last_mm * (I_K - kW)

        psi = torch.matmul(k_h.transpose(-1, -2), U)
        u = exp_G_last_mm * psi

        # Hillis-Steele scan
        Phi_cur = Phi.clone()
        u_cur = u.clone()
        n_steps = math.ceil(math.log2(NC)) if NC > 1 else 0
        for d in range(n_steps):
            stride = 1 << d
            if stride >= NC:
                break
            new_Phi = torch.matmul(Phi_cur[:, stride:], Phi_cur[:, :NC - stride])
            new_u = torch.matmul(Phi_cur[:, stride:], u_cur[:, :NC - stride]) + u_cur[:, stride:]
            Phi_cur = torch.cat([Phi_cur[:, :stride], new_Phi], dim=1)
            u_cur = torch.cat([u_cur[:, :stride], new_u], dim=1)

        states = torch.cat([u.new_zeros(B, 1, H, K_dim, V_dim), u_cur[:, :-1]], dim=1)
        return states

    # ---------- stage 3: output pass2 ----------
    def stage3_output(exp_G_h, W, U, states):
        V_corr = U - torch.matmul(W, states)
        inter = torch.matmul(q_h, states)

        QK = torch.matmul(q_h, k_h.transpose(-1, -2))
        causal = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))
        QK = QK.masked_fill(~causal, 0.0)
        intra = torch.matmul(QK, V_corr)

        o_h = exp_G_h.unsqueeze(-1) * (inter + intra)
        return rearrange(o_h, 'b n h c v -> b (n c) h v')

    # warm up all stages
    for _ in range(warmup):
        exp_G_h, W, U = stage1_wy()
        states = stage2_scan(exp_G_h, W, U)
        _ = stage3_output(exp_G_h, W, U, states)
    torch.cuda.synchronize()

    timings = {1: [], 2: [], 3: []}
    for _ in range(rep):
        torch.cuda.synchronize()
        e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

        e[0].record()
        exp_G_h, W, U = stage1_wy()
        e[1].record()
        states = stage2_scan(exp_G_h, W, U)
        e[2].record()
        _ = stage3_output(exp_G_h, W, U, states)
        e[3].record()

        torch.cuda.synchronize()
        timings[1].append(e[0].elapsed_time(e[1]))
        timings[2].append(e[1].elapsed_time(e[2]))
        timings[3].append(e[2].elapsed_time(e[3]))

    return timings


# =========================================================================== #
#  Instrumented mamba2_parallel  (forward only, returns per-stage timings)
# =========================================================================== #

def profile_mamba2_parallel(
    x: Tensor, a: Tensor, b: Tensor, c: Tensor,
    chunk_size: int = 64,
    warmup: int = 5,
    rep: int = 20,
):
    B, T, H, D = x.shape
    N = b.shape[-1]
    CS = chunk_size
    NC = T // CS

    x_h = rearrange(x, 'b (nc cs) h d -> b nc h cs d', cs=CS)
    a_h = rearrange(a, 'b (nc cs) h   -> b nc h cs',   cs=CS)
    b_h = rearrange(b, 'b (nc cs) h n -> b nc h cs n', cs=CS)
    c_h = rearrange(c, 'b (nc cs) h n -> b nc h cs n', cs=CS)

    # ---------- stage 1: intra-chunk (diagonal blocks) ----------
    def stage1_intra():
        A_cumsum = a_h.cumsum(dim=-1)

        L = (A_cumsum.unsqueeze(-1) - A_cumsum.unsqueeze(-2)).exp()
        causal = torch.tril(torch.ones(CS, CS, device=x.device, dtype=torch.bool))
        L = L.masked_fill(~causal, 0.0)

        G = torch.matmul(c_h, b_h.transpose(-1, -2))
        M = L * G
        Y_diag = torch.matmul(M, x_h)

        decay_to_end = (A_cumsum[..., -1:] - A_cumsum).exp()
        b_decay = b_h * decay_to_end.unsqueeze(-1)
        chunk_states = torch.matmul(b_decay.transpose(-1, -2), x_h)

        return Y_diag, A_cumsum, chunk_states

    # ---------- stage 2: inter-chunk state propagation ----------
    def stage2_inter(A_cumsum, chunk_states):
        log_chunk_decay = A_cumsum[..., -1]
        padded = F.pad(log_chunk_decay, (0, 0, 1, 0))
        padded_t = padded.permute(0, 2, 1)
        decay_matrix = segment_sum(padded_t).exp()

        chunk_states_pad = torch.cat(
            [x.new_zeros(B, 1, H, N, D), chunk_states], dim=1,
        )
        cs_t = chunk_states_pad.permute(0, 2, 1, 3, 4)
        all_states = torch.einsum('bhij,bhjnd->bhind', decay_matrix, cs_t)
        inter_states = all_states[:, :, :NC].permute(0, 2, 1, 3, 4)
        return inter_states

    # ---------- stage 3: state readout (off-diagonal) ----------
    def stage3_readout(Y_diag, A_cumsum, inter_states):
        decay_from_start = A_cumsum.exp()
        Y_off = torch.matmul(c_h, inter_states)
        Y_off = Y_off * decay_from_start.unsqueeze(-1)

        y = Y_diag + Y_off
        return rearrange(y, 'b nc h cs d -> b (nc cs) h d')

    # warm up
    for _ in range(warmup):
        Y_diag, A_cumsum, chunk_states = stage1_intra()
        inter_states = stage2_inter(A_cumsum, chunk_states)
        _ = stage3_readout(Y_diag, A_cumsum, inter_states)
    torch.cuda.synchronize()

    timings = {1: [], 2: [], 3: []}
    for _ in range(rep):
        torch.cuda.synchronize()
        e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

        e[0].record()
        Y_diag, A_cumsum, chunk_states = stage1_intra()
        e[1].record()
        inter_states = stage2_inter(A_cumsum, chunk_states)
        e[2].record()
        _ = stage3_readout(Y_diag, A_cumsum, inter_states)
        e[3].record()

        torch.cuda.synchronize()
        timings[1].append(e[0].elapsed_time(e[1]))
        timings[2].append(e[1].elapsed_time(e[2]))
        timings[3].append(e[2].elapsed_time(e[3]))

    return timings


# =========================================================================== #
#  End-to-end fwd+bwd benchmark
# =========================================================================== #

def benchmark_e2e(fn, *args, warmup=5, rep=20, **kwargs):
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


# =========================================================================== #
#  Main
# =========================================================================== #

if __name__ == '__main__':
    from gated_deltanet import chunk_gdn_parallel
    from mamba2_ssd import mamba2_parallel

    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.float32

    configs = [
        # (B, T, H, K/D, V/D, N, chunk_size, label)
        (2, 1024, 4, 64, 64, 16, 16, "small  (B=2  T=1024 H=4  K=64)"),
        (4, 2048, 8, 64, 64, 16, 16, "medium (B=4  T=2048 H=8  K=64)"),
        (4, 4096, 8, 64, 64, 16, 16, "large  (B=4  T=4096 H=8  K=64)"),
        (4, 4096, 8, 128, 128, 16, 16, "xl     (B=4  T=4096 H=8  K=128)"),
    ]

    for B, T, H, K, V, N, CS, label in configs:
        print()
        print("=" * 78)
        print(f"  {label}   chunk_size={CS}")
        print("=" * 78)

        # --- GDN inputs ---
        q = torch.randn(B, T, H, K, device=device, dtype=dtype)
        k = torch.randn(B, T, H, K, device=device, dtype=dtype)
        v = torch.randn(B, T, H, V, device=device, dtype=dtype)
        g = torch.empty(B, T, H, device=device, dtype=dtype).uniform_(-1, 0)
        beta = torch.rand(B, T, H, device=device, dtype=dtype)

        gdn_timings = profile_chunk_gdn_parallel(
            q, k, v, g, beta, chunk_size=CS, warmup=5, rep=20,
        )

        gdn_s1 = sum(gdn_timings[1]) / len(gdn_timings[1])
        gdn_s2 = sum(gdn_timings[2]) / len(gdn_timings[2])
        gdn_s3 = sum(gdn_timings[3]) / len(gdn_timings[3])
        gdn_total_fwd = gdn_s1 + gdn_s2 + gdn_s3

        print()
        print("  chunk_gdn_parallel  (forward only)")
        print(f"    Stage 1  WY/UT intra-chunk    : {gdn_s1:8.3f} ms  ({gdn_s1/gdn_total_fwd*100:5.1f}%)")
        print(f"    Stage 2  parallel scan (inter) : {gdn_s2:8.3f} ms  ({gdn_s2/gdn_total_fwd*100:5.1f}%)")
        print(f"    Stage 3  output pass2          : {gdn_s3:8.3f} ms  ({gdn_s3/gdn_total_fwd*100:5.1f}%)")
        print(f"    Total forward                  : {gdn_total_fwd:8.3f} ms")

        # --- Mamba2 inputs ---
        x = torch.randn(B, T, H, K, device=device, dtype=dtype)
        a = torch.empty(B, T, H, device=device, dtype=dtype).uniform_(-1, 0)
        b_mat = torch.randn(B, T, H, N, device=device, dtype=dtype) * 0.1
        c_mat = torch.randn(B, T, H, N, device=device, dtype=dtype) * 0.1

        m2_timings = profile_mamba2_parallel(
            x, a, b_mat, c_mat, chunk_size=CS, warmup=5, rep=20,
        )

        m2_s1 = sum(m2_timings[1]) / len(m2_timings[1])
        m2_s2 = sum(m2_timings[2]) / len(m2_timings[2])
        m2_s3 = sum(m2_timings[3]) / len(m2_timings[3])
        m2_total_fwd = m2_s1 + m2_s2 + m2_s3

        print()
        print("  mamba2_parallel  (forward only)")
        print(f"    Stage 1  intra-chunk (diag)    : {m2_s1:8.3f} ms  ({m2_s1/m2_total_fwd*100:5.1f}%)")
        print(f"    Stage 2  inter-chunk prop      : {m2_s2:8.3f} ms  ({m2_s2/m2_total_fwd*100:5.1f}%)")
        print(f"    Stage 3  state readout (off)   : {m2_s3:8.3f} ms  ({m2_s3/m2_total_fwd*100:5.1f}%)")
        print(f"    Total forward                  : {m2_total_fwd:8.3f} ms")

        # --- End-to-end fwd+bwd ---
        q.requires_grad_(True); k.requires_grad_(True)
        v.requires_grad_(True); g.requires_grad_(True)
        beta.requires_grad_(True)
        x.requires_grad_(True); a.requires_grad_(True)
        b_mat.requires_grad_(True); c_mat.requires_grad_(True)

        gdn_e2e = benchmark_e2e(chunk_gdn_parallel, q, k, v, g, beta, chunk_size=CS)
        m2_e2e = benchmark_e2e(mamba2_parallel, x, a, b_mat, c_mat, chunk_size=CS)

        gdn_avg = sum(gdn_e2e) / len(gdn_e2e)
        m2_avg = sum(m2_e2e) / len(m2_e2e)

        print()
        print("  End-to-end (fwd + bwd)")
        print(f"    chunk_gdn_parallel : {gdn_avg:8.3f} ms")
        print(f"    mamba2_parallel    : {m2_avg:8.3f} ms")
        print(f"    ratio (GDN / M2)   : {gdn_avg/m2_avg:8.2f}x")

    print()
    print("Done.")
