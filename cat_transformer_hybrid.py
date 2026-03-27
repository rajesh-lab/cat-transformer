"""
Hybrid CAT decoder mixing standard softmax attention and CAT-masked linear
attention layers.

Standard attention layers use flex_attention with get_cat_mask.
Linear attention layers use the two-pass chunked algorithm from
gated_deltanet.py (chunk_linear_attn_cat) with elu+1 feature maps.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn
from torch import Tensor, einsum
from torch.nn import functional as F
import einops
from einops import rearrange

from transformer import (
    TransformerConfig,
    TransformerBlock,
    Attention,
    RMSNorm,
    LLaMAMLP,
    LigerSwiGLUMLP,
    _init_weights,
    KVCache,
    get_mask_mod,
    build_rope_cache,
)
from cat_transformer import CAT_Config, Compressor, get_cat_mask
from gated_deltanet import naive_recurrent_gdn_cat, chunk_gdn_cat, chunk_gdn_cat_parallel
from mamba2_ssd import naive_mamba2_cat, mamba2_parallel_cat

from torch.nn.attention.flex_attention import create_block_mask, BlockMask
create_block_mask = torch.compile(create_block_mask)

torch._dynamo.config.cache_size_limit = 8

from liger_kernel.transformers import LigerRMSNorm, LigerFusedLinearCrossEntropyLoss
from liger_kernel.ops.swiglu import LigerSiLUMulFunction


# --------------------------------------------------------------------------- #
# CATLinearAttention — drop-in replacement for Attention
# --------------------------------------------------------------------------- #

class CATLinearAttention(nn.Module):
    """Linear attention that respects the CAT block-sparse mask.

    Uses elu+1 feature maps and the two-pass chunked algorithm where
    inter-block state only accumulates from position-0 (fx) tokens.
    RoPE is intentionally skipped — positional information flows through
    the standard-attention layers in the hybrid stack.
    """

    def __init__(self, config: TransformerConfig, layer_idx: int, cat_block_size: int,
                 use_naive: bool = False) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.cat_block_size = cat_block_size
        self.use_naive = use_naive

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

        self.wqkv = nn.Linear(
            config.dim,
            (config.n_head + 2 * config.n_local_heads) * config.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)

        self.o_norm = RMSNorm(config.head_dim, eps=config.norm_eps)

        self.kv_cache: Optional[KVCache] = None

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor,
        is_causal: bool = True, mask=None, input_pos=None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = F.elu(q, alpha=1.0) + 1.0
        k = F.elu(k, alpha=1.0) + 1.0

        C = self.cat_block_size
        pad_len = (C - seqlen % C) % C
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))

        if self.use_naive:
            y = self._naive_cat_attn(q, k, v)
        else:
            y = self._chunk_linear_attn_cat(q, k, v)

        if pad_len > 0:
            y = y[:, :seqlen, :, :]

        y = self.o_norm(y)
        y = y.contiguous().view(bsz, seqlen, self.dim)
        return self.wo(y)

    def _build_cat_mask(self, T: int, device: torch.device) -> Tensor:
        """Explicit CAT mask of shape (T, T) — same semantics as get_cat_mask."""
        C = self.cat_block_size
        idx = torch.arange(T, device=device)
        q_idx = idx.unsqueeze(1)
        kv_idx = idx.unsqueeze(0)
        within_block = (q_idx // C) == (kv_idx // C)
        divides_block = (kv_idx % C) == 0
        causal = q_idx >= kv_idx
        return ((within_block | divides_block) & causal)  # (T, T) bool

    def _naive_cat_attn(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Quadratic baseline: materialise full (Q K^T) * CAT_mask @ V."""
        scale = q.shape[-1] ** -0.5
        B, T, H, _ = q.shape

        q = q.transpose(1, 2) * scale   # (B, H, T, K)
        k = k.transpose(1, 2)           # (B, H, T, K)
        v = v.transpose(1, 2)           # (B, H, T, V)

        attn = q @ k.transpose(-2, -1)  # (B, H, T, T)
        mask = self._build_cat_mask(T, q.device)  # (T, T)
        attn = attn.masked_fill(~mask, 0.0)

        o = attn @ v                     # (B, H, T, V)
        return o.transpose(1, 2)         # (B, T, H, V)

    def _chunk_linear_attn_cat(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Two-pass chunked linear attention under the CAT mask."""
        scale = q.shape[-1] ** -0.5
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = self.cat_block_size
        NC = T // C

        q_c = rearrange(q, 'b (nc c) h k -> b nc c h k', c=C) * scale
        k_c = rearrange(k, 'b (nc c) h k -> b nc c h k', c=C)
        v_c = rearrange(v, 'b (nc c) h v -> b nc c h v', c=C)

        # Pass 1: inter-chunk state from first token of each block only
        k_first = k_c[:, :, 0, :, :]
        v_first = v_c[:, :, 0, :, :]
        kv_first = einsum('b n h k, b n h v -> b n h k v', k_first, v_first)
        h = kv_first.cumsum(dim=1)
        h = torch.cat([q.new_zeros(B, 1, H, K, V), h[:, :-1]], dim=1)

        # Pass 2: per-chunk (parallelizable)
        inter = einsum('b n c h k, b n h k v -> b n c h v', q_c, h)

        attn = einsum('b n i h k, b n j h k -> b n h i j', q_c, k_c)
        causal_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool))
        attn = attn.masked_fill(~causal_mask, 0.0)
        intra = einsum('b n h i j, b n j h v -> b n i h v', attn, v_c)

        o = inter + intra
        return rearrange(o, 'b nc c h v -> b (nc c) h v')


# --------------------------------------------------------------------------- #
# CATLinearBlock — TransformerBlock with linear attention
# --------------------------------------------------------------------------- #

class CATLinearBlock(nn.Module):
    """TransformerBlock variant that uses CATLinearAttention + MLP."""

    def __init__(self, config: TransformerConfig, layer_idx: int, cat_block_size: int,
                 use_naive: bool = False) -> None:
        super().__init__()
        self.config = config
        self.attention = CATLinearAttention(config, layer_idx, cat_block_size, use_naive=use_naive)

        if config.use_fused_ops:
            self.feed_forward = LigerSwiGLUMLP(config)
            self.ffn_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
            self.attention_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.feed_forward = LLaMAMLP(config)
            self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
            self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                is_causal: bool = True, mask=None, input_pos=None) -> Tensor:
        h = x + self.attention(self.attention_norm(x), cos, sin, is_causal, mask=mask, input_pos=input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# --------------------------------------------------------------------------- #
# CATGatedDeltaNet — GDN-based drop-in replacement for Attention
# --------------------------------------------------------------------------- #

class CATGatedDeltaNet(nn.Module):
    """Gated Delta Network attention with the CAT block-sparse mask.

    Uses L2-normalized Q/K, Mamba-style gated decay, and the two-pass
    chunked GDN algorithm from gated_deltanet.py.  Three modes:
      'naive'    — token-by-token  (naive_recurrent_gdn_cat)
      'chunk'    — two-pass, sequential intra-chunk  (chunk_gdn_cat)
      'parallel' — two-pass, WY-parallel intra-chunk (chunk_gdn_cat_parallel)

    Following fla & NVlabs references, includes causal depthwise short
    convolutions on Q/K/V and SiLU output gating.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int,
        cat_block_size: int,
        mode: str = 'parallel',
        use_short_conv: bool = True,
        conv_size: int = 4,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.cat_block_size = cat_block_size
        self.mode = mode
        assert mode in ('naive', 'chunk', 'parallel')

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.use_short_conv = use_short_conv

        key_dim = config.n_head * config.head_dim
        kv_dim = config.n_local_heads * config.head_dim

        self.q_proj = nn.Linear(config.dim, key_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, kv_dim, bias=False)

        if use_short_conv:
            self.q_conv = nn.Conv1d(
                key_dim, key_dim, conv_size,
                padding=conv_size - 1, groups=key_dim, bias=False,
            )
            self.k_conv = nn.Conv1d(
                kv_dim, kv_dim, conv_size,
                padding=conv_size - 1, groups=kv_dim, bias=False,
            )
            self.v_conv = nn.Conv1d(
                kv_dim, kv_dim, conv_size,
                padding=conv_size - 1, groups=kv_dim, bias=False,
            )

        # Mamba-style gated decay: g = -A_log.exp() * softplus(a_proj(x) + dt_bias)
        self.a_proj = nn.Linear(config.dim, config.n_head, bias=False)
        A = torch.empty(config.n_head, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(config.n_head) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        self.b_proj = nn.Linear(config.dim, config.n_head, bias=True)

        # Output: gate + RMSNorm + projection
        self.g_proj = nn.Linear(config.dim, key_dim, bias=False)
        self.o_norm = RMSNorm(config.head_dim, eps=config.norm_eps)
        self.wo = nn.Linear(key_dim, config.dim, bias=False)

        self.kv_cache: Optional[KVCache] = None

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor,
        is_causal: bool = True, mask=None, input_pos=None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_short_conv:
            q = F.silu(self.q_conv(q.transpose(1, 2))[..., :seqlen].transpose(1, 2))
            k = F.silu(self.k_conv(k.transpose(1, 2))[..., :seqlen].transpose(1, 2))
            v = F.silu(self.v_conv(v.transpose(1, 2))[..., :seqlen].transpose(1, 2))

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # L2 normalize Q, K
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # GQA: repeat K/V heads to match Q
        if self.n_local_heads != self.n_head:
            n_rep = self.n_head // self.n_local_heads
            k = k.unsqueeze(3).expand(-1, -1, -1, n_rep, -1)
            k = k.reshape(bsz, seqlen, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, n_rep, -1)
            v = v.reshape(bsz, seqlen, self.n_head, self.head_dim)

        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(x).float() + self.dt_bias
        )
        g = g.to(x.dtype)

        beta = self.b_proj(x).sigmoid()

        C = self.cat_block_size
        pad_len = (C - seqlen % C) % C
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
            g = F.pad(g, (0, 0, 0, pad_len))
            beta = F.pad(beta, (0, 0, 0, pad_len))

        if self.mode == 'naive':
            y = naive_recurrent_gdn_cat(q, k, v, g, beta, chunk_size=C)
        elif self.mode == 'chunk':
            y = chunk_gdn_cat(q, k, v, g, beta, chunk_size=C)
        else:
            y = chunk_gdn_cat_parallel(q, k, v, g, beta, chunk_size=C)

        if pad_len > 0:
            y = y[:, :seqlen, :, :]

        g_out = self.g_proj(x).view(bsz, seqlen, self.n_head, self.head_dim)
        y = self.o_norm(y) * F.silu(g_out)

        y = y.contiguous().view(bsz, seqlen, self.dim)
        return self.wo(y)


# --------------------------------------------------------------------------- #
# CATGatedDeltaNetBlock — TransformerBlock with GDN attention
# --------------------------------------------------------------------------- #

class CATGatedDeltaNetBlock(nn.Module):
    """TransformerBlock variant that uses CATGatedDeltaNet + MLP."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int,
        cat_block_size: int,
        mode: str = 'parallel',
        use_short_conv: bool = True,
        conv_size: int = 4,
    ) -> None:
        super().__init__()
        self.config = config
        self.attention = CATGatedDeltaNet(
            config, layer_idx, cat_block_size,
            mode=mode, use_short_conv=use_short_conv, conv_size=conv_size,
        )

        if config.use_fused_ops:
            self.feed_forward = LigerSwiGLUMLP(config)
            self.ffn_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
            self.attention_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.feed_forward = LLaMAMLP(config)
            self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
            self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                is_causal: bool = True, mask=None, input_pos=None) -> Tensor:
        h = x + self.attention(self.attention_norm(x), cos, sin, is_causal, mask=mask, input_pos=input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# --------------------------------------------------------------------------- #
# CATMamba2 — Mamba2/SSD-based drop-in replacement for Attention
# --------------------------------------------------------------------------- #

class CATMamba2(nn.Module):
    """Mamba2/SSD with the CAT block-sparse mask.

    Uses pre-discretized SSM inputs and the chunked parallel SSD algorithm
    from mamba2_ssd.py.  Two modes:
      'naive'    — token-by-token  (naive_mamba2_cat)
      'parallel' — chunked parallel (mamba2_parallel_cat)

    Following the fla Mamba2 layer reference:
    - Single in_proj producing (gate, x_BC, dt)
    - Depthwise causal convolution on (x, B, C) with SiLU activation
    - A / dt discretization
    - D skip connection
    - Per-head RMSNorm + SiLU-gated output
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int,
        cat_block_size: int,
        mode: str = 'parallel',
        state_size: int = 64,
        n_groups: int = 1,
        conv_kernel: int = 4,
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.cat_block_size = cat_block_size
        self.mode = mode
        assert mode in ('naive', 'parallel')

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.state_size = state_size
        self.n_groups = n_groups
        self.intermediate_size = config.n_head * config.head_dim
        self.use_conv = use_conv

        self.conv_dim = self.intermediate_size + 2 * n_groups * state_size
        projection_size = self.intermediate_size + self.conv_dim + config.n_head
        self.in_proj = nn.Linear(config.dim, projection_size, bias=False)

        if use_conv:
            self.conv1d = nn.Conv1d(
                self.conv_dim, self.conv_dim, conv_kernel,
                padding=conv_kernel - 1, groups=self.conv_dim, bias=False,
            )

        A = torch.empty(config.n_head, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(config.n_head) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.n_head))
        self.D._no_weight_decay = True

        self.o_norm = RMSNorm(config.head_dim, eps=config.norm_eps)
        self.wo = nn.Linear(self.intermediate_size, config.dim, bias=False)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor,
        is_causal: bool = True, mask=None, input_pos=None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        projected = self.in_proj(x)
        gate, x_BC, dt = projected.split(
            [self.intermediate_size, self.conv_dim, self.n_head], dim=-1,
        )

        if self.use_conv:
            x_BC = F.silu(
                self.conv1d(x_BC.transpose(1, 2))[..., :seqlen].transpose(1, 2)
            )

        groups_state_size = self.n_groups * self.state_size
        x_ssm, B, C = x_BC.split(
            [self.intermediate_size, groups_state_size, groups_state_size], dim=-1,
        )

        x_ssm = x_ssm.view(bsz, seqlen, self.n_head, self.head_dim)
        B = B.view(bsz, seqlen, self.n_groups, self.state_size)
        C = C.view(bsz, seqlen, self.n_groups, self.state_size)

        if self.n_groups < self.n_head:
            n_rep = self.n_head // self.n_groups
            B = B.unsqueeze(3).expand(-1, -1, -1, n_rep, -1)
            B = B.reshape(bsz, seqlen, self.n_head, self.state_size)
            C = C.unsqueeze(3).expand(-1, -1, -1, n_rep, -1)
            C = C.reshape(bsz, seqlen, self.n_head, self.state_size)

        A = -torch.exp(self.A_log.float())
        dt_val = F.softplus(dt.float() + self.dt_bias).to(x.dtype)

        a = A.to(x.dtype) * dt_val                       # (B, T, H) log-decay
        x_disc = x_ssm * dt_val.unsqueeze(-1)             # (B, T, H, D)
        D_skip = self.D[None, None, :, None] * x_ssm      # (B, T, H, D)

        C_bs = self.cat_block_size
        pad_len = (C_bs - seqlen % C_bs) % C_bs
        if pad_len > 0:
            x_disc = F.pad(x_disc, (0, 0, 0, 0, 0, pad_len))
            a = F.pad(a, (0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len))

        if self.mode == 'naive':
            y = naive_mamba2_cat(x_disc, a, B, C, chunk_size=C_bs)
        else:
            y = mamba2_parallel_cat(x_disc, a, B, C, chunk_size=C_bs)

        if pad_len > 0:
            y = y[:, :seqlen, :, :]

        y = y + D_skip

        gate = F.silu(gate.view(bsz, seqlen, self.n_head, self.head_dim))
        y = self.o_norm(y) * gate

        y = y.contiguous().view(bsz, seqlen, self.intermediate_size)
        return self.wo(y)


# --------------------------------------------------------------------------- #
# CATMamba2Block — TransformerBlock with Mamba2/SSD attention
# --------------------------------------------------------------------------- #

class CATMamba2Block(nn.Module):
    """TransformerBlock variant that uses CATMamba2 + MLP."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int,
        cat_block_size: int,
        mode: str = 'parallel',
        state_size: int = 64,
        n_groups: int = 1,
        conv_kernel: int = 4,
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.attention = CATMamba2(
            config, layer_idx, cat_block_size,
            mode=mode, state_size=state_size, n_groups=n_groups,
            conv_kernel=conv_kernel, use_conv=use_conv,
        )

        if config.use_fused_ops:
            self.feed_forward = LigerSwiGLUMLP(config)
            self.ffn_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
            self.attention_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.feed_forward = LLaMAMLP(config)
            self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
            self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                is_causal: bool = True, mask=None, input_pos=None) -> Tensor:
        h = x + self.attention(self.attention_norm(x), cos, sin, is_causal, mask=mask, input_pos=input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class HybridCAT_Config(CAT_Config):
    linear_attn_layers: List[int] = field(default_factory=list)
    gdn_layers: List[int] = field(default_factory=list)
    mamba2_layers: List[int] = field(default_factory=list)
    use_naive_linear_attn: bool = False
    gdn_mode: str = 'parallel'
    gdn_use_short_conv: bool = True
    gdn_conv_size: int = 4
    mamba2_mode: str = 'parallel'
    mamba2_state_size: int = 64
    mamba2_n_groups: int = 1
    mamba2_conv_kernel: int = 4
    mamba2_use_conv: bool = True


# --------------------------------------------------------------------------- #
# Hybrid CAT Transformer
# --------------------------------------------------------------------------- #

class CAT_Transformer_Hybrid(nn.Module):
    """CAT decoder with a configurable mix of softmax and linear attention layers.

    Layers whose indices appear in config.linear_attn_layers use
    CATLinearBlock (CAT-masked chunked linear attention).  All other
    layers use the standard TransformerBlock (flex_attention with
    get_cat_mask).
    """

    def __init__(self, config: HybridCAT_Config, f_config: CAT_Config) -> None:
        super().__init__()
        self.config = config

        self.num_chunks = config.num_chunks
        self.chunk_size = config.chunk_size
        self.block_size = config.block_size

        self.f = Compressor(f_config)

        self.dummy_fx = nn.Embedding(1, config.dim)
        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)

        cat_block_size = 1 + config.chunk_size
        self.layers = nn.ModuleList()
        for i in range(config.n_layer):
            if i in config.gdn_layers:
                self.layers.append(CATGatedDeltaNetBlock(
                    config, layer_idx=i, cat_block_size=cat_block_size,
                    mode=config.gdn_mode,
                    use_short_conv=config.gdn_use_short_conv,
                    conv_size=config.gdn_conv_size,
                ))
            elif i in config.mamba2_layers:
                self.layers.append(CATMamba2Block(
                    config, layer_idx=i, cat_block_size=cat_block_size,
                    mode=config.mamba2_mode,
                    state_size=config.mamba2_state_size,
                    n_groups=config.mamba2_n_groups,
                    conv_kernel=config.mamba2_conv_kernel,
                    use_conv=config.mamba2_use_conv,
                ))
            elif i in config.linear_attn_layers:
                self.layers.append(CATLinearBlock(
                    config, layer_idx=i, cat_block_size=cat_block_size,
                    use_naive=config.use_naive_linear_attn,
                ))
            else:
                self.layers.append(TransformerBlock(config, layer_idx=i))

        self.has_standard_attn = any(
            isinstance(layer, TransformerBlock) for layer in self.layers
        )

        self.output = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

        if config.use_fused_ops:
            self.norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
            self.fused_linear_cross_entropy = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.down_proj = nn.Identity()
        assert self.f.config.dim_fx == self.config.dim, \
            "f.dim_fx must equal config.dim"

        self.apply(lambda m: _init_weights(m, self.config.n_layer, self.config.dim))
        self.f.apply(lambda m: _init_weights(m, self.f.config.n_layer, self.f.config.dim))
        for layer in self.layers:
            if isinstance(layer, (CATLinearBlock, CATGatedDeltaNetBlock, CATMamba2Block)):
                nn.init.normal_(
                    layer.attention.wo.weight,
                    mean=0.0, std=1.0 / math.sqrt(config.dim) / config.n_layer,
                )

        self.get_mask_mod = get_mask_mod

    def setup_cache(self, device: torch.device):
        self.f.setup_cache(device=device)

        cos, sin = build_rope_cache(
            1 + self.chunk_size, self.config.rope_n_elem,
            device=device, base=self.config.rope_base,
        )
        cos = einops.repeat(cos, '1 l d -> 1 k l d', k=self.num_chunks + 1).clone()
        sin = einops.repeat(sin, '1 l d -> 1 k l d', k=self.num_chunks + 1).clone()
        cos = einops.rearrange(cos, '1 k l d -> 1 (k l) d')
        sin = einops.rearrange(sin, '1 k l d -> 1 (k l) d')
        cos = cos[:, :self.block_size + self.num_chunks + 1, :].contiguous()
        sin = sin[:, :self.block_size + self.num_chunks + 1, :].contiguous()

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        _cos, _sin = build_rope_cache(
            self.block_size, self.config.rope_n_elem,
            device=device, base=self.config.rope_base,
        )
        self.register_buffer("cos_gen", _cos, persistent=False)
        self.register_buffer("sin_gen", _sin, persistent=False)

        n_std = sum(1 for l in self.layers if isinstance(l, TransformerBlock))
        n_lin = sum(1 for l in self.layers if isinstance(l, CATLinearBlock))
        n_gdn = sum(1 for l in self.layers if isinstance(l, CATGatedDeltaNetBlock))
        n_m2 = sum(1 for l in self.layers if isinstance(l, CATMamba2Block))
        print(f"Hybrid CAT: {n_std} standard + {n_lin} linear + {n_gdn} GDN + {n_m2} Mamba2 layers")
        print("cos shape:", self.cos.shape)

    # ---- generation helpers ----

    def setup_kv_cache(self, max_batch_size: int, dtype, device: torch.device):
        for block in self.layers:
            if isinstance(block, TransformerBlock):
                block.attention.kv_cache = KVCache(
                    max_batch_size,
                    self.config.num_chunks + self.config.chunk_size,
                    self.config.n_local_heads, self.config.head_dim, dtype, device,
                )

    def forward_embeddings(
        self, x: Tensor,
        cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None,
        input_pos: Optional[Tensor] = None, rope_pos: Optional[Tensor] = None,
        mask: Optional[BlockMask] = None, is_input_token: bool = False,
    ) -> Tensor:
        bsz, seqlen = x.shape[0:2]
        if mask is not None and input_pos is not None:
            mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])
        if is_input_token:
            x = self.wte(x)
        if cos is None and sin is None:
            cos, sin = self.cos_gen, self.sin_gen
        if input_pos is not None:
            cos_g = cos[:, rope_pos]
            sin_g = sin[:, rope_pos]
        else:
            cos_g = cos[:, :seqlen, :]
            sin_g = sin[:, :seqlen, :]
        for layer in self.layers:
            x = layer(x, cos_g, sin_g, input_pos=input_pos, mask=mask)
        x = self.norm(x)
        return self.output(x)

    # ---- training forward ----

    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None) -> Tensor:
        bsz, seqlen = input_ids.shape

        pad_multiple = 512
        slice_end = False
        if seqlen % pad_multiple != 0:
            new_seqlen = ((seqlen // pad_multiple) + 1) * pad_multiple
            pad_len = new_seqlen - seqlen
            input_ids = F.pad(input_ids, (0, pad_len), value=0)
            old_seqlen = seqlen
            seqlen = new_seqlen
            slice_end = True

        cur_num_chunks = seqlen // self.chunk_size
        input_ids = input_ids.view(bsz, cur_num_chunks, self.chunk_size)

        # compress
        fx = torch.vmap(self.f.compress, in_dims=(1, 0), out_dims=1)(
            input_ids,
            torch.arange(cur_num_chunks, device=input_ids.device),
        )
        fx = self.down_proj(fx)
        fx_last = fx[:, -1, :].unsqueeze(1)

        dummy_fx = self.dummy_fx(torch.zeros(1, device=input_ids.device, dtype=torch.long))
        dummy_fx = einops.repeat(dummy_fx, '1 d -> b 1 d', b=bsz)

        fx = torch.cat([dummy_fx, fx[:, :-1, :]], dim=1)
        fx = einops.rearrange(fx, 'b k d -> b k 1 d')

        emb_x = self.wte(input_ids)
        x = torch.cat([fx, emb_x], dim=2)
        x = einops.rearrange(x, 'b k l d -> b (k l) d')
        x = torch.cat([x, fx_last], dim=1)

        cos = self.cos[:, :x.shape[1], :]
        sin = self.sin[:, :x.shape[1], :]

        # flex_attention mask only needed when standard layers exist
        mask = None
        if self.has_standard_attn:
            mask = create_block_mask(
                get_cat_mask(1 + self.chunk_size),
                B=None, H=None,
                Q_LEN=x.shape[1], KV_LEN=x.shape[1],
            )

        for layer in self.layers:
            x = layer(x, cos=cos, sin=sin, mask=mask)
        x = self.norm(x)

        # rearrange to (B, L, D) for next-token prediction
        x_last = x[:, -1:, :].contiguous()
        x = einops.rearrange(
            x[:, :-1, :], 'b (k l) d -> b k l d',
            k=cur_num_chunks, l=self.chunk_size + 1,
        )
        x_first = x[:, :1, 1:-1, :].contiguous()
        x_middle = x[:, 1:, :-1, :].contiguous()
        x_first = einops.rearrange(x_first, 'b 1 l d -> b (1 l) d')
        x_middle = einops.rearrange(x_middle, 'b k l d -> b (k l) d')
        x = torch.cat([x_first, x_middle, x_last], dim=1)

        if slice_end:
            x = x[:, :old_seqlen, :].contiguous()

        if labels is not None:
            if self.config.use_fused_ops:
                loss = self.fused_linear_cross_entropy(
                    self.output.weight, x.view(-1, x.size(-1)), labels.view(-1),
                )
                return loss
            else:
                logits = self.output(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
                )
                return loss

        return self.output(x)


# --------------------------------------------------------------------------- #
# Quick test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dim = 768
    num_layers = 12
    n_head = 12

    decoder_dim = 2 * dim
    dim_fx = decoder_dim
    n_head_decoder = 2 * n_head

    block_size = 2048
    chunk_size = 8

    linear_layers = list(range(0, num_layers, 2))

    compressor_config = CAT_Config(
        dim=dim, n_head=n_head, dim_fx=dim_fx,
        block_size=block_size, chunk_size=chunk_size,
        n_layer=max(1, num_layers // 4),
    )

    # --- chunked linear attention ---
    decoder_config_chunked = HybridCAT_Config(
        dim=decoder_dim, n_head=n_head_decoder,
        block_size=block_size, chunk_size=chunk_size,
        n_layer=num_layers,
        linear_attn_layers=linear_layers,
        use_naive_linear_attn=False,
    )

    # --- naive (quadratic) linear attention ---
    decoder_config_naive = HybridCAT_Config(
        dim=decoder_dim, n_head=n_head_decoder,
        block_size=block_size, chunk_size=chunk_size,
        n_layer=num_layers,
        linear_attn_layers=linear_layers,
        use_naive_linear_attn=True,
    )

    print(f"Linear attention at layers: {linear_layers}")

    # build chunked model
    model_chunked = CAT_Transformer_Hybrid(decoder_config_chunked, compressor_config)
    model_chunked = model_chunked.to(device=device)
    model_chunked.setup_cache(device=device)

    # build naive model, copy weights from chunked so outputs are directly comparable
    model_naive = CAT_Transformer_Hybrid(decoder_config_naive, compressor_config)
    model_naive.load_state_dict(model_chunked.state_dict())
    model_naive = model_naive.to(device=device)
    model_naive.setup_cache(device=device)

    n_params = sum(p.numel() for p in model_chunked.parameters())
    print(f"Total params (linear hybrid): {n_params / 1e6:.1f}M")

    input_ids = torch.randint(0, decoder_config_chunked.vocab_size, (4, block_size), device=device)
    print("input_ids shape:", input_ids.shape)

    with torch.no_grad():
        logits_chunked = model_chunked(input_ids)
        logits_naive = model_naive(input_ids)

    print("logits_chunked shape:", logits_chunked.shape)
    print("logits_naive   shape:", logits_naive.shape)

    diff = (logits_chunked - logits_naive).abs()
    print(f"Max abs diff:  {diff.max().item():.6e}")
    print(f"Mean abs diff: {diff.mean().item():.6e}")
    if diff.max().item() < 1e-2:
        print("PASS: chunked and naive linear attention agree")
    else:
        print("WARN: chunked and naive linear attention differ significantly")

    # ------------------------------------------------------------------- #
    # GDN hybrid: parallel vs naive modes
    # ------------------------------------------------------------------- #
    # print()
    # print("=" * 70)
    # print("GDN hybrid correctness check (parallel vs naive mode)")
    # print("=" * 70)

    # gdn_layers = list(range(0, num_layers, 2))

    # decoder_config_gdn_par = HybridCAT_Config(
    #     dim=decoder_dim, n_head=n_head_decoder,
    #     block_size=block_size, chunk_size=chunk_size,
    #     n_layer=num_layers,
    #     gdn_layers=gdn_layers,
    #     gdn_mode='parallel',
    # )
    # decoder_config_gdn_naive = HybridCAT_Config(
    #     dim=decoder_dim, n_head=n_head_decoder,
    #     block_size=block_size, chunk_size=chunk_size,
    #     n_layer=num_layers,
    #     gdn_layers=gdn_layers,
    #     gdn_mode='naive',
    # )

    # model_gdn_par = CAT_Transformer_Hybrid(decoder_config_gdn_par, compressor_config)
    # model_gdn_par = model_gdn_par.to(device=device)
    # model_gdn_par.setup_cache(device=device)

    # model_gdn_naive = CAT_Transformer_Hybrid(decoder_config_gdn_naive, compressor_config)
    # model_gdn_naive.load_state_dict(model_gdn_par.state_dict())
    # model_gdn_naive = model_gdn_naive.to(device=device)
    # model_gdn_naive.setup_cache(device=device)

    # n_params_gdn = sum(p.numel() for p in model_gdn_par.parameters())
    # print(f"Total params (GDN hybrid): {n_params_gdn / 1e6:.1f}M")
    # print(f"GDN layers: {gdn_layers}")

    # with torch.no_grad():
    #     logits_gdn_par = model_gdn_par(input_ids)
    #     logits_gdn_naive = model_gdn_naive(input_ids)

    # diff_gdn = (logits_gdn_par - logits_gdn_naive).abs()
    # print(f"Max abs diff (parallel vs naive):  {diff_gdn.max().item():.6e}")
    # print(f"Mean abs diff (parallel vs naive): {diff_gdn.mean().item():.6e}")
    # if diff_gdn.max().item() < 1e-2:
    #     print("PASS: GDN parallel and naive agree")
    # else:
    #     print("WARN: GDN parallel and naive differ significantly")

    # ------------------------------------------------------------------- #
    # Mamba2 hybrid: parallel vs naive modes
    # ------------------------------------------------------------------- #
    print()
    print("=" * 70)
    print("Mamba2 hybrid correctness check (parallel vs naive mode)")
    print("=" * 70)

    mamba2_layers = list(range(0, num_layers, 2))

    decoder_config_m2_par = HybridCAT_Config(
        dim=decoder_dim, n_head=n_head_decoder,
        block_size=block_size, chunk_size=chunk_size,
        n_layer=num_layers,
        mamba2_layers=mamba2_layers,
        mamba2_mode='parallel',
    )
    decoder_config_m2_naive = HybridCAT_Config(
        dim=decoder_dim, n_head=n_head_decoder,
        block_size=block_size, chunk_size=chunk_size,
        n_layer=num_layers,
        mamba2_layers=mamba2_layers,
        mamba2_mode='naive',
    )

    model_m2_par = CAT_Transformer_Hybrid(decoder_config_m2_par, compressor_config)
    model_m2_par = model_m2_par.to(device=device)
    model_m2_par.setup_cache(device=device)

    model_m2_naive = CAT_Transformer_Hybrid(decoder_config_m2_naive, compressor_config)
    model_m2_naive.load_state_dict(model_m2_par.state_dict())
    model_m2_naive = model_m2_naive.to(device=device)
    model_m2_naive.setup_cache(device=device)

    n_params_m2 = sum(p.numel() for p in model_m2_par.parameters())
    print(f"Total params (Mamba2 hybrid): {n_params_m2 / 1e6:.1f}M")
    print(f"Mamba2 layers: {mamba2_layers}")

    with torch.no_grad():
        logits_m2_par = model_m2_par(input_ids)
        logits_m2_naive = model_m2_naive(input_ids)

    diff_m2 = (logits_m2_par - logits_m2_naive).abs()
    print(f"Max abs diff (parallel vs naive):  {diff_m2.max().item():.6e}")
    print(f"Mean abs diff (parallel vs naive): {diff_m2.mean().item():.6e}")
    if diff_m2.max().item() < 1e-2:
        print("PASS: Mamba2 parallel and naive agree")
    else:
        print("WARN: Mamba2 parallel and naive differ significantly")
