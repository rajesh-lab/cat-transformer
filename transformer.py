"""
A vanilla transformer implementation. 
Nothing fancy here :)

Parts of it were taken from:
1. https://github.com/pytorch-labs/gpt-fast
2. https://github.com/Lightning-AI/litgpt

"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from liger_kernel.transformers import LigerRMSNorm, liger_rotary_pos_emb, LigerFusedLinearCrossEntropyLoss
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


from torch.nn.attention.flex_attention import flex_attention, BlockMask, _mask_mod_signature
_flex_attention_compiled = torch.compile(flex_attention, dynamic=False, mode="default")

@torch.compiler.disable(recursive=False)
def flex_attention_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: Optional[BlockMask] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    return _flex_attention_compiled(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)


# used during generation to shift the mask
def get_mask_mod(mask_mod: _mask_mod_signature, offset: int):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)
    return _mask_mod

@dataclass
class TransformerConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None

    n_layer: int = 6
    n_head: int = 12
    dim: int = 768
    intermediate_size: Optional[int] = None
    n_local_heads: int = -1
    head_dim: Optional[int] = None

    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_n_elem: Optional[int] = None

    # optional
    use_fused_ops: bool = False
    use_qk_norm: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)

        assert self.dim % self.n_head == 0
        self.head_dim = self.dim // self.n_head

        self.padded_vocab_size = find_multiple(self.vocab_size, 256)
        self.rope_n_elem = self.head_dim


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None: 
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, layer_idx=i) for i in range(config.n_layer))
        if self.config.use_fused_ops:
            self.norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

        if self.config.use_fused_ops:
            self.fused_linear_cross_entropy = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
        
        # initialize weights
        self.apply(lambda m: _init_weights(m, self.config.n_layer, self.config.dim))
        self.get_mask_mod = get_mask_mod

    def setup_cache(self, device=None):
        # force in fp32
        # this happens after the model has been created and move to respective device

        cos, sin = build_rope_cache(
            self.config.block_size, self.config.rope_n_elem, device=device, base=self.config.rope_base
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        print("created cos and sin cache ...")
        print("cos shape:", self.cos.shape)
        print("cos dtype:", self.cos.dtype)

    # used for generation
    def setup_kv_cache(self, max_batch_size: int, dtype, device: torch.device):
        print("Setting up kv cache ...")
        for block in self.layers:
            block.attention.kv_cache = KVCache(
                max_batch_size, self.config.block_size, self.config.n_local_heads, self.config.head_dim, dtype, device
            )

    def forward(
        self,
        input_ids: torch.LongTensor, 
        labels: Optional[torch.LongTensor] = None, 
        input_pos: Optional[Tensor] = None, 
        mask: Optional[BlockMask] = None
    ) -> Tensor:
        bsz, seqlen = input_ids.shape

        if (mask is not None) and (input_pos is not None):
            # doing generation
            mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])

        if input_pos is not None:
            cos = self.cos[:, input_pos]
            sin = self.sin[:, input_pos]
        else:
            # trim cos, sin
            cos = self.cos[:, :seqlen]
            sin = self.sin[:, :seqlen]

        x = self.wte(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, mask=mask, input_pos=input_pos)
        x = self.norm(x)

        if labels is not None:
            if self.config.use_fused_ops:
                loss = self.fused_linear_cross_entropy(
                    self.output.weight, x.view(-1, x.size(-1)), labels.view(-1)
                ) # need to reshape to x to (B*N, D) and labels to (B*N)
                return loss
            else:
                logits = self.output(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                return loss
        
        logits = self.output(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config

        self.attention = Attention(config, layer_idx)

        if config.use_fused_ops:
            self.feed_forward = LigerSwiGLUMLP(config)
        else:
            self.feed_forward = LLaMAMLP(config)

        if self.config.use_fused_ops:
            self.ffn_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
            self.attention_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
            self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, is_causal: Optional[bool] = True, mask: Optional[BlockMask] = None, input_pos: Optional[Tensor] = None) -> Tensor:
        h = x + self.attention(self.attention_norm(x), cos, sin, is_causal, mask=mask, input_pos=input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        # key, query and value projections for all heads, but in a batch
        self.wqkv = nn.Linear(
            config.dim,
            (config.n_head + 2 * config.n_local_heads) * config.head_dim,  # support for grouped/multi queries
            bias=False,
        )
        # output projection
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)

        self.config = config
        self.layer_idx = layer_idx

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.rope_n_elem = config.rope_n_elem

        self.kv_cache: Optional[KVCache] = None

        if self.config.use_qk_norm:
            if self.config.use_fused_ops:
                self.q_norm = LigerRMSNorm(config.head_dim, eps=config.norm_eps)
                self.k_norm = LigerRMSNorm(config.head_dim, eps=config.norm_eps)
            else:
                self.q_norm = RMSNorm(config.head_dim, eps=config.norm_eps)
                self.k_norm = RMSNorm(config.head_dim, eps=config.norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, is_causal: Optional[bool] = True, mask: Optional[BlockMask] = None, input_pos: Optional[Tensor] = None) -> Tensor:
        
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.config.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.config.use_fused_ops:
            q, k = liger_rotary_pos_emb(q, k, cos, sin)
        else:
            q = apply_rope_emb(q, cos, sin, self.rope_n_elem) # (B, n_head, N, head_dim)
            k = apply_rope_emb(k, cos, sin, self.rope_n_elem) # (B, n_local_heads, N, head_dim)

        if self.kv_cache is not None and input_pos is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        if mask is None:
            scale = 1.0 / math.sqrt(self.head_dim)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0,
                scale=scale, is_causal=is_causal, enable_gqa=(self.n_head != self.n_local_heads)
            ) # (B, n_head, N, head_dim)
        else:
            if input_pos is not None:
                # used during generation only!
                y = flex_attention(q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads))
            else:
                y = flex_attention_compiled(q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads))

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim) # (B, N, D)

        y = self.wo(y) # (B, N, D)

        # Output projection
        return y

# https://github.com/Lightning-AI/litgpt/blob/048633a7d08f75280e1f02fcd0ba58a7e47dfeb1/litgpt/model.py#L531
class LLaMAMLP(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.fc_2 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.proj = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class LigerSwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc_1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.fc_2 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.proj = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x):
        return self.proj(LigerSiLUMulFunction.apply(self.fc_1(x), self.fc_2(x)))


# https://github.com/Lightning-AI/litgpt/blob/048633a7d08f75280e1f02fcd0ba58a7e47dfeb1/litgpt/model.py#L812
class RMSNorm(torch.nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    # force float32
    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

@torch.amp.autocast("cuda", enabled=False)
def apply_rope_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_n_elem: int) -> Tensor:
    ### this does the following:
    # q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
    # q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)  # (B, nh_q, T, hs)
    
    x_roped = apply_rope(x[..., :rope_n_elem], cos, sin)
    x = torch.cat((x_roped, x[..., rope_n_elem:]), dim=-1)  # (B, nh_q, T, hs)
    return x

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.float16, device=None):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        k_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        v_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.register_buffer('k_cache', k_cache)
        self.register_buffer('v_cache', v_cache)

    def update(self, input_pos, k_val, v_val):
        # input_pos: [1], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2] # this checks that input_pos is equal to S

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

# https://arxiv.org/abs/2204.06745 : Section 2.1.3
# taken from: https://github.com/jzhang38/TinyLlama/blob/bf122247c486b6b897050e98cbb7bedae8eeba73/lit_gpt/model.py#L40
def _init_weights(module: nn.Module, n_layer: int, dim: int) -> None:
    """Meant to be used with `gpt.apply(gpt._init_weights)`."""
    # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
    if isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / dim))
        # RWKV: set it to 1e-4
        # torch.nn.init.uniform_(module.weight,  -1e-4, 1e-4)
    elif isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / dim))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    # GPT-NeoX
    for name, p in module.named_parameters():
        if (name == "wo.weight" and isinstance(module, Attention)) or (name == "proj.weight" and (isinstance(module, LLaMAMLP) or isinstance(module, LigerSwiGLUMLP))):
            nn.init.normal_(p, mean=0.0, std=(1 / math.sqrt(dim) / n_layer))


################################################################################################################
# RoPE related code below

# this file has been directly copied from: https://github.com/Lightning-AI/litgpt/blob/048633a7d08f75280e1f02fcd0ba58a7e47dfeb1/litgpt/model.py

# https://github.com/Lightning-AI/litgpt/blob/048633a7d08f75280e1f02fcd0ba58a7e47dfeb1/litgpt/model.py#L724
def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: Optional[dict] = None,
    rope_local_base_freq: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced Transformer with Rotary Position Embedding.

    Args:
        seq_len (int): Sequence length.
        n_elem (int): Number of elements (head dimension).
        device (torch.device, optional): Device for tensor allocations.
        base (int, optional): Base for computing inverse frequencies.
        condense_ratio (int, optional): Ratio to condense the position indices.
        extra_config (dict, optional): Configuration parameters for frequency adjustments (used by Llama 3.1 and 3.2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine caches for RoPE.
            Shapes are `(seq_len, n_elem)`.
    """

    # Compute the inverse frequencies theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    if extra_config is not None:
        factor = extra_config["factor"]
        if "original_max_seq_len" in extra_config:
            orig_context_len = extra_config["original_max_seq_len"]
            low_freq_factor = extra_config["low_freq_factor"]
            high_freq_factor = extra_config["high_freq_factor"]

            wavelen = 2 * torch.pi / theta
            ratio = orig_context_len / wavelen
            smooth_factor = (ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)

            # Compute adjusted_theta without masked indexing
            adjusted_theta = (1 - smooth_factor) * (theta / factor) + smooth_factor * theta
            theta = adjusted_theta
        else:
            theta = theta / factor

    # Create position indices `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    # If `n_elem` is odd, the final dimension of `idx_theta` has size
    # `n_elem + 1`, so need to cut something off.
    # Due to a current bug in Hugging Face, in the case `n_elem == 1`, we leave
    # `idx_theta`, `cos`, `sin` as is. Things work out in `apply_rope` due to
    # broadcasting. If we shorten `idx_theta`, unit tests comparing to
    # Hugging Face fail.
    # https://github.com/huggingface/transformers/issues/35233
    if idx_theta.shape[-1] > n_elem > 1:
        idx_theta = idx_theta[..., :n_elem]

    # if rope_local_base_freq is given, have a separate rope value for local embedding
    # For now, we use default RoPE for local embedding
    if rope_local_base_freq is not None:
        local_theta = 1.0 / (rope_local_base_freq ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
        local_idx_theta = torch.outer(seq_idx, local_theta)
        local_idx_theta = local_idx_theta.repeat(1, 2)
        if local_idx_theta.shape[-1] > n_elem > 1:
            local_idx_theta = local_idx_theta[..., :n_elem]

        idx_theta = torch.stack((idx_theta, local_idx_theta), dim=-1)

    # return torch.cos(idx_theta), torch.sin(idx_theta)
    return torch.cos(idx_theta).unsqueeze(0), torch.sin(idx_theta).unsqueeze(0) # NOTE: only this is different!!

# just to be sure, force float32 here
@torch.amp.autocast("cuda", enabled=False)
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies RoPE transform to `x`. Note that `cos`, `sin` need to have a batch
    dimension.

    Args:
        x: Input tensor, `(B, ..., T, head_size)`
        cos: Cached cosines, `(B, T, head_size)` or `(1, T, head_size)`
        sin: Cached sines, `(B, T, head_size)` or `(1, T, head_size)`

    Returns:
        Encoded tensor, `(B, ..., T, head_size)`
    """
    if cos.dim() != 3:
        raise ValueError(f"cos must be three-dimensional, but shape is {cos.shape}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos, sin must have same shape, but cos.shape={cos.shape}, sin.shape={sin.shape}")
    head_size_half = x.size(-1) // 2
    x1 = x[..., :head_size_half]  # (B, ..., T, head_size/2)
    x2 = x[..., head_size_half:]  # (B, ..., T, head_size/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, ..., T, head_size)
    dims_diff = x.dim() - cos.dim()
    if dims_diff > 0:
        # Ensure that shapes of `x`, `cos`, `sin` align
        new_shape = cos.shape[0:1] + (1,) * dims_diff + cos.shape[1:]
        cos = cos.view(*new_shape)
        sin = sin.view(*new_shape)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


if __name__ == "__main__":
    # test the model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TransformerConfig(
        block_size=2048,
        n_layer=12,

        n_head=12,
        dim=768,
        # use_fused_ops=True,
    )
    model = Transformer(config)
    model.to(device)
    model.setup_cache(device=device) # setup RoPE cache

    input_ids = torch.randint(0, config.vocab_size, (1, config.block_size), device=device)
    print("input_ids.shape:", input_ids.shape, input_ids.dtype)

    logits = model(input_ids)
    print("logits.shape:", logits.shape, logits.dtype)