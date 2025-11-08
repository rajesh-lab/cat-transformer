import math
import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import einops

# import some transformer components
from transformer import (
    TransformerConfig,
    TransformerBlock,
    RMSNorm,
    _init_weights,
    KVCache,
    get_mask_mod,
    build_rope_cache,
    apply_rope_emb,
)
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask, _mask_mod_signature

_flex_attention_compiled = torch.compile(flex_attention, dynamic=False, mode="default")
create_block_mask = torch.compile(create_block_mask)

@torch.compiler.disable(recursive=False)
def flex_attention_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: Optional[BlockMask] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    return _flex_attention_compiled(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)

from liger_kernel.transformers import LigerRMSNorm, liger_rotary_pos_emb, LigerFusedLinearCrossEntropyLoss
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

from cat_transformer import (
    CAT_Config,
    get_cat_mask
)


class CAT_Attention(torch.nn.Module):

    def __init__(self, config: CAT_Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.chunk_size = config.chunk_size
        self.block_size = config.block_size
        self.num_chunks = self.block_size // self.chunk_size

        self.cat_dim = config.dim_fx
        self.expand = nn.Linear(self.dim, self.cat_dim, bias=False)

        # compressor
        self.compressor = nn.Linear(self.chunk_size * self.dim, self.cat_dim)
        self.dummy_fx = nn.Embedding(1, self.cat_dim)

        # decoder
        self.attention = Attention(
            dim=self.cat_dim,
            n_head=(self.cat_dim // 64),
            config=config,
            layer_idx=layer_idx
        )

        # final projection
        self.final_proj = nn.Linear(self.cat_dim, self.dim, bias=False)

        self.mask = create_block_mask(
            get_cat_mask(1 + self.chunk_size), 
            B=None, H=None, 
            Q_LEN=self.block_size + self.num_chunks + 1, # K+L
            KV_LEN=self.block_size + self.num_chunks + 1, # K+L
        )

    def setup_cache(self, device: torch.device):

        # first, cache below for doing generation
        _cos, _sin = build_rope_cache(
            self.block_size, # JP: declare more than necessary here, its okay.
            self.config.rope_n_elem, device=device, base=self.config.rope_base
        )
        self.register_buffer("cos_gen", _cos, persistent=False)
        self.register_buffer("sin_gen", _sin, persistent=False)

        cos, sin = build_rope_cache(1 + self.chunk_size, self.config.rope_n_elem, device=device, base=self.config.rope_base)
        # careful here!
        cos = einops.repeat(cos, '1 l d -> 1 k l d', k=self.num_chunks+1).clone()
        sin = einops.repeat(sin, '1 l d -> 1 k l d', k=self.num_chunks+1).clone()

        cos = einops.rearrange(cos, '1 k l d -> 1 (k l) d')
        sin = einops.rearrange(sin, '1 k l d -> 1 (k l) d')

        # careful!
        # trim off extra
        cos = cos[:, :self.block_size + self.num_chunks + 1, :].contiguous()
        sin = sin[:, :self.block_size + self.num_chunks + 1, :].contiguous()

        # cache these for training
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        print("created cos and sin cache for CAT layer ...")
        print("cos shape:", self.cos.shape)
        print("cos dtype:", self.cos.dtype)

    def forward(
        self, 
        x: Tensor, 
        cos: Tensor, # not used
        sin: Tensor, # not used
    ) -> Tensor:

        bsz, seq_len, dim = x.size()

        assert seq_len == self.block_size
        assert seq_len % self.chunk_size == 0
        cur_num_chunks = seq_len // self.chunk_size

        # compress
        fx = self.compressor(
            einops.rearrange(x, "b (k l) d -> b k (l d)", l=self.chunk_size) # (b, k, l*D)
        )  # (b, k, 2D)
        dummy = self.dummy_fx.weight.unsqueeze(0).expand(bsz, -1, -1)  # (b, 1, 2D)
        fx = torch.cat([dummy, fx], dim=1)  # (b, k+1, 2D)

        fx_last = fx[:, -1:, :]  # (b, 1, 2D)
        fx = fx[:, :-1, :]  # (b, k, 2D)

        x = self.expand(x) # (b, k*l, 2D)

        x = einops.rearrange(x, "b (k l) d -> b k l d", l=self.chunk_size)  # (b, k, l, 2D)

        x = torch.cat([fx.unsqueeze(2), x], dim=2)  # (b, k, 1+l, 2D)
        # flatten x
        x = einops.rearrange(x, "b k l d -> b (k l) d")  # (b, k*(1+l), 2D)
        # attach the last fx token at the end
        x = torch.cat([x, fx_last], dim=1)  # (b, k*(1+l)+1, 2D)

        # pass through attention now
        x = self.attention(x, cos=self.cos, sin=self.sin, mask=self.mask)  # (b, k, 1+l, 2D)

        # Arrange everything nicely back to (B, L, D) for next token prediction
        x_last = x[:, -1:, :].contiguous() # (B, 1, D)
        x = einops.rearrange(x[:, :-1, :], 'b (k l) d -> b k l d', k=cur_num_chunks, l=self.chunk_size+1) # (B, K, 1+l, D)
        x_first = x[:, :1, 1:-1, :].contiguous() # (B, 1, l-1, D)
        x_middle = x[:, 1:, :-1, :].contiguous() # (B, K-1, l, D)
        x_first = einops.rearrange(x_first, 'b 1 l d -> b (1 l) d') # (B, l-1, D)
        x_middle = einops.rearrange(x_middle, 'b k l d -> b (k l) d') # (B, (K-1)*l, D)
        x = torch.cat([x_first, x_middle, x_last], dim=1) # (B, (K-1)*l + l-1 + 1, D) = (B, K*l, D) = (B, L, D)

        x = self.final_proj(x)  # (b, k*l, D)

        return x


class CAT_Transformer_Block(TransformerBlock):
    def __init__(self, config: CAT_Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention = CAT_Attention(config, layer_idx)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, is_causal: Optional[bool] = True, mask: Optional[BlockMask] = None, input_pos: Optional[Tensor] = None) -> Tensor:
        h = x + self.attention(self.attention_norm(x), cos, sin)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class CAT_Layer_Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None: 
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)
        self.layers = nn.ModuleList(CAT_Transformer_Block(config, layer_idx=i) for i in range(config.n_layer))
        if self.config.use_fused_ops:
            self.norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

        if self.config.use_fused_ops:
            self.fused_linear_cross_entropy = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
        
        # initialize weights
        self.apply(lambda m: _init_weights(m, self.config.n_layer, self.config.dim))

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

        # setup cache for each layer
        for layer in self.layers:
            layer.attention.setup_cache(device=device)

    def forward(
        self,
        input_ids: torch.LongTensor, 
        labels: Optional[torch.LongTensor] = None, 
        input_pos: Optional[Tensor] = None, 
        mask: Optional[BlockMask] = None
    ) -> Tensor:
        bsz, seqlen = input_ids.shape

        x = self.wte(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, self.cos, self.sin)
        x = self.norm(x)

        if labels is not None:
            if self.config.use_fused_linear_cross_entropy:
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


class Attention(nn.Module):
    def __init__(self, dim, n_head, config: CAT_Config, layer_idx: int) -> None:
        super().__init__()
        # key, query and value projections for all heads, but in a batch
        
        self.config = config
        self.dim = dim
        self.n_head = n_head
        self.n_local_heads = n_head
        self.head_dim = dim // n_head

        self.wqkv = nn.Linear(
            dim,
            (n_head + 2 * n_head) * self.head_dim,  # support for grouped/multi queries
            bias=False,
        )
        # output projection
        self.wo = nn.Linear(self.head_dim * n_head, dim, bias=False)

        self.layer_idx = layer_idx

        self.rope_n_elem = config.rope_n_elem

        if config.use_qk_norm:
            if config.use_fused_ops:
                self.q_norm = LigerRMSNorm(self.head_dim, eps=config.norm_eps)
                self.k_norm = LigerRMSNorm(self.head_dim, eps=config.norm_eps)
            else:
                self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
                self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

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

        # if self.kv_cache is not None and input_pos is not None:
        #     k, v = self.kv_cache.update(input_pos, k, v)

        # TODO: @jatin 15-04-2025: add flash-attn github API
        if mask is None:
            scale = 1.0 / math.sqrt(self.head_dim)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0,
                scale=scale, is_causal=is_causal, enable_gqa=(self.n_head != self.n_local_heads)
            ) # (B, n_head, N, head_dim)
        else:
            if input_pos is not None:
                # used during generation benchmarks
                y = flex_attention(q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads))
            else:
                y = flex_attention_compiled(q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads))

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim) # (B, N, D)

        y = self.wo(y) # (B, N, D)

        # Output projection.
        return y


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # simple test
    batch_size = 4
    seq_len = 2048
    chunk_size = 8
    dim = 768

    config = CAT_Config(
        dim=dim,
        n_head=16,
        chunk_size=chunk_size,

        # again, needs 2*dim for accurate decoding from compressed chunk representations
        dim_fx=2 * dim,

        block_size=seq_len,

        # right now, every layer is a CAT layer
        # but the implementation can be easily modified to create hybrid architectures :)
        n_layer=12,
    )

    model = CAT_Layer_Transformer(config)
    print(model)
    model.setup_cache(device=device)
    model.to(device)

    x = torch.randint(0, config.padded_vocab_size, (batch_size, seq_len), device=device)

    logits = model(x)

    # do stuff with logits ...
    print("logits shape:", logits.shape)