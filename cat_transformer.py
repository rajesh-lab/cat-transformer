import math
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
    build_rope_cache
)

from torch.nn.attention.flex_attention import create_block_mask

def get_cat_mask(block_size: int):
    """Attention mask for CATs: Figure 4 in the paper"""
    def cat_mask(b, h, q_idx, kv_idx):
        within_block = (q_idx // (block_size)) == (kv_idx // (block_size))
        divides_block = ((kv_idx % block_size) == 0)
        causal_mask = (q_idx >= kv_idx)
        return (divides_block | within_block) & causal_mask
    
    return cat_mask


@dataclass
class CATConfig(TransformerConfig):
    
    chunk_size: int = 16

    # compressor will compress to this dimension
    dim_fx: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        
        assert self.block_size % self.chunk_size == 0
        self.num_chunks = self.block_size // self.chunk_size

        if self.dim_fx is None:
            self.dim_fx = self.dim


class Compressor(nn.Module):
    def __init__(self, config: CATConfig) -> None:
        super().__init__()
        self.config = config

        self.num_chunks = config.num_chunks
        self.chunk_size = config.chunk_size
        self.block_size = config.block_size

        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)
        self.pos_tokens = nn.Embedding(config.num_chunks, config.dim)

        self.layers = nn.ModuleList(TransformerBlock(config, layer_idx=i) for i in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.proj_fx = nn.Linear(config.dim * config.chunk_size, config.dim_fx, bias=False)

        self.is_causal = False
        # self.mask = create_block_mask(
        #     get_bidirectional_sparse_mask(1 + self.chunk_size), 
        #     B=None, H=None, 
        #     Q_LEN=self.block_size + self.num_chunks, # K+L
        #     KV_LEN=self.block_size + self.num_chunks, # K+L
        # )
        # print(self.mask)
        self.apply(lambda m: _init_weights(m, self.config.n_layer, self.config.dim))

    def setup_cache(self, device=None):

        cos, sin = build_rope_cache(
            1 + self.chunk_size, # +1 for the position token
            self.config.rope_n_elem, 
            device=device, 
            base=self.config.rope_base
        )
        # cos, sin = einops.repeat(cos, '1 l d -> 1 (k l) d', k=self.num_chunks), \
        #     einops.repeat(sin, '1 l d -> 1 (k l) d', k=self.num_chunks) # (1, K*(1+chunk_size), dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        print("created cos and sin cache for chunked encoder ...")
        print("cos shape:", self.cos.shape)
        print("cos dtype:", self.cos.dtype)

    def compress(self, input_ids: torch.LongTensor, chunk_idx: torch.LongTensor) -> Tensor:
        # input_ids: (bsz, chunk_size)
        # chunk_idx: (1)
        bsz, seqlen = input_ids.shape
        assert seqlen == self.chunk_size

        x = self.wte(input_ids) # (bsz, chunk_size, dim)

        pos_token = self.pos_tokens(chunk_idx) # (dim)
        pos_token = einops.repeat(pos_token, 'd -> b 1 d', b=bsz) # (bsz, 1, dim)
        x = torch.cat([pos_token, x], dim=1) # (bsz, 1 + chunk_size, dim)

        for layer in self.layers:
            x = layer(x, self.cos, self.sin, is_causal=self.is_causal)
        x = self.norm(x) # (bsz, chunk_size + 1, dim)

        x = x[:, 1:, :] # (bsz, chunk_size, dim) # remove the position token
        x = einops.rearrange(x, 'b l d -> b (l d)') # (bsz, chunk_size * dim)
        x = self.proj_fx(x) # (bsz, chunk_size * dim) -> (bsz, dim)

        return x


class CAT_Transformer(nn.Module):
    def __init__(self, config: CATConfig, f_config: CATConfig) -> None:
        super().__init__()
        self.config = config

        self.num_chunks = config.num_chunks
        self.chunk_size = config.chunk_size
        self.block_size = config.block_size

        # encoder/compressor
        self.f = Compressor(f_config)

        # decoder params
        self.dummy_fx = nn.Embedding(1, config.dim)
        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, layer_idx=i) for i in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

        if self.f.config.dim_fx != self.config.dim:
            print(f"Warning: f.dim_fx ({self.f.config.dim_fx}) != config.dim ({self.config.dim}), using up/down projection.")
            self.down_proj = nn.Linear(self.f.config.dim_fx, self.config.dim, bias=False)
        else:
            print(f"f.dim_fx ({self.f.config.dim_fx}) == config.dim ({self.config.dim}), no down projection.")
            self.down_proj = nn.Identity()

        self.mask = create_block_mask(
            get_cat_mask(1 + self.chunk_size), 
            B=None, H=None, 
            Q_LEN=self.block_size + self.num_chunks + 1, # K+L
            KV_LEN=self.block_size + self.num_chunks + 1, # K+L
        )
        print("Using mask:")
        print(self.mask)

        # init weights
        self.apply(lambda m: _init_weights(m, self.config.n_layer, self.config.dim))
        self.f.apply(lambda m: _init_weights(m, self.f.config.n_layer, self.f.config.dim))

        self.get_mask_mod = get_mask_mod

    def setup_cache(self, device: torch.device):

        self.f.setup_cache(device=device) # called for the encoder/compressor

        # first, cache below for doing generation
        _cos, _sin = build_rope_cache(self.num_chunks + self.chunk_size, self.config.rope_n_elem, device=device, base=self.config.rope_base)
        self.register_buffer("cos_gen", _cos, persistent=False)
        self.register_buffer("sin_gen", _sin, persistent=False)

        cos, sin = build_rope_cache(1 + self.chunk_size, self.config.rope_n_elem, device=device, base=self.config.rope_base)
        cos = einops.repeat(cos, '1 l d -> 1 k l d', k=self.num_chunks).clone()
        sin = einops.repeat(sin, '1 l d -> 1 k l d', k=self.num_chunks).clone()

        for i in range(self.num_chunks):
            cos[0, i, :, :] = _cos[0, i : i + 1+self.chunk_size, :].clone() # (1, 1+chunk_size, d)
            sin[0, i, :, :] = _sin[0, i : i + 1+self.chunk_size, :].clone() # (1, 1+chunk_size, d)

        cos = einops.rearrange(cos, '1 k l d -> 1 (k l) d')
        sin = einops.rearrange(sin, '1 k l d -> 1 (k l) d')

        # careful!
        cos_last = _cos[0, self.num_chunks, :].clone() # (d,)
        sin_last = _sin[0, self.num_chunks, :].clone() # (d,)
        cos = torch.cat([cos, einops.rearrange(cos_last, 'd -> 1 1 d')], dim=1) # (1, k*l+1, d)
        sin = torch.cat([sin, einops.rearrange(sin_last, 'd -> 1 1 d')], dim=1) # (1, k*l+1, d)

        # cache these for training
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        print("created cos and sin cache for CAT ...")
        print("cos shape:", self.cos.shape)
        print("cos dtype:", self.cos.dtype)


    # used for generation
    def setup_kv_cache(self, max_batch_size: int, dtype, device: torch.device):
        print("Setting up kv cache ...")
        for block in self.layers:
            block.attention.kv_cache = KVCache(
                max_batch_size, 
                (self.config.num_chunks + self.config.chunk_size), 
                self.config.n_local_heads, self.config.head_dim, dtype, device
            )


    def forward(self, input_ids: torch.LongTensor) -> Tensor:
        # input_ids: (B, L)
        bsz, seqlen = input_ids.shape
        cur_num_chunks = seqlen // self.chunk_size

        input_ids = input_ids.view(bsz, cur_num_chunks, self.chunk_size) # (B, K, l)

        # compress all chunks in parallel
        fx = torch.vmap(self.f.compress, in_dims=(1, 0), out_dims=1)(
            input_ids, # (B, K, l)
            torch.arange(cur_num_chunks, device=input_ids.device) # (K)
        ) # (B, K, D_fx)
        fx = self.down_proj(fx) # (B, K, D_fx) -> (B, K, D)
        fx_last = fx[:, -1, :].unsqueeze(1) # (B, 1, D)

        dummy_fx = self.dummy_fx(torch.zeros(1, device=input_ids.device, dtype=torch.long)) # (1, D)
        dummy_fx = einops.repeat(dummy_fx, '1 d -> b 1 d', b=bsz) # (B, 1, D)

        # directly pass the fx to the decoder
        fx = torch.cat([dummy_fx, fx[:, :-1, :]], dim=1) # concat{ (B, 1, D), (B, K-1, D) } = (B, K, D)
        fx = einops.rearrange(fx, 'b k d -> b k 1 d') # (B, K, 1, D)

        emb_x = self.wte(input_ids) # (B, K, l, D)
        x = torch.cat([fx, emb_x], dim=2) # (B, K, 1+l, D)
        x = einops.rearrange(x, 'b k l d -> b (k l) d') # (B, K + L, D) # flatten the chunks
        x = torch.cat([x, fx_last], dim=1) # (B, K + L + 1, D) # add the last chunk's representation

        for layer in self.layers:
            # pass through cat with masking
            x = layer(x, self.cos, self.sin, mask=self.mask)
        x = self.norm(x)

        # Arrange everything nicely back to (B, L, D) for next token prediction
        x_last = x[:, -1:, :].contiguous() # (B, 1, D)
        x = einops.rearrange(x[:, :-1, :], 'b (k l) d -> b k l d', k=cur_num_chunks, l=self.chunk_size+1) # (B, K, 1+l, D)
        x_first = x[:, :1, 1:-1, :].contiguous() # (B, 1, l-1, D)
        x_middle = x[:, 1:, :-1, :].contiguous() # (B, K-1, l, D)
        x_first = einops.rearrange(x_first, 'b 1 l d -> b (1 l) d') # (B, l-1, D)
        x_middle = einops.rearrange(x_middle, 'b k l d -> b (k l) d') # (B, (K-1)*l, D)
        x = torch.cat([x_first, x_middle, x_last], dim=1) # (B, (K-1)*l + l-1 + 1, D) = (B, K*l, D) = (B, L, D)

        logits = self.output(x) # (B, L, D) -> (B, L, V)
        return logits


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # look at table 1 for the paper for how this choice can affect language modeling and recall
    dim = 768
    dim_fx = 2 * dim # recommended to be 2*dim, but can be same as dim as well!
    decoder_dim = 2 * dim # recommended to be 2*dim
    
    block_size = 512
    chunk_size = 8

    compressor_config = CATConfig(dim=dim, dim_fx=dim_fx, block_size=block_size, chunk_size=chunk_size)
    decoder_config = CATConfig(dim=decoder_dim, block_size=block_size, chunk_size=chunk_size)

    model = CAT_Transformer(decoder_config, compressor_config)
    model = model.to(device=device)
    model.setup_cache(device=device)
    print(model)

    input_ids = torch.randint(0, decoder_config.vocab_size, (2, block_size), device=device)
    print("input_ids shape:", input_ids.shape)

    logits = model(input_ids)

    print("logits shape:", logits.shape)