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

from torch.nn.attention.flex_attention import create_block_mask, BlockMask
create_block_mask = torch.compile(create_block_mask)

from liger_kernel.transformers import LigerRMSNorm, liger_rotary_pos_emb, LigerFusedLinearCrossEntropyLoss
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

def get_cat_mask(block_size: int):
    """Attention mask for CATs: Figure 8 in the appendix of the paper."""
    def cat_mask(b, h, q_idx, kv_idx):
        within_block = (q_idx // (block_size)) == (kv_idx // (block_size))
        divides_block = ((kv_idx % block_size) == 0)
        causal_mask = (q_idx >= kv_idx)
        return (divides_block | within_block) & causal_mask
    
    return cat_mask


@dataclass
class CAT_Config(TransformerConfig):
    
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
    def __init__(self, config: CAT_Config) -> None:
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
        self.apply(lambda m: _init_weights(m, self.config.n_layer, self.config.dim))

    def setup_cache(self, device=None):

        cos, sin = build_rope_cache(
            1 + self.chunk_size, # +1 for the position token
            self.config.rope_n_elem, 
            device=device, 
            base=self.config.rope_base
        )

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
    def __init__(self, config: CAT_Config, f_config: CAT_Config) -> None:
        super().__init__()
        self.config = config

        self.num_chunks = config.num_chunks
        self.chunk_size = config.chunk_size
        self.block_size = config.block_size

        # compressor
        self.f = Compressor(f_config)

        # decoder params
        self.dummy_fx = nn.Embedding(1, config.dim)
        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, layer_idx=i) for i in range(config.n_layer))
        self.output = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

        # fused rmsnorm
        if self.config.use_fused_ops:
            self.norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Fused cross entropy loss
        if self.config.use_fused_ops:
            self.fused_linear_cross_entropy = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

        self.down_proj = nn.Identity()
        assert self.f.config.dim_fx == self.config.dim, \
            "f.dim_fx (compressed chunk representation size) must be equal to config.dim (decoder hidden size)"

        self.mask = create_block_mask(
            get_cat_mask(1 + self.chunk_size), 
            B=None, H=None, 
            Q_LEN=self.block_size + self.num_chunks + 1, # K+L
            KV_LEN=self.block_size + self.num_chunks + 1, # K+L
        )

        # init weights
        self.apply(lambda m: _init_weights(m, self.config.n_layer, self.config.dim))
        self.f.apply(lambda m: _init_weights(m, self.f.config.n_layer, self.f.config.dim))

        self.get_mask_mod = get_mask_mod

    def setup_cache(self, device: torch.device):

        self.f.setup_cache(device=device) # called for the encoder/compressor

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

    # used for generation!
    def forward_embeddings(self, x: Tensor, cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None, input_pos: Optional[Tensor] = None, rope_pos: Optional[Tensor] = None, mask: Optional[BlockMask] = None, is_input_token=False) -> Tensor:
        # x: (B, l, D)
        bsz, seqlen = x.shape[0:2]
        assert seqlen <= self.chunk_size + self.num_chunks

        if mask is not None and input_pos is not None:
            # doing generation
            mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])

        if is_input_token:
            x = self.wte(x) # (B, l, D)

        if cos is None and sin is None: # if cos and sin are not provided, use the cached ones
            cos, sin = self.cos_gen, self.sin_gen

        if input_pos is not None:
            cos_gen = cos[:, rope_pos]
            sin_gen = sin[:, rope_pos]
        else:
            cos_gen = cos[:, :seqlen, :]
            sin_gen = sin[:, :seqlen, :]

        for layer in self.layers:
            x = layer(x, cos_gen, sin_gen, input_pos=input_pos, mask=mask)
        x = self.norm(x)

        logits = self.output(x)
        return logits


    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None) -> Tensor:
        # input_ids: (B, L)
        bsz, seqlen = input_ids.shape

        # handle non-multiple of chunk_size seqlen by padding
        slice_end = False
        if seqlen % self.chunk_size != 0:
            # pad to the next multiple of chunk_size
            new_seqlen = ((seqlen // self.chunk_size) + 1) * self.chunk_size
            pad_len = new_seqlen - seqlen
            # padding with zero (which is usually the pad token)
            # JP: replace with appropriate pad token of tokenizer
            input_ids = F.pad(input_ids, (0, pad_len), value=0) 
            old_seqlen = seqlen
            seqlen = new_seqlen
            slice_end = True

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

        # trim off extra cos and sin
        cos = self.cos[:, :x.shape[1], :]
        sin = self.sin[:, :x.shape[1], :]

        # create mask for this input size
        mask = create_block_mask(
            get_cat_mask(1 + self.chunk_size),
            B=None, H=None,
            Q_LEN=x.shape[1], # K+L
            KV_LEN=x.shape[1], # K+L
        )

        for layer in self.layers:
            # pass through cat with masking
            x = layer(x, cos=cos, sin=sin, mask=mask)
        x = self.norm(x)

        # Arrange everything nicely back to (B, L, D) for next token prediction
        x_last = x[:, -1:, :].contiguous() # (B, 1, D)
        x = einops.rearrange(x[:, :-1, :], 'b (k l) d -> b k l d', k=cur_num_chunks, l=self.chunk_size+1) # (B, K, 1+l, D)
        x_first = x[:, :1, 1:-1, :].contiguous() # (B, 1, l-1, D)
        x_middle = x[:, 1:, :-1, :].contiguous() # (B, K-1, l, D)
        x_first = einops.rearrange(x_first, 'b 1 l d -> b (1 l) d') # (B, l-1, D)
        x_middle = einops.rearrange(x_middle, 'b k l d -> b (k l) d') # (B, (K-1)*l, D)
        x = torch.cat([x_first, x_middle, x_last], dim=1) # (B, (K-1)*l + l-1 + 1, D) = (B, K*l, D) = (B, L, D)

        # if we padded, slice off the extra
        if slice_end:
            x = x[:, :old_seqlen, :].contiguous()

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

        logits = self.output(x) # (B, L, D) -> (B, L, V)
        return logits


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # below assumes that one wishes to instantiate a CAT that matches
    # a vanilla transformer containing 12 layers, and hidden size of 768
    dim = 768
    num_layers = 12
    n_head = 12

    # this is the hidden size of decoder, which is recommended to be 2*dim
    # however, it can be 1.5*dim, or 1.25*dim depending on the task
    # dim_fx means the size of the compressed chunk representations (f(c)'s), which
    # is same as hidden size of the decoder
    decoder_dim = 2 * dim # hidden size of the decoder
    dim_fx = decoder_dim # size of compressed chunk representations
    n_head_decoder = 2 * n_head # increase heads too proportionally

    block_size = 2048 # context length
    chunk_size = 8 # chunk size

    # instantiate the model
    compressor_config = CAT_Config(dim=dim, n_head=n_head, dim_fx=dim_fx, block_size=block_size, chunk_size=chunk_size, n_layer=(num_layers // 4)) # layers are defined according to the paper, but one may use lower number of layers in the compressor
    decoder_config = CAT_Config(dim=decoder_dim, n_head=n_head_decoder, block_size=block_size, chunk_size=chunk_size, n_layer=num_layers)
    model = CAT_Transformer(decoder_config, compressor_config)
    model = model.to(device=device)
    model.setup_cache(device=device)

    # do forward pass
    input_ids = torch.randint(0, decoder_config.vocab_size, (4, block_size), device=device)
    print("input_ids shape:", input_ids.shape)

    logits = model(input_ids)
    print("logits shape:", logits.shape)
    # do stuff with logits ...