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

# some helpers
def power_of_2(x: int) -> int:
    # 2^(x % iter_num)
    return int(2 ** x)

def power_of_2_exponent(n: int) -> int:
    if n <= 0:
        raise ValueError("Input must be a positive integer.")
    if (n & (n - 1)) != 0:
        raise ValueError(f"{n} is not a power of 2.")
    return int(math.log2(n))

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
        self.pos_tokens = nn.Embedding(self.block_size, config.dim) # this can have block_size tokens now

        self.power_of_two = int(math.log2(self.chunk_size))
        self.adaptive_token = nn.Embedding(self.power_of_two + 1, config.dim) # (1, dim)

        self.layers = nn.ModuleList(TransformerBlock(config, layer_idx=i) for i in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.proj_fx = nn.Linear(config.dim * config.chunk_size, config.dim_fx, bias=False)

        self.is_causal = False
        self.apply(lambda m: _init_weights(m, self.config.n_layer, self.config.dim))

    def setup_cache(self, device=None):

        cos, sin = build_rope_cache(
            2 + self.chunk_size, # +1 for the position and adaptive token
            self.config.rope_n_elem, 
            device=device, 
            base=self.config.rope_base
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        print("created cos and sin cache for chunked encoder ...")
        print("cos shape:", self.cos.shape)
        print("cos dtype:", self.cos.dtype)

    def compress(self, input_ids: torch.LongTensor, chunk_idx: torch.LongTensor, chunk_size_power: torch.LongTensor) -> Tensor:
        # input_ids: (bsz, chunk_size)
        # chunk_idx: (1)
        bsz, seqlen = input_ids.shape
        # assert seqlen == self.chunk_size # can't assert anymore :)

        x = self.wte(input_ids) # (bsz, chunk_size, dim)

        # trim cos, sin
        cos = self.cos[:, :2 + seqlen] # due to adptive
        sin = self.sin[:, :2 + seqlen]

        # pos_token tells the compressor which position this chunk is in the sequence
        pos_token = self.pos_tokens(chunk_idx) # (dim)
        pos_token = einops.repeat(pos_token, 'd -> b 1 d', b=bsz) # (bsz, 1, dim)

        # adaptive_token tells compressor which chunk size is being used
        adaptive_token = self.adaptive_token(chunk_size_power) # (dim)
        adaptive_token = einops.repeat(adaptive_token, 'd -> b 1 d', b=bsz) # (bsz, 1, dim)

        x = torch.cat([adaptive_token, pos_token, x], dim=1) # (bsz, 1 + chunk_size, dim)

        for layer in self.layers:
            x = layer(x, cos, sin, is_causal=self.is_causal)
        x = self.norm(x) # (bsz, chunk_size + 1, dim)

        x = x[:, 2:, :] # (bsz, chunk_size, dim) # remove the position token
        x = einops.rearrange(x, 'b l d -> b (l d)') # (bsz, chunk_size * dim)

        # below is inspired from: https://arxiv.org/abs/2212.08013
        # reshape proj_fx to adapt to different chunk sizes
        new_proj_fx_weight = torch.nn.functional.interpolate(
            self.proj_fx.weight.unsqueeze(0).unsqueeze(0), # (1, 1, out_features, in_features)
            size=(self.config.dim_fx, (2 ** chunk_size_power) * self.config.dim), # (out_features, in_features)
            mode="bilinear"
        ).squeeze(0).squeeze(0) # (out_features, in_features)
        x = torch.nn.functional.linear(x, new_proj_fx_weight, bias=None) # (bsz, dim_fx)

        return x


class CAT_Transformer(nn.Module):
    def __init__(self, config: CAT_Config, f_config: CAT_Config) -> None:
        super().__init__()
        self.config = config

        self.power_of_2_exponent = power_of_2_exponent(self.config.chunk_size)

        self.num_chunks = config.num_chunks
        self.chunk_size = config.chunk_size
        self.block_size = config.block_size

        # compressor
        self.f = Compressor(f_config)

        # decoder params
        # this also acts as the token which tells decoder the chunk size being used
        self.dummy_fx = nn.Embedding(self.power_of_2_exponent + 1, config.dim) 
        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, layer_idx=i) for i in range(config.n_layer))
        self.output = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

        # seperates the chunk representations and the token embeddings
        self.seperator = nn.Embedding(1, config.dim)

        # declare these early
        self.cos = dict()
        self.sin = dict()
        self.cos_gen = None
        self.sin_gen = None

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

        self.f.setup_cache(device=device) # called for the compressor
        print("power_of_2_exponent:", self.power_of_2_exponent)

        for c in range(1 + self.power_of_2_exponent):

            chunk_size = int(2 ** c)
            print("creating cos and sin cache for chunk size:", chunk_size)

            assert self.block_size % chunk_size == 0
            num_chunks = (self.block_size // chunk_size)

            _cos, _sin = build_rope_cache(
                num_chunks + chunk_size + 2, # this is okay
                self.config.rope_n_elem, 
                device=device,
                base=self.config.rope_base
            )
            if c == self.power_of_2_exponent:
                # this is the max chunk size
                # we need to cache these for doing generation
                self.cos_gen = _cos.clone()
                self.sin_gen = _sin.clone()

            cos, sin = build_rope_cache(
                2 + chunk_size,
                
                self.config.rope_n_elem, 
                device=device,
                base=self.config.rope_base
            )
            cos = einops.repeat(cos, '1 l d -> 1 k l d', k=num_chunks+1).clone()
            sin = einops.repeat(sin, '1 l d -> 1 k l d', k=num_chunks+1).clone()

            cos = einops.rearrange(cos, '1 k l d -> 1 (k l) d').clone()
            sin = einops.rearrange(sin, '1 k l d -> 1 (k l) d').clone()

            cos = cos[:, :self.block_size + 2*num_chunks + 2, :].clone()
            sin = sin[:, :self.block_size + 2*num_chunks + 2, :].clone()

            # cache for training
            self.cos[c] = cos.clone()
            self.sin[c] = sin.clone()

            print("created cos and sin cache for chunked decoder ...")
            print("cos shape:", self.cos[c].shape)
            print("cos dtype:", self.cos[c].dtype)


    # used for generation
    def setup_kv_cache(self, max_batch_size: int, dtype, device: torch.device):
        print("Setting up kv cache ...")
        for block in self.layers:
            block.attention.kv_cache = KVCache(
                max_batch_size, 
                (self.config.num_chunks + self.config.chunk_size), 
                self.config.n_local_heads, self.config.head_dim, dtype, device
            )


    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None, chunk_size_power: Optional[int] = None) -> Tensor:
        # input_ids: (B, L)
        bsz, seqlen = input_ids.shape

        assert chunk_size_power is not None
        cur_iter_chunk_size = power_of_2(chunk_size_power)

        # handle non-multiple of chunk_size seqlen by padding
        slice_end = False
        if seqlen % cur_iter_chunk_size != 0:
            # pad to the next multiple of chunk_size
            new_seqlen = ((seqlen // cur_iter_chunk_size) + 1) * cur_iter_chunk_size
            pad_len = new_seqlen - seqlen
            # padding with zero (which is usually the pad token)
            # JP: replace with appropriate pad token of tokenizer
            input_ids = F.pad(input_ids, (0, pad_len), value=0) 
            old_seqlen = seqlen
            seqlen = new_seqlen
            slice_end = True

        cur_num_chunks = seqlen // cur_iter_chunk_size

        input_ids = input_ids.view(bsz, cur_num_chunks, cur_iter_chunk_size) # (B, K, l)

        # compress all chunks in parallel
        fx = torch.vmap(self.f.compress, in_dims=(1, 0, None), out_dims=1)(
            input_ids, # (B, K, l)
            torch.arange(cur_num_chunks, device=input_ids.device), # (K)
            torch.tensor(chunk_size_power, device=input_ids.device, dtype=torch.long) # (K)
        ) # (B, K, D_fx)
        fx = self.down_proj(fx) # (B, K, D_fx) -> (B, K, D)
        fx_last = fx[:, -1, :].unsqueeze(1) # (B, 1, D)

        # attach the conditioning vector, which is dummy_fx
        # this tells the decoder which chunk size is being used
        dummy_fx = self.dummy_fx(torch.tensor([chunk_size_power], device=input_ids.device, dtype=torch.long)) # (1, D) # this is now the conditioning vector
        dummy_fx = einops.repeat(dummy_fx, '1 d -> b 1 d', b=bsz) # (B, 1, D)

        fx = torch.cat([dummy_fx, fx[:, :-1, :]], dim=1) # concat{ (B, 1, D), (B, K-1, D) } = (B, K, D)
        fx = einops.rearrange(fx, 'b k d -> b k 1 d') # (B, K, 1, D)

        # create seperator tokens
        # these seperate the chunk representations and the token embeddings in decoder
        sep_token = self.seperator(torch.zeros(1, device=input_ids.device, dtype=torch.long)) # (1, D)
        sep_token = einops.repeat(sep_token, '1 d -> b k 1 d', b=bsz, k=cur_num_chunks) # (B, K, 1, D)
        last_sep_token = self.seperator(torch.zeros(1, device=input_ids.device, dtype=torch.long)) # (1, D)
        last_sep_token = einops.repeat(last_sep_token, '1 d -> b 1 d', b=bsz) # (B, 1, D)

        emb_x = self.wte(input_ids) # (B, K, l, D)
        x = torch.cat([fx, sep_token, emb_x], dim=2) # (B, K, 2+l, D)
        x = einops.rearrange(x, 'b k l d -> b (k l) d') # (B, K + L, D) # flatten the chunks
        x = torch.cat([x, fx_last, last_sep_token], dim=1) # (B, K + L + 2, D) # add the last chunk's representation

        cos = self.cos[chunk_size_power][:, :x.shape[1], :]
        sin = self.sin[chunk_size_power][:, :x.shape[1], :]

        mask = create_block_mask(
            get_cat_mask(1 + 1 + cur_iter_chunk_size),
            B=None, H=None,
            Q_LEN=x.shape[1], # K+L
            KV_LEN=x.shape[1], # K+L
        )

        for layer in self.layers:
            x = layer(x, cos, sin, mask=mask)
        x = self.norm(x)

        # Arrange everything nicely back to (B, L, D) for next token prediction
        x_last = x[:, -1:, :].contiguous() # (B, 1, D)
        x = einops.rearrange(x[:, :-2, :], 'b (k l) d -> b k l d', k=cur_num_chunks, l=cur_iter_chunk_size+2) # (B, K, 2+l, D)
        x_first = x[:, :1, 2:-1, :].contiguous() # (B, 1, l-1, D)
        x_middle = x[:, 1:, 1:-1, :].contiguous() # (B, K-1, l, D)
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
    num_layers = 4
    n_head = 12

    # this is the hidden size of decoder, which is recommended to be 2*dim
    # however, it can be 1.5*dim, or 1.25*dim depending on the task
    # dim_fx means the size of the compressed chunk representations (f(c)'s), which
    # is same as hidden size of the decoder
    decoder_dim = 2 * dim # hidden size of the decoder
    dim_fx = decoder_dim # size of compressed chunk representations
    n_head_decoder = 2 * n_head # increase heads too proportionally

    block_size = 2048 # context length
    chunk_size = 32 # chunk size

    # instantiate the model
    compressor_config = CAT_Config(dim=dim, n_head=n_head, dim_fx=dim_fx, block_size=block_size, chunk_size=chunk_size, n_layer=(num_layers // 4)) # layers are defined according to the paper, but one may use lower number of layers in the compressor
    decoder_config = CAT_Config(dim=decoder_dim, n_head=n_head_decoder, block_size=block_size, chunk_size=chunk_size, n_layer=num_layers)
    model = CAT_Transformer(decoder_config, compressor_config)
    model = model.to(device=device)
    model.setup_cache(device=device)

    # do forward pass
    input_ids = torch.randint(0, decoder_config.vocab_size, (4, block_size), device=device)
    print("input_ids shape:", input_ids.shape)

    # choose which chunk size to use for this forward pass
    # must be power of 2, and and less than or equal to chunk_size
    # only powers of two supported for now
    cur_chunk_size_power = 4 # corresponds to chunk size of 16 (2^4)

    logits = model(input_ids, chunk_size_power=cur_chunk_size_power)

    print("logits shape:", logits.shape)
    # do stuff with logits ...