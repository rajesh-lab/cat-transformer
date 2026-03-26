import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import einops

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

torch._dynamo.config.cache_size_limit = 32

from liger_kernel.transformers import LigerRMSNorm, liger_rotary_pos_emb, LigerFusedLinearCrossEntropyLoss
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

from cat_transformer import CAT_Config, Compressor


class CAT_Transformer_Looped(nn.Module):
    """Looped variant of CAT_Transformer.

    Instead of processing all chunks in parallel with a custom block-sparse
    attention mask (get_cat_mask), this version processes chunks sequentially
    in a loop.  For each chunk the decoder receives the compressed
    representations of all preceding chunks as history tokens, concatenated
    with the current chunk's token embeddings, and applies standard causal
    attention.

    Label alignment matches the parallel CAT: output[i] predicts input_ids[i+1].
    Chunk 0 contributes l-1 outputs, chunks 1..K-1 contribute l each, and
    fx_last contributes 1, totalling (l-1) + (K-1)*l + 1 = K*l = L.
    """

    def __init__(self, config: CAT_Config, f_config: CAT_Config) -> None:
        super().__init__()
        self.config = config

        self.num_chunks = config.num_chunks
        self.chunk_size = config.chunk_size
        self.block_size = config.block_size

        # compressor (same architecture as the parallel CAT)
        self.f = Compressor(f_config)

        # decoder params
        self.dummy_fx = nn.Embedding(1, config.dim)
        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, layer_idx=i) for i in range(config.n_layer))
        self.output = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

        if self.config.use_fused_ops:
            self.norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if self.config.use_fused_ops:
            self.fused_linear_cross_entropy = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

        self.down_proj = nn.Identity()
        assert self.f.config.dim_fx == self.config.dim, \
            "f.dim_fx (compressed chunk representation size) must be equal to config.dim (decoder hidden size)"

        # init weights
        self.apply(lambda m: _init_weights(m, self.config.n_layer, self.config.dim))
        self.f.apply(lambda m: _init_weights(m, self.f.config.n_layer, self.f.config.dim))

        self.get_mask_mod = get_mask_mod

    def setup_cache(self, device: torch.device):
        self.f.setup_cache(device=device)

        # max per-chunk sequence length is num_chunks + chunk_size
        # (the last chunk has K history tokens + l current tokens)
        cos, sin = build_rope_cache(
            self.num_chunks + self.chunk_size,
            self.config.rope_n_elem,
            device=device,
            base=self.config.rope_base
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # separate cache for generation (may need full block_size)
        _cos, _sin = build_rope_cache(
            self.block_size,
            self.config.rope_n_elem, device=device, base=self.config.rope_base
        )
        self.register_buffer("cos_gen", _cos, persistent=False)
        self.register_buffer("sin_gen", _sin, persistent=False)

        print("created cos and sin cache for Looped CAT ...")
        print("cos shape:", self.cos.shape)
        print("cos dtype:", self.cos.dtype)

    # ---- generation helpers (mirror the parallel CAT interface) ----

    def setup_kv_cache(self, max_batch_size: int, dtype, device: torch.device):
        print("Setting up kv cache ...")
        for block in self.layers:
            block.attention.kv_cache = KVCache(
                max_batch_size,
                (self.config.num_chunks + self.config.chunk_size),
                self.config.n_local_heads, self.config.head_dim, dtype, device
            )

    def forward_embeddings(
        self,
        x: Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        input_pos: Optional[Tensor] = None,
        rope_pos: Optional[Tensor] = None,
        mask: Optional[BlockMask] = None,
        is_input_token: bool = False,
    ) -> Tensor:
        bsz, seqlen = x.shape[0:2]

        if mask is not None and input_pos is not None:
            mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])

        if is_input_token:
            x = self.wte(x)

        if cos is None and sin is None:
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

    # ---- core looped forward ----

    def _forward_fx_last(self, fx_last: Tensor, history: Tensor) -> Tensor:
        """Run fx_last through the decoder with full history context.

        Args:
            fx_last: (B, 1, D) compressed representation of the last chunk.
            history: (B, K, D) all shifted history tokens.

        Returns:
            (B, 1, D) hidden state for predicting the token after the sequence.
        """
        x = torch.cat([history, fx_last], dim=1)  # (B, K+1, D)

        total_len = x.shape[1]
        cos = self.cos[:, :total_len, :]
        sin = self.sin[:, :total_len, :]

        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        return x[:, -1:, :]  # (B, 1, D)

    def forward_chunk(self, input_ids: Tensor, history_tokens: Tensor) -> Tensor:
        """Decode a single chunk with causal attention over history + current tokens.

        Args:
            input_ids:      (B, l) token ids for the current chunk.
            history_tokens: (B, k, D) compressed representations of prior chunks.

        Returns:
            (B, l, D) hidden states for next-token prediction within this chunk.
        """
        bsz, seqlen = input_ids.shape
        assert seqlen == self.chunk_size

        x = self.wte(input_ids)                        # (B, l, D)
        x = torch.cat([history_tokens, x], dim=1)      # (B, k+l, D)

        total_len = x.shape[1]
        cos = self.cos[:, :total_len, :]
        sin = self.sin[:, :total_len, :]

        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        # positions [k-1 .. k+l-2] predict tokens [tok_0 .. tok_{l-1}]
        k = history_tokens.shape[1]
        x = x[:, k - 1:-1, :]                          # (B, l, D)

        return x

    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None) -> Tensor:
        bsz, seqlen = input_ids.shape

        # pad to nearest chunk_size multiple if needed
        slice_end = False
        if seqlen % self.chunk_size != 0:
            new_seqlen = ((seqlen // self.chunk_size) + 1) * self.chunk_size
            pad_len = new_seqlen - seqlen
            input_ids = F.pad(input_ids, (0, pad_len), value=0)
            old_seqlen = seqlen
            seqlen = new_seqlen
            slice_end = True

        cur_num_chunks = seqlen // self.chunk_size
        input_ids = input_ids.view(bsz, cur_num_chunks, self.chunk_size)  # (B, K, l)

        # compress all chunks in parallel via vmap
        fx = torch.vmap(self.f.compress, in_dims=(1, 0), out_dims=1)(
            input_ids,                                                     # (B, K, l)
            torch.arange(cur_num_chunks, device=input_ids.device)          # (K)
        )  # (B, K, D_fx)
        fx = self.down_proj(fx)                                            # (B, K, D)
        fx_last = fx[:, -1:, :]                                            # (B, 1, D)

        # build shifted history: chunk i sees [dummy_fx, fx_0, ..., fx_{i-1}]
        dummy_fx = self.dummy_fx(torch.zeros(1, device=input_ids.device, dtype=torch.long))  # (1, D)
        dummy_fx = einops.repeat(dummy_fx, '1 d -> b 1 d', b=bsz)                           # (B, 1, D)
        fx = torch.cat([dummy_fx, fx[:, :-1, :]], dim=1)                                     # (B, K, D)

        # loop over chunks with growing history
        all_x = []
        for i in range(cur_num_chunks):
            history_tokens = fx[:, :i + 1, :]                              # (B, i+1, D)
            cur_x = self.forward_chunk(input_ids[:, i, :], history_tokens)  # (B, l, D)
            if i == 0:
                cur_x = cur_x[:, 1:, :]                                    # (B, l-1, D) skip dummy_fx pred
            all_x.append(cur_x)

        # fx_last predicts the token after the full sequence
        x_last = self._forward_fx_last(fx_last, fx)                        # (B, 1, D)
        all_x.append(x_last)

        all_x = torch.cat(all_x, dim=1)  # (B, (l-1)+(K-1)*l+1, D) = (B, L, D)

        if slice_end:
            all_x = all_x[:, :old_seqlen, :].contiguous()

        if labels is not None:
            if self.config.use_fused_ops:
                loss = self.fused_linear_cross_entropy(
                    self.output.weight, all_x.view(-1, all_x.size(-1)), labels.view(-1)
                )
                return loss
            else:
                logits = self.output(all_x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                return loss

        logits = self.output(all_x)  # (B, L, V)
        return logits


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # matches the example in cat_transformer.py
    dim = 768
    num_layers = 6
    n_head = 12

    decoder_dim = 2 * dim
    dim_fx = decoder_dim
    n_head_decoder = 2 * n_head

    block_size = 512
    chunk_size = 8

    compressor_config = CAT_Config(
        dim=dim, n_head=n_head, dim_fx=dim_fx,
        block_size=block_size, chunk_size=chunk_size,
        n_layer=(num_layers // 4),
    )
    decoder_config = CAT_Config(
        dim=decoder_dim, n_head=n_head_decoder,
        block_size=block_size, chunk_size=chunk_size,
        n_layer=num_layers,
    )

    model = CAT_Transformer_Looped(decoder_config, compressor_config)
    model = model.to(device=device)
    model.setup_cache(device=device)

    input_ids = torch.randint(0, decoder_config.vocab_size, (4, block_size), device=device)
    print("input_ids shape:", input_ids.shape)

    logits = model(input_ids)
    print("logits shape:", logits.shape)
    # do stuff with logits ...
