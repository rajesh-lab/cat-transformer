import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from itertools import permutations
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import einops

from accelerate import Accelerator


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


from .transformer import (
    TransformerConfig,
    TransformerBlock,
    RMSNorm,
    _init_weights,
    LigerRMSNorm,
    LigerFusedLinearCrossEntropyLoss
)

from .utils.projection import (
    ProjectionHead,
    ProjectionBlock
)

from .utils.rope import build_rope_cache

from torch.nn.attention.flex_attention import (
    create_block_mask, 
    BlockMask
)


def get_bidirectional_sparse_mask(block_size: int):

    def bidirectional_sparse_mask(b, h, q_idx, kv_idx):
        within_block = (q_idx // (block_size)) == (kv_idx // (block_size))
        return within_block
    
    return bidirectional_sparse_mask

# util for generating permutations for augmentation
def generate_perm_dict(n):
    base_array = np.arange(n)
    all_perms = np.array(list(permutations(base_array)))
    
    # Filter out the identity permutation
    filtered_perms = all_perms[~np.all(all_perms == base_array, axis=1)]
    
    # Create the dictionary
    perm_dict = {i: row.tolist() for i, row in enumerate(filtered_perms)}
    
    return perm_dict

@dataclass
class ChunkedTransformerConfig(TransformerConfig):

    chunk_size: int = 16
    proj_n_layer: int = 2

    train_from_chunk: Optional[int] = -1
    single_history: bool = False

    use_h: bool = False
    augment_mqar: bool = False

    perm_dict: dict = None

    dim_fx: Optional[int] = None

    # concat_summary: bool = True

    norm_w: float = 1.0

    learn_norm_w: bool = False

    chunked_type: str = "encoder"  # "encoder", "decoder" or "scorer"

    use_register_tokens: bool = False

    use_cannon: bool = False

    def __post_init__(self):
        super().__post_init__()
        
        assert self.block_size % self.chunk_size == 0
        self.num_chunks = self.block_size // self.chunk_size

        assert self.proj_n_layer >= 2

        print("Using ChunkedTransformer type:", self.chunked_type)

        if self.chunked_type == "encoder":

            if self.dim_fx is None:
                print("Warning: dim_fx is not set, using dim as dim_fx....")
                self.dim_fx = self.dim

            assert self.dim_fx is not None
            print(f"~~~~~~ Using f(x) dim: {self.dim_fx} ~~~~~~")

            if self.train_from_chunk != -1:
                print(f"~~~~~~ Training from chunk_idx {self.train_from_chunk} ~~~~~~~")

            if self.use_h:
                print("~~~~~~ Using h for dot-product ~~~~~~")

            if self.augment_mqar:
                print("~~~~~~ Using mqar augmentation: swap key-values~~~~~~")
                assert self.dataset == "mqar"

            if self.use_register_tokens:
                self.num_register_tokens = self.dim_fx // self.dim
                print(f"~~~~~~ Using register tokens {self.num_register_tokens} .... ~~~~~~")

            # if self.dataset == "mqar":
            #     print(f"Generating permutations for chunk_size {self.chunk_size}... may take a while...")
            #     self.perm_dict = generate_perm_dict(self.chunk_size)

        if self.chunked_type == "decoder":
            if self.norm_w != 1.0:
                print(f"~~~~~~ Using norm_w {self.norm_w} ~~~~~~")

            if self.learn_norm_w:
                print("~~~~~~ Using learnable norm_w ~~~~~~")

            if self.use_cannon:
                print("~~~~~~ Using cannon 🎶 residuals for chunked decoder ~~~~~~")

class Scorer(nn.Module):
    def __init__(self, config: ChunkedTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.chunk_size = config.chunk_size
        self.num_chunks = config.num_chunks
        if config.use_pos_emb:
            self.pos_embed = nn.Embedding(config.num_chunks, config.dim)

        self.layers = nn.ModuleList(TransformerBlock(config, layer_idx=i) for i in range(config.n_layer))
        if self.config.use_fused_rmnsnorm:
            self.norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # init weights in encoder

    def setup_cache(self, device=None):
        # init RoPE
        cos, sin = build_rope_cache(
            self.num_chunks, 
            self.config.rope_n_elem, 
            device=device, 
            base=self.config.rope_base
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        print("created cos and sin cache for scorer ...")
        print("cos shape:", self.cos.shape)
        print("cos dtype:", self.cos.dtype)


    def forward(self, x: Tensor) -> Tensor:
        # x: (B, k, D)

        bsz, seqlen, _ = x.shape

        cos = self.cos[:, :seqlen]
        sin = self.sin[:, :seqlen]

        # add position embedding
        if self.config.use_pos_emb:
            pos_embed = self.pos_embed(torch.arange(x.shape[1], device=x.device)) # (K, D)
            pos_embed = einops.rearrange(pos_embed, 'l d -> 1 l d') # (1, K, D)
            x = x + pos_embed # (B, K, D) # broadcast

        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        return x


class ChunkedEncoder(nn.Module):
    def __init__(self, config: ChunkedTransformerConfig, scorer: Optional[Scorer] = None) -> None:
        super().__init__()
        self.config = config

        self.num_chunks = config.num_chunks
        self.chunk_size = config.chunk_size
        self.block_size = config.block_size

        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)
        if self.config.use_pos_emb:
            self.pos_embed = nn.Embedding(1 + config.chunk_size, config.dim)
        self.pos_tokens = nn.Embedding(config.num_chunks, config.dim)

        self.layers = nn.ModuleList(TransformerBlock(config, layer_idx=i) for i in range(config.n_layer))
        if self.config.use_fused_rmnsnorm:
            self.norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if config.use_register_tokens:
            self.register_tokens = nn.Embedding(config.num_register_tokens, config.dim)
            print(f"~~~~~~ Using register tokens {config.num_register_tokens} !!!! ~~~~~~")
        else:
            self.proj_fx = nn.Sequential(
                nn.Linear(config.dim * self.chunk_size, config.dim_fx, bias=False),
                # RMSNorm(config.dim_fx, eps=config.norm_eps),
                # nn.ReLU(),

                # nn.Linear(config.dim_fx, config.dim_fx, bias=False),
                # RMSNorm(config.dim_fx, eps=config.norm_eps),
            )

        # TODO: create mask
        self.is_causal = False
        # self.mask = create_block_mask(
        #     get_bidirectional_sparse_mask(1 + self.chunk_size), 
        #     B=None, H=None, 
        #     Q_LEN=self.block_size + self.num_chunks, # K+L
        #     KV_LEN=self.block_size + self.num_chunks, # K+L
        # )
        # print(self.mask)

        # contrastive loss
        self.g = scorer
        # self.left_proj = ProjectionHead(config)
        # self.right_proj = ProjectionHead(config)

        # this should be non-linear due to contrastive loss
        if self.config.dataset == "mqar":
            self.right_proj = nn.Sequential(
                nn.Linear(self.config.dim_fx, self.config.dim_fx, bias=False),
                RMSNorm(config.dim_fx, eps=config.norm_eps),
                nn.ReLU(),

                nn.Linear(self.config.dim_fx, self.config.dim_fx, bias=False),
                RMSNorm(self.config.dim_fx, eps=config.norm_eps),
            )
        else:
            print("Using 3-layer projection for f(x) right_proj ...")
            self.right_proj = nn.Sequential(
                nn.Linear(self.config.dim_fx, self.config.dim, bias=False),
                RMSNorm(config.dim, eps=config.norm_eps),
                nn.ReLU(),

                nn.Linear(self.config.dim, self.config.dim, bias=False),
                RMSNorm(config.dim, eps=config.norm_eps),
                nn.ReLU(),

                nn.Linear(self.config.dim, self.config.dim, bias=False),
                RMSNorm(config.dim, eps=config.norm_eps),
            )

        if config.use_h:
            self.h = nn.Sequential(
                ProjectionBlock(2 * config.dim, config.norm_eps),
                ProjectionBlock(2 * config.dim, config.norm_eps),

                nn.Linear(2 * config.dim, 1, bias=False),
                # RMSNorm(1, eps=config.norm_eps),
            )

        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # init the encoder first
        self.apply(lambda m: _init_weights(m, self.config.n_layer, self.config.dim))
        self.g.apply(lambda m: _init_weights(m, self.g.config.n_layer, self.g.config.dim))

    def setup_cache(self, device=None):

        # call for the scorer
        self.g.setup_cache(device=device)

        cos, sin = build_rope_cache(
            self.config.num_register_tokens + 1 + self.chunk_size if self.config.use_register_tokens \
                else 1 + self.chunk_size, # +1 for the position token
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

    def encode(self, input_ids: torch.LongTensor, chunk_idx: torch.LongTensor) -> Tensor:
        # input_ids: (bsz, chunk_size)
        # chunk_idx: (1)
        
        bsz, seqlen = input_ids.shape
        assert seqlen == self.chunk_size

        # trim cos, sin
        # cos = self.cos[:, :1 + seqlen] # due to pos token
        # sin = self.sin[:, :1 + seqlen]

        x = self.wte(input_ids) # (bsz, chunk_size, dim)

        pos_token = self.pos_tokens(chunk_idx) # (dim)
        pos_token = einops.repeat(pos_token, 'd -> b 1 d', b=bsz) # (bsz, 1, dim)

        if self.config.use_register_tokens:
            register_tokens = self.register_tokens.weight
            register_tokens = einops.repeat(register_tokens, 'r d -> b r d', b=bsz) # (bsz, num_register_tokens, dim)
            x = torch.cat([register_tokens, pos_token, x], dim=1) # (bsz, num_register_tokens + 1 + chunk_size, dim)
        else:
            x = torch.cat([pos_token, x], dim=1) # (bsz, 1 + chunk_size, dim)

        # add position embedding
        if self.config.use_pos_emb:
            pos_embed = self.pos_embed(torch.arange(1 + self.chunk_size, device=input_ids.device)) # (1 + chunk_size, dim)
            pos_embed = einops.rearrange(pos_embed, 'l d -> 1 l d') # (1, 1 + chunk_size, dim)
            x = x + pos_embed # (bsz, 1 + chunk_size, dim) # broadcast

        for layer in self.layers:
            x = layer(x, self.cos, self.sin, is_causal=self.is_causal)
        x = self.norm(x) # (bsz, chunk_size + 1, dim)

        # dont project for now, use the first token
        if self.config.use_register_tokens:
            x = x[:, :self.config.num_register_tokens, :] # (bsz, r, dim)
            x = einops.rearrange(x, 'b r d -> b (r d)') # (bsz, num_register_tokens * dim)
        else:
            x = x[:, 1:, :] # (bsz, chunk_size, dim) # remove the position token
            x = einops.rearrange(x, 'b l d -> b (l d)') # (bsz, chunk_size * dim)
            x = self.proj_fx(x) # (bsz, chunk_size * dim) -> (bsz, dim)

        return x
    
    # def encode_parallel(self, x: Tensor, chunk_idx: torch.LongTensor) -> Tensor:
    #     # input_ids: (bsz, K, l, D)
    #     # chunk_idx: (K)
    #     # output: (bsz, K, D)
        
    #     bsz, num_chunks, seqlen = x.shape
    #     assert seqlen == self.chunk_size
    #     assert not self.config.use_register_tokens # does not support register tokens

    #     x = self.wte(x) # (bsz, K, chunk_size, dim)
    #     pos_token = self.pos_tokens(chunk_idx) # (K, dim)
    #     pos_token = einops.repeat(pos_token, 'k d -> b k 1 d', b=bsz) # (bsz, K, 1, dim)
    #     x = torch.cat([pos_token, x], dim=2) # (bsz, K, 1 + chunk_size, dim)

    #     # add position embedding
    #     if self.config.use_pos_emb:
    #         pos_embed = self.pos_embed(torch.arange(1 + self.chunk_size, device=x.device))
    #         pos_embed = einops.rearrange(pos_embed, 'l d -> 1 1 l d') # (1, 1, 1 + chunk_size, dim)
    #         x = x + pos_embed # (bsz, K, 1 + chunk_size, dim) # broadcast
        
    #     x = einops.rearrange(x, 'b k l d -> b (k l) d') # (bsz, K * (1 + chunk_size), dim)
    #     for layer in self.layers:
    #         x = layer(x, self.cos, self.sin, mask=self.mask)
    #     x = self.norm(x) # (bsz, K * (1 + chunk_size), dim)

    #     x = einops.rearrange(x, 'b (k l) d -> b k l d', k=num_chunks) # (bsz, K, 1 + chunk_size, dim)

    #     x = x[:, :, 1:, :] # (bsz * K, chunk_size, dim) # remove the position token
    #     x = einops.rearrange(x, '... l d -> ... (l d)') # (bsz, chunk_size * dim)
    #     x = self.proj_fx(x) # (bsz, K, chunk_size * dim) -> (bsz, K, dim)

    #     return x

    def forward(self, accelerate: Accelerator, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None, validate: bool = False) -> Tensor:

        bsz, seqlen = input_ids.shape
        assert seqlen == self.block_size

        input_ids = input_ids.view(bsz, self.num_chunks, self.chunk_size) # (bsz, K, l)

        if self.config.augment_mqar or validate:
            coin_toss = np.random.randint(len(self.config.perm_dict))
            negative_input_ids = input_ids.clone()
            negative_input_ids = negative_input_ids[..., self.config.perm_dict[coin_toss]]  # permute the tokens in chunk
            input_ids = torch.cat([input_ids, negative_input_ids], dim=0) # (2*bsz, K, l)
            
        pos_ids = torch.arange(self.num_chunks, device=input_ids.device) # (K)

        fx = torch.vmap(self.encode, in_dims=(1, 0), out_dims=1, randomness="different")(input_ids, pos_ids) # (bsz, K, l, D) -> (bsz, K, D)
        # fx = self.encode_parallel(input_ids, pos_ids) # (bsz, K, l, D) -> (bsz, K, D)
        gfx = self.g(fx[:bsz]) # (bsz, K, D) -> (bsz, K, D)

        fx = self.right_proj(fx) # (2*bsz, K, D)

        gfx = gfx[:, :-1, :] # (bsz, K-1, D)
        fx = fx[:, 1:, :] # (2*bsz, K-1, D)

        # since num_kv_pairs=16, and chunk_size=8, we skip the first 16*2/8=4 chunks
        # NOTE: only apply this during mqar dataset training!!
        if self.config.train_from_chunk != -1:
            # print("helloooo")
            gfx = gfx[:, self.config.train_from_chunk:, :]
            fx = fx[:, self.config.train_from_chunk:, :]

        if self.config.use_h:
            contrastive_loss = torch.vmap(clip_loss_h, in_dims=(1, 1, None, None), out_dims=0)(gfx, fx, self.logit_scale, self.h) # (bsz, K, D) -> (bsz)
        else:
            contrastive_loss = torch.vmap(clip_loss, in_dims=(1, 1, None), out_dims=0)(gfx, fx, self.logit_scale) # (bsz, K, D) -> (bsz)

        if self.config.dataset == "mqar":
            return contrastive_loss
        else:
            return contrastive_loss.mean()


class ChunkedDecoder(nn.Module):
    def __init__(self, config: ChunkedTransformerConfig, encoder: ChunkedEncoder) -> None:
        super().__init__()
        self.config = config

        self.num_chunks = config.num_chunks
        self.chunk_size = config.chunk_size
        self.block_size = config.block_size

        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)
        self.dummy_history = nn.Embedding(1, config.dim)

        self.layers = nn.ModuleList(TransformerBlock(config, layer_idx=i) for i in range(config.n_layer))
        if self.config.use_fused_rmnsnorm:
            self.norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.output = nn.Linear(config.dim, config.padded_vocab_size, bias=False)
        if self.config.use_fused_linear_cross_entropy:
            if self.config.dataset == "mqar":
                print("Disabling fused cross entropy for mqar dataset")
                self.config.use_fused_linear_cross_entropy = False
            else:
                self.fused_linear_cross_entropy = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

        self.f = encoder

        # init weights
        self.apply(lambda m: _init_weights(m, self.config.n_layer, self.config.dim))
        self.f.apply(lambda m: _init_weights(m, self.f.config.n_layer, self.f.config.dim))
        self.f.g.apply(lambda m: _init_weights(m, self.f.g.config.n_layer, self.f.g.config.dim))

        # should be non-linear since info might be complex
        # self.down_proj = nn.Sequential(
        #     nn.Linear(self.f.config.dim_fx, self.config.dim, bias=False),
        #     RMSNorm(self.config.dim, eps=config.norm_eps),
        #     nn.ReLU(),

        #     nn.Linear(self.config.dim, self.config.dim, bias=False),
        # )


    def setup_cache(self, device: torch.device):
        self.f.setup_cache(device=device) # called for the encoder

        cos, sin = build_rope_cache(
            1 + self.chunk_size if self.config.single_history else \
            self.num_chunks + self.chunk_size, # +1 for the history token
            self.config.rope_n_elem,
            device=device,
            base=self.config.rope_base
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)


    def forward_embeddings(self, x: Tensor) -> Tensor:
        # x: (B, l, D)
        bsz, seqlen, _ = x.shape
        assert seqlen <= self.chunk_size + self.num_chunks # account for the history token

        cos = self.cos[:, :seqlen, :]
        sin = self.sin[:, :seqlen, :]

        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        logits = self.output(x)
        return logits


    def forward_chunk(self, input_ids: Tensor, history_tokens: Tensor) -> Tensor:
        # input_ids: (B, l)
        # history_tokens: (B, k, D)

        bsz, seqlen = input_ids.shape
        assert seqlen == self.chunk_size # (B, l)

        x = self.wte(input_ids) # (B, l, D)
        if self.config.use_cannon:
            x = x + history_tokens
        x = torch.cat([history_tokens, x], dim=1) # (B, k+l, D)

        seqlen = x.shape[1] # (B, k+l, D)

        cos = self.cos[:, :seqlen, :] # (1, k+l, D)
        sin = self.sin[:, :seqlen, :] # (1, k+l, D)

        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        # logits = self.output(x)

        start_idx = history_tokens.shape[1] - 1 # this is 0 if there is only a single history token
        x = x[:, start_idx:-1, :] # ignore last token

        return x # (B, l, D)


    def forward(self, accelerate: Accelerator, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None) -> Tensor:
        # input_ids: (B, L)
        bsz, seqlen = input_ids.shape
        # assert seqlen == self.block_size # mqar dataset has variable seq lens
        cur_num_chunks = seqlen // self.chunk_size # this can be different due to mqar dataset

        input_ids = input_ids.view(bsz, cur_num_chunks, self.chunk_size) # (B, K, l)

        fx = torch.vmap(self.f.encode, in_dims=(1, 0), out_dims=1)(
            input_ids, # (B, K, l)
            torch.arange(cur_num_chunks, device=input_ids.device) # (K)
        ) # (B, K, D_fx)
        # fx = self.down_proj(fx) # (B, K, D_fx) -> (B, K, D)

        dummy_history = self.dummy_history(torch.zeros(1, device=input_ids.device, dtype=torch.long)) # (1, D)
        dummy_history = einops.repeat(dummy_history, '1 d -> b 1 d', b=bsz) # (B, 1, D)

        if self.config.single_history:

            gfx = self.f.g(fx) # (B, K, D)
            gfx = torch.cat([dummy_history, gfx[:, :-1, :]], dim=1) # concat{ (B, 1, D), (B, K-1, D) } = (B, K, D)

            # pass through a non-linearity before passing to the decoder
            # gfx = self.proj(gfx) # (B, K, D)
            # gfx = self.f.left_proj(gfx)

            all_x = torch.vmap(self.forward_chunk, in_dims=(1, 1), out_dims=1)(
                input_ids, # (B, K, l)
                gfx.unsqueeze(2) * self.config.norm_w # (B, K, 1, D)
            ) # (B, K, l, V)
            all_x = einops.rearrange(all_x, 'b k l d -> b (k l) d')
        else:
            # directly pass the fx to the decoder
            fx = torch.cat([dummy_history, fx[:, :-1, :]], dim=1) # concat{ (B, 1, D), (B, K-1, D) } = (B, K, D)

            all_x = list()

            for i in range(cur_num_chunks):
                
                history_tokens = fx[:, :i+1, :] # (B, i+1, D)

                cur_x = self.forward_chunk(input_ids[:, i, :], history_tokens * self.config.norm_w)
                all_x.append(cur_x) # (B, l, D)
            
            all_x = torch.cat(all_x, dim=1) # (B, K*l, D)

        if labels is not None:
            if self.config.use_fused_linear_cross_entropy:
                loss = self.fused_linear_cross_entropy(
                    self.output.weight, all_x.view(-1, all_x.size(-1)), labels.view(-1)
                ) # need to reshape to x to (B*N, D) and labels to (B*N)
                return loss
            else:
                logits = self.output(all_x)
                if self.config.dataset == "mqar":
                    logits = logits[:, 1:, :].contiguous()
                    # note that target is already shifted in torch.dataset
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                return loss
        
        logits = self.output(all_x) # (B, L, V)
        if self.config.dataset == "mqar":
            logits = logits[:, 1:, :].contiguous()
            # note that target is already shifted in torch.dataset
        return logits


# https://github.com/rajesh-lab/symile/blob/287f82102313b6be44a8ff2a7d56e5165a3dfb4e/src/losses.py#L49
def clip_loss(X: Tensor, Y: Tensor, logit_scale: Tensor):
    # X: [B1, D], Y: [B2, D]
    # clip loss is being done in the direction X -> Y
    logits = logit_scale * X @ Y.T
    labels = torch.arange(logits.shape[0]).to(logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)

def clip_loss_h(X, Y, logit_scale, h):
    # X: [B1, D], Y: [B2, D]
    # clip loss is being done in the direction X -> Y
    B1, D = X.shape
    B2, _ = Y.shape

    # Expand dimensions for broadcasting
    X_exp = X.unsqueeze(2).expand(B1, D, B2)  # shape: [B1, D, B2]
    Y_exp = Y.T.unsqueeze(0).expand(B1, D, B2)  # shape: [B1, D, B2]

    # Concatenate along the D dimension (axis=1)
    Z = torch.cat([X_exp, Y_exp], dim=1)  # shape: [B, 2D, B]
    Z = Z.transpose(1, 2)  # shape: [B, B, 2D]

    # Apply the projection head
    Z = h(Z)  # shape: [B, B, 1]
    Z = Z.squeeze(-1)  # shape: [B, B]
    logits = logit_scale * Z
    labels = torch.arange(logits.shape[0]).to(logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)



if __name__ == "__main__":
    
    device = "cuda"
    dtype = torch.float16

    scorer_config = ChunkedTransformerConfig()
    # print(scorer_config)
    scorer = Scorer(scorer_config)
    # print(scorer)

    encoder_config = ChunkedTransformerConfig(dim_g=scorer_config.dim)
    # print(encoder_config)
    encoder = ChunkedEncoder(encoder_config, scorer)
    # print(encoder)

    # input_ids = torch.randint(0, 100, (8, encoder_config.block_size))
    # print(input_ids.shape)

    # loss = encoder(None, input_ids)
    # print(loss)

    decoder_config = ChunkedTransformerConfig(single_history=False)
    # print(decoder_config)
    decoder = ChunkedDecoder(decoder_config, encoder)
    decoder.to(device=device, dtype=dtype)

    # freeze the encoder
    print("Freezing encoder parameters...")
    for param in decoder.f.parameters():
        param.requires_grad = False
    # print(decoder)

    input_ids = torch.randint(0, 100, (8, decoder_config.block_size), device=device)
    labels = torch.randint(0, 100, (8, decoder_config.block_size), device=device)
    print(input_ids.shape)
    loss = decoder(None, input_ids, labels)
    print(loss)

    from triton.testing import do_bench
    fwd = lambda: decoder(None, input_ids, labels)
    fwd_time = do_bench(fwd, warmup=5, rep=10)
    print("Fwd time:", fwd_time)

    # compile
    print("Compiling decoder...")
    decoder = torch.compile(decoder, mode="default", fullgraph=True)
    loss = decoder(None, input_ids, labels)
    print(loss)

    fwd = lambda: decoder(None, input_ids, labels)
    fwd_time = do_bench(fwd, warmup=5, rep=10)
    print("Fwd time:", fwd_time)