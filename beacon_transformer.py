"""
Activation Beacon: https://openreview.net/forum?id=1eQT9OzfNQ

The sequence is split into chunks of `l` tokens. Each chunk gets `n_beacons` learned
beacon tokens appended at its end. A chunk's raw tokens are visible only inside that
chunk; every later position sees the chunk exclusively through its beacons, whose
per-layer activations act as the compressed memory.

Compared to CAT there is no separate compressor network: compression happens in-place,
layer by layer, inside the decoder stack.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import einops

# import some transformer components
from transformer import (
    TransformerConfig,
    Attention,
    LLaMAMLP,
    LigerSwiGLUMLP,
    RMSNorm,
    _init_weights,
    KVCache,
    get_mask_mod,
    build_rope_cache,
    apply_rope_emb,
    flex_attention_compiled,
)

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
create_block_mask = torch.compile(create_block_mask)

torch._dynamo.config.cache_size_limit = 8

from liger_kernel.transformers import LigerRMSNorm, liger_rotary_pos_emb, LigerFusedLinearCrossEntropyLoss


def get_beacon_mask(chunk_size: int, n_beacons: int = 1):
    """Attention mask for Activation Beacon.

    Same structure as `get_cat_mask`, except the compressed slots sit at the *end* of
    each block instead of the start.
    """
    blk = chunk_size + n_beacons

    def beacon_mask(b, h, q_idx, kv_idx):
        within_block = (q_idx // blk) == (kv_idx // blk)
        is_beacon = (kv_idx % blk) >= chunk_size
        causal_mask = (q_idx >= kv_idx)
        return (within_block | is_beacon) & causal_mask

    return beacon_mask


# some helpers
def power_of_2(x: int) -> int:
    return int(2 ** x)


def power_of_2_exponent(n: int) -> int:
    if n <= 0:
        raise ValueError("Input must be a positive integer.")
    if (n & (n - 1)) != 0:
        raise ValueError(f"{n} is not a power of 2.")
    return int(math.log2(n))


@dataclass
class Beacon_Config(TransformerConfig):

    # largest chunk size, must be a power of two
    chunk_size: int = 16

    # number of beacon tokens appended to every chunk; compression ratio is chunk_size / n_beacons
    n_beacons: int = 1

    # smallest chunk size the model will ever be run with, must be a power of two
    min_chunk_size: int = 4

    # if True, beacons reuse the token Q/K/V/O projections instead of getting their own
    beacon_share_proj: bool = False

    # how RoPE positions are handed out over the interleaved [tokens, beacons] sequence:
    #   "chunk_reset" - positions restart at 0 in every chunk, as in CAT
    #   "compact"     - as in the paper: the beacons a chunk can see are packed into a
    #                   prefix, so a beacon costs one position instead of chunk_size
    rope_position_scheme: str = "chunk_reset"

    def __post_init__(self):
        super().__post_init__()

        assert self.block_size % self.chunk_size == 0
        assert self.min_chunk_size <= self.chunk_size
        assert self.rope_position_scheme in ("chunk_reset", "compact"), \
            f"unknown rope_position_scheme {self.rope_position_scheme!r}"
        self.num_chunks = self.block_size // self.chunk_size


def beacon_position_ids(
    num_chunks: int,
    chunk_size: int,
    n_beacons: int,
    scheme: str,
    device: Optional[torch.device] = None,
) -> Tensor:
    """RoPE position index for every slot of the interleaved sequence.

    The sequence is laid out as `[tok(k,0..l-1), beacon(k,0..r-1)]` for k = 0..K-1.
    Both schemes are prefix-consistent: the first K' chunks of a K-chunk assignment are
    exactly the assignment for a K'-chunk sequence, so short inputs can slice the cache.
    """
    if scheme == "chunk_reset":
        return torch.arange(chunk_size + n_beacons, device=device).repeat(num_chunks)

    # "compact": when the paper encodes chunk k, the k*r beacons it can see are the whole
    # of its KV cache and occupy positions [0, k*r), with the chunk's own slots following.
    # A beacon is given the position it holds inside that prefix, since that is the one
    # every later chunk reads it at. The paper can also give it a second, larger position
    # in the window that creates it; a single forward pass has to pick one.
    offset = (torch.arange(num_chunks, device=device) * n_beacons).view(-1, 1)
    tokens = offset + torch.arange(chunk_size, device=device).view(1, -1)
    beacons = offset + torch.arange(n_beacons, device=device).view(1, -1)
    return torch.cat([tokens, beacons], dim=1).flatten()


def _init_weights_beacon(module: nn.Module, n_layer: int, dim: int) -> None:
    """`_init_weights`, extended to give `wo_beacon` the same residual scaling as `wo`."""
    _init_weights(module, n_layer, dim)

    if isinstance(module, BeaconAttention):
        for name, p in module.named_parameters():
            if name == "wo_beacon.weight":
                nn.init.normal_(p, mean=0.0, std=(1 / math.sqrt(dim) / n_layer))


class BeaconAttention(Attention):
    """Attention where beacon positions get their own Q/K/V/O projections.

    Beacons live at a fixed stride, so routing is a reshape rather than a gather.
    """

    def __init__(self, config: Beacon_Config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)

        self.n_beacons = config.n_beacons

        if config.beacon_share_proj:
            self.wqkv_beacon = None
            self.wo_beacon = None
        else:
            self.wqkv_beacon = nn.Linear(
                config.dim,
                (config.n_head + 2 * config.n_local_heads) * config.head_dim,
                bias=False,
            )
            self.wo_beacon = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)

    @staticmethod
    def _split_proj(w: nn.Linear, w_beacon: Optional[nn.Linear], x: Tensor, blk: int, n_beacons: int) -> Tensor:
        if w_beacon is None:
            return w(x)

        bsz, seqlen, feat = x.shape
        assert seqlen % blk == 0, \
            f"sequence length {seqlen} must be a multiple of the block size {blk} to route beacon projections"

        x = x.view(bsz, seqlen // blk, blk, feat)
        out = torch.cat(
            [
                w(x[:, :, : blk - n_beacons]),
                w_beacon(x[:, :, blk - n_beacons :]),
            ],
            dim=2,
        )
        return out.flatten(1, 2)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        is_causal: Optional[bool] = True,
        mask: Optional[BlockMask] = None,
        input_pos: Optional[Tensor] = None,
        blk: Optional[int] = None,
    ) -> Tensor:

        bsz, seqlen, _ = x.shape
        assert blk is not None, "BeaconAttention needs the interleaved block size"

        kv_size = self.n_local_heads * self.head_dim
        qkv = self._split_proj(self.wqkv, self.wqkv_beacon, x, blk, self.n_beacons)
        q, k, v = qkv.split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.config.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        if self.config.use_fused_ops:
            q, k = liger_rotary_pos_emb(q, k, cos, sin)
        else:
            q = apply_rope_emb(q, cos, sin, self.rope_n_elem)
            k = apply_rope_emb(k, cos, sin, self.rope_n_elem)

        if self.kv_cache is not None and input_pos is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        if mask is None:
            scale = 1.0 / math.sqrt(self.head_dim)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0,
                scale=scale, is_causal=is_causal, enable_gqa=(self.n_head != self.n_local_heads)
            )
        else:
            if input_pos is not None:
                # used during generation only!
                y = flex_attention(q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads))
            else:
                y = flex_attention_compiled(q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads))

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self._split_proj(self.wo, self.wo_beacon, y, blk, self.n_beacons)

        return y


class BeaconBlock(nn.Module):
    def __init__(self, config: Beacon_Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config

        self.attention = BeaconAttention(config, layer_idx)

        if config.use_fused_ops:
            self.feed_forward = LigerSwiGLUMLP(config)
        else:
            self.feed_forward = LLaMAMLP(config)

        if config.use_fused_ops:
            self.ffn_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
            self.attention_norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
            self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        is_causal: Optional[bool] = True,
        mask: Optional[BlockMask] = None,
        input_pos: Optional[Tensor] = None,
        blk: Optional[int] = None,
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), cos, sin, is_causal, mask=mask, input_pos=input_pos, blk=blk)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Beacon_Transformer(nn.Module):
    def __init__(self, config: Beacon_Config) -> None:
        super().__init__()
        self.config = config

        self.power_of_2_exponent = power_of_2_exponent(config.chunk_size)
        self.min_power_of_2_exponent = power_of_2_exponent(config.min_chunk_size)

        self.num_chunks = config.num_chunks
        self.chunk_size = config.chunk_size
        self.block_size = config.block_size
        self.n_beacons = config.n_beacons

        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)
        self.layers = nn.ModuleList(BeaconBlock(config, layer_idx=i) for i in range(config.n_layer))
        self.output = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

        # a single beacon token, repeated wherever a beacon is needed, as in the paper.
        # It is indexed by chunk size power only so the model can tell which compression ratio
        # is active; the paper gets that for free because the ratio changes the beacon count.
        self.beacon_emb = nn.Embedding(self.power_of_2_exponent + 1, config.dim)

        # declare these early
        self.cos = dict()
        self.sin = dict()
        self.cos_gen = None
        self.sin_gen = None

        if config.use_fused_ops:
            self.norm = LigerRMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if config.use_fused_ops:
            self.fused_linear_cross_entropy = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

        # init weights
        self.apply(lambda m: _init_weights_beacon(m, config.n_layer, config.dim))

        self.get_mask_mod = get_mask_mod

    def setup_cache(self, device: torch.device):

        # global cache, kept around for generation
        _cos, _sin = build_rope_cache(
            self.block_size, self.config.rope_n_elem, device=device, base=self.config.rope_base
        )
        self.cos_gen = _cos.clone()
        self.sin_gen = _sin.clone()

        for c in range(self.min_power_of_2_exponent, self.power_of_2_exponent + 1):

            chunk_size = power_of_2(c)
            assert self.block_size % chunk_size == 0
            num_chunks = self.block_size // chunk_size

            pos = beacon_position_ids(
                num_chunks, chunk_size, self.n_beacons,
                self.config.rope_position_scheme, device=device,
            )
            cos, sin = build_rope_cache(
                int(pos.max()) + 1, self.config.rope_n_elem, device=device, base=self.config.rope_base
            )
            self.cos[c] = cos[:, pos, :].clone()
            self.sin[c] = sin[:, pos, :].clone()

            print(f"created cos and sin cache for beacon decoder (chunk size {chunk_size}) ...")
            print(f"rope scheme: {self.config.rope_position_scheme}, max position: {int(pos.max())}")
            print("cos shape:", self.cos[c].shape)
            print("cos dtype:", self.cos[c].dtype)

    # used for generation
    def setup_kv_cache(self, max_batch_size: int, dtype, device: torch.device):
        print("Setting up kv cache ...")
        for block in self.layers:
            block.attention.kv_cache = KVCache(
                max_batch_size,
                (self.num_chunks * self.n_beacons + self.chunk_size),
                self.config.n_local_heads, self.config.head_dim, dtype, device
            )

    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None, chunk_size_power: Optional[int] = None) -> Tensor:
        # input_ids: (B, L)
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        assert chunk_size_power is not None
        cur_chunk_size = power_of_2(chunk_size_power)
        assert cur_chunk_size >= self.config.min_chunk_size, \
            f"chunk size {cur_chunk_size} is below min_chunk_size {self.config.min_chunk_size}"

        # pad to nearest 512 multiple to reduce flex attention recompilations
        pad_multiple = 512
        slice_end = False
        if seqlen % pad_multiple != 0:
            new_seqlen = ((seqlen // pad_multiple) + 1) * pad_multiple
            pad_len = new_seqlen - seqlen
            input_ids = F.pad(input_ids, (0, pad_len), value=0)
            old_seqlen = seqlen
            seqlen = new_seqlen
            slice_end = True

        cur_num_chunks = seqlen // cur_chunk_size
        blk = cur_chunk_size + self.n_beacons

        x = self.wte(input_ids) # (B, L, D)
        x = x.view(bsz, cur_num_chunks, cur_chunk_size, -1) # (B, K, l, D)

        # one beacon token, repeated at the end of every chunk
        beacons = self.beacon_emb(torch.tensor(chunk_size_power, device=device, dtype=torch.long)) # (D)
        beacons = beacons.view(1, 1, 1, self.config.dim) # (1, 1, 1, D)
        beacons = beacons.expand(bsz, cur_num_chunks, self.n_beacons, self.config.dim)

        x = torch.cat([x, beacons], dim=2) # (B, K, l+r, D)
        x = einops.rearrange(x, 'b k n d -> b (k n) d') # (B, K*(l+r), D)

        assert x.shape[1] <= self.cos[chunk_size_power].shape[1], \
            f"interleaved sequence of length {x.shape[1]} exceeds the RoPE cache built for block_size={self.block_size}"
        cos = self.cos[chunk_size_power][:, :x.shape[1], :]
        sin = self.sin[chunk_size_power][:, :x.shape[1], :]

        mask = create_block_mask(
            get_beacon_mask(cur_chunk_size, self.n_beacons),
            B=None, H=None,
            Q_LEN=x.shape[1],
            KV_LEN=x.shape[1],
        )

        for layer in self.layers:
            x = layer(x, cos, sin, mask=mask, blk=blk)
        x = self.norm(x)

        # drop the beacon positions; token positions map 1:1 back onto the input
        x = einops.rearrange(x, 'b (k n) d -> b k n d', k=cur_num_chunks, n=blk) # (B, K, l+r, D)
        x = x[:, :, :cur_chunk_size, :].contiguous() # (B, K, l, D)
        x = einops.rearrange(x, 'b k l d -> b (k l) d') # (B, L, D)

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


def _reference_beacon_mask(num_chunks: int, chunk_size: int, n_beacons: int) -> Tensor:
    """Dense mask built straight from the method description, used to check `get_beacon_mask`."""
    blk = chunk_size + n_beacons
    n = num_chunks * blk
    allowed = torch.zeros(n, n, dtype=torch.bool)

    for q in range(n):
        q_chunk, q_off = divmod(q, blk)
        for kv in range(n):
            kv_chunk, kv_off = divmod(kv, blk)
            if kv > q:
                continue
            if kv_chunk == q_chunk:
                # everything earlier inside my own chunk, raw tokens included
                allowed[q, kv] = True
            elif kv_off >= chunk_size:
                # earlier chunks are only visible through their beacons
                allowed[q, kv] = True

    return allowed


def check_mask(num_chunks: int = 5, chunk_size: int = 4, n_beacons: int = 1) -> None:
    blk = chunk_size + n_beacons
    n = num_chunks * blk

    idx = torch.arange(n)
    q_idx = idx.view(-1, 1).expand(n, n)
    kv_idx = idx.view(1, -1).expand(n, n)

    got = get_beacon_mask(chunk_size, n_beacons)(None, None, q_idx, kv_idx)
    want = _reference_beacon_mask(num_chunks, chunk_size, n_beacons)

    assert torch.equal(got, want), "beacon mask does not match the reference construction"
    print(f"mask check passed (K={num_chunks}, l={chunk_size}, r={n_beacons})")


def check_flex_matches_dense(num_chunks: int, chunk_size: int, n_beacons: int, device) -> None:
    """`create_block_mask` must agree with a dense mask, including when the interleaved
    length is not a multiple of flex attention's 128-wide blocks."""
    n = num_chunks * (chunk_size + n_beacons)
    n_head, head_dim = 2, 32

    q, k, v = (torch.randn(1, n_head, n, head_dim, device=device, dtype=torch.float32) for _ in range(3))

    block_mask = create_block_mask(
        get_beacon_mask(chunk_size, n_beacons), B=None, H=None, Q_LEN=n, KV_LEN=n
    )
    got = flex_attention(q, k, v, block_mask=block_mask)

    dense = _reference_beacon_mask(num_chunks, chunk_size, n_beacons).to(device)
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=dense.view(1, 1, n, n))

    max_diff = (got - want).abs().max().item()
    assert max_diff < 1e-4, f"flex block mask disagrees with the dense mask (max diff {max_diff})"
    print(f"flex-vs-dense check passed (N={n}, N%128={n % 128}, max diff {max_diff:.2e})")


def check_positions(block_size: int, chunk_size: int, n_beacons: int = 1) -> None:
    """Under "compact", every beacon a chunk reads must sit strictly before that chunk."""
    num_chunks = block_size // chunk_size
    pos = beacon_position_ids(num_chunks, chunk_size, n_beacons, "compact")
    blk = chunk_size + n_beacons

    grid = pos.view(num_chunks, blk)
    tokens, beacons = grid[:, :chunk_size], grid[:, chunk_size:]

    # beacons are the whole memory, so their order has to survive
    assert torch.equal(beacons.flatten(), torch.arange(num_chunks * n_beacons)), \
        "compact beacon positions are not a dense increasing prefix"
    # chunk k's tokens come after the beacons of chunks 0..k-1
    assert (tokens[1:, 0] > beacons[:-1, -1]).all(), \
        "a chunk's tokens overlap the beacons it attends to"

    span = int(pos.max()) + 1
    reset_span = chunk_size + n_beacons
    print(
        f"position check passed (L={block_size}, l={chunk_size}, r={n_beacons}): "
        f"compact spans 0..{span - 1} over {num_chunks * blk} slots, "
        f"vs 0..{reset_span - 1} for chunk_reset and 0..{num_chunks * blk - 1} for global"
    )


@torch.enable_grad()
def check_causality(model: "Beacon_Transformer", chunk_size_power: int, seqlen: int, device) -> None:
    """A token's logits must not depend on any later token's embedding."""
    captured = {}

    def hook(module, inputs, output):
        output.retain_grad()
        captured["emb"] = output

    handle = model.wte.register_forward_hook(hook)
    input_ids = torch.randint(0, model.config.vocab_size, (1, seqlen), device=device)
    logits = model(input_ids, chunk_size_power=chunk_size_power)

    probe = seqlen // 2
    logits[0, probe].sum().backward()
    handle.remove()

    grads = captured["emb"].grad[0] # (L, D)
    leak = grads[probe + 1:].abs().max().item()
    seen = grads[: probe + 1].abs().max().item()

    assert leak == 0.0, f"logits at position {probe} leak information from later tokens (max |grad| = {leak})"
    assert seen > 0.0, "logits appear disconnected from earlier tokens, something is wrong"
    print(f"causality check passed (probe position {probe}, max |grad| on future tokens = {leak})")

    model.zero_grad(set_to_none=True)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    check_mask(num_chunks=5, chunk_size=4, n_beacons=1)
    check_mask(num_chunks=3, chunk_size=8, n_beacons=2)

    for l in (4, 8, 16, 32):
        check_positions(block_size=4096, chunk_size=l, n_beacons=1)
    check_positions(block_size=4096, chunk_size=32, n_beacons=2)

    if torch.cuda.is_available():
        # 2048 tokens with chunk size 32 gives 2112 interleaved positions, not a multiple of 128
        check_flex_matches_dense(num_chunks=64, chunk_size=32, n_beacons=1, device=device)
        check_flex_matches_dense(num_chunks=128, chunk_size=16, n_beacons=1, device=device)

    dim = 768
    num_layers = 4
    n_head = 12

    block_size = 2048 # context length
    chunk_size = 32 # largest chunk size

    config = Beacon_Config(
        dim=dim,
        n_head=n_head,
        n_layer=num_layers,
        block_size=block_size,
        chunk_size=chunk_size,
        n_beacons=1,
        rope_position_scheme="compact",
    )
    model = Beacon_Transformer(config)
    model = model.to(device=device)
    model.setup_cache(device=device)

    input_ids = torch.randint(0, config.vocab_size, (4, block_size), device=device)
    print("input_ids shape:", input_ids.shape)

    # choose which chunk size to use for this forward pass
    # must be a power of 2, between min_chunk_size and chunk_size
    cur_chunk_size_power = 4 # corresponds to chunk size of 16 (2^4)

    logits = model(input_ids, chunk_size_power=cur_chunk_size_power)
    print("logits shape:", logits.shape)

    check_causality(model, chunk_size_power=cur_chunk_size_power, seqlen=512, device=device)
