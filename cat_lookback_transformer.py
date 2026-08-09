"""
CAT with a one-chunk raw lookback.

Plain CAT lets a chunk see every earlier chunk only through its compressed vector
f(c). Here the immediately preceding chunk is handed over as raw tokens instead, so
a query sees: the previous chunk uncompressed, everything older compressed, and its
own chunk causally. Each past chunk is therefore represented exactly once.

Giving the previous chunk correct relative positions is impossible if a token appears
once in the sequence: own-chunk keys must sit at distance i-j and previous-chunk keys
at l+i-j, which forces positions to grow by l per chunk and blows up the RoPE range.
So each chunk is written into the sequence twice, once as a window's readout half and
once as the next window's context half. A chunk then holds two different positions and
CAT's per-block position reset survives, at the cost of ~2x sequence length.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import einops

from transformer import (
    TransformerBlock,
    RMSNorm,
    _init_weights,
    KVCache,
    get_mask_mod,
    build_rope_cache,
)

from cat_transformer_adaptive import (
    CAT_Config,
    Compressor,
    get_cat_mask,
    power_of_2,
    power_of_2_exponent,
)

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
create_block_mask = torch.compile(create_block_mask)

torch._dynamo.config.cache_size_limit = 8

from liger_kernel.transformers import LigerRMSNorm, LigerFusedLinearCrossEntropyLoss

# slot roles inside a block of width 2 + 2*chunk_size
FX_SLOT = 0
SEP_SLOT = 1
FIRST_RAW_SLOT = 2


def block_width(chunk_size: int) -> int:
    """Width of one window: the summary slot, the separator, and two raw chunks."""
    return 2 + 2 * chunk_size


class CAT_Lookback_Transformer(nn.Module):
    """CAT whose decoder additionally reads the previous chunk's raw tokens.

    Block `w` holds `[f(c_{w-1}), sep, chunk_w, chunk_{w+1}]` for w = 0 .. K-2, with
    block 0 carrying the conditioning vector in place of a summary. Because the newest
    summary in scope is always of a strictly earlier chunk than either raw chunk in the
    window, causality needs no special-casing.
    """

    def __init__(self, config: CAT_Config, f_config: CAT_Config) -> None:
        super().__init__()
        self.config = config

        self.power_of_2_exponent = power_of_2_exponent(config.chunk_size)

        self.num_chunks = config.num_chunks
        self.chunk_size = config.chunk_size
        self.block_size = config.block_size

        self.f = Compressor(f_config)

        # also tells the decoder which chunk size is active
        self.dummy_fx = nn.Embedding(self.power_of_2_exponent + 1, config.dim)
        self.wte = nn.Embedding(config.padded_vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, layer_idx=i) for i in range(config.n_layer))
        self.output = nn.Linear(config.dim, config.padded_vocab_size, bias=False)

        # separates the chunk representation from the token embeddings
        self.seperator = nn.Embedding(1, config.dim)

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

        self.down_proj = nn.Identity()
        assert self.f.config.dim_fx == self.config.dim, \
            "f.dim_fx (compressed chunk representation size) must be equal to config.dim (decoder hidden size)"

        self.apply(lambda m: _init_weights(m, self.config.n_layer, self.config.dim))
        self.f.apply(lambda m: _init_weights(m, self.f.config.n_layer, self.f.config.dim))

        self.get_mask_mod = get_mask_mod

    def setup_cache(self, device: torch.device):

        self.f.setup_cache(device=device)
        print("power_of_2_exponent:", self.power_of_2_exponent)

        for c in range(1 + self.power_of_2_exponent):

            chunk_size = power_of_2(c)
            assert self.block_size % chunk_size == 0
            num_chunks = self.block_size // chunk_size
            num_blocks = num_chunks - 1
            blk = block_width(chunk_size)

            _cos, _sin = build_rope_cache(
                num_chunks + chunk_size + 2,
                self.config.rope_n_elem,
                device=device,
                base=self.config.rope_base,
            )
            if c == self.power_of_2_exponent:
                # kept around for generation
                self.cos_gen = _cos.clone()
                self.sin_gen = _sin.clone()

            # positions restart at every block boundary, so the previous chunk sits at
            # 2..l+1 and the current one at l+2..2l+1, exactly l apart
            cos, sin = build_rope_cache(
                blk, self.config.rope_n_elem, device=device, base=self.config.rope_base
            )
            cos = einops.repeat(cos, '1 l d -> 1 k l d', k=num_blocks).clone()
            sin = einops.repeat(sin, '1 l d -> 1 k l d', k=num_blocks).clone()

            self.cos[c] = einops.rearrange(cos, '1 k l d -> 1 (k l) d').clone()
            self.sin[c] = einops.rearrange(sin, '1 k l d -> 1 (k l) d').clone()

            print(f"created cos and sin cache for lookback decoder (chunk size {chunk_size}) ...")
            print("cos shape:", self.cos[c].shape)
            print("cos dtype:", self.cos[c].dtype)

    # used for generation
    def setup_kv_cache(self, max_batch_size: int, dtype, device: torch.device):
        print("Setting up kv cache ...")
        for block in self.layers:
            block.attention.kv_cache = KVCache(
                max_batch_size,
                (self.num_chunks + block_width(self.chunk_size)),
                self.config.n_local_heads, self.config.head_dim, dtype, device
            )

    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None, chunk_size_power: Optional[int] = None) -> Tensor:
        # input_ids: (B, L)
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        assert chunk_size_power is not None
        cur_chunk_size = power_of_2(chunk_size_power)

        # pad to nearest 512 multiple to reduce flex attention recompilations
        pad_multiple = 512
        slice_end = False
        if seqlen % pad_multiple != 0:
            new_seqlen = ((seqlen // pad_multiple) + 1) * pad_multiple
            input_ids = F.pad(input_ids, (0, new_seqlen - seqlen), value=0)
            old_seqlen = seqlen
            seqlen = new_seqlen
            slice_end = True

        cur_num_chunks = seqlen // cur_chunk_size
        assert cur_num_chunks >= 2, \
            f"need at least two chunks to form a lookback window, got {cur_num_chunks}"
        num_blocks = cur_num_chunks - 1
        blk = block_width(cur_chunk_size)

        input_ids = input_ids.view(bsz, cur_num_chunks, cur_chunk_size) # (B, K, l)

        # compress all chunks in parallel
        fx = torch.vmap(self.f.compress, in_dims=(1, 0, None), out_dims=1)(
            input_ids, # (B, K, l)
            torch.arange(cur_num_chunks, device=device), # (K)
            torch.tensor(chunk_size_power, device=device, dtype=torch.long),
        ) # (B, K, D_fx)
        fx = self.down_proj(fx) # (B, K, D)

        # the conditioning vector tells the decoder which chunk size is being used
        dummy_fx = self.dummy_fx(torch.tensor([chunk_size_power], device=device, dtype=torch.long)) # (1, D)
        dummy_fx = einops.repeat(dummy_fx, '1 d -> b 1 d', b=bsz) # (B, 1, D)

        # block w carries f(c_{w-1}), so the newest summary a window sees is always of a
        # chunk older than both of its raw chunks
        fx_slots = torch.cat([dummy_fx, fx[:, : num_blocks - 1, :]], dim=1) # (B, W, D)
        fx_slots = einops.rearrange(fx_slots, 'b w d -> b w 1 d') # (B, W, 1, D)

        sep_token = self.seperator(torch.zeros(1, device=device, dtype=torch.long)) # (1, D)
        sep_token = einops.repeat(sep_token, '1 d -> b w 1 d', b=bsz, w=num_blocks) # (B, W, 1, D)

        emb_x = self.wte(input_ids) # (B, K, l, D)

        # chunk k appears as the context half of block k and the readout half of block k-1
        x = torch.cat([fx_slots, sep_token, emb_x[:, :num_blocks], emb_x[:, 1:]], dim=2) # (B, W, 2+2l, D)
        x = einops.rearrange(x, 'b w s d -> b (w s) d') # (B, W*(2+2l), D)

        assert x.shape[1] <= self.cos[chunk_size_power].shape[1], \
            f"windowed sequence of length {x.shape[1]} exceeds the RoPE cache built for block_size={self.block_size}"
        cos = self.cos[chunk_size_power][:, :x.shape[1], :]
        sin = self.sin[chunk_size_power][:, :x.shape[1], :]

        # within-block causal plus a globally visible slot 0 is already the pattern we
        # want once the block is wide enough to hold two raw chunks
        mask = create_block_mask(
            get_cat_mask(blk),
            B=None, H=None,
            Q_LEN=x.shape[1],
            KV_LEN=x.shape[1],
        )

        for layer in self.layers:
            x = layer(x, cos, sin, mask=mask)
        x = self.norm(x)

        # every prediction comes from a real token's hidden state: chunk 0 from the
        # context half of block 0, chunk k from the readout half of block k-1
        x = einops.rearrange(x, 'b (w s) d -> b w s d', w=num_blocks, s=blk) # (B, W, 2+2l, D)
        x_first = x[:, 0, FIRST_RAW_SLOT : FIRST_RAW_SLOT + cur_chunk_size, :] # (B, l, D)
        x_rest = x[:, :, FIRST_RAW_SLOT + cur_chunk_size :, :] # (B, W, l, D)
        x = torch.cat([x_first.unsqueeze(1), x_rest], dim=1) # (B, K, l, D)
        x = einops.rearrange(x, 'b k l d -> b (k l) d') # (B, L, D)

        # if we padded, slice off the extra
        if slice_end:
            x = x[:, :old_seqlen, :].contiguous()

        if labels is not None:
            if self.config.use_fused_ops:
                loss = self.fused_linear_cross_entropy(
                    self.output.weight, x.view(-1, x.size(-1)), labels.view(-1)
                )
                return loss
            else:
                logits = self.output(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                return loss

        logits = self.output(x) # (B, L, D) -> (B, L, V)
        return logits


def _reference_mask(num_blocks: int, chunk_size: int) -> Tensor:
    """Dense mask built straight from the design, used to check `get_cat_mask` at 2+2l.

    A query may read a summary slot from any earlier-or-equal block, and anything
    earlier-or-equal inside its own block. Nothing else.
    """
    blk = block_width(chunk_size)
    n = num_blocks * blk
    allowed = torch.zeros(n, n, dtype=torch.bool)

    for q in range(n):
        q_blk, q_slot = divmod(q, blk)
        for kv in range(n):
            kv_blk, kv_slot = divmod(kv, blk)
            if kv_slot == FX_SLOT and kv_blk <= q_blk:
                allowed[q, kv] = True
            elif kv_blk == q_blk and kv_slot <= q_slot:
                allowed[q, kv] = True

    return allowed


def check_mask(num_blocks: int = 4, chunk_size: int = 4) -> None:
    blk = block_width(chunk_size)
    n = num_blocks * blk

    idx = torch.arange(n)
    q_idx = idx.view(-1, 1).expand(n, n)
    kv_idx = idx.view(1, -1).expand(n, n)

    got = get_cat_mask(blk)(None, None, q_idx, kv_idx)
    want = _reference_mask(num_blocks, chunk_size)

    assert torch.equal(got, want), "get_cat_mask at 2+2l does not match the reference construction"
    print(f"mask check passed (W={num_blocks}, l={chunk_size}, blk={blk})")


def check_visibility(num_chunks: int = 6, chunk_size: int = 4) -> None:
    """The readout copy of a token must see exactly the intended context.

    For token (k, i): all of chunk k-1 raw, its own chunk up to i, and the summaries of
    chunks 0..k-2. Crucially no summary of chunk k-1 or later, which is what keeps the
    duplicated layout causal.
    """
    blk = block_width(chunk_size)
    num_blocks = num_chunks - 1
    allowed = _reference_mask(num_blocks, chunk_size)

    for k in range(num_chunks):
        for i in range(chunk_size):
            # readout slot for token (k, i)
            if k == 0:
                q = 0 * blk + FIRST_RAW_SLOT + i
            else:
                q = (k - 1) * blk + FIRST_RAW_SLOT + chunk_size + i

            raw_seen, fx_seen = set(), set()
            for kv in range(num_blocks * blk):
                if not allowed[q, kv]:
                    continue
                kv_blk, kv_slot = divmod(kv, blk)
                if kv_slot == FX_SLOT:
                    # block 0 holds the conditioning vector, block w holds f(c_{w-1})
                    if kv_blk > 0:
                        fx_seen.add(kv_blk - 1)
                elif kv_slot >= FIRST_RAW_SLOT:
                    off = kv_slot - FIRST_RAW_SLOT
                    chunk = kv_blk + (1 if off >= chunk_size else 0)
                    raw_seen.add((chunk, off % chunk_size))

            want_raw = {(k, j) for j in range(i + 1)}
            if k >= 1:
                want_raw |= {(k - 1, j) for j in range(chunk_size)}
            want_fx = set(range(max(0, k - 1)))

            assert raw_seen == want_raw, f"token ({k},{i}) raw context {sorted(raw_seen)} != {sorted(want_raw)}"
            assert fx_seen == want_fx, f"token ({k},{i}) summary context {sorted(fx_seen)} != {sorted(want_fx)}"

    print(f"visibility check passed (K={num_chunks}, l={chunk_size}): "
          f"previous chunk raw, older chunks compressed, no summary of the current or previous chunk")


def check_pack_unpack(num_chunks: int = 8, chunk_size: int = 4) -> None:
    """Packing tokens into windows then slicing the readout must recover 0..L-1."""
    blk = block_width(chunk_size)
    num_blocks = num_chunks - 1

    tok = torch.arange(num_chunks * chunk_size).view(1, num_chunks, chunk_size, 1)
    fx = torch.full((1, num_blocks, 1, 1), -1)
    sep = torch.full((1, num_blocks, 1, 1), -2)

    x = torch.cat([fx, sep, tok[:, :num_blocks], tok[:, 1:]], dim=2)
    x = einops.rearrange(x, 'b w s d -> b (w s) d')

    x = einops.rearrange(x, 'b (w s) d -> b w s d', w=num_blocks, s=blk)
    x_first = x[:, 0, FIRST_RAW_SLOT : FIRST_RAW_SLOT + chunk_size, :]
    x_rest = x[:, :, FIRST_RAW_SLOT + chunk_size :, :]
    out = torch.cat([x_first.unsqueeze(1), x_rest], dim=1)
    out = einops.rearrange(out, 'b k l d -> b (k l) d').flatten()

    want = torch.arange(num_chunks * chunk_size)
    assert torch.equal(out, want), "readout slicing does not recover the original token order"
    print(f"pack/unpack check passed (K={num_chunks}, l={chunk_size}, L={num_chunks * chunk_size})")


def check_flex_matches_dense(num_blocks: int, chunk_size: int, device) -> None:
    """`create_block_mask` must agree with a dense mask, including when the windowed
    length is not a multiple of flex attention's 128-wide blocks."""
    blk = block_width(chunk_size)
    n = num_blocks * blk
    n_head, head_dim = 2, 32

    q, k, v = (torch.randn(1, n_head, n, head_dim, device=device, dtype=torch.float32) for _ in range(3))

    block_mask = create_block_mask(get_cat_mask(blk), B=None, H=None, Q_LEN=n, KV_LEN=n)
    got = flex_attention(q, k, v, block_mask=block_mask)

    dense = _reference_mask(num_blocks, chunk_size).to(device)
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=dense.view(1, 1, n, n))

    max_diff = (got - want).abs().max().item()
    assert max_diff < 1e-4, f"flex block mask disagrees with the dense mask (max diff {max_diff})"
    print(f"flex-vs-dense check passed (N={n}, N%128={n % 128}, max diff {max_diff:.2e})")


def check_training_step(model: "CAT_Lookback_Transformer", chunk_size_power: int, seqlen: int, device) -> None:
    """One full training step, in bf16 because that is how the trainer runs.

    flex attention's fp32 backward rejects these sequence lengths ("LSE is not correctly
    aligned"), which hits plain CAT the same way, so the forward-only checks above are
    not enough to catch a broken backward.
    """
    input_ids = torch.randint(0, model.config.vocab_size, (2, seqlen), device=device)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        loss = model(input_ids, labels=input_ids, chunk_size_power=chunk_size_power)
    loss.backward()

    grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    for name, g in grads:
        assert torch.isfinite(g).all(), f"non-finite gradient in {name}"

    touched_compressor = any(n.startswith("f.") for n, _ in grads)
    assert touched_compressor, "the compressor received no gradient, the summaries are unused"

    print(f"training step passed (chunk size {power_of_2(chunk_size_power)}, loss {loss.item():.4f}, "
          f"{len(grads)} tensors with finite grads)")
    model.zero_grad(set_to_none=True)


@torch.no_grad()
def check_causality(model: "CAT_Lookback_Transformer", chunk_size_power: int, seqlen: int, device) -> None:
    """Logits at a position must not move when later tokens are replaced.

    Perturbation rather than gradients, because the compressor runs under vmap and its
    embeddings cannot be hooked; rewriting the input covers the decoder and compressor
    paths at once. Chunk boundaries are probed explicitly since the duplicated layout
    makes those the easiest place to get an off-by-one wrong.
    """
    vocab = model.config.vocab_size
    chunk_size = power_of_2(chunk_size_power)

    base_ids = torch.randint(0, vocab, (1, seqlen), device=device)
    base_logits = model(base_ids, chunk_size_power=chunk_size_power)

    probes = sorted({chunk_size - 1, chunk_size, 2 * chunk_size - 1, seqlen // 2, seqlen - 2})
    for probe in probes:
        assert 0 <= probe < seqlen - 1

        pert_ids = base_ids.clone()
        pert_ids[:, probe + 1:] = torch.randint(0, vocab, (1, seqlen - probe - 1), device=device)
        pert_logits = model(pert_ids, chunk_size_power=chunk_size_power)

        leak = (base_logits[:, : probe + 1] - pert_logits[:, : probe + 1]).abs().max().item()
        assert leak < 1e-4, \
            f"logits up to position {probe} moved by {leak} when only later tokens changed"

        # make sure the test has teeth: the rewritten region must actually respond
        moved = (base_logits[:, probe + 1:] - pert_logits[:, probe + 1:]).abs().max().item()
        assert moved > 1e-3, f"logits after position {probe} did not respond to new tokens"

    print(f"causality check passed (probes {probes}, no leak from future tokens)")


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    check_mask(num_blocks=4, chunk_size=4)
    check_mask(num_blocks=3, chunk_size=8)

    check_visibility(num_chunks=6, chunk_size=4)
    check_visibility(num_chunks=5, chunk_size=8)

    check_pack_unpack(num_chunks=8, chunk_size=4)
    check_pack_unpack(num_chunks=17, chunk_size=16)

    if torch.cuda.is_available():
        # 2 chunks of 32 plus overhead gives blk=66, so these lengths are not 128-aligned
        check_flex_matches_dense(num_blocks=31, chunk_size=32, device=device)
        check_flex_matches_dense(num_blocks=63, chunk_size=16, device=device)

    dim = 768
    num_layers = 4
    n_head = 12

    decoder_dim = 2 * dim
    dim_fx = decoder_dim
    n_head_decoder = 2 * n_head

    block_size = 2048 # context length
    chunk_size = 32 # largest chunk size

    compressor_config = CAT_Config(
        dim=dim, n_head=n_head, dim_fx=dim_fx,
        block_size=block_size, chunk_size=chunk_size, n_layer=(num_layers // 4),
    )
    decoder_config = CAT_Config(
        dim=decoder_dim, n_head=n_head_decoder,
        block_size=block_size, chunk_size=chunk_size, n_layer=num_layers,
    )
    model = CAT_Lookback_Transformer(decoder_config, compressor_config)
    model = model.to(device=device)
    model.setup_cache(device=device)

    input_ids = torch.randint(0, decoder_config.vocab_size, (4, block_size), device=device)
    print("input_ids shape:", input_ids.shape)

    # choose which chunk size to use for this forward pass
    # must be a power of 2, and less than or equal to chunk_size
    cur_chunk_size_power = 4 # corresponds to chunk size of 16 (2^4)

    logits = model(input_ids, chunk_size_power=cur_chunk_size_power)
    print("logits shape:", logits.shape)

    check_causality(model, chunk_size_power=cur_chunk_size_power, seqlen=512, device=device)

    if torch.cuda.is_available():
        for power in range(2, power_of_2_exponent(chunk_size) + 1):
            check_training_step(model, chunk_size_power=power, seqlen=1024, device=device)
