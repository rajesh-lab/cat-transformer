"""
Training-free block-sparse attention for the vanilla Transformer.

Adapted from the prefill-stage BlockSparseAttention in:
    https://github.com/PiotrNawrot/nano-sparse-attention

Algorithm (per attention layer, per forward pass):
    1. Compute Q, K, V and apply RoPE as usual.
    2. Divide Q and K into chunks of size `chunk_size`.
    3. Average within each chunk to get chunk-level representations.
    4. Score every (query-chunk, key-chunk) pair, apply causal chunk mask.
    5. Force the first chunk (attention sinks) to always be selected.
    6. Keep top-K key-chunks per query-chunk.
    7. Expand to a token-level mask, OR with a local window, AND with causal.
    8. Run manual scaled dot-product attention with the mask.
"""

import math
import types
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def create_block_sparse_mask(
    q: Tensor,
    k: Tensor,
    chunk_size: int,
    top_chunks: int,
) -> Tensor:
    """Build a boolean attention mask using block-sparse chunk selection.

    Args:
        q: (B, H, L, D) query tensor (after RoPE).
        k: (B, H, L, D) key tensor (after RoPE).
        chunk_size: number of tokens per chunk/block.
        top_chunks: how many key-chunks each query-chunk selects.

    Returns:
        Boolean mask of shape (B, H, L, L) where True = attend, False = masked.
    """
    B, H, L, D = q.shape
    device = q.device

    num_chunks = (L + chunk_size - 1) // chunk_size
    padded_len = num_chunks * chunk_size
    pad_amt = padded_len - L

    if pad_amt > 0:
        q_padded = F.pad(q, (0, 0, 0, pad_amt))  # pad seq dim
        k_padded = F.pad(k, (0, 0, 0, pad_amt))
    else:
        q_padded = q
        k_padded = k

    # (B, H, num_chunks, chunk_size, D)
    q_chunks = q_padded.reshape(B, H, num_chunks, chunk_size, D)
    k_chunks = k_padded.reshape(B, H, num_chunks, chunk_size, D)

    # chunk-level representations: mean over tokens in each chunk
    q_repr = q_chunks.mean(dim=-2)  # (B, H, num_chunks, D)
    k_repr = k_chunks.mean(dim=-2)  # (B, H, num_chunks, D)

    # chunk-to-chunk attention scores
    chunk_scores = torch.matmul(q_repr, k_repr.transpose(-2, -1))  # (B, H, nc, nc)

    # causal mask at chunk level: query-chunk i can only attend to key-chunk j <= i
    chunk_causal = torch.triu(
        torch.ones(num_chunks, num_chunks, device=device, dtype=torch.bool), diagonal=1
    )
    chunk_scores.masked_fill_(chunk_causal, float("-inf"))

    # always select the first chunk (attention sinks)
    chunk_scores[..., 0] = float("inf")

    effective_top = min(top_chunks, num_chunks)
    _, top_indices = torch.topk(chunk_scores, effective_top, dim=-1, sorted=False)

    # build chunk-level boolean mask: True = attend
    chunk_mask = torch.zeros(B, H, num_chunks, num_chunks, device=device, dtype=torch.bool)
    chunk_mask.scatter_(-1, top_indices, True)

    # expand to token-level: (B, H, padded_len, padded_len)
    token_mask = chunk_mask.repeat_interleave(chunk_size, dim=-2).repeat_interleave(chunk_size, dim=-1)

    # local window: each token can attend to tokens within chunk_size distance
    positions = torch.arange(padded_len, device=device)
    local_mask = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() < chunk_size  # (pL, pL)
    token_mask = token_mask | local_mask.unsqueeze(0).unsqueeze(0)

    # causal mask at token level
    causal_mask = ~torch.triu(
        torch.ones(padded_len, padded_len, device=device, dtype=torch.bool), diagonal=1
    )
    token_mask = token_mask & causal_mask.unsqueeze(0).unsqueeze(0)

    # trim padding
    if pad_amt > 0:
        token_mask = token_mask[:, :, :L, :L]

    return token_mask


def _block_sparse_forward(
    self,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    is_causal: Optional[bool] = True,
    mask=None,
    input_pos: Optional[Tensor] = None,
) -> Tensor:
    """Drop-in replacement for Attention.forward that applies block-sparse masking."""
    bsz, seqlen, _ = x.shape

    kv_size = self.n_local_heads * self.head_dim
    q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

    q = q.view(bsz, seqlen, self.n_head, self.head_dim)
    k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

    if self.config.use_qk_norm:
        q = self.q_norm(q)
        k = self.k_norm(k)

    q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

    if self.config.use_fused_ops:
        from liger_kernel.transformers import liger_rotary_pos_emb
        q, k = liger_rotary_pos_emb(q, k, cos, sin)
    else:
        from transformer import apply_rope_emb
        q = apply_rope_emb(q, cos, sin, self.rope_n_elem)
        k = apply_rope_emb(k, cos, sin, self.rope_n_elem)

    if self.kv_cache is not None and input_pos is not None:
        k, v = self.kv_cache.update(input_pos, k, v)

    # --- block-sparse path (pure PyTorch, no GQA) ---
    assert self.n_head == self.n_local_heads, (
        "Block-sparse attention does not support GQA (n_head != n_local_heads)"
    )

    bs_mask = create_block_sparse_mask(
        q, k,
        chunk_size=self._block_sparse_chunk_size,
        top_chunks=self._block_sparse_top_chunks,
    )  # (B, H, L, L) bool

    scale = 1.0 / math.sqrt(self.head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)
    scores = scores.masked_fill(~bs_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = attn.masked_fill(~bs_mask, 0.0)  # zero out NaN from all-masked rows
    y = torch.matmul(attn, v)  # (B, H, L, D)

    y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
    y = self.wo(y)
    return y


def apply_block_sparse_attention(
    model: "Transformer",
    chunk_size: int = 64,
    top_chunks: int = 4,
) -> "Transformer":
    """Monkey-patch all Attention layers in a Transformer to use block-sparse attention.

    This is training-free: no new parameters are added, the same weights are used.
    """
    from transformer import Attention

    for module in model.modules():
        if isinstance(module, Attention):
            module._block_sparse_chunk_size = chunk_size
            module._block_sparse_top_chunks = top_chunks
            module.forward = types.MethodType(_block_sparse_forward, module)

    return model
