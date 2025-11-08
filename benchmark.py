"""
This file contains code for generation benchmarking for CAT Transformer.

Inspired from: github.com/meta-pytorch/gpt-fast
"""
import os
import argparse
import torch
from torch import Tensor

from typing import Optional

import einops
from tqdm import trange
import pandas as pd

from transformers import AutoTokenizer

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
create_block_mask = torch.compile(create_block_mask)

torch._inductor.config.fx_graph_cache = True 
torch._functorch.config.enable_autograd_cache = True

from transformer import (
    TransformerConfig,
    Transformer
)

from cat_transformer import (
    CAT_Config,
    CAT_Transformer
)

# slow autoregressive generation for correctness checking
# or can be used for accuracy benchmarks
@torch.no_grad()
def generate_autoregressive_slow(
    input_ids: Tensor, model: CAT_Transformer, num_new_tokens: int
):
    cur_input_ids = input_ids.clone()
    for _ in range(num_new_tokens):
        logits = model(cur_input_ids) # [B, l, V]
        
        logits = logits[:, -1, ...] # [B, V]
        next_token = torch.argmax(logits, dim=-1, keepdim=True) # [B, 1]

        cur_input_ids = torch.cat([cur_input_ids, next_token.clone()], dim=1) # [B, l+1]

    return cur_input_ids


def causal_mask(b, h, q, kv):
    return q >= kv

def prefill_fx(model: CAT_Transformer, fx, input_pos, rope_pos, block_mask):
    assert input_pos.shape[-1] == 1

    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, (model.config.num_chunks + model.config.chunk_size))
    # print("input pos for fx:", input_pos.shape)
    # [B, l, V]
    # logits = model(None, input_ids=input_ids, input_pos=input_pos, mask=mask)
    logits = model.forward_embeddings(fx, input_pos=input_pos, rope_pos=rope_pos, mask=mask)
    # [B, V]
    logits = logits[:, -1, ...]
    # [B, 1]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    # breakpoint()
    return next_token

# we change this slightly
def decode_one_token(model: CAT_Transformer, x: torch.Tensor, input_pos: torch.Tensor, rope_pos: torch.Tensor, block_mask: BlockMask):
    # input_pos: [B, 1]
    # print("input pos:", input_pos.shape)
    assert input_pos.shape[-1] == 1

    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, (model.config.num_chunks + model.config.chunk_size))

    logits = model.forward_embeddings(x, input_pos=input_pos, rope_pos=rope_pos, mask=mask, is_input_token=True)
    logits = logits[:, -1, ...]
    # return sample(logits, **sampling_kwargs)
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    # breakpoint()
    return next_token

# we change this slightly
def decode_n_tokens(model: CAT_Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, rope_pos: torch.Tensor, num_new_tokens: int, block_mask):
    # new_tokens, new_probs = [], []
    new_tokens = []
    for i in range(num_new_tokens):
        next_token = decode_one_token(
            model, cur_token, input_pos, rope_pos, block_mask, # **sampling_kwargs
        )
        input_pos += 1
        rope_pos += 1
        new_tokens.append(next_token.clone())
        # callback(new_tokens[-1])
        # new_probs.append(next_prob.clone())
        cur_token = next_token.clone()

    return new_tokens

def generate_chunk(model: CAT_Transformer, fx, input_pos, rope_pos, block_mask):
    # fx: (B, 1, D)
    # chunk_idx: [1]

    next_token = prefill_fx(model, fx, input_pos, rope_pos, block_mask).clone() # (B, 1)

    input_pos += 1
    rope_pos += 1
    
    generated_tokens = decode_n_tokens(model, next_token, input_pos, rope_pos, model.config.chunk_size - 1, block_mask)
    generated_tokens = torch.cat([next_token] + generated_tokens, dim=-1) # (B, k)

    return generated_tokens
    # then decode_n_tokens, or actually n-1 tokens

# https://github.com/pytorch-labs/gpt-fast/blob/7dd5661e2adf2edd6a1042a2732dcd3a94064ad8/generate.py#L154
@torch.no_grad()
def generate_chunk_by_chunk(
    input_ids: Tensor, model: CAT_Transformer, # num_new_tokens: int
):
    bsz = input_ids.shape[0]
    assert input_ids.shape[1] % model.config.chunk_size == 0

    block_mask = create_block_mask(
        causal_mask, # i think this is okay
        B=None, H=None, 
        Q_LEN=(model.config.num_chunks + model.config.chunk_size),
        KV_LEN=(model.config.num_chunks + model.config.chunk_size),
        device=input_ids.device
    )

    # get the fx
    # do a prefill for dummy embedding
    input_pos = torch.tensor([0], device=input_ids.device, dtype=torch.int)
    rope_pos = torch.tensor([0], device=input_ids.device, dtype=torch.int)
    fx = model.dummy_fx(torch.zeros(1, device=input_ids.device, dtype=torch.int)) # [1, D]
    fx = einops.repeat(fx, "1 d -> b 1 d", b=bsz) # [B, 1, D]
    prefill_fx(model, fx, input_pos, rope_pos, block_mask) # [B, 1]

    input_pos += 1
    chunk_pos = torch.tensor([0], device=input_ids.device, dtype=torch.int)
    fx = model.f.compress(input_ids, chunk_pos.squeeze(0)).unsqueeze(1) # (B, l, D), [1] -> (B, D) -> (B, 1, D)

    new_chunks = list()
    
    for i in range(model.config.num_chunks - 1):
        
        rope_pos = torch.tensor([0], device=input_ids.device, dtype=torch.int)
        next_chunk = generate_chunk(model, fx, input_pos.clone(), rope_pos.clone(), block_mask)
        new_chunks.append(next_chunk.clone())

        # update the fx
        input_pos += 1
        chunk_pos += 1
        next_fx = model.f.compress(next_chunk, chunk_pos.squeeze(0)).unsqueeze(1) # (B, l, D), [1] -> (B, 1, D)
        fx = next_fx

    new_chunks = torch.cat(new_chunks, dim=1).view(bsz, -1) # (B, k * (num_chunks - 1))
    return new_chunks


def benchmark():

    dtype = torch.bfloat16
    device = "cuda:0"
    do_compile = True
    warmup_iters = 3
    repetitions = 3
    benchmark_name = "temp"

    batch_size = 320

    block_size = 2048 # context length
    chunk_size = 8
    
    n_layer = 12
    dim = 768
    n_head = 12

    # decoder config, according to the paper
    # but can be changed
    dim_fx = 2 * dim
    decoder_n_head = 2 * n_head
    
    compressor_config = CAT_Config(
        block_size=block_size,
        chunk_size=chunk_size,

        n_layer=(n_layer // 4), # according to the paper, but can be changed
        dim=dim,
        n_head=n_head,

        dim_fx=dim_fx,
    )
    decoder_config = CAT_Config(
        block_size=block_size,
        chunk_size=chunk_size,

        n_layer=n_layer,
        dim=dim_fx,
        n_head=decoder_n_head,
    )
    model = CAT_Transformer(decoder_config, compressor_config)
    print(model)

    model.to(device=device, dtype=dtype)
    model.setup_cache(device=device)

    # # Optionally load model weights here
    # state_dict = "path/to/state_dict.pt"
    # state_dict = torch.load(state_dict, map_location="cpu")
    # print(model.load_state_dict(state_dict, strict=True))

    model.setup_kv_cache(max_batch_size=batch_size, device=device, dtype=dtype)
    model.eval()

    if do_compile:
        print("Compiling...")
        global decode_one_token
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        global prefill_fx
        prefill_fx = torch.compile(prefill_fx, mode="reduce-overhead", fullgraph=True)
    else:
        print("Not compiling...")

    print(f"Benchmarking... bs-{batch_size} chunk_size-{chunk_size} block_size-{block_size}")

    # warm up
    for _ in trange(warmup_iters, desc="Warmup..."):
        input_ids = torch.randint(0, 1000, (batch_size, chunk_size), device=device)

        # # Alternatively, use custom text input
        # text = input("Enter text to generate: ")
        # input_ids = tokenizer(text, return_tensors="pt").input_ids
        # input_ids = input_ids.to(device=device) # (1, l)
        # input_ids = input_ids[:, :chunk_size]
        # print("Input text is:", tokenizer.decode(input_ids[0]))

        output_ids = generate_chunk_by_chunk(input_ids.clone(), model)

        # # and then, decode output ids to view the generated text
        # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # print("Output text is:", output_text)

        # # Optionally: Correctness check with slow generation
        # # Note that due to differences in compiled kernels in flex attention and pytorch's provided sdpa, do this test in float32 only
        # # also, turn off compile (CUDA graphs) for correctness check
        # # use batch size 1, reduce model size since slow generation is really slow due to repeated flex attention compilations
        ### Corretness check ###
        # expected_output_ids = generate_autoregressive_slow(input_ids.clone(), model, num_new_tokens=block_size - chunk_size)
        # expected_output_ids = expected_output_ids[:, chunk_size:]
        # if not torch.all(output_ids == expected_output_ids):
        #     print("❌ Check failed.")
        # else:
        #     print("✅ Check passed.")
        ########################

    torch.cuda.synchronize()

    latency = list()
    mem_taken = list()

    for iter in trange(repetitions, desc="Benchmarking..."):
        torch.cuda.reset_peak_memory_stats()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        input_ids = torch.randint(0, 1000, (batch_size, chunk_size), device=device)
        
        start_event.record()
        generate_chunk_by_chunk(input_ids, model)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        peak_mem = torch.cuda.max_memory_allocated(device=device) / 1024 / 1024 / 1024 # in GB
        latency.append(elapsed_time_ms)
        mem_taken.append(peak_mem) # in MB

        print(f"Benchmark {iter}/{repetitions}: batch_size={batch_size}, chunk_size={chunk_size}, block_size={block_size}, dim={dim}, n_layer={n_layer}, device={device}, dtype={dtype}")
        print(f"Latency: {elapsed_time_ms:.2f} ms, Peak Memory: {peak_mem:.2f} GB")
        print()
    print("\n\n")

    # take mean
    latency = torch.tensor(latency).mean().item()
    mem_taken = torch.tensor(mem_taken).mean().item()

    # create a dict
    result = {
        "latency": latency,
        "mem_taken": mem_taken,
        "batch_size": batch_size,
        "block_size": block_size,
        "chunk_size": chunk_size,
        "dim": dim,
        "n_layer": n_layer,
        "device": device,
        "dtype": dtype,
    }

    # convert to dataframe
    df = pd.DataFrame([result])
    file_path = f"{benchmark_name}.csv"
    write_header = not os.path.exists(file_path)
    df.to_csv(file_path, index=False, mode="a", header=write_header)

    print("Benchmarking done!")

if __name__ == "__main__":
    # torch.set_float32_matmul_precision('high')
    benchmark()
    