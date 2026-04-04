"""
This script loads a model and generates a few tokens from it.

Download the weights from: https://huggingface.co/collections/bicycleman15/cat-transformer

Sample command:

python generate.py \
--model_type chunked \
--chunk_size_power 3 \
--model_path "/scratch/jp7467/cat-transformer/Results/fineweb-15b/2026-03-20/00:35:07.052056/state_dict.pt" \
--prompt "The meaning of life is"

"""

import os
import sys
import argparse

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformer import TransformerConfig, Transformer
from cat_transformer_adaptive import CAT_Config, CAT_Transformer

block_size = 4096 # context length

def get_model(model_type):
    if model_type == "vanilla":
        config = TransformerConfig(
            vocab_size=50257,
            block_size=block_size,
            dim=1024,
            n_head=16,
            n_layer=12,
            use_qk_norm=True,
            use_fused_ops=True,
        )
        return Transformer(config)

    elif model_type == "chunked":
        chunk_size = 32
        compressor_config = CAT_Config(
            vocab_size=50257,
            block_size=block_size,
            chunk_size=chunk_size,
            dim=1024,
            n_head=16,
            n_layer=3,
            dim_fx=2048,
            use_qk_norm=True,
            # we don't use fused ops here due to no support of vmap in liger-kernels :((
        )
        decoder_config = CAT_Config(
            vocab_size=50257,
            block_size=block_size,
            chunk_size=chunk_size,
            dim=2048,
            n_head=32,
            n_layer=12,
            use_qk_norm=True,
            use_fused_ops=True,
        )
        return CAT_Transformer(decoder_config, compressor_config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


@torch.no_grad()
def generate(input_ids, model, num_new_tokens=100, do_sample=True, temperature=0.8, chunk_size_power=None):
    cur = input_ids.clone()
    for _ in range(num_new_tokens):
        if isinstance(model, CAT_Transformer):
            logits = model(cur, chunk_size_power=chunk_size_power)
        else:
            logits = model(cur)
        logits = logits[:, -1, :]
        if do_sample:
            logits = logits / temperature
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        cur = torch.cat([cur, next_token], dim=1)
    return cur


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["vanilla", "chunked"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--num_tokens", type=int, default=100)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--chunk_size_power", type=int, default=4) # this decides the chunk size to use (in powers of two, so chunk_size_power=3 means we are using chunk size of 2^3=8 in CAT)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = get_model(args.model_type)
    state_dict = torch.load(args.model_path, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    model.setup_cache(device=device)

    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"Prompt: {args.prompt}\n")

    with torch.autocast(device_type=device, dtype=dtype):
        output_ids = generate(
            input_ids, model,
            num_new_tokens=args.num_tokens,
            do_sample=not args.greedy,
            temperature=args.temperature,
            chunk_size_power=args.chunk_size_power,
        )

    print(tokenizer.decode(output_ids[0].cpu().numpy()))
