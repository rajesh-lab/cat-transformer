"""
This script loads a model and generates a few tokens from it.

Download the weights from: https://huggingface.co/collections/bicycleman15/cat-transformer

Sample command:

python generate.py \
--model_type chunked \
--chunk_size_power 3 \
--model_path "/path/to/model" \
--prompt "The meaning of life is"

model_type is one of: vanilla, chunked (CAT), beacon (Activation Beacon).

"""

import os
import sys
import argparse

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformer import TransformerConfig, Transformer
from cat_transformer_adaptive import CAT_Config, CAT_Transformer
from beacon_transformer import Beacon_Config, Beacon_Transformer
from cat_lookback_transformer import CAT_Lookback_Transformer

# models that take a chunk_size_power at forward time
ADAPTIVE_MODELS = (CAT_Transformer, CAT_Lookback_Transformer, Beacon_Transformer)

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

    elif model_type == "chunked_lookback":
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
        return CAT_Lookback_Transformer(decoder_config, compressor_config)

    elif model_type == "beacon":
        config = Beacon_Config(
            vocab_size=50257,
            block_size=block_size,
            chunk_size=32,
            n_beacons=1,
            # must match the checkpoint being loaded
            rope_position_scheme="chunk_reset",
            dim=1024,
            n_head=16,
            n_layer=12,
            use_qk_norm=False,
            use_fused_ops=True,
        )
        return Beacon_Transformer(config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


@torch.no_grad()
def generate(input_ids, model, num_new_tokens=100, do_sample=True, temperature=0.8, chunk_size_power=None):
    cur = input_ids.clone()
    for _ in range(num_new_tokens):
        if isinstance(model, ADAPTIVE_MODELS):
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
    parser.add_argument("--model_type", type=str, required=True, choices=["vanilla", "chunked", "chunked_lookback", "beacon"])
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
