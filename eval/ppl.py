# single file eval for perplexity
"""
Perplexity evaluation on PG19 (test split).

Usage:
    python eval/ppl.py --model_type vanilla
    python eval/ppl.py --model_type chunked --chunk_size_power 4
    python eval/ppl.py --model_type chunked --chunk_size_power 3 --output_path eval/results/ppl_cat8.json
"""

import os
import sys
import json
import math
import argparse

import torch
import torch.nn.functional as F
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer import TransformerConfig, Transformer
from cat_transformer_adaptive import CAT_Config, CAT_Transformer

# fill in the correct hyper-parameters
BLOCK_SIZE = 2048
VOCAB_SIZE = 32000
TOKENIZER_NAME = "meta-llama/Llama-2-7b-hf"

# fill in the correct paths
MODEL_TYPE_TO_PATH = {
    "vanilla": "/path/to/model",
    "chunked": "/path/to/model",
}


# fill in the correct hyper-parameters
def get_model(model_type):
    if model_type == "vanilla":
        config = TransformerConfig(
            vocab_size=VOCAB_SIZE,
            block_size=BLOCK_SIZE,
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
            vocab_size=VOCAB_SIZE,
            block_size=BLOCK_SIZE,
            chunk_size=chunk_size,
            dim=1024,
            n_head=16,
            n_layer=3,
            dim_fx=2048,
            use_qk_norm=True,
        )
        decoder_config = CAT_Config(
            vocab_size=VOCAB_SIZE,
            block_size=BLOCK_SIZE,
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


def load_model(model_type, device="cuda"):
    model = get_model(model_type)
    model_path = MODEL_TYPE_TO_PATH[model_type]

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    print(model.load_state_dict(new_state_dict, strict=True))

    model.eval()
    model.to(device=device)
    model.setup_cache(device=device)
    return model


def tokenize_pg19(tokenizer, context_length):
    """Load PG19 test split, take the first context_length tokens from each document."""
    ds = datasets.load_dataset("emozilla/pg19", split="test")

    chunks = []
    skipped = 0
    for example in tqdm(ds, desc="Tokenizing PG19 test"):
        tokens = tokenizer(example["text"], return_attention_mask=False)["input_ids"]
        if len(tokens) < context_length:
            skipped += 1
            continue
        chunks.append(tokens[:context_length])

    chunks = torch.tensor(chunks, dtype=torch.long)
    print(f"PG19 test: {len(chunks)} documents with {context_length} tokens each ({skipped} skipped, too short)")
    return chunks


@torch.no_grad()
def evaluate_perplexity(model, chunks, model_type, chunk_size_power, device, dtype):
    """Compute perplexity over all chunks."""
    total_loss = 0.0
    total_tokens = 0

    for i in tqdm(range(len(chunks)), desc="Evaluating perplexity"):
        input_ids = chunks[i].unsqueeze(0).to(device)  # (1, context_length)

        with torch.autocast(device_type=device, dtype=dtype):
            if isinstance(model, CAT_Transformer):
                logits = model(input_ids, chunk_size_power=chunk_size_power)
            else:
                logits = model(input_ids)

        # shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += shift_labels.numel()

    avg_nll = total_loss / total_tokens
    ppl = math.exp(avg_nll)
    return avg_nll, ppl, total_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate perplexity on PG19 (test split)")
    parser.add_argument("--model_type", type=str, required=True, help="Model type: vanilla, chunked")
    parser.add_argument("--chunk_size_power", type=int, default=4, help="Chunk size power for CAT (default: 4)")
    parser.add_argument("--context_length", type=int, default=2048, help="Context length (default: 2048)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of chunks to evaluate")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16
    context_length = args.context_length
    assert context_length <= BLOCK_SIZE, f"context_length {context_length} exceeds BLOCK_SIZE {BLOCK_SIZE}"

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    chunks = tokenize_pg19(tokenizer, context_length)

    if args.limit is not None:
        chunks = chunks[: args.limit]
        print(f"Limiting to {args.limit} chunks")

    model = load_model(args.model_type, device=device)

    avg_nll, ppl, total_tokens = evaluate_perplexity(
        model, chunks, args.model_type, args.chunk_size_power, device, dtype
    )

    print(f"\n{'=' * 60}")
    print(f"Model type:       {args.model_type}")
    print(f"Chunk size power: {args.chunk_size_power}")
    print(f"Context length:   {context_length}")
    print(f"Total tokens:     {total_tokens:,}")
    print(f"Avg NLL:          {avg_nll:.4f}")
    print(f"Perplexity:       {ppl:.4f}")
    print(f"{'=' * 60}")

    results = {
        "model_type": args.model_type,
        "chunk_size_power": args.chunk_size_power,
        "context_length": context_length,
        "total_tokens": total_tokens,
        "avg_nll": avg_nll,
        "perplexity": ppl,
        "model_path": MODEL_TYPE_TO_PATH[args.model_type],
        "dataset": "emozilla/pg19",
        "split": "test",
    }

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_path}")
