"""
NIAH (Needle-In-A-Haystack) evaluation for Transformer, CAT_Transformer, and CAT_Transformer_Hybrid.

Uses RULER benchmark JSONL datasets.

Usage:
    python eval/niah.py --model_type cat_transformer_hybrid --chunk_size_power 4
    python eval/niah.py --model_type vanilla --file_name niah_vanilla
    python eval/niah.py --model_type cat_transformer_hybrid --model_path /path/to/ckpt --chunk_size_power 4
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import argparse

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer import (
    TransformerConfig,
    Transformer
)

from cat_transformer_adaptive import (
    CAT_Config,
    CAT_Transformer
)

from cat_transformer_hybrid import (
    HybridCAT_Config,
    CAT_Transformer_Hybrid,
)

@torch.no_grad()
def generate_autoregressive(input_ids, model, num_new_tokens=32, do_sample=False, chunk_size_power=None):
    """Simple autoregressive generation (slow but correct)."""
    cur_input_ids = input_ids.clone()
    for _ in range(num_new_tokens):
        if isinstance(model, (CAT_Transformer, CAT_Transformer_Hybrid)):
            logits = model(cur_input_ids, chunk_size_power=chunk_size_power)
        else:
            logits = model(cur_input_ids)
        logits = logits[:, -1, ...]
        if do_sample:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        cur_input_ids = torch.cat([cur_input_ids, next_token], dim=1)
    return cur_input_ids


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

block_size = 1024

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
        model = Transformer(config)

    elif model_type == "chunked":
        chunk_size = 16

        compressor_config = CAT_Config(
            vocab_size=50257,
            block_size=block_size,
            chunk_size=chunk_size,
            dim=1024,
            n_head=16,
            n_layer=3,
            dim_fx=2048,
            use_qk_norm=True,
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

        model = CAT_Transformer(decoder_config, compressor_config)

    elif model_type == "cat_transformer_hybrid":
        chunk_size = 16

        compressor_config = CAT_Config(
            vocab_size=50257,
            block_size=block_size,
            chunk_size=chunk_size,
            dim=1024,
            n_head=16,
            n_layer=3,
            dim_fx=1536,
            use_qk_norm=True,
        )

        decoder_config = HybridCAT_Config(
            vocab_size=50257,
            block_size=block_size,
            chunk_size=chunk_size,
            dim=1536,
            n_head=24,
            n_layer=12,
            use_qk_norm=True,
            use_fused_ops=True,
            gdn_layers=[1, 3, 5, 7, 9, 11],
            fla_gdn_num_heads=8,
            fla_gdn_head_dim=128,
            fla_gdn_expand_v=2.0,
        )

        model = CAT_Transformer_Hybrid(decoder_config, compressor_config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


if __name__ == "__main__":

    device = "cuda"
    dtype = torch.bfloat16
    MAX_VAL_TOKENS = 100

    model_type_to_path = {
        "vanilla" : "/scratch/jp7467/cat-transformer/Results/test-fineweb-1b/2026-03-06/12:07:04.421670/state_dict.pt",
        "chunked" : "/scratch/jp7467/cat-transformer/Results/test-fineweb-1b/2026-03-06/15:40:25.010151/state_dict.pt",
        "cat_transformer_hybrid" : "/gpfs/data/ranganathlab/Jatin/cat-transformer/Results/fineweb-5b/2026-03-28/01:25:03.783948/intermediate_state_dict_0033750.pt",
    }

    # python eval/niah.py --model_type cat_transformer_hybrid --chunk_size_power 2
    # python eval/niah.py --model_type cat_transformer_hybrid --chunk_size_power 2 --datasets "niah-numbers-1k" --file_name test

    parser = argparse.ArgumentParser(description="NIAH evaluation using RULER benchmark")
    parser.add_argument("--model_type", type=str, required=True, help="Model type: vanilla, chunked, cat_transformer_hybrid")
    parser.add_argument("--model_path", type=str, default=None, help="Override model checkpoint path")
    parser.add_argument("--file_name", type=str, default="test", help="File name to save results")
    parser.add_argument("--chunk_size_power", type=int, default=4, help="Chunk size power (default: 4)")
    parser.add_argument("--dataset_prefix", type=str,
                        default="/gpfs/data/ranganathlab/Jatin/RULER/scripts/data/synthetic/{}/validation.jsonl",
                        help="Dataset path template with {} for dataset name")
    parser.add_argument("--datasets", type=str, default="niah-numbers-1k,niah-numbers-2k,niah-numbers-4k",
                        help="Comma-separated NIAH dataset names")
    parser.add_argument("--num_new_tokens", type=int, default=32, help="Number of tokens to generate (default: 32)")

    args = parser.parse_args()
    model_type = args.model_type
    model_path = args.model_path if args.model_path else model_type_to_path[model_type]
    file_name = args.file_name

    model = get_model(model_type)
    print(model)

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    print(model.load_state_dict(new_state_dict, strict=True))

    model.eval()
    model.to(device=device)
    model.setup_cache(device=device)

    dataset_names = [d.strip() for d in args.datasets.split(",")]

    for dataset_name in dataset_names:

        print("Evaluating on dataset: ", dataset_name)

        acc = list()

        tokenized_text = list()
        in_path = Path(args.dataset_prefix.format(dataset_name))
        assert in_path.exists(), f"Dataset not found: {in_path}"
        with in_path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Tokenizing"):
                obj = json.loads(line)
                ids = tokenizer.encode(obj["input"] + obj["answer_prefix"], add_special_tokens=False)
                obj["input_ids"] = ids
                tokenized_text.append(obj)
        N = len(tokenized_text)

        print("Evaluating on dataset: ", dataset_name)
        print("Model type: ", model_type)
        print("Model path: ", model_path)
        print("Max val tokens: ", MAX_VAL_TOKENS)
        print("Num samples: ", N)
        print()

        bar = tqdm(range(N))
        for i in bar:

            input_ids = torch.tensor(tokenized_text[i]["input_ids"], dtype=torch.long, device=device)
            input_ids = input_ids.unsqueeze(0)
            start_idx = input_ids.shape[1]

            with torch.autocast(device_type=device, dtype=dtype):
                output_ids = generate_autoregressive(
                    input_ids, model,
                    num_new_tokens=args.num_new_tokens,
                    do_sample=False,
                    chunk_size_power=args.chunk_size_power,
                )

            output_ids = output_ids[0, start_idx:]
            answer_span = tokenizer.decode(output_ids.cpu().numpy()).strip()

            cur_acc = 0
            for x in tokenized_text[i]["outputs"]:
                if x in answer_span:
                    cur_acc = 1
                    break

            acc.append(cur_acc)
            bar.set_postfix_str(f"acc: {np.array(acc).mean():.4f}, accepted samples: {len(acc)}, seq length: {input_ids.shape[1]}")

        acc = np.array(acc)
        import pandas as pd

        results = {
            "model_type": model_type,
            "dataset_name": dataset_name,
            "chunk_size_power": args.chunk_size_power,
            "acc": acc.mean(),
            "num_samples": len(acc),
            "num_correct": np.sum(acc),
            "max_val_tokens": MAX_VAL_TOKENS,
            "model_path": model_path,
        }

        print("model_type: ", model_type)
        print("dataset_name: ", dataset_name)
        print("Accuracy: ", np.mean(acc))
        print("total samples: ", len(acc))
        print("correct samples: ", np.sum(acc))
        print("max_val_tokens: ", MAX_VAL_TOKENS)

        print("Dumping results to csv...")

        folder_path = "eval/hybrid_cat"
        os.makedirs(folder_path, exist_ok=True)

        df = pd.DataFrame(results, index=[0])
        file_path = f"{folder_path}/{file_name}.csv"
        write_header = not os.path.exists(file_path)
        df.to_csv(file_path, index=False, mode="a", header=write_header)
