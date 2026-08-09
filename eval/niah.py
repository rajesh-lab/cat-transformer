# S-NIAH evaluation (RULER single needle-in-a-haystack).
#
# The datasets are the pre-generated RULER jsonl dumps that live in
# contrastive-generative-models/eval_data, e.g. niah-numbers-4k is S-NIAH-2 at 4K:
# a 7-digit "magic number" hidden inside a Paul Graham essay haystack.
#
# Scoring follows RULER's reference scorer: partial credit for the fraction of
# needles that appear anywhere in the greedy continuation, lower-cased.
#
# python eval/niah.py --model_type beacon --model_path /path/to/state_dict.pt \
#     --chunk_size_power 5 --datasets niah-numbers-4k --output_dir /tmp/niah

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# reuse the architecture table so the eval configs can never drift apart
from recall import get_model, ADAPTIVE_MODELS

DEFAULT_DATASET_ROOT = "contrastive-generative-models/eval_data"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


@torch.no_grad()
def generate_autoregressive(input_ids, model, num_new_tokens=32, chunk_size_power=None):
    """Greedy decoding by re-running the full forward each step.

    Slow, but it sidesteps the per-architecture KV cache paths and is what the
    other recall evals in this repo already do, so numbers stay comparable.
    """
    cur_input_ids = input_ids.clone()
    for _ in range(num_new_tokens):
        if isinstance(model, ADAPTIVE_MODELS):
            logits = model(cur_input_ids, chunk_size_power=chunk_size_power)
        else:
            logits = model(cur_input_ids)
        next_token = torch.argmax(logits[:, -1, ...], dim=-1, keepdim=True)
        cur_input_ids = torch.cat([cur_input_ids, next_token], dim=1)
    return cur_input_ids


def load_examples(dataset_path, max_samples=None):
    examples = []
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in tqdm(file, desc=f"Tokenizing {dataset_path.parent.name}"):
            obj = json.loads(line)
            obj["input_ids"] = tokenizer.encode(
                obj["input"] + obj["answer_prefix"], add_special_tokens=False
            )
            examples.append(obj)
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on S-NIAH")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Model type: vanilla, chunked, chunked_lookback, beacon")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--file_name", type=str, default="niah")
    parser.add_argument("--chunk_size_power", type=int, default=5)
    parser.add_argument("--datasets", type=str, default="niah-numbers-4k",
                        help="Comma-separated dataset folder names under --dataset_root")
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_dir", type=str, default="benchmark_logs_v2/niah")
    parser.add_argument("--rope_position_scheme", type=str, default="chunk_reset",
                        choices=["chunk_reset", "compact"],
                        help="Beacon RoPE scheme, must match the one the checkpoint was trained with")
    parser.add_argument("--dim", type=int, default=2048, help="chunked_lookback decoder width")
    parser.add_argument("--compressor_dim", type=int, default=1024, help="chunked_lookback compressor width")
    parser.add_argument("--block_size", type=int, default=4096, help="context length to evaluate at")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional per-dataset sample cap for smoke tests")

    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    model = get_model(
        args.model_type,
        rope_position_scheme=args.rope_position_scheme,
        dim=args.dim,
        compressor_dim=args.compressor_dim,
        block_size=args.block_size,
    )
    print(model)

    state_dict = torch.load(args.model_path, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    print(model.load_state_dict(state_dict, strict=True))

    model.eval()
    model.to(device=device)
    model.setup_cache(device=device)

    dataset_root = Path(args.dataset_root).expanduser().resolve()

    for dataset_name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        dataset_path = dataset_root / dataset_name / "validation.jsonl"
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Missing evaluation dataset: {dataset_path}")

        examples = load_examples(dataset_path, max_samples=args.max_samples)

        print("Evaluating on dataset: ", dataset_name)
        print("Model type: ", args.model_type)
        print("Model path: ", args.model_path)
        print("Chunk size: ", 2 ** args.chunk_size_power)
        print("Num samples: ", len(examples))
        print()

        scores = []
        num_skipped = 0
        bar = tqdm(examples, desc=f"Evaluating {dataset_name}")
        for example in bar:
            prompt_length = len(example["input_ids"])
            # the models have a hard context limit; a truncated haystack would
            # silently drop the needle, so skip rather than report a wrong 0
            if prompt_length + args.max_new_tokens > args.block_size:
                num_skipped += 1
                continue

            input_ids = torch.tensor(
                example["input_ids"], dtype=torch.long, device=device
            ).unsqueeze(0)

            with torch.autocast(device_type=device, dtype=dtype):
                output_ids = generate_autoregressive(
                    input_ids, model,
                    num_new_tokens=args.max_new_tokens,
                    chunk_size_power=args.chunk_size_power,
                )

            prediction = tokenizer.decode(
                output_ids[0, prompt_length:].cpu().numpy()
            ).strip().lower()

            needles = [answer.strip().lower() for answer in example["outputs"]]
            score = sum(needle in prediction for needle in needles) / len(needles) if needles else 0.0
            scores.append(score)
            bar.set_postfix_str(
                f"acc: {np.mean(scores):.4f}, scored: {len(scores)}, skipped: {num_skipped}"
            )

        scores = np.asarray(scores, dtype=np.float32)

        results = {
            "model_type": args.model_type,
            "chunk_size_power": args.chunk_size_power,
            "dataset_name": dataset_name,
            "acc": float(scores.mean()) if len(scores) else float("nan"),
            "num_samples": len(scores),
            "num_correct": float(scores.sum()),
            "num_skipped": num_skipped,
            "max_new_tokens": args.max_new_tokens,
            "block_size": args.block_size,
            "model_path": args.model_path,
        }
        print(json.dumps(results, indent=2))

        os.makedirs(args.output_dir, exist_ok=True)
        file_path = os.path.join(args.output_dir, f"{args.file_name}.csv")
        pd.DataFrame([results]).to_csv(
            file_path, index=False, mode="a", header=not os.path.exists(file_path)
        )
        print("Wrote", file_path)
