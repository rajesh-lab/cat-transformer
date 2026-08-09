"""
Books3 tokenization for the 16K long-context finetune.
https://huggingface.co/datasets/Geralt-Targaryen/books3

Only keeps documents with >= MIN_DOC_TOKENS tokens so that every 16384-token
training window lands inside a single book instead of straddling two.

The upstream dataset is 52.8GB across 35 parquet shards, so this streams rather
than materialising the whole thing: we stop after MAX_TOTAL_TOKENS and only pay
for the parquet shards actually consumed.

Batches are encoded with the fast (Rust) tokenizer, which parallelises internally.
An mp.Pool over the streaming iterator deadlocks on teardown when you break out
early, which is exactly the access pattern here.

Shard 0 becomes the val split, the rest are train, matching prepare_sharded_data.py.
Also writes the test.pt that data.get_dataset loads for validation.

$ python prepare_books3.py
"""

import os
import argparse

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ------------------------------------------
OUT_DIR = "datasets/books3-16k-gpt2"
DATASET_NAME = "Geralt-Targaryen/books3"

SHARD_SIZE = 100_000_000       # 100M tokens per shard
USE_EOS_TOKEN = True
TOKENIZER_NAME = "gpt2"

MIN_DOC_TOKENS = 16_000        # only keep docs long enough for a 16K window

# The finetune consumes 1 x 4 GPUs x 16384 x 8 accum x 16000 iters = 1.048B tokens.
# Shard 0 is held out for val, so 1.2B total leaves 1.1B for train: one clean pass.
MAX_TOTAL_TOKENS = 1_200_000_000

# how much of the val shard to keep as test.pt; eval_iters=50 at batch 1 uses far less
TEST_PT_TOKENS = 20_000_000

DOC_BATCH = 32                 # documents per tokenizer call
DTYPE = np.uint16
# ------------------------------------------


def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    # the length warning is irrelevant here: we tokenize raw text, never feed it to gpt2
    tok.model_max_length = int(1e9)
    eos_id = tok.eos_token_id
    if eos_id is None:
        tok.add_special_tokens({"eos_token": "<|endoftext|>"})
        eos_id = tok.eos_token_id
    return tok, int(eos_id)


def encode_batch(tok, eot, texts):
    """Tokenize a batch of documents, returning only those long enough to keep."""
    encoded = tok(texts, add_special_tokens=False)["input_ids"]
    out = []
    for ids in encoded:
        if len(ids) < MIN_DOC_TOKENS:
            out.append(None)
            continue
        arr = np.asarray(([eot] + ids) if USE_EOS_TOKEN else ids, dtype=np.int64)
        assert (0 <= arr).all() and (arr < 2**16).all(), (
            "Token IDs exceed uint16 range. Use a tokenizer with vocab <= 65535 "
            "or set DTYPE=np.uint32."
        )
        out.append(arr.astype(DTYPE))
    return out


def shard_path(out_dir, shard_index):
    split = "val" if shard_index == 0 else "train"
    return os.path.join(out_dir, f"shard_{split}_{shard_index:06d}")


def write_test_pt(out_dir, max_tokens=TEST_PT_TOKENS):
    """data.get_dataset validates from test.pt, which the sharding loop never writes."""
    val_shard = os.path.join(out_dir, "shard_val_000000.npy")
    if not os.path.exists(val_shard):
        raise FileNotFoundError(f"no val shard at {val_shard}")

    tokens = np.load(val_shard)[:max_tokens]
    tensor = torch.from_numpy(tokens.astype(np.int32)).to(torch.long)
    out_path = os.path.join(out_dir, "test.pt")
    torch.save(tensor, out_path)
    print(f"Wrote {out_path}: {tensor.numel():,} tokens")


def main():
    parser = argparse.ArgumentParser(description="Tokenize books3 for 16K finetuning")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    parser.add_argument("--max_total_tokens", type=int, default=MAX_TOTAL_TOKENS)
    parser.add_argument("--test_pt_tokens", type=int, default=TEST_PT_TOKENS)
    parser.add_argument("--test_pt_only", action="store_true",
                        help="Skip tokenization and just rebuild test.pt from the val shard")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.test_pt_only:
        write_test_pt(args.out_dir, args.test_pt_tokens)
        return

    shard_size = args.shard_size
    max_total_tokens = args.max_total_tokens

    tok, eot = build_tokenizer()

    # streaming: the full dataset is 52.8GB but we only need ~1.2B tokens of it
    stream = load_dataset(DATASET_NAME, split="train", streaming=True)
    print(f"Streaming {DATASET_NAME}")
    print(f"Keeping docs with >= {MIN_DOC_TOKENS:,} tokens, up to {max_total_tokens/1e9:.2f}B tokens total.")
    print(f"Shards of {shard_size:,} tokens, writing to {args.out_dir}")

    total_tokens_written = 0
    docs_kept = 0
    docs_skipped = 0

    shard_index = 0
    buffer = np.empty((shard_size,), dtype=DTYPE)
    token_count = 0
    progress = tqdm(total=max_total_tokens, unit="tok", unit_scale=True, desc="tokenizing")

    def flush_shard(n_tokens):
        nonlocal shard_index, total_tokens_written
        np.save(shard_path(args.out_dir, shard_index), buffer[:n_tokens])
        total_tokens_written += n_tokens
        shard_index += 1

    def feed(tokens):
        """Append one document, flushing whole shards as they fill."""
        nonlocal token_count
        offset = 0
        while offset < len(tokens):
            room = shard_size - token_count
            take = min(room, len(tokens) - offset)
            buffer[token_count:token_count + take] = tokens[offset:offset + take]
            token_count += take
            offset += take
            if token_count == shard_size:
                flush_shard(shard_size)
                token_count = 0
                if total_tokens_written >= max_total_tokens:
                    return True
        return False

    done = False
    batch = []
    for doc in stream:
        batch.append(doc["text"])
        if len(batch) < DOC_BATCH:
            continue

        for tokens in encode_batch(tok, eot, batch):
            if tokens is None:
                docs_skipped += 1
                continue
            docs_kept += 1
            progress.update(len(tokens))
            if feed(tokens):
                done = True
                break
        batch = []
        if done:
            break

    # trailing partial batch and partial shard
    if not done:
        for tokens in encode_batch(tok, eot, batch):
            if tokens is None:
                docs_skipped += 1
                continue
            docs_kept += 1
            progress.update(len(tokens))
            if feed(tokens):
                done = True
                break
        if not done and token_count > 0:
            flush_shard(min(token_count, max_total_tokens - total_tokens_written))

    progress.close()

    print(f"\nDocs kept: {docs_kept:,}, docs skipped (<{MIN_DOC_TOKENS:,} tokens): {docs_skipped:,}")
    print(f"Shards written: {shard_index} (shard 0 is val, the rest are train)")
    print(f"Total tokens written: {total_tokens_written:,} ({total_tokens_written/1e9:.3f}B)")
    print(f"Train tokens available: {(total_tokens_written - shard_size):,}")

    write_test_pt(args.out_dir, args.test_pt_tokens)


if __name__ == "__main__":
    main()
