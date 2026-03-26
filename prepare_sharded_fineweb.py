"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python prepare_sharded_fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm
from transformers import AutoTokenizer  # pip install transformers

# ------------------------------------------
local_dir = "/gpfs/data/ranganathlab/Jatin/Datasets/new-fineweb-5b-gpt2"
dataset_name = "HuggingFaceFW/fineweb-edu"
remote_name = "sample-10BT"
shard_size = 100_000_000  # 500M tokens per shard
use_eos_token = True # append EOS token before each document
TOKENIZER_NAME = "gpt2"
# TOKENIZER_NAME = "meta-llama/Llama-2-7b-hf"

# local_dir = "/gpfs/data/ranganathlab/Jatin/Datasets/wikitext-sharded-fixed"
# dataset_name = "wikitext"
# remote_name = "wikitext-103-raw-v1"
# shard_size = 50_000_000  # 50M tokens per shard, total of ~3 shards
# use_eos_token = False
# TOKENIZER_NAME = "meta-llama/Llama-2-7b-hf"
# import re
# def wt_detokenizer(string):
#     # contractions
#     string = string.replace("s '", "s'")
#     string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
#     # number separators
#     string = string.replace(" @-@ ", "-")
#     string = string.replace(" @,@ ", ",")
#     string = string.replace(" @.@ ", ".")
#     # punctuation
#     string = string.replace(" : ", ": ")
#     string = string.replace(" ; ", "; ")
#     string = string.replace(" . ", ". ")
#     string = string.replace(" ! ", "! ")
#     string = string.replace(" ? ", "? ")
#     string = string.replace(" , ", ", ")
#     # double brackets
#     string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
#     string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
#     string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
#     string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
#     string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
#     # miscellaneous
#     string = string.replace("= = = =", "====")
#     string = string.replace("= = =", "===")
#     string = string.replace("= =", "==")
#     string = string.replace(" " + chr(176) + " ", chr(176))
#     string = string.replace(" \n", "\n")
#     string = string.replace("\n ", "\n")
#     string = string.replace(" N ", " 1 ")
#     string = string.replace(" 's", "'s")
#     return string

# Choose the HF tokenizer (default: GPT-2, 50k vocab -> fits uint16)
DTYPE = np.uint16  # if you use a tokenizer with vocab > 65535, switch to np.uint32

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset(dataset_name, name=remote_name, split="train")
fw = fw.train_test_split(test_size=0.01, shuffle=False, seed=42)

fw = fw["train"]  # use train split for now
fw = fw.shard(num_shards=2, index=0)  # first ~5B tokens

print(fw)

# --- HF tokenizer (lazy-init per process so mp.Pool works cleanly) ---
_TOKENIZER = None
_EOT_ID = None

def _get_tokenizer():
    """Create (once per process) and return the fast HF tokenizer + eos id."""
    global _TOKENIZER, _EOT_ID
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
        # Prefer eos_token_id; if missing, try common alternatives or add one.
        eos_id = _TOKENIZER.eos_token_id
        if eos_id is None:
            # Try common special token names if eos not set
            for tok in ("</s>", "<|endoftext|>", "<eos>"):
                tok_id = _TOKENIZER.convert_tokens_to_ids(tok)
                if tok_id != _TOKENIZER.unk_token_id and tok_id is not None and tok_id != -1:
                    eos_id = tok_id
                    break
        if eos_id is None:
            # As a last resort, add an EOS token
            _TOKENIZER.add_special_tokens({"eos_token": "<|eos|>"})
            eos_id = _TOKENIZER.eos_token_id
        _EOT_ID = int(eos_id)
    return _TOKENIZER, _EOT_ID

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of tokens
    tok, eot = _get_tokenizer()
    ids = tok.encode(doc["text"], add_special_tokens=False)
    # ids = tok.encode(wt_detokenizer(doc["text"]), add_special_tokens=False)

    if use_eos_token:
        tokens = [eot]
    else:
        tokens = []
    tokens.extend(ids)
    tokens_np = np.asarray(tokens, dtype=DTYPE)

    # Safety check if sticking with uint16
    if DTYPE == np.uint16:
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
            "Token IDs exceed uint16 range. Use a tokenizer with vocab <= 65535 "
            "or set DTYPE=np.uint32."
        )
    return tokens_np

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 10) # leave some CPUs free
print(f"Using {nprocs} processes to tokenize and write shards of {shard_size} tokens each.")
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=DTYPE)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one

            split = "val" if shard_index == 0 else "train"
            # always train for now
            # split = "train"

            filename = os.path.join(DATA_CACHE_DIR, f"shard_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            leftover = len(tokens) - remainder
            if leftover > 0:
                all_tokens_np[0:leftover] = tokens[remainder:]
            token_count = leftover

    # write any remaining tokens as the last shard
    if token_count != 0:

        split = "val" if shard_index == 0 else "train"
        # always train for now
        # split = "train"

        filename = os.path.join(DATA_CACHE_DIR, f"shard_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
