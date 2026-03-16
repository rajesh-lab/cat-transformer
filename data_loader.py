# Inspired from: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
# and some o3 edits
# uses dataset in the folder: /scratch/jp7467/Datasets/fineweb-50b

import os
import numpy as np
import torch

from transformers import AutoTokenizer

def load_tokens(filename):
    npt = np.load(filename).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)

class TokenBatchIterable(torch.utils.data.IterableDataset):
    """Single-worker iterable that yields pre-batched (x, y) for LM."""
    def __init__(self, config, split, process_rank=0, num_processes=1, seed=42):
        assert split in {"train", "test"}
        self.config = config
        self.split = split
        self.rank = process_rank
        self.num_processes = num_processes
        self.seed = seed
        self._epoch = 0

        self.B = config.train.batch_size
        self.T = config.model.block_size + 1 # +1 for next-token prediction
        self.model_name = config.model.name

        data_root = config.dataset.path
        shards = sorted(
            os.path.join(data_root, s)
            for s in os.listdir(data_root)
            if split in s
        )
        assert len(shards) > 0, f"no shards found for split {split}"
        self.shards = shards

        print("########### [shared_data_loader] Loaded shards:", self.shards)

        # stride each step across processes (single worker, so just processes)
        self.stride = self.B * self.T * self.num_processes

        # initialize to the first viable shard for this rank
        self._shard_idx = 0
        self._load_shard(self._shard_idx)

        # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    def _load_shard(self, idx):
        self.tokens = load_tokens(self.shards[idx])

        n = len(self.tokens)
        n_cut = n % (self.B * self.T * self.num_processes)
        if n_cut > 0:
            print(f"######### Rank {self.rank}: cutting off {n_cut} tokens from shard {idx} (original length {n})")
            self.tokens = self.tokens[:-n_cut]
        self.tokens = self.tokens.view(-1, self.T)
        # deterministic shuffle: seed derived from base seed, epoch, and shard index
        # so every rank sees the same permutation for the same shard/epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch * len(self.shards) + idx)
        perm = torch.randperm(self.tokens.size(0), generator=g)
        self.tokens = self.tokens[perm]
        self.tokens = self.tokens.view(-1)  # flatten back

        # starting offset for this process (no workers)
        self.pos = self.B * self.T * self.rank

    def _advance_shard(self):
        start_idx = self._shard_idx
        while True:
            self._shard_idx = (self._shard_idx + 1) % len(self.shards)
            if self._shard_idx == 0:
                self._epoch += 1
            self._load_shard(self._shard_idx)

            # Check if this shard is big enough to produce at least one batch for this rank
            if self.pos + (self.B * self.T) <= len(self.tokens):
                return
            # If we wrapped around and still nothing, bail with a clear error
            if self._shard_idx == start_idx:
                raise RuntimeError(
                    "No shard is large enough for the current (B, T, process_rank, num_processes). "
                    f"Need at least {self.B*self.T*(self.rank+1)+1} tokens per shard."
                )

    def __iter__(self):
        B, T = self.B, self.T
        while True:
            # ensure we have enough tokens for this batch; if not, move to next shard
            if self.pos + (B * T) > len(self.tokens):
                print(f"Rank {self.rank}: advancing from shard {self._shard_idx} (pos {self.pos})")
                self._advance_shard()

            buf = self.tokens[self.pos : self.pos + B * T].clone() # clone for safety
            buf = buf.view(B, T)  # (B, L+1) for next-token prediction

            # different slices according to the model
            # if "chunked" in self.model_name:
            #     x = buf[:, :-1]
            #     y = buf[:, :-1].clone()
            # else:
            x = buf[:, :-1]
            y = buf[:, 1:].clone()

            # advance for next step across processes
            self.pos += self.stride
            yield x, y


class DataLoaderLite(torch.utils.data.DataLoader):
    """
    DataLoader yielding pre-batched (x, y)
    Assumes single-worker (num_workers=0) and that the dataset yields full batches
    Shouldn't be a problem in compute heavy regimes like LM training
    """
    def __init__(self, config, process_rank=0, num_processes=1, split="train", seed=42, **kwargs):
        dataset = TokenBatchIterable(
            config=config,
            split=split,
            process_rank=process_rank,
            num_processes=num_processes,
            seed=seed,
        )
        # dataset already yields full batches; keep batch_size=None
        super().__init__(dataset=dataset, batch_size=None, num_workers=0, drop_last=False, **kwargs)

    def reset(self):
        ds = self.dataset
        ds._epoch = 0
        ds._shard_idx = 0
        ds._load_shard(0)
