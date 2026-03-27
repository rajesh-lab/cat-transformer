import os
import torch

from data_loader import DataLoaderLite

class RandomBatchDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, context_length):

        # deliberately convert to long just to be sure
        self.input_ids = input_ids.to(torch.long)

        self.context_length = context_length
        self.context_length_plus_one = context_length + 1

        # trim off extra tokens from the end if context_length_plus_one does not evenly divide the input_ids length
        self.input_ids = self.input_ids[: (len(self.input_ids) // self.context_length_plus_one) * self.context_length_plus_one]

        # reshape the input_ids to be of shape (N, context_length_plus_one)
        self.input_ids = self.input_ids.view(-1, self.context_length_plus_one)

        # print stats for the dataset
        print("**** Total tokens in the dataset:", self.input_ids.shape[0]*self.input_ids.shape[1], "****")

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        total_tokens = self.input_ids[idx, :]
        train_tokens = total_tokens[:-1] # omit the last token
        target_tokens = total_tokens[1:] # shift left by one
        return train_tokens, target_tokens


# inspired from: https://github.com/HazyResearch/zoology/blob/c42ae3370b9b13a04a23c5f9f4d967469ecb8958/zoology/data/utils.py#L126
class MQARDataset(torch.utils.data.Dataset):
    """Pre-batched MQAR dataset. Each segment corresponds to a different num_kv_pairs."""
    def __init__(self, cfg, split="train"):
        self.model_name = cfg.model.name
        self.segments = torch.load(os.path.join(cfg.dataset.path, f"{split}.pt"), weights_only=False)
        self.batch_size = cfg.train.batch_size
        self.batches = [
            (segment_idx, batch_start)
            for segment_idx, segment in enumerate(self.segments)
            for batch_start in range(0, len(segment[0][0]), self.batch_size)
        ]
        num_tokens = sum(x[0][0].shape[0] * x[0][0].shape[1] for x in self.segments)

        print("Loading MQAR dataset from:", cfg.dataset.path)
        print(f"~~~~~~ Total tokens in the dataset: {num_tokens:,} ~~~~~~")

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        segment_idx, batch_start = self.batches[idx]
        config = self.segments[segment_idx][1]
        input_ids, labels, _examples = self.segments[segment_idx][0]

        slc = slice(batch_start, batch_start + self.batch_size)
        input_ids = input_ids[slc]
        labels = labels[slc]

        return input_ids, labels, config


def get_dataset(cfg, process_rank=0, num_processes=1):
    if cfg.dataset.name == "mqar":
        train_dataset = MQARDataset(cfg, split="train")
        test_dataset = MQARDataset(cfg, split="test")
        return train_dataset, test_dataset, dict(
            vocab_size=cfg.dataset.vocab_size,
            tokenizer_name=getattr(cfg.dataset, "tokenizer_name", None),
        )

    train_dataset = DataLoaderLite(
        cfg, split="train", process_rank=process_rank, num_processes=num_processes, seed=cfg.seed,
    )
    test_dataset = RandomBatchDataset(
        torch.load(os.path.join(cfg.dataset.path, "test.pt"), weights_only=False), cfg.model.block_size
    )
    return train_dataset, test_dataset, dict(vocab_size=cfg.dataset.vocab_size, tokenizer_name=cfg.dataset.tokenizer_name)