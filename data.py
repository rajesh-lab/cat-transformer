import os
import torch

from data_loader import DataLoaderLite, load_tokens

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


def get_dataset(cfg, process_rank=0, num_processes=1):
    train_dataset = DataLoaderLite(
        cfg, split="train", process_rank=process_rank, num_processes=num_processes, seed=cfg.seed,
    )
    test_dataset = RandomBatchDataset(
        load_tokens(os.path.join(cfg.dataset.path, "shard_val_000000.npy")), cfg.model.block_size
    )
    return train_dataset, test_dataset, dict(vocab_size=cfg.dataset.vocab_size, tokenizer_name=cfg.dataset.tokenizer_name)