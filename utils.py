# most things here are taken from: https://github.com/Lightning-AI/litgpt/blob/main/litgpt/utils.py
from datetime import datetime
import math
import random
import os
import numpy as np

from typing import Any, Iterable, List, Optional, Union
from typing_extensions import Self
from functools import partial

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.distributed as dist

from accelerate import Accelerator

from transformer import (
    TransformerConfig,
    Transformer,
)

from cat_transformer_adaptive import (
    CAT_Config,
    CAT_Transformer
)


def get_model(accelerate: Accelerator, cfg):
    # pass hyperparameters from the yaml config file to the transformer config

    if "transformer" == cfg.model.name:

        config = TransformerConfig(
            vocab_size=cfg.dataset.vocab_size,
            block_size=cfg.model.block_size,
            n_layer=cfg.model.n_layer,
            dim=cfg.model.dim,
            n_head=cfg.model.n_head,
            norm_eps=cfg.train.norm_eps,

            use_fused_ops=cfg.model.use_fused_ops,
            use_qk_norm=cfg.model.use_qk_norm,
        )
        model = Transformer(config)
        accelerate.print("Transformer config:", config)
        accelerate.print(model)
        return model

    elif "cat_transformer" == cfg.model.name:

        compressor_config = CAT_Config(

            vocab_size=cfg.dataset.vocab_size,
            block_size=cfg.model.block_size, 

            use_fused_ops=cfg.model.use_fused_ops,
            use_qk_norm=cfg.model.use_qk_norm,

            chunk_size=cfg.model.chunk_size, 
            dim=cfg.model.compressor_dim, 
            n_head=cfg.model.compressor_n_head, 
            dim_fx=cfg.model.dim_fx,  
            
            n_layer=cfg.model.compressor_num_layers,
        ) # layers are defined according to the paper, but one may use lower number of layers in the compressor

        decoder_config = CAT_Config(

            vocab_size=cfg.dataset.vocab_size,
            block_size=cfg.model.block_size, 

            use_fused_ops=cfg.model.use_fused_ops,
            use_qk_norm=cfg.model.use_qk_norm,

            chunk_size=cfg.model.chunk_size, 
            dim=cfg.model.dim, 
            n_head=cfg.model.n_head,
            n_layer=num_layers
        )

        model = CAT_Transformer(decoder_config, compressor_config)

        accelerate.print("CAT compressor config:", compressor_config)
        accelerate.print("CAT decoder config:", decoder_config)
        accelerate.print(model)
        return model
    
    else:
        raise ValueError(f"Unknown model type: {cfg['name']}")


# https://github.com/Lightning-AI/litgpt/blob/main/litgpt/pretrain.py#L384
@torch.no_grad()
def validate(accelerate: Accelerator, model: nn.Module, val_dataloader: torch.utils.data.DataLoader, cfg) -> torch.Tensor:
    
    print("Validating ...")
    model.eval()

    max_iters = cfg.eval.eval_iters

    losses = []
    val_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Evaluating", disable=(not accelerate.is_main_process))
    for k, batch in val_bar:
        input_ids, targets = batch

        if k >= max_iters:
            break

        input_ids, targets = input_ids.to(accelerate.device), targets.to(accelerate.device)
        
        with accelerate.autocast():
            loss = model(input_ids, targets)

        losses.append(loss)
        val_bar.set_postfix_str(f"val loss: {(sum(losses) / len(losses)):.4f}")

    val_loss = torch.stack(losses).mean()
    model.train()
    return val_loss


# taken from: https://github.com/Lightning-AI/litgpt/blob/main/litgpt/pretrain.py#L299
# learning rate decay scheduler (cosine with linear warmup)
def get_lr(learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                total += math.prod(p.quant_state.shape)
            else:
                total += p.numel()
    return total


class CycleIterator:
    """An iterator that cycles through an iterable indefinitely.

    Example:
        >>> iterator = CycleIterator([1, 2, 3])
        >>> [next(iterator) for _ in range(5)]
        [1, 2, 3, 1, 2]

    Note:
        Unlike ``itertools.cycle``, this iterator does not cache the values of the iterable.
    """

    def __init__(self, iterable: Iterable, upper: Optional[int] = 999999) -> None:
        self.iterable = iterable
        self.epoch = 0
        self.upper = upper
        self.count = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            if self.count >= self.upper:
                self._iterator = iter(self.iterable)
                self.count = 0
            self.count += 1
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self) -> Self:
        return self

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # # try to get deterministic training, but its slow!
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_experiment_name(cfg, datetime_str, accelerate) -> str:

    # start with todays date and time to make sure it is unique
    name = ""
    name += f"{datetime_str}"
    return name


def create_results_dir(cfg, datetime_str, accelerate):
    # first create a save dir name
    name = f"{datetime_str}"
    name = "/".join(name.split(" ")) # day/time

    result_dir = os.path.join(cfg.results_dir, cfg.wandb.project, name)

    os.makedirs(result_dir, exist_ok=True)

    return result_dir


@torch.no_grad
def calculate_grad_norm(model, norm_type=2.0, scaler=None):
    norm_type = float(norm_type)
    grads = []

    scale = scaler.get_scale() if scaler is not None else None

    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad.detach()
            if scale is not None:
                grad = grad / scale  # No clone; just use the result of the division
            grads.append(grad.float())  # Ensure float32 for stable norm computation

    if not grads:
        return 0.0

    if norm_type == float("inf"):
        total_norm = max(g.abs().max() for g in grads)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g, norm_type) for g in grads]), norm_type)

    return total_norm.item()
