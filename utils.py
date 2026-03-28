# most things here are taken from: https://github.com/Lightning-AI/litgpt/blob/main/litgpt/utils.py
from datetime import datetime
import math
import random
import os
import hydra
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

from cat_transformer import CAT_Config as CAT_Config_Fixed
from cat_transformer_hybrid import (
    HybridCAT_Config,
    CAT_Transformer_Hybrid,
)


def _make_compressor_config(cfg):
    """Build compressor config shared by cat_transformer and cat_transformer_hybrid."""
    return CAT_Config(
        vocab_size=cfg.dataset.vocab_size,
        block_size=cfg.model.block_size,

        use_fused_ops=False,  # liger-kernels doesn't support vmaps
        use_qk_norm=cfg.model.use_qk_norm,

        chunk_size=cfg.model.chunk_size,

        dim=cfg.model.compressor_dim,
        n_head=cfg.model.compressor_n_head,
        n_layer=cfg.model.compressor_n_layer,
        
        norm_eps=cfg.train.norm_eps,

        dim_fx=cfg.model.dim_fx,
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

            use_fused_ops=False,
            use_qk_norm=cfg.model.use_qk_norm,

            chunk_size=cfg.model.chunk_size,

            dim=cfg.model.compressor_dim,
            n_head=cfg.model.compressor_n_head,
            n_layer=cfg.model.compressor_n_layer,

            dim_fx=cfg.model.dim_fx,
        )

        decoder_config = CAT_Config(
            vocab_size=cfg.dataset.vocab_size,
            block_size=cfg.model.block_size,

            use_fused_ops=cfg.model.use_fused_ops,
            use_qk_norm=cfg.model.use_qk_norm,

            chunk_size=cfg.model.chunk_size,

            dim=cfg.model.dim,
            n_head=cfg.model.n_head,
            n_layer=cfg.model.n_layer,
        )

        model = CAT_Transformer(decoder_config, compressor_config)

        accelerate.print("CAT compressor config:", compressor_config)
        accelerate.print("CAT decoder config:", decoder_config)
        accelerate.print(model)
        return model

    elif "cat_transformer_hybrid" == cfg.model.name:

        compressor_config = _make_compressor_config(cfg)

        decoder_config = HybridCAT_Config(
            vocab_size=cfg.dataset.vocab_size,
            block_size=cfg.model.block_size,

            use_fused_ops=cfg.model.use_fused_ops,
            use_qk_norm=cfg.model.use_qk_norm,

            chunk_size=cfg.model.chunk_size,

            dim=cfg.model.dim,
            n_head=cfg.model.n_head,
            n_layer=cfg.model.n_layer,

            norm_eps=cfg.train.norm_eps,

            mamba2_layers=list(cfg.model.get("mamba2_layers", [])),
            gdn_layers=list(cfg.model.get("gdn_layers", [])),
            linear_attn_layers=list(cfg.model.get("linear_attn_layers", [])),

            mamba2_mode=cfg.model.get("mamba2_mode", "parallel"),
            mamba2_state_size=cfg.model.get("mamba2_state_size", 64),
            mamba2_n_groups=cfg.model.get("mamba2_n_groups", 1),
            mamba2_conv_kernel=cfg.model.get("mamba2_conv_kernel", 4),
            mamba2_use_conv=cfg.model.get("mamba2_use_conv", True),

            gdn_mode=cfg.model.get("gdn_mode", "parallel"),
            gdn_use_short_conv=cfg.model.get("gdn_use_short_conv", True),
            gdn_conv_size=cfg.model.get("gdn_conv_size", 4),

            fla_gdn_head_dim=cfg.model.get("fla_gdn_head_dim", 128),
            fla_gdn_num_heads=cfg.model.get("fla_gdn_num_heads", 8),
            fla_gdn_expand_v=cfg.model.get("fla_gdn_expand_v", 2.0),
        )

        model = CAT_Transformer_Hybrid(decoder_config, compressor_config)

        accelerate.print("Hybrid CAT compressor config:", compressor_config)
        accelerate.print("Hybrid CAT decoder config:", decoder_config)
        accelerate.print(model)
        return model

    else:
        raise ValueError(f"Unknown model type: {cfg.model.name}")


# https://github.com/Lightning-AI/litgpt/blob/main/litgpt/pretrain.py#L384
@torch.no_grad()
def validate(accelerate: Accelerator, model: nn.Module, val_dataloader: torch.utils.data.DataLoader, cfg, chunk_size_powers=None):
    
    print("Validating ...")
    model.eval()

    max_iters = cfg.eval.eval_iters

    if chunk_size_powers is not None:
        results = {}
        for power in chunk_size_powers:
            total_loss = 0.0
            total_tokens = 0
            desc = f"Evaluating (chunk={2**power})"
            val_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=desc, disable=(not accelerate.is_main_process))
            for k, batch in val_bar:
                if len(batch) == 3:
                    input_ids, targets, _batch_config = batch
                else:
                    input_ids, targets = batch
                if k >= max_iters:
                    break
                input_ids, targets = input_ids.to(accelerate.device), targets.to(accelerate.device)
                num_tokens = (targets != -100).sum().item()
                with accelerate.autocast():
                    loss = model(input_ids, targets, chunk_size_power=power)
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                val_bar.set_postfix_str(f"val loss: {total_loss / total_tokens:.4f}")
            val_loss = total_loss / total_tokens
            perplexity = math.exp(val_loss)
            results[power] = (val_loss, perplexity)
            accelerate.print(f"  chunk_size={2**power}: loss={val_loss:.4f}, ppl={perplexity:.4f}")
        model.train()
        return results

    total_loss = 0.0
    total_tokens = 0
    val_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Evaluating", disable=(not accelerate.is_main_process))
    for k, batch in val_bar:
        if len(batch) == 3:
            input_ids, targets, _batch_config = batch
        else:
            input_ids, targets = batch

        if k >= max_iters:
            break

        input_ids, targets = input_ids.to(accelerate.device), targets.to(accelerate.device)
        
        num_tokens = (targets != -100).sum().item()

        with accelerate.autocast():
            loss = model(input_ids, targets)

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        val_bar.set_postfix_str(f"val loss: {total_loss / total_tokens:.4f}")

    val_loss = total_loss / total_tokens
    perplexity = math.exp(val_loss)
    model.train()
    return val_loss, perplexity


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
    name = cfg.wandb.exp_name
    name += f" {datetime_str}"
    return name


def create_results_dir(cfg, datetime_str, accelerate):
    # first create a save dir name
    name = f"{datetime_str}"
    name = "/".join(name.split(" ")) # day/time

    original_cwd = hydra.utils.get_original_cwd()
    result_dir = os.path.join(original_cwd, cfg.results_dir, cfg.wandb.project, name)

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


@torch.no_grad()
def measure_accuracy(accelerate: Accelerator, model: nn.Module, val_dataloader: torch.utils.data.DataLoader, cfg, wandb, split="val", step=None, max_iters=-1, chunk_size_power=None) -> torch.Tensor:
    
    print("Accuracy validation...")
    model.eval()

    acc = list()
    predictions = list()
    acc_kv = dict()

    val_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Evaluating", disable=(not accelerate.is_main_process))
    for k, (input_ids, targets, batch_config) in val_bar:
        
        # NOTE: remove max_iters in accuracy eval 16-04-2025
        if max_iters != -1 and k >= max_iters:
            break

        input_ids, targets = input_ids.to(accelerate.device), targets.to(accelerate.device)
        
        if cfg.dataset.name == "mqar":
            input_ids = input_ids.squeeze(0)
            if targets is not None:
                targets = targets.squeeze(0)
                
        with accelerate.autocast():
            if chunk_size_power is not None:
                output_logits = model(input_ids, chunk_size_power=chunk_size_power)
            else:
                output_logits = model(input_ids)

        cur_pred = output_logits.argmax(dim=-1)
        # cur_probs = output_logits.softmax(dim=-1).max(dim=-1).values
        mask = targets != -100
        # calculate accuracy
        cur_acc = (cur_pred == targets).float().masked_select(mask)
        # cur_pred = cur_pred.masked_select(mask)
        # cur_targets = targets.masked_select(mask)
        # cur_probs = cur_probs.masked_select(mask)

        cur_acc = cur_acc.view(input_ids.shape[0], -1)
        # cur_pred = cur_pred.view(input_ids.shape[0], -1)
        # cur_targets = cur_targets.view(input_ids.shape[0], -1)
        # cur_probs = cur_probs.view(input_ids.shape[0], -1)

        acc.append(cur_acc.mean().unsqueeze(0))
        # predictions.append(cur_pred)

        # also print the predictions for first batch
        # if k == -1: # disable for now
        #     num_to_display = 10
        #     # print the table first
        #     print("#### KV Table:", input_ids[:num_to_display, :input_ids.shape[1] // 2].view(num_to_display, -1, 2), flush=True)
        #     # print("Targets:", cur_targets[:num_to_display, :])
        #     # print("Predictions:", cur_pred[:num_to_display, :])
        #     t = cur_targets[:num_to_display, :].view(num_to_display, -1, 1)
        #     p = cur_pred[:num_to_display, :].view(num_to_display, -1, 1)
        #     prob = cur_probs[:num_to_display, :].view(num_to_display, -1, 1).cpu().numpy().round(3)
        #     print("(Target, Prediction):", flush=True)
        #     print(torch.cat((t, p), dim=-1), flush=True)
        #     print("Probabilities:", flush=True)
        #     print(prob, flush=True)

        # OLD stuff
        # val_bar.set_postfix_str(f"accuracy: {(sum(acc) / len(acc)):.4f}")
        log_str = batch_config["num_kv_pairs"]
        if log_str not in acc_kv:
            acc_kv[log_str] = []
        acc_kv[log_str].append(cur_acc)

    # acc = torch.stack(acc).mean()
    acc = torch.concat(acc)
    # predictions = torch.concat(predictions, dim=0)

    if accelerate.is_main_process: # only doing it for the test, where max_iters = -1
        if split == "val":
            # log everything to wandb
            prefix_str = "val/"

            for kv, kv_acc in acc_kv.items():
                print("KV:", kv, "Accuracy:", torch.concat(kv_acc, dim=0).mean(dim=0).cpu().numpy().round(3), sep=" ")
                wandb.log({f"{prefix_str}accuracy_{kv}": torch.concat(kv_acc, dim=0).mean()})
            
            # plot the accuracy per KV pair
            # acc_per_pos = acc.mean(dim=0)
            # for i in range(acc_per_pos.shape[0]):
            #     wandb.log({f"val_extra/accuracy_token_{i}": acc_per_pos[i].item()})
        else:
            # log everything to wandb
            prefix_str = "train/"

            for kv, kv_acc in acc_kv.items():
                print("KV:", kv, "Accuracy:", torch.concat(kv_acc, dim=0).mean(dim=0).cpu().numpy().round(3), sep=" ")
                wandb.log({f"{prefix_str}accuracy_{kv}": torch.concat(kv_acc, dim=0).mean()})


            # acc_per_pos = acc.mean(dim=0)
            # for i in range(acc_per_pos.shape[0]):
            #     wandb.log({f"train_extra/accuracy_token_{i}": acc_per_pos[i].item()})

    # if accelerate.is_main_process:
    #     if max_iters == -1:
    #         # only save for test data
    #         save_path = os.path.join(cfg.result_dir, f"test_predictions_step_{step:07d}.pt")
    #         print("Shape of predictions:", predictions.shape, flush=True)
    #         print("Saving test predictions to:", save_path, flush=True)
    #         torch.save(predictions, save_path)
    #     else:
    #         save_path = os.path.join(cfg.result_dir, f"train_predictions_step_{step:07d}.pt")
    #         print("Shape of predictions:", predictions.shape, flush=True)
    #         print("Saving train predictions to:", save_path, flush=True)
    #         torch.save(predictions, save_path)

    model.train()
    # return acc
    return acc.mean()
