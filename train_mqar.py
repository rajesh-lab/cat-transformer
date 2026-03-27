import os
import math
import time
import datetime
from tqdm import tqdm

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import torch
import wandb

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from data import get_dataset

from utils import (
    CycleIterator,
    validate,
    measure_accuracy,
    get_lr,
    num_parameters,
    seed_everything,
    get_experiment_name,
    create_results_dir,
    get_model,
)

@hydra.main()
def main(cfg: DictConfig):

    # init accelerate
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerate = Accelerator(kwargs_handlers=[ddp_kwargs])

    seed_everything(42)

    if accelerate.is_main_process:
        datetime_str = str(datetime.datetime.now())
        experiment_name = get_experiment_name(cfg, datetime_str, accelerate)
        result_dir = create_results_dir(cfg, datetime_str, accelerate)
        cfg.result_dir = result_dir

        # change batch-size due to ddp for logging
        cfg.train.batch_size = cfg.train.batch_size * accelerate.num_processes

        # dump hydra configs
        OmegaConf.save(HydraConfig.get().overrides.task, os.path.join(result_dir, "overrides.yaml"))
        OmegaConf.save(cfg, os.path.join(result_dir, "config.yaml"))
        OmegaConf.save(cfg.model, os.path.join(result_dir, "model.yaml"))

        # Initialize wandb
        wandb.init(
            project=cfg.wandb.project, 
            name=experiment_name,
            dir=result_dir,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        wandb.define_metric("val_loss", summary="min")

        accelerate.print("******* Results Dir *******")
        accelerate.print("Experiment:", experiment_name)
        accelerate.print("Results path:", result_dir)
        accelerate.print("***************************\n")

        # revert batch-size
        cfg.train.batch_size = cfg.train.batch_size // accelerate.num_processes

    accelerate.print(OmegaConf.to_container(cfg, resolve=True), "\n")

    train_dataset, test_dataset, data_config = get_dataset(cfg)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    train_val_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    train_dataloader = accelerate.prepare_data_loader(train_dataloader)
    train_iterator = CycleIterator(train_dataloader)

    # get model
    model = get_model(accelerate, cfg)

    is_cat = (cfg.model.name == "cat_transformer")
    if is_cat:
        max_power = int(math.log2(cfg.model.chunk_size))
        min_power = 2  # chunk_size = 4
        chunk_size_powers = list(range(min_power, max_power + 1))
        accelerate.print(f"CAT chunk_size_powers: {chunk_size_powers} (sizes: {[2**p for p in chunk_size_powers]})")

    accelerate.print("*****************************************************************")
    accelerate.print(f"Using #GPUs:", accelerate.num_processes)
    accelerate.print(f"Using Mixed Precision:", accelerate.mixed_precision)
    accelerate.print("Using Model type:", cfg.model.name)
    accelerate.print("Block size:", cfg.model.block_size)
    if is_cat:
        accelerate.print("Chunk size:", cfg.model.chunk_size)
    accelerate.print(f"Total parameters: {num_parameters(model):,}")
    accelerate.print("Batch size on single device:", cfg.train.batch_size)
    accelerate.print("Total effective batch size:", cfg.train.batch_size * accelerate.num_processes)

    if cfg.train.train_epochs > -1:
        accelerate.print(f"\nTraining for {cfg.train.train_epochs} epochs")
        cfg.train.train_iters = int(len(train_dataloader) * cfg.train.train_epochs)
    
    accelerate.print(f"\nTrain steps in one epoch: {accelerate.num_processes * len(train_dataloader):,}")
    accelerate.print(f"Train steps in this training: {cfg.train.train_iters:,}")
    accelerate.print(f"Effective train epochs in this training: {(cfg.train.train_iters * accelerate.num_processes / len(train_dataloader)):.2f}\n")

    if cfg.train.grad_norm > 0:
        accelerate.print(f"Using gradient clipping as: {cfg.train.grad_norm}\n")
    else:
        accelerate.print("Not using gradient clipping!!\n")

    if cfg.train.warmup_steps > 0:
        accelerate.print(f"Setting warm up steps to: {cfg.train.warmup_steps} !")
        warmup_steps = cfg.train.warmup_steps
    else:
        warmup_steps = int(cfg.train.train_iters * cfg.train.warmup_steps_percentage)
        accelerate.print(f"Setting warm up steps as {cfg.train.warmup_steps_percentage} * {cfg.train.train_iters} train_iters: {warmup_steps}")
    accelerate.print()
    accelerate.print("*****************************************************************")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, betas=cfg.optimizer.betas,
    )

    model, optimizer = accelerate.prepare(model, optimizer)
    accelerate.unwrap_model(model).setup_cache(device=accelerate.device)
    
    accelerate.wait_for_everyone()

    # training loop
    bar = tqdm(range(cfg.train.train_iters), desc="Training", disable=(not accelerate.is_main_process))
    for iter_num in bar:
        input_ids, targets, batch_config = next(train_iterator)

        start_time = time.time()

        optimizer.zero_grad()

        lr = get_lr(optimizer.defaults["lr"], iter_num, warmup_steps, cfg.train.train_iters, cfg.optimizer.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with accelerate.autocast():
            if is_cat:
                chunk_size_power = chunk_size_powers[iter_num % len(chunk_size_powers)]
                loss = model(input_ids, targets, chunk_size_power=chunk_size_power)
            else:
                loss = model(input_ids, targets)

        accelerate.backward(loss)

        if torch.isnan(loss):
            accelerate.print("Loss is NaN. Exiting...", flush=True)
            exit(1)

        if cfg.train.grad_norm > 0:
            accelerate.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_norm)

        optimizer.step()

        time_taken = time.time() - start_time
        token_throughput = input_ids.shape[0] * input_ids.shape[1] / time_taken / 1000

        if accelerate.is_main_process:
            wandb.log({
                "train/lr": optimizer.param_groups[0]['lr'], 
                "train/loss": loss.item(),
                "perf/Ktokens_s": token_throughput,
            })
            log_str = f'train/loss_num_kv_pairs_{batch_config["num_kv_pairs"]}'
            wandb.log({log_str: loss.item()})

        bar.set_postfix_str(f"loss: {loss.item():.4f}; lr: {lr:.6f}; {token_throughput:.2f}K tokens/s;")

        if (iter_num % cfg.eval.eval_interval == 0) and (iter_num > 0):
            eval_chunk_power = chunk_size_powers[-1] if is_cat else None

            train_accuracy = measure_accuracy(
                accelerate, model, train_val_dataloader, cfg, wandb,
                split="train", step=iter_num, max_iters=10,
                chunk_size_power=eval_chunk_power,
            )
            accelerate.print(f"Training Accuracy: {train_accuracy.item():.4f}")

            if is_cat:
                val_results = validate(
                    accelerate, model, test_dataloader, cfg,
                    chunk_size_powers=chunk_size_powers,
                )
                val_loss, val_ppl = val_results[chunk_size_powers[-1]]
            else:
                val_loss, val_ppl = validate(accelerate, model, test_dataloader, cfg)
            accelerate.print(f"Validation Loss: {val_loss:.4f}")

            accuracy = measure_accuracy(
                accelerate, model, test_dataloader, cfg, wandb,
                step=iter_num,
                chunk_size_power=eval_chunk_power,
            )
            accelerate.print(f"Validation Accuracy: {accuracy.item():.4f}")

            if accelerate.is_main_process:
                wandb.log({
                    "val/loss": val_loss,
                    "val/perplexity": val_ppl,
                })

            if accuracy.item() > 0.97:
                accelerate.print("Early stopping as accuracy is > 0.97")
                break

            accelerate.wait_for_everyone()

    # end-of-training eval
    if accelerate.is_main_process:
        model_save_path = os.path.join(result_dir, "state_dict.pt")
        accelerate.print("Training complete. Saving model at:", model_save_path)
        accelerate.save(model.state_dict(), model_save_path)

    accelerate.print("Evaluating at the end of training...")

    eval_chunk_power = chunk_size_powers[-1] if is_cat else None
    if is_cat:
        val_results = validate(
            accelerate, model, test_dataloader, cfg,
            chunk_size_powers=chunk_size_powers,
        )
        val_loss, val_ppl = val_results[chunk_size_powers[-1]]
    else:
        val_loss, val_ppl = validate(accelerate, model, test_dataloader, cfg)
    accuracy = measure_accuracy(
        accelerate, model, test_dataloader, cfg, wandb,
        step=cfg.train.train_iters, chunk_size_power=eval_chunk_power,
    )
    accelerate.print(f"Final val loss: {val_loss:.4f}, ppl: {val_ppl:.4f}, accuracy: {accuracy.item():.4f}")

    if accelerate.is_main_process:
        wandb.log({
            "val/end_loss": val_loss,
            "val/end_accuracy": accuracy.item(),
        })
        wandb.finish()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
