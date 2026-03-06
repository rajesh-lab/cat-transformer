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

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
torch._dynamo.config.optimize_ddp=False

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from data import get_dataset

from utils import (
    CycleIterator,
    validate, 
    get_lr,
    num_parameters,
    seed_everything,
    get_experiment_name,
    create_results_dir,
    calculate_grad_norm,
    get_model
)

@hydra.main()
def main(cfg: DictConfig):

    # init accelerate
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerate = Accelerator(kwargs_handlers=[ddp_kwargs])

    seed_everything(cfg.seed)

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
            dir=result_dir, # save in seperate wandb dir
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("end_val_loss", summary="min")

        accelerate.print("******* Results Dir *******")
        accelerate.print("Experiment:", experiment_name)
        accelerate.print("Results path:", result_dir)
        accelerate.print("***************************\n")

        # revert batch-size
        cfg.train.batch_size = cfg.train.batch_size // accelerate.num_processes

    accelerate.print(OmegaConf.to_container(cfg, resolve=True), "\n")

    train_dataset, test_dataset, data_config = get_dataset(cfg)
    tokenizer = AutoTokenizer.from_pretrained(data_config["tokenizer_name"])

    # Create a generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(334) # same data order

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=g,
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    ) # we don't prepare the val_loader since we test on all processes!

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

    accelerate.print(f"Total parameters: {num_parameters(model):,}")

    accelerate.print("Batch size on single device:", cfg.train.batch_size)
    accelerate.print("Total effective batch size:", cfg.train.batch_size * accelerate.num_processes * cfg.train.grad_accum)

    grad_accum = cfg.train.grad_accum
    accelerate.print(f"Gradient Accumulation steps: {grad_accum} ~~~~~")

    accelerate.print(f"\nTrain iters in this training: {cfg.train.train_iters:,}")
    
    accelerate.print(f"\nGradient steps in this training: {cfg.train.train_iters // grad_accum:,}")
    effective_batch_size_tokens = cfg.train.batch_size * accelerate.num_processes * grad_accum * cfg.model.block_size
    accelerate.print(f"Effective batch size (tokens): {effective_batch_size_tokens:,}")
    accelerate.print(f"Iterating over tokens: {cfg.train.train_iters * cfg.train.batch_size * accelerate.num_processes * cfg.model.block_size:,}")

    if cfg.train.grad_norm > 0:
        accelerate.print(f"\nUsing gradient clipping as: {cfg.train.grad_norm}\n")
    else:
        accelerate.print("\nNot using gradient clipping!!\n")

    if cfg.train.warmup_steps > 0:
        accelerate.print(f"Setting warm up steps to: {cfg.train.warmup_steps} !")
        warmup_steps = cfg.train.warmup_steps
    else:
        warmup_steps = int(cfg.train.train_iters * cfg.train.warmup_steps_percentage)
        accelerate.print(f"Setting warm up steps as {cfg.train.warmup_steps_percentage} * {cfg.train.train_iters} train_iters: {warmup_steps}")
    accelerate.print()
    accelerate.print("*****************************************************************")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, betas=cfg.optimizer.betas,
    )

    model, optimizer = accelerate.prepare(model, optimizer)
    accelerate.unwrap_model(model).setup_cache(device=accelerate.device) # required for fp32 rope
    
    # wait for all processes just to be safe :)
    accelerate.wait_for_everyone()

    # do training for train_iters steps
    bar = tqdm(range(cfg.train.train_iters), desc="Training", disable=(not accelerate.is_main_process))
    for iter_num in bar:
        input_ids, targets = next(train_iterator)

        # move to device
        input_ids = input_ids.to(accelerate.device, non_blocking=True)
        targets = targets.to(accelerate.device, non_blocking=True)

        start_time = time.time()

        # Zero gradients only at the start of each accumulation cycle
        if iter_num % grad_accum == 0:
            optimizer.zero_grad()

        # determine and set the learning rate for this iteration
        lr = get_lr(optimizer.defaults["lr"], iter_num, warmup_steps, cfg.train.train_iters, cfg.optimizer.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with accelerate.autocast():
            if is_cat:
                chunk_size_power = chunk_size_powers[iter_num % len(chunk_size_powers)]
                original_loss = model(input_ids, targets, chunk_size_power=chunk_size_power)
            else:
                original_loss = model(input_ids, targets)
            loss = original_loss / grad_accum  # Normalize loss to account for accumulation

        accelerate.backward(loss)

        if (iter_num + 1) % grad_accum == 0:
            if cfg.train.grad_norm > 0:
                grad_norm = accelerate.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_norm)

            optimizer.step()

            time_taken = time.time() - start_time # optimizer.step() syncs the processes in distributed settings!
            token_throughput = input_ids.shape[0] * input_ids.shape[1] / time_taken / 1000

            if accelerate.is_main_process:
                log_dict = {
                    "train/lr": optimizer.param_groups[0]['lr'], 
                    "train/loss": original_loss.item(),
                    "perf/Ktokens_s" : token_throughput,
                }
                if is_cat:
                    log_dict["extra/train_chunk_size_power"] = chunk_size_power
                wandb.log(log_dict)
                if cfg.train.grad_norm > 0 and iter_num % cfg.train.grad_norm_interval == 0:
                    wandb.log({"train/grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm})
                if accelerate.mixed_precision == "fp16":
                    wandb.log({
                        "train/grad_scaler": accelerate.scaler.get_scale(),
                    })

            if accelerate.mixed_precision == "fp16":
                bar.set_postfix_str(f"loss: {original_loss.item():.4f}; lr: {lr:.6f}; {token_throughput:.2f}K tokens/s; grad_scaler: {accelerate.scaler.get_scale():.4f};")
            else:
                bar.set_postfix_str(f"loss: {original_loss.item():.4f}; lr: {lr:.6f}; {token_throughput:.2f}K tokens/s;")
            

        if (iter_num % cfg.eval.eval_interval == 0) and (iter_num > 0) :
            
            accelerate.print("Validating log loss...")
            if is_cat:
                val_results = validate(accelerate, model, test_dataloader, cfg, chunk_size_powers=chunk_size_powers)
                if accelerate.is_main_process:
                    max_power = chunk_size_powers[-1]
                    for power, (vl, vp) in val_results.items():
                        if power == max_power:
                            wandb.log({"val/loss": vl, "val/perplexity": vp})
                        
                        wandb.log({
                            f"extra/loss_chunk{2**power}": vl,
                            f"extra/perplexity_chunk{2**power}": vp,
                        })
            else:
                val_loss, val_perplexity = validate(accelerate, model, test_dataloader, cfg)
                accelerate.print(f"Validation Loss: {val_loss:.4f}")
                wandb.log({
                    "val/loss": val_loss,
                    "val/perplexity": val_perplexity,
                })

            if accelerate.is_main_process and (iter_num % cfg.train.save_interval == 0) and (iter_num > 0):

                model_save_path = os.path.join(result_dir, f"intermediate_state_dict_{iter_num:07d}.pt")
                accelerate.print("Saving intermediate model at:", model_save_path)
                accelerate.save(model.state_dict(), model_save_path)

            # wait for all GPUs
            accelerate.wait_for_everyone()
            

    if accelerate.is_main_process:
        model_save_path = os.path.join(result_dir, "state_dict.pt")
        accelerate.print("training complete. Saving model at:", model_save_path)
        accelerate.save(model.state_dict(), model_save_path)

    accelerate.print("evaluating at the end of training...")
   
    # perform a full validation at the end
    accelerate.print("Validating log loss...")
    if is_cat:
        val_results = validate(accelerate, model, test_dataloader, cfg, chunk_size_powers=chunk_size_powers)
        if accelerate.is_main_process:
            max_power = chunk_size_powers[-1]
            for power, (vl, vp) in val_results.items():
                if power == max_power:
                    wandb.log({"val/loss": vl, "val/perplexity": vp})
                
                wandb.log({
                    f"extra/loss_chunk{2**power}": vl,
                    f"extra/perplexity_chunk{2**power}": vp,
                })
    else:
        val_loss, val_perplexity = validate(accelerate, model, test_dataloader, cfg)
        accelerate.print(f"Validation: {val_loss:.4f}")
        if accelerate.is_main_process:
            wandb.log({
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
            })

    accelerate.wait_for_everyone()

    if accelerate.is_main_process:
        # flush wandb
        wandb.finish()

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    main()