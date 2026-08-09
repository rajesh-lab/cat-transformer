"""
Long-context finetuning entrypoint.

Reuses train.py's loop verbatim and injects the one thing pretraining does not do:
initialise the model from an existing checkpoint. train.py calls the module-global
`get_model` exactly once, right after the dataloaders are built and before the
optimizer, which is precisely where weights need to land, so we wrap that instead
of forking the training loop.

Adds a `finetune` config group:
    finetune.checkpoint_path   weights to start from (required)
    finetune.rope_base         RoPE theta for the finetune; applied to cfg.model

Note for the Activation Beacon `chunk_reset` scheme: RoPE positions restart inside
every chunk, so the maximum position index is chunk_size + n_beacons - 1 regardless
of block_size. Extending context introduces no unseen positions and rope_base has
no length-extrapolation role. It matters for the `compact` scheme, where the max
position grows with the number of chunks.

Sample:

accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 \
  --num_processes=4 --multi_gpu finetune.py \
  --config-path configs/books3 --config-name books3.yaml \
  finetune.checkpoint_path=/path/to/state_dict.pt \
  model_type=beacon_transformer beacon_transformer.block_size=16384
"""

import torch
from omegaconf import open_dict

import train as train_module
from utils import get_model as build_model, num_parameters


def load_checkpoint(accelerate, model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # checkpoints are saved from the DDP-wrapped model
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    accelerate.print(f"Loaded {len(state_dict)} tensors from checkpoint")


def get_model(accelerate, cfg):
    """train.get_model, plus rope_base override and checkpoint init."""
    finetune_cfg = cfg.get("finetune", None)
    if finetune_cfg is None:
        raise ValueError(
            "finetune.py requires a `finetune` config group; use a config such as "
            "configs/books3/books3.yaml, or run train.py for from-scratch training."
        )

    rope_base = finetune_cfg.get("rope_base", None)
    if rope_base is not None:
        # rope_base is a model property; get_model reads it off cfg.model.
        # open_dict restores the struct flag afterwards instead of forcing it on.
        with open_dict(cfg):
            cfg.model.rope_base = rope_base

    model = build_model(accelerate, cfg)

    checkpoint_path = finetune_cfg.get("checkpoint_path", None)
    if not checkpoint_path:
        raise ValueError("finetune.checkpoint_path must be set")

    load_checkpoint(accelerate, model, checkpoint_path)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_chunks = cfg.model.block_size // cfg.model.chunk_size

    accelerate.print("\n[FINETUNE] Long Context Extension")
    accelerate.print(f"Checkpoint: {checkpoint_path}")
    accelerate.print(f"RoPE base: {cfg.model.get('rope_base', 10000)}")
    accelerate.print(f"Block size: {cfg.model.block_size}")
    accelerate.print(f"Chunk size: {cfg.model.chunk_size}")
    accelerate.print(f"Num chunks: {num_chunks}")
    accelerate.print(f"Total parameters: {num_parameters(model):,}")
    accelerate.print(f"Trainable parameters: {trainable:,}\n")

    return model


# train.main resolves `get_model` from the train module namespace at call time
train_module.get_model = get_model
main = train_module.main


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
