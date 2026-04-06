# train vanilla transformer and cat_transformer on ~15B tokens
# effective batch size: 32 * 4 * 4096 = 524,288 tokens (~0.5M tokens/step)
# train_iters: 114,440 micro-steps => 28,610 optimizer steps => ~15B tokens

# optionally disable wandb if you want
# WANDB_MODE=offline \ 
accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
--config-path . \
--config-name fineweb.yaml \
wandb.project="cats fineweb" \
wandb.exp_name="vanilla 12L lr 8e-4" \
\
train.batch_size=32 \
train.grad_accum=4 \
train.train_iters=114440 \
train.warmup_steps=2000 \
eval.eval_interval=11444 \
train.save_interval=11444 \
optimizer.lr=8e-4 \
optimizer.min_lr=8e-5 \
\
model_type=transformer \
transformer.block_size=4096 \
transformer.n_layer=12 \
transformer.dim=1024 \
transformer.n_head=16


# train cat_transformer
# we only set the max chunk size, trainer then iterates through all chunk sizes in powers of two
# you can set a lower max chunk size if you want

accelerate launch --config-file accelerate.yaml --mixed_precision=bf16 --num_processes=1 train.py \
--config-path . \
--config-name fineweb.yaml \
wandb.project="cats fineweb" \
wandb.exp_name="cat-transformer 4-8-16-32 12L lr 8e-4" \
\
train.batch_size=32 \
train.grad_accum=4 \
train.train_iters=114440 \
train.warmup_steps=2000 \
eval.eval_interval=11444 \
train.save_interval=11444 \
optimizer.lr=8e-4 \
optimizer.min_lr=8e-5 \
\
model_type=cat_transformer \
cat_transformer.block_size=4096 \
cat_transformer.chunk_size=32 \
\
cat_transformer.n_layer=12 \
cat_transformer.dim=2048 \
cat_transformer.n_head=32 \
cat_transformer.dim_fx=2048 \
\
cat_transformer.compressor_dim=1024 \
cat_transformer.compressor_n_head=16 \
cat_transformer.compressor_n_layer=3
