# config for training GPT-2 (124M) with DeepSpeed ZeRO-3
# ZeRO-3: Full parameter, gradient, and optimizer state partitioning
# Best for: Maximum memory efficiency, can train larger models
# launch: deepspeed --num_gpus=8 train_deepspeed.py config/train_gpt2_deepspeed_zero3.py

wandb_log = True
wandb_project = 'pretrain-data-analysis'
wandb_run_name = 'gpt2-124M-deepspeed-zero3'

# DeepSpeed configuration
deepspeed_config = 'deepspeed_configs/zero3_config.json'
use_deepspeed = True

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12 
block_size = 1024 
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# Disable compile for DeepSpeed compatibility
compile = False
