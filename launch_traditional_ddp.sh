#!/bin/bash

# Launch script for traditional DDP training (baseline comparison)
# Use this to compare performance against DeepSpeed approaches

echo "Launching traditional DDP training (baseline)..."
echo "Configuration: PyTorch Distributed Data Parallel"
echo "Use this for performance baseline comparison"

# Default to 8 GPUs, can be overridden with argument
NUM_GPUS=${1:-8}

echo "Using $NUM_GPUS GPUs"

torchrun --standalone --nproc_per_node=$NUM_GPUS train.py config/train_gpt2.py

echo "Training completed!"
