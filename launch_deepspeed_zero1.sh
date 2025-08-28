#!/bin/bash

# Launch script for DeepSpeed ZeRO-1 training
# ZeRO-1: Optimizer state partitioning only
# Best for: Medium-scale training with minimal communication overhead

echo "Launching DeepSpeed ZeRO-1 training..."
echo "Configuration: ZeRO Stage 1 (Optimizer state partitioning)"
echo "Best for: Moderate memory savings with maximum speed"

# Default to 8 GPUs, can be overridden with argument
NUM_GPUS=${1:-8}

echo "Using $NUM_GPUS GPUs"

deepspeed --num_gpus=$NUM_GPUS train_deepspeed.py config/train_gpt2_deepspeed_zero1.py

echo "Training completed!"
