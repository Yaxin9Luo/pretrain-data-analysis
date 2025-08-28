#!/bin/bash

# Launch script for DeepSpeed ZeRO-2 training
# ZeRO-2: Optimizer state + gradient partitioning
# Best for: Balanced memory savings and communication overhead

echo "Launching DeepSpeed ZeRO-2 training..."
echo "Configuration: ZeRO Stage 2 (Optimizer + Gradient partitioning)"
echo "Best for: Balanced memory efficiency and speed"

# Default to 8 GPUs, can be overridden with argument
NUM_GPUS=${1:-8}

echo "Using $NUM_GPUS GPUs"

deepspeed --num_gpus=$NUM_GPUS train_deepspeed.py config/train_gpt2_deepspeed_zero2.py

echo "Training completed!"
