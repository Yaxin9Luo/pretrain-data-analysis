#!/bin/bash

# Launch script for DeepSpeed ZeRO-3 training
# ZeRO-3: Full parameter, gradient, and optimizer state partitioning
# Best for: Maximum memory efficiency

echo "Launching DeepSpeed ZeRO-3 training..."
echo "Configuration: ZeRO Stage 3 (Full parameter + optimizer + gradient partitioning)"
echo "Best for: Maximum memory efficiency, can train larger models"

# Default to 8 GPUs, can be overridden with argument
NUM_GPUS=${1:-8}

echo "Using $NUM_GPUS GPUs"

deepspeed --num_gpus=$NUM_GPUS train_deepspeed.py config/train_gpt2_deepspeed_zero3.py

echo "Training completed!"
