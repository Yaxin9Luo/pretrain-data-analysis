#!/bin/bash

# Launch script for DeepSpeed ZeRO-3 + CPU Offloading training
# ZeRO-3 with CPU offloading: Maximum memory efficiency for very large models
# Best for: Training models that don't fit in GPU memory even with ZeRO-3

echo "Launching DeepSpeed ZeRO-3 + CPU Offloading training..."
echo "Configuration: ZeRO Stage 3 with CPU offloading"
echo "Best for: Training very large models that exceed GPU memory"
echo "Note: This will be slower due to CPU-GPU data movement"

# Default to 8 GPUs, can be overridden with argument
NUM_GPUS=${1:-8}

echo "Using $NUM_GPUS GPUs"

deepspeed --num_gpus=$NUM_GPUS train_deepspeed.py config/train_gpt2_deepspeed_zero3_offload.py

echo "Training completed!"
