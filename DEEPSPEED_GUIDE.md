# DeepSpeed Integration Guide

This guide explains how to use DeepSpeed with your GPT training codebase to choose different parallelism strategies based on your needs.

## Quick Start

### 1. Install DeepSpeed
```bash
pip install deepspeed
```

### 2. Choose Your Strategy

| Strategy | Memory Efficiency | Speed | Best For |
|----------|-------------------|-------|----------|
| **ZeRO-1** | Low | Highest | Small to medium models, maximum speed |
| **ZeRO-2** | Medium | High | Balanced training, moderate memory savings |
| **ZeRO-3** | High | Medium | Large models, significant memory savings |
| **ZeRO-3 + Offload** | Highest | Lowest | Very large models, maximum memory efficiency |

### 3. Launch Training

#### Option A: Use Launch Scripts (Recommended)
```bash
# ZeRO-1 (fastest, minimal memory savings)
./launch_deepspeed_zero1.sh 8

# ZeRO-2 (balanced)
./launch_deepspeed_zero2.sh 8

# ZeRO-3 (memory efficient)
./launch_deepspeed_zero3.sh 8

# ZeRO-3 with CPU offloading (most memory efficient)
./launch_deepspeed_zero3_offload.sh 8

# Traditional DDP (baseline comparison)
./launch_traditional_ddp.sh 8
```

#### Option B: Manual Launch
```bash
# ZeRO-2 example
deepspeed --num_gpus=8 train_deepspeed.py config/train_gpt2_deepspeed_zero2.py

# ZeRO-3 with custom parameters
deepspeed --num_gpus=8 train_deepspeed.py config/train_gpt2_deepspeed_zero3.py --batch_size=16
```

## Detailed Strategy Explanations

### ZeRO-1: Optimizer State Partitioning
- **What it does**: Partitions optimizer states across GPUs
- **Memory savings**: ~4x reduction in optimizer memory
- **Communication overhead**: Minimal
- **Best for**: When you want some memory savings without sacrificing speed

### ZeRO-2: Optimizer + Gradient Partitioning  
- **What it does**: Partitions optimizer states AND gradients across GPUs
- **Memory savings**: ~8x reduction in optimizer + gradient memory
- **Communication overhead**: Low
- **Best for**: Balanced approach between memory and speed

### ZeRO-3: Full Parameter Partitioning
- **What it does**: Partitions parameters, gradients, AND optimizer states
- **Memory savings**: Linear scaling with number of GPUs
- **Communication overhead**: Higher (parameters need to be gathered for forward/backward)
- **Best for**: Training large models that don't fit with ZeRO-2

### ZeRO-3 + CPU Offloading
- **What it does**: ZeRO-3 + offloads optimizer states and parameters to CPU
- **Memory savings**: Maximum possible
- **Communication overhead**: Highest (CPU-GPU transfers)
- **Best for**: Training models larger than GPU memory capacity

## Configuration Files

Each strategy has a corresponding configuration:

- `deepspeed_configs/zero1_config.json` - ZeRO-1 settings
- `deepspeed_configs/zero2_config.json` - ZeRO-2 settings  
- `deepspeed_configs/zero3_config.json` - ZeRO-3 settings
- `deepspeed_configs/zero3_offload_config.json` - ZeRO-3 + CPU offloading

Training configs:
- `config/train_gpt2_deepspeed_zero1.py`
- `config/train_gpt2_deepspeed_zero2.py`
- `config/train_gpt2_deepspeed_zero3.py`
- `config/train_gpt2_deepspeed_zero3_offload.py`

## Multi-Node Training

For multi-node training, use the `deepspeed` launcher with hostfile:

```bash
# Create hostfile
echo "node1 slots=8" > hostfile
echo "node2 slots=8" >> hostfile

# Launch multi-node training
deepspeed --hostfile=hostfile train_deepspeed.py config/train_gpt2_deepspeed_zero2.py
```

## Performance Monitoring

Monitor your training with:

1. **Weights & Biases**: Automatically logs training metrics
2. **DeepSpeed profiling**: Set `"wall_clock_breakdown": true` in config for detailed timing
3. **Memory usage**: Set `"memory_breakdown": true` for memory profiling

## Troubleshooting

### Common Issues

1. **OOM even with ZeRO-3**: Try ZeRO-3 + CPU offloading or reduce batch size
2. **Slow training with offloading**: Expected - CPU-GPU transfers are expensive
3. **Hanging during initialization**: Check NCCL settings, try `export NCCL_DEBUG=INFO`

### Performance Tips

1. **Start with ZeRO-2** for most use cases
2. **Use ZeRO-3 only when necessary** (memory constraints)
3. **Avoid CPU offloading unless required** (significant speed penalty)
4. **Monitor communication overhead** with profiling enabled

## Compatibility Notes

- **PyTorch Compilation**: Disabled when using DeepSpeed (handled internally)
- **Gradient Accumulation**: Handled automatically by DeepSpeed
- **Mixed Precision**: Configured in DeepSpeed config files
- **Checkpointing**: Uses DeepSpeed's native checkpointing format

## Example Workflow

```bash
# 1. Start with traditional DDP baseline
./launch_traditional_ddp.sh 8

# 2. Try ZeRO-2 for improved memory efficiency
./launch_deepspeed_zero2.sh 8

# 3. Scale up model size and use ZeRO-3 if needed
# Edit config files to increase model size, then:
./launch_deepspeed_zero3.sh 8

# 4. For very large models, use CPU offloading
./launch_deepspeed_zero3_offload.sh 8
```

This progressive approach helps you find the optimal balance between memory efficiency and training speed for your specific use case.
