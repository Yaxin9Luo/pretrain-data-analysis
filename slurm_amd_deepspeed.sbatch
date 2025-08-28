#!/bin/bash
#SBATCH --job-name=lentdd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --mem=120G
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err


# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate data_analysis

# WandB settings
export WANDB_MODE=online  # or 'offline' if you don't want to sync during training

# Set environment variables for AMD MI250X GPUs
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

# AMD MI210 specific GPU architecture
export PYTORCH_ROCM_ARCH=gfx90a
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=8

# Set visible devices
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# DeepSpeed environment variables
export DS_ACCELERATOR=cuda  # DeepSpeed uses CUDA API which is translated by HIP
export DS_BUILD_OPS=0  # Disable JIT compilation of ops
export DS_BUILD_CPU_ADAM=0  # Disable CPU Adam
export DS_BUILD_FUSED_ADAM=0  # Disable fused Adam to avoid compilation issues
export DS_BUILD_UTILS=0  # Disable utils
export DS_BUILD_CCL_COMM=0  # Disable CCL
export TORCH_CUDA_ARCH_LIST=""  # Don't set architecture list
export TORCH_EXTENSIONS_DIR=/vast/users/zhiqiang.shen/.cache/torch_extensions/py311_rocm
export DEEPSPEED_DISABLE_NINJA=1  # Disable ninja builder

# RCCL environment variables for DeepSpeed on AMD
export RCCL_DEBUG=WARN
# Auto-detect network interface
NETWORK_INTERFACE=$(ip -o -4 addr show | grep -v '127.0.0.1' | awk '{print $2}' | head -n1)
export RCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
echo "Using network interface: $NETWORK_INTERFACE"
export RCCL_IB_DISABLE=1  # Disable InfiniBand if causing issues
export RCCL_P2P_USE_HIP_MEMCPY=1

# RCCL/AMD-specific optimizations
export RCCL_TREE_THRESHOLD=0
export RCCL_TIMEOUT=600  # Increase timeout for stability
export HSA_FORCE_FINE_GRAIN_PCIE=1  # AMD GPU memory optimization
export RCCL_NCHANNELS=16  # Number of RCCL channels

# Compatibility layer - DeepSpeed might still look for NCCL vars
export NCCL_DEBUG=$RCCL_DEBUG
export NCCL_SOCKET_IFNAME=$RCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE=$RCCL_IB_DISABLE

# CPU optimization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Print system information
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 8))"
echo "Working directory: $(pwd)"

# Check DeepSpeed installation
echo "=== DeepSpeed Information ==="
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
ds_report

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# DeepSpeed configuration
DEEPSPEED_CONFIG=${1:-"config/train_gpt2_deepspeed_zero2.py"}
STRATEGY=${2:-"zero2"}

echo "=== Starting DeepSpeed Training ==="
echo "Configuration: $DEEPSPEED_CONFIG"
echo "Strategy: $STRATEGY"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 8))"

# Create hostfile for DeepSpeed
HOSTFILE="hostfile_${SLURM_JOB_ID}.txt"
scontrol show hostnames $SLURM_JOB_NODELIST | awk '{print $1" slots=4"}' > $HOSTFILE

echo "=== Hostfile ==="
cat $HOSTFILE

# Set distributed backend for AMD
export TORCH_DISTRIBUTED_BACKEND=rccl  # Use RCCL backend for AMD GPUs

# Launch DeepSpeed training
if [[ $SLURM_JOB_NUM_NODES -eq 1 ]]; then
    # Single node - don't use hostfile and use localhost for better stability
    export MASTER_ADDR=localhost
    export MASTER_PORT=29500
    deepspeed --num_gpus=8 train_deepspeed.py --backend=rccl $DEEPSPEED_CONFIG
else
    # Multi-node - use hostfile
    deepspeed \
        --hostfile=$HOSTFILE \
        --num_gpus=8 \
        --num_nodes=$SLURM_JOB_NUM_NODES \
        --master_addr=$(head -n 1 $HOSTFILE | cut -d' ' -f1) \
        --master_port=29500 \
        --launcher=openmpi \
        train_deepspeed.py --backend=rccl $DEEPSPEED_CONFIG
fi

# Clean up
rm -f $HOSTFILE

echo "=== Training Completed ==="

