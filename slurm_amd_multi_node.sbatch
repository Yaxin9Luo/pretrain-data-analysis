#!/bin/bash
#SBATCH --job-name=nanogpt-amd-multi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --mem=0
#SBATCH --partition=faculty
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --exclusive


# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate data_analysis

# Set environment variables for AMD MI250X GPUs
export ROCM_HOME=/opt/rocm-5.4.2
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

# AMD MI250X specific GPU architecture
export PYTORCH_ROCM_ARCH=gfx90a
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=8

# Set visible devices (MI250X has 2 GCDs per GPU card)
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Get master node address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# RCCL (AMD's NCCL) environment variables for multi-node
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_TREE_THRESHOLD=0
export RCCL_P2P_USE_HIP_MEMCPY=1

# Network optimizations for InfiniBand
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=14
export NCCL_IB_SL=0
export NCCL_IB_TC=0

# CPU affinity optimization
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
echo "Master node: $MASTER_ADDR"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"

# Check GPU and network on first node
if [[ $SLURM_NODEID -eq 0 ]]; then
    echo "=== GPU Information (Node 0) ==="
    rocm-smi --showproductname
    echo ""
    echo "=== InfiniBand Status ==="
    ibstat | grep -E "State:|Physical state:|Rate:"
fi

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Training configuration
TRAINING_CONFIG=${1:-"config/train_gpt2.py"}

echo "=== Starting Multi-Node Training ==="
echo "Configuration: $TRAINING_CONFIG"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 8))"
echo "Training script: train_amd.py"

# Launch training with torchrun on all nodes
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_amd.py $TRAINING_CONFIG

echo "=== Training Completed ==="

