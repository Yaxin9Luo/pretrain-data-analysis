#!/bin/bash
#SBATCH --job-name=nanogpt-amd-single
#SBATCH --nodes=1
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

# RCCL (AMD's NCCL) environment variables
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export RCCL_P2P_USE_HIP_MEMCPY=1

# CPU affinity optimization for dual-socket systems
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Print system information
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of GPUs: 8"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Check GPU availability
echo "=== GPU Information ==="
rocm-smi --showproductname
echo ""
rocm-smi --showmeminfo vram

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Training configuration
TRAINING_CONFIG=${1:-"config/train_gpt2.py"}

echo "=== Starting Training ==="
echo "Configuration: $TRAINING_CONFIG"
echo "Number of GPUs: 8"
echo "Training script: train_amd.py"

# Launch training with torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train_amd.py $TRAINING_CONFIG

echo "=== Training Completed ==="

