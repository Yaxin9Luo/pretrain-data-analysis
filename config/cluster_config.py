"""
Cluster-aware configuration system for nanoGPT training.
This module automatically detects cluster environment and optimizes settings accordingly.
"""

import os
import socket
import subprocess
from typing import Dict, Any, Optional


def detect_cluster_environment() -> Dict[str, Any]:
    """
    Detect cluster environment and return appropriate configuration.
    
    Returns:
        Dict containing cluster configuration parameters
    """
    config = {
        'is_slurm': False,
        'is_amd_gpu': False,
        'num_nodes': 1,
        'num_gpus_per_node': 1,
        'total_gpus': 1,
        'node_rank': 0,
        'local_rank': 0,
        'master_addr': 'localhost',
        'master_port': 29500,
        'backend': 'nccl',
        'timeout_minutes': 30,
    }
    
    # Detect Slurm environment
    if 'SLURM_JOB_ID' in os.environ:
        config['is_slurm'] = True
        config['num_nodes'] = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        config['total_gpus'] = int(os.environ.get('SLURM_NTASKS', 1))
        config['num_gpus_per_node'] = config['total_gpus'] // config['num_nodes']
        config['node_rank'] = int(os.environ.get('SLURM_NODEID', 0))
        config['local_rank'] = int(os.environ.get('SLURM_LOCALID', 0))
        
        # Get master node address
        if 'SLURM_JOB_NODELIST' in os.environ:
            try:
                nodelist = subprocess.check_output(
                    ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']]
                ).decode().strip().split('\n')
                config['master_addr'] = nodelist[0]
            except subprocess.CalledProcessError:
                config['master_addr'] = socket.gethostname()
    
    # Detect GPU type
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            config['is_amd_gpu'] = any(keyword in gpu_name for keyword in ['amd', 'radeon', 'gfx', 'mi100', 'mi200'])
            
            if not config['is_slurm']:
                config['total_gpus'] = torch.cuda.device_count()
                config['num_gpus_per_node'] = config['total_gpus']
    except ImportError:
        pass
    
    # Adjust timeout for larger clusters
    if config['total_gpus'] > 16:
        config['timeout_minutes'] = 60
    elif config['total_gpus'] > 8:
        config['timeout_minutes'] = 45
    
    return config


def get_optimal_batch_config(cluster_config: Dict[str, Any], model_size: str = 'gpt2-124M') -> Dict[str, Any]:
    """
    Get optimal batch configuration based on cluster setup and model size.
    
    Args:
        cluster_config: Output from detect_cluster_environment()
        model_size: Model size identifier
        
    Returns:
        Dict with batch size and gradient accumulation settings
    """
    total_gpus = cluster_config['total_gpus']
    is_amd = cluster_config['is_amd_gpu']
    
    # Base configurations for different model sizes
    configs = {
        'gpt2-124M': {
            'base_batch_size': 12,
            'base_grad_accum': 40,  # 5 * 8 from original config
            'max_batch_per_gpu': 16 if not is_amd else 12,  # AMD might need smaller batches
        },
        'gpt2-350M': {
            'base_batch_size': 8,
            'base_grad_accum': 60,
            'max_batch_per_gpu': 12 if not is_amd else 8,
        },
        'gpt2-774M': {
            'base_batch_size': 6,
            'base_grad_accum': 80,
            'max_batch_per_gpu': 8 if not is_amd else 6,
        },
        'gpt2-1.5B': {
            'base_batch_size': 4,
            'base_grad_accum': 120,
            'max_batch_per_gpu': 6 if not is_amd else 4,
        }
    }
    
    base_config = configs.get(model_size, configs['gpt2-124M'])
    
    # Calculate optimal batch size per GPU
    batch_size = min(base_config['base_batch_size'], base_config['max_batch_per_gpu'])
    
    # Calculate gradient accumulation to maintain effective batch size
    target_total_batch = base_config['base_batch_size'] * base_config['base_grad_accum']
    actual_total_batch = batch_size * total_gpus
    gradient_accumulation_steps = max(1, target_total_batch // actual_total_batch)
    
    # Ensure gradient accumulation is divisible by world size for DDP
    if cluster_config['total_gpus'] > 1:
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // total_gpus) * total_gpus
    
    return {
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'effective_batch_size': batch_size * gradient_accumulation_steps * total_gpus,
        'tokens_per_iter': batch_size * gradient_accumulation_steps * total_gpus * 1024,  # assuming block_size=1024
    }


def get_training_config(model_size: str = 'gpt2-124M', cluster_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get complete training configuration optimized for the cluster environment.
    
    Args:
        model_size: Model size identifier
        cluster_config: Optional cluster config (will be detected if not provided)
        
    Returns:
        Complete training configuration dictionary
    """
    if cluster_config is None:
        cluster_config = detect_cluster_environment()
    
    batch_config = get_optimal_batch_config(cluster_config, model_size)
    
    # Base training configuration
    config = {
        # Model configuration
        'init_from': 'scratch',
        'dataset': 'openwebtext',
        'block_size': 1024,
        
        # Batch configuration (optimized for cluster)
        'batch_size': batch_config['batch_size'],
        'gradient_accumulation_steps': batch_config['gradient_accumulation_steps'],
        
        # Training schedule
        'max_iters': 600000,
        'lr_decay_iters': 600000,
        'warmup_iters': 2000,
        'learning_rate': 6e-4,
        'min_lr': 6e-5,
        'weight_decay': 1e-1,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        
        # Evaluation
        'eval_interval': 2000,
        'eval_iters': 200,
        'log_interval': 10,
        'always_save_checkpoint': True,
        
        # System configuration
        'device': 'cuda',
        'compile': False if cluster_config['is_amd_gpu'] else True,  # Disable for AMD compatibility
        'backend': 'nccl',
        
        # Logging
        'wandb_log': True,
        'wandb_project': 'nanogpt-cluster',
        'wandb_run_name': f'{model_size}-{cluster_config["total_gpus"]}gpu',
    }
    
    # Model size specific configurations
    if model_size == 'gpt2-124M':
        config.update({
            'n_layer': 12,
            'n_head': 12,
            'n_embd': 768,
            'dropout': 0.0,
            'bias': False,
        })
    elif model_size == 'gpt2-350M':
        config.update({
            'n_layer': 24,
            'n_head': 16,
            'n_embd': 1024,
            'dropout': 0.0,
            'bias': False,
        })
    elif model_size == 'gpt2-774M':
        config.update({
            'n_layer': 36,
            'n_head': 20,
            'n_embd': 1280,
            'dropout': 0.0,
            'bias': False,
        })
    elif model_size == 'gpt2-1.5B':
        config.update({
            'n_layer': 48,
            'n_head': 25,
            'n_embd': 1600,
            'dropout': 0.0,
            'bias': False,
        })
    
    # AMD-specific optimizations
    if cluster_config['is_amd_gpu']:
        # Use float16 instead of bfloat16 for better AMD compatibility
        config['dtype'] = 'float16'
        # Reduce batch size slightly for memory safety
        config['batch_size'] = max(1, config['batch_size'] - 2)
        # Increase gradient accumulation to maintain effective batch size
        config['gradient_accumulation_steps'] += 1
    else:
        config['dtype'] = 'bfloat16'
    
    return config


def print_cluster_info(cluster_config: Dict[str, Any], training_config: Dict[str, Any]):
    """Print cluster and training configuration information."""
    print("=" * 60)
    print("CLUSTER CONFIGURATION")
    print("=" * 60)
    print(f"Environment: {'Slurm' if cluster_config['is_slurm'] else 'Local'}")
    print(f"GPU Type: {'AMD (ROCm)' if cluster_config['is_amd_gpu'] else 'NVIDIA (CUDA)'}")
    print(f"Total Nodes: {cluster_config['num_nodes']}")
    print(f"GPUs per Node: {cluster_config['num_gpus_per_node']}")
    print(f"Total GPUs: {cluster_config['total_gpus']}")
    print(f"Master Address: {cluster_config['master_addr']}")
    print(f"Backend: {cluster_config['backend']}")
    print()
    
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model: {training_config.get('n_layer', 'N/A')} layers, {training_config.get('n_embd', 'N/A')} dim")
    print(f"Batch Size per GPU: {training_config['batch_size']}")
    print(f"Gradient Accumulation: {training_config['gradient_accumulation_steps']}")
    print(f"Effective Batch Size: {training_config['batch_size'] * training_config['gradient_accumulation_steps'] * cluster_config['total_gpus']}")
    print(f"Tokens per Iteration: {training_config['batch_size'] * training_config['gradient_accumulation_steps'] * cluster_config['total_gpus'] * training_config['block_size']:,}")
    print(f"Learning Rate: {training_config['learning_rate']}")
    print(f"Max Iterations: {training_config['max_iters']:,}")
    print(f"Precision: {training_config.get('dtype', 'N/A')}")
    print(f"Compilation: {training_config.get('compile', False)}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    cluster_config = detect_cluster_environment()
    training_config = get_training_config('gpt2-124M', cluster_config)
    print_cluster_info(cluster_config, training_config)
