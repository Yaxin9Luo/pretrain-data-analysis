"""
AMD GPU optimized training script for nanoGPT.
This script is based on train.py but includes AMD-specific optimizations for ROCm.

To run on a single AMD GPU:
$ python train_amd.py --batch_size=32 --compile=False

To run with DDP on multiple AMD GPUs on 1 node:
$ torchrun --standalone --nproc_per_node=4 train_amd.py

To run with DDP on AMD GPUs across multiple nodes (Slurm):
$ sbatch slurm_multi_node.sbatch
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# AMD GPU detection and optimization
def setup_amd_optimizations():
    """Setup AMD-specific optimizations"""
    if torch.cuda.is_available():
        # Check if we're using AMD GPUs (ROCm)
        gpu_name = torch.cuda.get_device_name(0).lower()
        if 'amd' in gpu_name or 'radeon' in gpu_name or 'gfx' in gpu_name:
            print(f"✅ AMD GPU detected: {torch.cuda.get_device_name(0)}")
            
            # AMD-specific optimizations
            os.environ.setdefault('HSA_FORCE_FINE_GRAIN_PCIE', '1')
            os.environ.setdefault('HIP_FORCE_DEV_KERNARG', '1')
            
            # ROCm memory optimizations
            torch.backends.cuda.matmul.allow_tf32 = False  # TF32 not available on AMD
            torch.backends.cudnn.allow_tf32 = False
            
            # Enable ROCm optimizations
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            
            return True
        else:
            print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
            # Keep NVIDIA optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return False
    return False

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # Disable compilation for AMD GPUs by default for better compatibility
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Setup AMD optimizations
is_amd_gpu = setup_amd_optimizations()

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # Set NCCL timeout to a longer value for AMD GPUs
    os.environ.setdefault('NCCL_TIMEOUT', '1800')  # 30 minutes for AMD
    
    if is_amd_gpu:
        # AMD-specific NCCL settings
        os.environ.setdefault('NCCL_DEBUG', 'WARN')
        os.environ.setdefault('NCCL_IB_DISABLE', '1')  # Disable InfiniBand
        os.environ.setdefault('NCCL_P2P_DISABLE', '1')  # Disable P2P for stability
        os.environ.setdefault('NCCL_MIN_NCHANNELS', '2')
        os.environ.setdefault('NCCL_MAX_NCHANNELS', '4')
        os.environ.setdefault('NCCL_BUFFSIZE', '2097152')  # 2MB buffer
        os.environ.setdefault('NCCL_NTHREADS', '64')
        
        # ROCm-specific settings
        os.environ.setdefault('HIP_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7'))
    else:
        # NVIDIA-specific NCCL settings
        os.environ.setdefault('NCCL_DEBUG', 'WARN')
        os.environ.setdefault('NCCL_IB_DISABLE', '1')
        os.environ.setdefault('NCCL_P2P_DISABLE', '1')
    
    import datetime
    init_process_group(backend=backend, timeout=datetime.timedelta(seconds=1800))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    
    # CPU affinity optimization for NUMA systems
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
        if ddp_local_rank < 4:
            # GPUs 0-3 on NUMA node 0
            start_cpu = ddp_local_rank * (cpus_per_task // 4)
            end_cpu = start_cpu + (cpus_per_task // 4) - 1
            os.system(f"taskset -cp {start_cpu}-{end_cpu} {os.getpid()} 2>/dev/null")
        else:
            # GPUs 4-7 on NUMA node 1
            start_cpu = (ddp_local_rank - 4) * (cpus_per_task // 4) + (cpus_per_task // 2)
            end_cpu = start_cpu + (cpus_per_task // 4) - 1
            os.system(f"taskset -cp {start_cpu}-{end_cpu} {os.getpid()} 2>/dev/null")
    
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_local_rank = 0  # Add this for non-DDP case

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# AMD GPUs may have issues with bfloat16, fallback to float16
if is_amd_gpu and dtype == 'bfloat16':
    if not torch.cuda.is_bf16_supported():
        print("⚠️  AMD GPU doesn't support bfloat16, falling back to float16")
        dtype = 'float16'
        ptdtype = torch.float16

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model (disabled by default for AMD compatibility)
if compile and not is_amd_gpu:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
elif compile and is_amd_gpu:
    print("⚠️  Model compilation disabled for AMD GPU compatibility")

# wrap model into DDP container
if ddp:
    if master_process:
        print("Wrapping model with DDP...")
    
    model = DDP(model, device_ids=[ddp_local_rank], 
                find_unused_parameters=False, 
                broadcast_buffers=True)
    
    if master_process:
        print("DDP model wrapping complete")

# Print system information
if master_process:
    print(f"=== System Information ===")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"AMD GPU: {is_amd_gpu}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print(f"Compile enabled: {compile}")
    print(f"Mixed precision: {dtype}")

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
training_start_time = time.time()  # Track overall training start time
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        
        # Calculate time estimation for evaluation log
        elapsed_time = time.time() - training_start_time
        progress_pct = (iter_num / max_iters) * 100
        eval_time_str = ""
        
        if iter_num > 0:
            avg_time_per_iter = elapsed_time / iter_num
            remaining_iters = max_iters - iter_num
            estimated_remaining_time = remaining_iters * avg_time_per_iter
            
            # Estimate completion time
            completion_time = time.time() + estimated_remaining_time
            completion_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(completion_time))
            
            # Format time remaining
            if estimated_remaining_time > 86400:  # More than 1 day
                days = int(estimated_remaining_time // 86400)
                hours = int((estimated_remaining_time % 86400) // 3600)
                eval_time_str = f" | ETA: {days}d {hours}h ({completion_str}) | Progress: {progress_pct:.1f}%"
            elif estimated_remaining_time > 3600:  # More than 1 hour
                hours = int(estimated_remaining_time // 3600)
                minutes = int((estimated_remaining_time % 3600) // 60)
                eval_time_str = f" | ETA: {hours}h {minutes}m ({completion_str}) | Progress: {progress_pct:.1f}%"
            else:  # Less than 1 hour
                minutes = int(estimated_remaining_time // 60)
                seconds = int(estimated_remaining_time % 60)
                eval_time_str = f" | ETA: {minutes}m {seconds}s ({completion_str}) | Progress: {progress_pct:.1f}%"
        
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}{eval_time_str}")
        if wandb_log:
            wandb_log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            # Add time estimation to wandb if available
            if iter_num > 0:
                wandb_log_dict.update({
                    "progress_pct": progress_pct,
                    "estimated_remaining_hours": estimated_remaining_time / 3600,
                })
            wandb.log(wandb_log_dict)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        # Calculate time estimation
        elapsed_time = time.time() - training_start_time
        progress_pct = (iter_num / max_iters) * 100
        
        time_str = ""
        if iter_num > 0:
            avg_time_per_iter = elapsed_time / iter_num
            remaining_iters = max_iters - iter_num
            estimated_remaining_time = remaining_iters * avg_time_per_iter
            
            # Format time remaining
            if estimated_remaining_time > 86400:  # More than 1 day
                days = int(estimated_remaining_time // 86400)
                hours = int((estimated_remaining_time % 86400) // 3600)
                time_str = f", ETA: {days}d {hours}h"
            elif estimated_remaining_time > 3600:  # More than 1 hour
                hours = int(estimated_remaining_time // 3600)
                minutes = int((estimated_remaining_time % 3600) // 60)
                time_str = f", ETA: {hours}h {minutes}m"
            else:  # Less than 1 hour
                minutes = int(estimated_remaining_time // 60)
                seconds = int(estimated_remaining_time % 60)
                time_str = f", ETA: {minutes}m {seconds}s"
            
            time_str = f"{time_str}, progress: {progress_pct:.1f}%"
        
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%{time_str}")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
