"""
Training Script
=====================================
"""

from contextlib import nullcontext
import math
import os
import csv
import json
from typing import Literal, Optional, Tuple, Any
from datetime import datetime
import time
import numpy as np
from tqdm import trange
from termcolor import colored,cprint
import matplotlib.pyplot as plt
from custom_tokenizers import tiktoken_tokenizer, char_level_tokenizer

import torch
from torch.nn import functional as F

from Config import GPTConfig, GPT2Config, ModelConfig
from model_gpt2 import GPT2Model

import torch.distributed as dist # Run script with: torchrun --standalone --nproc_per_node=1 train_gpt2.py
from torch.nn.parallel import DistributedDataParallel as DDP

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# DDP (Distributed Data Parallel): torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
# -------------------------------------------------------------------------------

DDP_ACTIVE = int(os.environ.get('RANK', -1)) != -1 # RANK equals -1 if DDP is not available
if DDP_ACTIVE:
    # DDP mode
    cprint("DDP mode", color="yellow",attrs=["bold"])
    assert torch.cuda.is_available(), "DDP requires CUDA to be available"
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    compute_device = f'cuda:{ddp_local_rank}'
    device = torch.device(compute_device)
    
    # Initialize the process group for DDP
    if ddp_world_size == 1:
        dist.init_process_group(backend='gloo') 
    else:
        dist.init_process_group(backend='nccl')  # TODO: Test on multiple GPUs

    torch.cuda.set_device(device)  # Set the current device to the local rank's GPU

else:
    # NON DDP mode
    cprint(f"Single process mode (not DDP)", color="yellow", attrs=["bold"])
    ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1  # Default values for single process mode

    compute_device = "cpu"
    if torch.cuda.is_available():
        compute_device = f'cuda:{ddp_local_rank}'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        compute_device = "mps"
    device = torch.device(compute_device)

master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

compute_color_map = {}
color_list = ["cyan", "red", "orange", "green", "blue", "indigo", "violet"]
for i in range(len(color_list)):
    compute_color_map[f"cuda:{i}"] = color_list[i % len(color_list)]
compute_color_map["cpu"] = "light_grey"
compute_color_map["mps"] = "dark_grey"
# Set color for the compute device
compute_color = compute_color_map.get(compute_device, "white")  # Default to white if not found
device_type = "cuda" if compute_device.startswith("cuda") else compute_device
cprint(f"DDP active = {DDP_ACTIVE}. Device: {compute_device} of type {device_type}: ddp_rank = {ddp_rank}, ddp_local_rank = {ddp_local_rank}, ddp_world_size = {ddp_world_size}", compute_color)

# Reproducibility settings
# -----------------------
seed = 1337
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed) 

# GPU Optimization Settings
# -----------------------
TENSOR_CORES = True  # Set to True to enable Tensor Cores for faster matrix multiplications on supported GPUs
TORCH_COMPILATION = True  # Set to True to enable PyTorch 2.0's compile feature for performance optimization
AUTOCAST = True  # Mixed precision training https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html

torch.set_float32_matmul_precision('high') if TENSOR_CORES else None

# File paths
TRAIN_ID = datetime.now().strftime("%Y%m%d_%H%M") # Unique identifier for this training session
DATA_PATH = "data/edu_fineweb10B/shards/"
REPORT_DIR = f'logs/GPT2_training_{TRAIN_ID}_{compute_device}'
# Create plots directory
os.makedirs(REPORT_DIR, exist_ok=True)
# =============================================================================
# HYPERPARAMETERS
# =============================================================================

config = GPT2Config(compute_device=compute_device)

if master_process:
    cprint("HYPERPARAMETERS", compute_color)
    cprint(config.model_dump_json(indent=2), compute_color)

# =============================================================================
# DATA PREPARATION
# =============================================================================
if master_process:
    cprint("DATA PREPARATION", compute_color)

tokenizer = char_level_tokenizer if config.selected_tokenizer == char_level_tokenizer.name else tiktoken_tokenizer
config.vocab_size = tokenizer.n_vocab  if config.vocab_size is None else config.vocab_size # Set vocabulary size based on the tokenizer if it is not provided in the config

class DataLoaderFromTokenShards:
    def __init__(self, B:int, T:int, process_rank:int=0, num_processes:int=1, split:Literal['train', 'val']='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split  # Store split type
        assert split in {'train','val'} 

        # get the shard filenames
        shard_names = os.listdir(DATA_PATH) # Get all shard file names in the root
        shard_names = [s for s in shard_names if split in s] # Check if 'test' or 'train' is in the shard file name
        shard_names = sorted(shard_names) # Sort shard names
        shard_names = [os.path.join(DATA_PATH, s) for s in shard_names] # Create absolute path names for each shard
        self.shard_names = shard_names
        
        assert len(shard_names) > 0, f"No shard files found for split {split}"
        if master_process:
            cprint(f"found {len(shard_names)} for split {split}", compute_color)

        self.reset()
        self.vocab_size = tokenizer.n_vocab
            
    def reset(self):
        """
        Reset the data loader to the initial state.
        """
        self.current_shard = 0
        self.tokens = torch.tensor(np.load(self.shard_names[self.current_shard]), dtype = torch.long)  # Convert to long tensor
        self.current_position = self.process_rank * self.B * self.T  # state: initialize current position based on process rank to ensure each process gets a unique subset of the data
    
    # Batch generator
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of input-target pairs for training.
            
            How batching works:
            - Sample random starting positions in the dataset
            - Extract sequences of length seq_size starting from those positions
            - Create targets by shifting input sequences by one position
            
            Example with seq_size=5, batch_size=2:
                Input:  [[23, 30, 31, 7, 21], [45, 12, 8, 33, 9]]
                Target: [[30, 31, 7, 21, 14], [12, 8, 33, 9, 41]]
            
            This gives us seq_size training examples per sequence:
                - Predict 30 given [23]
                - Predict 31 given [23, 30]
                - Predict 7 given [23, 30, 31]
                - etc.
                
        This implementation supports DDP training by ensuring that each process gets a unique subset of the data based on its rank.
        This implementation assumes that the data is already tokenized and stored in numpy arrays in the specified shard files.
        """
        
        B,T = self.B, self.T
        data = self.tokens
        
        # Ensure current_position does not exceed data length
        if self.current_position >= len(data):
            self.current_position = self.process_rank * self.B * self.T

        buf = data[self.current_position: self.current_position+ B*T + 1]

        # Safeguard against empty slices
        if len(buf) < B * T + 1:
            raise ValueError(f"Insufficient data to generate batch: requested {B * T + 1}, but got {len(buf)}")

        xb = buf[:-1].view(B,T) # inputs
        yb = buf[1:].view(B,T)  # targets (shifted) 
        self.current_position += B * T * self.num_processes # advance the current position by B*T*num_processes tokens to ensure each process gets a unique subset of the data

        # If we reach the end of the tokens, move to the next shard at the first position
        if self.current_position + (B * T * self.num_processes + 1) > len(data):
            self.current_shard = (self.current_shard + 1) % len(self.shard_names)  # Move to the next shard
            self.tokens = torch.tensor(np.load(self.shard_names[self.current_shard]), dtype = torch.long)  # Convert to long tensor
            self.current_position = self.process_rank * B * T

        # Move to compute device (GPU or CPU)
        xb, yb = xb.to(device), yb.to(device)

        return xb, yb

# Create dataloader instance
train_loader = DataLoaderFromTokenShards(B=config.batch_size, T=config.seq_size, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader_logs = DataLoaderFromTokenShards(B=config.batch_size, T=config.seq_size, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
train_loader_logs = DataLoaderFromTokenShards(B=config.batch_size, T=config.seq_size, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')

# Macro batch: We will repeat the forward - backward grad_accumulation_steps times to simulate a larger batch size. We will call this micro-step.
tokens_per_step = config.tokens_per_step #config.tokens_per_step # 2**19  approx. 0.5M like in the paper
total_batch_size = config.batch_size * config.seq_size  # Total batch size across all processes (DDP)
assert tokens_per_step % (total_batch_size*ddp_world_size) == 0, "tokens_per_step must be divisible by total_batch_size * ddp_world_size"
grad_accumulation_steps = tokens_per_step // (total_batch_size*ddp_world_size)

if master_process:
    cprint("\nBATCH CALCULATIONS", compute_color)
    cprint(f"Macro batch size: {tokens_per_step} tokens", compute_color)
    cprint(f"Total batch size (B*T): {config.batch_size} * {config.seq_size} = {total_batch_size} tokens", compute_color)
    cprint(f"Grad accumulation steps (tokens_per_step // (total_batch_size*ddp_world_size)): {grad_accumulation_steps}", compute_color)

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
if master_process:
    cprint("\nMODEL INITIALIZATION", compute_color)


# Create model instance
model = GPT2Model(config)
model.to(device) # this only works for the model, for tensors do tensor = tensor.to(device)


# Speed ups the model with PyTorch 2.0's compile feature (optional, but recommended for performance) 
# https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html 
if TORCH_COMPILATION:
    model = torch.compile(model)

# DDP Wrapper for the model
if DDP_ACTIVE:
    model = DDP(model, device_ids = [ddp_local_rank]) # DDP fails here if using 1 GPU and backed = 'nccl' (it requires at least 2 GPUs), so we use 'gloo' backend for single GPU training


if master_process:
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = total_params * 4 / 1024**2  # Size in MB (assuming float32, 4 bytes per parameter)
    model_summary = (
        f"\nModel Details:\n"
        f"  Total parameters: {total_params:,}\n"
        f"  Trainable parameters: {trainable_params:,}\n"
        f"  Model size: ~{model_size:.2f} MB (float32)\n"
    )

    cprint(model_summary, compute_color)


# Print model architecture
# -----------------------
# Option 1
#cprint(model, compute_color)
# Option 2
# for k, v in model.state_dict().items():
#     cprint(f"{k}: {v.shape} - {v.dtype}", compute_color)  # Print each parameter's shape and dtype


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

# New! Learning rate scheduler function
def get_lr(step: int, config: ModelConfig) -> float:
    """
    Get the learning rate for the current training step.
    Uses a cosine decay schedule with warmup.
    """
    max_lr = config.lr  # Maximum learning rate
    min_lr = max_lr * 0.1  # Minimum learning rate (10% of max)
    lr_warmup_steps = config.lr_warmup_steps  # Number of warmup steps
    total_steps = config.training_steps  # Total training steps
    # If no decay and no warmup, return max_lr
    if not config.lr_decay and lr_warmup_steps == 0:
        return max_lr
    # (1) warmup
    if step < lr_warmup_steps:
        return max_lr * (step+1) / lr_warmup_steps  # Linear warmup
    # (2) cosine decay
    elif step >= lr_warmup_steps and step <= total_steps and config.lr_decay:
        decay_ratio = (step - lr_warmup_steps) / (total_steps - lr_warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)
    # (3) after training
    else: 
        return min_lr

# =============================================================================
# TRAINING 
# =============================================================================

cprint("TRAINING", compute_color)


# Initialize lists to store metrics for plotting
train_loss_list, val_loss_list, lr_list, norm_list, duration_list, tokens_per_sec_list = [], [], [], [], [], []

# Initialize optimizer
if DDP_ACTIVE:
    optimizer = model.module.configure_optimizers(config, device_type=device_type)
else:
    optimizer = model.configure_optimizers(config, device_type=device_type)

log_file = os.path.join(REPORT_DIR, f"log.csv")
header_row = f"Step; Train Loss Accum; Val Loss Accum ; lr;  norm ;  dt (s);  Tokens/s;\n"
with open(log_file, "w") as f: # open for writing to clear the file
    f.write(header_row)

start_train_loop = datetime.now()
pbar = trange(config.training_steps, desc="Training steps", unit="step", disable=not master_process, colour=compute_color)
for step in pbar:
    last_step = (step == config.training_steps - 1)
    t0 = time.time()

    # EVALUATION PHASE
    # -------------------------------------------------------------------------------------------------------------------------------------
    # (1) Once every eval_interval steps, evaluate the model on the validation set for eval_loss_steps steps
    if step % config.eval_interval == 0 or last_step:
        model.eval()
        val_loader_logs.reset()  # Reset the validation loader to the start of the validation data
        train_loader_logs.reset()  # Reset the training loader to the start of the training data
        with torch.no_grad():
            val_loss_log = 0.0
            train_loss_log = 0.0
            for _ in range(config.eval_loss_steps):
                xb_val, yb_val = val_loader_logs.next_batch()
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == "cuda" else torch.float32) if AUTOCAST else nullcontext():
                    _, val_loss = model(xb_val, yb_val)
                    val_loss = val_loss / config.eval_loss_steps
                    val_loss_log += val_loss.detach()  # Accumulate validation loss

                xb_train, yb_train = train_loader_logs.next_batch()
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == "cuda" else torch.float32) if AUTOCAST else nullcontext():
                    _, train_loss = model(xb_train, yb_train)
                    train_loss = train_loss / config.eval_loss_steps
                    train_loss_log += train_loss.detach()  # Accumulate training loss

        if DDP_ACTIVE:
            dist.all_reduce(val_loss_log, op=dist.ReduceOp.SUM)
            val_loss_log /= ddp_world_size  # Average the validation loss across all DDP processes
            dist.all_reduce(train_loss_log, op=dist.ReduceOp.SUM)
            train_loss_log /= ddp_world_size  # Average the training loss across all DDP processes

        if master_process:
            pass
            # TODO Checkpoints
            

    # (2) Once every eval_interval steps generate text from the model
    if (step % config.eval_interval == 0 or last_step) and False:
        # Context tokens
        context_text = "Hello, I'm a language model,"
        context_tokens = tokenizer.encode(context_text)
        context_tokens = torch.tensor(context_tokens, dtype=torch.long) # 1, 8

        # Manually generating a batch with the same sequence context_tokens 5 times 
        num_generated_sequences = 5
        context_tokens = context_tokens.unsqueeze(0).repeat(num_generated_sequences, 1) # 5, 8
        idx = context_tokens.to(device)
        sample_rgn = torch.Generator(device=device)  # Create a random number generator for sampling, to avoid using the global random state and affect the training process
        sample_rgn.manual_seed(seed + ddp_rank)  # Seed the random number generator with the current step for reproducibility
        max_new_tokens = 30

        # Generate from context tokens (manually instead of using model.generate() not implemented yet)
        while idx.size(1) < max_new_tokens:

            with torch.no_grad():
                # right now idx is (B, T) where B = 5, T = 8

                # forward the model
                logits, _ = model(idx)  # (B, T, vocab_size)

                # Focus on the last time step (next token prediction)
                logits = logits[:, -1, :] # (B, vocab_size) 

                # Convert to probabilities
                probs = F.softmax(logits, dim = -1) # (B, vocab_size)

                # Do top-k sampling of 50 (huggingface default pipeline)
                # topk_probs here becomes (5,50) and topk_indices becomes (5,50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # Get top 50 probabilities and indices
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rgn)  # Select a token from the top 50 probabilities (B, 1)
                xcol = torch.gather(topk_indices, -1, ix) # Gather the corresponding token indices based on the sampled probabilities
                
                # Append to sequence
                idx = torch.cat((idx, xcol), dim = 1) # (B, T+1)

        # print the generated sequences
        for i in range(num_generated_sequences):
            generated_tokens = idx[i, :max_new_tokens].tolist() # Get the generated tokens for this sequence
            generated_text = tokenizer.decode(generated_tokens)  # Decode the tokens to text
            cprint(f"DDP rank {ddp_rank} - Generated text {i+1}: <START> {generated_text}<END>", compute_color)
        
    
    # TRAINING PHASE
    # ---------------------------------------------------------------------------------------------------------------------------------------
    model.train()
    optimizer.zero_grad() # Reset gradients

    # Macro/Micro Batch: Accumulate gradients over multiple micro-steps to simulate a larger macro batch size
    train_loss_acc = 0.0
    for micro_step in range(grad_accumulation_steps):
        
        # Sample a batch random of training data
        xb, yb = train_loader.next_batch()

        if DDP_ACTIVE:
            model.require_backward_grad_sync = (micro_step == grad_accumulation_steps - 1)  # Only sync gradients on the last micro-step
        
        # Forward pass
        with (torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == "cuda" else torch.float32) if AUTOCAST else nullcontext()):
            logits, loss = model(xb, yb)
        
        loss = loss / grad_accumulation_steps # Scale the loss by the number of micro-steps to average it out
        train_loss_acc += loss.detach()  # Accumulate loss over micro-steps
        
        # Backward pass
        loss.backward()

    if DDP_ACTIVE:

        dist.all_reduce(train_loss_acc, op=dist.ReduceOp.SUM)  # Aggregate loss across all processes in DDP
        train_loss_acc /= ddp_world_size  # Average the loss across all DDP processes
        # Note! dist.ReduceOp.AVG is not a primitive operation and failed with 'goo' backend, while SUM is primitive operation that does not fail. Manually dividing by ddp_world_size later gets the average loss across all processes.
        
    # Clipping gradients: stabilize training 
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) if config.gradient_clipping else torch.tensor(0.0, device=device)  # Clip gradients to prevent exploding gradients, norm is 0 if clipping is disabled

    # Learning Rate Scheduler
    lr = get_lr(step, config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Update weights
    optimizer.step() 
    
    
    torch.cuda.synchronize() if device_type == "cuda" else None  # Synchronize for accurate timing
    t1 = time.time()
    dt = t1 - t0

    tokens_processed = train_loader.B * train_loader.T * grad_accumulation_steps * ddp_world_size # Total tokens processed in this step
    tokens_per_second = tokens_processed / dt

    # LOGGING EVALUATION AND TRAINING METRICS
    # ---------------------------------------------------------------------------------------------------------------------------------------
    if master_process:
        csv_row = f"{step}; {train_loss_log:.4}; {val_loss_log:.4}; {lr:.4}; {norm:.4}; {dt:.4}; {tokens_per_second};\n"
        #terminal_row = f"step {step}; train_loss_log {train_loss_log:.4}; val_loss_log {val_loss_log:.4}; lr {lr:.4}; norm {norm:.4}; dt {dt:.4}; tokens_per_second {tokens_per_second};\n"
        #cprint(terminal_row, compute_color)

        if step % config.eval_interval == 0 or last_step:
            with open(log_file, "a") as f:
                f.write(csv_row)

            # Store metrics for plotting
            train_loss_list.append(train_loss_log.cpu().item() if train_loss_log.is_cuda else train_loss_log.item())  # Convert to Python float for plotting
            val_loss_list.append(val_loss_log.cpu().item() if val_loss_log.is_cuda else val_loss_log.item())
            lr_list.append(lr)
            norm_list.append(norm.cpu().item() if norm.is_cuda else norm.item())
            duration_list.append(dt)
            tokens_per_sec_list.append((tokens_per_second))
            steps_list = np.linspace(0, step, num=len(train_loss_list), dtype=int)  # Create a list of steps for plotting
            # Plot metrics in real-time
            plt.figure(figsize=(20, 12))

            # Loss plot
            plt.plot(steps_list, train_loss_list, label='Train Loss')
            plt.plot(steps_list, val_loss_list, label='Val Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.yticks(np.arange(min(train_loss_list)-1, max(train_loss_list) + 1, 1))  # Set y-ticks for better readability
            plt.title('Loss over Steps')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{REPORT_DIR}/losses.png')

            # Learning rate plot
            plt.figure(figsize=(20, 12))

            plt.subplot(2, 2, 1)
            plt.plot(steps_list, lr_list, label='Learning Rate', color='orange')
            plt.xlabel('Step')
            plt.ylabel('Learning Rate')
            plt.yticks(np.arange(0, max(lr_list) + 5e-5, 5e-5))  # Set y-ticks for better readability
            plt.title('Learning Rate over Steps')
            plt.legend()
            plt.grid(True)

            # Gradient norm plot
            plt.subplot(2, 2, 2)
            plt.plot(steps_list, norm_list, label='Gradient Norm', color='green')
            plt.xlabel('Step')
            plt.ylabel('Norm')
            plt.title('Gradient Norm over Steps')
            plt.legend()
            plt.grid(True)

            # Duration plot
            plt.subplot(2, 2, 3)
            plt.plot(steps_list, duration_list, label='Duration', color='red')
            plt.xlabel('Step')
            plt.ylabel('Duration (s)')
            plt.yticks(np.arange(0, max(duration_list) + 0.1, 0.1))  # Set y-ticks for better readability
            plt.title('Duration per Step')
            plt.legend()
            plt.grid(True)

            # Tokens per second plot
            plt.subplot(2, 2, 4)
            plt.plot(steps_list, tokens_per_sec_list, label='Tokens/sec', color='purple')
            plt.xlabel('Step')
            plt.ylabel('Tokens/sec')
            plt.yticks(np.arange(0, max(tokens_per_sec_list) + 10000, 10000))  # Set y-ticks for better readability
            plt.title('Tokens/sec over Steps')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f'{REPORT_DIR}/training_metrics_summary.png')
  
        plt.close('all')
        pbar.postfix = f"train_loss_acc {train_loss_acc}"  # Update the progress bar postfix with the current step


 
  
end_train_loop = datetime.now()
if master_process:
    cprint(f"\nTraining completed in {end_train_loop - start_train_loop} (HH:MM:SS)", compute_color)
    # Save configuration as JSON
    with open(f'{REPORT_DIR}/config.json', 'w', encoding='utf-8') as f:
        config_dict = config.model_dump()
        #config_dict["compute_device"] = compute_device
        #config_dict["selected_tokenizer"] = tokenizer.name
        f.write(json.dumps(config_dict, indent=2))



dist.destroy_process_group() if DDP_ACTIVE else None  # Clean up DDP resources
