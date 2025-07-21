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
from model_gpt import GPTModel

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
TENSOR_CORES = False  # Set to True to enable Tensor Cores for faster matrix multiplications on supported GPUs
TORCH_COMPILATION = False  # Set to True to enable PyTorch 2.0's compile feature for performance optimization
AUTOCAST = False  # Mixed precision training https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html

torch.set_float32_matmul_precision('high') if TENSOR_CORES else None

# File paths
TRAIN_ID = datetime.now().strftime("%Y%m%d_%H%M") # Unique identifier for this training session
DATA_PATH = 'data/tiny_shakespeare/text/tinyshakespeare.txt'
REPORT_DIR = f'logs/GPT_training_{TRAIN_ID}_{compute_device}'
# Create plots directory
os.makedirs(REPORT_DIR, exist_ok=True)
# =============================================================================
# HYPERPARAMETERS
# =============================================================================

config = GPTConfig(compute_device=compute_device)

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

class DataLoaderFromTxt:
    def __init__(self, B:int, T:int, process_rank:int=0, num_processes:int=1, split:Literal['train', 'val']='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split  # Store split type
        assert split in {'train','val'}

        # At init load tokens from disk and store them in memory
        try:
            with open(DATA_PATH, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        
        tokens = tokenizer.encode(text)  # Encode the text into tokens
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        # Split the data into training and validation sets
        split_idx = int(config.train_val_ratio * len(self.tokens)) 
        self.train_tokens = self.tokens[:split_idx] # train split 
        self.val_tokens = self.tokens[split_idx:] # validation split
        
        if master_process:
            cprint(f"loaded {len(self.tokens)} tokens", compute_color)
            cprint(f"1 epoch = {len(self.tokens)// (self.B*self.T)} batches", compute_color)
            data_preparation_summary = (
            f"\nTokenziation summary:\n"
            f"  Tokenizer: {tokenizer.name}\n"
            f"  Tokenized text: {len(self.tokens):,} tokens\n"
            f"  Vocabulary size: {tokenizer.n_vocab} unique tokens\n"
            f"\nData split:\n"
            f"  Training:   {len(self.train_tokens):,} tokens ({len(self.train_tokens)/len(self.tokens)*100:.1f}%)\n"
            f"  Validation: {len(self.val_tokens):,} tokens ({len(self.val_tokens)/len(self.tokens)*100:.1f}%)\n"
            )
        
            cprint(data_preparation_summary, compute_color)

        self.reset()
        self.vocab_size = tokenizer.n_vocab
            
    def reset(self):
        """
        Reset the data loader to the initial state.
        """
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
        """
        B, T = self.B, self.T
        data = self.train_tokens if self.split == 'train' else self.val_tokens  # Use train or validation tokens based on split
        # v1
        # starting_idx = torch.randint(len(data) - T, (B,))         # Sample random starting indices for each sequence in the batch
        # xb = torch.stack([data[i:i+T] for i in starting_idx])          # Extract sequences and create targets
        # yb = torch.stack([data[i+1:i+T+1] for i in starting_idx])      # Shift the sequence by one to the right to create the target

        # v2
        #Ensure current_position does not exceed data length
        if self.current_position >= len(data):
            self.reset()

        buf = data[self.current_position: self.current_position + B * T + 1]

        # Safeguard against empty slices
        if len(buf) < B * T + 1:
            raise ValueError(f"Insufficient data to generate batch: requested {B * T + 1}, but got {len(buf)}")

        xb = buf[:-1].view(B,T) # inputs
        yb = buf[1:].view(B,T)  # targets (shifted) 
        self.current_position += B * T * self.num_processes # advance the current position by B*T*num_processes tokens to ensure each process gets a unique subset of the data

        # If we reach the end of the tokens, reset to the beginning
        if self.current_position + (B * T * self.num_processes + 1) > len(data):
            self.reset()

        # Move to compute device (GPU or CPU)
        xb, yb = xb.to(device), yb.to(device)

        return xb, yb

# Create dataloader instance
train_loader = DataLoaderFromTxt(B=config.batch_size, T=config.seq_size, process_rank=ddp_rank, num_processes=ddp_world_size, split = 'train')
val_loader = DataLoaderFromTxt(B=config.batch_size, T=config.seq_size, process_rank=ddp_rank, num_processes=ddp_world_size, split = 'val')

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
model = GPTModel(config)
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

def train_val_loss_plot(train_losses: list, val_losses: list, steps_recorded: list):
    step = steps_recorded[-1] if steps_recorded else 0  # Get the last recorded step
    
    # Initialize Loss Plot
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.set_xlabel('Steps')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title(f'Training and Validation Loss at step {step}')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_xlim(0, config.training_steps + 2 * config.eval_interval)
    ax_loss.tick_params(axis='x', labelsize=6)  # x-axis ticks
    ax_loss.plot(steps_recorded, train_losses, color = "tab:blue", label='Training Loss', linewidth=2)
    ax_loss.plot(steps_recorded, val_losses, color = "tab:orange", label='Validation Loss', linewidth=2)
    ax_loss.legend(loc='upper right')
    ax_loss.set_ylim(min(min(train_losses), min(val_losses))*0.9, max(max(train_losses), max(val_losses)) * 1.1 if train_losses else 1)
    fig_loss.canvas.draw()
    fig_loss.savefig(f'{REPORT_DIR}/losses.png', dpi=300, bbox_inches='tight')
    plt.close(fig_loss)  # Close the figure to free memory

    
def pwe_plot(model: GPTModel, step: int, starting_limits: tuple = (None, None)):
    """
    Plot the position embeddings of the model.
    """
    layer_prefix = "module." if DDP_ACTIVE else ""
    layer_prefix += "_orig_mod." if TORCH_COMPILATION else ""
    layer_name = "wpe"
    weights = model.state_dict()[f"{layer_prefix}{layer_name}.weight"].cpu().detach().numpy()

    if starting_limits == (None, None):
        starting_limits = (weights.min(), weights.max())
        
    # --- Plot 1: Heatmap of the position embeddings ---
    fig_pwe, ax_pwe = plt.subplots(figsize=(10, 6))

    values_pwe = ax_pwe.imshow(weights, vmin=starting_limits[0], vmax=starting_limits[1])

    ax_pwe.set_xlabel('Position Embedding (n_embd)')
    tick_positions = [np.arange(weights.shape[1])[0], np.arange(weights.shape[1])[len(np.arange(weights.shape[1]-1))//2], np.arange(weights.shape[1])[-1]]
    ax_pwe.set_xticks(tick_positions)
    #ax_pwe.set_xlim(0, weights.shape[1]-1)  # Set x-axis limits to avoid empty space

    ax_pwe.set_ylabel('Sequence Position (T)')
    tick_positions = [np.arange(weights.shape[0])[0], np.arange(weights.shape[0])[len(np.arange(weights.shape[0]))//2], np.arange(weights.shape[0])[-1]]
    ax_pwe.set_yticks(tick_positions)
    #ax_pwe.set_ylim(0, weights.shape[0]-1)  # Set y-axis limits to avoid empty space

    ax_pwe.set_title(f'Position Embeddings Weights at step {step}')
    
    cbar = fig_pwe.colorbar(values_pwe, ax=ax_pwe, label='Weight Value')
    #cbar.set_ticks(np.arange(weights.min(), weights.max(), 0.1))
    
    fig_pwe.savefig(f"{REPORT_DIR}/{layer_name}.png")
    plt.close(fig_pwe)  # Close the figure to free memory
    
    # --- Plot 2: Histogram of values ---
    fig_pwe_hist, ax_pwe_hist = plt.subplots(figsize=(10, 4))
    ax_pwe_hist.hist(weights.flatten(), bins=100, color='gray', edgecolor='black')
    ax_pwe_hist.set_title(f'Position Embedding Value Distribution at step {step}')
    ax_pwe_hist.set_xlabel('Weight Value')
    ax_pwe_hist.set_ylabel('Frequency')
    ax_pwe_hist.grid(True, linestyle='--', alpha=0.6)

    fig_pwe_hist.tight_layout()
    fig_pwe_hist.savefig(f"{REPORT_DIR}/{layer_name}_hist.png")
    plt.close(fig_pwe_hist)
    
    return starting_limits
    
def ffn_weight_plot(model: GPTModel, step: int, starting_limits: tuple):
    """
    Plot the FFN first-layer weights of the first transformer block.
    """
    layer_prefix = "module." if DDP_ACTIVE else ""
    layer_prefix += "_orig_mod." if TORCH_COMPILATION else ""
    layer_name = "transformer_blocks.0.mlp.c_fc"
    weights = model.state_dict()[f"{layer_prefix}{layer_name}.weight"].cpu().detach().numpy()
    
    if starting_limits == (None, None):
        starting_limits = (weights.min(), weights.max())
    
    # --- Plot 1: Heatmap of the FFN weights ---
    fig_ffn, ax_ffn = plt.subplots(figsize=(10, 6))

    im = ax_ffn.imshow(weights, vmin=starting_limits[0], vmax=starting_limits[1])

    ax_ffn.set_xlabel('Input Features (n_embd)')
    tick_positions = [np.arange(weights.shape[1])[0], np.arange(weights.shape[1])[len(np.arange(weights.shape[1]))//2], np.arange(weights.shape[1])[-1]]
    ax_ffn.set_xticks(tick_positions)
    #ax_ffn.set_xlim(0, weights.shape[1]-1)  # Set x-axis limits to avoid empty space
    ax_ffn.set_ylabel('FFN Neurons (4*n_embd)')
    tick_positions = [np.arange(weights.shape[0])[0], np.arange(weights.shape[0])[len(np.arange(weights.shape[0]))//2], np.arange(weights.shape[0])[-1]]
    ax_ffn.set_yticks(tick_positions)
    #ax_ffn.set_ylim(0, weights.shape[0]-1)  # Set y-axis limits to avoid empty space

    ax_ffn.set_title(f'FFN Layer 0 Weights (Block 0) at step {step}')
    
    cbar = fig_ffn.colorbar(im, ax=ax_ffn, label='Weight Value')
    #cbar.set_ticks(np.arange(weights.min(), weights.max(), 0.1))

    fig_ffn.savefig(f"{REPORT_DIR}/{layer_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig_ffn)
    
    # --- Plot 2: Histogram of values ---
    fig_ffn_hist, ax_ffn_hist = plt.subplots(figsize=(10, 4))
    ax_ffn_hist.hist(weights.flatten(), bins=100, color='gray', edgecolor='black')
    ax_ffn_hist.set_title(f'FFN Layer 0 Weight Distribution (Block 0) at step {step}')
    ax_ffn_hist.set_xlabel('Weight Value')
    ax_ffn_hist.set_ylabel('Frequency')
    ax_ffn_hist.grid(True, linestyle='--', alpha=0.6)

    fig_ffn_hist.tight_layout()
    fig_ffn_hist.savefig(f"{REPORT_DIR}/{layer_name}_hist.png")
    plt.close(fig_ffn_hist)
    
    return starting_limits 
    
        
def train_val_loss_csv(train_losses: list, val_losses: list, steps_recorder: list):
    """
    Write training and validation losses to CSV, overwriting existing content.
    """
    with open(f'{REPORT_DIR}/losses.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Step', 'Train_Loss', 'Val_Loss'])  # Header row
        
        for step, train_loss, val_loss in zip(steps_recorder, train_losses, val_losses):
            writer.writerow([step, float(train_loss), float(val_loss)])
            
starting_limits_ffn = (None, None)  # Initialize starting limits for FFN weight plot
starting_limits_pwe = (None, None)  # Initialize starting limits for PWE weight plot
def update_visualizations(step, train_losses, val_losses, losses, steps_recorded, model, starting_limits_ffn, starting_limits_pwe):
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        steps_recorded.append(step)
        train_val_loss_plot(train_losses, val_losses, steps_recorded)
        train_val_loss_csv(train_losses, val_losses, steps_recorded)
        starting_limits_pwe = pwe_plot(model, step, starting_limits_pwe)
        starting_limits_ffn = ffn_weight_plot(model, step, starting_limits_ffn)
        return starting_limits_ffn, starting_limits_pwe

@torch.no_grad()
def estimate_loss():
    """
    Calculates mean loss over val_loss_steps batches, for each the training and the validation splits.
    """
    mean_losses = {}
    model.eval() # indicate the model is in 'evaluation' mode
    for split in ['train', 'val']:
        losses = torch.zeros(config.val_loss_steps)
        for k in range(config.val_loss_steps): 
            X,Y = train_loader.next_batch() if split == 'train' else val_loader.next_batch() # Get a batch of data
            _, loss = model(X,Y)
            losses[k] = loss.item()
        mean_losses[split] = losses.mean()
    model.train() # indicate the model is in 'training' mode
    return mean_losses


# =============================================================================
# TRAINING 
# =============================================================================

cprint("TRAINING", compute_color)


# Initialize lists to store metrics for plotting
train_losses, val_losses, steps_recorded, final_losses = [], [], [], {}
losses_accumulated, lrs, norms, durations, tokens_per_sec = [], [], [], [], []

# Initialize optimizer
if DDP_ACTIVE:
    optimizer = model.module.configure_optimizers(config, device_type=device_type)
else:
    optimizer = model.configure_optimizers(config, device_type=device_type)

log_file = os.path.join(REPORT_DIR, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

start_train_loop = datetime.now()
for step in trange(config.training_steps, desc="Training steps", unit="step", disable=False):
    last_step = (step == config.training_steps - 1)
    t0 = time.time()


    # EVALUATION PHASE
    # -------------------------------------------------------------------------------------------------------------------------------------
    # (1) Once every eval_interval steps, evaluate the model on the validation set for val_loss_steps steps
    if step % config.eval_interval == 0 or last_step:
        losses = estimate_loss()
        starting_limits_ffn, starting_limits_pwe = update_visualizations(step, train_losses, val_losses, losses, steps_recorded, model, starting_limits_ffn, starting_limits_pwe)
    
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

    if master_process:
        cprint(f"Step {step+1:03d} | Train Loss Accum: {train_loss_acc:.4} | lr: {lr:.4e} | norm: {norm:.4e} | dt: {dt:.4}s | Tokens/sec: {tokens_per_second}", compute_color)
        with open(log_file, "a") as f:
            f.write(f"{step} train {train_loss_acc.item():.6f}\n")
        # Store metrics for plotting
        losses_accumulated.append(train_loss_acc.cpu().item() if train_loss_acc.is_cuda else train_loss_acc.item())  # Convert to Python float for plotting
        lrs.append(lr)
        norms.append(norm.cpu().item() if norm.is_cuda else norm.item())
        durations.append(dt)
        tokens_per_sec.append((tokens_per_second))
        
end_train_loop = datetime.now()

# Estimate loss after the last training step
# This is to ensure the final losses are recorded even if the last step is not an evaluation 
step = config.training_steps - 1
losses = estimate_loss()

if master_process:
    cprint(f"\nTraining completed in {end_train_loop - start_train_loop} (HH:MM:SS)", compute_color)
    starting_limits_ffn, starting_limits_pwe = update_visualizations(step, train_losses, val_losses, losses, steps_recorded, model, starting_limits_ffn, starting_limits_pwe)


    final_losses = losses # Store final losses for reporting

    training_summary = (
        f"\nTraining Summary:\n"
        f"  Final training loss: {final_losses['train']:.4f}\n"
        f"  Final validation loss: {final_losses['val']:.4f}\n"
    )
    print(training_summary)
# =============================================================================
# INFERENCE & TEXT GENERATION
# =============================================================================
# Context tokens
context_text = "\n"
tokenizer
context_tokens = tokenizer.encode(context_text)
context_tokens = torch.tensor(context_tokens, dtype=torch.long)
idx = context_tokens.to(device).unsqueeze(0)  # Shape becomes (1, seq_len)

# Generate from context tokens
generated_tokens = model.module.generate(idx, max_new_tokens=500)[0].tolist()
generated_text = tokenizer.decode(generated_tokens)
print("Generated text: <START>", colored(generated_text, "cyan"), "<END>")
    

# =============================================================================
# REPORT GENERATION
# =============================================================================

report = f"""# Training Report

**Training Session:** `{TRAIN_ID}`

**Training Device:** `{compute_device}`

## 🎯 Training Result

- **Final Training Loss:** `{final_losses['train']:.4f}` | **Final Validation Loss:** `{final_losses['val']:.4f}`
- **Training duration:** `{end_train_loop - start_train_loop}` (HH:MM:SS)

### 📈 Loss evolution

<img src="losses.png" alt="Training and Validation Loss" width="60%"/>

## Generation Example:
```
{generated_text}
```

## Hyperparameters and Configuration

| Hyperparameters and Architecture |                            | | | Model Dimension         |                                                  | | | Dataset Details      |                                                                         |
|----------------------------------|----------------------------|-|-|-------------------------|--------------------------------------------------|-|-|----------------------|-------------------------------------------------------------------------|
| seq_size                       | `{config.seq_size}` tokens   | | | Total Parameters        | `{total_params:,}`                               | | | Dataset              | `{DATA_PATH}`                                                           |
| batch_size                     | `{config.batch_size}`        | | | Trainable Parameters    | `{trainable_params:,}`                           | | | Dataset Size         | `{len(train_loader.train_tokens) + len(val_loader.val_tokens):,}` tokens  |
| n_embd (dim)                   | `{config.n_embd}`            | | | Model Size              | ~`{total_params * 4 / 1024**2:.2f}` MB (float32) | | | Training Tokens      | `{len(train_loader.train_tokens):,}` tokens ({config.train_val_ratio:.1%})|
| n_head                         | `{config.n_head}`            | | | Optimizer               | AdamW with learning rate `{config.lr}`| | | Validation Tokens    | `{len(val_loader.val_tokens):,}` tokens ({1-config.train_val_ratio:.1%})|
| n_layer                        | `{config.n_layer}`           | | | Tokenizer               | `{tokenizer.name}`                               | | |                      |                                                                         |
| dropout                        | `{config.dropout}`           | | | Vocabulary Size         | `{tokenizer.n_vocab:,}` tokens                | | |                      |                                                                         |
| training_steps                 | `{config.training_steps:,}`  | | |                         |                                                  | | |                      |                                                                         |
| lr                  | `{config.lr}`     | | |                         |                                                  | | |                      |                                                                         |
| eval_interval                  | `{config.eval_interval}`     | | |                         |                                                  | | |                      |                                                                         |
| val_loss_steps                     | `{config.val_loss_steps}`        | | |                         |                                                  | | |                      |                                                                         |


"""

with open(f'{REPORT_DIR}/report.md', 'w', encoding='utf-8') as f:
    f.write(report)
    
# Save configuration as JSON
with open(f'{REPORT_DIR}/config.json', 'w', encoding='utf-8') as f:
    config_dict = config.model_dump()
    #config_dict["compute_device"] = compute_device
    #config_dict["selected_tokenizer"] = tokenizer.name
    f.write(json.dumps(config_dict, indent=2))




dist.destroy_process_group() if DDP_ACTIVE else None  # Clean up DDP resources
