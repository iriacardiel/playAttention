"""
Training Script
=====================================
"""

import math
import os
import csv
import json
from typing import Tuple, Any
from datetime import datetime
import time
import numpy as np
from tqdm import trange
from termcolor import colored
import matplotlib.pyplot as plt
from custom_tokenizers import tiktoken_tokenizer, char_level_tokenizer

import torch
from torch.nn import functional as F

from Config import GPT2Config
from model_gpt2 import GPT2Model



# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# For reproducibility
seed = 1337
torch.manual_seed(seed)

# Determine computation device (GPU or CPU)
compute_device = "cpu"
if torch.cuda.is_available():
    compute_device = "cuda"
    torch.cuda.manual_seed(seed) 
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    compute_device = "mps"
    
print("Using device:", compute_device)

device = torch.device(compute_device)

# GPU Settings
TENSOR_CORES = True  # Set to True to enable Tensor Cores for faster matrix multiplications on supported GPUs
TORCH_COMPILATION = True  # Set to True to enable PyTorch 2.0's compile feature for performance optimization
AUTOCAST = True  # Set to True to enable mixed precision training with autocast for performance optimization
PRETRAINED = False  # Set to False if you want to use randomly initialized weights

# New! Set high precision for matmul operations TF32: This will be faster on GPUs with Tensor Cores
torch.set_float32_matmul_precision('high') if TENSOR_CORES else None


# File paths
TRAIN_ID = datetime.now().strftime("%Y%m%d_%H%M") # Unique identifier for this training session
DATA_PATH = 'data/tinyshakespeare.txt'
# =============================================================================
# HYPERPARAMETERS
# =============================================================================

config = GPT2Config(compute_device=compute_device)


print("HYPERPARAMETERS")

print(config.model_dump_json(indent=2))  # Print the configuration in JSON format


# =============================================================================
# DATA PREPARATION
# =============================================================================

print("DATA PREPARATION")

tokenizer = tiktoken_tokenizer
class DataLoaderLite:
    def __init__(self, B, T):
        self.batch_size = B
        self.seq_size = T

        # At init load tokens from disk and store them in memory
        try:
            with open(DATA_PATH, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        
        tokens = tokenizer.encode(text)  # Encode the text into tokens
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)// (self.batch_size*self.seq_size)} batches")
        
        self.vocab_size = tokenizer.n_vocab

        # state
        self.current_position = 0
        
        data_preparation_summary = (
            f"\nTokenziation summary:\n"
            f"  Tokenizer: {tokenizer.name}\n"
            f"  Tokenized text: {len(self.tokens):,} tokens\n"
            f"  Vocabulary size: {tokenizer.n_vocab} unique tokens\n"
        )
        
        print(data_preparation_summary)
        
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
        """
        B,T = self.batch_size, self.seq_size
        
        buf = self.tokens[self.current_position: self.current_position+ B*T + 1]
        xb = buf[:-1].view(B,T) # inputs
        yb = buf[1:].view(B,T)  # targets (shifted) 
        self.current_position += B*T # advance the current position by B*T tokens

        # If we reach the end of the tokens, reset to the beginning
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0
        
        # Move to compute device (GPU or CPU)
        xb, yb = xb.to(device), yb.to(device)
        
        return xb, yb  

# Create dataloader instance
train_loader = DataLoaderLite(B=config.batch_size, T=config.seq_size)

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

print("MODEL INITIALIZATION")


# Create model instance
#model = GPT2Model.from_pretrained('gpt2') # Load the pre-trained GPT-2 model from Huggingface
model = GPT2Model(config)
model.to(device) # this only works for the model, for tensors do tensor = tensor.to(device)

# New! https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html 
# Speed ups the model with PyTorch 2.0's compile feature (optional, but recommended for performance). 
# Speedup mainly comes from reducing Python overhead and GPU read/writes, and so the observed speedup 
# may vary on factors such as model architecture and batch size
model = torch.compile(model) if TORCH_COMPILATION else model 

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size = total_params * 4 / 1024**2  # Size in MB (assuming float32, 4 bytes per parameter)

model_summary = (
    f"\nModel Details:\n"
    f"  Architecture: GPT-style Transformer\n"
    f"  Total parameters: {total_params:,}\n"
    f"  Trainable parameters: {trainable_params:,}\n"
    f"  Model size: ~{model_size:.2f} MB (float32)\n"
    f"\n\nOptimizer: AdamW with learning rate {config.learning_rate}\n"
)

print(model_summary)


# Print model architecture
# -----------------------
# Option 1
#print(colored(model, "green"))
# Option 2
# for k, v in model.state_dict().items():
#     print(colored(f"{k}: {v.shape} - {v.dtype}", "green"))  # Print each parameter's shape and dtype


# =============================================================================
# TRAINING 
# =============================================================================

print("TRAINING")

# New! Learning rate scheduler function
def get_lr(step: int, config: GPT2Config) -> float:
    """
    Get the learning rate for the current training step.
    
    Uses a cosine decay schedule with warmup.
    """
    max_lr = config.learning_rate  # Maximum learning rate
    min_lr = max_lr * 0.1  # Minimum learning rate (10% of max)
    warmup_steps = config.warmup_steps  # Number of warmup steps
    total_steps = config.training_steps  # Total training steps
    # (1) warmup
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps  # Linear warmup
    # (2) decay
    elif step >= warmup_steps and step <= total_steps:
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)
    # (3) after training
    else: 
        return min_lr
    
    
    
start_train_loop = datetime.now() # Record start time of training
# Initialize optimizer
#optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.eps) # New! Use AdamW optimizer with beta1, beta2 and epsilon parameters for better convergence
optimizer = model.configure_optimizers(config, device_type=compute_device)
# Initialize lists to store metrics for plotting
losses = []
lrs = []
norms = []
durations = []
tokens_per_sec = []

for step in range(config.training_steps):
    
    t0 = time.time()
    
    # TRAINING PHASE

    # Sample a batch random of training data
    xb, yb = train_loader.next_batch()
    
    # New! Use autocast for mixed precision training: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    if AUTOCAST:
        with torch.autocast(device_type=compute_device, dtype=torch.bfloat16 if compute_device == "cuda" else torch.float32): 
            logits, loss = model(xb, yb) # Forward pass
    else:
        logits, loss = model(xb, yb) # Forward pass
    
    optimizer.zero_grad() # Reset gradients
    loss.backward() # Backward pass
    
    # New! Clipping gradients to stabilize training and avoid exploding gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
    
    # New! Determine and set the learning rate for this step
    lr = get_lr(step, config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    optimizer.step() # Update weights

    torch.cuda.synchronize() if compute_device == "cuda" else None  # Synchronize for accurate timing
    t1 = time.time()
    duration = t1 - t0

    # Store metrics for plotting
    losses.append(loss.item())
    lrs.append(lr)
    norms.append(norm.cpu().item() if norm.is_cuda else norm.item())
    durations.append(duration)
    tokens_per_sec.append((train_loader.batch_size * train_loader.seq_size / duration))

    print(f"Step {step+1:03d} | Loss: {loss.item():.4} | lr: {lr:.4e} | norm: {norm:.4e} | dt: {duration:.4}s | Tokens/sec: {train_loader.batch_size * train_loader.seq_size / duration}")

end_train_loop = datetime.now() # Record start time of training
print(f"\nTraining completed in {end_train_loop - start_train_loop} (HH:MM:SS)")

# Save plots to files
plt.figure(figsize=(20, 12))

# Loss plot
plt.subplot(2, 2, 1)
plt.plot(losses, label='Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.yticks(np.arange(min(losses)-1, max(losses) + 1, 1))  # Set y-ticks for better readability
plt.title('Loss over Steps')
plt.legend()
plt.grid(True)

# Learning rate plot
plt.subplot(2, 2, 2)
plt.plot(lrs, label='Learning Rate', color='orange')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.yticks(np.arange(0, max(lrs) + 5e-5, 5e-5))  # Set y-ticks for better readability
plt.title('Learning Rate over Steps')
plt.legend()
plt.grid(True)

# # Gradient norm plot
# plt.subplot(2, 3, 3)
# plt.plot(norms, label='Gradient Norm', color='green')
# plt.xlabel('Step')
# plt.ylabel('Norm')
# plt.title('Gradient Norm over Steps')
# plt.legend()
# plt.grid(True)

# Duration plot
plt.subplot(2, 2, 3)
plt.plot(durations, label='Duration', color='red')
plt.xlabel('Step')
plt.ylabel('Duration (s)')
plt.yticks(np.arange(0, max(durations) + 0.1, 0.1))  # Set y-ticks for better readability
plt.title('Duration per Step')
plt.legend()
plt.grid(True)

# Tokens per second plot
plt.subplot(2, 2, 4)
plt.plot(tokens_per_sec, label='Tokens/sec', color='purple')
plt.xlabel('Step')
plt.ylabel('Tokens/sec')
plt.yticks(np.arange(0, max(tokens_per_sec) + 10000, 10000))  # Set y-ticks for better readability
plt.title('Tokens/sec over Steps')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics_summary.png')

# # =============================================================================
# # INFERENCE & TEXT GENERATION
# # =============================================================================
# # Context tokens
# context_text = "Hello, I'm a language model,"
# context_tokens = tokenizer.encode(context_text)
# context_tokens = torch.tensor(context_tokens, dtype=torch.long) # 1, 8

# # Manually generating a batch with the same sequence context_tokens 5 times 
# num_generated_sequences = 5
# context_tokens = context_tokens.unsqueeze(0).repeat(num_generated_sequences, 1) # 5, 8
# idx = context_tokens.to(device)

# max_new_tokens = 30

# # Generate from context tokens (manually instead of using model.generate() not implemented yet)
# while idx.size(1) < max_new_tokens:

#     with torch.no_grad():
#         # right now idx is (B, T) where B = 5, T = 8

#         # forward the model
#         logits, _ = model(idx)  # (B, T, vocab_size)

#         # Focus on the last time step (next token prediction)
#         logits = logits[:, -1, :] # (B, vocab_size) 

#         # Convert to probabilities
#         probs = F.softmax(logits, dim = -1) # (B, vocab_size)

#         # Do top-k sampling of 50 (huggingface default pipeline)
#         # topk_probs here becomes (5,50) and topk_indices becomes (5,50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # Get top 50 probabilities and indices
#         ix = torch.multinomial(topk_probs, num_samples=1)  # Select a token from the top 50 probabilities
#         xcol = torch.gather(topk_indices, -1, ix) # Gather the corresponding token indices based on the sampled probabilities
        
#         # Append to sequence
#         idx = torch.cat((idx, xcol), dim = 1) # (B, T+1)

# # print the generated sequences
# for i in range(num_generated_sequences):
#     generated_tokens = idx[i, :max_new_tokens].tolist() # Get the generated tokens for this sequence
#     generated_text = tokenizer.decode(generated_tokens)  # Decode the tokens to text
#     print(f"Generated text {i+1}: <START>", colored(generated_text, "cyan"), "<END>")

