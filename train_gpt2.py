"""
Training Script
=====================================
"""

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

config = GPT2Config()
print(f"\n{'='*60}")
print("HYPERPARAMETERS")
print('='*60)
print(config.model_dump_json(indent=2))  # Print the configuration in JSON format


# =============================================================================
# DATA PREPARATION
# =============================================================================
print(f"\n{'='*60}")
print("DATA PREPARATION")
print('='*60)
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
train_loader = DataLoaderLite(B=16, T=1024)

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
print(f"\n{'='*60}")
print("MODEL INITIALIZATION")
print('='*60)

# Create model instance
#model = GPT2Model.from_pretrained('gpt2') # Load the pre-trained GPT-2 model from Huggingface
model = GPT2Model(config)
model.to(device) # this only works for the model, for tensors do tensor = tensor.to(device)
model = torch.compile(model) if TORCH_COMPILATION else model # New! https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html Speed ups the model with PyTorch 2.0's compile feature (optional, but recommended for performance). Speedup mainly comes from reducing Python overhead and GPU read/writes, and so the observed speedup may vary on factors such as model architecture and batch size

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

model_summary = (
    f"\nModel Details:\n"
    f"  Architecture: GPT-style Transformer\n"
    f"  Total parameters: {total_params:,}\n"
    f"  Trainable parameters: {trainable_params:,}\n"
    f"  Model size: ~{total_params * 4 / 1024**2:.2f} MB (float32)\n" # asu
    f"\n\nOptimizer: AdamW with learning rate PENDING\n"
)

print(model_summary)


# Print model architecture
# -----------------------
# Option 1
#print(colored(model, "green"))
# Option 2
for k, v in model.state_dict().items():
    print(colored(f"{k}: {v.shape} - {v.dtype}", "green"))  # Print each parameter's shape and dtype


# =============================================================================
# TRAINING 
# =============================================================================
print(f"\n{'='*60}")
print("TRAINING")
print('='*60)
print(f"\nStarting training loop...")

start_train = datetime.now() # Record start time of training
# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for step in range(50):
    t0 = time.time()
    
    # TRAINING PHASE
    
    # Sample a batch random of training data
    xb, yb = train_loader.next_batch()
    
    if AUTOCAST:
        with torch.autocast(device_type=compute_device, dtype=torch.bfloat16 if compute_device == "cuda" else torch.float32): # New! Use autocast for mixed precision training: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            logits, loss = model(xb, yb) # Forward pass optimized for speed
    else:
        logits, loss = model(xb, yb) # Forward pass without autocast
    
    optimizer.zero_grad() # Reset gradients
    loss.backward() # Backward pass
    optimizer.step() # Update weights 
    torch.cuda.synchronize() if compute_device == "cuda" else None  # Synchronize for accurate timing
    t1 = time.time()
    duration = t1 - t0

    print(f"Step {step+1:03d} | Loss: {loss.item():.4f} | Time: {duration:.2f}s | Tokens/sec: {train_loader.batch_size * train_loader.seq_size / duration:.2f}")

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
#         # right now idx is (B,T) where B = 5, T = 8

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

