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

from Config import GPTConfig
from model_gpt import GPTModel


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

# File paths
TRAIN_ID = datetime.now().strftime("%Y%m%d_%H%M") # Unique identifier for this training session
DATA_PATH = 'data/tinyshakespeare.txt'
REPORT_DIR = f'reports/training_{TRAIN_ID}_{compute_device}'
# Create plots directory
os.makedirs(REPORT_DIR, exist_ok=True)
# =============================================================================
# HYPERPARAMETERS
# =============================================================================

config = GPTConfig(compute_device=compute_device)


print("HYPERPARAMETERS")

print(config.model_dump_json(indent=2))  # Print the configuration in JSON format


# =============================================================================
# DATA PREPARATION
# =============================================================================

print("DATA PREPARATION")

tokenizer = char_level_tokenizer if config.selected_tokenizer == char_level_tokenizer.name else tiktoken_tokenizer
config.vocab_size = tokenizer.n_vocab  # Set vocabulary size based on the tokenizer
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


        # Generate splits: train , val
        split_idx = int(config.train_val_ratio * len(self.tokens)) 
        train_data = self.tokens[:split_idx] # train split
        val_data = self.tokens[split_idx:] # validation split
        
        self.train_data = train_data
        self.val_data = val_data

        
        data_preparation_summary = (
            f"\nTokenziation summary:\n"
            f"  Tokenizer: {tokenizer.name}\n"
            f"  Tokenized text: {len(self.tokens):,} tokens\n"
            f"  Vocabulary size: {tokenizer.n_vocab} unique tokens\n"
            f"\nData split:\n"
            f"  Training:   {len(train_data):,} tokens ({len(train_data)/len(self.tokens)*100:.1f}%)\n"
            f"  Validation: {len(val_data):,} tokens ({len(val_data)/len(self.tokens)*100:.1f}%)\n"
        )
        
        print(data_preparation_summary)
        
        
    
    # Batch generator
    def next_batch(self, split_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        data = self.train_data if split_type == "train" else self.val_data
        starting_idx = torch.randint(len(data) - T, (B,))         # Sample random starting indices for each sequence in the batch
        xb = torch.stack([data[i:i+T] for i in starting_idx])          # Extract sequences and create targets
        yb = torch.stack([data[i+1:i+T+1] for i in starting_idx])         # Shift the sequence by one to the right to create the target       
        
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
model = GPTModel(config)
model.to(device) # this only works for the model, for tensors do tensor = tensor.to(device)

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

model_summary = (
    f"\nModel Details:\n"
    f"  Architecture: GPT-style Transformer\n"
    f"  Total parameters: {total_params:,}\n"
    f"  Trainable parameters: {trainable_params:,}\n"
    f"  Model size: ~{total_params * 4 / 1024**2:.2f} MB (float32)\n" # asu
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
# TRAINING UTILITIES
# =============================================================================
        
        
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
    layer_name = "wpe"
    weights = model.state_dict()[f"{layer_name}.weight"].cpu().detach().numpy()

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
    layer_name = "transformer_blocks.0.mlp.c_fc"
    weights = model.state_dict()[f"{layer_name}.weight"].cpu().detach().numpy()
    
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
    Calculates mean loss over eval_iters batches, for each the training and the validation splits.
    """
    mean_losses = {}
    model.eval() # indicate the model is in 'evaluation' mode
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters): 
            X,Y = train_loader.next_batch(split_type=split)
            _, loss = model(X,Y)
            losses[k] = loss.item()
        mean_losses[split] = losses.mean()
    model.train() # indicate the model is in 'training' mode
    return mean_losses


# =============================================================================
# TRAINING 
# =============================================================================

print("TRAINING")


# Initialize lists to store losses for plotting and logging
train_losses, val_losses, steps_recorded, final_losses = [], [], [], {}


start_train = datetime.now() # Record start time of training
# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
for step in trange(config.training_steps, desc="Training steps", unit="step", disable=False):
    # EVALUATION PHASE
    if step % config.eval_interval == 0: # Every eval_interval steps pause training and evaluate the mean loss on train and val sets on eval_iters batches
        
        losses = estimate_loss()
        starting_limits_ffn, starting_limits_pwe = update_visualizations(step, train_losses, val_losses, losses, steps_recorded, model, starting_limits_ffn, starting_limits_pwe)
    
    t0 = time.time()
    # TRAINING PHASE

    # Sample a batch random of training data
    xb, yb = train_loader.next_batch(split_type='train')

    logits, loss = model(xb, yb) # Forward pass
    optimizer.zero_grad() # Reset gradients
    loss.backward() # Backward pass
    optimizer.step() # Update weights 
    torch.cuda.synchronize() if compute_device == "cuda" else None  # Synchronize for accurate timing
    t1 = time.time()
    duration = t1 - t0

    print(f"Step {step+1:03d} | Loss: {loss.item():.4f} | Time: {duration:.2f}s | Tokens/sec: {train_loader.batch_size * train_loader.seq_size / duration:.2f}")


# Estimate loss after the last training step
# This is to ensure the final losses are recorded even if the last step is not an evaluation 
step = config.training_steps - 1
losses = estimate_loss()
starting_limits_ffn, starting_limits_pwe = update_visualizations(step, train_losses, val_losses, losses, steps_recorded, model, starting_limits_ffn, starting_limits_pwe)


end_train = datetime.now() # Record start time of training
total_time = end_train - start_train
final_losses = losses # Store final losses for reporting

training_summary = (
    f"\nTraining Summary:\n"
    f"  Final training loss: {final_losses['train']:.4f}\n"
    f"  Final validation loss: {final_losses['val']:.4f}\n"
    f"  Training duration: {total_time}\n"
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
generated_tokens = model.generate(idx, max_new_tokens=500)[0].tolist()
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
- **Training duration:** `{total_time}`

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
| batch_size                     | `{config.batch_size}`        | | | Trainable Parameters    | `{trainable_params:,}`                           | | | Dataset Size         | `{len(train_loader.train_data) + len(train_loader.val_data):,}` tokens  |
| n_embd (dim)                   | `{config.n_embd}`            | | | Model Size              | ~`{total_params * 4 / 1024**2:.2f}` MB (float32) | | | Training Tokens      | `{len(train_loader.train_data):,}` tokens ({config.train_val_ratio:.1%})|
| n_head                         | `{config.n_head}`            | | | Optimizer               | AdamW with learning rate `{config.learning_rate}`| | | Validation Tokens    | `{len(train_loader.val_data):,}` tokens ({1-config.train_val_ratio:.1%})|
| n_layer                        | `{config.n_layer}`           | | | Tokenizer               | `{tokenizer.name}`                               | | |                      |                                                                         |
| dropout                        | `{config.dropout}`           | | | Vocabulary Size         | `{tokenizer.n_vocab:,}` tokens                | | |                      |                                                                         |
| training_steps                 | `{config.training_steps:,}`  | | |                         |                                                  | | |                      |                                                                         |
| learning_rate                  | `{config.learning_rate}`     | | |                         |                                                  | | |                      |                                                                         |
| eval_interval                  | `{config.eval_interval}`     | | |                         |                                                  | | |                      |                                                                         |
| eval_iters                     | `{config.eval_iters}`        | | |                         |                                                  | | |                      |                                                                         |


"""

with open(f'{REPORT_DIR}/report.md', 'w', encoding='utf-8') as f:
    f.write(report)
    
# Save configuration as JSON
with open(f'{REPORT_DIR}/config.json', 'w', encoding='utf-8') as f:
    config_dict = config.model_dump()
    #config_dict["compute_device"] = compute_device
    #config_dict["selected_tokenizer"] = tokenizer.name
    f.write(json.dumps(config_dict, indent=2))




