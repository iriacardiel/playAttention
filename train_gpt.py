"""
Training Script
=====================================
"""

import os
import csv
from typing import Tuple, Any
from datetime import datetime
import numpy as np
from tqdm import trange
from termcolor import colored
import matplotlib.pyplot as plt
import torch

from custom_tokenizers import tiktoken_tokenizer, char_level_tokenizer
from Config import GPTConfig
from model_gpt import GPTModel
import json


TRAIN = True # Set to False to skip training and just load the model
# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# For reproducibility
torch.manual_seed(1337) 

# Determine computation device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Training compute device --> {device}\n")

# File paths
TRAIN_ID = datetime.now().strftime("%Y%m%d_%H%M") # Unique identifier for this training session
DATA_PATH = 'data/tinyshakespeare.txt'
REPORT_DIR = f'reports/training_{TRAIN_ID}_{device.type}'
# Create plots directory
os.makedirs(REPORT_DIR, exist_ok=True)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# config = GPTConfig(
#             compute_device=device,
#             tokenizer=char_level_tokenizer,
#             vocab_size=None,  # Will be set after data preparation
#             seq_size=256,
#             batch_size=64,
#             n_embd=384,
#             n_head=6,
#             n_layer=6,
#             dropout=0.2,
#             training_steps=5000,
#             learning_rate=3e-4,
#             eval_iters=100,
#             eval_interval=100, 
#             train_val_ratio=0.9
#             )

config = GPTConfig(
            compute_device=device,
            tokenizer=char_level_tokenizer,
            vocab_size=None,  # Will be set after data preparation
            seq_size=8,
            batch_size=32,
            n_embd=32,
            n_head=4,
            n_layer=3,
            dropout=0,
            training_steps=20000,
            learning_rate=1e-3,
            eval_iters=100,
            eval_interval=100,
            train_val_ratio=0.9
            )

print(f"\n{'='*60}")
print("HYPERPARAMETERS")
print('='*60)

hyperparams_summary = (f""
    f"\nModel Architecture:\n"
    f"  seq_size        : {config.seq_size} tokens (max context length)\n"
    f"  batch_size      : {config.batch_size} sequences\n"
    f"  n_embd          : {config.n_embd} (embedding dimension)\n"
    f"  n_head       : {config.n_head} heads\n"
    f"  n_layer        : {config.n_layer} transformer blocks\n"
    f"  dropout         : {config.dropout}\n"
    f"\n\nTraining Parameters:\n"
    f"  training_steps  : {config.training_steps:,} steps\n"
    f"  learning_rate   : {config.learning_rate}\n"
    f"  eval_iters      : {config.eval_iters} batches\n"
    f"  eval_interval   : {config.eval_interval} steps\n"
    f"  train_val_ratio : {config.train_val_ratio}\n"
)

print(hyperparams_summary)


# =============================================================================
# DATA PREPARATION
# =============================================================================
print(f"\n{'='*60}")
print("DATA PREPARATION")
print('='*60)

def load_and_prepare_data() -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Load text data, tokenize it, and split into train/validation sets.
    
    Returns:
        train_data: Training data tensor
        val_data: Validation data tensor
        vocab_size: Size of vocabulary
    """
    
    # Load text file
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    # Tokenize text using character-level tokenizer [TODO: Try tiktoken tokenizer]
    text_ids = config.tokenizer.encode(text)
    data = torch.tensor(text_ids, dtype=torch.long)
    vocab_size = config.tokenizer.n_vocab


    # Generate splits: train , val
    split_idx = int(config.train_val_ratio * len(data)) 
    train_data = data[:split_idx] # train split
    val_data = data[split_idx:] # validation split
    

    
    data_preparation_summary = (
        f"\nTokenziation summary:\n"
        f"  Tokenizer: {config.tokenizer.name}\n"
        f"  Tokenized text: {len(data):,} tokens\n"
        f"  Vocabulary size: {vocab_size} unique tokens\n"
        f"\nData split:\n"
        f"  Training:   {len(train_data):,} tokens ({len(train_data)/len(data)*100:.1f}%)\n"
        f"  Validation: {len(val_data):,} tokens ({len(val_data)/len(data)*100:.1f}%)\n"
    )
    print(data_preparation_summary)

    return train_data, val_data, vocab_size

# Load and prepare data
train_data, val_data, vocab_size = load_and_prepare_data()
config.vocab_size = vocab_size # Update config with vocabulary size

# Batch generator
def get_batch(split_type: str, batch_size: int, seq_size: int)-> Tuple[torch.Tensor, torch.Tensor]:
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
    data = train_data if split_type == "train" else val_data
    
    # Sample random starting indices for each sequence in the batch
    starting_idx = torch.randint(
        len(data) - seq_size,
        (batch_size,)
    ) 
    
    # Extract sequences and create targets
    xb = torch.stack([
        data[i:i+seq_size] 
        for i in starting_idx
    ]) 
    # Shift the sequence by one to the right to create the target
    yb = torch.stack([
        data[i+1:i+seq_size+1] 
        for i in starting_idx
    ]) 
    
    # Move to compute device (GPU or CPU)
    xb, yb = xb.to(device), yb.to(device) 
    
    return xb, yb



# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
print(f"\n{'='*60}")
print("MODEL INITIALIZATION")
print('='*60)

# Create model instance
model = GPTModel(config).to(device)

# Check dtype of all parameters. Default is float32, but can be changed to float16 for memory efficiency
# print("\nModel parameters data types:")
# for name, param in model.named_parameters():
#     print(f"{name}: {param.dtype}")

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate) 

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
for k, v in model.state_dict().items():
    print(colored(f"{k}: {v.shape} - {v.dtype}", "green"))  # Print each parameter's shape and dtype

# =============================================================================
# TRAINING 
# =============================================================================
           
if TRAIN:
    print("Training the model...")

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
                X,Y = get_batch(split, batch_size=config.batch_size, seq_size=config.seq_size)
                _, loss = model(X,Y)
                losses[k] = loss.item()
            mean_losses[split] = losses.mean()
        model.train() # indicate the model is in 'training' mode
        return mean_losses


    # =============================================================================
    # TRAINING LOOP
    # =============================================================================
    print(f"\n{'='*60}")
    print("TRAINING")
    print('='*60)

    # Initialize lists to store losses for plotting and logging
    train_losses, val_losses, steps_recorded, final_losses = [], [], [], {}
    print(f"\nStarting training loop...")

    start_train = datetime.now() # Record start time of training
    # In each time step a different batch is sampled randomly with 32 sequences of 8 tokens
    for step in trange(config.training_steps, desc="Training steps", unit="step", disable=False):
        # EVALUATION PHASE
        if step % config.eval_interval == 0: # Every eval_interval steps pause training and evaluate the mean loss on train and val sets on eval_iters batches
            
            losses = estimate_loss()
            starting_limits_ffn, starting_limits_pwe = update_visualizations(step, train_losses, val_losses, losses, steps_recorded, model, starting_limits_ffn, starting_limits_pwe)

        # TRAINING PHASE

        # Get a batch of training data
        xb, yb = get_batch(split_type='train', batch_size=config.batch_size, seq_size=config.seq_size) # Sample a batch of data (this is a random batch from the train data)
        
        # Forward pass: compute model output and loss
        _, loss = model(xb, yb) # Forward pass
        
        # Backward pass: compute gradients
        optimizer.zero_grad(set_to_none=True) # Clear previous gradients
        loss.backward() # Compute gradients via backpropagation
        
        # Update model parameters
        optimizer.step() # Update model weights

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
    enc = config.tokenizer
    context_tokens = enc.encode(context_text)
    context_tokens = torch.tensor(context_tokens, dtype=torch.long)
    idx = context_tokens.to(device).unsqueeze(0)  # Shape becomes (1, seq_len)

    # Generate from context tokens
    generated_tokens = model.generate(idx, max_new_tokens=500)[0].tolist()
    generated_text = enc.decode(generated_tokens)
    print("Generated text: <START>", colored(generated_text, "cyan"), "<END>")
     

    # =============================================================================
    # REPORT GENERATION
    # =============================================================================

    report = f"""# Training Report

**Training Session:** `{TRAIN_ID}`

**Training Device:** `{device}`

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

| Hyperparameters and Architecture |                            | | | Model Dimension         |                                                  | | | Dataset Details      |                                                            |
|----------------------------------|----------------------------|-|-|-------------------------|--------------------------------------------------|-|-|----------------------|------------------------------------------------------------|
| seq_size                       | `{config.seq_size}` tokens   | | | Total Parameters        | `{total_params:,}`                               | | | Dataset              | `{DATA_PATH}`                                              |
| batch_size                     | `{config.batch_size}`        | | | Trainable Parameters    | `{trainable_params:,}`                           | | | Vocabulary Size      | `{vocab_size:,}` tokens                                    |
| n_embd (dim)                   | `{config.n_embd}`            | | | Model Size              | ~`{total_params * 4 / 1024**2:.2f}` MB (float32)  | | | Dataset Size         | `{len(train_data) + len(val_data):,}` tokens               |
| n_head                      | `{config.n_head}`         | | | Optimizer               | AdamW with learning rate `{config.learning_rate}`| | | Training Tokens      | `{len(train_data):,}` tokens ({config.train_val_ratio:.1%})|
| n_layer                       | `{config.n_layer}`          | | | Tokenizer               | `{config.tokenizer.name}`                        | | | Validation Tokens    | `{len(val_data):,}` tokens ({1-config.train_val_ratio:.1%})|
| dropout                        | `{config.dropout}`           | | |                         |                                                  | | |                      |                                                            |
| training_steps                 | `{config.training_steps:,}`  | | |                         |                                                  | | |                      |                                                            |
| learning_rate                  | `{config.learning_rate}`     | | |                         |                                                  | | |                      |                                                            |
| eval_interval                  | `{config.eval_interval}`     | | |                         |                                                  | | |                      |                                                            |
| eval_iters                     | `{config.eval_iters}`        | | |                         |                                                  | | |                      |                                                            |


    """

    with open(f'{REPORT_DIR}/report.md', 'w', encoding='utf-8') as f:
        f.write(report)
        
    # Save configuration as JSON
    with open(f'{REPORT_DIR}/config.json', 'w', encoding='utf-8') as f:
        config_dict = config.model_dump()
        config_dict["compute_device"] = str(config_dict["compute_device"])
        config_dict["tokenizer"] = config_dict["tokenizer"].name  # Store tokenizer name
        f.write(json.dumps(config_dict, indent = 2))
        
    
else:
    print("Skipping training. Just loading the model...")
    # maybe load model weights, run evaluation, etc.




