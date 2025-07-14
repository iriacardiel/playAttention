"""
Training Script
=====================================
"""

import os
import csv
from typing import Tuple, Any
from datetime import datetime
from tqdm import trange
from termcolor import colored
import matplotlib.pyplot as plt
import torch

from tokenizers import tiktoken_tokenizer, char_level_tokenizer
from Config import GPTConfig
from model import GPTLanguageModel


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
CSV_FILE = f'{REPORT_DIR}/losses.csv'
PLOT_FILE = f'{REPORT_DIR}/losses.png'
REPORT_FILE = f'{REPORT_DIR}/report.md'
REPORT_HTML_FILE = f'{REPORT_DIR}/report.html'

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

config = GPTConfig(
            compute_device=device,
            tokenizer=char_level_tokenizer,
            vocab_size=None,  # Will be set after data preparation
            seq_size=256,
            batch_size=64,
            n_embd=384,
            num_heads=6,
            N_layers=6,
            dropout=0.2,
            training_steps=1000,
            learning_rate=3e-4,
            eval_iters=500,
            eval_interval=200,
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
    f"  num_heads       : {config.num_heads} heads\n"
    f"  N_layers        : {config.N_layers} transformer blocks\n"
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
model = GPTLanguageModel(config).to(device)

# Print model architecture
print(colored(model, "green"))

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

if TRAIN:
    print("Training the model...")

    # =============================================================================
    # TRAINING UTILITIES
    # =============================================================================

    def set_up_visualization():
        """
        Set up the live visualization for training and validation losses.
        """
        # Create plots directory
        os.makedirs(REPORT_DIR, exist_ok=True)
        
        # Initialize Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, config.training_steps + 2 * config.eval_interval)
        ax.set_xticks(range(0, config.training_steps + 2 * config.eval_interval, 2 * config.eval_interval)) # Set x-ticks to show every 2 eval_interval steps
        ax.tick_params(axis='x', labelsize=6)  # x-axis ticks
        train_line, = ax.plot([], [], color = "tab:blue", label='Training Loss', linewidth=2)
        val_line, = ax.plot([], [],  color = "tab:orange", label='Validation Loss', linewidth=2)
        ax.legend(loc='upper right')

        # Initialize CSV
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Train_Loss', 'Val_Loss'])  # Header row
            
        return fig, ax, train_line, val_line
    
    def update_visualization(step: int, train_loss: float, val_loss: float, 
                            train_losses: list, val_losses: list, steps_recorded: list,
                            fig: plt.Figure, ax: plt.Axes, train_line: Any, val_line: Any): 
        """Update training visualization and save progress."""
        # Update data lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        steps_recorded.append(step)

        # Update CSV
        with open(CSV_FILE, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([step, losses['train'].item(), losses['val'].item()])
        
        # Update Plot
        train_line.set_data(steps_recorded, train_losses)
        val_line.set_data(steps_recorded, val_losses)
        ax.set_ylim(min(min(train_losses), min(val_losses))*0.9, max(max(train_losses), max(val_losses)) * 1.1 if train_losses else 1)
        
        # Save plot
        fig.canvas.draw()
        plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
        

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
    # Initialize visualization plot
    fig, ax, train_line, val_line = set_up_visualization()

    print(f"\nStarting training loop...")

    start_train = datetime.now() # Record start time of training
    # In each time step a different batch is sampled randomly with 32 sequences of 8 tokens
    for step in trange(config.training_steps, desc="Training steps", unit="step", disable=False):
        # EVALUATION PHASE
        if step % config.eval_interval == 0: # Every eval_interval steps pause training and evaluate the mean loss on train and val sets on eval_iters batches
            
            losses = estimate_loss()
            
            # Update visualization
            update_visualization(
                step, losses['train'], losses['val'],
                train_losses, val_losses, steps_recorded,
                fig, ax, train_line, val_line
            )
        
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
    losses = estimate_loss()    
    # Update visualization
    update_visualization(
        config.training_steps-1, losses['train'], losses['val'],
        train_losses, val_losses, steps_recorded,
        fig, ax, train_line, val_line
    )


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

    # Turn off Plot
    plt.close(fig)



    # =============================================================================
    # INFERENCE & TEXT GENERATION
    # =============================================================================
    context = torch.zeros((1,1), dtype = torch.long, device=device)
    generated_text = config.tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist())
    print("Generated text: <START>", colored(generated_text, "cyan"), "<END>")


    # =============================================================================
    # REPORT GENERATION
    # =============================================================================

    report = f"""# GPT Training Report

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

## Hyperparameters Summary

| Hyperparameter | Value |
|-----------|-------|
| seq_size | `{config.seq_size}` tokens |
| batch_size | `{config.batch_size}` |
| n_embd (dim) | `{config.n_embd}` |
| num_heads | `{config.num_heads}` |
| N_layers | `{config.N_layers}` |
| dropout | `{config.dropout}` |
| training_steps | `{config.training_steps:,}` |
| learning_rate | `{config.learning_rate}` |
| eval_interval | `{config.eval_interval}` steps |
| eval_iters | `{config.eval_iters}` |

## Model Details

| Metric | Value |
|--------|-------|
| **Total Parameters** | `{total_params:,}` |
| **Trainable Parameters** | `{trainable_params:,}` |
| **Model Size** | ~`{total_params * 4 / 1024**2:.2f}` MB (float32) |
| **Optimizer** | AdamW with learning rate `{config.learning_rate}` |
| **Tokenizer** | `{config.tokenizer.name}` |

## Dataset Details

| Metric | Value |
|--------|-------|
| **Dataset** | `{DATA_PATH}` |
| **Vocabulary Size** | `{vocab_size:,}` tokens |
| **Total Dataset Size** | `{len(train_data) + len(val_data):,}` tokens |
| **Training Tokens** | `{len(train_data):,}` tokens ({config.train_val_ratio:.1%})|
| **Validation Tokens** | `{len(val_data):,}` tokens ({1-config.train_val_ratio:.1%})|


    """

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)




    print(f"✅ Training report saved to: {REPORT_FILE}")
    print(f"📊 Training data saved to: {CSV_FILE}")
    print(f"📈 Loss plot saved to: {PLOT_FILE}")

else:
    print("Skipping training. Just loading the model...")
    # maybe load model weights, run evaluation, etc.




