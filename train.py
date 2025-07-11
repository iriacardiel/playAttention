"""
GPT-style Transformer Training Script
=====================================

This script implements a small GPT-style transformer from scratch for educational purposes.
It includes detailed comments explaining each component and process.

Architecture:
- Character-level tokenization
- Multi-head self-attention
- Feed-forward networks
- Positional embeddings
- Residual connections and layer normalization
"""
import os
import csv
import pathlib as Path
from typing import Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange
import matplotlib.pyplot as plt
from termcolor import colored

from tokenization.tokenizers import tiktoken_tokenizer, char_level_tokenizer

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# For reproducibility
torch.manual_seed(1337) 

# Determine computation device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Training compute device --> {device}\n")

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# Model Architecture Parameters
seq_size = 8 # Number of tokens in the input sequence. Maximum context length for the predictions
batch_size = 32 # Number of sequences in a batch to be processed in parallel
n_embd = 32 # Embedding dimension: size of the embedding vector for each token
num_heads = 4
N_layers = 3 # Number of transformer blocks in the model
dropout = 0 # Dropout rate for regularization (to avoid overfitting)

# Training Parameters
training_steps = 20000 # Number of training steps
learning_rate = 1e-3 
eval_iters = 200 # NUmber of batches to evaluate the loss on train and val splits
eval_interval = 500 # Number of training steps between evaluations
train_val_ratio = 0.9 # 90% for training, 10% for validation

# File paths
DATA_PATH = 'data/tinyshakespeare.txt'
PLOTS_DIR = 'plots'
CSV_FILE = 'plots/training_losses.csv'
PLOT_FILE = 'plots/loss_curves_live.png'

print(f"\n{'='*60}")
print("HYPERPARAMETERS")
print('='*60)
print(f"Model Architecture:")
print(f"  seq_size        : {seq_size} tokens (max context length)")
print(f"  batch_size      : {batch_size} sequences")
print(f"  n_embd          : {n_embd} (embedding dimension)")
print(f"  num_heads       : {num_heads} heads")
print(f"  N_layers        : {N_layers} transformer blocks")
print(f"  dropout         : {dropout}")
print(f"\nTraining Parameters:")
print(f"  training_steps  : {training_steps:,} steps")
print(f"  learning_rate   : {learning_rate}")
print(f"  eval_iters      : {eval_iters} batches")
print(f"  eval_interval   : {eval_interval} steps")
print(f"  train_val_ratio : {train_val_ratio}")


# =============================================================================
# DATA PREPARATION
# =============================================================================
print(f"\n{'='*60}")
print("DATA PREPARATION")
print('='*60)

def load_and_prepare_data():
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
        print(f"✓ Loaded text file: {len(text):,} characters")
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    # Tokenize text using character-level tokenizer [TODO: Try tiktoken tokenizer]
    text_ids = char_level_tokenizer.encode(text)
    data = torch.tensor(text_ids, dtype=torch.long)
    vocab_size = char_level_tokenizer.n_vocab
    print(f"✓ Tokenized text: {len(data):,} tokens")
    print(f"✓ Vocabulary size: {vocab_size} unique tokens")

    # Generate splits: train , val
    split_idx = int(train_val_ratio * len(data)) 
    train_data = data[:split_idx] # train split
    val_data = data[split_idx:] # validation split
    
    print(f"✓ Data split:")
    print(f"  Training:   {len(train_data):,} tokens ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Validation: {len(val_data):,} tokens ({len(val_data)/len(data)*100:.1f}%)")
    
    return train_data, val_data, vocab_size

# Load and prepare data
train_data, val_data, vocab_size = load_and_prepare_data()


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
# MODEL ARCHITECTURE
# =============================================================================

class AttentionHead(nn.Module):
    """
    Single head of self-attention.
    
    Self-attention allows each token to "look at" all other tokens in the sequence
    and decide how much to focus on each one when creating its representation.
    
    Key concepts:
    - Query (Q): "What am I looking for?"
    - Key (K): "What do I contain?"
    - Value (V): "What should I contribute if I'm relevant?"
    
    The attention mechanism:
    1. Compute attention scores: Q @ K^T (how much each token "likes" every other token)
    2. Apply causal masking (prevent looking at future tokens)
    3. Normalize with softmax to get attention weights
    4. Apply weights to values: attention_weights @ V
    
    """
    def __init__(self, head_size: int):
        super().__init__()
        
        # Linear transformations for Q, K, V (no bias needed)
        self.key = nn.Linear(n_embd, head_size, bias=False) 
        self.query = nn.Linear(n_embd, head_size, bias=False) 
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Causal mask: lower triangular matrix prevents looking at future tokens
        self.register_buffer('causal_mask', torch.tril(torch.ones(seq_size, seq_size)))
        
        # Dropout layer for regularization 
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B,T,C = x.shape # B: batch size, T: sequence length, C: head size
        
        # Compute Q, K, V projections
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)

        # Compute attention scores ("affinities") between query and key vectors
        scores = q @ k.transpose(-2,-1) # (B,T,C) @ (B, C, T) --> (B,T,T) # dot product between query and key vectors
        
        # Scale by sqrt(head_size) to prevent large values that cause vanishing gradients
        scores = scores * C ** -0.5 # Scale the scores by the square root of the embedding dimension to prevent large values that can lead to numerical instability
        
        # Apply causal masking (set future positions to -inf). only for decoder self-attention, which needs to be autoregressive
        scores = scores.masked_fill(
            self.causal_mask[:T,:T] == 0, 
            float('-inf')
        )
        
        # Attention scores are normalized to probabilities
        attention_weights = torch.functional.F.softmax(scores, dim=-1)
        
        # Dropout for regularization
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values: weighted sum of the value vectors
        out = attention_weights @ v  # (B,T,T) @ (B,T,C) --> (B,T,C)
        
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.
    
    Why Multiple Heads?
    ==================
    
    Multiple attention heads allow the model to attend to different aspects
    of the input simultaneously. Each head can focus on different relationships:
    
    - Head 1: Might focus on syntactic relationships (subject-verb)
    - Head 2: Might focus on semantic relationships (word meanings)
    - Head 3: Might focus on positional relationships (nearby words)
    - Head 4: Might focus on long-range dependencies
    
    The outputs of all heads are concatenated and projected back to the
    original embedding dimension, allowing the model to use all these
    different types of attention simultaneously.
    """
    
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        # Multiple attention heads operating in parallel
        self.heads = nn.ModuleList([
            AttentionHead(head_size)
            for _ in range(num_heads)
        ])
        # Project concatenated heads back to embedding dimension
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run all heads in parallel and concatenate outputs
        head_outputs = [h(x) for h in self.heads] # List of outputs from each head, each of shape (B,T,C)
        out = torch.cat(head_outputs, dim=-1) # (B, T, n_embd)
        
        # Project back to embedding dimension
        out = self.proj(out) 
        
        # Apply dropout for regularization
        out = self.dropout(out) 
        
        return out
    
    
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Purpose:
    ========
    
    After attention has gathered information from different positions,
    the feed-forward network processes this information for each position
    independently. It's a simple MLP that:
    
    1. Expands the representation to a larger dimension (4x is standard)
    2. Applies non-linear activation (ReLU)
    3. Projects back to the original dimension
    
    This allows the model to perform complex computations on the attention
    output and introduces non-linearity that's crucial for learning
    complex patterns.
    
    Architecture: Linear -> ReLU -> Linear -> Dropout
    """
    def __init__(self, n_embd: int):
        super().__init__()
        # Standard transformer uses 4x expansion
        hidden_size = 4 * n_embd 
        
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_size),  # Linear layer to expand to larger dimension
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, n_embd),  # Linear layer to project back to embedding dimension
            nn.Dropout(dropout) # Dropout for regularization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x

class TransformerBlock(nn.Module):
    """
    Complete transformer block: Multi-head attention + Feed-forward.
    
    Architecture (Pre-LayerNorm style):
    ===================================
    
    1. LayerNorm -> Multi-Head Attention -> Residual Connection
    2. LayerNorm -> Feed-Forward Network -> Residual Connection
    
    Key Components:
    - Pre-LayerNorm: Normalizes inputs before each sub-layer (more stable training)
    - Residual Connections: Allow gradients to flow directly, enabling deep networks
    - Multi-Head Attention: Allows tokens to communicate with each other
    - Feed-Forward: Processes the attended information
    
    Why Pre-LayerNorm?
    - Original transformers used Post-LayerNorm (after residual connection)
    - Pre-LayerNorm has been shown to be more stable and easier to train
    - It helps with gradient flow in deep networks
    """
    def __init__(self, n_embd: int, num_heads: int):
        super().__init__()
        head_size = n_embd // num_heads # head_size is a divisor of n_embd, the embedding dimension
        
        # Sub-layers of the transformer block
        self.ln1 = nn.LayerNorm(n_embd)                      # Pre-norm for attention
        self.mha = MultiHeadAttention(num_heads, head_size)  # Multi-head attention
        self.ln2 = nn.LayerNorm(n_embd)                      # Pre-norm for feed-forward
        self.ffn = FeedForward(n_embd)                       # Feed-forward network

    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x = self.ln1(x)
        x = x + self.mha(x) 
        
        # Feed-forward with residual connection
        x = self.ln2(x)
        x = x + self.ffn(x) 
        return x
    

class GPTLanguageModel(nn.Module):
    """
    GPT-style Language Model.
    
    Architecture Overview:
    =====================
    
    1. Token Embedding: Converts token IDs to dense vectors
    2. Positional Embedding: Adds position information to each token
    3. Transformer Blocks: Stack of attention + feed-forward layers
    4. Layer Normalization: Final normalization before output
    5. Language Modeling Head: Projects to vocabulary size for next-token prediction
    
    Key Concepts:
    - Autoregressive: Predicts next token based on previous tokens
    - Causal: Cannot look at future tokens during training
    - Transformer: Uses attention mechanism for token interactions
    """
    def __init__(self):
        super().__init__()
        # Embedding layers
        self.emmbeding_layer = nn.Embedding(vocab_size, n_embd)         # Token ID -> embedding vector
        self.position_embbedings_layer = nn.Embedding(seq_size, n_embd) # Position -> embedding vector
       
        # Stack of N_layers transformer blocks
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(n_embd, num_heads) 
            for _ in range(N_layers)
        ])
        
        self.ln = nn.LayerNorm(n_embd)                                  # Final normalization 
        
        self.llm_head = nn.Linear(n_embd, vocab_size)                   # Project to vocabulary 
        
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            idx: Input token indices, shape (batch_size, sequence_length)
            targets: Target token indices for loss calculation, shape (batch_size, sequence_length)
        
        Returns:
            logits: Predicted token probabilities, shape (batch_size, sequence_length, vocab_size)
            loss: Cross-entropy loss if targets provided, None otherwise
        """
        
        B,T = idx.shape # B: batch size, T: sequence length
        
        # Create embeddings for the input tokens and add positional embeddings
        token_emb = self.emmbeding_layer(idx) # (B, T, n_embd) 
        pos_emb = self.position_embbedings_layer(torch.arange(T, device=device)) # (T, n_embd)
        x = token_emb + pos_emb # (B, T, n_embd)
        
        # Pass through the transformer blocks
        x = self.transformer_blocks(x) # (B, T, n_embd)
        
        # Final layer normalization
        x = self.ln(x) # (B,T,n_embd) Layer normalization for the final output 
        
        # Linear layer to project the embeddings to the vocabulary size
        logits = self.llm_head(x) # (B,T,vocab_size)
        
        # Calculate loss if targets provided (for training)
        loss = None
        # For inference, no need to calculate loss
        if targets is not None:
            # Reshaping for the loss
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generate new tokens autoregressively in all the batch dimensions
        (BxT) --> BxT+1, BxT+2, BxT+3, ...., BxT+max_new_tokens
        
        Process:
        1. Take current sequence
        2. Get predictions for next token
        3. Sample from the probability distribution
        4. Append to sequence
        5. Repeat
        
        Note: We crop the input to seq_size to respect the model's context window.
        
        Args:
            idx: Starting context, shape (batch_size, current_length)
            max_new_tokens: Number of tokens to generate
        
        Returns:
            Generated sequence, shape (batch_size, current_length + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop input sequence to maximum context length
            idx_cropped = idx[:, -seq_size:] # (B, T) 
            
            # Get predictions
            logits, _ = self(idx_cropped) # (B, T, vocab_size)
            
            # Focus on the last time step (next token prediction)
            logits = logits[:, -1, :] # (B, vocab_size) 

            # Convert to probabilities
            probs = F.softmax(logits, dim = 1) # (B, vocab_size)

            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
            
        return idx

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
print(f"\n{'='*60}")
print("MODEL INITIALIZATION")
print('='*60)
# Create model instance
model = GPTLanguageModel().to(device)

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✓ Model initialized:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: ~{total_params * 4 / 1024**2:.2f} MB (float32)")

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate) 
print(f"✓ Optimizer: AdamW with learning rate {learning_rate}")



# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def set_up_visualization():
    """
    Set up the live visualization for training and validation losses.
    """
    # Create plots directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Initialize Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, training_steps + 2 * eval_interval)
    ax.set_xticks(range(0, training_steps + 2 * eval_interval, 2 * eval_interval)) # Set x-ticks to show every 2 eval_interval steps
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
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters): 
            X,Y = get_batch(split, batch_size=batch_size, seq_size=seq_size)
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
train_losses, val_losses, steps_recorded = [], [], []
# Initialize visualization plot
fig, ax, train_line, val_line = set_up_visualization()

print(f"Training configuration:")
print(f"  Steps: {training_steps:,}")
print(f"  Batch size: {batch_size}")
print(f"  Evaluation every {eval_interval} steps")
print(f"  Logging to: {CSV_FILE}")
print(f"  Plots saved to: {PLOT_FILE}")
print(f"\nStarting training loop...")

# In each time step a different batch is sampled randomly with 32 sequences of 8 tokens
for step in trange(training_steps, desc="Training steps", unit="step", disable=False):
    
    # EVALUATION PHASE
    if step % eval_interval == 0: # Every eval_interval steps pause training and evaluate the mean loss on train and val sets on eval_iters batches

        losses = estimate_loss()
        
        # Update visualization
        update_visualization(
            step, losses['train'], losses['val'],
            train_losses, val_losses, steps_recorded,
            fig, ax, train_line, val_line
        )
    
    # TRAINING PHASE

    # Get a batch of training data
    xb, yb = get_batch(split_type='train', batch_size=batch_size, seq_size=seq_size) # Sample a batch of data (this is a random batch from the train data)
    
    # Forward pass: compute model output and loss
    _, loss = model(xb, yb) # Forward pass
    
    # Backward pass: compute gradients
    optimizer.zero_grad(set_to_none=True) # Clear previous gradients
    loss.backward() # Compute gradients via backpropagation
    
    # Update model parameters
    optimizer.step() # Update model weights

# Final evaluation
final_losses = estimate_loss()
print(f"\nTraining completed!")
print(f"Final train loss: {final_losses['train']:.4f}")
print(f"Final validation loss: {final_losses['val']:.4f}")

# Turn off Plot
plt.close(fig)



# =============================================================================
# INFERENCE & TEXT GENERATION
# =============================================================================
context = torch.zeros((1,1), dtype = torch.long, device=device)
print("Generated text: <START>", colored(char_level_tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()), "cyan"), "<END>")
