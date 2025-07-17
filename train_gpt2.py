from typing import Tuple, Any
from datetime import datetime
import torch
from torch.nn import functional as F
from Config import GPT2Config
from model_gpt2 import GPT2Model
import tiktoken
from termcolor import colored

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Determine computation device (GPU or CPU)
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Determine computation device (GPU or CPU)
compute_device = "cpu"
if torch.cuda.is_available():
    compute_device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    compute_device = "mps"
    
print("Using device:", compute_device)

device = torch.device(compute_device)

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
tokenizer = tiktoken.get_encoding("gpt2")

# Load text file
try:
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

# Batch generator
def get_batch(text)-> Tuple[torch.Tensor, torch.Tensor]:
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
    text = text[:1000] # first 1000 characters for testing purposes
    tokens = tokenizer.encode(text)  # Encode the text into tokens
    B, T = 4, 32
    buf = torch.tensor(tokens[:B*T + 1], dtype=torch.long)  # Convert tokens to tensor
    xb = buf[:-1].view(B,T)
    yb = buf[1:].view(B,T)  # Shifted version for targets
    
    # Move to compute device (GPU or CPU)
    xb, yb = xb.to(device), yb.to(device) 
    
    return xb, yb

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

PRETRAINED = False  # Set to False if you want to use randomly initialized weights

print(f"\n{'='*60}")
print("MODEL INITIALIZATION")
print('='*60)

# Create model instance
if PRETRAINED: 
    model = GPT2Model.from_pretrained('gpt2') # Load the pre-trained GPT-2 model from Huggingface
else: 
    model = GPT2Model(config) # If you want to use the model with randomly initialized weights (before any training)

model.eval()
model.to(device)

# =============================================================================
# TRAINING 
# =============================================================================
print(f"\n{'='*60}")
print("TRAINING")
print('='*60)

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    
    # TRAINING PHASE
    
    # Get a batch of training data
    xb, yb = get_batch(text)
    
    _, loss = model(xb, yb) # Forward pass
    optimizer.zero_grad() # Reset gradients
    loss.backward() # Backward pass
    optimizer.step() # Update weights
    print(f"Step {i+1}, Loss: {loss.item():.4f}")
# =============================================================================
# INFERENCE & TEXT GENERATION
# =============================================================================
# Context tokens
context_text = "Hello, I'm a language model,"
context_tokens = tokenizer.encode(context_text)
context_tokens = torch.tensor(context_tokens, dtype=torch.long) # 1, 8

# Manually generating a batch with the same sequence context_tokens 5 times 
num_generated_sequences = 5
context_tokens = context_tokens.unsqueeze(0).repeat(num_generated_sequences, 1) # 5, 8
idx = context_tokens.to(device)

max_new_tokens = 30

# Generate from context tokens (manually instead of using model.generate() not implemented yet)
while idx.size(1) < max_new_tokens:

    with torch.no_grad():
        # right now idx is (B,T) where B = 5, T = 8

        # forward the model
        logits, loss = model(idx)  # (B, T, vocab_size)

        # Focus on the last time step (next token prediction)
        logits = logits[:, -1, :] # (B, vocab_size) 

        # Convert to probabilities
        probs = F.softmax(logits, dim = -1) # (B, vocab_size)

        # Do top-k sampling of 50 (huggingface default pipeline)
        # topk_probs here becomes (5,50) and topk_indices becomes (5,50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # Get top 50 probabilities and indices
        ix = torch.multinomial(topk_probs, num_samples=1)  # Select a token from the top 50 probabilities
        xcol = torch.gather(topk_indices, -1, ix) # Gather the corresponding token indices based on the sampled probabilities
        
        # Append to sequence
        idx = torch.cat((idx, xcol), dim = 1) # (B, T+1)

# print the generated sequences
for i in range(num_generated_sequences):
    generated_tokens = idx[i, :max_new_tokens].tolist() # Get the generated tokens for this sequence
    generated_text = tokenizer.decode(generated_tokens)  # Decode the tokens to text
    print(f"Generated text {i+1}: <START>", colored(generated_text, "cyan"), "<END>")

