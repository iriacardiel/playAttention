from termcolor import colored
from tokenization.tokenizers import tiktoken_tokenizer, char_level_tokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange
import matplotlib.pyplot as plt
import os
import csv

# For reproducibility
torch.manual_seed(1337) 

# Use computation available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Training compute device --> {device}\n")

# HYPERPARAMETERS
# ---------------
seq_size = 8 # Number of tokens in the input sequence. Maximum context length for the predictions
batch_size = 32 # Number of sequences in a batch to be processed in parallel
n_embd = 32 # Embedding dimension: size of the embedding vector for each token
num_heads = 4
N_layers = 3 # Number of transformer blocks in the model
training_steps = 20000 # Number of training steps
learning_rate = 1e-3 
eval_iters = 200 # NUmber of batches to evaluate the loss on train and val splits
eval_interval = 500 # Number of training steps between evaluations
train_val_ratio = 0.9 # 90% for training, 10% for validation
dropout = 0 # Dropout rate for regularization (to avoid overfitting)
# ---------------

print(f"Hyperparameters")
print(f"  seq_size        : {seq_size} tokens")
print(f"  batch_size      : {batch_size} sequences")
print(f"  n_embd          : {n_embd}")
print(f"  num_heads       : {num_heads} heads")
print(f"  N_layers        : {N_layers} transformer layers")
print(f"  training_steps  : {training_steps} steps")
print(f"  learning_rate   : {learning_rate}")
print(f"  eval_iters      : {eval_iters} steps")
print(f"  eval_interval   : {eval_interval} steps")
print(f"  dropout         : {dropout}")


# PREPARE TRAINING AND VALIDATION DATA
# ------------------------------------
# Load train text file
with open('data/tinyshakespeare.txt', 'r') as f:
    text = f.read()
    
text_ids = char_level_tokenizer.encode(text)
data = torch.tensor(text_ids, dtype=torch.long)
vocab_size=char_level_tokenizer.n_vocab
print(f"  vocabulary size : {vocab_size} tokens\n")

# Generate splits: train , val
idx = int(train_val_ratio * len(data)) 
train_data = data[:idx] # train split
val_data = data[idx:] # validation split

# Batch generator
def get_batch(split_type, batch_size, seq_size):
    """
    The transformer receives batches of sequences of the train data
    1 batch contains batch_size sequences of seq_size tokens
    This can be sampled to seq_size+1 examples on how to predict the next token: input(x) and target(y) pairs.

    seq_size =  5 # Size of the input sequence. Maximum context length for the predictions
    batch_size = 4 # Number of sequences in a batch to be processed in parallel. In this case, 4 sequences of 5 tokens each

    1 batch contains 4 sequences of 5 tokens. 
    This can be sampled to 5+1 examples on how to predict the next token: input x and target y pairs.
    
        sequence = [[23,30,31,7,21]]]
        
            input x = [[23]], target y = [[30]]
            input x = [[23,30]], target y = [[31]]
            input x = [[23,30,31]], target y = [[7]]
            input x = [[23,30,31, 7]], target y = [[21]]
            input x = [[23,30,31, 7, 21]], target y = [[14]]


    In a batch:
        x batch = [[23,30,31,7,21,14,0,1], [...], [...], [...]]
        y batch = [[30,31,7,21,14,0,1,2], [...], [...], [...]]
    """
    data = train_data if split_type == "train" else val_data
    starting_idx = torch.randint(len(data) - seq_size, (batch_size,)) # Randomly select STARTING indices for each sequence in the batch
    xb = torch.stack([data[i:i+seq_size] for i in starting_idx]) # Extract sequences of seq_size tokens starting from the indices
    yb = torch.stack([data[i+1:i+seq_size+1] for i in starting_idx]) # Shift the sequence by one to the right to create the target
    xb, yb = xb.to(device), yb.to(device) # Move to device (GPU or CPU)
    return xb, yb

# MODEL
# ------

class Head(nn.Module):
    """
    One head of self-attention.
    
    Every single token at each position t will emmit a query vector and a key vector.
    The query vector: "What am i looking for?"
    The key vector: "What do i have to offer?"
    The value vector: "What is my value?" The value to be aggregated if it is considered relevant by the query.
    Doing dot product (wei) between my query vector and the key vectors of all other tokens will give me a score of how much i like each of them.
    The higher the score, the more i like that token.
    The softmax of the scores will give me a probability distribution over all tokens.
    The weighted sum of the value vectors of all tokens will give me a new representation of the token.

    """
    def __init__(self, head_size):
        super().__init__()
        # head_size is a divisor of n_embd, the embedding dimension
        self.key = nn.Linear(n_embd, head_size, bias=False) 
        self.query = nn.Linear(n_embd, head_size, bias=False) 
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # This is not a module, it is a buffer that will not be updated during training.
        self.register_buffer('tril', torch.tril(torch.ones(seq_size, seq_size))) # Lower triangular matrix for causal masking
        self.dropout = nn.Dropout(dropout) # Dropout layer for regularization 

    def forward(self, x):
        
        B,T,C = x.shape # B: batch size, T: sequence length, C: head size)
        
        # let's see a single Head perform self-attention
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)

        # Compute attention scores ("affinities") between query and key vectors
        scores = q @ k.transpose(-2,-1) # (B,T,C) @ (B, C, T) --> (B,T,T) # dot product between query and key vectors
        scores = scores*n_embd ** -0.5 # Scale the scores by the square root of the embedding dimension to prevent large values that can lead to numerical instability
        scores = scores.masked_fill(self.tril[:T,:T] == 0, float('-inf'))  # CAUSAL MASKING (only for decoder self-attention, which needs to be autoregressive)
        
        # Attention scores are normalized to probabilities
        att = torch.functional.F.softmax(scores, dim=-1) # Normalize the triangular matrix so that every row sums to 1
        att = self.dropout(att) # Apply dropout to the scores for regularization

        # Perform weighter aggreation of the value vectors based on the attention scores
        out = att @ v  # (B,T,T) @ (B,T,n_embd) --> (B,T,n_embd) # weighted sum of the value vectors
        
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention.
    
    This module allows the model to jointly attend to information from different representation subspaces at different positions.
    It consists of multiple heads, each performing self-attention independently and then concatenating the results.
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # List of heads
        self.proj = nn.Linear(n_embd, n_embd) # Linear layer to project the concatenated outputs of all heads back to the embedding dimension
        self.dropout = nn.Dropout(dropout) # Dropout layer for regularization (not used in the original GPT, but can be useful for stability)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # Concatenate the outputs of all heads over the channel dimension. Final output shape: (B,T,n_embd)
        out = self.proj(out) # Project the concatenated outputs back to the embedding dimension
        out = self.dropout(out) # Apply dropout for regularization
        return out
    
    
class FeedForward(nn.Module):
    """
    Feed-forward neural network with a hidden layer. 
    A simple linear layer followed by a ReLU activation.
    """
    def __init__(self, n_embd):
        super().__init__()
        d_ffn = 4 * n_embd # Hidden layer size, typically larger than the embedding dimension
        self.linear1 = nn.Linear(n_embd,d_ffn)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, n_embd)
        self.dropout = nn.Dropout(dropout) # Dropout layer for regularization (not used in the original GPT, but can be useful for stability)

    def forward(self, x):
        x = self.linear1(x) # rotation
        x = self.activation(x) # squash
        x = self.linear2(x) # rotation
        x = self.dropout(x) # apply dropout
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward neural network.
    
    This block consists of a multi-head self-attention layer followed by a feed-forward neural network.
    It also includes residual connections and layer normalization.
    """
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads # head_size is a divisor of n_embd, the embedding dimension
        self.mha = MultiHeadAttention(num_heads, head_size) # Multi-head self-attention
        self.ffn = FeedForward(n_embd) # Feed-forward neural network
        self.layernorm1 = nn.LayerNorm(n_embd) # Layer normalization (we will do PRE-NORM before the self-attention)
        self.layernorm2 = nn.LayerNorm(n_embd) # Layer normalization (we will do PRE-NORM before the feed-forward network)

    def forward(self, x):
        x = self.layernorm1(x)
        x = x + self.mha(x) # Residual connection + LayerNorm after self-attention
        x = self.layernorm2(x)
        x = x + self.ffn(x) # Residual connection + LayerNorm after feed-forward network
        return x
    

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token reads off the logits for the next token from a lookup table
        # Randomly initialized embedding layer of size (vocab_size, vocab_size)
        
        self.emmbeding_layer = nn.Embedding(vocab_size, n_embd) # Vocab_size = vocabulary size, embedding dimension = n_embd. Embedding layer to convert token indices to embeddings
        self.position_embbedings_layer = nn.Embedding(seq_size, n_embd) # Positional embeddings for each token in the sequence
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(n_embd, num_heads) for _ in range(N_layers)] # N_layers is the number of transformer blocks
        )
        self.layernorm = nn.LayerNorm(n_embd) # Layer normalization for the final output (not used in the original GPT, but can be useful for stability)
        self.llm_head = nn.Linear(n_embd, vocab_size) # Linear layer to project the embeddings to the vocabulary size
        
    def forward(self, idx, targets=None):
        B,T = idx.shape # B: batch size, T: sequence length
        # idx (xb) input tokens shape: (B,T), target tokens (yb) shape: (B,T)
        # B: batch size, T: sequence length
        # Access the embedding table to get the logits for the next token is equivalent to multiplying the one-hot encoded vector of the input token with the embedding matrix
        
        token_embeddings = self.emmbeding_layer(idx) # (B,T,n_embd) 
        pos_embeddings = self.position_embbedings_layer(torch.arange(T, device=device)) # (T,n_embd)
        x = token_embeddings + pos_embeddings # (B,T,n_embd) + (T,n_embd) --> (B,T,n_embd)
        x = self.transformer_blocks(x) # (B,T,n_embd) # Pass through the transformer blocks
        x = self.layernorm(x) # (B,T,n_embd) Layer normalization for the final output 
        logits = self.llm_head(x) # (B,T,vocab_size)
        
        
        # For inference, no need to calculate loss
        if targets is None:
            loss = None
        # For training, calculate loss
        else:
            # Reshaping for the loss
            B,T,vocab_size = logits.shape
            logits = logits.view(B*T,vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """_summary_
        Generates the next token in the sequence in all the batch dimensions in the time dimension :
        (BxT) --> BxT+1, BxT+2, BxT+3, ...., BxT+max_new_tokens
        """

        for _ in range(max_new_tokens):
            idx_clip = idx[:,-seq_size:] # (B, T) clip the input sequence to the last seq_size tokens to avoid exceeding the maximum context length
            
            # get the predictions
            logits, _ = self(idx_clip) # (B, T, C) # para cada batch B, los logits de cada time step 1,..., T para los C elementos del vocabulario
            
            # focus on the last step (the predicted) 
            logits = logits[:,-1,:] # (B, C) 

            # apply softmax to get probabiliites
            probs = F.softmax(logits, dim = 1) # (B,C) para cada batch B, la distribución de probabilidad sobre los C elementos del vocabulario para el siguiente time step

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) para cada batch B, el idx sampleado para el siguiente time step
            
            # append sampled index to the runnin sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1) concatenación del next token con el resto de la secuencia
            
        return idx

# Model instance
m = GPTLanguageModel()
model = m.to(device)

# OPTIMIZER
# ---------
# PyTorch optimizer: takes the gradients and updates paramenters
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate) 


# TRAINING LOOP
# -------------

# Initialize lists to store losses for plotting and logging
train_losses = []
val_losses = []
steps_recorded = []

# Initialize Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('Steps')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, training_steps + 2*eval_interval)
ax.set_xticks(range(0, training_steps + 2*eval_interval, 2*eval_interval)) # Set x-ticks to show every 2 eval_interval steps
ax.tick_params(axis='x', labelsize=6)  # x-axis ticks
train_line, = ax.plot([], [], color = "tab:blue", label='Training Loss', linewidth=2)
val_line, = ax.plot([], [],  color = "tab:orange", label='Validation Loss', linewidth=2)
ax.legend()
os.makedirs('plots', exist_ok=True)

# Initialize CSV
csv_file_path = 'plots/training_losses.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Step', 'Train_Loss', 'Val_Loss'])  # Header row
    
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

# In each time step a different batch is sampled randomly with 32 sequences of 8 tokens
for step in trange(training_steps, desc="Training steps", unit="step", disable=False):
    
    # LOG TRAINING PROGRESS
    if step % eval_interval == 0: # Every eval_interval steps pause training and evaluate the mean loss on train and val sets on eval_iters batches

        mean_losses = estimate_loss()
        #print(f"Step {step}: train loss {mean_losses['train']:.4f}, val loss {mean_losses['val']:.4f}")
        
        # Store losses for plotting and logging
        train_losses.append(mean_losses['train'].item())
        val_losses.append(mean_losses['val'].item())
        steps_recorded.append(step)
    
        # Update CSV
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([step, mean_losses['train'].item(), mean_losses['val'].item()])
        
        # Update Plot
        train_line.set_data(steps_recorded, train_losses)
        val_line.set_data(steps_recorded, val_losses)
        ax.set_ylim(min(min(train_losses), min(val_losses))*0.9, max(max(train_losses), max(val_losses)) * 1.1 if train_losses else 1)
        fig.canvas.draw()
        plt.savefig('plots/loss_curves_live.png', dpi=300, bbox_inches='tight')

    # GET BATCH
    xb, yb = get_batch(split_type='train', batch_size=batch_size, seq_size=seq_size) # Sample a batch of data (this is a random batch from the train data)
    # FORWARD PASS
    _, loss = model(xb, yb) # Forward pass
    # BACKWARD PASS
    optimizer.zero_grad(set_to_none=True) # Clear gradients
    loss.backward() # Backward pass: PyTorch automatically computes gradients of the loss with respect to all model parameters, using autograd
    # UPDATE WEIGHTS
    optimizer.step() # Update model weights

print(f"Final loss : {loss.item()}") 

# Turn off Plot
plt.close(fig)



# INFERENCE
# ---------
context = torch.zeros((1,1), dtype = torch.long, device=device)
print("Generated text: <START>", colored(char_level_tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()), "cyan"), "<END>")
