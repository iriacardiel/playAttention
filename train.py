from termcolor import colored
from tokenization.tokenizers import tiktoken_tokenizer, char_level_tokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F

# For reproducibility
torch.manual_seed(1337) 

# Use computation available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# HYPERPARAMETERS
# ---------------
seq_size = 8 
batch_size = 32
training_steps = 10000
learning_rate = 1e-3
eval_iters = 200
eval_interval = 300
# ---------------

# PREPARE TRAINING AND VALIDATION DATA
# ------------------------------------
# Load train text file
with open('data/tinyshakespeare.txt', 'r') as f:
    text = f.read()
    
text_ids = char_level_tokenizer.encode(text)
data = torch.tensor(text_ids, dtype=torch.long)

# Generate splits: train , val
ratio = 0.9 # 90% for training, 10% for validation
idx = int(ratio * len(data)) 
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
class BigramLanguageModel(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        # Each token reads off the logits for the next token from a lookup table
        # Randomly initialized embedding layer of size (vocab_size, vocab_size)
        
        self.emmbeding_layer = nn.Embedding(vocab_size, d_model) # C = vocab_size
    
    def forward(self, idx, targets=None):
        
        # idx (xb) input tokens shape: (B,T), target tokens (yb) shape: (B,T)
        # B: batch size, T: sequence length
        # Access the embedding table to get the logits for the next token is equivalent to multiplying the one-hot encoded vector of the input token with the embedding matrix
        
        embeddings = self.emmbeding_layer(idx) # Score fo the next token on the sequence (B,T,C) 
        
        logits = embeddings # in this case, the logits are the outputs of the unique layer
        
        # For inference, no need to calculate loss
        if targets is None:
            loss = None
        # For training, calculate loss
        else:
            # Reshaping for the loss
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """_summary_
        Generates the next token in the sequence in all the batch dimensions in the time dimension :
        (BxT) --> BxT+1, BxT+2, BxT+3, ...., BxT+max_new_tokens
        """
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(idx) # (B, T, C) # para cada batch B, los logits de cada time step 1,..., T para los C elementos del vocabulario
            
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
m = BigramLanguageModel(d_model=char_level_tokenizer.n_vocab, vocab_size=char_level_tokenizer.n_vocab)
model = m.to(device)

# OPTIMIZER
# ---------
# PyTorch optimizer: takes the gradients and updates paramenters
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate) 


# TRAINING LOOP
# -------------
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
for step in range(training_steps):
    
    # every eval_interval steps pause training and evaluate the mean loss on train and val sets on eval_iters batches
    if step % eval_interval == 0:
        mean_losses = estimate_loss()
        print(f"Step {step}: train loss {mean_losses['train']:.4f}, val loss {mean_losses['val']:.4f}")

    # sample a batch of data (this is a random batch from the train data)
    xb, yb = get_batch(split_type='train', batch_size=batch_size, seq_size=seq_size) # Get batch of inputs / targets
    
    # evaluate the loss (no need for the logits now)
    _, loss = model(xb, yb) # Forward pass
    optimizer.zero_grad(set_to_none=True) # Clear gradients
    loss.backward() # Backward pass: PyTorch automatically computes gradients of the loss with respect to all model parameters, using autograd
    optimizer.step() # Update model weights

print(f"Final loss --> {loss.item()}") 

# INFERENCE
# ---------
context = torch.zeros((1,1), dtype = torch.long, device=device)
print("Generated text: <START>", colored(char_level_tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()), "cyan"), "<END>")