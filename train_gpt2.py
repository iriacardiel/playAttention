import torch
from torch.nn import functional as F
from Config import GPT2Config
from model_gpt2 import GPT2Model
import tiktoken
from termcolor import colored


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

PRETRAINED = True  # Set to False if you want to use randomly initialized weights

print(f"\n{'='*60}")
print("MODEL INITIALIZATION")
print('='*60)

# Create model instance
if PRETRAINED: 
    model = GPT2Model.from_pretrained('gpt2') # Load the pre-trained GPT-2 model from Huggingface
else: 
    model = GPT2Model(config=GPT2Config()) # If you want to use the model with randomly initialized weights (before any training)

model.eval()
model.to('cuda')

# =============================================================================
# TRAINING 
# =============================================================================

# not implemented yet.
# not necessary if loading pre-trained weights

# =============================================================================
# INFERENCE & TEXT GENERATION
# =============================================================================
# Context tokens
context_text = "Hello, I'm a language model,"
enc = tiktoken.get_encoding("gpt2")
context_tokens = enc.encode(context_text)
context_tokens = torch.tensor(context_tokens, dtype=torch.long) # 1, 8

# Manually generating a batch with the same sequence context_tokens 5 times 
num_generated_sequences = 5
context_tokens = context_tokens.unsqueeze(0).repeat(num_generated_sequences, 1) # 5, 8

idx = context_tokens.to('cuda')

max_new_tokens = 30
torch.manual_seed(42)  # For reproducibility
torch.cuda.manual_seed(42)  # For reproducibility on GPU
# Generate from context tokens (manually instead of using model.generate() not implemented yet)
while idx.size(1) < max_new_tokens:

    with torch.no_grad():
        # right now idx is (B,T) where B = 5, T = 8

        # forward the model
        logits = model(idx)  # (B, T, vocab_size)

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
    generated_text = enc.decode(generated_tokens)  # Decode the tokens to text
    print(f"Generated text {i+1}: <START>", colored(generated_text, "cyan"), "<END>")

