from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# We are building the GPT-2 model architecture (skelleton) and loading the weights like in: https://www.youtube.com/watch?v=l8pRSuU81PU&t=123s
# It is important to build it with the same names (different from the model.py script for the first GPT) and structure as the original GPT-2 model to be able to load the weights correctly.

@dataclass
class GPT2Config:
    seq_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens in the vocabulary: 50000 BPE Merges + 256 Bytes tokens + 1 EOS token 
    n_layer: int = 12 # number of transformer blocks
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension (size of the hidden states)

class CausalSelfAttention(nn.Module):
    """
    Same algorithm as the original GPT, but more efficient. Number of heads works as a new batch dimension.
    Optimized for training on GPUs, where we can use the efficient matrix multiplication operations.

    Mathematically equivalent to implementing each Head separately and concatenating.
    """
    def __init__(self, config:  GPT2Config): # KARPATHY IS MISSING THIS LINE, WHY?
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # 3 because we have query, key, and value. This replaces the three linear layers for each Q,K,V
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a bias, more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.seq_size, config.seq_size)).view(1, 1, config.seq_size, config.seq_size))

    def forward(self, x):

        B,T,C = x.size() # batch size, sequence length, embedding dimensionallity (n_embd)
        # calculate query, key, value for all heads in a batch and move head forward to be the batch dimnension
        # nh is "number of heads", hs is "head size", and C (number of channels) is nh *hs = n_embd
        # e.g. in GPT-2, n_embd = 768, n_head = 12, so hs = 64
        qkv = self.c_attn(x) # this is a linear layer that takes the input x and projects it to 3 * n_embd
        q, k, v = qkv.split(self.n_embd, dim = 2) # split the output into three parts: query, key, value
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # reshape k to (B, n_head, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # reshape q to (B, n_head, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # reshape v to (B, n_head, T, hs)
        
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q@k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, n_head, T, T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, n_head, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble  all head outputs side by side 
        # output projection
        y = self.c_proj(y) # (B, T, C)
        
        return y 
    
    
    
class MLP(nn.Module):
    def __init__(self, config:  GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # replaced previous RELU with GELU for better performance
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config:  GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.seq_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def forward(self, idx):
        # idx is of shape (B,T) where B is the batch size and T is the sequence length
        B, T = idx.size()
        assert T <= self.config.seq_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # position indices
        pos_emb = self.transformer.wpe(pos) # position embeddings (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings (B,T, n_embd)

        x = tok_emb + pos_emb # (B,T, n_embd) broadcasting the position embeddings to the batch size
        
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
            
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        return logits
    
    @classmethod
    def from_pretrained(cls, model_type: str):
        """
        Loads the GPT-2 model weights from huggingface
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, "Model type must be one of: gpt2, gpt2-medium, gpt2-large, gpt2-xl"
        print("loading weights from huggingface pretrained gpt: %s" % model_type)
        
        # n_layer, n_head, n_embd are determined by the model_type
        config_args = {
            'gpt2':        {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
            'gpt2-medium': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},
            'gpt2-large':  {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},
            'gpt2-xl':     {'n_layer': 48, 'n_head': 25, 'n_embd': 1600}
        }[model_type]
        
        config_args['vocab_size'] = 50257 # GPT-2 uses a fixed vocabulary size
        config_args['seq_size'] = 1024 # GPT-2 uses a fixed sequence length
        # create a from-scratch initialized minGPT model

        config = GPT2Config(**config_args)
        model = GPT2Model(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, it is not needed
        
        # init huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ...
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # ...
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # Basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla ...
        # this means that we have to transpose these weights to match when we import them
        assert len(sd_keys_hf) == len(sd_keys), "Mismatch in number of keys between HF and our model"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t()) # transpose 
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
        
        
# -------------------------------------------------------------------

num_return_sequences = 5
max_length = 30

model = GPT2Model.from_pretrained('gpt2')
print("didn't crash, model loaded successfully")


model.eval()
model.to('cuda')

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)

# Manually generating a batch instead of using get_batch() function
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # 5, 8
x = tokens.to('cuda')

# Manually generating a batch instead of model.generate() metho
# right now x is (B,T) where B = 5, T = 8
while x.size(1) < max_length:

    with torch.no_grad():
        # forward the model
        logits = model(x)  # (B, T, vocab_size)
                
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
        x = torch.cat((x, xcol), dim = 1) # (B, T+1)

# print the generated sequences
for i in range(num_return_sequences):
    generated_tokens = x[i, :max_length].tolist() # Get the generated tokens for this sequence
    generated_text = enc.decode(generated_tokens)  # Decode the tokens to text
    print(f"Generated sequence {i+1}: {generated_text}")
    
