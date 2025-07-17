"""
GPT-2
=====================================

This script implements a small language model based on the decoder-only transformer.
It includes detailed comments explaining each component and process.

Architecture:
- Character-level tokenization
- Causal Multi-head Self-attention
- Feedforward networks
- Positional embeddings
- Residual connections and layer normalization

Note:
 - We are building the GPT-2 model architecture (skelleton) so we can load the weights like in: https://www.youtube.com/watch?v=l8pRSuU81PU&t=123s
 - It is important to build it with the same names (different from the model_gpt.py script for the first GPT) and structure as the original GPT-2 model to be able to load the weights correctly.

"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel # Huggingface model that we will use to load the weights
from Config import GPT2Config, ModelConfig


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Same algorithm as the original GPT, but more efficient. Number of heads works as a new batch dimension.
    Optimized for training on GPUs, where we can use the efficient matrix multiplication operations.

    Mathematically equivalent to implementing each Head separately and concatenating.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # 3 because we have query, key, and value. This replaces the three linear layers for each Q,K,V


        # Project concatenated heads back to embedding dimension
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
    
    
    
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Purpose:
    ========
    
    After attention has gathered information from different positions,
    the feed-forward network processes this information for each position
    independently. It's a simple MLP that:
    
    1. Expands the representation to a larger dimension (4x is standard)
    2. Applies non-linear activation (GeLU)
    3. Projects back to the original dimension
    
    This allows the model to perform complex computations on the attention
    output and introduces non-linearity that's crucial for learning
    complex patterns.
    
    Architecture: Linear -> GeLU -> Linear -> Dropout
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        hidden_size = 4 * config.n_embd                      # Standard transformer uses 4x expansion
        self.c_fc = nn.Linear(config.n_embd, hidden_size)    # Linear layer to expand to larger dimension
        self.gelu = nn.GELU(approximate='tanh')              # Activation function (GeLU) replaced previous ReLU
        self.c_proj = nn.Linear(hidden_size, config.n_embd)  # Linear layer to project back to embedding dimension
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        
        return x

class TransformerBlock(nn.Module):
    """
    Complete transformer block: Multi-head attention + Feed-forward.
    
    Architecture (Pre-LayerNorm style):
    ===================================
    
    1. LayerNorm -> Multi-Head Attention -> Residual Connection
    2. LayerNorm -> Feed-Forward Network (MLP) -> Residual Connection
    
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
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embd)              # Pre-Norm
        self.attn = MultiHeadAttention(config)               # Multi-Head Causal Self-Attention
        self.ln_2 = nn.LayerNorm(config.n_embd)              # Pre-Norm
        self.mlp = FeedForward(config)                       # Feedforward (MLP)

    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        
        # Multi-Head Causal Self-Attention with Pre-Norm and Skip Connection
        x = x + self.attn(self.ln_1(x))
        
        # Feedforward (MLP) with Pre-Norm and Skip Connection
        x = x + self.mlp(self.ln_2(x))          
           
        return x
    
class GPT2Model(nn.Module):
    """
    GPT-2 Language Model.
    
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
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Token ID -> embedding vector
            wpe = nn.Embedding(config.seq_size, config.n_embd), # Position -> embedding vector
            h = nn.ModuleList([
                TransformerBlock(config) 
                for _ in range(config.n_layer)
                ]),
            ln_f = nn.LayerNorm(config.n_embd)
        )) # Stack of n_layer transformer blocks
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)                   # Project to vocabulary 
    def forward(self, idx: torch.Tensor):
        """
        Forward pass through the model.
        
        Args:
            idx: Input token indices, shape (batch_size, sequence_length)
        Returns:
            logits: Predicted token probabilities, shape (batch_size, sequence_length, vocab_size)
        """
        
        B,T = idx.shape # B: batch size, T: sequence length
        assert T <= self.config.seq_size, f"Cannot forward sequence of length {T}, block size is only {self.config.seq_size}"
        
        # Create embeddings for the input tokens and add positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # position indices
        pos_emb = self.transformer.wpe(pos) # position embeddings (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings (B,T, n_embd)

        x = tok_emb + pos_emb # (B, T, n_embd)
        
        # Pass through the transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        # Final layer normalization
        x = self.transformer.ln_f(x) # (B,T,C)
        
        # Linear layer to project the embeddings to the vocabulary size
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
        # NOT IMPLEMENTED YET
        raise NotImplementedError("Generation method not implemented yet.")