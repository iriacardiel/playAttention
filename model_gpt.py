"""
GPT-style Transformer
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

from typing import Tuple, Optional
from Config import GPTConfig

import torch
import torch.nn as nn
from torch.nn import functional as F


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
    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embd // config.num_heads # head_size is a divisor of n_embd, the embedding dimension

        # Linear transformations for Q, K, V (no bias needed)
        self.key = nn.Linear(config.n_embd, head_size, bias=False) 
        self.query = nn.Linear(config.n_embd, head_size, bias=False) 
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        
        # Causal mask: lower triangular matrix prevents looking at future tokens
        self.register_buffer('causal_mask', torch.tril(torch.ones(config.seq_size, config.seq_size)))
        
        # Dropout layer for regularization 
        self.dropout = nn.Dropout(config.dropout) 

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
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Multiple attention heads operating in parallel

        self.heads = nn.ModuleList([
            AttentionHead(config)
            for _ in range(config.num_heads)
        ])
        # Project concatenated heads back to embedding dimension
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.dropout) 

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
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Standard transformer uses 4x expansion
        hidden_size = 4 * config.n_embd 
        
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_size),  # Linear layer to expand to larger dimension
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, config.n_embd),  # Linear layer to project back to embedding dimension
            nn.Dropout(config.dropout) # Dropout for regularization
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
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # Sub-layers of the transformer block
        self.ln1 = nn.LayerNorm(config.n_embd)                      # Pre-norm for attention
        self.mha = MultiHeadAttention(config)  # Multi-head attention
        self.ln2 = nn.LayerNorm(config.n_embd)                      # Pre-norm for feed-forward
        self.ffn = FeedForward(config)                       # Feed-forward network

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
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Embedding layers
        self.emmbeding_layer = nn.Embedding(config.vocab_size, config.n_embd)         # Token ID -> embedding vector
        self.position_embeddings_layer = nn.Embedding(config.seq_size, config.n_embd) # Position -> embedding vector
       
        # Stack of N_layers transformer blocks
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(config) 
            for _ in range(config.N_layers)
        ])
        
        self.ln = nn.LayerNorm(config.n_embd)                                         # Final normalization 
        
        self.llm_head = nn.Linear(config.n_embd, config.vocab_size)                   # Project to vocabulary 
        
        self.config = config
        
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
        pos_emb = self.position_embeddings_layer(torch.arange(T, device=self.config.compute_device)) # (T, n_embd)
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
            idx_cropped = idx[:, -self.config.seq_size:] # (B, T) 
            
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