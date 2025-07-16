"""
GPT Model Configuration
=====================================
"""

from typing import Optional, Any
from pydantic import BaseModel
from dataclasses import dataclass

class GPTConfig(BaseModel):
        
    # Compute Device
    compute_device: Any  # Choose device: 'cpu' or 'cuda'
    
    # Tokenizer
    tokenizer: Any  # Choose tokenizer: tiktoken or char_level_tokenizer
    vocab_size: Optional[int] 
    
    # Model Architecture Parameters
    seq_size : int # Number of tokens in the input sequence. Maximum context length for the predictions
    batch_size : int # Number of sequences in a batch to be processed in parallel
    n_embd : int # Embedding dimension: size of the embedding vector for each token
    num_heads : int 
    N_layers : int # Number of transformer blocks in the model
    dropout : float  # Dropout rate for regularization (to avoid overfitting)

    # Training Parameters
    training_steps : int # Number of training steps
    learning_rate : float # Lower if the model is bigger, higher if the model is smaller. 
    eval_iters : int  # NUmber of batches to evaluate the loss on train and val splits
    eval_interval : int  # Number of training steps between evaluations
    train_val_ratio : float  


@dataclass
class GPT2Config:
    seq_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens in the vocabulary: 50000 BPE Merges + 256 Bytes tokens + 1 EOS token 
    n_layer: int = 12 # number of transformer blocks
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension (size of the hidden states)