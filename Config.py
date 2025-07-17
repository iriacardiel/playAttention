"""
Model Configurations
=====================================
"""

from typing import Optional, Any
from pydantic import BaseModel
from dataclasses import dataclass
class ModelConfig(BaseModel):
    pass

class GPTConfig(ModelConfig):

    # Compute Device
    compute_device: str # Choose device: 'cpu' or 'cuda'
    
    # Tokenizer
    selected_tokenizer: str # Choose tokenizer: "CharTokenizer" or "TiktokenGPT2"
    vocab_size: Optional[int] # Number of tokens in the vocabulary: (50000 BPE Merges + 256 Bytes tokens + 1 EOS token) for GPT-2, or 256 for char-level tokenizer, etc.
    
    # Model Architecture Parameters
    seq_size : int # Number of tokens in the input sequence. Maximum context length for the predictions
    batch_size : int # Number of sequences in a batch to be processed in parallel
    n_embd : int # Embedding dimension (size of the hidden states)
    n_head : int # Number of attention heads
    n_layer : int # Number of transformer blocks
    dropout : float  # Dropout rate for regularization (to avoid overfitting)

    # Training Parameters
    training_steps : int # Number of training steps
    learning_rate : float # Lower if the model is bigger, higher if the model is smaller. 
    eval_iters : int  # Number of batches to evaluate the loss on train and val splits
    eval_interval : int  # Number of training steps between evaluations
    train_val_ratio : float # Ratio of training to validation data, e.g., 0.9 means 90% training and 10% validation


class GPT2Config(ModelConfig):

    # Compute Device
    # TODO
    
    # Tokenizer
    # TODO
    vocab_size: int = 50257 # Number of tokens in the vocabulary: (50000 BPE Merges + 256 Bytes tokens + 1 EOS token) for GPT-2, or 256 for char-level tokenizer, etc. 

    # Model Architecture Parameters
    seq_size: int = 1024 # Number of tokens in the input sequence. Maximum context length for the predictions
    n_head: int = 12 # Number of attention heads
    n_layer: int = 12 # Number of transformer blocks
    n_embd: int = 768 # Embedding dimension (size of the hidden states)
    
    # Training Parameters
    # TODO