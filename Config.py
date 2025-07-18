"""
Model Configurations
=====================================
"""

from typing import Optional
from pydantic import BaseModel

class ModelConfig(BaseModel):
    pass

class GPTConfig(ModelConfig):

    # Compute Device
    compute_device: Optional[str] = None # 'cpu' or 'cuda'
    
    # Tokenizer
    selected_tokenizer: str = "CharTokenizer" # Choose tokenizer: "CharTokenizer" or "TiktokenGPT2"
    vocab_size: Optional[int] = None # Number of tokens in the vocabulary: (50000 BPE Merges + 256 Bytes tokens + 1 EOS token) for GPT-2, or 256 for char-level tokenizer, etc.
    
    # Model Architecture Parameters
    seq_size : int = 8 # Number of tokens in the input sequence. Maximum context length for the predictions
    batch_size : int = 32 # Number of sequences in a batch to be processed in parallel
    n_embd : int = 32 # Embedding dimension (size of the hidden states)
    n_head : int = 4 # Number of attention heads
    n_layer : int = 3 # Number of transformer blocks
    dropout : float = 0  # Dropout rate for regularization (to avoid overfitting)

    macro_batch_size: int = None # New! Macro batch size for training. Set to none to disable macro batching. If set, it will be used to accumulate gradients over multiple batches before updating the model weights.

    # Training Parameters
    training_steps : int = 2000 # Number of training steps
    lr : float = 1e-3 # Lower if the model is bigger, higher if the model is smaller. 
    lr_warmup_steps: int = 0 # New! Number of warmup steps for the learning rate scheduler  
    lr_decay: bool = False # New! Whether to decay the learning rate during training
    gradient_clipping: bool = False # New! Whether to clip gradients to stabilize training and avoid exploding gradients
    beta1: float = 0.9 # New! Beta1 for AdamW optimizer
    beta2: float = 0.999 # New! Beta2 for AdamW optimizer
    eps: float  = 1e-8 # New! Epsilon for AdamW optimizer
    weight_decay: float = 0 # New! Weight decay for AdamW optimizer: default is 0.1, but set to 0 to avoid regularization in small models
    
    # Evaluation Parameters
    eval_iters : int  = 100  # Number of batches to evaluate the loss on train and val splits
    eval_interval : int = 100  # Number of training steps between evaluations
    train_val_ratio : float = 0.9 # Ratio of training to validation data, e.g., 0.9 means 90% training and 10% validation
            
            
class GPT2Config(ModelConfig):

    # Compute Device
    compute_device: Optional[str] # 'cpu' or 'cuda'
        
    # Tokenizer
    selected_tokenizer: str = "TiktokenGPT2" # Choose tokenizer: "CharTokenizer" or "TiktokenGPT2"
    vocab_size: int = 50304 # New! 50304 Increased from 50257 to the nearest power of 2 (ugly number: 50000 BPE Merges + 256 Bytes tokens + 1 EOS token) for GPT-2

    # Model Architecture Parameters
    seq_size: int = 1024 # Number of tokens in the input sequence. Maximum context length for the predictions
    batch_size : int = 16 # Number of sequences in a batch to be processed in parallel
    n_head: int = 12 # Number of attention heads
    flash_attention: bool = True # New! Use flash attention if available
    n_layer: int = 12 # Number of transformer blocks
    n_embd: int = 768 # Embedding dimension (size of the hidden states)
    dropout : float = 0  # Dropout rate for regularization (to avoid overfitting) TODO: NOT IMPLENTED YET

    macro_batch_size: int = 524288 # New! Macro batch size for training. Set to none to disable macro batching. If set, it will be used to accumulate gradients over multiple batches before updating the model weights.
    
    # Training Parameters
    training_steps : int = 100 # Number of training steps
    lr : float = 6e-4 # Lower if the model is bigger, higher if the model is smaller.
    lr_warmup_steps: int = 10 # New! Number of warmup steps for the learning rate scheduler  
    lr_decay: bool = True # New! Whether to decay the learning rate during training
    gradient_clipping: bool = True # New! Whether to clip gradients to stabilize training and avoid exploding gradients
    beta1: float = 0.9 # New! Beta1 for AdamW optimizer: default is 0.9
    beta2: float = 0.95 # New! Beta2 for AdamW optimizer: default is 0.999
    eps: float  = 1e-8 # New! Epsilon for AdamW optimizer: default is 1e-8
    weight_decay: float = 0.1 # New! Weight decay for AdamW optimizer: default is 0.1, but set to 0 to avoid regularization in small models
    
    # Evaluation Parameters
    # TODO
