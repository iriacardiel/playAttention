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
    #flash_attention: bool = False # Use flash attention if available (not implemented for GPT1)
    n_layer : int = 3 # Number of transformer blocks
    dropout : float = 0  # Dropout rate for regularization (to avoid overfitting)

    tokens_per_step: int = seq_size*batch_size # Tokens per step. It will be used to accumulate gradients over multiple batches before updating the model weights. Use B*T to force the model to process all tokens in the batch at once.

    # Training Parameters
    training_steps : int = 20000 # Number of training steps
    lr : float = 1e-3 # Lower if the model is bigger, higher if the model is smaller. 
    lr_warmup_steps: int = 0 # Number of warmup steps for the learning rate scheduler  
    lr_decay: bool = False # Whether to decay the learning rate during training
    gradient_clipping: bool = False # Whether to clip gradients to stabilize training and avoid exploding gradients
    beta1: float = 0.9 # Beta1 for AdamW optimizer (default is 0.9)
    beta2: float = 0.999 # Beta2 for AdamW optimizer (default is 0.999)
    eps: float  = 1e-8 # Epsilon for AdamW optimizer (default is 1e-8)
    weight_decay: float = 0 # Weight decay for AdamW optimizer: default is 0.1, but set to 0 to avoid regularization in small models
    
    # Evaluation Parameters
    eval_loss_steps : int  = 100  # Number of batches to evaluate the loss on train and val splits
    eval_interval : int = 100  # Number of training steps between evaluations
    train_val_ratio : float = 0.9 # Ratio of training to validation data, e.g., 0.9 means 90% training and 10% validation
            
            
class GPT2Config(ModelConfig):

    # Compute Device
    compute_device: Optional[str] # 'cpu' or 'cuda'
        
    # Tokenizer
    selected_tokenizer: str = "TiktokenGPT2" # Choose tokenizer: "CharTokenizer" or "TiktokenGPT2"
    vocab_size: int = 50304 # (I) 

    # Model Architecture Parameters
    seq_size: int = 1024 # Number of tokens in the input sequence. Maximum context length for the predictions
    batch_size : int = 8 # Number of sequences in a batch to be processed in parallel 16, 32, 64, etc.
    n_head: int = 12 # Number of attention heads
    flash_attention: bool = True # Use flash attention if available
    n_layer: int = 12 # Number of transformer blocks
    n_embd: int = 768 # Embedding dimension (size of the hidden states)
    dropout : float = 0  # Dropout rate for regularization (to avoid overfitting) ((not implemented for GPT2)

    tokens_per_step: int = seq_size*batch_size  #2**19 # (II) Tokens per step. It will be used to accumulate gradients over multiple batches before updating the model weights. Use B*T to force the model to process all tokens in the batch at once.

    # Training Parameters
    training_steps : int = 19073 # (IV) Number of training steps
    lr : float = 6e-4 # Lower if the model is bigger, higher if the model is smaller. 
    lr_warmup_steps: int = 715 # (V) Number of warmup steps for the learning rate scheduler
    lr_decay: bool = True # Whether to decay the learning rate during training
    gradient_clipping: bool = True # Whether to clip gradients to stabilize training and avoid exploding gradients
    beta1: float = 0.9 # Beta1 for AdamW optimizer (default is 0.9)
    beta2: float = 0.95 # Beta2 for AdamW optimizer (default is 0.999)
    eps: float  = 1e-8 # Epsilon for AdamW optimizer (default is 1e-8)
    weight_decay: float = 0.1 # Weight decay for AdamW optimizer (default is 0.1, but set to 0 to avoid regularization in small models)

    # Evaluation Parameters
    eval_loss_steps : int = 10  # Number of steps to evaluate the validation loss
    eval_interval : int = 10  # Number of training steps between evaluations


'''
Calculations for parameters in GPT2Config on Edufineweb:

(I) vocab_size
vocab_size for gpt-2 tokenizer is 50257, (ugly number: 50000 BPE Merges + 256 Bytes tokens + 1 EOS token)
but we use 50304 to make it a power of 2 

(II) tokens_per_step or tokens per step
in the paper, around 0.5 M "macro" batch size was used
2**19 = 524288 ~ 0.5M TOKENS PER STEP (macro batch size)

(III) tokens to process with the fineweb dataset
100M tokens per shard, total of 100 shards = 10e9 TOTAL TOKENS

(IV) training_steps
that gives 19073 training steps = 10e9 TOTAL TOKENS / 2**19 tokens per step

(V) lr_warmup_steps
in GPT-3 paper they used 375e6 warmup tokens, which is 715 steps (375e6 tokens / 2**19 tokens per step)
so 715 warmup steps
'''