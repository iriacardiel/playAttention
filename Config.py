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
    
    # Dataset and Tokenizer
    selected_tokenizer: str = "CharTokenizer" # Choose tokenizer: "CharTokenizer" or "TiktokenGPT2"
    vocab_size: Optional[int] = None # Number of tokens in the vocabulary: (50000 BPE Merges + 256 Bytes tokens + 1 EOS token) for GPT-2, or 256 for char-level tokenizer, etc.
    train_val_ratio : float = 0.9 # Ratio of training to validation data, e.g., 0.9 means 90% training and 10% validation
    dataset_tokens : int = 1115394 # Number of tokens in the tinyshakespeare dataset. It will be used to calculate training steps.
    dataset_training_tokens : int = int(dataset_tokens * train_val_ratio) # Number of training tokens in the tinyshakespeare dataset
    dataset_validation_tokens : int = int(dataset_tokens - dataset_training_tokens) # Number of validation tokens in the tinyshakespeare dataset

    # Model Architecture Parameters
    seq_size : int = 16 # Number of tokens in the input sequence. Maximum context length for the predictions
    batch_size : int = 64 # Number of sequences in a batch to be processed in parallel 16, 32, 64, etc.
    n_embd : int = 32 # Embedding dimension (size of the hidden states)
    n_head : int = 8 # Number of attention heads
    flash_attention: bool = False # Use flash attention if available (NOT IMPLEMENTED FOR GPT)
    n_layer : int = 4 # Number of transformer blocks
    dropout : float = 0  # Dropout rate for regularization (to avoid overfitting)

    # Training Parameters

    n_epochs: int = 5 # Number of epochs (times the model sees all the training data) to train the model. It will be used to calculate tokens_per_step. TOUCH THIS FIRST
    grad_accum : int = 1 # Gradient accumulation steps. It allows to accumulate gradients over multiple batches before updating the model weights. Use 1 to force the model to process all tokens in the batch at once.
    
    tokens_per_step: int = int(seq_size * batch_size * grad_accum) # Tokens per step. It will be used to calculate the training steps.
    training_steps : int = int(dataset_training_tokens* n_epochs / tokens_per_step) # Number of training steps
    
    lr : float = 1e-3 # Lower if the model is bigger
    lr_warmup_steps: int = 0 # Number of warmup steps for the learning rate scheduler  
    lr_decay: bool = False # To decay the learning rate during training
    gradient_clipping: bool = False # To clip gradients to stabilize training and avoid exploding gradients
    beta1: float = 0.9 # Beta1 for AdamW optimizer (default is 0.9)
    beta2: float = 0.999 # Beta2 for AdamW optimizer (default is 0.999)
    eps: float  = 1e-8 # Epsilon for AdamW optimizer (default is 1e-8)
    weight_decay: float = 0  # Weight decay for AdamW optimizer (default is 0.1, but set to 0 to avoid regularization in small models)
    
    # Evaluation Parameters
    eval_loss_steps : int  = 100 #int(dataset_validation_tokens / (seq_size*batch_size) ) # Number of batches to evaluate the loss on train and val splits
    eval_interval : int = 100 #int(dataset_validation_tokens / (seq_size*batch_size)) # Number of training steps between evaluations
    
    # To be filled after the model is created
    total_params: Optional[int] = None # Total number of parameters in the model, to be calculated after the model is created
    trainable_params: Optional[int] = None # Number of trainable parameters in the model, to be calculated after the model is created
    non_trainable_params: Optional[int] = None # Number of non-trainable parameters in the model, to be calculated after the model is created
    model_size: Optional[str] = None # Size of the model, to be calculated after the model is created (e.g., "small", "medium", "large", etc.)
            
            
class GPT2Config(ModelConfig):

    # Compute Device
    compute_device: Optional[str] = None # 'cpu' or 'cuda'
        
    # Dataset and Tokenizer
    selected_tokenizer: str = "TiktokenGPT2" # Choose tokenizer: "CharTokenizer" or "TiktokenGPT2"
    vocab_size: int = 50304 # (I) Number of tokens in the vocabulary: (50000 BPE Merges + 256 Bytes tokens + 1 EOS token) for GPT-2,  256 or 65 for char-level tokenizer, etc.
    dataset_training_tokens : int = int(10e9) # Number of approx. training tokens in the fineweb dataset (exact number is 9853989344)
    dataset_validation_tokens : int = int(1e8) # Number of validation tokens in the fineweb dataset

    # Model Architecture Parameters
    seq_size: int = 1024 # Number of tokens in the input sequence. Maximum context length for the predictions
    batch_size : int = 32 # Number of sequences in a batch to be processed in parallel 16, 32, 64, etc. 64 is too much for ITB GPU, 32 is ok
    n_head: int = 12 # Number of attention heads
    flash_attention: bool = True # Use flash attention if available
    n_layer: int = 12 # Number of transformer blocks
    n_embd: int = 768 # Embedding dimension (size of the hidden states)
    dropout : float = 0  # Dropout rate for regularization (to avoid overfitting) (NOT IMPLEMENTED FOR GPT2)
    
    # Training Parameters
    
    n_epochs: int = 0.05 # Should be 1 Number of epochs (times the model sees all the training data) to train the model. It will be used to calculate tokens_per_step. TOUCH THIS FIRST
    grad_accum : int = 1 # should be 64 Gradient accumulation steps. It allows to accumulate gradients over multiple batches before updating the model weights. Use 1 to force the model to process all tokens in the batch at once.
    
    tokens_per_step: int = int(seq_size * batch_size * grad_accum) # Tokens per step. It will be used to calculate the training steps. (2**19 = 524288 ~ 0.5M TOKENS PER STEP for GPT2 on fineweb dataset)
    training_steps : int = int(dataset_training_tokens* n_epochs / tokens_per_step) # Number of training steps (19073 for GPT2 on fineweb dataset)
    
    lr : float = 6e-4 # Lower if the model is bigger
    lr_warmup_steps: int = 715 # (V) Number of warmup steps for the learning rate scheduler
    lr_decay: bool = True # To decay the learning rate during training
    gradient_clipping: bool = True # To clip gradients to stabilize training and avoid exploding gradients
    beta1: float = 0.9 # Beta1 for AdamW optimizer (default is 0.9)
    beta2: float = 0.95 # Beta2 for AdamW optimizer (default is 0.999)
    eps: float  = 1e-8 # Epsilon for AdamW optimizer (default is 1e-8)
    weight_decay: float = 0.1 # Weight decay for AdamW optimizer (default is 0.1, but set to 0 to avoid regularization in small models)

    # Evaluation Parameters
    eval_loss_steps : int  = 100 # int(dataset_validation_tokens / (seq_size*batch_size) ) # Number of batches to evaluate the loss on train and val splits
    eval_interval : int = 100 # int(dataset_validation_tokens / (seq_size*batch_size)) # Number of training steps between evaluations
    
    # To be filled after the model is created
    total_params: Optional[int] = None # Total number of parameters in the model, to be calculated after the model is created
    trainable_params: Optional[int] = None # Number of trainable parameters in the model, to be calculated after the model is created
    non_trainable_params: Optional[int] = None # Number of non-trainable parameters in the model, to be calculated after the model is created
    model_size: Optional[str] = None # Size of the model, to be calculated after the model is created (e.g., "small", "medium", "large", etc.)


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