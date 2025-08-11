"""
Training Script
=====================================
"""

from contextlib import nullcontext
import inspect
import math
import os
import sys
import json
from typing import Literal, Tuple
import datetime
import time
import numpy as np
from tqdm import trange
from termcolor import colored,cprint
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist # Run script with: torchrun --standalone --nproc_per_node=1 train_gpt2.py
from torch.nn.parallel import DistributedDataParallel as DDP


# Add the parent directory to the Python path to allow imports from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.custom_tokenizers import tiktoken_tokenizer, char_level_tokenizer
from models.model_gpt import GPTModel
from Config import GPTConfig, ModelConfig


# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# DDP (Distributed Data Parallel): torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
# -------------------------------------------------------------------------------

DDP_ACTIVE = int(os.environ.get('RANK', -1)) != -1 # RANK equals -1 if DDP is not available
if DDP_ACTIVE:
    # DDP mode
    cprint("DDP mode", color="yellow",attrs=["bold"])
    assert torch.cuda.is_available(), "DDP requires CUDA to be available"
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    compute_device = f'cuda:{ddp_local_rank}'
    device = torch.device(compute_device)
    
    # Initialize the process group for DDP
    if ddp_world_size == 1:
        dist.init_process_group(backend='gloo') 
    else:
        dist.init_process_group(backend='nccl')  # TODO: Test on multiple GPUs

    torch.cuda.set_device(device)  # Set the current device to the local rank's GPU

else:
    # NON DDP mode
    cprint(f"Single process mode (not DDP)", color="yellow", attrs=["bold"])
    ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1  # Default values for single process mode

    compute_device = "cpu"
    if torch.cuda.is_available():
        compute_device = f'cuda:{ddp_local_rank}'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        compute_device = "mps"
    device = torch.device(compute_device)

master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

compute_color_map = {}
color_list = ["cyan", "red", "orange", "green", "blue", "indigo", "violet"]
for i in range(len(color_list)):
    compute_color_map[f"cuda:{i}"] = color_list[i % len(color_list)]
compute_color_map["cpu"] = "light_grey"
compute_color_map["mps"] = "dark_grey"
# Set color for the compute device
compute_color = compute_color_map.get(compute_device, "white")  # Default to white if not found
device_type = "cuda" if compute_device.startswith("cuda") else compute_device
cprint(f"DDP active = {DDP_ACTIVE}. Device: {compute_device} of type {device_type}: ddp_rank = {ddp_rank}, ddp_local_rank = {ddp_local_rank}, ddp_world_size = {ddp_world_size}", compute_color)
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float16', 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

# Reproducibility settings
# -----------------------
seed = 1337
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed) 

# GPU Optimization Settings
# -----------------------
TENSOR_CORES = False  # Set to True to enable Tensor Cores for faster matrix multiplications on supported GPUs
TORCH_COMPILATION = False  # Set to True to enable PyTorch 2.0's compile feature for performance optimization
AUTOCAST = False  # Mixed precision training https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html

torch.set_float32_matmul_precision('high') if TENSOR_CORES else None
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' or AUTOCAST == False else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# File paths
TRAIN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M") # Unique identifier for this training session
DATA_PATH = 'data/tiny_shakespeare/text/tinyshakespeare.txt'
REPORT_DIR = f'results/GPT_training_{TRAIN_ID}_{compute_device}'
# Create plots directory
os.makedirs(REPORT_DIR, exist_ok=True)
# =============================================================================
# CONFIGURATION 
# =============================================================================

config = GPTConfig(compute_device=compute_device)

# =============================================================================
# DATA PREPARATION
# =============================================================================
if master_process:
    cprint("DATA PREPARATION", compute_color)

tokenizer = char_level_tokenizer if config.selected_tokenizer == char_level_tokenizer.name else tiktoken_tokenizer
config.vocab_size = tokenizer.n_vocab  if config.vocab_size is None else config.vocab_size # Set vocabulary size based on the tokenizer if it is not provided in the config

class DataLoaderFromTxt:
    def __init__(self, B:int, T:int, process_rank:int=0, num_processes:int=1, split:Literal['train', 'val']='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split  # Store split type
        assert split in {'train','val'}

        # At init load tokens from disk and store them in memory
        try:
            with open(DATA_PATH, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        
        tokens = tokenizer.encode(text)  # Encode the text into tokens
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        # Split the data into training and validation sets
        split_idx = int(config.train_val_ratio * len(self.tokens)) 
        self.train_tokens = self.tokens[:split_idx] # train split 
        self.val_tokens = self.tokens[split_idx:] # validation split
        
        if master_process:
            cprint(f"loaded {len(self.tokens)} tokens", compute_color)
            cprint(f"1 epoch = {len(self.tokens)// (self.B*self.T)} batches", compute_color)
            data_preparation_summary = (
            f"\n DATASET CALCULATIONS (Sanity Check):\n"
            f"  Tokenizer: {tokenizer.name}\n"
            f"  Tokenized text: {len(self.tokens):,} tokens\n"
            f"  Vocabulary size: {tokenizer.n_vocab} unique tokens\n"
            f"\nData split:\n"
            f"  Training:   {len(self.train_tokens):,} tokens ({len(self.train_tokens)/len(self.tokens)*100:.1f}%)\n"
            f"  Validation: {len(self.val_tokens):,} tokens ({len(self.val_tokens)/len(self.tokens)*100:.1f}%)\n"
            )
        
            cprint(data_preparation_summary, compute_color)

        self.reset()
        self.vocab_size = tokenizer.n_vocab
            
    def reset(self):
        """
        Reset the data loader to the initial state.
        """
        self.current_position = self.process_rank * self.B * self.T  # state: initialize current position based on process rank to ensure each process gets a unique subset of the data
    
    # Batch generator
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of input-target pairs for training.
            
            How batching works:
            - Sample random starting positions in the dataset
            - Extract sequences of length seq_size starting from those positions
            - Create targets by shifting input sequences by one position
            
            Example with seq_size=5, batch_size=2:
                Input:  [[23, 30, 31, 7, 21], [45, 12, 8, 33, 9]]
                Target: [[30, 31, 7, 21, 14], [12, 8, 33, 9, 41]]
            
            This gives us seq_size training examples per sequence:
                - Predict 30 given [23]
                - Predict 31 given [23, 30]
                - Predict 7 given [23, 30, 31]
                - etc.
                
        This implementation supports:
        - Supports DDP training by ensuring that each process gets a unique subset of the data based on its rank.
        
        """
        B, T = self.B, self.T
        data = self.train_tokens if self.split == 'train' else self.val_tokens  # Use train or validation tokens based on split
        # v1
        # starting_idx = torch.randint(len(data) - T, (B,))         # Sample random starting indices for each sequence in the batch
        # xb = torch.stack([data[i:i+T] for i in starting_idx])          # Extract sequences and create targets
        # yb = torch.stack([data[i+1:i+T+1] for i in starting_idx])      # Shift the sequence by one to the right to create the target

        # v2
        #Ensure current_position does not exceed data length
        if self.current_position >= len(data):
            self.reset()

        buf = data[self.current_position: self.current_position + B * T + 1]

        # Safeguard against empty slices
        if len(buf) < B * T + 1:
            raise ValueError(f"Insufficient data to generate batch: requested {B * T + 1}, but got {len(buf)}")

        xb = buf[:-1].view(B,T) # inputs
        yb = buf[1:].view(B,T)  # targets (shifted) 
        self.current_position += B * T * self.num_processes # advance the current position by B*T*num_processes tokens to ensure each process gets a unique subset of the data

        # If we reach the end of the tokens, reset to the beginning
        if self.current_position + (B * T * self.num_processes + 1) > len(data):
            self.reset()

        # Move to compute device (GPU or CPU)
        xb, yb = xb.to(device), yb.to(device)

        return xb, yb

# Create dataloader instance
train_loader = DataLoaderFromTxt(B=config.batch_size, T=config.seq_size, process_rank=ddp_rank, num_processes=ddp_world_size, split = 'train')
val_loader_avg = DataLoaderFromTxt(B=config.batch_size, T=config.seq_size, process_rank=ddp_rank, num_processes=ddp_world_size, split = 'val')
train_loader_avg = DataLoaderFromTxt(B=config.batch_size, T=config.seq_size, process_rank=ddp_rank, num_processes=ddp_world_size, split = 'train')

# Microsteps in training steps: We will repeat the forward - backward grad_acc_microsteps times to simulate a larger batch size. We will call this micro-step.
tokens_per_step = config.tokens_per_step #config.tokens_per_step # 2**19  approx. 0.5M like in the paper
total_batch_size = config.batch_size * config.seq_size  # Total batch size across all processes (DDP)
assert tokens_per_step % (total_batch_size*ddp_world_size) == 0, "tokens_per_step must be divisible by total_batch_size * ddp_world_size"
grad_acc_microsteps = tokens_per_step // (total_batch_size*ddp_world_size)

if master_process:
    cprint("\nBATCH CALCULATIONS (Sanity Check)", compute_color)
    cprint(f"{tokens_per_step} tokens_per_step to cover {config.n_epochs} epochs of training data", compute_color)
    cprint(f"Microstep batch size (B*T): {config.batch_size} * {config.seq_size} = {total_batch_size} tokens", compute_color)
    cprint(f"Grad accumulation microsteps (tokens_per_step // (total_batch_size*ddp_world_size)): {grad_acc_microsteps}", compute_color)

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
if master_process:
    cprint("\nMODEL INITIALIZATION", compute_color)


# Create model instance
model = GPTModel(config)
model.to(device) # this only works for the model, for tensors do tensor = tensor.to(device)


# Speed ups the model with PyTorch 2.0's compile feature (optional, but recommended for performance) 
# https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html 
if TORCH_COMPILATION:
    model = torch.compile(model)

# DDP Wrapper for the model
if DDP_ACTIVE:
    model = DDP(model, device_ids = [ddp_local_rank]) # DDP fails here if using 1 GPU and backed = 'nccl' (it requires at least 2 GPUs), so we use 'gloo' backend for single GPU training

if master_process:
    # Count model parameters
    config.total_params = sum(p.numel() for p in model.parameters())
    config.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config.non_trainable_params = config.total_params - config.trainable_params
    model_size = config.total_params * 4 / 1024**2  # Size in MB (assuming float32, 4 bytes per parameter)
    config.model_size = f"{model_size:.2f} MB (float32)"
    
    cprint("Configuration", compute_color)
    cprint(config.model_dump_json(indent=2), compute_color)

# Print model architecture
# -----------------------
# Option 1
#cprint(model, compute_color)
# Option 2
# for k, v in model.state_dict().items():
#     cprint(f"{k}: {v.shape} - {v.dtype}", compute_color)  # Print each parameter's shape and dtype


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

# New! Learning rate scheduler function
def get_lr(step: int, config: ModelConfig) -> float:
    """
    Get the learning rate for the current training step.
    Uses a cosine decay schedule with warmup.
    """
    max_lr = config.lr  # Maximum learning rate
    min_lr = max_lr * 0.1  # Minimum learning rate (10% of max)
    lr_warmup_steps = config.lr_warmup_steps  # Number of warmup steps
    total_steps = config.training_steps  # Total training steps
    # If no decay and no warmup, return max_lr
    if not config.lr_decay and lr_warmup_steps == 0:
        return max_lr
    # (1) warmup
    if step < lr_warmup_steps:
        return max_lr * (step+1) / lr_warmup_steps  # Linear warmup
    # (2) cosine decay
    elif step >= lr_warmup_steps and step <= total_steps and config.lr_decay:
        decay_ratio = (step - lr_warmup_steps) / (total_steps - lr_warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)
    # (3) after training
    else: 
        return min_lr

# New! Configure optimizers
def configure_optimizers(model:torch.nn.Module, config: ModelConfig, device_type:str) -> torch.optim.Optimizer:
    """
    Configure optimizers for the model:
    - AdamW optimizer with weight decay for regularization
    - Fused version if available for better performance on CUDA
    - Set betas, epsilon and learning rate according to the config
    """
    param_dict = {pn: p for pn, p in model.named_parameters()} # Make dictionary of parameter names and tensors
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # Filter out parameters that do not require gradients
    
    # create optimization groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    params_to_optimize = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    #print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(params_to_optimize, lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, fused=use_fused)

    return optimizer
# =============================================================================
# TRAINING 
# =============================================================================

cprint("TRAINING", compute_color)


# Initialize lists to store metrics for plotting
train_loss_list, train_loss_avg_list, val_loss_avg_list, lr_list, norm_list, duration_list, tokens_per_sec_list = [], [], [], [], [], [], []
train_steps_list, train_steps_avg_list, val_steps_avg_list = [], [], []
# Initialize dict at the top of your script
gradient_l2_log = {"wte":[], "wpe":[], "lm_head": []}

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler('cuda', enabled=(device_type == 'cuda' and AUTOCAST and dtype == 'float16'))

# Initialize optimizer
if DDP_ACTIVE:
    optimizer = configure_optimizers(model.module, config, device_type)
else:
    optimizer = configure_optimizers(model, config, device_type)
    
generated_samples_log = os.path.join(REPORT_DIR, "generated_samples.txt")
log_file = os.path.join(REPORT_DIR, f"log.csv")
with open(log_file, "w") as f: # open for writing to clear the file
    f.write(f"Step;Train Loss;Train Loss Avg;Val Loss Avg;lr;norm;dt (s);Tokens/s\n")

start_train_loop = time.time()
pbar = trange(config.training_steps, desc="Training steps", unit="step", disable=not master_process, colour=compute_color)
for step in pbar:
    last_step = (step == config.training_steps - 1)

    # EVALUATION PHASE
    # ---------------------------------------------------------------------------------------------------------------------------------------
    if step % config.eval_interval == 0 or last_step:
        
        # (1) Evaluate the model on the validation set for eval_loss_steps steps
        model.eval()
        val_loader_avg.reset()  # Reset the validation loader to the start of the validation data
        train_loader_avg.reset()  # Reset the training loader to the start of the training data
        with torch.no_grad():
            val_loss_avg = 0.0
            train_loss_avg = 0.0
            for _ in range(config.eval_loss_steps):
                xb_val, yb_val = val_loader_avg.next_batch()
                xb_train, yb_train = train_loader_avg.next_batch()
                with ctx:
                    _, val_loss = model(xb_val, yb_val)
                    val_loss = val_loss / config.eval_loss_steps
                    val_loss_avg += val_loss.detach()  # Accumulate validation loss

                    _, train_loss = model(xb_train, yb_train)
                    train_loss = train_loss / config.eval_loss_steps
                    train_loss_avg += train_loss.detach()  # Accumulate training loss


        if DDP_ACTIVE:
            dist.all_reduce(val_loss_avg, op=dist.ReduceOp.SUM)
            val_loss_avg /= ddp_world_size  # Average the validation loss across all DDP processes
            dist.all_reduce(train_loss_avg, op=dist.ReduceOp.SUM)
            train_loss_avg /= ddp_world_size  # Average the training loss across all DDP processes
            
        if master_process:
            # Store metrics for plotting
            val_loss_avg_list.append(val_loss_avg.cpu().item() if val_loss_avg.is_cuda else val_loss_avg.item())
            val_steps_avg_list.append(step)
            train_loss_avg_list.append(train_loss_avg.cpu().item() if train_loss_avg.is_cuda else train_loss_avg.item())
            train_steps_avg_list.append(step)
            
            # TODO: Checkpoints

        # (2) Generate text from the model
        if True:
            # Context tokens
            context_text = "\n"
            context_tokens = tokenizer.encode(context_text)
            context_tokens = torch.tensor(context_tokens, dtype=torch.long)

            # Manually generating a batch with the same sequence context_tokens 5 times 
            num_generated_sequences = 5
            max_new_tokens = 24

            context_tokens = context_tokens.unsqueeze(0).repeat(num_generated_sequences, 1)
            idx = context_tokens.to(device)

            # Generate from context tokens
            if DDP_ACTIVE:
                generated_tokens = model.module.generate(idx, max_new_tokens, ddp_rank, device)
            else:
                model.eval()
                generated_tokens = model.generate(idx, max_new_tokens, ddp_rank, device)

            if master_process:
                with open(generated_samples_log, "a") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Step {step} — Generated Samples\n")
                    f.write(f"{'-'*80}\n")
                    for i in range(num_generated_sequences):
                        sample_tokens = generated_tokens[i].tolist()
                        decoded_text = tokenizer.decode(sample_tokens)
                        f.write(f"[Sample {i+1}]\n")
                        f.write(f"{decoded_text.strip()}\n\n")

    
    # TRAINING PHASE
    # ---------------------------------------------------------------------------------------------------------------------------------------
    t0 = time.time()

    model.train()
    optimizer.zero_grad() # Reset gradients

    # Macro/Micro Batch: Accumulate gradients over multiple micro-steps to simulate a larger macro batch size
    train_loss_acc = 0.0
    for micro_step in range(grad_acc_microsteps):
        
        # Sample a batch random of training data
        xb, yb = train_loader.next_batch()

        if DDP_ACTIVE:
            model.require_backward_grad_sync = (micro_step == grad_acc_microsteps - 1)  # Only sync gradients on the last micro-step
        
        # Forward pass
        with ctx:
            logits, loss = model(xb, yb)
        
        loss = loss / grad_acc_microsteps # Scale the loss by the number of micro-steps to average it out
        train_loss_acc += loss.detach()  # Accumulate loss over micro-steps
        
        # Backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    if DDP_ACTIVE:

        dist.all_reduce(train_loss_acc, op=dist.ReduceOp.SUM)  # Aggregate loss across all processes in DDP
        train_loss_acc /= ddp_world_size  # Average the loss across all DDP processes
        # Note! dist.ReduceOp.AVG is not a primitive operation and failed with 'goo' backend, while SUM is primitive operation that does not fail. Manually dividing by ddp_world_size later gets the average loss across all processes.
        
    # Clipping gradients: stabilize training 
    if config.gradient_clipping:
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    else:
        norm = torch.tensor(0.0, device=device)
        
    # Log gradient L2 norms 
    for name, param in model.named_parameters():
        if param.grad is None:
            continue  # Skip parameters without gradients
        
        grad_norm = param.grad.data.norm(2).item()
        for substr in gradient_l2_log.keys():
            if substr in name.lower():
                gradient_l2_log[substr].append(grad_norm)
   
    # Learning Rate Scheduler
    lr = get_lr(step, config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Update weights and scaler
    scaler.step(optimizer)
    scaler.update()
    
    
    torch.cuda.synchronize() if device_type == "cuda" else None  # Synchronize for accurate timing
    t1 = time.time()
    dt = t1 - t0

    tokens_processed = train_loader.B * train_loader.T * grad_acc_microsteps * ddp_world_size # Total tokens processed in this step
    tokens_per_second = tokens_processed / dt

    # LOGGING EVALUATION AND TRAINING METRICS
    # ---------------------------------------------------------------------------------------------------------------------------------------
    if master_process:
        # Store metrics for plotting
        train_loss_list.append(train_loss_acc.cpu().item() if train_loss_acc.is_cuda else train_loss_acc.item())  # Convert to Python float for plotting
        lr_list.append(lr)
        norm_list.append(norm.cpu().item() if norm.is_cuda else norm.item())
        duration_list.append(dt)
        tokens_per_sec_list.append((tokens_per_second))
        train_steps_list.append(step)
        
        with open(log_file, "a") as f:
            val_str = f"{val_loss_avg:.4}" if step % config.eval_interval == 0 or last_step else "NA"
            train_str = f"{train_loss_avg:.4}" if step % config.eval_interval == 0 or last_step else "NA"

            f.write(f"{step};{train_loss_acc:.4};{train_str};{val_str};{lr:.4};{norm:.4};{dt:.4};{tokens_per_second}\n")

        if step % config.eval_interval == 0 or last_step:
            
            # Plot metrics in real-time
            if step != 0:
                
                # Loss plot
                plt.figure(figsize=(10, 6))
                plt.plot(train_steps_list, train_loss_list, label=f'Train Loss', color='tab:blue', alpha = 0.5)
                plt.plot(train_steps_avg_list, train_loss_avg_list, label=f'Train Loss Avg (last={train_loss_avg_list[-1]:.4f})', color='royalblue')
                plt.plot(val_steps_avg_list, val_loss_avg_list, label=f'Val Loss Avg (last={val_loss_avg_list[-1]:.4f})', color='tab:orange')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.yticks(np.arange(round(min(train_loss_list),1)-1, round(max(train_loss_list),1)+ 1, 0.2))  # Set y-ticks for better readability
                plt.xlim(0, config.training_steps +1)  # Set x-axis limit to training steps
                plt.xticks(np.arange(0,config.training_steps + 1, config.training_steps // 10))  # Set x-ticks for better readability
                plt.title(f'Loss over Steps | Training time ~ {str(datetime.timedelta(seconds=int(t1-start_train_loop)))}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{REPORT_DIR}/losses.png')
                
                # Gradient L2 plot
                plt.figure(figsize=(10, 6))
                for substr in gradient_l2_log.keys():
                    if len(gradient_l2_log[substr]) > 0:  # Check if there are any gradients logged for this substring
                        plt.plot(train_steps_list, gradient_l2_log[substr], label=f'{substr} Grad L2 Norm', alpha=0.5)
                plt.xlabel('Step')
                plt.ylabel('Gradient L2 Norm')
                plt.xlim(0, config.training_steps +1)  # Set x-axis limit to training steps
                plt.xticks(np.arange(0,config.training_steps + 1, config.training_steps // 10))  # Set x-ticks for better readability
                plt.title('Gradient L2 Norm over Steps')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{REPORT_DIR}/gradient_l2_norms.png')

                # Learning rate plot
                plt.figure(figsize=(20, 12))

                plt.subplot(2, 2, 1)
                plt.plot(train_steps_list, lr_list, label='Learning Rate', color='orange')
                plt.xlabel('Step')
                plt.ylabel('Learning Rate')
                #plt.yticks(np.arange(0, max(lr_list) + 1e-4, 1e-4))  # Set y-ticks for better readability
                plt.xlim(0, config.training_steps +1)  # Set x-axis limit to training steps
                plt.xticks(np.arange(0,config.training_steps + 1, config.training_steps // 10))  # Set x-ticks for better readability
                plt.title('Learning Rate over Steps')
                plt.legend()
                plt.grid(True)

                # Gradient norm plot
                plt.subplot(2, 2, 2)
                plt.plot(train_steps_list, norm_list, label='Gradient Norm', color='green')
                plt.xlabel('Step')
                plt.ylabel('Norm')
                plt.xlim(0, config.training_steps +1)  # Set x-axis limit to training steps
                plt.xticks(np.arange(0,config.training_steps + 1, config.training_steps // 10))  # Set x-ticks for better readability
                plt.title('Gradient Norm over Steps')
                plt.legend()
                plt.grid(True)

                # Duration plot
                plt.subplot(2, 2, 3)
                plt.plot(train_steps_list, duration_list, label=f'Duration', color='red', alpha = 0.7)
                mean_duration = round(np.mean(duration_list[1:]), 3)
                plt.plot(train_steps_list, [mean_duration] * len(train_steps_list), label=f'Mean Duration ~ {mean_duration:.4f}', color='crimson', linestyle='--')
                plt.xlabel('Step')
                plt.ylabel('Duration (s)')
                #plt.yticks(np.arange(round(min(duration_list[1:])) - round(0.1* max(duration_list[1:])), round(max(duration_list[1:])) + round(0.1* max(duration_list[1:])), 0.1))  # Set y-ticks for better readability
                plt.xlim(0, config.training_steps +1)  # Set x-axis limit to training steps
                plt.xticks(np.arange(0,config.training_steps + 1, config.training_steps // 10))  # Set x-ticks for better readability
                plt.title('Duration per Step')
                plt.legend()
                plt.grid(True)

                # Tokens per second plot
                plt.subplot(2, 2, 4)
                plt.plot(train_steps_list, tokens_per_sec_list, label='Tokens/sec', color='purple', alpha = 0.7)
                mean_tokens_per_sec = round(np.mean(tokens_per_sec_list[1:]),0)
                plt.plot(train_steps_list, [mean_tokens_per_sec] * len(train_steps_list), label=f'Mean Tokens/sec ~ {mean_tokens_per_sec:.4f}', color='blueviolet', linestyle='--')
                plt.xlabel('Step')
                plt.ylabel('Tokens/sec')
                #plt.yticks(np.arange(round(min(tokens_per_sec_list[1:])) - round(0.1* max(tokens_per_sec_list[1:])), round(max(tokens_per_sec_list[1:])) + round(0.1* max(tokens_per_sec_list[1:])), 1000))  # Set y-ticks for better readability
                plt.xlim(0, config.training_steps +1)  # Set x-axis limit to training steps
                plt.xticks(np.arange(0,config.training_steps + 1, config.training_steps // 10))  # Set x-ticks for better readability
                plt.title('Tokens/sec over Steps')
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(f'{REPORT_DIR}/training_metrics_summary.png')
    
            plt.close('all')
        pbar.postfix = f"train_loss_acc {train_loss_acc}"  # Update the progress bar postfix with the current step


end_train_loop = time.time()
if master_process:
    model.eval()
    cprint(f"\nTraining completed in {end_train_loop - start_train_loop} (HH:MM:SS)", compute_color)
    cprint(f"Final training loss: {train_loss_acc:.4f}, Final validation loss: {val_loss_avg:.4f}", compute_color)
    
    # Save configuration as JSON
    with open(f'{REPORT_DIR}/config.json', 'w', encoding='utf-8') as f:
        config_dict = config.model_dump()
        f.write(json.dumps(config_dict, indent=2))
        
        # =============================================================================
        # INFERENCE & TEXT GENERATION
        # =============================================================================
        # Context tokens
        context_text = "\n"
        context_tokens = tokenizer.encode(context_text)
        context_tokens = torch.tensor(context_tokens, dtype=torch.long)
        idx = context_tokens.to(device).unsqueeze(0)  # Shape becomes (1, seq_len)

        # Generate from context tokens
        if DDP_ACTIVE:
            generated_tokens = model.module.generate(idx, max_new_tokens=500, ddp_rank=ddp_rank, device=device)[0].tolist()
        else:
            model.eval()
            generated_tokens = model.generate(idx, max_new_tokens=500)[0].tolist()
        generated_text = tokenizer.decode(generated_tokens)
        print("Generated text: <START>", colored(generated_text, compute_color), "<END>")

dist.destroy_process_group() if DDP_ACTIVE else None  # Clean up DDP resources
