"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "data/tiny_shakespeare/shards/".
"""

import os
import multiprocessing as mp
import numpy as np
from custom_tokenizers import tiktoken_tokenizer, char_level_tokenizer
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
shards_local_dir = "data/tiny_shakespeare/shards/"
text_local_dir = "data/tiny_shakespeare/text/"
shard_size = int(2e6) # 1115394 < 1e7 tokens per shard, total of 1 shard

# Create the cache local directory
SHARDS_CACHE_DIR = os.path.join(os.path.dirname(__file__), shards_local_dir)
os.makedirs(SHARDS_CACHE_DIR, exist_ok=True)

TEXT_CACHE_DIR = os.path.join(os.path.dirname(__file__), text_local_dir)
os.makedirs(TEXT_CACHE_DIR, exist_ok=True)

# Tokenize all the documents
enc = char_level_tokenizer

# Load dataset from local file 
shakespeare_file = "data/tiny_shakespeare/text/tinyshakespeare.txt"
try:
    with open(shakespeare_file, 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found: {shakespeare_file}")
        
# Each document text can't be longer than 2e6 tokens so we will split it if necessary
tokens = enc.encode(text)  # Encode the text into tokens
train_val_ratio = 0.9  # 90% training, 10% validation
total_tokens = len(tokens)  # Total number of tokens in the dataset
print(f"Total tokens in dataset: {total_tokens}")
print(f"Using {train_val_ratio} for train/val split.")


# Generate splits: train , val
split_idx = int(train_val_ratio * total_tokens) 
train_tokens = tokens[:split_idx] # train split
val_tokens = tokens[split_idx:] # validation split

# Convert to numpy arrays
# Note: This is necessary for saving to .npy files later
train_tokens_np = np.array(train_tokens)
val_tokens_np = np.array(val_tokens)
train_tokens_np_uint16 = train_tokens_np.astype(np.uint16)
val_tokens_np_uint16 = val_tokens_np.astype(np.uint16)

    
split = "val"  # validation split
shard_index = 0
filename = os.path.join(SHARDS_CACHE_DIR, f"tinyshakespeare_{split}_{shard_index:06d}")
np.save(filename, val_tokens_np_uint16) # Saves the tokens to a numpy file: shards


split = "train"  # training split
shard_index = 1
filename = os.path.join(SHARDS_CACHE_DIR, f"tinyshakespeare_{split}_{shard_index:06d}")
np.save(filename, train_tokens_np_uint16)

# Sanity Check: Reconstruct the dataset from shards
def reconstruct_dataset():
    dataset = []
    # list of all shard file names
    shard_files = [f for f in os.listdir(SHARDS_CACHE_DIR)]
    shard_files.sort()
    for shard_file in shard_files:
        filename = os.path.join(SHARDS_CACHE_DIR, shard_file)
        if os.path.exists(filename):
            tokens_np = np.load(filename)
            for token in tokens_np:
                dataset.append(enc.decode([token]))
    print(f"Reconstructed dataset with {len(dataset)} items from {len(shard_files)} shards.")
    # Optionally, sort the dataset to maintain order
    return dataset

dataset = reconstruct_dataset()

# Save to txt file to verify
output_file = os.path.join(TEXT_CACHE_DIR, "reconstructed_tinyshakespeare.txt")
with open(output_file, 'w', encoding='utf-8') as f:
    for item in dataset:
        f.write(item)
        
        
# Sanity Check: Count the token for each saved shard
def count_dataset():
    dataset = []
    # list of all shard file names
    shard_files = [f for f in os.listdir(SHARDS_CACHE_DIR)]
    shard_files.sort()
    for shard_file in shard_files:
        filename = os.path.join(SHARDS_CACHE_DIR, shard_file)
        if os.path.exists(filename):
            tokens_np = np.load(filename)
            print(len(tokens_np))
    # Optionally, sort the dataset to maintain order
    return dataset

dataset = count_dataset()