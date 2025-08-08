"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "data/edu_fineweb10B/shards/".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
shards_local_dir = "edu_fineweb10B/shards/"
text_local_dir = "edu_fineweb10B/text/"

remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# Create the cache local directory
SHARDS_CACHE_DIR = os.path.join(os.path.dirname(__file__), shards_local_dir)
os.makedirs(SHARDS_CACHE_DIR, exist_ok=True)
TEXT_CACHE_DIR = os.path.join(os.path.dirname(__file__), text_local_dir)
os.makedirs(TEXT_CACHE_DIR, exist_ok=True)

# # Download dataset with HuggingFace datasets library
# fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# # Tokenize all the documents
# enc = tiktoken.get_encoding("gpt2")
# eot = enc._special_tokens['<|endoftext|>'] # end of text token
# def tokenize(doc):
#     # Tokenizes a single document and returns a numpy array of uint16 tokens
#     tokens = [eot] # Special <|endoftext|> token delimits all documents
#     tokens.extend(enc.encode_ordinary(doc["text"]))
#     tokens_np = np.array(tokens)
#     # Make sure the tokens are within the range of uint16
#     assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16 (range 0 to 2**16-1)"
    
#     # Convert to uint16
#     tokens_np_uint16 = tokens_np.astype(np.uint16)
#     return tokens_np_uint16

# # Saves the tokens to a numpy file: shards
# def write_datafile(filename, tokens_np):
#     np.save(filename, tokens_np)

# # Tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
# nprocs = max(1, os.cpu_count()//2)
# with mp.Pool(nprocs) as pool:
#     shard_index = 0
#     # preallocate buffer to hold current shard
#     all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
#     token_count = 0
#     progress_bar = None
#     for tokens in pool.imap(tokenize, fw, chunksize=16):

#         # is there enough space in the current shard for the new tokens?
#         if token_count + len(tokens) < shard_size:
#             # simply append tokens to current shard
#             all_tokens_np[token_count:token_count+len(tokens)] = tokens
#             token_count += len(tokens)
#             # update progress bar
#             if progress_bar is None:
#                 progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
#             progress_bar.update(len(tokens))
#         else:
#             # write the current shard and start a new one
#             split = "val" if shard_index == 0 else "train"
#             filename = os.path.join(SHARDS_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
#             # split the document into whatever fits in this shard; the remainder goes to next one
#             remainder = shard_size - token_count
#             progress_bar.update(remainder)
#             all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
#             write_datafile(filename, all_tokens_np)
#             shard_index += 1
#             progress_bar = None
#             # populate the next shard with the leftovers of the current doc
#             all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
#             token_count = len(tokens)-remainder

#     # write any remaining tokens as the last shard
#     if token_count != 0:
#         split = "val" if shard_index == 0 else "train"
#         filename = os.path.join(SHARDS_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
#         write_datafile(filename, all_tokens_np[:token_count])
        

# Sanity Check: Count the token for each saved shard
def count_dataset():
    dataset = []
    total_training_tokens = 0
    total_validation_tokens = 0
    # list of all shard file names
    shard_files = [f for f in os.listdir(SHARDS_CACHE_DIR)]
    shard_files.sort()
    for i, shard_file in enumerate(shard_files):
        filename = os.path.join(SHARDS_CACHE_DIR, shard_file)
        if os.path.exists(filename):
            tokens_np = np.load(filename)
            print( filename, i, len(tokens_np))
            total_training_tokens += len(tokens_np) if "train" in shard_file else 0
            total_validation_tokens += len(tokens_np) if "val" in shard_file else 0
    print(f"Total training tokens: {total_training_tokens}, Total validation tokens: {total_validation_tokens}")
    # Optionally, sort the dataset to maintain order
    return dataset

dataset = count_dataset()
