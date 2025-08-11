.PHONY: all train_gpt train_gpt2

all: train_gpt train_gpt2

train_gpt:
	@echo "Running train_GPT.py..."
	torchrun --standalone --nproc_per_node=1 train/train_GPT.py

train_gpt2:
	@echo "Running train_GPT2.py..."
	torchrun --standalone --nproc_per_node=1 train/train_GPT2.py