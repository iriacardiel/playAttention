.PHONY: all train_gpt train_gpt2

all: train_gpt train_gpt2

train_gpt:
	@echo "Running train_gpt.py..."
	torchrun --standalone --nproc_per_node=1 train_gpt.py

train_gpt2:
	@echo "Running train_gpt2.py..."
	torchrun --standalone --nproc_per_node=1 train_gpt2.py