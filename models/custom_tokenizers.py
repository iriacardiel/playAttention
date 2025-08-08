import tiktoken

# Load text file
with open('data/tiny_shakespeare/text/tinyshakespeare.txt', 'r') as f:
    train_text = f.read()
    
# CHARACTER LEVEL TOKENIZERS
# --------------------------
# (option 0) Self implementation of character level tokenizer ("no training" needed)
class CharTokenizer:
    def __init__(self, text):
        self.name = "CharTokenizer"
        self.text = text
        self.vocab = sorted(list(set(text)))
        self.n_vocab = len(self.vocab)
        self.char_to_int = {ch: i for i, ch in enumerate(self.vocab)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, s):
        return [self.char_to_int[ch] for ch in s]

    def decode(self, l):
        return ''.join([self.int_to_char[i] for i in l])
    
char_level_tokenizer = CharTokenizer(train_text)


# WORD LEVEL TOKENIZERS (BPE)
# --------------------------
# (option 1) Self implementation of Byte Pair Encoding (BPE) tokenizer in playBPE repository (https://github.com/iriacardiel/playBPE) (needs training to merge byte pairs)
# (option 2) Loading the tiktoken tokenizer for GPT-2 (already trained based on BPE)
tiktoken_tokenizer = tiktoken.get_encoding("gpt2")
tiktoken_tokenizer.name = "TiktokenGPT2"

if __name__ == "__main__":
    tokenizer = char_level_tokenizer 
    print(f"Tokenizer: {tokenizer.name}")
    print(f"Vocabulary size: {tokenizer.n_vocab}")
    encoded = tokenizer.encode("hello world")
    decoded = tokenizer.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    