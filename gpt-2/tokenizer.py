import regex as re
import tiktoken


# helper functions for tokenization
def get_stats(ids, counts):
    # return a dictionary of (token, frequency) pairs
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids 

class BasicTokenizer:
    
    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes
        self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern)
    
    
    def train(self, text, vocab_size, verbose=False):
        gpt4 = re.compile(self.pattern)
        parsed = re.findall(gpt4, text)
        num_merges = vocab_size - 256
        
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        print(ids)
        
        def count_pairs(tok):
            pairs = {}
            for pair in zip(tok, tok[1:]):
                pairs[pair] = pairs.get(pair, 0) + 1
            return pairs
        
        def merge(text, pair, idx):
            newids = []
            i = 0
            while i < len(text):
                if i < len(text)-1 and text[i] == pair[0] and text[i+1] == pair[1]:
                    newids.append(idx)
                    i+=2
                else:
                    newids.append(text[i])
                    i+=1
            return newids
        
        for i in range(num_merges):
            pairs = count_pairs(flat_tokens)
            sorted_pairs = sorted(pairs, key = pairs.get, reverse = True)
            to_merge = sorted_pairs[0]
            idx = 256 + i
            flat_tokens = merge(flat_tokens, to_merge, idx)
            self.merges[to_merge] = idx
            self.vocab[idx] = self.vocab[to_merge[0]] + self.vocab[to_merge[1]]
            
            if verbose:
                print(f"Merging {to_merge} → ID {idx}, bytes: {self.vocab[idx]}")
        
        return flat_tokens
        
    def encode(self, text):
        enc = list(text.encode('utf-8'))
        return enc
        
    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        dec = tokens.decode('utf-8', errors = 'replace')
        return dec
        
    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab


