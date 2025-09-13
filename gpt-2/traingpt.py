import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.shape # batch, time, channels    
        qkv = self.c_attn(x)
        
        q, k, v = qkv.split(self.n_embd, dim = 2) # split the neurons of the attention layer into the qkv vectors
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, n_head, T, C -> n_head is like a second batch dimension!
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        #apply attention formula
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf")) # masks out the future tokens with -inf
        att = F.softmax(att, dim = -1) # softmax over the channels to get probabilities and zero out future
        
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C) # return to initial shape
        y = self.c_proj(y)
        
        return y
        
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd*4)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(config.n_embd*4, config.n_embd)

    def forward(self,x):
        x = self.c_fc(x) # maps up
        x = self.gelu(x) # approximate gelu activation
        x= self.c_proj(x) # maps down
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # residual connections
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size = 1024 # context length
    vocab_size = 50257 # 50,000 BPE + 256 basic bytes + <|endoftoken|>
    n_layer: int = 12 # number of blocks i guess
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # series of blocks
            ln_f = nn.LayerNorm(config.n_embd) # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)  # output layer to softmax tokens
    
    def forward(self, idx):
        B, T = idx.size() # gives us the batch and position in the context window this token is
        
        assert T <= self.config.block_size, f"Cannot forward sequence of this size"
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device) # all the positional embeddings up to the token
        pos_emb = self.transformer.wpe(pos) 
        tok_emb = self.transformer.wte(idx) # accesses the token's literal embedding
        x = pos_emb + tok_emb
        
        for layer in self.transformer.h: # run through the blocks
            x = layer(x)
        x = self.transformer.ln_f(x) # final layer norm
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        return logits
        