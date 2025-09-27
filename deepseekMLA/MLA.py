import torch
import torch.nn as nn
from torch.nn import functional as F
import time

#define hyperparameters
batch_size = 64 # num parallel sequences
block_size = 128 # context length
max_iters = 5000  
eval_intervals = 500
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # access the gpu
eval_iters = 200
n_embd = 256
n_heads = 8
n_blocks = 6
dropout = 0.2

torch.manual_seed(1332)

#load the dataset (tiny shakespeare)
with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) # list all unique characters in the dataset
vocab_size = len(chars) # number of unique characters
stoi = {ch:s for s,ch in enumerate(chars)} # hash map of a character to an index
itos = {s:ch for s,ch in enumerate(chars)} # map from index back to character
encode = lambda s: [stoi[c] for c in s] # hash a string into a list of indexes
decode = lambda l: ''.join(itos[i] for i in l) # hash a list of indexes into a string

data = torch.tensor(encode(text), dtype = torch.long) # encode the dataset
# train-test splits
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#splits data into batches
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 1D tensor of batch_num indexes
    x = torch.stack([data[i:i+block_size] for i in ix]) # stacks batches of indexes to predict on (32, 8)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # stacks batches of indexes to predict (32, 8)
    x, y = x.to(device), y.to(device) # send to gpu
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # sets evaluation mode (more consistent results)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) 
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss, kv_cache = model(X,Y) # computes the loss from the model
            losses[k] = loss.item()
        out[split] = losses.mean() 
    model.train() # reverts to training mode
    return out

class Head(nn.Module):
    """A self-attention head that supports KV Caching"""
    
    def __init__(self, head_size):
        super().__init__()
        
        head_size_r = head_size // 4
        self.head_size_r = head_size_r
        self.head_size_c = head_size - head_size_r
        
        self.q_compress = nn.Linear(n_embd, n_embd, bias = False)
        self.kv_compress = nn.Linear(n_embd, n_embd, bias = False)
        
        self.q_content_proj = nn.Linear(n_embd, self.head_size_c, bias = False)
        self.q_rotary_proj = nn.Linear(n_embd, self.head_size_r, bias = False)
          
        self.k_content_proj = nn.Linear(n_embd, self.head_size_c, bias = False)
        self.k_rotary_proj = nn.Linear(n_embd, self.head_size_r, bias = False)
        
        self.v_proj = nn.Linear(n_embd, head_size, bias = False) 
                
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('causal_mask', torch.tril(torch.ones(block_size, block_size, dtype = torch.bool)))
        
    def forward(self, x, cached = None, start_pos = 0):
        B,T,C = x.shape 
        # initialize QKV vectors
        c_q = self.q_compress(x)
        c_kv = self.kv_compress(x)
        
        q_c = self.q_content_proj(c_q) # (B, nh, T, hs)
        q_prer = self.q_rotary_proj(c_q)
        q_r = self.rope(q_prer, start_pos = start_pos)
        q = torch.cat((q_c, q_r), dim = -1)
        
        k_c = self.k_content_proj(c_kv)
        k_prer = self.k_rotary_proj(c_kv)
        k_r = self.rope(k_prer, start_pos = start_pos)
        k = torch.cat((k_c, k_r), dim = -1)
        
        v = self.v_proj(c_kv)        

        if cached is not None:
            cached_k, cached_v = cached
            k = torch.cat((cached_k, k), dim = 1) # concatenate along the time dimension
            v = torch.cat((cached_v, v), dim = 1)
        
        full_k = k
        full_v = v

        wei = q @ full_k.transpose(-2,-1) * (full_k.size(-1)**-0.5)
        T_q, T_k = q.size(1), k.size(1)
        
        
        if cached is not None and T_q == 1:
            pass
        else:
            # Handle sequences longer than registered buffer
            mask = torch.tril(torch.ones(T_q, T_k, device=wei.device, dtype=torch.bool))
            wei = wei.masked_fill(~mask, float('-inf'))
        
        wei = F.softmax(wei, dim=-1) # applies an average to each row of weights
        wei = self.dropout(wei) # dropout layer
        out = wei @ full_v # important QK connections get a strong V applied
        return out, (full_k, full_v) # return the key and value for caching
    
    def rope(self, x, start_pos = 0):
        B, T, head_dim_r = x.size()
        assert head_dim_r % 2 == 0, "Head dimension must be even for RoPE."

        half_dim = head_dim_r // 2
        theta = 1.0 / (10000 ** (torch.arange(0, half_dim, device=x.device) / half_dim))  # [half_dim]
        seq_idx = torch.arange(start_pos, start_pos+T, device=x.device)  # [T]
        idx_theta = torch.outer(seq_idx, theta)  # [T, half_dim]

        sin = torch.sin(idx_theta) 
        cos = torch.cos(idx_theta)  

        x_split = x.view(B, T, half_dim, 2)  # [B*n_head, T, half_dim, 2]
        x1, x2 = x_split[..., 0], x_split[..., 1]  # [B*n_head, T, half_dim]
        x_rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim = -1)
        return x_rotated.view(B, T, head_dim_r)

class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.proj = nn.Linear(n_embd, n_embd) 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, cached_kv = None, start_pos = 0):
        if cached_kv is None:
            cached_kv = [None] * len(self.heads)
        
        head_outs = []
        new_kv = []
        for i, h in enumerate(self.heads):
            out, kv = h(x, cached = cached_kv[i], start_pos = start_pos)
            head_outs.append(out)
            new_kv.append(kv)
        
        out = torch.cat(head_outs, dim = -1) # concatenate along the embedding dimension
        out = self.proj(out) # linear layer to mix the heads
        out = self.dropout(out)
        return out, new_kv
            
            
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n = n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # scale up the embedding
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd), # scale back down to the embedding dimension
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

# complete transformer block with layer norms
class Block(nn.Module):
    
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    # residual/skip connections
    def forward(self, x, cached_kv = None, start_pos = 0):
        sa_out, new_kv = self.sa(self.ln1(x), cached_kv, start_pos = start_pos)
        x = x + sa_out
        x = x + self.ff(self.ln2(x))
        return x, new_kv


# overall calling model 
class Transformer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # token embedding table to map into a high-dimensional space
        self.blocks = nn.ModuleList([Block(n_embd, n_heads) for _ in range(n_blocks)]) # stacks multiple blocks
        self.ln = nn.LayerNorm(n_embd) # final layer norm
        self.lm = nn.Linear(n_embd, vocab_size) # maps back to the number of characters
        
        
    # moves the model forward by returning the logits from the embedding table
    def forward(self, idx, targets=None, kv_cache=None, start_pos = 0):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        x = tok_emb

        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, kv = block(x, layer_cache, start_pos = start_pos)
            new_kv_cache.append(kv)

        x = self.ln(x)
        logits = self.lm(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, new_kv_cache
    
    #samples the model
    def generate(self, idx, max_new_tokens):
        
        self.eval()
        kv_cache = [None] * len(self.blocks)  # initialize kv cache for each block
        current_pos = idx.size(1)
        for _ in range(max_new_tokens):
            # crops idx to the last block_size tokens
            idx_cond = idx[:, -1:]
            logits, loss, kv_cache = self(idx_cond, kv_cache = kv_cache, start_pos = current_pos-1)  # calls the forward function to get logits
            logits = logits[:, -1, :] # accesses the last character in the context (B,C)
            probs = F.softmax(logits, dim = -1) # probabilities of the logits
            idx_next = torch.multinomial(probs, num_samples = 1) # selects a character according to the probability
            idx = torch.cat((idx, idx_next), dim = 1) # appends the new character to the context window
            current_pos += 1
        self.train()
        return idx


model = Transformer()
m = model.to(device)

#adam optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

# training with gradient descent
for iter in range(max_iters):
    if iter % eval_intervals == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss = {losses['val']:.4f}")

    xb, yb = get_batch('train')
        
    logits, loss, kv_cache = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

#generation 
context = torch.zeros((1,1), dtype = torch.long, device = device) # initializes the context window as the new line character
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))