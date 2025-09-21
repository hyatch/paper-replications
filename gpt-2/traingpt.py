import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import math

# attention block with QKV and masking to prevent looking into the future
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # for QKV projections for all heads, we 3x our embedding
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.shape # batch, time, channels    
        qkv = self.c_attn(x)
        
        q, k, v = qkv.split(self.n_embd, dim = 2) # split the neurons of the attention layer into the qkv vectors
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, n_head, T, C -> n_head is like a second batch dimension!
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # apply attention formula
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # apply the mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf")) # masks out the future tokens with -inf
        
        att = F.softmax(att, dim = -1) # softmax over the channels to get probabilities and zero out future

        # use the value vector for output
        y = att @ v
        
        #FlashAttention
        # y = F.scaled_dot_product_attention(q, k, v, iscausal=True)
        # Basically allows us to more efficietly compute attention because we don't have to store the entire NxN matrix. We can calculate softmax in tiles that break up the context. 
        
        y = y.transpose(1,2).contiguous().view(B,T,C) # return to initial shape
        
        # output projection
        y = self.c_proj(y)
        
        return y
        
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd*4)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(config.n_embd*4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        x = self.c_fc(x) # maps up 4x
        x = self.gelu(x) # approximate gelu activation
        x= self.c_proj(x) # maps down 4x
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # residual connections with layer norm
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # context length
    vocab_size: int = 50257 # 50,000 BPE + 256 basic bytes + <|endoftoken|>
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
    
        # weight sharing tech that reduces total parameters
        self.transformer.wte.weight = self.lm_head.weight
        
        # initializes weights
        self.apply(self._init_weights)


    # basically we initial our weights according to a normal distribution with std 0.02, but if the layer has the NANOGPT_SCALE_INIT attribute (which is set for certain layers above), we scale it down by sqrt(2*num_layers)
    # this is supposed to help with training stability in deep transformers as proposed in the GPT-2 paper
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
    
    def forward(self, idx, targets=None):
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
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #reshape to fit cross_entropy input needs
        
        
        return logits, loss
    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        # loads all the parameters 
        parameters = {pn: p for pn, p in self.named_parameters()}
        # loads all the parameters that require gradients
        parameters = {pn:p for pn, p in parameters.items() if p.requires_grad}
        
        # separate out all the parameters that will and wont be decayed
        decay_params = [p for n, p in parameters.items() if p.ndim >= 2]
        # these are the parameters that will not be decayed (typically biases and layernorm weights)
        nodecay_params = [p for n, p in parameters.items() if p.ndim < 2]
        
        # sets the weight decay for our two groups
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        
        # checks for a faster fused kernel in PyTorch's AdamW implementation
        import inspect
        fused_avail = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avail and "cuda" in device
        
        # creates the desired AdamW optimizer with proper weight decay and fused kernel if possible
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
        
# access the gpu
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
import os

ddp = int(os.environ.get("RANK", -1)) != -1 # asks if we are running ddp
if ddp:
    assert torch.cuda.is_available() #"DDP requires CUDA"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"]) # the global rank of the GPU we are using
    ddp_local_rank = int(os.environ["LOCAL_RANK"]) # the local rank of the GPU we are using (of the node)
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device =  f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    
else:
    # if not ddp, we are running on a single GPU or CPU
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # and then we can initialize a single GPU or CPU as before

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("using "+device)
    
torch.manual_seed(1337)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
    
total_batch_size = 524288 # 2^19 ~500k tokens per batch, as used in the GPT-2 paper
B = 4
T = 1024
assert total_batch_size % (B*T*ddp_world_size) == 0
gradient_accumulation_steps = total_batch_size // (B*T) # if on ddp, this also divides by the number of gpus
if master_process:
    print(f"running with {ddp_world_size} processes")
    print(f"batch size {B}, context size {T}, gradient accumulation steps {gradient_accumulation_steps}, total batch size {B*T*gradient_accumulation_steps}")

import tiktoken

# loads data from Tiny Shakespeare dataset
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B # batch size
        self.T = T # context window size
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        with open("input.txt", "r", encoding = "utf-8") as f:  
            text = f.read()
        enc = tiktoken.get_encoding("gpt2") # gets the GPT-2 BPE tokenizer
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype = torch.long)
        print(f"loaded {self.tokens.size(0)} tokens")
        
        self.current_position = self.process_rank * self.B * self.T # each process starts at a different part of the data
    
    def next_batch(self):
        B, T = self.B, self.T
        chunk = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = chunk[:-1].view(B, T) # input is all but last token
        y = chunk[1:].view(B, T)  # target is the next token
        
        self.current_position += self.num_processes * B * T # move forward in the data by num_processes * B * T tokens
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_position = self.process_rank * B * T # if we reach the end of the data, start over
    

# can alternatively load Hugging Face weights for GPT-2 from the transformers library
model = GPT(GPTConfig(vocab_size = 50304)) # nicely divisble by powers of 2 which quickens the run time on GPU kernels
model.to(device)
# model = torch.compile(model) # should be used for faster training in PyTorch 2.0 but my GPU is too old
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model # unwrap DDP model for logging if needed

# implements the learning rate schedule from the GPT-2 paper
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_iters = 10
max_iters = 50
def get_lr(it):
    if it < warmup_iters:
        return max_lr * it / warmup_iters # linear warmup
    if it > max_iters:
        return min_lr # reached the minimum lr after reading enough tokens from multiple iterations
    
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # cosine decay
    return min_lr + coeff * (max_lr - min_lr)



train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")



# sets the matmul from FP32 to TF32 for faster training on modern GPUs (i dont have one so i cant test it)
# reduces the mantissa bits from 23 to 10, so it is less precise but faster
# torch.set_float_32_matmul_precision('high')

# bugfixed AdamW optimizer from PyTorch
optimizer = model.configure_optimizer(weight_decay=1e-1, learning_rate=max_lr, device=device)
# sets to training mode
for i in range(50): # arbitrary number of iterations
    # validation loss
    
    
    
    # training step
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(gradient_accumulation_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        #with torch.autocast(device_type=device, dtype=torch.bfloat16): # reduces the precision of certain matmuls and convolutions to BF16 from FP32 for faster training on modern GPUs (i dont have one so i cant test it)
        logits, loss = model(x, y)
        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        loss_accum += loss.detach()
        loss.backward() # scale the loss to account for gradient accumulation
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # average the loss across all processes for logging
    lr = get_lr(micro_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping at 1.0 to prevent gradient explosions
    optimizer.step()
    print(f"step {i} | loss {loss_accum.item()} | norm {norm:.4f}")


import sys; sys.exit(0)

# set to eval mode for inference
model.eval()
max_length = 30
num_return_sequences = 5

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim = -1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim = 1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">" + decoded)
