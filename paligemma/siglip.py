import configparser
from typing import Optional, Tuple
import torch
import torch.nn as nn

# hyperparameter definitions
class SiglipVisionConfig:
    
    def __init__(
        self,
        hidden_size = 768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        
        # Core transformer parameters
        self.hidden_size = hidden_size  # embed dimension
        self.intermediate_size = intermediate_size  # MLP scale up (x4)
        self.num_hidden_layers = num_hidden_layers 
        self.num_attention_heads = num_attention_heads
        
        # Vision parameters
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_image_tokens = num_image_tokens or (image_size // patch_size) ** 2
        
        # Regularization parameters
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout

# overall transformer model
class SiglipTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SiglipEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)
    
        # layernorm resolves covariate shift which is the variance in magnitudes
        # across different input vectors, which destabilizes the network
        # The gradients will be too variate.
        
        # Layernorm achieves this by finding the mean and STD of each input 
        # embeddings of a (list of tokens or convs of an image)
        # and normalizing them
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        
        # passes to the embeddings class
        hidden_states = self.embeddings(pixel_values)
        
        # passes the embeddings to the encoder
        last_hidden_state = self.encoder(hidden_states)
        
        # normalizes after encoding so that the gradients are not too variate
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        return last_hidden_state

class SiglipEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # embedding dimension
        self.embed_dim = config.hidden_size
        # H x W of the image
        self.image_size = config.image_size
        # size of the patches/conv filter
        self.patch_size = config.patch_size
        
        # RGB processing to embedding
        # each pixel is seen once as in no overlap in patches
        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            padding = 0,
        )
        
        # defines the number of patches in the overall image
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        
        # embedding table of different patches x embeddings
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # position ids are the indices of the patches in the overall image
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1,-1)),
            persistent = False,
        )
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # B, C, H, W
        
        patch_embeds = self.patch_embedding(pixel_values) # B, Embed, H_Patches, W_Patches
        
        embeddings = patch_embeds.flatten(2) # Flatten H_Patches
        # Now at shape B, Embed, Num_Patches
        
        embeddings = embeddings.transpose(1,2) # B, NumPatches, Embed
        
        embeddings = embeddings + self.position_embedding(self.position_ids) # add the relevant positional embeddings to our patches
        
        return embeddings #B, N_Patches, Embed

# multi-layer transformer encoder
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states

# single layer of the encoder
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # we start with a shape of [batch, seq_len, embed_dim]
        residual = hidden_states # for the skip connection
        
        # common architecture is now layer norm, self-attention, and skip connection
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states = hidden_states)
        
        hidden_states = residual + hidden_states
        # for the 2nd skip connection
        residual = hidden_states
        
        # layer norm into MLP into skip connection
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return hidden_states
    
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states) # scale up the dimensions
        hidden_states = nn.functional.gelu(hidden_states, approximate = "tanh") # GeLU non-linearity
        hidden_states = self.fc2(hidden_states) # scale down the dimensions
        return hidden_states

class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads # must be divisible
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # accepts states after the layernorm
        # the shape is [batch, patch, dim]
        batch_size, seq_length, _ = hidden_states.size()
        
        # projects the hidden states to q, k, v via linear projections
        q_states = self.q_proj(hidden_states)
        k_states = self.k_proj(hidden_states)
        v_states = self.v_proj(hidden_states)
        
        
        # [Batch, Heads, Patches, HeadDim]
        # we transpose to enforce an idea that each head works with a number of patches with this split dimension
        # this intialization is actually a slowdown compared to the fused attention commonly used in GPT
        q_states = q_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        k_states = k_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        v_states = v_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
    
        # tells us which tokens correspond to others, this is the attention formula
        attn_weights = (torch.matmul(q_states, k_states.transpose(2,3)) * self.scale)
        
        # Check if attention weights have expected shape [batch, num_heads, seq_len, seq_len]
        expected_shape = (batch_size, self.num_heads, seq_length, seq_length)
        if attn_weights.shape != expected_shape:
            raise ValueError(f"Attention weights shape {attn_weights.shape} does not match expected shape {expected_shape}")


        # applying the softmax, a causal mask would be applied if we were in a decoder 
        attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float32).to(q_states.dtype)
        
        # dropout layer for fun
        attn_dropout = nn.functional.dropout(attn_weights, p = self.dropout, training = self.training)
        
        # brings us to shape [batch, num_heads, seq_len, head_dim]
        attn_output = (torch.matmul(attn_dropout, v_states))
        
        # [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1,2).contiguous()
        
        # this acts as the concatenation of the heads along the head dimension
        attn_output = attn_output.reshape(batch_size, seq_length, self.embed_dim)
        
        # shuffling of the heads in a linear projection
        output = self.out_proj(attn_output)
        return output

# overall model
class SiglipModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipTransformer(config)
    
    def forward(self, pixel_values):
        return self.vision_model(pixel_values=pixel_values)
        