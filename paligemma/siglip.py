from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
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
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

# processes the image embeddings
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # accesses the actual values inside the patch
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicates no padding is added
        )

        # solves for the number of patches needed to convert the entire image (224 / 16)^2 across rows and columns
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        self.num_positions = self.num_patches
        
        # positional embedding based on patch placement
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        
        # a buffer mask to access all the positions 
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        
        # send the pixel values to a 2D CNN to read pixel values
        # the CNN reads each pixel value once since stride and kernel width are equal
        # Based on this, we will now have [Batch, Embed Dim, H_patches, W_patches]
        patch_embeds = self.patch_embedding(pixel_values)
          
        # flatten the h_patch dimension [Batch, Embed Dim, Total Patches]
        embeddings = patch_embeds.flatten(2)
        
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        # this reinforces an idea that each patch has this embedding that describes it
        embeddings = embeddings.transpose(1, 2)
        
        # Add position embeddings to each patch straight into the embedding dimension
        # the buffer unpacks the positional embedding table to add to the corresponding patches
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # standard MHA !!!
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # seq_len is interchangeable with the number of patches
        batch_size, seq_len, _ = hidden_states.size()
        
        # projection layers to create independent QKV vectors 
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # we reshape the QKV vectors [Batch, Patches, Embed Dim] -> [Batch, N_Heads, Patches, Head_Dim]
        # note that embed dim // heads must be an integer so that we have complete layers
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # attention formula (QK^T / sqrt(head dim)) where multiply across the head_dim dimension
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)
        
        # sanity check that enforces proper dimensions and divisble heads
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # bfloat16 is accurate enough and faster, also my cuda can't tolerate torch.float32
        # stabilize the attention weights with a softmax for a probability distribution
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.bfloat16).to(query_states.dtype)
        # useful in training to learn new connections, but this model is purely inference so this layer is optional
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # finish the formula by multiplying with value states which gets us back to the shape [Batch, Heads, Patch, Head Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        # another sanity check
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
            
        # map back to the original shape while also concatenating the attention weights
        # [Batch, Patch, Embed Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # shuffles the weights in this projection layer
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        # The attention paper suggests for the intermediate dim to be 4x embed dim
        hidden_states = self.fc1(hidden_states)
        # gelu non-linearity to draw meaningful connections
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # map back down to [Batch, Patch, Embed Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Overall encoding architecture
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # residual for the skip connection
        residual = hidden_states
        # pass to a standard layer normalization for stability
        hidden_states = self.layer_norm1(hidden_states)
        # pass to a self-attention layer 
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # skip connection
        hidden_states = residual + hidden_states
        # make another residual
        residual = hidden_states
        # another layer norm
        hidden_states = self.layer_norm2(hidden_states)
        # multi-layer perception
        hidden_states = self.mlp(hidden_states)
        # skip connection
        hidden_states = residual + hidden_states
    
        # shape is left unchanged overall [Batch, Num Patches, Embed Dim]
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        
        # series of encoder layers
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    # Ignore copy
    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # recall inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        # pass the embeddings into recurrent encoder layers
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        # pass our processed embeddings to the encoder
        # we get a well-encoded shape of [Batch Size, Num Patches, Embedding Dim]
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        # final layer normalization
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed Dim]
        # send it back to the gemma.py file
        return self.vision_model(pixel_values=pixel_values) 