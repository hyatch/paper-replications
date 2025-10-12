import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import math
from typing import Optional, Tuple, List
from siglip import SiglipModel, SiglipVisionConfig


class GemmaConfig():

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads, 
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id



class PaliGemmaConfig():

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class KVCache():
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # [Batch, heads, sequence length, dims]
            return self.key_cache[0].shape[-2]
    
    def update(self,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # initialized if not already initialized
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # concatenate the new keys with the cached ones
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim = -2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim = -2)
            
        # return the KV values for the layer
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias = True)
    
    def forward(self, image_features):
        # [Batch, Path, Embed_dim] -> [Batch, Patch, Proj_dim (same as the text tokens)]
        hidden_states = self.linear(image_features)
        return hidden_states

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embedding = 2048, base = 10000, device = None):
        super().__init__()
        
        self.dim = dim # head dimensions
        self.max_position_embeddings = max_position_embedding
        self.base = base
    
        # calculation for theta given by theta_i = base ^ (2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent = False)
    
    @torch.no_grad()
    def forward(self, x, position_ids, seq_len = None):
        # x: [batch, heads, sequence, dims]
        self.inv_freq.to(x.device)
        
        # [Batch, Head_Dim //2 , 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # [Batch, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        
        device_type = x.device.type
        device_type = device_type if isinstance(device_type,str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type = device_type, enabled = False):
            # We multiply theta via dot product with the position of the vector
            # [Batch, Head_Dim // 2, 1] @ [Batch, 1, Seq_Len] -> [Batch, Seq_Len, Head_Dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1,2)
            
            # emb: [Batch, Seq_Len, Head_dim]
            emb = torch.cat((freqs, freqs), dim = -1)
            # cos, sin [Batch, Seq_Len, Head_dim]
            cos = emb.cos()
            sin = emb.sin()
        
        return cos.to(dtype = x.dtype), sin.to(dtype = x.dtype) 
        

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1):
    cos = cos.unsqueeze(unsqueeze_dim) # adds the head dimension
    sin = sin.unsqueeze(unsqueeze_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    # Simulating the [-x2, x1, -x4, x3 ...] shape in RoPE paper
    x1 = x[..., : x.shape[-1] // 2] # first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # second half of the last dimension
    return torch.cat((-x2, x1), dim = -1)

def repeat_kv(hidden_states, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    # duplication of the heads
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


# THIS IS WHERE IT KINDA COMES TOGETHER
class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0            

        # [1024, 1024]
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        
        # [1024, 512] (only one KV head per 2 Q heads)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        
        # [1024, 1024]
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embedding=self.max_position_embeddings,
            base=self.rope_theta,
        )
    
    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        # [Batch, SeqLen, NumHeads * Headdim]
        query_states = self.q_proj(hidden_states)
        
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        
        # rotary embedding
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len = None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # append the most recent token generation
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)
        
        # reverses GQA due to hardware limitations
        # maps a head to every query
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype = torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p = self.attention_dropout, training=self.training)
        
        attn_outputs = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_outputs.view(bsz, q_len, -1)
        
        # shuffling
        attn_output = self.o_proj(attn_output)
        
        return attn_output
            
class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias = False)
        self.proj = nn.Linear(config.hidden_size, config.intermediate_size, bias = False)
        self.unproj = nn.Linear(config.intermediate_size, config.hidden_size, bias = False)
        
    def forward(self, x):
        # y = self.gate_proj(x) [Batch, Sequence, Inter]
        # y = nn.functional.gelu(y, approximate = "tanh") [Batch, Seq, Inter]
        # j = self.proj(x) [Batch, Seq, Inter]
        # z = y * j [Batch, Seq, Inter]
        # z = self.unproj(z) [Batch, Seq, Hidden]
        # return z
        return self.unproj(nn.functional.gelu(self.gate_proj(x), approximate = "tanh")*self.proj(x))
    
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GemmaAttention(config = config, layer_idx = layer_idx)
        
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        # [Batch, Seq_Len, Hidden_dim]
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            kv_cache = kv_cache
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers) 
])
        
        self.norm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        # [Batch, Seq_Len, Hidden_size]
        hidden_states = inputs_embeds
        # [Batch, Seq_Len, Hidden_size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype = hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask = attention_mask,
                position_ids = position_ids,
                kv_cache = kv_cache
            )
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
    
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # dimension of the token embeddings 
        self.weight = nn.Parameter(torch.zeros(dim))
    
    def _norm(self, x):
        # normalization across the embeddings
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps) # 1 / sqrt(...)
    
    def forward(self, x):
        output = self._norm(x.float())
        # LlaMa model prefer x.to() * w, Gemma prefers (x*w).to()
        output = output * (1.0 +self.weight.float())
        return output.type_as(x)

class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data
    

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipModel(config.vision_config)
        self.multi_model_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        
        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model
        
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        
    # matches the parameters from the embedding to the transofrmation of the final linear layer
    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds, input_ids, attention_mask, kv_cache
    ):
        _,_, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
        
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype = dtype, device = device)
        # True for Text Tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id
        
        # For example [111,111,111,111,1,12,43,21,42,2]
        # where 111 is <image> token, 1 is <bos>, 2 is \n, and all else are text
        # textmask would give [0,0,0,0,1,1,1,1,1,1]
        # pad would give all [0...]
        # image would give [1,1,1,1,0,0,0,0,0,0,0]
        
        
        # expanding out masks to fit the original shape [batch, seq_len, dim]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1,-1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1,-1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1,-1, embed_dim)
        
        # add the text embeddings via the mask 
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        
        # add the image embeddings via the mask, torch.where cannot be used because of shape mismatch in image
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        # Attention Mask Creation
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
        
        if kv_cache is None or kv_cache.num_items() == 0:
            # no masking in prefilling except for padding
            # no padding in this implementation
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value = 0, dtype=dtype, device=device
            )
        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value = 0, dtype=dtype, device=device
            )
            
        # extracts a head dimension
        # [Batch, Q_Len, KV_Len] -> [Batch, Q_Head, Q_HeadLen, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)
            
        if kv_cache is not None and kv_cache.num_items() > 0:
            # grab the query token (last position)
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
                
        else:
            # create position ids based on the size of the attention mask
            # Use the 1 token for masked tokens
            position_ids = (attention_mask.cumsum(-1)).masked_fill((attention_mask == 0), 1).to(device)
        
        return final_embedding, causal_mask, position_ids
        
        
    def forward(self,
                input_ids: torch.LongTensor = None,
                pixel_values: torch.FloatTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None,
                ) -> Tuple:
        assert torch.all(attention_mask == 1), "Padding is not supported in this custom implementation"
        
        # get the input embeddings
        # (Batch, Seq_Len, Hidden Size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # Batch, Channels, Height, Width -> Batch, Patch, Embed Dim
        selected_image_feature = self.vision_tower(pixel_values.to(input_embeds.dtype))
        # Batch, Patch, Embed Dim -> Batch, Patch, Hidden Size (matches the shape of text)
        image_features = self.multi_model_projector(selected_image_feature)
        
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)
        
        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            kv_cache = kv_cache
        )
        
        return outputs