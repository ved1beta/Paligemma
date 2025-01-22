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

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels, 
            out_channels=self.embed_dim, 
            kernel_size=self.patch_size, 
            padding="vaild"
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)  
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    
    def forward(self, input_embeds : torch.Tensor)-> torch.Tensor:
        hidden_states = input_embeds
        for encoder_layer in self.layer:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states

class SiglipMLP(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = (config.hidden_size, config.intermediate_size)
        self.fc2 = (config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states : torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)

        hidden_states = nn.functional.gelu(hidden_states , approximate="tanh")

class Attention(nn.Module):
    def __init__(self, config : SiglipVisionConfig) :
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim//self.num_heads
        self.scale = self.head_dim**-0.5 # eqv 1/ root(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim , self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim , self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim , self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim , self.embed_dim)

    def forward(self, 
                hidden_state :torch.Tensor,)-> Tuple[torch.Tensor , Optional[torch.Tensor]]:
        
        batch_size , seq_length, _ = hidden_state.size()
        query_states = self.q_proj(hidden_state) 
        key_states = self.k_proj(hidden_state) 
        value_states = self.v_proj(hidden_state) 

        query_states = query_states.view(batch_size, seq_length , self.num_heads , self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_length , self.num_heads , self.head_dim).transpose(1,2)
        valur_states = value_states.view(batch_size, seq_length , self.num_heads , self.head_dim).transpose(1,2)
        
        attn_weight = (torch.matmul(query_states , key_states).transpose(2,3) * self.scale)

        if attn_weight.size() != (batch_size, self.num_heads , seq_length , seq_length):
            raise ValueError(
                f"Attention weights size {(batch_size, self.num_heads , seq_length , seq_length)} , is "
                f"{(attn_weight.size())}"
            )
        attn_weight = nn.functional.softmax(attn_weight , dim = -1 , dtype=torch.float32).to(query_states.dtype)
        attn_weight = nn.functional.dropout(attn_weight , p=self.dropout , training=self.training )

        attn_output = torch.matmul(attn_weight , value_states)

        if attn_output.size() != (batch_size, self.num_heads , seq_length, self.head_dim):
            raise ValueError(
                f"Attention weights size {(batch_size, self.num_heads , seq_length , self.head_dim)} , is "
                f"{(attn_output.size())}"
            )
        attn_output = attn_output.transpose(1,2).contiguous()

        attn_output = attn_output.reshape(batch_size , seq_length , self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weight


class SiglipEncoderLayer(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim , esp= config.layer_norm_eps )

    def forward(self
                ,hidden_state :torch.Tensor
                ) -> torch.Tensor:
        residual = hidden_state

        hidden_states = self.layer_norm(hidden_state)

        hidden_states , _ = self.self_attn(hidden_states=hidden_states )

        hidden_states = residual + hidden_states

        residual = hidden_states 

        hidden_states = self.layer_norm2(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config =config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values)-> Tuple:
        return self.vision_model(pixel_values = pixel_values)
