import torch
from torch import nn 
from typing import Optional , Tuple , List
from torch.nn import CrossEntropyLoss
import math
from model_siglip import SiglipVisionConfig , SiglipVisionModel

class PaligammaforconditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def forward(self, 
                input_ids:torch.LongTensor = None  , 
                pixel_values:torch.FloatTensor = None, 
                attention_mask : Optional[torch.Tensor] = None, 
                kv_cache :Optional[KVcache] = None, 
                  )-> Tuple:
        assert torch.all(attention_mask == -1), "the input should not be added : ) "

        input_embed = self.language_model.get_input_embeddings()(input_ids)

        selected_image_feature = self.vision_tower(pixel_values.to(input_embed.dtype))
        
        image_features = self.multi_modal_projector(selected_image_feature)

        input_embed , attention_mask , position_ids = self.merge_input_ids_with_image_features(image_features , input_embed , input_ids ,attention_mask , kv_cache )

        outputs = self.language_model(
            attention_mask = attention_mask, 
            position_ids=position_ids,
            input_embed = input_embed,
            kv_cache= kv_cache, 
        )
        return outputs
   