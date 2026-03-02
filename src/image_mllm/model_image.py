import torch
import torch.nn as nn
from src.image_mllm.vision_image import ImageViT
from src.image_mllm.language_image import ImageGPT, ImageLLMConfig

class ImageProjector(nn.Module):
    def __init__(self, input_dim, llm_dim, mlp_depth=2):
        super().__init__()
        layers = [nn.Linear(input_dim, llm_dim), nn.GELU()]
        for _ in range(mlp_depth - 1):
            layers.append(nn.Linear(llm_dim, llm_dim))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)

class ImageMLLM(nn.Module):
    def __init__(
        self,
        vocab_size=5000,
        vision_dim=512,
        llm_dim=512,
        img_size=64,
        patch_size=16
    ):
        super().__init__()
        
        self.vision_encoder = ImageViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vision_dim
        )
        
        self.projector = ImageProjector(
            input_dim=vision_dim,
            llm_dim=llm_dim,
            mlp_depth=2
        )
        
        config = ImageLLMConfig(
            vocab_size=vocab_size,
            n_embd=llm_dim,
            n_head=8,
            n_layer=8,
            max_len=512
        )
        self.llm = ImageGPT(config)
        self.llm_dim = llm_dim
        
    def forward(self, image, input_ids, targets=None, numeric_mask=None, numeric_loss_lambda=1.0):
        vis_embeds = self.vision_encoder(image) 
        proj_embeds = self.projector(vis_embeds) 
        
        logits, loss = self.llm(
            input_ids=input_ids,
            vision_embeds=proj_embeds,
            targets=targets,
            numeric_mask=numeric_mask,
            numeric_loss_lambda=numeric_loss_lambda
        )
        
        return logits, loss
        
    @torch.no_grad()
    def generate(self, image, input_ids, max_new_tokens=40, temperature=1.0):
        self.eval()
        vis_embeds = self.vision_encoder(image)
        proj_embeds = self.projector(vis_embeds)
        
        generated_ids = self.llm.generate(
            input_ids=input_ids,
            vision_embeds=proj_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        return generated_ids
