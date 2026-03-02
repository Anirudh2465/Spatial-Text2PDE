import torch
import torch.nn as nn

class PatchEmbed2D(nn.Module):
    """
    Splits an image into patches and embeds them.
    Input: (B, C, H, W)
    Output: (B, Num_Patches, Dim)
    """
    def __init__(self, img_size=64, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        x = self.proj(x) 
        x = x.flatten(2).transpose(1, 2)
        return x

class PositionalEncoding2D(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        return x + self.pos_embed

class ImageViT(nn.Module):
    def __init__(
        self, 
        img_size=64, 
        patch_size=16, 
        in_chans=3, 
        embed_dim=256, 
        depth=6, 
        num_heads=8, 
        mlp_ratio=4., 
        drop_rate=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed2D(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = PositionalEncoding2D(num_patches, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=drop_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.pos_drop(x)
        
        x = self.blocks(x)
        x = self.norm(x)
        
        return x
