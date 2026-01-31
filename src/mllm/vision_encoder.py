import torch
import torch.nn as nn
import math

class PatchEmbed3D(nn.Module):
    """
    Splits video into patches and embeds them.
    Input: (B, C, T, H, W)
    Output: (B, Num_Patches, Dim)
    """
    def __init__(self, img_size=64, patch_size=16, in_chans=3, embed_dim=256, num_frames=24):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size * num_frames
        
        # We use Conv3d to handle the patch embedding.
        # Kernel: (1, patch, patch) -> Each frame is patched independently spatially.
        # Stride: (1, patch, patch)
        # This keeps T dimension separate initially, maintaining temporal order.
        self.proj = nn.Conv3d(
            in_chans, 
            embed_dim, 
            kernel_size=(1, patch_size, patch_size), 
            stride=(1, patch_size, patch_size)
        )
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x) 
        # Output: (B, Dim, T, H//P, W//P) -> (B, 256, 24, 4, 4)
        
        # Flatten spatio-temporal dimensions
        x = x.flatten(2).transpose(1, 2) # (B, T*H'*W', Dim) -> (B, 384, 256)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        return x + self.pos_embed

class PhysicsViT(nn.Module):
    def __init__(
        self, 
        img_size=64, 
        patch_size=16, 
        in_chans=3, 
        num_frames=24, 
        embed_dim=256, 
        depth=6, 
        num_heads=8, 
        mlp_ratio=4., 
        drop_rate=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Patch Embedding
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim, num_frames)
        num_patches = self.patch_embed.num_patches
        
        # Positional Embedding (Spatio-Temporal)
        self.pos_embed = PositionalEncoding(num_patches, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=drop_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-Norm is standard for ViT
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Final Norm
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
        """
        x: (B, C, T, H, W)
        Returns: (B, Seq_Len, Dim) -> (B, 384, 256)
        """
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.pos_drop(x)
        
        x = self.blocks(x)
        x = self.norm(x)
        
        return x

if __name__ == "__main__":
    # Quick Test
    model = PhysicsViT()
    dummy = torch.randn(2, 3, 24, 64, 64)
    out = model(dummy)
    print(f"Input: {dummy.shape}")
    print(f"Output: {out.shape}")
    
    # Param Count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params/1e6:.2f}M")
