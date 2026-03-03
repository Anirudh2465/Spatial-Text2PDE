"""
vision_video.py
===============
VideoViT: Vision Transformer for spatio-temporal video input.
Input  : (B, C, T, H, W)
Output : (B, T * H' * W', embed_dim)    where H' = H // patch_size, W' = W // patch_size

Architecture
------------
1. PatchEmbed3D  — Conv3d with kernel (1, P, P), stride (1, P, P)
                   Each frame is independently patched; temporal stride = 1
                   so we preserve every frame's contribution.
2. Factorised Positional Encoding — separate learnable spatial + temporal embeddings
                   added together, giving the model the ability to reason about
                   both where (spatial) and when (temporal) a patch appeared.
3. TransformerEncoder — standard Pre-Norm ViT blocks with GELU.
4. LayerNorm on output.

Why factorised positional encoding?
  A flat (T*H'*W') positional table would have 384 entries for (24, 4, 4) and
  would be hard to generalise. Factorised spatial + temporal embeddings are
  far cheaper and transfer better within a temporal range.
"""

import math
import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """
    Embed a video into patch tokens.
    Input:  (B, C, T, H, W)
    Output: (B, T * Hs * Ws, embed_dim)   where Hs = H // P, Ws = W // P
    """
    def __init__(self, img_size=64, patch_size=16, in_chans=3, embed_dim=512, num_frames=24):
        super().__init__()
        self.patch_size  = patch_size
        self.num_frames  = num_frames
        self.grid_size   = img_size // patch_size          # spatial grid per frame
        self.num_patches = self.grid_size * self.grid_size * num_frames

        # kernel (1, P, P) — patch spatially per frame, stride 1 in time
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride     =(1, patch_size, patch_size)
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)                     # (B, D, T, Hs, Ws)
        B, D, T, Hs, Ws = x.shape
        x = x.flatten(2).transpose(1, 2)     # (B, T*Hs*Ws, D)
        return x


class FactorisedPositionalEncoding(nn.Module):
    """
    Learnable spatial embedding  (1, Hs*Ws, D)
    + learnable temporal embedding (1, T, D)
    broadcast-added over the token sequence (B, T*Hs*Ws, D).
    """
    def __init__(self, num_frames, spatial_patches, embed_dim):
        super().__init__()
        self.T  = num_frames
        self.Sp = spatial_patches   # Hs * Ws

        self.spatial_embed  = nn.Parameter(torch.zeros(1, spatial_patches, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        nn.init.trunc_normal_(self.spatial_embed,  std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)

    def forward(self, x):
        """x: (B, T*Sp, D)"""
        B, N, D = x.shape
        # spatial_embed: (1, Sp, D)  → tile T times → (1, T*Sp, D)
        sp  = self.spatial_embed.repeat(1, self.T, 1)          # (1, T*Sp, D)
        # temporal_embed: (1, T, D) → repeat Sp times interleaved
        # Each block of Sp tokens belongs to the same frame → repeat_interleave
        te  = self.temporal_embed.repeat_interleave(self.Sp, dim=1)  # (1, T*Sp, D)
        return x + sp + te


class VideoViT(nn.Module):
    """
    Video Vision Transformer.
    Input:  (B, C, T, H, W)
    Output: (B, T*Hs*Ws, embed_dim)
    """
    def __init__(
        self,
        img_size    = 64,
        patch_size  = 16,
        in_chans    = 3,
        num_frames  = 24,
        embed_dim   = 512,
        depth       = 6,
        num_heads   = 8,
        mlp_ratio   = 4.0,
        drop_rate   = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim, num_frames)
        Hs = Ws = img_size // patch_size
        spatial_patches = Hs * Ws

        self.pos_embed = FactorisedPositionalEncoding(num_frames, spatial_patches, embed_dim)
        self.pos_drop  = nn.Dropout(p=drop_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = embed_dim,
            nhead          = num_heads,
            dim_feedforward= int(embed_dim * mlp_ratio),
            dropout        = drop_rate,
            activation     = "gelu",
            batch_first    = True,
            norm_first     = True,   # Pre-Norm (standard for ViT)
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm   = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (B, C, T, H, W)  →  out: (B, T*Hs*Ws, embed_dim)"""
        x = self.patch_embed(x)   # (B, T*Hs*Ws, D)
        x = self.pos_embed(x)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


if __name__ == "__main__":
    model = VideoViT(img_size=64, patch_size=16, num_frames=24, embed_dim=512)
    dummy = torch.randn(2, 3, 24, 64, 64)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")   # expect (2, 384, 512)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {params/1e6:.2f}M")
