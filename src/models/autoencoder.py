import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Utilities
def normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h

class Downsample3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)

class Upsample3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalize(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, d, h, w = q.shape
        q = q.reshape(b, c, -1).permute(0, 2, 1)
        k = k.reshape(b, c, -1)
        v = v.reshape(b, c, -1).permute(0, 2, 1)
        attn = torch.bmm(q, k) * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        h_ = torch.bmm(attn, v)
        h_ = h_.permute(0, 2, 1).reshape(b, c, d, h, w)
        return x + self.proj_out(h_)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        
        # Down 0
        self.down_0 = nn.ModuleList([ResnetBlock3D(64, 64), ResnetBlock3D(64, 64)])
        self.down_0_ds = Downsample3D(64)
        
        # Down 1
        self.down_1 = nn.ModuleList([ResnetBlock3D(64, 128, conv_shortcut=False), ResnetBlock3D(128, 128)])
        self.down_1_ds = Downsample3D(128)
        
        # Down 2 (New Layer from Inspection)
        # block.0 (128->256), block.1 (256). attn.0, attn.1
        # Keys show: block.0, block.1, attn.0, attn.1.
        # Structure is usually: Block -> Attn -> Block? Or Block -> Block -> Attn?
        # Keys: down.2.block.0, down.2.attn.0, down.2.attn.1, down.2.block.1?
        # Let's assume sequential: Block0(128->256), Attn0, Attn1, Block1.
        # Wait, usually 1 attn per block? Or AttnBlock is separate?
        # Keys show `attn.0` and `attn.1`.
        # I'll add them in likely order: Block0, Attn0, Attn1, Block1.
        # Weights for block.0 input 128 out 256. (Conv1 [256, 128]).
        self.down_2_block_0 = ResnetBlock3D(128, 256, conv_shortcut=False)
        self.down_2_attn_0 = AttnBlock3D(256)
        self.down_2_attn_1 = AttnBlock3D(256)
        self.down_2_block_1 = ResnetBlock3D(256, 256)
        # No downsample here (keys don't show down.2.downsample)
        
        # Mid
        # Keys: mid.block_1, mid.attn_1, mid.block_2.
        self.mid_block_1 = ResnetBlock3D(256, 256)
        self.mid_attn_1 = AttnBlock3D(256)
        self.mid_block_2 = ResnetBlock3D(256, 256)
        
        self.norm_out = normalize(256)
        self.conv_out = nn.Conv3d(256, 32, kernel_size=3, padding=1) # 32 = 16 Mean + 16 LogVar

    def forward(self, x):
        h = self.conv_in(x)
        # Down 0
        for m in self.down_0: h = m(h)
        h = self.down_0_ds(h)
        # Down 1
        for m in self.down_1: h = m(h)
        h = self.down_1_ds(h)
        # Down 2
        h = self.down_2_block_0(h)
        h = self.down_2_attn_0(h)
        h = self.down_2_attn_1(h)
        h = self.down_2_block_1(h)
        
        # Mid
        h = self.mid_block_1(h)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv3d(16, 256, kernel_size=3, padding=1) # Input 16 (Sampled)
        
        # Mid
        self.mid_block_1 = ResnetBlock3D(256, 256)
        self.mid_attn_1 = AttnBlock3D(256)
        self.mid_block_2 = ResnetBlock3D(256, 256)
        
        # Up (Reverse of Down)
        # Encoder Down: 0(64), 1(128), 2(256).
        # Decoder Up: 2(256->256, Upsample?), 1(256->128, Upsample), 0(128->64, Upsample?).
        # Keys: `decoder.cnn_decoder.up.0`, `up.1`, `up.2`.
        # up.2: Inputs 256. (Mid out).
        # up.2.block.0? (256->256). attn.0, attn.1, attn.2?
        # up.2.upsample?
        
        # I'll create generic Up 2, 1, 0 containers and trust the key mapper to fill them or dynamic list.
        # But for shape correctness I need channels right.
        
        # Up 2 (Corresponds to Enc Down 2 - 256ch)
        self.up_2 = nn.ModuleList([
            ResnetBlock3D(256, 256),
            ResnetBlock3D(256, 256),
            ResnetBlock3D(256, 256),
            Upsample3D(256) # 256 out
        ])
        # Keys: up.2.attn.0, up.2.attn.1, up.2.attn.2. (THREE attns?).
        # Ok, I'll add them. The order is tricky. Block, Attn, Block?
        # I'll make a sequential block in forward.
        self.up_2_attns = nn.ModuleList([AttnBlock3D(256), AttnBlock3D(256), AttnBlock3D(256)])
        
        # Up 2 (Corresponds to Enc Down 2 - 256ch)
        self.up_2 = nn.ModuleList([
            ResnetBlock3D(256, 256),
            ResnetBlock3D(256, 256),
            ResnetBlock3D(256, 256),
            Upsample3D(256) # 256 out
        ])
        
        # Up 1
        self.up_1 = nn.ModuleList([
            ResnetBlock3D(256, 128, conv_shortcut=False), 
            ResnetBlock3D(128, 128),
            ResnetBlock3D(128, 128),
            Upsample3D(128)
        ])
        
        # Up 0
        self.up_0 = nn.ModuleList([
            ResnetBlock3D(128, 64, conv_shortcut=False),
            ResnetBlock3D(64, 64),
            ResnetBlock3D(64, 64)
        ])
        
        self.norm_out = normalize(64)
        self.conv_out = nn.Conv3d(64, 64, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        
        # Mid
        h = self.mid_block_1(h)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h)
        
        # Up 2
        for i in range(3): 
            h = self.up_2[i](h)
            h = self.up_2_attns[i](h) 
        h = self.up_2[3](h) # Upsample
        
        for m in self.up_1: h = m(h)
        for m in self.up_0: h = m(h)
          
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

class Autoencoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 64)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.proj_out = nn.Linear(64, 3) 
        
    def encode(self, x):
        b, t, c, h_dim, w_dim = x.shape
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(-1, c) 
        embed = self.proj(x_flat) 
        embed = embed.reshape(b, t, h_dim, w_dim, 64).permute(0, 4, 1, 2, 3) # (B, 64, T, H, W)
        moments = self.encoder(embed) # (B, 32, T', H', W')
        
        # Reparameterization (Split 32 -> 16, 16)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        z = mean # Use mean for deterministic encoding (or sample: mean + std * eps)
        # For verification/inference context, mean is usually preferred.
        # If training, we sample.
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
        return z

    def decode(self, z):
        h = self.decoder(z) # (B, 64, T, H, W)
        b, c, t, h_dim, w_dim = h.shape
        h_flat = h.permute(0, 2, 3, 4, 1).reshape(-1, c)
        out = self.proj_out(h_flat)
        out = out.reshape(b, t, h_dim, w_dim, 3).permute(0, 1, 4, 2, 3) 
        return out
