import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, affine=True)
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
        h = F.silu(h) # Swish
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
            
        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        if with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        else:
            self.conv = nn.AvgPool3d(2, stride=2)

    def forward(self, x):
        # Check padding for even division
        pad = (0, 1, 0, 1, 0, 1) # Simple pad for now, strictly should check dims
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        # Flatten for attn
        b, c, d, h, w = q.shape
        q = q.reshape(b, c, -1)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, -1)
        v = v.reshape(b, c, -1)
        v = v.permute(0, 2, 1)
        
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)
        
        # attend to values
        h_ = torch.bmm(w_, v)
        h_ = h_.permute(0, 2, 1) # b, c, dhw
        h_ = h_.reshape(b, c, d, h, w)
        
        h_ = self.proj_out(h_)
        return x + h_

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1) # Input channels 64 from Projection
        
        # Down 0
        self.down_0_block_0 = ResnetBlock(64, 64)
        self.down_0_block_1 = ResnetBlock(64, 64)
        self.down_0_down = Downsample(64)
        
        # Down 1
        self.down_1_block_0 = ResnetBlock(64, 128)
        self.down_1_block_1 = ResnetBlock(128, 128)
        self.down_1_down = Downsample(128)
        
        # Mid
        self.mid_block_0 = ResnetBlock(128, 256)
        self.mid_attn_0 = AttnBlock(256)
        self.mid_block_1 = ResnetBlock(256, 256)
        
        self.norm_out = nn.GroupNorm(32, 256, eps=1e-6, affine=True)
        self.conv_out = nn.Conv3d(256, 16, kernel_size=3, padding=1) # To 4*4 channels for Quant? 
        # Check quant_conv weights: [4, 64, 1, 1, 1] means input 64?
        # My model scales to 256. 
        # I check weights: `decoder.conv_in` is [256, ...].
        # So Latent must be mapped to 256?
        # Wait, `quant_conv` was `[4, 64]`? 
        # I need to match EXACT keys.
        # I will structure this generally.
        
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input Projection (from 3ch to 64ch)
        self.proj = nn.Linear(3, 64)
        
        # Encoder Backbone (Simplified based on inspection)
        # Assuming standard structure map
        # I'll construct modules dynamically to match dict or use specific names
        # Based on logs:
        # encoder.cnn_encoder.down.0 (64->64)
        # encoder.cnn_encoder.down.1 (64->128) -> downsample -> (128)
        # encoder.cnn_encoder.mid (128->256)
        # encoder.cnn_encoder.norm_out, conv_out
        
        # Actually implementing dynamic loading is safer.
        self.encoder = nn.ModuleDict({
            'conv_in': nn.Conv3d(64, 64, 3, 1, 1)
        })
        # ... complex construction ...
        
    def load_weights(self, path):
         # ...
         pass

# I will provide a dedicated script `load_and_test.py` that builds the model dynamically using the keys!
# This is safer than hardcoding classes.
# No, constructing classes is required for forward pass.
