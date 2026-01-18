import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """
    2D Fourier Layer.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to a factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNOBlock(nn.Module):
    def __init__(self, width, modes):
        super(FNOBlock, self).__init__()
        self.conv = SpectralConv2d(width, width, modes, modes)
        self.w = nn.Conv2d(width, width, 1) # 1x1 convolution for residual
        self.bn = nn.BatchNorm2d(width) # Optional, usually Gelu is enough

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        return F.gelu(x1 + x2)

class RecurrentFNO(nn.Module):
    def __init__(self, modes=12, width=32):
        super(RecurrentFNO, self).__init__()
        
        self.modes = modes
        self.width = width
        
        # Input: u, v, p, Re -> 4 channels (mask/grid could be added)
        # Assuming Re is expanded as a channel map
        self.fc0 = nn.Linear(4, self.width) # Lifting
        
        self.fno1 = FNOBlock(width, modes)
        self.fno2 = FNOBlock(width, modes)
        self.fno3 = FNOBlock(width, modes)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3) # Output: u_next, v_next, p_next

    def forward(self, u, v, p, re_num):
        # x: (batch, 3, 64, 64) -> current state
        # re_num: (batch, 1) -> scalar
        
        batch_size, _, H, W = u.shape
        
        # Create Re channel
        re_map = re_num.view(batch_size, 1, 1, 1).expand(batch_size, 1, H, W)
        
        # Stack inputs: (B, 4, H, W)
        x = torch.cat([u, v, p, re_map], dim=1)
        
        # Rearrange for Linear layer: (B, H, W, 4)
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # (B, Width, H, W)
        
        # FNO Layers
        x = self.fno1(x)
        x = self.fno2(x)
        x = self.fno3(x)
        
        # Projection
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x) # (B, H, W, 3)
        x = x.permute(0, 3, 1, 2)
        
        return x # u, v, p next step delta or absolute? Usually absolute for FNO
