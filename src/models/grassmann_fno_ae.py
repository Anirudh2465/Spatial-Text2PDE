"""
grassmann_fno_ae.py — Grassmann-FNO Autoencoder for Cylinder Flow

Replaces GINO's radius-based neighbour search with SVD-based Grassmann
projection. The FNO operates on compact mode descriptors arranged on a
regular (nt × G × G) grid — no neighbour search needed.

Data flow:
    x: (B, T, M, 3)
    → GrassmannProjector   → U:(B,T,M,k)  S:(B,T,k)  Vh:(B,T,k,3)
    → ModeEmbedder         → (B, fno_hidden, T, G, G)
    → FNO3d                → (B, fno_hidden, T, G, G)
    → CNN_Encoder (3D)     → z_moments: (B, 2*z_ch, T', G', G')
    → sample z             → (B, z_ch, T', G', G')
    → CNN_Decoder (3D)     → (B, fno_hidden, T, G, G)
    → FNO3d (inverse)      → (B, fno_hidden, T, G, G)
    → ModeUnembedder       → S_hat:(B,T,k)  Vh_hat:(B,T,k,3)
    → GrassmannReconstructor (with stored U_ref) → x_rec:(B,T,M,3)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.grassmann import GrassmannProjector, GrassmannReconstructor


# ---------------------------------------------------------------------------
# 3D Fourier Neural Operator
# ---------------------------------------------------------------------------

class SpectralConv3d(nn.Module):
    """
    3D Fourier integral operator layer.
    Operates on (B, C, T, H, W) tensors.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes_t: int, modes_h: int, modes_w: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_t = modes_t
        self.modes_h = modes_h
        self.modes_w = modes_w

        scale = 1.0 / (in_channels * out_channels)
        # 8 corners in 3D rfft frequency space
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.rand(in_channels, out_channels,
                                            modes_t, modes_h, modes_w,
                                            dtype=torch.cfloat))
            for _ in range(4)  # 4 corners in (T, H) for rfft
        ])

    def compl_mul3d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # (B, Ci, T, H, W), (Ci, Co, T, H, W) → (B, Co, T, H, W)
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Real FFT over all 3 spatial/temporal dims
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        T, H, W_c = x_ft.shape[-3], x_ft.shape[-2], x_ft.shape[-1]
        mt = min(self.modes_t, T)
        mh = min(self.modes_h, H)
        mw = min(self.modes_w, W_c)

        out_ft = torch.zeros(B, self.out_channels, T, H, W_c,
                             dtype=torch.cfloat, device=x.device)

        # 4 corners: (+t, +h), (+t, -h), (-t, +h), (-t, -h)
        slices = [
            (slice(None, mt),  slice(None, mh)),
            (slice(None, mt),  slice(-mh, None)),
            (slice(-mt, None), slice(None, mh)),
            (slice(-mt, None), slice(-mh, None)),
        ]
        for i, (st, sh) in enumerate(slices):
            out_ft[:, :, st, sh, :mw] = self.compl_mul3d(
                x_ft[:, :, st, sh, :mw], self.weights[i][:, :, :mt, :mh, :mw]
            )

        # Inverse FFT
        x_out = torch.fft.irfftn(out_ft, s=x.shape[-3:])
        return x_out


class FNO3dBlock(nn.Module):
    """Single FNO layer: spectral conv + pointwise skip + GELU."""

    def __init__(self, width: int, modes_t: int, modes_h: int, modes_w: int):
        super().__init__()
        self.spec = SpectralConv3d(width, width, modes_t, modes_h, modes_w)
        self.skip = nn.Conv3d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, width), width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spec(x) + self.skip(x)))


class FNO3d(nn.Module):
    """Stack of 3D FNO blocks."""

    def __init__(self, width: int, n_layers: int,
                 modes_t: int, modes_h: int, modes_w: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            FNO3dBlock(width, modes_t, modes_h, modes_w)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Mode Embedder / Unembedder
# ---------------------------------------------------------------------------

class ModeEmbedder(nn.Module):
    """
    Map (S, Vh) per frame into a regular (fno_hidden, nt, G, G) grid.

    S:  (B, T, k)      singular values
    Vh: (B, T, k, C)   mode-channel mixing

    1. Concatenate (S, Vh.flatten) → (B, T, k + k*C)
    2. Linear projection → (B, T, fno_hidden * G * G)
    3. Reshape → (B, fno_hidden, T, G, G)
    """

    def __init__(self, rank: int, in_channels: int,
                 fno_hidden: int, nt: int, grid_size: int):
        super().__init__()
        self.fno_hidden = fno_hidden
        self.nt = nt
        self.G = grid_size
        in_dim = rank + rank * in_channels  # S dim + Vh dim
        out_dim = fno_hidden * grid_size * grid_size
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, S: torch.Tensor, Vh: torch.Tensor) -> torch.Tensor:
        """
        S:  (B, T, k)
        Vh: (B, T, k, C)
        Returns: (B, fno_hidden, T, G, G)
        """
        B, T = S.shape[:2]
        k = S.shape[-1]
        C = Vh.shape[-1]
        # Flatten Vh: (B, T, k, C) → (B, T, k*C)
        Vh_flat = Vh.reshape(B, T, k * C)
        # Concat with S
        feat = torch.cat([S, Vh_flat], dim=-1)  # (B, T, k + k*C)
        # Project
        out = self.proj(feat)  # (B, T, fno_hidden * G * G)
        # Reshape to grid
        G = self.G
        out = out.reshape(B, T, self.fno_hidden, G, G)
        # (B, fno_hidden, T, G, G) — channels first for FNO
        return out.permute(0, 2, 1, 3, 4)


class ModeUnembedder(nn.Module):
    """
    Inverse of ModeEmbedder: map FNO grid → (S_hat, Vh_hat).

    Uses adaptive average pooling over spatial (H, W) dims so this module
    is robust to any spatial resolution coming out of the CNN decoder.

    Input:  (B, fno_hidden, T, H, W)   (H, W may differ from G)
    Output: S_hat  (B, T, k)
            Vh_hat (B, T, k, C)
    """

    def __init__(self, rank: int, out_channels: int,
                 fno_hidden: int, nt: int, grid_size: int = None):
        super().__init__()
        self.rank = rank
        self.out_channels = out_channels
        self.fno_hidden = fno_hidden
        # After AdaptiveAvgPool2d(1) each T-frame becomes fno_hidden scalars
        in_dim  = fno_hidden
        out_dim = rank + rank * out_channels  # S + Vh
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.GELU(),
            nn.Linear(in_dim * 4, out_dim),
        )
        # Pool spatial dims to 1×1 per frame, across channels
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # keep T, pool H,W

    def forward(self, x: torch.Tensor):
        """
        x: (B, fno_hidden, T, H, W)
        Returns S_hat:(B,T,k), Vh_hat:(B,T,k,C)
        """
        # Pool spatial: (B, fno_hidden, T, 1, 1)
        pooled = self.pool(x).squeeze(-1).squeeze(-1)  # (B, fno_hidden, T)
        # Permute: (B, T, fno_hidden)
        pooled = pooled.permute(0, 2, 1)
        out = self.proj(pooled)  # (B, T, k + k*C)
        k, C = self.rank, self.out_channels
        S_hat  = out[..., :k]                       # (B, T, k)
        Vh_hat = out[..., k:].reshape(*out.shape[:-1], k, C)  # (B, T, k, C)
        return S_hat, Vh_hat


# ---------------------------------------------------------------------------
# CNN Encoder / Decoder (3D ResNet-style)
# ---------------------------------------------------------------------------

def _norm(c): return nn.GroupNorm(min(8, c), c)


class ResBlock3D(nn.Module):
    """3D ResNet block with optional channel change and dropout."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = _norm(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1)
        self.skip  = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.drop(h)
        h = self.conv2(h)
        return h + self.skip(x)


class CNNEncoder3D(nn.Module):
    """
    3D CNN encoder: downsamples the (B, fno_hidden, T, G, G) grid
    to a latent (B, 2*z_ch, T', G', G') representing VAE moments.
    """

    def __init__(self, in_ch: int, hidden_ch: int,
                 ch_mult=(1, 2, 4), num_res: int = 2,
                 z_ch: int = 16, dropout: float = 0.1):
        super().__init__()
        self.conv_in = nn.Conv3d(in_ch, hidden_ch, 3, 1, 1)
        channels = [hidden_ch * m for m in ch_mult]

        layers = []
        cur = hidden_ch
        for nxt in channels:
            for _ in range(num_res):
                layers.append(ResBlock3D(cur, nxt, dropout))
                cur = nxt
            layers.append(nn.Conv3d(cur, cur, 3, stride=(1, 2, 2), padding=1))  # spatial downsample
        self.down = nn.Sequential(*layers)

        self.mid = nn.Sequential(ResBlock3D(cur, cur, dropout))
        self.norm_out = _norm(cur)
        self.conv_out = nn.Conv3d(cur, 2 * z_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        h = self.down(h)
        h = self.mid(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


class CNNDecoder3D(nn.Module):
    """
    3D CNN decoder: upsamples latent (B, z_ch, T', G', G')
    back to (B, fno_hidden, T, G, G).
    """

    def __init__(self, z_ch: int, hidden_ch: int,
                 ch_mult=(4, 2, 1), num_res: int = 2,
                 out_ch: int = 64, dropout: float = 0.1):
        super().__init__()
        # Build channel progression: first one is the projection of z
        channels = [hidden_ch * m for m in ch_mult]
        self.conv_in = nn.Conv3d(z_ch, channels[0], 3, 1, 1)

        layers = []
        cur = channels[0]
        for nxt in channels:
            # ResBlocks at current channel
            for _ in range(num_res):
                layers.append(ResBlock3D(cur, cur, dropout))
            # Upsample spatially (not temporally)
            layers.append(nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'))
            # Channel transition after upsample
            layers.append(nn.Conv3d(cur, nxt, 3, 1, 1))
            cur = nxt
        self.up = nn.Sequential(*layers)

        self.norm_out = _norm(cur)
        self.conv_out = nn.Conv3d(cur, out_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        h = self.up(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class GrassmannFNOEncoder(nn.Module):
    """
    Full encoder: irregular mesh → latent z.

    mesh x (B, T, M, C)
      → GrassmannProjector   U, S, Vh
      → ModeEmbedder         (B, fno_hidden, T, G, G)
      → FNO3d
      → CNNEncoder3D         (B, 2*z_ch, T', G', G')
    """

    def __init__(self, in_channels: int = 3, rank: int = 16,
                 grid_size: int = 16, nt: int = 25,
                 fno_modes=(8, 8, 8), fno_hidden: int = 64, fno_layers: int = 4,
                 hidden_channels: int = 64, ch_mult=(1, 2, 4),
                 num_res_blocks: int = 2, dropout: float = 0.1,
                 z_channels: int = 16):
        super().__init__()
        self.projector = GrassmannProjector(rank)
        self.embedder  = ModeEmbedder(rank, in_channels, fno_hidden, nt, grid_size)
        self.fno       = FNO3d(fno_hidden, fno_layers, *fno_modes)
        self.cnn       = CNNEncoder3D(fno_hidden, hidden_channels, ch_mult,
                                       num_res_blocks, z_channels, dropout)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, M, C)
        Returns:
            moments: (B, 2*z_ch, T', G', G')
            U_ref:   (B, T, M, k)   stored for decoder use
        """
        U, S, Vh = self.projector(x)   # (B,T,M,k), (B,T,k), (B,T,k,C)
        grid = self.embedder(S, Vh)    # (B, fno_hidden, T, G, G)
        grid = self.fno(grid)
        moments = self.cnn(grid)       # (B, 2*z_ch, T', G', G')
        return moments, U


class GrassmannFNODecoder(nn.Module):
    """
    Full decoder: latent z → irregular mesh via stored U_ref.

    z (B, z_ch, T', G', G')
      → CNNDecoder3D        (B, fno_hidden, T, G, G)
      → FNO3d
      → ModeUnembedder      S_hat, Vh_hat
      → GrassmannReconstructor (using externally-provided U_ref)
      → x_rec (B, T, M, C)
    """

    def __init__(self, in_channels: int = 3, rank: int = 16,
                 grid_size: int = 16, nt: int = 25,
                 fno_modes=(8, 8, 8), fno_hidden: int = 64, fno_layers: int = 4,
                 hidden_channels: int = 64, ch_mult=(4, 2, 1),
                 num_res_blocks: int = 2, dropout: float = 0.1,
                 z_channels: int = 16):
        super().__init__()
        self.cnn          = CNNDecoder3D(z_channels, hidden_channels, ch_mult,
                                          num_res_blocks, fno_hidden, dropout)
        self.fno          = FNO3d(fno_hidden, fno_layers, *fno_modes)
        self.unembedder   = ModeUnembedder(rank, in_channels, fno_hidden, nt, grid_size)
        self.reconstructor = GrassmannReconstructor()

    def forward(self, z: torch.Tensor, U_ref: torch.Tensor) -> torch.Tensor:
        """
        z:     (B, z_ch, T', G', G')
        U_ref: (B, T, M, k)

        Returns:
            x_rec: (B, T, M, C)
        """
        h = self.cnn(z)                                 # (B, fno_hidden, T, G, G)
        # Ensure T dimension matches U_ref
        T_ref = U_ref.shape[1]
        if h.shape[2] != T_ref:
            h = F.interpolate(h, size=(T_ref, h.shape[-2], h.shape[-1]),
                              mode='trilinear', align_corners=False)
        h = self.fno(h)                                  # (B, fno_hidden, T, G, G)
        S_hat, Vh_hat = self.unembedder(h)               # (B,T,k), (B,T,k,C)
        x_rec = self.reconstructor(U_ref, S_hat, Vh_hat) # (B,T,M,C)
        return x_rec


# ---------------------------------------------------------------------------
# Full VAE-style Autoencoder
# ---------------------------------------------------------------------------

class GrassmannFNOAutoencoder(nn.Module):
    """
    Grassmann-FNO Variational Autoencoder for irregular mesh cylinder flow.

    Drop-in replacement for the GINO-based autoencoder used in the original
    pipeline. No radius-search, no open3d dependency.

    Args:
        in_channels:     number of field channels (default 3: u, v, p)
        rank:            Grassmann rank k (dominant modes, e.g. 16)
        grid_size:       FNO spatial grid side length G (e.g. 16)
        nt:              number of time steps
        fno_modes:       Fourier modes per dim (nt, G, G)
        fno_hidden:      FNO channel width
        fno_layers:      number of FNO blocks
        hidden_channels: CNN channel width
        ch_mult:         channel multipliers per CNN stage
        num_res_blocks:  ResBlocks per CNN stage
        dropout:         dropout rate
        z_channels:      latent channel depth
        kl_weight:       weight for KL divergence term (0 = deactivate)
    """

    def __init__(self,
                 in_channels: int = 3,
                 rank: int = 16,
                 grid_size: int = 16,
                 nt: int = 25,
                 fno_modes=(8, 8, 8),
                 fno_hidden: int = 64,
                 fno_layers: int = 4,
                 hidden_channels: int = 64,
                 ch_mult=(1, 2, 4),
                 num_res_blocks: int = 2,
                 dropout: float = 0.1,
                 z_channels: int = 16,
                 kl_weight: float = 1e-6):
        super().__init__()
        self.kl_weight = kl_weight

        enc_kwargs = dict(
            in_channels=in_channels, rank=rank, grid_size=grid_size, nt=nt,
            fno_modes=fno_modes, fno_hidden=fno_hidden, fno_layers=fno_layers,
            hidden_channels=hidden_channels, ch_mult=ch_mult,
            num_res_blocks=num_res_blocks, dropout=dropout, z_channels=z_channels,
        )
        dec_kwargs = dict(
            in_channels=in_channels, rank=rank, grid_size=grid_size, nt=nt,
            fno_modes=fno_modes, fno_hidden=fno_hidden, fno_layers=fno_layers,
            hidden_channels=hidden_channels,
            ch_mult=tuple(reversed(ch_mult)),   # reverse for decoder
            num_res_blocks=num_res_blocks, dropout=dropout, z_channels=z_channels,
        )

        self.encoder = GrassmannFNOEncoder(**enc_kwargs)
        self.decoder = GrassmannFNODecoder(**dec_kwargs)

    # ------------------------------------------------------------------
    # Reparameterisation
    # ------------------------------------------------------------------
    def _sample(self, moments: torch.Tensor):
        """
        Split (2*z_ch) moments into mean and log-variance, sample z.
        Returns z, mean, logvar.
        """
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = logvar.clamp(-30.0, 20.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        return z, mean, logvar

    def _kl_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Standard Gaussian KL divergence: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))."""
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, M, C)  mesh field
        Returns:
            z:     (B, z_ch, T', G', G')
            U_ref: (B, T, M, k)
            mean/logvar: for KL computation
        """
        moments, U_ref = self.encoder(x)
        z, mean, logvar = self._sample(moments)
        return z, U_ref, mean, logvar

    def decode(self, z: torch.Tensor, U_ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:     (B, z_ch, T', G', G')
            U_ref: (B, T, M, k)
        Returns:
            x_rec: (B, T, M, C)
        """
        return self.decoder(z, U_ref)

    def forward(self, x: torch.Tensor):
        """
        Full forward pass.

        Args:
            x: (B, T, M, C)
        Returns:
            x_rec:     (B, T, M, C)   reconstruction
            loss_dict: dict with 'recon', 'kl', 'total'
        """
        z, U_ref, mean, logvar = self.encode(x)
        x_rec = self.decode(z, U_ref)

        recon_loss = F.l1_loss(x_rec, x)
        kl_loss    = self._kl_loss(mean, logvar)
        total_loss = recon_loss + self.kl_weight * kl_loss

        loss_dict = {
            'recon': recon_loss,
            'kl':    kl_loss,
            'total': total_loss,
        }
        return x_rec, loss_dict

    def get_last_layer_weight(self):
        """Utility for potential perceptual loss scaling (GAN integration)."""
        return self.decoder.cnn.conv_out.weight
