"""
mesh_dataset.py — Dataset for irregular-mesh cylinder flow data (HDF5).

Loads (u, v, pressure) from train_downsampled_labeled.h5 and returns
tensors of shape (T, M, 3) for the Grassmann-FNO autoencoder.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class CylinderMeshDataset(Dataset):
    """
    Loads cylinder flow data from an HDF5 file containing unstructured mesh
    simulations.

    Each sample returns a tensor of shape (T, M, 3) representing the
    velocity (u, v) and pressure fields at T time steps on M mesh nodes.

    Args:
        file_path:     path to the HDF5 file
        num_timesteps: number of temporal frames to extract per sample
        normalise:     if True, normalise per-channel globally (z-score)
        stat_path:     optional path to a .pkl stats file (same format as
                       src.data.normalization) for consistent normalisation
        max_samples:   if set, cap the dataset at this many samples
    """

    def __init__(self, file_path: str,
                 num_timesteps: int = 25,
                 normalise: bool = True,
                 stat_path: str = None,
                 max_samples: int = None):
        self.file_path     = file_path
        self.num_timesteps = num_timesteps
        self.normalise     = normalise
        self.stat_path     = stat_path

        # Build index of valid keys
        with h5py.File(file_path, 'r') as f:
            keys = sorted(f.keys(), key=lambda x: int(x) if x.isdigit() else x)
        if max_samples is not None:
            keys = keys[:max_samples]
        self.keys = keys

        # Load normalisation stats (optional)
        self.mean = None
        self.std  = None
        if normalise and stat_path is not None and os.path.exists(stat_path):
            import pickle
            with open(stat_path, 'rb') as f:
                stats = pickle.load(f)
            # stats = [m_u, s_u, m_v, s_v, m_p, s_p]
            self.mean = torch.tensor([stats[0], stats[2], stats[4]],
                                     dtype=torch.float32)  # (3,)
            self.std  = torch.tensor([stats[1], stats[3], stats[5]],
                                     dtype=torch.float32)
            self.std[self.std < 1e-6] = 1.0
        elif normalise:
            # Compute statistics from the dataset on first access (lazy)
            # We'll do it inline during __getitem__ if mean/std not set
            pass

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            x: (T, M, 3) float32 tensor — mesh fields [u, v, pressure]
        """
        key = self.keys[idx]
        with h5py.File(self.file_path, 'r') as f:
            grp = f[key]

            # Temporal downsampling
            total_steps = int(grp['u'].shape[0])
            if total_steps >= self.num_timesteps:
                indices = np.linspace(0, total_steps - 1,
                                      self.num_timesteps, dtype=int)
            else:
                indices = np.arange(total_steps)

            u = grp['u'][indices]        # (T, M)
            v = grp['v'][indices]        # (T, M)
            p = grp['pressure'][indices] # (T, M) or (T, M, 1)

        # Squeeze trailing dim if present (some files: (T, M, 1))
        if p.ndim == 3 and p.shape[2] == 1:
            p = p[..., 0]

        # Stack → (T, M, 3)
        x = torch.tensor(
            np.stack([u, v, p], axis=-1), dtype=torch.float32
        )

        # Normalise
        if self.normalise:
            if self.mean is not None:
                x = (x - self.mean.to(x.device)) / self.std.to(x.device)
            else:
                # Per-sample z-score (fallback when no global stats)
                mean = x.mean(dim=(0, 1), keepdim=True)
                std  = x.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
                x    = (x - mean) / std

        return x

    def get_mesh_info(self, idx: int = 0):
        """
        Returns mesh positions and cell connectivity for sample `idx`.

        Returns:
            mesh_pos:  (M, 2) float array — node XY coordinates
            cells:     (Nc, 3) int array  — triangle connectivity
            node_type: (M, 1) int array   — boundary/interior flags
        """
        key = self.keys[idx]
        with h5py.File(self.file_path, 'r') as f:
            grp = f[key]
            mesh_pos  = grp['mesh_pos'][:]
            cells     = grp['cells'][:]
            node_type = grp['node_type'][:]
        return mesh_pos, cells, node_type


class PaddedMeshCollator:
    """
    Collate function for DataLoader that pads variable-length mesh dimensions.

    When different samples have different numbers of mesh nodes (M varies),
    this pads all samples in a batch to the maximum M with zeros.
    """

    def __call__(self, batch):
        """
        batch: list of (T, M_i, C) tensors

        Returns:
            x:    (B, T, M_max, C)  padded batch
            mask: (B, M_max)        True where valid (not padded)
        """
        Ts  = [b.shape[0] for b in batch]
        Ms  = [b.shape[1] for b in batch]
        C   = batch[0].shape[2]
        T   = max(Ts)
        M   = max(Ms)
        B   = len(batch)

        x_pad    = torch.zeros(B, T, M, C, dtype=batch[0].dtype)
        mask     = torch.zeros(B, M, dtype=torch.bool)

        for i, b in enumerate(batch):
            ti, mi = b.shape[0], b.shape[1]
            x_pad[i, :ti, :mi, :] = b
            mask[i, :mi] = True

        return x_pad, mask
