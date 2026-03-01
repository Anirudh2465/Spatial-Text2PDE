"""
plot_reconstruction.py — Generate images comparing GT and AE reconstruction

Usage:
    python scripts/06_grassmann_fno/plot_reconstruction.py \
        --ckpt_path checkpoints/grassmann_fno/best.pth \
        --rank 8 --grid_size 8
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT)

from src.models.grassmann_fno_ae import GrassmannFNOAutoencoder
from src.data.mesh_dataset import CylinderMeshDataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', type=str, default='d:/Semester 6/Natural Language Processing/Project 3/checkpoints/grassmann_fno/best.pth')
    p.add_argument('--data_path', type=str, default='d:/Semester 6/Natural Language Processing/Project 3/train_downsampled_labeled.h5')
    p.add_argument('--stat_path', type=str, default='d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl')
    p.add_argument('--output_path', type=str, default='d:/Semester 6/Natural Language Processing/Project 3/grassmann_ae_reconstruction.png')
    
    # Needs to match training exactly
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--grid_size', type=int, default=8)
    p.add_argument('--num_timesteps', type=int, default=25)
    
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Dataset
    print(f"Loading data from {args.data_path}")
    dataset = CylinderMeshDataset(
        file_path=args.data_path,
        num_timesteps=args.num_timesteps,
        normalise=True,
        stat_path=args.stat_path,
        max_samples=100  # only need a few for plotting
    )
    
    # 2. Build Model
    model = GrassmannFNOAutoencoder(
        in_channels=3,
        rank=args.rank,
        grid_size=args.grid_size,
        nt=args.num_timesteps,
        fno_modes=(8, 8, 8),
        fno_hidden=64,
        fno_layers=4,
        hidden_channels=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        dropout=0.1,
        z_channels=16,
        kl_weight=0.0
    ).to(device)
    
    # 3. Load Checkpoint
    print(f"Loading checkpoint {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 4. Inference on Sample
    sample_idx = 0
    x_norm = dataset[sample_idx]  # (T, M, 3)
    x_input = x_norm.unsqueeze(0).to(device)  # (1, T, M, 3)
    
    print("Running inference...")
    with torch.no_grad():
        x_rec_norm, _ = model(x_input)
    
    x_rec_norm = x_rec_norm.squeeze(0).cpu()  # (T, M, 3)
    
    # Un-normalize
    if dataset.mean is not None:
        mean = dataset.mean
        std = dataset.std
        x_orig = x_norm * std + mean
        x_rec  = x_rec_norm * std + mean
    else:
        x_orig = x_norm
        x_rec = x_rec_norm

    # 5. Get Mesh Info
    mesh_pos, cells, _ = dataset.get_mesh_info(sample_idx)
    
    # 6. Plot Magnitude of Velocity at mid-time step
    t_idx = args.num_timesteps // 2
    
    def get_mag(tensor):
        # tensor is (T, M, 3) -> return (M,) magnitude for t_idx
        u = tensor[t_idx, :, 0].numpy()
        v = tensor[t_idx, :, 1].numpy()
        return np.sqrt(u**2 + v**2)
    
    mag_gt = get_mag(x_orig)
    mag_rec = get_mag(x_rec)
    mag_err = np.abs(mag_gt - mag_rec)
    
    # Plot formatting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin, vmax = mag_gt.min(), mag_gt.max()
    
    # GT
    im0 = axes[0].tripcolor(mesh_pos[:, 0], mesh_pos[:, 1], cells, mag_gt, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Ground Truth Velocity (t={t_idx})")
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0])
    
    # Recon
    im1 = axes[1].tripcolor(mesh_pos[:, 0], mesh_pos[:, 1], cells, mag_rec, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Reconstructed Velocity (t={t_idx})")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1])
    
    # Error
    im2 = axes[2].tripcolor(mesh_pos[:, 0], mesh_pos[:, 1], cells, mag_err, shading='gouraud', cmap='inferno')
    axes[2].set_title(f"Absolute Error")
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2])
    
    # Compute relative L2 norm for this frame
    rel_error = np.linalg.norm(mag_err) / np.linalg.norm(mag_gt)
    
    plt.suptitle(f"Grassmann-FNO AE Reconstruction (Rel. Error: {rel_error:.4f})", fontsize=16)
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=200)
    print(f"Saved plot to {args.output_path}")

if __name__ == '__main__':
    main()
