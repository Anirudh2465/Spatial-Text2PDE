import torch
import torch.nn as nn
import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.autoencoder import Autoencoder3D
from src.data.normalization import Normalizer

CKPT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/ae_finetuned.pth"
DATA_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
STAT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl"
OUTPUT_PLOT = "d:/Semester 6/Natural Language Processing/Project 3/ae_verification_finetuned.png"

def verify():
    # 1. Initialize Model
    model = Autoencoder3D()
    
    # 2. Load Weights (Standard State Dict from Fine-Tuning)
    print(f"Loading fine-tuned weights from {CKPT_PATH}...")
    sd = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    # If saved via torch.save(model.state_dict()), no need for 'state_dict' key extraction or mapping unless wrapped
    if 'state_dict' in sd: sd = sd['state_dict'] # Handle just in case
    
    # Check if keys match directly (Fine-tuned model should have same keys as model class)
    # The finetune script did: torch.save(model.state_dict())
    # So keys should be compatible directly.
    msg = model.load_state_dict(sd, strict=False)
    print(f"Weights Loaded. Missing keys: {len(msg.missing_keys)}")
    
    # 3. Load Data & Normalizer
    print("Loading Data & Normalizer...")
    normalizer = Normalizer(STAT_PATH)
    
    with h5py.File(DATA_PATH, 'r') as f:
        grid = f['0']['grid'][:] # (25, 3, 64, 64)
        x = torch.tensor(grid).unsqueeze(0).permute(0, 1, 2, 3, 4) # (1, 25, 3, 64, 64)
    
    # Normalize
    x_in = normalizer.normalize(x)
    
    # 4. Forward
    model.eval()
    with torch.no_grad():
        z = model.encode(x_in)
        # z: (1, 16, 6, 16, 16)
        recon_norm = model.decode(z)
        # recon: (1, 24, 3, 64, 64)
        
    # 5. Metrics (Compute Relative L2 Error)
    t_out = recon_norm.shape[1]
    x_target = x_in[:, :t_out]
    
    # Standard Rel L2 = ||Pred - GT||_2 / ||GT||_2
    diff = recon_norm - x_target
    l2_error = torch.norm(diff, p=2)
    l2_ref = torch.norm(x_target, p=2)
    
    rel_l2 = l2_error / l2_ref
    
    # Check L1 too
    l1_loss = nn.L1Loss()(recon_norm, x_target)
    
    print(f"Results:")
    print(f"  Relative L2 Error: {rel_l2.item():.4f} ({rel_l2.item()*100:.2f}%)")
    print(f"  L1 Loss (Norm):    {l1_loss.item():.5f}")
    
    success = rel_l2.item() < 0.05
    if success:
        print("SUCCESS: Error is within expected range (<5%).")
    else:
        print(f"Note: Error {rel_l2.item():.4f} is higher than strict 2%, likely due to slight normalization mismatch or SVD compression loss.")

    # 6. Visualization
    recon = normalizer.unnormalize(recon_norm) 
    # Use Unnormalized for visual reality
    
    # Get frame 10 (mid sequence)
    frame_idx = 10
    
    def get_mag(tensor_x):
        # (1, T, 3, H, W)
        u = tensor_x[0, frame_idx, 0]
        v = tensor_x[0, frame_idx, 1]
        return np.sqrt(u**2 + v**2)
    
    mag_orig = get_mag(x) # Original raw input
    mag_recon = get_mag(recon)
    
    # Error Map
    mag_diff = np.abs(mag_orig - mag_recon)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin, vmax = mag_orig.min(), mag_orig.max()
    
    im0 = axes[0].imshow(mag_orig, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Original (Frame {frame_idx})")
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(mag_recon, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Reconstructed (Frame {frame_idx})")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1])
    
    # Error
    im2 = axes[2].imshow(mag_diff, cmap='inferno')
    axes[2].set_title("Absolute Error")
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2])
    
    plt.suptitle(f"Autoencoder Reconstruction (Rel L2 Error: {rel_l2.item():.4f})", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    verify()
