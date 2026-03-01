import torch
import torch.nn as nn
import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Add root to path if needed (assuming running from root)
if os.path.abspath('.') not in sys.path:
    sys.path.append(os.path.abspath('.'))

from src.models.autoencoder import Autoencoder3D
from src.data.normalization import Normalizer

# Patch for legacy checkpoint that might refer to modules.utils.Normalizer
import sys
import types
if 'modules' not in sys.modules:
    sys.modules['modules'] = types.ModuleType('modules')
if 'modules.utils' not in sys.modules:
    sys.modules['modules.utils'] = sys.modules['src.data.normalization']

CKPT_PATH = "ae_cylinder.ckpt"
DATA_PATH = "train_grid_64.h5"
STAT_PATH = "train_normal_stat.pkl"
OUTPUT_GIF = "ae_output_sample.gif"
OUTPUT_IMAGE = "ae_output_sample.png"

def display_ae_output():
    # 1. Initialize Model
    print("Initializing Model...")
    model = Autoencoder3D()
    
    # 2. Load Weights
    print(f"Loading weights from {CKPT_PATH}...")
    if not os.path.exists(CKPT_PATH):
        print(f"Error: Checkpoint {CKPT_PATH} not found.")
        return

    # Load checkpoint - handle both full checkpoint and state_dict
    # Set weights_only=False because the checkpoint might contain arbitrary objects (like Normalizer)
    sd = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    
    # Load state dict
    msg = model.load_state_dict(sd, strict=False)
    print(f"Weights Loaded. Missing keys: {len(msg.missing_keys)}")
    
    # 3. Load Data & Normalizer
    print("Loading Data & Normalizer...")
    if not os.path.exists(STAT_PATH):
        print(f"Error: Stat file {STAT_PATH} not found.")
        return
        
    normalizer = Normalizer(STAT_PATH)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file {DATA_PATH} not found.")
        return

    with h5py.File(DATA_PATH, 'r') as f:
        # Load just one sample (index 0)
        grid = f['0']['grid'][:] # (25, 3, 64, 64)
        x = torch.tensor(grid).unsqueeze(0).permute(0, 1, 2, 3, 4) # (1, 25, 3, 64, 64)
        print(f"Input shape: {x.shape}")

    # Normalize
    x_in = normalizer.normalize(x)
    
    # 4. Forward
    model.eval()
    with torch.no_grad():
        z = model.encode(x_in)
        recon_norm = model.decode(z)
        
    # Unnormalize
    recon = normalizer.unnormalize(recon_norm)

    # 5. Display Text Info
    print("\n" + "="*40)
    print("SAMPLE INFO")
    print("="*40)
    print(f"Sample Index: 0")
    print(f"Data Source: {DATA_PATH}")
    print(f"Original Shape: {x.shape}")
    print(f"Reconstructed Shape: {recon.shape}")
    print(f"Latent Representation Shape: {z.shape}")
    print("Description: Flow around a cylinder simulation data.")
    print("="*40 + "\n")

    # 6. Generate Visualization (GIF)
    print("Generating Visualization...")
    
    # Helper to get magnitude
    def get_mag(tensor, frame):
        # tensor: (1, T, 3, H, W)
        u = tensor[0, frame, 0]
        v = tensor[0, frame, 1]
        return np.sqrt(u**2 + v**2)

    num_frames = x.shape[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Calculate global vmin/vmax for consistent colorbar
    mag_orig_all = np.sqrt(x[0, :, 0]**2 + x[0, :, 1]**2)
    vmin, vmax = mag_orig_all.min(), mag_orig_all.max()

    im0 = axes[0].imshow(np.zeros((64, 64)), cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title("Original Input")
    axes[0].axis('off')
    
    im1 = axes[1].imshow(np.zeros((64, 64)), cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstructed Output")
    axes[1].axis('off')
    
    plt.tight_layout()

    def update(frame):
        mag_orig = get_mag(x, frame)
        mag_recon = get_mag(recon, frame)
        
        im0.set_data(mag_orig)
        im1.set_data(mag_recon)
        return im0, im1

    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True)
    
    try:
        ani.save(OUTPUT_GIF, writer='pillow', fps=5)
        print(f"Saved GIF to {os.path.abspath(OUTPUT_GIF)}")
    except Exception as e:
        print(f"Could not save GIF: {e}")
        # Fallback to saving a single frame image
        mag_orig = get_mag(x, 10)
        mag_recon = get_mag(recon, 10)
        im0.set_data(mag_orig)
        im1.set_data(mag_recon)
        plt.savefig(OUTPUT_IMAGE)
        print(f"Saved single frame comparison to {os.path.abspath(OUTPUT_IMAGE)}")

if __name__ == "__main__":
    display_ae_output()
