import torch
import torch.nn as nn
import numpy as np
import h5py
import sys
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Mock modules
import types
m = types.ModuleType('modules')
sys.modules['modules'] = m
m.utils = types.ModuleType('utils')
class MockClass:
    def __init__(self, *args, **kwargs): pass
m.utils.Normalizer = MockClass
sys.modules['modules.utils'] = m.utils

# Add src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.autoencoder import Autoencoder3D
from src.models.diffusion import DiffusionTransformer, DDIMSampler
from src.data.normalization import Normalizer

AE_CKPT = "d:/Semester 6/Natural Language Processing/Project 3/ae_cylinder.ckpt"
DIT_CKPT = "d:/Semester 6/Natural Language Processing/Project 3/ldm_DiT_text_cylinder.ckpt"
DATA_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
STAT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl"
OUTPUT_DIR = "d:/Semester 6/Natural Language Processing/Project 3/generated_results"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0, help='Sample index from dataset')
    parser.add_argument('--prompt', type=str, default="", help='Text prompt for conditioning')
    parser.add_argument('--steps', type=int, default=50, help='DDIM sampling steps')
    parser.add_argument('--strength', type=float, default=0.8, help='Refinement strength (0.0=No Change, 1.0=Full Generation)')
    parser.add_argument('--no-clip', action='store_true', help='Skip CLIP loading (use zero embedding)')
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Models
    print("Loading Models...")
    ae = Autoencoder3D().to(device).eval()
    load_ae_weights(ae, AE_CKPT)
    
    dit = DiffusionTransformer(
        input_size=(6, 16, 16), patch_size=2, in_channels=16, hidden_size=1024, depth=28, num_heads=16, context_dim=768
    ).to(device).eval()
    dit.load_from_ckpt(DIT_CKPT)
    
    sampler = DDIMSampler(dit, ckpt_path=DIT_CKPT, ddim_steps=args.steps)
    normalizer = Normalizer(STAT_PATH)
    
    # 2. Load Data (First Frame)
    with h5py.File(DATA_PATH, 'r') as f:
        grid = f[str(args.index)]['grid'][:] # (25, 3, 64, 64)
    
    x_gt = torch.tensor(grid).unsqueeze(0).permute(0, 1, 2, 3, 4).float().to(device) # (1, 25, 3, 64, 64)
    x_gt = x_gt[:, :24] # Crop to 24
    
    # 3. Create Coarse Guess
    # Strategy: "First Frame Extrapolation"
    # Create a tensor where every frame is the first frame
    first_frame = x_gt[:, 0:1, :, :, :] # (1, 1, 3, 64, 64)
    x_coarse = first_frame.repeat(1, 24, 1, 1, 1) # (1, 24, 3, 64, 64)
    
    # Add noise to "unlock" diffusion? 
    # Actually, diffusion works by adding noise schedule.
    # We normalized coarse input.
    x_coarse_norm = normalizer.normalize(x_coarse)
    
    # 4. Text Embedding
    print(f"Refining with strength {args.strength}...")
    
    if args.no_clip:
        print("Skipping CLIP. Using Zero Embedding.")
        text_emb = torch.zeros(1, 77, 768).to(device)
    else:
        # Try to load CLIP, else random
        text_emb = get_text_embedding(args.prompt, device)
    
    # 5. Refinement
    with torch.no_grad():
        z_coarse = ae.encode(x_coarse_norm)
        
        # Refine
        z_refined = sampler.refine(z_coarse, text_emb, strength=args.strength)
        
        # Decode
        x_refined_norm = ae.decode(z_refined)
        x_refined = normalizer.unnormalize(x_refined_norm)
    
    # 6. Visualization
    save_gif(x_gt, x_coarse, x_refined, args.index, args.prompt, OUTPUT_DIR)

def get_text_embedding(prompt, device):
    try:
        from transformers import CLIPTokenizer, CLIPTextModel
        print("Loading CLIP for text embedding...")
        # Using standard SD v1.5 CLIP or similar (openai/clip-vit-large-patch14)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
        tokens = tokenizer([prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        with torch.no_grad():
             text_emb = text_encoder(tokens.input_ids.to(device))[0]
        return text_emb
    except Exception as e:
        print(f"CLIP Loading failed ({e}). Using Zero Embedding.")
        return torch.zeros(1, 77, 768).to(device)

def save_gif(gt, coarse, refined, index, prompt, output_dir):
    # gt, coarse, refined: (1, 24, 3, 64, 64)
    # Extract Velocity Magnitude: sqrt(u^2 + v^2)
    # Channel 0=u, 1=v
    
    def get_mag(x):
        x_np = x[0].cpu().numpy()
        u = x_np[:, 0]
        v = x_np[:, 1]
        return np.sqrt(u**2 + v**2)
    
    mag_gt = get_mag(gt)
    mag_coarse = get_mag(coarse)
    mag_refined = get_mag(refined)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    vmin, vmax = mag_gt.min(), mag_gt.max()
    
    im0 = axes[0].imshow(mag_gt[0], vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    im1 = axes[1].imshow(mag_coarse[0], vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1].set_title("Coarse Input (First Frame)")
    axes[1].axis('off')
    
    im2 = axes[2].imshow(mag_refined[0], vmin=vmin, vmax=vmax, cmap='viridis')
    axes[2].set_title(f"Refined (DiT)")
    axes[2].axis('off')
    
    fig.suptitle(f"Sample {index} | Prompt: '{prompt}'")
    
    def update(frame):
        im0.set_data(mag_gt[frame])
        im1.set_data(mag_coarse[frame])
        im2.set_data(mag_refined[frame])
        return im0, im1, im2
    
    ani = animation.FuncAnimation(fig, update, frames=len(mag_gt), interval=200, blit=True)
    
    out_path = os.path.join(output_dir, f"sample_{index}_refined.gif")
    print(f"Saving GIF to {out_path}...")
    ani.save(out_path, writer='pillow', fps=5)
    print("Done.")

def load_ae_weights(model, path):
    # Same logic as refine_fno.py
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if 'state_dict' in sd: sd = sd['state_dict']
    new_sd = {}
    for k, v in sd.items():
        if 'encoder.gino_encoder.x_projection.fcs.0' in k:
            if 'weight' in k: new_sd['proj.weight'] = v
            elif 'bias' in k: new_sd['proj.bias'] = v
            continue 
        if 'encoder.cnn_encoder' in k:
            name = k.replace('encoder.cnn_encoder.', 'encoder.')
            if 'down.0' in name: name = name.replace('down.0.block', 'down_0').replace('down.0.downsample', 'down_0_ds')
            elif 'down.1' in name: name = name.replace('down.1.block', 'down_1').replace('down.1.downsample', 'down_1_ds')
            elif 'down.2' in name: name = name.replace('down.2.', 'down_2_').replace('block.', 'block_').replace('attn.', 'attn_')
            elif 'mid.block' in name: name = name.replace('mid.block', 'mid_block')
            elif 'mid.attn' in name: name = name.replace('mid.attn', 'mid_attn')
            new_sd[name] = v
        elif 'decoder.cnn_decoder' in k:
            name = k.replace('decoder.cnn_decoder.', 'decoder.')
            if 'up.2' in name: name = name.replace('up.2.block', 'up_2').replace('up.2.upsample', 'up_2.3').replace('up.2.attn', 'up_2_attns')
            elif 'up.1' in name: name = name.replace('up.1.block', 'up_1').replace('up.1.upsample', 'up_1.3')
            elif 'up.0' in name: name = name.replace('up.0.block', 'up_0')
            elif 'mid.block' in name: name = name.replace('mid.block', 'mid_block')
            elif 'mid.attn' in name: name = name.replace('mid.attn', 'mid_attn')
            new_sd[name] = v
    model.load_state_dict(new_sd, strict=False)

if __name__ == "__main__":
    main()
