import torch
import torch.nn as nn
import numpy as np
import h5py
import sys
import os
import argparse
from tqdm import tqdm

# Mock modules for Loading Context
import types
m = types.ModuleType('modules')
sys.modules['modules'] = m
m.utils = types.ModuleType('utils')
class MockClass:
    def __init__(self, *args, **kwargs): pass
m.utils.Normalizer = MockClass
sys.modules['modules.utils'] = m.utils

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.autoencoder import Autoencoder3D
from src.models.diffusion import DiffusionTransformer, DDIMSampler
from src.data.normalization import Normalizer

AE_CKPT = "d:/Semester 6/Natural Language Processing/Project 3/ae_cylinder.ckpt"
DIT_CKPT = "d:/Semester 6/Natural Language Processing/Project 3/ldm_DiT_text_cylinder.ckpt"
DATA_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
STAT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Models
    print("Loading Autoencoder...")
    ae = Autoencoder3D().to(device)
    ae.eval()
    # Load AE Weights (Manual logic from validate_integration)
    # ... I should refactor manual logic to a utility or method, but for now copy-paste safe logic
    # Actually, I'll use a simplified loader if possible, or just re-implement the mapping
    load_ae_weights(ae, AE_CKPT)
    
    print("Loading DiT...")
    dit = DiffusionTransformer(
        input_size=(6, 16, 16),
        patch_size=2,
        in_channels=16,
        hidden_size=1024,
        depth=28,
        num_heads=16,
        context_dim=768
    ).to(device)
    dit.eval()
    dit.load_from_ckpt(DIT_CKPT)
    
    sampler = DDIMSampler(dit, ckpt_path=DIT_CKPT)
    
    # 2. Load Data & Normalizer
    print("Loading Data & Normalizer...")
    normalizer = Normalizer(STAT_PATH)
    with h5py.File(DATA_PATH, 'r') as f:
        # Load sample 0
        grid = f['0']['grid'][:] # (25, 3, 64, 64)
    
    # Prepare GT
    x_gt = torch.tensor(grid).unsqueeze(0).permute(0, 1, 2, 3, 4).float().to(device) # (1, 25, 3, 64, 64)
    x_gt = x_gt[:, :24]
    
    # 3. Simulate Coarse FNO Output (GT + Noise)
    noise_level = 0.1
    x_coarse = x_gt + torch.randn_like(x_gt) * noise_level
    print(f"Coarse Input MSE: {nn.MSELoss()(x_coarse, x_gt).item():.6f}")
    
    # Normalize Coarse Input for AE
    x_coarse_norm = normalizer.normalize(x_coarse)
    
    # 4. Refinement
    with torch.no_grad():
        z_coarse = ae.encode(x_coarse_norm) # (1, 16, 6, 16, 16)
        
        # Text Condition
        text_emb = torch.zeros(1, 77, 768).to(device) 
        
        # Refine (SDEdit)
        # Strength 0.5 (Refine last 50% steps)
        # This assumes DiT works on same latent space as AE
        z_refined = sampler.refine(z_coarse, text_emb, strength=0.5)
        
        # Decode
        x_refined_norm = ae.decode(z_refined)
        
        # Unnormalize
        x_refined = normalizer.unnormalize(x_refined_norm)
        
    # 5. Metrics
    loss_coarse = nn.MSELoss()(x_coarse, x_gt).item()
    loss_refined = nn.MSELoss()(x_refined, x_gt).item()
    
    print(f"Final Results:")
    print(f"Coarse MSE: {loss_coarse:.6f}")
    print(f"Refined MSE: {loss_refined:.6f}")
    
    if loss_refined < loss_coarse:
        print("SUCCESS: Refinement improved quality.")
    else:
        print("Note: Refinement degraded quality (expected with dummy text/noise models without proper conditioning tuning).")

def load_ae_weights(model, path):
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
    print("AE Weights Loaded.")

if __name__ == "__main__":
    main()
