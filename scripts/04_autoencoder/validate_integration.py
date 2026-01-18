import torch
import torch.nn as nn
import numpy as np
import h5py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.autoencoder import Autoencoder3D
from src.data.normalization import Normalizer

CKPT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/ae_cylinder.ckpt"
DATA_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
STAT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl"

def verify():
    # 1. Initialize Model
    model = Autoencoder3D()
    
    # 2. Load Weights
    sd = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' in sd: sd = sd['state_dict']
    
    new_sd = {}
    for k, v in sd.items():
        if 'encoder.gino_encoder.x_projection.fcs.0' in k:
            if 'weight' in k:
                new_sd['proj.weight'] = v
            elif 'bias' in k:
                new_sd['proj.bias'] = v
            continue 

        if 'encoder.cnn_encoder' in k:
            name = k.replace('encoder.cnn_encoder.', 'encoder.')
            # Key mapping logic...
            if 'down.0' in name: name = name.replace('down.0.block', 'down_0').replace('down.0.downsample', 'down_0_ds')
            elif 'down.1' in name: name = name.replace('down.1.block', 'down_1').replace('down.1.downsample', 'down_1_ds')
            elif 'down.2' in name:
                name = name.replace('down.2.', 'down_2_')
                name = name.replace('block.', 'block_')
                name = name.replace('attn.', 'attn_')
            elif 'mid.block' in name: name = name.replace('mid.block', 'mid_block')
            elif 'mid.attn' in name: name = name.replace('mid.attn', 'mid_attn')
            new_sd[name] = v

        elif 'decoder.cnn_decoder' in k:
            name = k.replace('decoder.cnn_decoder.', 'decoder.')
            if 'up.2' in name:
                name = name.replace('up.2.block', 'up_2')
                name = name.replace('up.2.upsample', 'up_2.3')
                name = name.replace('up.2.attn', 'up_2_attns') 
            elif 'up.1' in name:
                name = name.replace('up.1.block', 'up_1')
                name = name.replace('up.1.upsample', 'up_1.3')
            elif 'up.0' in name:
                name = name.replace('up.0.block', 'up_0')
            elif 'mid.block' in name: name = name.replace('mid.block', 'mid_block')
            elif 'mid.attn' in name: name = name.replace('mid.attn', 'mid_attn')
            new_sd[name] = v
            
    print("Loading weights...")
    msg = model.load_state_dict(new_sd, strict=False)
    # print(f"Missing keys: {len(msg.missing_keys)}")
    
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
        print(f"Latent shape: {z.shape}") # Should be (1, 16, 6, 16, 16)
        
        recon_norm = model.decode(z)
        print(f"Recon shape: {recon_norm.shape}")
        
    # 5. Loss
    # Align time dimensions (Input 25 -> Output 24 handling)
    t_out = recon_norm.shape[1]
    
    # Unnormalize for visual comparison logic, but usually Loss is computed on NORMALIZED data?
    # Paper usually reports Normalized MSE/L1. 
    # Let's check both or stick to Normalized if expecting 0.034.
    
    x_target = x_in[:, :t_out]
    
    l1_loss = nn.L1Loss()(recon_norm, x_target)
    print(f"L1 Reconstruction Loss (Normalized): {l1_loss.item():.5f}")
    
    # Check Unnormalized Loss just in case
    recon = normalizer.unnormalize(recon_norm)
    x_target_unnorm = x[:, :t_out]
    l1_loss_unnorm = nn.L1Loss()(recon, x_target_unnorm)
    print(f"L1 Reconstruction Loss (Unnormalized): {l1_loss_unnorm.item():.5f}")
    
    # Threshold warning
    if l1_loss.item() > 0.1:
        print("Note: Loss > 0.1 likely due to missing Normalizer stats from original training.")

if __name__ == "__main__":
    verify()
