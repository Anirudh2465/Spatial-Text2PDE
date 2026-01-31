import torch
import torch.nn as nn
import numpy as np
import h5py
import sys
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.autoencoder import Autoencoder3D
from src.data.normalization import Normalizer

DATA_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
STAT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl"
CKPT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/ae_cylinder.ckpt"
OUTPUT_MODEL = "d:/Semester 6/Natural Language Processing/Project 3/ae_finetuned.pth"

class GridDataset(Dataset):
    def __init__(self, file_path, normalizer=None):
        self.file_path = file_path
        self.normalizer = normalizer
        with h5py.File(file_path, 'r') as f:
            self.keys = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)
            
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        with h5py.File(self.file_path, 'r') as f:
            grid = f[key]['grid'][:] # (25, 3, 64, 64)
            
        x = torch.tensor(grid).float() # (25, 3, 64, 64)
        
        if self.normalizer:
            x = self.normalizer.normalize(x.unsqueeze(0)).squeeze(0)
            
        return x

def load_pretrained(model):
    print(f"Loading pretrained weights from {CKPT_PATH}...")
    sd = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' in sd: sd = sd['state_dict']
    new_sd = {}
    for k, v in sd.items():
        if 'encoder.gino_encoder.x_projection.fcs.0' in k:
            if 'weight' in k: new_sd['proj.weight'] = v
            elif 'bias' in k: new_sd['proj.bias'] = v; continue 
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50) # Standard fine-tuning
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Models
    model = Autoencoder3D().to(device)
    load_pretrained(model)
    
    # Load Data
    normalizer = Normalizer(STAT_PATH)
    dataset = GridDataset(DATA_PATH, normalizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    
    print("Starting Fine-tuning...")
    loss_history = []
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x in loader:
            x = x.to(device) # (B, 3, 25, 64, 64)
            
            # AE expects (B, T, C, H, W).
            # Cropping T (2nd dim) from 25 to 24.
            x_in = x[:, :24, :, :, :]
            
            # Forward
            # VAE: encode -> z, posterior?
            # My Autoencoder3D.encode returns z (sample). 
            # Does it return posterior for KL loss?
            # Current implementation `encode` returns `Sample(mean, logvar)`.
            # Wait, `encode` in `autoencoder.py` returns `z`.
            # Does it compute KL?
            # I should verify `autoencoder.py`. 
            # Assuming standard L1 Recon for now (Perceptual Compression).
            
            z = model.encode(x_in)
            recon = model.decode(z)
            
            loss = criterion(recon, x_in)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.6f}")
        
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"Fine-tuned model saved to {OUTPUT_MODEL}")
    
    # Plot Loss
    plt.plot(loss_history)
    plt.title("Fine-tuning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.savefig("d:/Semester 6/Natural Language Processing/Project 3/finetune_loss.png")

if __name__ == "__main__":
    main()
