import torch
import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.fno import RecurrentFNO

DATA_FILE = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
OUTPUT_PLOT = "d:/Semester 6/Natural Language Processing/Project 3/fno_verification.png"
MODEL_PATH = "d:/Semester 6/Natural Language Processing/Project 3/fno_checkpoint.pth"

class CylinderDataset(Dataset):
    def __init__(self, file_path, num_samples=None):
        self.file_path = file_path
        with h5py.File(file_path, 'r') as f:
            self.keys = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)
        if num_samples:
            self.keys = self.keys[:num_samples]
            
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        with h5py.File(self.file_path, 'r') as f:
            grp = f[key]
            grid = grp['grid'][:] # (25, 3, 64, 64)
            re = grp['reynolds_number'][()]
        return torch.tensor(grid, dtype=torch.float32), torch.tensor(re, dtype=torch.float32)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Train logic (Short run to establish functionality)
    dataset = CylinderDataset(DATA_FILE, num_samples=50) # Small subset
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = RecurrentFNO(modes=12, width=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    print("Training FNO (Demonstration)...")
    epochs = 10
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for grid_seq, re in loader:
            grid_seq = grid_seq.to(device)
            re = re.to(device)
            
            # Predict t=1 from t=0
            state_t0 = grid_seq[:, 0]
            target_t1 = grid_seq[:, 1]
            
            pred_t1 = model(state_t0[:, 0:1], state_t0[:, 1:2], state_t0[:, 2:3], re)
            loss = criterion(pred_t1, target_t1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.5f}")
        
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # 2. Visualize
    print("Generating verification plot...")
    model.eval()
    
    # Get one sample
    grid_seq, re = dataset[0]
    grid_seq = grid_seq.unsqueeze(0).to(device) # (1, 25, 3, 64, 64)
    re = re.unsqueeze(0).to(device)
    
    state_t0 = grid_seq[:, 0]
    target_t1 = grid_seq[:, 1]
    
    with torch.no_grad():
        pred_t1 = model(state_t0[:, 0:1], state_t0[:, 1:2], state_t0[:, 2:3], re)
        
    # Plot Magnitude
    def get_mag(x):
        # x: (1, 3, 64, 64) -> (64, 64)
        u = x[0, 0].cpu().numpy()
        v = x[0, 1].cpu().numpy()
        return np.sqrt(u**2 + v**2)
    
    mag_Input = get_mag(state_t0)
    mag_GT = get_mag(target_t1)
    mag_Pred = get_mag(pred_t1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    vmin, vmax = mag_GT.min(), mag_GT.max()
    
    # Input
    im0 = axes[0, 0].imshow(mag_Input, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Input (t=0)")
    axes[0, 0].axis('off')
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # GT
    im1 = axes[0, 1].imshow(mag_GT, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Ground Truth (t=1)")
    axes[0, 1].axis('off')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Pred
    im2 = axes[1, 0].imshow(mag_Pred, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("FNO Prediction (t=1)")
    axes[1, 0].axis('off')
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Loss
    axes[1, 1].plot(loss_history, marker='o')
    axes[1, 1].set_title("Training Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MSE Loss")
    axes[1, 1].grid(True)
    
    plt.suptitle(f"FNO Verification (Re={re[0].item():.1f})", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
