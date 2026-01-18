import torch
import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.fno import RecurrentFNO

DATA_FILE = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"

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
            
        # Standardize? For now raw values.
        # Returns: input=(u,v,p)_t, Re -> target=(u,v,p)_{t+1}
        # We model single step dynamics here for testing
        # Or better: return sequence
        return torch.tensor(grid, dtype=torch.float32), torch.tensor(re, dtype=torch.float32)

def train_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load limited data for quick test
    dataset = CylinderDataset(DATA_FILE, num_samples=100)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = RecurrentFNO(modes=8, width=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Starting training loop...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        baseline_loss = 0
        
        for grid_seq, re in loader:
            grid_seq = grid_seq.to(device) # (B, 25, 3, 64, 64)
            re = re.to(device)
            
            # Predict t=1 from t=0
            state_t0 = grid_seq[:, 0] # (B, 3, H, W)
            target_t1 = grid_seq[:, 1]
            
            # Forward
            # Unpack u, v, p channels
            pred_t1 = model(state_t0[:, 0:1], state_t0[:, 1:2], state_t0[:, 2:3], re)
            
            loss = criterion(pred_t1, target_t1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Identity Baseline: Prediction = Previous Frame
            base_loss = criterion(state_t0, target_t1)
            baseline_loss += base_loss.item()
            
        avg_loss = total_loss / len(loader)
        avg_base = baseline_loss / len(loader)
        ratio = avg_loss / avg_base if avg_base > 0 else 0
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f} | Baseline = {avg_base:.6f} | Ratio = {ratio:.2f}")

    if avg_loss < avg_base:
        print("\nSUCCESS: Model outperformed identity baseline.")
    else:
        print("\nWARNING: Model did not beat identity baseline (more training/data needed).")

if __name__ == "__main__":
    train_test()
