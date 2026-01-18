import torch
import sys
import types


CKPT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/ae_cylinder.ckpt"

if __name__ == "__main__":
    # Checkpoint path
    checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    state_dict = checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Found 'state_dict' key.")
        
    print("\n--- Model Layers ---")
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")
