import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.loader import load_simulation_data

FILE_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_downsampled_labeled.h5"

def verify():
    print("Starting verification...")
    loader = load_simulation_data(FILE_PATH, num_timesteps=25)
    
    # Check first 5 samples
    for i, data in enumerate(loader):
        if i >= 5:
            break
            
        key = data['key']
        num_nodes = data['mesh_pos'].shape[0]
        
        print(f"Verifying sample {key}, Nodes: {num_nodes}")
        
        # Verify shapes
        assert data['u'].shape == (25, num_nodes), f"u shape mismatch: {data['u'].shape}"
        assert data['v'].shape == (25, num_nodes), f"v shape mismatch: {data['v'].shape}"
        assert data['p'].shape == (25, num_nodes), f"p shape mismatch: {data['p'].shape}"
        assert data['mesh_pos'].shape == (num_nodes, 2), "mesh_pos shape mismatch"
        
        # Verify node_type
        # Expected to be integer, maybe 1-hot or class index. Inspection showed (num_nodes, 1) int32.
        assert data['node_type'].ndim >= 1
        
        # Verify prompt
        assert isinstance(data['prompt'], str) and len(data['prompt']) > 0, "Prompt decoding failed"
        
        print(f"  > Shapes OK.")
        print(f"  > Prompt: {data['prompt']}")
        print(f"  > u range: [{data['u'].min():.4f}, {data['u'].max():.4f}]")
        
    print("\nVerification passed for first 5 samples.")

if __name__ == "__main__":
    verify()
