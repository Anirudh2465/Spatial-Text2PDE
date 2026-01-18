import h5py
import numpy as np

def load_simulation_data(file_path, num_timesteps=25):
    """
    Generator that yields simulation data from the HDF5 file.
    
    Args:
        file_path (str): Path to the HDF5 file.
        num_timesteps (int): Number of timesteps to extract. 
                             Assumes input has 100 steps and we want to downsample evenly.
                             
    Yields:
        dict: Dictionary containing mesh and physics fields.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            keys = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)
            
            for key in keys:
                group = f[key]
                
                # Read static mesh data
                mesh_pos = group['mesh_pos'][:]
                node_type = group['node_type'][:]
                cells = group['cells'][:] # Connectivity for triangulation
                
                # Determine stride for temporal downsampling
                # Default is 100 steps in file (based on inspection)
                total_steps = group['u'].shape[0]
                if total_steps >= num_timesteps:
                    indices = np.linspace(0, total_steps - 1, num_timesteps, dtype=int)
                else:
                    # If fewer steps than requested, take what we have (or handle error)
                    # For now, just taking all if less
                    indices = np.arange(total_steps)
                
                # Read temporal data with striding
                u = group['u'][indices]
                v = group['v'][indices]
                
                # Pressure might have an extra channel dim based on inspection: (100, 1876, 1)
                p = group['pressure'][indices]
                if p.ndim == 3 and p.shape[2] == 1:
                    p = p[..., 0] # Squeeze the last dim if it exists
                
                # Read metadata
                prompt_bytes = group['metadata']['prompt'][()]
                if isinstance(prompt_bytes, bytes):
                    prompt = prompt_bytes.decode('utf-8')
                else:
                    prompt = str(prompt_bytes)
                    
                yield {
                    'key': key,
                    'mesh_pos': mesh_pos,   # (num_nodes, 2)
                    'u': u,                 # (25, num_nodes)
                    'v': v,                 # (25, num_nodes)
                    'p': p,                 # (25, num_nodes)
                    'node_type': node_type, # (num_nodes, 1)
                    'cells': cells,         # (num_cells, 3)
                    'prompt': prompt
                }
                
    except Exception as e:
        print(f"Error reading file: {e}")
        raise

if __name__ == "__main__":
    # Quick test if run directly
    loader = load_simulation_data("d:/Semester 6/Natural Language Processing/Project 3/train_downsampled_labeled.h5")
    sample = next(loader)
    print(f"Loaded sample {sample['key']}")
    print(f"u shape: {sample['u'].shape}")
    print(f"Prompt: {sample['prompt']}")
