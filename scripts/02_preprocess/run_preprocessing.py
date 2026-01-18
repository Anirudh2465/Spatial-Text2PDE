import h5py
import numpy as np
import math
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.interpolation import kernel_interpolate

INPUT_FILE = "d:/Semester 6/Natural Language Processing/Project 3/train_downsampled_labeled.h5"
OUTPUT_FILE = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
GRID_SIZE = 64
SIGMA = 0.05
NUM_TIMESTEPS = 25

def process_single_sample(key):
    """
    Worker function to process a single sample key.
    Reads from HDF5 independently (safe for concurrent reads in SWMR or simple read mode usually).
    Returns (key, grid_tensor, prompt_str, reynolds)
    """
    try:
        # Re-open file in each process to avoid pickling lock issues
        with h5py.File(INPUT_FILE, 'r') as f:
            grp = f[key]
            
            # Metadata
            meta = grp['metadata']
            domain_x = meta['domain_x'][:]
            domain_y = meta['domain_y'][:]
            prompt_bytes = meta['prompt'][()]
            prompt = prompt_bytes.decode('utf-8') if isinstance(prompt_bytes, bytes) else str(prompt_bytes)
            reynolds = meta['reynolds_number'][()]
            
            # Grid Setup
            x = np.linspace(domain_x[0], domain_x[1], GRID_SIZE)
            y = np.linspace(domain_y[0], domain_y[1], GRID_SIZE)
            grid_x, grid_y = np.meshgrid(x, y)
            
            mesh_pos = grp['mesh_pos'][:]
            
            # Temporal Downsample Indices
            total_steps = grp['u'].shape[0]
            indices = np.linspace(0, total_steps - 1, NUM_TIMESTEPS, dtype=int)
            
            # Read Data
            u_seq = grp['u'][indices]
            v_seq = grp['v'][indices]
            p_raw = grp['pressure'][indices]
            if p_raw.ndim == 3: p_raw = p_raw[..., 0]
            p_seq = p_raw
            
            # Vectorized Interpolation Logic
            H, W = GRID_SIZE, GRID_SIZE
            grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1) # (H*W, 2)
            dists = cdist(grid_points, mesh_pos)
            
            weights = np.exp(- (dists ** 2) / (SIGMA ** 2))
            weight_sum = weights.sum(axis=1, keepdims=True)
            weight_sum[weight_sum == 0] = 1.0
            norm_weights = weights / weight_sum
            
            # Stack (T, 3, N)
            data_block = np.stack([u_seq, v_seq, p_seq], axis=1)
            # Flatten to (T*3, N) or similar. Let's do (25, 3, N) -> (75, N)
            data_flat = data_block.reshape(-1, mesh_pos.shape[0])
            
            # Interpolate: (H*W, N) @ (N, 75) -> (H*W, 75)
            interp_flat = norm_weights @ data_flat.T
            
            # Reshape back: (H*W, 75) -> (H, W, 25, 3)
            interp_reshaped = interp_flat.reshape(H, W, NUM_TIMESTEPS, 3)
            
            # Transpose to (25, 3, H, W)
            grid_tensor = interp_reshaped.transpose(2, 3, 0, 1).astype(np.float32)
            
            return key, grid_tensor, prompt, reynolds
            
    except Exception as e:
        return key, None, str(e), None

def main():
    start_time = time.time()
    
    # Get keys first
    with h5py.File(INPUT_FILE, 'r') as f:
        keys = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)
        print(f"Total samples to process: {len(keys)}")
    
    # Prepare output file
    mode = 'w'
    with h5py.File(OUTPUT_FILE, mode) as f_out:
        pass # Create/Truncate
        
    # Parallel Processing
    # Adjust max_workers based on system. usually os.cpu_count()
    max_workers = min(os.cpu_count() or 4, 16) 
    print(f"Starting pool with {max_workers} processed...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_key = {executor.submit(process_single_sample, key): key for key in keys}
        
        # Write results as they arrive
        with h5py.File(OUTPUT_FILE, 'r+') as f_out:
            for i, future in enumerate(as_completed(future_to_key)):
                key, tensor, prompt, reynolds = future.result()
                
                if tensor is not None:
                    grp = f_out.create_group(key)
                    grp.create_dataset('grid', data=tensor, compression='gzip')
                    grp.create_dataset('prompt', data=prompt)
                    grp.create_dataset('reynolds_number', data=reynolds)
                    
                    if i % 10 == 0:
                        print(f"Processed {i+1}/{len(keys)} (Last: {key})")
                else:
                    print(f"Failed {key}: {prompt}")
                    
    print(f"Done. File saved to {OUTPUT_FILE}")
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
