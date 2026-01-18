import h5py
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from interpolation import kernel_interpolate

INPUT_FILE = "d:/Semester 6/Natural Language Processing/Project 3/train_downsampled_labeled.h5"
OUTPUT_FILE = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
GRID_SIZE = 64
SIGMA = 0.05 # Tunable parameter for smoothness

def preprocess():
    print(f"Opening {INPUT_FILE}...")
    with h5py.File(INPUT_FILE, 'r') as f_in, h5py.File(OUTPUT_FILE, 'w') as f_out:
        keys = sorted(list(f_in.keys()), key=lambda x: int(x) if x.isdigit() else x)
        
        # Use tqdm for progress bar
        for key in tqdm(keys, desc="Processing samples"):
            try:
                grp_in = f_in[key]
                
                # Create output group
                grp_out = f_out.create_group(key)
                
                # Read metadata for domain bounds
                meta = grp_in['metadata']
                domain_x = meta['domain_x'][:]
                domain_y = meta['domain_y'][:]
                
                # Check bounds (assuming constant for robustness, but reading per sample)
                min_x, max_x = domain_x[0], domain_x[1]
                min_y, max_y = domain_y[0], domain_y[1]
                
                # Create Grid
                x = np.linspace(min_x, max_x, GRID_SIZE)
                y = np.linspace(min_y, max_y, GRID_SIZE)
                grid_x, grid_y = np.meshgrid(x, y)
                
                # Read mesh and data
                mesh_pos = grp_in['mesh_pos'][:]
                
                # Perform downsampling to 25 steps (same logic as data_loader)
                # This ensures consistent preprocessing
                num_timesteps = 25
                time_indices = np.linspace(0, grp_in['u'].shape[0] - 1, num_timesteps, dtype=int)
                
                u_seq = grp_in['u'][time_indices]
                v_seq = grp_in['v'][time_indices]
                
                p_raw = grp_in['pressure'][time_indices]
                if p_raw.ndim == 3: p_raw = p_raw[..., 0]
                p_seq = p_raw
                
                # Interpolation Optimization
                # Compute weights once for this mesh because grid and mesh are static for the sample
                # Flatten the grid for vectorized distance calculation
                # mesh_pos: (N, 2)
                # grid points: (H*W, 2)
                H, W = GRID_SIZE, GRID_SIZE
                grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1) # (H*W, 2)
                dists = cdist(grid_points, mesh_pos) # (H*W, N)
                
                weights = np.exp(- (dists ** 2) / (SIGMA ** 2))
                weight_sum = weights.sum(axis=1, keepdims=True)
                weight_sum[weight_sum == 0] = 1.0
                norm_weights = weights / weight_sum # (H*W, N)
                
                # Stack all data to shape (T*3, N) or similar to do single matmul
                # u_seq: (25, N), v_seq: (25, N), p_seq: (25, N)
                # Stack to (25, 3, N)
                data_block = np.stack([u_seq, v_seq, p_seq], axis=1) # (25, 3, N)
                data_flat = data_block.reshape(-1, mesh_pos.shape[0]) # (75, N)
                
                # Interpolate: (H*W, N) @ (N, 75) -> (H*W, 75)
                # Transpose data_flat to (N, 75)
                interp_flat = norm_weights @ data_flat.T # (H*W, 75)
                
                # Reshape back to (H, W, 25, 3) then transpose to (25, 3, H, W)
                interp_reshaped = interp_flat.reshape(H, W, num_timesteps, 3)
                # Current: H, W, T, C. Target: T, C, H, W
                grid_tensor = interp_reshaped.transpose(2, 3, 0, 1)
                
                # Save datasets
                grp_out.create_dataset('grid', data=grid_tensor, compression='gzip')
                
                # Copy prompt
                grp_out.create_dataset('prompt', data=meta['prompt'][()])
                
                # Copy other useful metadata just in case
                grp_out.create_dataset('reynolds_number', data=meta['reynolds_number'][()])
                
            except Exception as e:
                print(f"\nFailed on key {key}: {e}")
                
    print(f"\nPreprocessing complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess()
