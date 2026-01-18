import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

INPUT_FILE = "d:/Semester 6/Natural Language Processing/Project 3/train_downsampled_labeled.h5"
GRID_SIZE = 64
SIGMA = 0.05

def test_single():
    print("Testing interpolation on sample '0'...")
    with h5py.File(INPUT_FILE, 'r') as f:
        grp = f['0']
        mesh_pos = grp['mesh_pos'][:]
        u = grp['u'][0] # T=0
        v = grp['v'][0]
        mag = np.sqrt(u**2 + v**2)
        
        meta = grp['metadata']
        domain_x = meta['domain_x'][:]
        domain_y = meta['domain_y'][:]
        
        # Create Grid
        x = np.linspace(domain_x[0], domain_x[1], GRID_SIZE)
        y = np.linspace(domain_y[0], domain_y[1], GRID_SIZE)
        grid_x, grid_y = np.meshgrid(x, y)
        
        # Interpolate
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        dists = cdist(grid_points, mesh_pos)
        weights = np.exp(- (dists ** 2) / (SIGMA ** 2))
        weight_sum = weights.sum(axis=1)
        weight_sum[weight_sum == 0] = 1.0
        
        interp_flat = (weights @ mag) / weight_sum
        interp_grid = interp_flat.reshape(GRID_SIZE, GRID_SIZE)
        
        # Plot
        plt.figure(figsize=(6, 5))
        plt.imshow(interp_grid, origin='lower', extent=[domain_x[0], domain_x[1], domain_y[0], domain_y[1]], cmap='viridis')
        plt.colorbar(label='Velocity Mag')
        plt.title("Test Interpolation (Single Sample)")
        plt.savefig("d:/Semester 6/Natural Language Processing/Project 3/single_test_interp.png")
        print("Saved test plot to single_test_interp.png")

if __name__ == "__main__":
    test_single()
