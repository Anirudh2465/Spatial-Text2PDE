import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

GRID_FILE = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
ORIG_FILE = "d:/Semester 6/Natural Language Processing/Project 3/train_downsampled_labeled.h5"
OUTPUT_IMG = "d:/Semester 6/Natural Language Processing/Project 3/interpolation_check.png"

def verify():
    # Load sample '0' from both
    with h5py.File(GRID_FILE, 'r') as f_grid, h5py.File(ORIG_FILE, 'r') as f_orig:
        if '0' not in f_grid:
            print("Sample '0' not found in grid file yet.")
            return

        # Grid Data
        grid_tensor = f_grid['0']['grid'][:] # (25, 3, 64, 64)
        print(f"Grid Tensor Shape: {grid_tensor.shape}")
        
        # Original Data (timestep 0)
        orig_grp = f_orig['0']
        mesh_pos = orig_grp['mesh_pos'][:]
        cells = orig_grp['cells'][:]
        u_orig = orig_grp['u'][0] # T=0
        v_orig = orig_grp['v'][0]
        mag_orig = np.sqrt(u_orig**2 + v_orig**2)
        
        # Grid Data (timestep 0)
        u_grid = grid_tensor[0, 0]
        v_grid = grid_tensor[0, 1]
        mag_grid = np.sqrt(u_grid**2 + v_grid**2)
        
        # Visualization Comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original Mesh Plot
        triang = tri.Triangulation(mesh_pos[:,0], mesh_pos[:,1], cells)
        t = axes[0].tripcolor(triang, mag_orig, shading='flat', cmap='viridis')
        axes[0].set_title("Original Mesh (T=0)")
        plt.colorbar(t, ax=axes[0])
        axes[0].set_aspect('equal')
        
        # Grid Interpolated Plot
        # Origin lower is important if y was linspace(min, max)
        im = axes[1].imshow(mag_grid, origin='lower', extent=[-1, 2, -1, 1], cmap='viridis') # Extent is approx from visual
        # Note: Exact extent depends on metadata, simpler here to just show pixel map
        # axes[1].imshow(mag_grid, origin='lower', cmap='viridis')
        axes[1].set_title("Interpolated Grid 64x64 (T=0)")
        plt.colorbar(im, ax=axes[1])
        
        plt.suptitle("Mesh vs Grid Interpolation Check")
        plt.savefig(OUTPUT_IMG)
        print(f"Saved comparison to {OUTPUT_IMG}")
        plt.close()

if __name__ == "__main__":
    verify()
