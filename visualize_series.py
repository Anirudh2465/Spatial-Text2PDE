import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from data_loader import load_simulation_data

FILE_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_downsampled_labeled.h5"
OUTPUT_FILE = "d:/Semester 6/Natural Language Processing/Project 3/time_series_visualization.png"

def plot_time_series():
    # Load just one sample
    loader = load_simulation_data(FILE_PATH, num_timesteps=25)
    sample = next(loader)
    
    print(f"Visualizing sample: {sample['key']}")
    print(f"Prompt: {sample['prompt']}")
    
    mesh_pos = sample['mesh_pos']
    cells = sample['cells']
    u = sample['u']
    v = sample['v']
    
    # Create triangulation
    triangulation = tri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], cells)
    
    # Calculate velocity magnitude for all steps
    velocity_mag = np.sqrt(u**2 + v**2)
    
    # Determine global min/max for consistent colorbar
    vmin, vmax = velocity_mag.min(), velocity_mag.max()
    
    # Setup 5x5 grid
    fig, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for t in range(25):
        ax = axes[t]
        # Use tripcolor for flat shading on triangles
        tpc = ax.tripcolor(triangulation, velocity_mag[t], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"t={t}")
        ax.axis('off')
        
    # Add a global colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(tpc, cax=cbar_ax, label='Velocity Magnitude')
    
    plt.suptitle(f"Simulation: {sample['prompt']}", fontsize=16)
    plt.savefig(OUTPUT_FILE, dpi=100, bbox_inches='tight')
    print(f"Saved visualization to {OUTPUT_FILE}")
    plt.close()

if __name__ == "__main__":
    plot_time_series()
