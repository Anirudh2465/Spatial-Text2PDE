import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import numpy as np
from data_loader import load_simulation_data

FILE_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_downsampled_labeled.h5"
OUTPUT_FILE = "d:/Semester 6/Natural Language Processing/Project 3/simulation_series.gif"

def generate_video():
    # Load just one sample
    loader = load_simulation_data(FILE_PATH, num_timesteps=25)
    sample = next(loader)
    
    print(f"Animating sample: {sample['key']}")
    
    mesh_pos = sample['mesh_pos']
    cells = sample['cells']
    u = sample['u']
    v = sample['v']
    
    triangulation = tri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], cells)
    velocity_mag = np.sqrt(u**2 + v**2)
    vmin, vmax = velocity_mag.min(), velocity_mag.max()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initialize plot
    tripcolor = ax.tripcolor(triangulation, velocity_mag[0], cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(tripcolor, ax=ax, label='Velocity Magnitude')
    ax.set_aspect('equal')
    ax.axis('off')
    
    title = ax.set_title(f"Time: 0")
    
    def update(frame):
        ax.clear()
        ax.axis('off')
        ax.set_aspect('equal')
        tpc = ax.tripcolor(triangulation, velocity_mag[frame], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"Time: {frame}")
        return tpc,
    
    ani = animation.FuncAnimation(fig, update, frames=25, interval=200, blit=False)
    
    print("Saving GIF (this might take a moment)...")
    ani.save(OUTPUT_FILE, writer='pillow', fps=5)
    print(f"Saved animation to {OUTPUT_FILE}")
    plt.close()

if __name__ == "__main__":
    generate_video()
