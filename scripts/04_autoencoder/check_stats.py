import h5py
import numpy as np

DATA_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"

def check():
    with h5py.File(DATA_PATH, 'r') as f:
        grid = f['0']['grid'][:] # (25, 3, 64, 64)
        print(f"Min: {grid.min()}, Max: {grid.max()}, Mean: {grid.mean()}")
        for c in range(3):
            print(f"Ch {c}: Min {grid[:,c].min():.4f} Max {grid[:,c].max():.4f} Mean {grid[:,c].mean():.4f}")

if __name__ == "__main__":
    check()
