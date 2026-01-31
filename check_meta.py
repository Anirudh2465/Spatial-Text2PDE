import h5py

DATA_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"

def check():
    with h5py.File(DATA_PATH, 'r') as f:
        keys = list(f.keys())
        print(f"Total keys: {len(keys)}")
        k = keys[0]
        print(f"Checking key: {k}")
        grp = f[k]
        print("Datasets in group:", list(grp.keys()))
        
        if 'reynolds_number' in grp:
            print("Reynolds Number:", grp['reynolds_number'][()])
        
        if 'prompt' in grp:
            print("Prompt:", grp['prompt'][()])

if __name__ == "__main__":
    check()
