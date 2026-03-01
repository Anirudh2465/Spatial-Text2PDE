import h5py
import numpy as np
import os

file_path = r'd:\Semester 6\Natural Language Processing\Project 3\train_downsampled_labeled.h5'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

def print_dataset_info(name, data):
    print(f"  Dataset: {name}")
    print(f"    Shape: {data.shape}")
    print(f"    Type: {data.dtype}")
    
    val = data[()]
    if data.size < 20:
        if isinstance(val, bytes):
            print(f"    Value: {val.decode('utf-8')}")
        else:
            print(f"    Value: {val}")
    else:
        if np.issubdtype(data.dtype, np.number):
            print(f"    Stats - Min: {np.min(val):.4f}, Max: {np.max(val):.4f}, Mean: {np.mean(val):.4f}")
            # Print first few elements flattened
            flat = val.flatten()
            print(f"    First 5: {flat[:5]}")
        else:
            print("    (Large non-numeric dataset)")

with h5py.File(file_path, 'r') as f:
    if '0' not in f:
        print("Sample '0' not found in file.")
        exit(1)
        
    sample = f['0']
    print(f"Inspecting sample '0' in {file_path}")
    
    # Sort keys for consistent output
    keys = sorted(sample.keys())
    
    for key in keys:
        item = sample[key]
        if isinstance(item, h5py.Group):
            print(f"\n[Group] {key}")
            subkeys = sorted(item.keys())
            for subkey in subkeys:
                subitem = item[subkey]
                if isinstance(subitem, h5py.Dataset):
                    print_dataset_info(subkey, subitem)
        elif isinstance(item, h5py.Dataset):
            print(f"\n[Dataset] {key}")
            print_dataset_info(key, item)
