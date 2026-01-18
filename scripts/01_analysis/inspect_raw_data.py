import h5py
import sys

file_path = "d:/Semester 6/Natural Language Processing/Project 3/train_downsampled_labeled.h5"
output_file = "d:/Semester 6/Natural Language Processing/Project 3/inspection_log.txt"

def inspect():
    with open(output_file, 'w', encoding='utf-8') as log:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                log.write(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}\n")
            elif isinstance(obj, h5py.Group):
                log.write(f"Group: {name}\n")

        try:
            with h5py.File(file_path, 'r') as f:
                log.write(f"Root keys: {list(f.keys())[:5]} ... (total {len(f.keys())})\n")
                
                # Inspect the first sample '0'
                if '0' in f:
                    log.write("\n--- Structure of sample '0' ---\n")
                    f['0'].visititems(print_structure)
                    
                    # Check for attributes if any
                    log.write("\n--- Attributes of sample '0' ---\n")
                    for k, v in f['0'].attrs.items():
                        log.write(f"{k}: {v}\n")
                        
        except Exception as e:
            log.write(f"Error: {e}\n")

if __name__ == "__main__":
    inspect()
