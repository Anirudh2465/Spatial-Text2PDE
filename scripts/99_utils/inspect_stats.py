import pickle
import torch
import numpy as np

FILES = ["d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl", 
         "d:/Semester 6/Natural Language Processing/Project 3/train_minmax_stat.pkl"]

def inspect():
    for fpath in FILES:
        print(f"--- Inspecting {fpath} ---")
        try:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            print(f"Type: {type(data)}")
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"Key: {k}, Type: {type(v)}")
                    if hasattr(v, 'shape'):
                        print(f"  Shape: {v.shape}")
                    print(f"  Value: {v}")
            else:
                print(f"Value: {data}")
        except Exception as e:
            print(f"Error: {e}")
        print("\n")

if __name__ == "__main__":
    inspect()
