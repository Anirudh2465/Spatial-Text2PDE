import sys
import os
sys.path.append(os.getcwd())

import torch
from src.mllm.dataset import PhysicsMLLMDataset
from tokenizers import Tokenizer

def verify():
    print("=== Phase 1 Verification ===")
    
    # 1. Verify Tokenizer
    tok_path = "d:/Semester 6/Natural Language Processing/Project 3/src/mllm/mllm_tokenizer.json"
    if not os.path.exists(tok_path):
        print("X Tokenizer file not found!")
        return
        
    tokenizer = Tokenizer.from_file(tok_path)
    print("✓ Tokenizer loaded.")
    test_str = "Vortex shedding at Re=200."
    enc = tokenizer.encode(test_str)
    print(f"  Input: '{test_str}'")
    print(f"  Tokens: {enc.tokens}")
    print(f"  IDs: {enc.ids}")
    
    # Check special tokens
    special = ["<IMG>", "[SEP]", "[EOS]"]
    for s in special:
        id_ = tokenizer.token_to_id(s)
        if id_ is None:
            print(f"X Special token {s} missing!")
        else:
            print(f"✓ Special token {s} ID: {id_}")

    # 2. Verify Dataset
    print("\n--- Verifying Dataset ---")
    data_path = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
    stat_path = "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl"
    
    try:
        ds = PhysicsMLLMDataset(data_path, tok_path, stat_path)
        print(f"✓ Dataset initialized. Size: {len(ds)}")
        
        sample = ds[0]
        print("✓ Sample 0 loaded.")
        print(f"  Image Shape: {sample['image'].shape} (Expected: T, C, H, W)")
        print(f"  Input IDs: {sample['input_ids']}")
        print(f"  Generated Text: '{sample['text']}'")
        
        # Check shapes
        if sample['image'].shape[1] != 3:
             print("X Image channel dimension mismatch!")
        if sample['image'].shape[2] != 64:
             print("X Image spatial dimension mismatch!")
             
    except Exception as e:
        print(f"X Dataset verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
