import torch
import sys
import types

# Mock modules to bypass import errors
# The checkpoint likely refers to 'modules.diffusion...' or similar.
m = types.ModuleType('modules')
sys.modules['modules'] = m
m.utils = types.ModuleType('utils')
class MockClass:
    def __init__(self, *args, **kwargs): pass
m.utils.Normalizer = MockClass

CKPT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/ldm_DiT_text_cylinder.ckpt"

def inspect():
    try:
        sd = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
        if 'state_dict' in sd: sd = sd['state_dict']
        print(f"Loaded successfully. Keys: {len(sd)}")
        
        # Dump keys
        with open("ldm_keys.txt", "w") as f:
            for k in sorted(sd.keys()):
                f.write(f"{k}: {sd[k].shape}\n")
        print("Keys dumped to ldm_keys.txt")
        
        # Print first few keys
        for i, k in enumerate(sorted(sd.keys())):
            if i < 20: print(f"{k}: {sd[k].shape}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
