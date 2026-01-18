import torch

CKPT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/ae_cylinder.ckpt"

def check():
    checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
    
    prefixes = set()
    for k in state_dict.keys():
        if k.startswith("encoder."):
            parts = k.split('.')
            if len(parts) > 1:
                prefixes.add(f"encoder.{parts[1]}")
            else:
                prefixes.add(k)
                
    print(f"Encoder Submodules: {prefixes}")

if __name__ == "__main__":
    check()
