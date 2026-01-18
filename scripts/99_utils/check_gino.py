import torch

CKPT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/ae_cylinder.ckpt"

def check():
    checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
    
    for k in state_dict.keys():
        if k.startswith("encoder.gino_encoder"):
             print(f"{k}: {state_dict[k].shape}")

if __name__ == "__main__":
    check()
