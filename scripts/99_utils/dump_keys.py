import torch

CKPT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/ae_cylinder.ckpt"

def dump():
    checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
    
    with open("all_keys.txt", "w") as f:
        for k in sorted(state_dict.keys()):
            f.write(f"{k}: {state_dict[k].shape}\n")

if __name__ == "__main__":
    dump()
