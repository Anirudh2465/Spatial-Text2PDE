import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from src.mllm.label_generator import generate_description
from src.data.normalization import Normalizer

class PhysicsMLLMDataset(Dataset):
    """
    Dataset for Physics MLLM.
    Returns:
        image: (3, 25, 64, 64) Normalized Simulation tensors.
        text_input: Tokenized text sequence with <IMG> tokens.
    """
    def __init__(self, data_path, tokenizer_path, stat_path=None, num_frames=24):
        self.data_path = data_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.num_frames = num_frames
        
        # Load Normalizer if provided
        self.normalizer = Normalizer(stat_path) if stat_path else None
        
        # Cache keys
        with h5py.File(data_path, 'r') as f:
            self.keys = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)
            
        # Special Tokens
        self.sos_token = self.tokenizer.token_to_id("[SOS]")
        self.eos_token = self.tokenizer.token_to_id("[EOS]")
        self.sep_token = self.tokenizer.token_to_id("[SEP]")
        self.img_token = self.tokenizer.token_to_id("<IMG>")
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        
        with h5py.File(self.data_path, 'r') as f:
            grp = f[key]
            
            # 1. Load Image Data
            grid = grp['grid'][:] # (25, 3, 64, 64)
            print(f"DEBUG DATASET: Key {key}, Grid {grid.shape}")
            
            # Slice time dimension
            if grid.shape[0] > self.num_frames:
                grid = grid[:self.num_frames]
            
            # Prepare tensor
            # Prepare tensor
            grid_tensor = torch.tensor(grid).float() 
            if grid_tensor.shape[0] > self.num_frames:
               grid_tensor = grid_tensor[:self.num_frames]

            # Normalize (Expects B, T, C, H, W)
            if self.normalizer:
                 grid_tensor = self.normalizer.normalize(grid_tensor.unsqueeze(0)).squeeze(0)
            
            # Permute to (C, T, H, W) for Model
            image = grid_tensor.permute(1, 0, 2, 3)

            # 2. Generate Text
            re = grp['reynolds_number'][()]
            prompt = grp['prompt'][()]
            
            description = generate_description({'reynolds_number': re, 'prompt': prompt})
            
            # 3. Tokenize
            # Return pure caption for training. Model will prepend Vision tokens.
            text_tokens = self.tokenizer.encode(description).ids
            
            # Caption: [SOS] text [EOS]
            caption_ids = [self.sos_token] + text_tokens + [self.eos_token]
            
            return {
                'image': image, #(T, C, H, W)
                'caption_ids': torch.tensor(caption_ids, dtype=torch.long),
                'text': description,
                'reynolds_number': torch.tensor(re, dtype=torch.float32)
            }

if __name__ == "__main__":
    # Test
    ds = PhysicsMLLMDataset(
        "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5",
        "d:/Semester 6/Natural Language Processing/Project 3/src/mllm/mllm_tokenizer.json",
        "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl"
    )
    sample = ds[0]
    print("Image shape:", sample['image'].shape)
    print("Caption IDs:", sample['caption_ids'])
    print("Text:", sample['text'])
