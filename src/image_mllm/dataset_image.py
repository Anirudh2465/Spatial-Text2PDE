import torch
import h5py
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from src.data.normalization import Normalizer

class ImageCylinderDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, stat_path=None, num_frames=24):
        self.data_path = data_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.normalizer = Normalizer(stat_path) if stat_path else None
        self.num_frames = num_frames
        
        with h5py.File(data_path, 'r') as f:
            self.keys = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)
            
        self.sos_token = self.tokenizer.token_to_id("[SOS]")
        self.eos_token = self.tokenizer.token_to_id("[EOS]")
        
    def __len__(self):
        return len(self.keys) * self.num_frames
        
    def __getitem__(self, idx):
        sim_idx = idx // self.num_frames
        frame_idx = idx % self.num_frames
        key = self.keys[sim_idx]
        
        with h5py.File(self.data_path, 'r') as f:
            grp = f[key]
            
            # Extract the specific frame
            grid = grp['grid'][frame_idx] # (3, 64, 64)
            grid_tensor = torch.tensor(grid).float()
            
            if self.normalizer:
                # Normalizer expects (B, T, C, H, W)
                grid_fake = grid_tensor.unsqueeze(0).unsqueeze(0) 
                grid_norm = self.normalizer.normalize(grid_fake).squeeze(0).squeeze(0) 
                image = grid_norm
            else:
                image = grid_tensor
                
            prompt = grp['prompt'][()]
            if isinstance(prompt, bytes): prompt = prompt.decode('utf-8')
            
            # The prompt is our pure caption!
            caption = prompt
            
            text_tokens = self.tokenizer.encode(caption).ids
            ids = [self.sos_token] + text_tokens + [self.eos_token]
            
            input_tensor = torch.tensor(ids, dtype=torch.long)
            labels = input_tensor.clone()
            
            numeric_mask = torch.zeros_like(input_tensor, dtype=torch.float)
            for i, token_id in enumerate(ids):
                token = self.tokenizer.decode([token_id])
                if any(c.isdigit() for c in token) or token in ['=', '.']:
                    numeric_mask[i] = 1.0
                    
            return {
                "image": image,
                "input_ids": input_tensor,
                "labels": labels,
                "numeric_mask": numeric_mask
            }
