import torch
import h5py
import re
import random
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from src.data.normalization import Normalizer

class ImageCylinderDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, stat_path=None, num_frames=24, split='train', split_ratios=(0.8, 0.1, 0.1)):
        self.data_path = data_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.normalizer = Normalizer(stat_path) if stat_path else None
        self.split = split
        
        with h5py.File(data_path, 'r') as f:
            all_keys = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)
            
        n_total = len(all_keys)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        
        if split == 'train':
            self.keys = all_keys[:n_train]
        elif split == 'val':
            self.keys = all_keys[n_train:n_train+n_val]
        else: # test
            self.keys = all_keys[n_train+n_val:]
            
        # First, last, and 2 from the middle
        if num_frames >= 4:
            self.frame_indices = [0, num_frames // 3, 2 * (num_frames // 3), num_frames - 1]
        else:
            self.frame_indices = list(range(num_frames))
            
        self.sos_token = self.tokenizer.token_to_id("[SOS]")
        self.eos_token = self.tokenizer.token_to_id("[EOS]")
        
    def variate_prompt(self, prompt, split):
        # Only variate during training
        if split != 'train':
            return prompt
            
        radius = re.search(r"radius of ([\d\.]+)", prompt)
        pos = re.search(r"position: ([\d\.]+),\s*([\d\.]+)", prompt)
        vel = re.search(r"velocity of ([\d\.]+)", prompt)
        reynolds = re.search(r"Reynolds number is (\d+)", prompt)
        flow_match = re.search(r"The flow is (.*?)\.", prompt)
        
        if not (radius and pos and vel and reynolds and flow_match):
            return prompt
            
        rad = radius.group(1).rstrip('.')
        px = pos.group(1).rstrip('.')
        py = pos.group(2).rstrip('.')
        v = vel.group(1).rstrip('.')
        re_num = reynolds.group(1).rstrip('.')
        flow = flow_match.group(1)
        
        templates = [
            f"Fluid passes over a cylinder with a radius of {rad} and position: {px}, {py}. Fluid enters with a velocity of {v}. The Reynolds number is {re_num}. The flow is {flow}.",
            f"A cylinder of radius {rad} is located at ({px}, {py}) in a fluid stream. The inlet velocity is {v}, resulting in a Reynolds number of {re_num}. We observe that the flow is {flow}.",
            f"The flow is {flow}. It has a Reynolds number of {re_num} and an initial velocity of {v}. This is caused by a cylinder (radius: {rad}) positioned at X={px}, Y={py}.",
            f"With a Reynolds number of {re_num}, the flow is {flow}. The fluid velocity at the inlet is {v}, passing around a cylinder at {px}, {py} with radius {rad}.",
            f"An obstacle cylinder (radius {rad}) at coordinates {px}, {py} interacts with a fluid moving at velocity {v}. The flow is {flow} with Re={re_num}.",
            f"Re = {re_num}. Inlet velocity = {v}. Cylinder radius = {rad} at position ({px}, {py}). The resulting flow state is {flow}."
        ]
        return random.choice(templates)
        
    def __len__(self):
        return len(self.keys) * len(self.frame_indices)
        
    def __getitem__(self, idx):
        sim_idx = idx // len(self.frame_indices)
        f_idx = idx % len(self.frame_indices)
        key = self.keys[sim_idx]
        real_frame_idx = self.frame_indices[f_idx]
        
        with h5py.File(self.data_path, 'r') as f:
            grp = f[key]
            
            # Extract the specific frame
            grid = grp['grid'][real_frame_idx] # (3, 64, 64)
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
            caption = self.variate_prompt(prompt, getattr(self, 'split', 'train'))
            
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
