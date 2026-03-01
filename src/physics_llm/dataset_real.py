import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import re
from .tokenizer import PhysicsTokenizer

class RealPhysicsDataset(Dataset):
    def __init__(self, h5_path, tokenizer: PhysicsTokenizer, num_frames=24):
        self.h5_path = h5_path
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        
        with h5py.File(h5_path, 'r') as f:
            self.keys = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)
            
    def __len__(self):
        return len(self.keys)
    
    def parse_prompt(self, prompt_str):
        data = {
            'Re': '0', 'Velocity': '0.00', 'Radius': '0.00',
            'Pos_X': '0.00', 'Pos_Y': '0.00'
        }
        
        # Radius
        m_rad = re.search(r"radius of ([\d\.]+)", prompt_str)
        if m_rad: data['Radius'] = m_rad.group(1).rstrip('.')
        
        # Position
        m_pos = re.search(r"position: ([\d\.]+), ([\d\.]+)", prompt_str)
        if m_pos:
            data['Pos_X'] = m_pos.group(1).rstrip('.')
            data['Pos_Y'] = m_pos.group(2).rstrip('.')
            
        # Velocity
        m_vel = re.search(r"velocity of ([\d\.]+)", prompt_str)
        if m_vel: data['Velocity'] = m_vel.group(1).rstrip('.')
        
        # Re
        m_re = re.search(r"Reynolds number is (\d+)", prompt_str)
        if m_re: data['Re'] = m_re.group(1)
        
        return data

    def __getitem__(self, idx):
        key = self.keys[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            grp = f[key]
            
            # 1. Vision (Grid)
            # (25, 3, 64, 64) -> Take first num_frames
            grid = grp['grid'][:] 
            if grid.shape[0] > self.num_frames:
                grid = grid[:self.num_frames]
            elif grid.shape[0] < self.num_frames:
                # Pad? Or wrap. Simple repetition for now if needed.
                # Assuming dataset is mostly consistent.
                pass
                
            # Tensor (T, C, H, W) -> Model expects (B, C, T, H, W)
            # Dataset returns (C, T, H, W) usually for DataLoader to stack.
            # Grid is (T, C, H, W).
            grid_tensor = torch.tensor(grid).float()
            vision_tensor = grid_tensor.permute(1, 0, 2, 3) # (C, T, H, W)
            
            # 2. Text (Prompt -> Physics Block)
            prompt = grp['prompt'][()]
            if isinstance(prompt, bytes): prompt = prompt.decode('utf-8')
            
            # Parse physics
            parsed = self.parse_prompt(prompt)
            
            physics_block = (
                "<PHYSICS>\n"
                f"Re = {parsed['Re']} ;\n"
                f"Velocity = {parsed['Velocity']} ;\n"
                f"Radius = {parsed['Radius']} ;\n"
                f"Pos_X = {parsed['Pos_X']} ;\n"
                f"Pos_Y = {parsed['Pos_Y']} ;\n"
                "</PHYSICS>\n"
            )
            
            # Extract Caption (The description part)
            # "The Reynolds number is 230. The flow is transitioning in the wake."
            # Split by "Reynolds number is X."?
            # Or just use the whole Prompt as the "Target Caption" (including physics description redundancies)?
            # Phase 1: "The flow is laminar with Reynolds number Re = 230."
            # Real Prompt: "... The Reynolds number is 230. The flow is transitioning..."
            # Let's clean it up to be the caption.
            
            # Find where "The Reynolds number" starts
            caption_start = prompt.find("The Reynolds number is")
            if caption_start != -1:
                caption = prompt[caption_start:]
            else:
                caption = prompt
                
            full_text = physics_block + caption
            
            # Tokenize
            ids = self.tokenizer.encode(full_text, add_special_tokens=True)
            input_tensor = torch.tensor(ids, dtype=torch.long)
            
            # Labels (Shifted in Model, but here same as input)
            labels = input_tensor.clone()
            
            # Mask Physics (Don't compute loss on Physics tokens?)
            # Phase 1 rules: "Physics tokens are part of the input sequence and are not predicted"
            # So masked -100 in labels.
            # Find </PHYSICS>
            phys_end_id = self.tokenizer.token_to_id.get("</PHYSICS>")
            
            # Find end position
            # This is simple scanning
            try:
                # First occurrence
                split_idx = (input_tensor == phys_end_id).nonzero()[0].item()
                # Mask 0..split_idx
                labels[:split_idx+1] = -100
            except:
                pass # If not found? Should generally be found.
                
            # Numeric Mask (for weighted loss)
            numeric_mask = torch.zeros_like(input_tensor, dtype=torch.float)
            for i, token_id in enumerate(ids):
                token = self.tokenizer.decode([token_id])
                if any(c.isdigit() for c in token) or token in ['=', '.']:
                    numeric_mask[i] = 1.0
                    
            return {
                "input_ids": input_tensor,
                "vision_tensor": vision_tensor,
                "labels": labels,
                "numeric_mask": numeric_mask
            }
