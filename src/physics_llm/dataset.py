import torch
from torch.utils.data import Dataset
import random
import numpy as np
from .tokenizer import PhysicsTokenizer

class PhysicsDataset(Dataset):
    def __init__(self, tokenizer: PhysicsTokenizer, size: int = 1000, max_len: int = 128):
        self.tokenizer = tokenizer
        self.size = size
        self.max_len = max_len
        self.samples = [self._generate_sample() for _ in range(size)]

    def _generate_sample(self):
        # Generate random physics values
        re = random.randint(100, 3000)
        velocity = f"{random.uniform(0.1, 5.0):.2f}"
        radius = f"{random.uniform(0.01, 1.0):.2f}"
        pos_x = f"{random.uniform(0.0, 1.0):.2f}"
        pos_y = f"{random.uniform(0.0, 1.0):.2f}"
        
        # Phase 2: Decouple Re and Flow Type to force Vision Grounding
        # We want the model to look at Vision for the adjective, not Re.
        # Randomly choose flow type
        flow_type = random.choice(["laminar", "turbulent"])
        
        # Synthetic Vision Features (T=24, D=64)
        # Laminar: Smooth low-freq sine waves
        # Turbulent: High-freq noise + sine
        t = torch.linspace(0, 4*np.pi, 24).unsqueeze(1) # (24, 1)
        base_feature = torch.randn(1, 64) # (1, 64) random direction
        
        if flow_type == "laminar":
            # Smooth variation
            modulation = torch.sin(t)
            vision_noise = torch.randn(24, 64) * 0.1
            vision_tensor = (modulation * base_feature) + vision_noise
        else:
            # Chaotic
            modulation = torch.sin(t * 3) # higher freq
            vision_noise = torch.randn(24, 64) * 2.0 # High noise
            vision_tensor = (modulation * base_feature) + vision_noise
            
        vision_tensor = vision_tensor.float()
        
        # Construct Physics Block
        # Strict canonical format
        physics_block = (
            "<PHYSICS>\n"
            f"Re = {re} ;\n"
            f"Velocity = {velocity} ;\n"
            f"Radius = {radius} ;\n"
            f"Pos_X = {pos_x} ;\n"
            f"Pos_Y = {pos_y} ;\n"
            "</PHYSICS>\n"
        )
        
        # Target Caption
        # "The flow is laminar with Reynolds number Re = 230."
        caption = f"The flow is {flow_type} with Reynolds number Re = {re}."
        
        full_text = physics_block + caption
        return full_text, vision_tensor

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        text, vision_tensor = self.samples[idx]
        
        # Tokenize (full sequence)
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate or Pad
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            # Ensure we don't chop in a weird way, but for phase 1 simple truncation is ok
        
        padding_len = self.max_len - len(input_ids)
        if padding_len > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
            
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        
        # Create Labels
        # We want to predict the caption, not the physics block.
        # Find the end of the physics block
        end_physics_token_id = self.tokenizer.token_to_id["</PHYSICS>"]
        
        labels = input_tensor.clone()
        
        # Mask out padding in labels
        labels[input_tensor == self.tokenizer.pad_token_id] = -100
        
        # Mask out the physics block (including </PHYSICS>)
        # Find index of </PHYSICS>
        try:
            split_idx = (input_tensor == end_physics_token_id).nonzero(as_tuple=True)[0][0].item()
            # Mask everything up to and including split_idx
            labels[:split_idx+1] = -100
        except IndexError:
            # If </PHYSICS> not found (truncated?), mask everything?
            # Or assume start is masked.
            pass
            
        # Create Numeric Mask for Weighted Loss
        # We only care about numerics in the TARGET (unmasked labels)
        numeric_mask = torch.zeros_like(labels, dtype=torch.float)
        
        vocab_arr = self.tokenizer.id_to_token
        digits = set(self.tokenizer.digits)
        symbols = set(self.tokenizer.symbols)
        
        for i, token_id in enumerate(input_tensor):
            if labels[i] == -100:
                continue
            
            token = vocab_arr.get(token_id.item(), "")
            if token in digits or token in symbols:
                numeric_mask[i] = 1.0
                
        return {
            "input_ids": input_tensor,
            "vision_tensor": vision_tensor,
            "labels": labels,
            "numeric_mask": numeric_mask
        }
