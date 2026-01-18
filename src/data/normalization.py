import pickle
import torch
import numpy as np

class Normalizer:
    def __init__(self, stat_path):
        self.stat_path = stat_path
        self.mean = None
        self.std = None
        self.load_stats()

    def load_stats(self):
        with open(self.stat_path, 'rb') as f:
            stats = pickle.load(f)
        
        # specific logic for the list format [m0, s0, m1, s1, m2, s2]
        # Channels: U, V, P
        self.mean = torch.tensor([stats[0], stats[2], stats[4]]).view(1, 1, 3, 1, 1) # B, T, C, H, W
        self.std = torch.tensor([stats[1], stats[3], stats[5]]).view(1, 1, 3, 1, 1)
        
        # Check for near-zero std to avoid nan
        self.std[self.std < 1e-6] = 1.0
        
        print(f"Normalizer loaded from {self.stat_path}")
        print(f"Mean: {self.mean.flatten()}")
        print(f"Std:  {self.std.flatten()}")

    def normalize(self, x):
        # x: (B, T, C, H, W)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnormalize(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

# MinMax Normalizer if needed
class MinMaxNormalizer:
    def __init__(self, stat_path):
        self.stat_path = stat_path
        self.load_stats()
        
    def load_stats(self):
        with open(self.stat_path, 'rb') as f:
             stats = pickle.load(f)
        # [min0, max0, min1, max1, min2, max2]
        self.min = torch.tensor([stats[0], stats[2], stats[4]]).view(1, 1, 3, 1, 1)
        self.max = torch.tensor([stats[1], stats[3], stats[5]]).view(1, 1, 3, 1, 1)
        
    def normalize(self, x):
        # Scale to [-1, 1]
        x_mins = self.min.to(x.device)
        x_maxs = self.max.to(x.device)
        
        # 0..1
        x_norm = (x - x_mins) / (x_maxs - x_mins + 1e-6)
        # -1..1
        return 2 * x_norm - 1

    def unnormalize(self, x):
        # -1..1 -> 0..1
        x_norm = (x + 1) / 2
        # 0..1 -> min..max
        x_mins = self.min.to(x.device)
        x_maxs = self.max.to(x.device)
        return x_norm * (x_maxs - x_mins) + x_mins
