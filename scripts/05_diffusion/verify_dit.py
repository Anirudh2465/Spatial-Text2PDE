import torch
import sys
import os

# Mock modules for checkpoint loading context
import types
m = types.ModuleType('modules')
sys.modules['modules'] = m
m.utils = types.ModuleType('utils')
class MockClass:
    def __init__(self, *args, **kwargs): pass
m.utils.Normalizer = MockClass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.diffusion import DiffusionTransformer

CKPT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/ldm_DiT_text_cylinder.ckpt"

def verify():
    print("Initializing Model...")
    model = DiffusionTransformer(
        input_size=(6, 16, 16),
        patch_size=2,
        in_channels=16,
        hidden_size=1024,
        depth=28,
        num_heads=16,
        context_dim=768
    )
    
    print("Loading Weights...")
    try:
        model.load_from_ckpt(CKPT_PATH)
    except Exception as e:
        print(f"Loading Error: {e}")
        # Proceed to test forward even if load partial
    
    # Test Forward
    B = 1
    x = torch.randn(B, 16, 6, 16, 16)
    t = torch.tensor([500]) # Timestep
    c = torch.randn(B, 77, 768) # Text context (Length 77 standard for CLIP/BERT)
    
    print("Running Forward Pass...")
    out = model(x, t, c)
    print(f"Output Shape: {out.shape}")
    
    if out.shape == x.shape:
        print("SUCCESS: Output shape matches input shape.")
    else:
        print("FAILURE: Shape mismatch.")

if __name__ == "__main__":
    verify()
