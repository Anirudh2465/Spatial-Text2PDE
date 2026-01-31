import torch
import os
import sys
import argparse
from tokenizers import Tokenizer
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.mllm.model import PhysicsMLLM
from src.mllm.dataset import PhysicsMLLMDataset

# Config (Must match training)
DATA_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
TOKENIZER_PATH = "d:/Semester 6/Natural Language Processing/Project 3/src/mllm/mllm_tokenizer.json"
STAT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl"
CHECKPOINT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/checkpoints/mllm_last.pth"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on {device}")

    # 1. Load Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    # 2. Load Dataset (for samples)
    dataset = PhysicsMLLMDataset(DATA_PATH, TOKENIZER_PATH, STAT_PATH, num_frames=24)
    print(f"Dataset loaded. Size: {len(dataset)}")
    
    # 3. Load Model
    model = PhysicsMLLM(
        vocab_size=tokenizer.get_vocab_size(),
        vision_dim=256,
        llm_dim=256
    ).to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # 4. Inference loop
    model.eval()
    
    # Pick a few random samples
    indices = [0, 10, 50] 
    
    for idx in indices:
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device) # (1, 3, T, H, W) -> wait, dataset is (C, T, H, W)
        # Model forward expects (B, C, T, H, W) if it uses 3D Conv or similar?
        # Let's check model.py / vision_encoder.py expectations.
        # vision_encoder.py: PhysicsViT forward(self, x): x is (B, C, T, H, W) normally for video?
        # Dataset returns: image: (3, 25, 64, 64) -> (C, T, H, W) after my fix?
        # Let's double check dataset fix.
        # Fix was: permute(1, 0, 2, 3) -> (C, T, H, W)
        # So yes, unsqueeze(0) gives (1, C, T, H, W)
        
        true_text = sample['text']
        true_re = sample['reynolds_number'].item()
        
        print(f"\n--- Sample {idx} ---")
        print(f"Ground Truth Re: {true_re:.2f}")
        print(f"Ground Truth Text: {true_text}")
        
        with torch.no_grad():
            # 1. Regression
            vis = model.vision_encoder(image)
            proj = model.projector(vis)
            re_pred = model.regression_head(proj.mean(dim=1))
            print(f"Predicted Re:    {re_pred.item():.2f}")
            
            # 2. Text Generation
            generated_text = model.generate(image, tokenizer, max_new_tokens=100)
            
        print(f"Generated Text:  {generated_text}")

if __name__ == "__main__":
    main()
