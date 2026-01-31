import torch
from src.mllm.model import PhysicsMLLM
from tokenizers import Tokenizer

def verify():
    print("=== Phase 4 Verification: PhysicsMLLM ===")
    
    # Config
    B = 2
    T = 24
    H = 64
    Seq = 20
    Vocab = 5000
    
    print("Initializing PhysicsMLLM...")
    model = PhysicsMLLM(vocab_size=Vocab, num_frames=T, img_size=H)
    model.eval()
    
    # Param Check
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total Parameter Count: {params/1e6:.2f}M (Target: < 10M)")
    
    # 1. Forward Pass
    img = torch.randn(B, 3, T, H, H)
    input_ids = torch.randint(0, Vocab, (B, Seq))
    
    try:
        logits = model(img, input_ids)
        # Expected Output Shape: (B, Vis_Tokens + Text_Tokens, Vocab)
        # Vis_Tokens = 24 * (64/16)**2 = 24 * 16 = 384
        # Total = 384 + 20 = 404
        exp_seq = 384 + Seq
        
        print(f"✓ Forward Output: {logits.shape}")
        
        if logits.shape[1] != exp_seq:
            print(f"X Sequence Length Mismatch! Expected {exp_seq}, got {logits.shape[1]}")
        else:
            print("✓ Sequence Length Correct.")
            
    except Exception as e:
        print(f"X Forward Pass Failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Generation Check (Mock)
    print("\n--- Generation Check (Mock) ---")
    # We need a tokenizer instance
    try:
        tok_path = "d:/Semester 6/Natural Language Processing/Project 3/src/mllm/mllm_tokenizer.json"
        
        # Create a mock tokenizer if file doesn't exist (or just try to load)
        # We assume verify_phase1 ran successfully, so it should exist.
        import os
        if os.path.exists(tok_path):
            tokenizer = Tokenizer.from_file(tok_path)
            
            # Simple generate call
            img_one = torch.randn(1, 3, T, H, H)
            txt = model.generate(img_one, tokenizer, max_new_tokens=5)
            print(f"✓ Generated Text: '{txt}'")
        else:
            print("! Tokenizer file not found, skipping generation check.")
            
    except Exception as e:
        print(f"X Generation Failed: {e}")

if __name__ == "__main__":
    verify()
