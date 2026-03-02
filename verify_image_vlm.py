import torch
from src.image_mllm.model_image import ImageMLLM
from tokenizers import Tokenizer

def verify():
    print("=== Verification: Image MLLM ===")
    
    B, H = 2, 64
    Seq = 20
    Vocab = 5000
    
    print("Initializing ImageMLLM...")
    model = ImageMLLM(vocab_size=Vocab, img_size=H)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {params/1e6:.2f}M")
    
    # 1. Forward Pass
    img = torch.randn(B, 3, H, H)
    input_ids = torch.randint(0, Vocab, (B, Seq))
    
    try:
        logits = model(img, input_ids)
        # Total tokens = Vision Tokens (16 if 64x64, patch_size 16 -> 4x4) + Text Tokens (Seq)
        exp_seq = 16 + Seq
        
        print(f"Forward Output: {logits.shape}")
        if logits.shape[1] != exp_seq:
            print(f"Mismatch! Expected {exp_seq}, got {logits.shape[1]}")
        else:
            print("Sequence Length Correct.")
            
    except Exception as e:
        print(f"Forward Pass Failed: {e}")

    # 2. Mock Generation
    print("\n--- Mock Generation ---")
    try:
        tok_path = "src/mllm/mllm_tokenizer.json"
        import os
        if os.path.exists(tok_path):
            tokenizer = Tokenizer.from_file(tok_path)
            img_one = torch.randn(1, 3, H, H)
            txt = model.generate(img_one, tokenizer, max_new_tokens=5)
            print(f"Generated Text: '{txt}'")
    except Exception as e:
        print(f"Generation Failed: {e}")

if __name__ == "__main__":
    verify()
