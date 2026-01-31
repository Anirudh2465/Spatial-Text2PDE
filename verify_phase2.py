import torch
from src.mllm.vision_encoder import PhysicsViT

def verify():
    print("=== Phase 2 Verification: Vision Encoder ===")
    
    # Config
    B, C, T, H, W = 4, 3, 24, 64, 64
    
    print("Initializing PhysicsViT...")
    model = PhysicsViT(
        img_size=H,
        patch_size=16,
        num_frames=T,
        embed_dim=256,
        depth=6,
        num_heads=8
    )
    
    # Param Check
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model Parameter Count: {params/1e6:.2f}M (Target: Small, e.g. < 5M)")
    
    if params > 10e6:
        print("X Warning: Model is larger than expected!")
        
    # Forward Pass
    x = torch.randn(B, C, T, H, W)
    print(f"\nForward Pass Input: {x.shape}")
    
    try:
        out = model(x)
        print(f"✓ Output Shape: {out.shape}")
        
        expected_tokens = 24 * (64//16) * (64//16) # 24 * 4 * 4 = 384
        if out.shape[1] != expected_tokens:
            print(f"X Sequence length mismatch! Expected {expected_tokens}, got {out.shape[1]}")
        else:
            print("✓ Sequence length correct.")
            
        if out.shape[2] != 256:
            print(f"X Embedding dim mismatch! Expected 256, got {out.shape[2]}")
            
    except Exception as e:
        print(f"X Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
