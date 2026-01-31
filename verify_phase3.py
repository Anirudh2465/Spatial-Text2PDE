import torch
from src.mllm.language_decoder import PhysicsGPT

def verify():
    print("=== Phase 3 Verification: PhysicsGPT Decoder ===")
    
    # Config
    vocab_size = 5000
    dim = 256
    depth = 6
    heads = 8
    seq_len = 128
    
    print("Initializing PhysicsGPT...")
    model = PhysicsGPT(vocab_size, dim, depth, heads)
    model.eval() # IMPORTANT: Disable dropout for deterministic check
    
    # Param Check
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model Parameter Count: {params/1e6:.2f}M")
    
    # 1. Forward Pass with Input IDs (Text Only)
    x_ids = torch.randint(0, vocab_size, (2, seq_len))
    logits = model(input_ids=x_ids)
    print(f"✓ Text-only Forward Output: {logits.shape} (Expected: 2, {seq_len}, {vocab_size})")
    
    # 2. Forward Pass with Embeddings (Multimodal Sim)
    # Simulate: [Vision (10)] + [Text (20)]
    vision_emb = torch.randn(2, 10, dim)
    text_emb = model.token_emb(torch.randint(0, vocab_size, (2, 20)))
    combined = torch.cat([vision_emb, text_emb], dim=1) # (2, 30, 256)
    
    logits_mm = model(inputs_embeds=combined)
    print(f"✓ Multimodal Forward Output: {logits_mm.shape} (Expected: 2, 30, {vocab_size})")
    
    # 3. Causal Mask Verification
    # If we change the last token of input, the output for the FIRST token should NOT change.
    input1 = torch.randint(0, vocab_size, (1, 10))
    input2 = input1.clone()
    input2[0, -1] = 123 # Change last token
    
    out1 = model(input_ids=input1)
    out2 = model(input_ids=input2)
    
    # Check if index 0-8 are identical
    diff = (out1[0, :-1] - out2[0, :-1]).abs().sum()
    if diff < 1e-5:
        print("✓ Causal Masking Verified (Past is independent of Future).")
    else:
        print(f"X Causal Masking FAILED! Diff: {diff}")

if __name__ == "__main__":
    verify()
