import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.physics_llm.model import PhysicsGPT, PhysicsGPTConfig
from src.physics_llm.tokenizer import PhysicsTokenizer
from src.physics_llm.vision import SimpleVisionEncoder, GrassmannProjector
from src.physics_llm.dataset import PhysicsDataset

def adversarial_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints/physics_llm'
    
    tokenizer = PhysicsTokenizer()
    phase2_max_len = 160
    config = PhysicsGPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_len=phase2_max_len,
        n_embd=256,
        n_head=4,
        n_layer=4,
        dropout=0.1
    )
    model = PhysicsGPT(config).to(device)
    
    vision_encoder = SimpleVisionEncoder(input_dim=64, output_dim=256).to(device)
    grassmann = GrassmannProjector(input_dim=256, k_tokens=8, proj_dim=256).to(device)
    
    checkpoint = torch.load(os.path.join(save_dir, 'phase2_checkpoint.pth'))
    model.load_state_dict(checkpoint['model'])
    vision_encoder.load_state_dict(checkpoint['vision_encoder'])
    grassmann.load_state_dict(checkpoint['grassmann'])
    
    # Adversarial Test
    # Swap physics between samples
    # Sample A: Laminar (Re=500), Vision=Laminar
    # Sample B: Turbulent (Re=2500), Vision=Turbulent
    # Input: Physics A + Vision B
    # Expected: "The flow is turbulent with Reynolds number Re = 500."
    
    print("\n--- Adversarial Verification ---")
    
    # Create inputs manually
    t = torch.linspace(0, 4*3.14159, 24).unsqueeze(1)
    base = torch.randn(1, 64)
    
    # Vision B (Turbulent)
    mod_b = torch.sin(t * 3)
    noise_b = torch.randn(24, 64) * 2.0
    vis_b = (mod_b * base) + noise_b
    vis_b = vis_b.float().unsqueeze(0).to(device) # (1, 24, 64)
    
    # Physics A (Laminar Re)
    # Must match training format exactly
    phys_text_a = (
        "<PHYSICS>\n"
        "Re = 500 ;\n"
        "Velocity = 1.00 ;\n"
        "Radius = 0.50 ;\n"
        "Pos_X = 0.50 ;\n"
        "Pos_Y = 0.50 ;\n"
        "</PHYSICS>\n"
    )
    prompt_ids = tokenizer.encode(phys_text_a, add_special_tokens=True)
    # Remove EOS (last token)
    if prompt_ids[-1] == tokenizer.eos_token_id:
        prompt_ids = prompt_ids[:-1]
        
    prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(device)
    
    physics_end_token_id = tokenizer.token_to_id["</PHYSICS>"]

    with torch.no_grad():
        v = vision_encoder(vis_b)
        vis_emb = grassmann(v)
        
        gen_ids = model.generate(
            prompt_tensor, 
            vision_embeds=vis_emb, 
            max_new_tokens=40, 
            temperature=0, 
            physics_end_token_id=physics_end_token_id
        )
        
    out = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=False)
    caption = out.split("</PHYSICS>")[-1].strip().replace("<EOS>", "")
    
    print(f"Physics Input: {phys_text_a.strip()}")
    print(f"Vision Input: Turbulent Pattern")
    print(f"Generated: {caption}")
    
    if "turbulent" in caption:
        print("Vision Check: PASS (Description follows Vision)")
    else:
        print("Vision Check: FAIL")
        
    if "Re = 500" in caption:
        print("Physics Check: PASS (Numbers follow Physics)")
    else:
        print("Physics Check: FAIL")

if __name__ == "__main__":
    adversarial_test()
