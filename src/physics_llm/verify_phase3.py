import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.physics_llm.model import PhysicsGPT, PhysicsGPTConfig
from src.physics_llm.tokenizer import PhysicsTokenizer
from src.physics_llm.vision import SimpleVisionEncoder, GrassmannProjector

def verify_phase3():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints/physics_llm'
    
    tokenizer = PhysicsTokenizer()
    phase3_max_len = 160
    config = PhysicsGPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_len=phase3_max_len,
        n_embd=256,
        n_head=4,
        n_layer=4,
        dropout=0.1
    )
    
    model = PhysicsGPT(config).to(device)
    vision_encoder = SimpleVisionEncoder(input_dim=64, output_dim=256).to(device)
    grassmann = GrassmannProjector(input_dim=256, k_tokens=8, proj_dim=256).to(device)
    
    checkpoint = torch.load(os.path.join(save_dir, 'phase3_checkpoint.pth'))
    model.load_state_dict(checkpoint['model'])
    vision_encoder.load_state_dict(checkpoint['vision_encoder'])
    grassmann.load_state_dict(checkpoint['grassmann'])
    
    model.eval()
    
    print("\n--- Phase 3 Final Verification ---")
    
    # 1. Adversarial Test (Re=500 + Turbulent Vision)
    print("\n[Test 1: Adversarial Robustness]")
    t = torch.linspace(0, 4*3.14159, 24).unsqueeze(1)
    base = torch.randn(1, 64)
    mod_b = torch.sin(t * 3) # Turbulent-ish
    noise_b = torch.randn(24, 64) * 2.0
    vis_b = (mod_b * base) + noise_b
    vis_b = vis_b.float().unsqueeze(0).to(device)
    
    phys_text = (
        "<PHYSICS>\n"
        "Re = 500 ;\n"
        "Velocity = 1.00 ;\n"
        "Radius = 0.50 ;\n"
        "Pos_X = 0.50 ;\n"
        "Pos_Y = 0.50 ;\n"
        "</PHYSICS>\n"
    )
    
    prompt_ids = tokenizer.encode(phys_text, add_special_tokens=True)
    if prompt_ids[-1] == tokenizer.eos_token_id: prompt_ids = prompt_ids[:-1]
    prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(device)
    physics_end_token_id = tokenizer.token_to_id["</PHYSICS>"]
    
    with torch.no_grad():
        v = vision_encoder(vis_b)
        vis_emb = grassmann(v)
        gen_ids = model.generate(prompt_tensor, vision_embeds=vis_emb, max_new_tokens=40, temperature=0, physics_end_token_id=physics_end_token_id)
        
    out = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=False)
    caption = out.split("</PHYSICS>")[-1].strip().replace("<EOS>", "")
    
    print(f"Prop Input: {phys_text.strip()}")
    print(f"Vis Input:  Turbulent")
    print(f"Output:     {caption}")
    
    if "turbulent" in caption and "Re = 500" in caption:
        print("Result: PASS (Language follows Vision, Number follows Physics)")
    else:
        print("Result: FAIL")

    # 2. Generalization Test (Unseen Re=9999)
    print("\n[Test 2: Generalization (Unseen Re)]")
    phys_text_2 = phys_text.replace("Re = 500", "Re = 9999")
    prompt_ids_2 = tokenizer.encode(phys_text_2, add_special_tokens=True)
    if prompt_ids_2[-1] == tokenizer.eos_token_id: prompt_ids_2 = prompt_ids_2[:-1]
    prompt_tensor_2 = torch.tensor(prompt_ids_2).unsqueeze(0).to(device)
    
    # Use Laminar Vision (Smooth)
    mod_a = torch.sin(t * 0.5) 
    vis_a = (mod_a * base) * 0.5
    vis_a = vis_a.float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        v = vision_encoder(vis_a)
        vis_emb = grassmann(v)
        gen_ids = model.generate(prompt_tensor_2, vision_embeds=vis_emb, max_new_tokens=40, temperature=0, physics_end_token_id=physics_end_token_id)
        
    out_2 = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=False)
    caption_2 = out_2.split("</PHYSICS>")[-1].strip().replace("<EOS>", "")
    
    print(f"Prop Input: {phys_text_2.strip()}")
    print(f"Vis Input:  Laminar")
    print(f"Output:     {caption_2}")
    
    if "laminar" in caption_2 and "Re = 9999" in caption_2:
        print("Result: PASS (Correctly generalized to unseen number)")
    else:
        print("Result: FAIL")

if __name__ == "__main__":
    verify_phase3()
