import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.physics_llm.model import PhysicsGPT, PhysicsGPTConfig
from src.physics_llm.tokenizer import PhysicsTokenizer
from src.physics_llm.dataset_real import RealPhysicsDataset
from src.physics_llm.vision_real import PhysicsViT
from src.physics_llm.vision import GrassmannProjector

def verify_final():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints/physics_llm'
    
    tokenizer = PhysicsTokenizer()
    dataset = RealPhysicsDataset("train_grid_64.h5", tokenizer)
    
    # Load Real Model
    config = PhysicsGPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_len=256,
        n_embd=256,
        n_head=4,
        n_layer=4,
        dropout=0.1
    )
    model = PhysicsGPT(config).to(device)
    vision_encoder = PhysicsViT(img_size=64, num_frames=24, embed_dim=256).to(device)
    grassmann = GrassmannProjector(input_dim=256, k_tokens=8, proj_dim=256).to(device)
    
    ckpt_path = os.path.join(save_dir, 'real_model.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        vision_encoder.load_state_dict(ckpt['vision_encoder'])
        grassmann.load_state_dict(ckpt['grassmann'])
        model.eval()
        print("Loaded Real Model Checkpoint.")
    else:
        print("Checkpoint not found!")
        return

    physics_end_token_id = tokenizer.token_to_id.get("</PHYSICS>")
    
    print("\n--- Final Verification on Real Samples ---")
    
    # Check 3 known samples
    indices = [0, 20, 40]
    
    for idx in indices:
        if idx >= len(dataset): continue
        item = dataset[idx]
        
        # Prepare inputs
        input_ids = item['input_ids'].to(device)
        vision_tensor = item['vision_tensor'].to(device).unsqueeze(0) # (1, C, T, H, W)
        
        # Ground Truth Caption extraction
        full_text = tokenizer.decode(item['input_ids'].tolist(), skip_special_tokens=False)
        try:
            prompt_split = full_text.split("</PHYSICS>")
            physics_prompt = prompt_split[0] + "</PHYSICS>\n"
            target_caption = prompt_split[1].strip().replace("<EOS>", "")
        except:
            physics_prompt = full_text # Fail safe
            target_caption = "???"

        # Encode Prompt
        prompt_ids = tokenizer.encode(physics_prompt, add_special_tokens=True)
        # Remove EOS
        if prompt_ids[-1] == tokenizer.eos_token_id: prompt_ids = prompt_ids[:-1]
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(device)

        # Generate
        with torch.no_grad():
            v = vision_encoder(vision_tensor)
            vis_emb = grassmann(v)
            gen_ids = model.generate(prompt_tensor, vision_embeds=vis_emb, max_new_tokens=40, temperature=0, physics_end_token_id=physics_end_token_id)
            
        out = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=False)
        try: caption = out.split("</PHYSICS>")[-1].strip().replace("<EOS>", "")
        except: caption = out
        
        print(f"\n[Sample {idx}]")
        print(f"Vision Input: (Video Tensor)")
        print(f"Prompt: {physics_prompt.strip()}")
        print(f"Target: {target_caption}")
        print(f"Output: {caption}")
        
        # Checks
        if "The flow is" in caption:
            print("Fluency: PASS")
        if "Re =" in caption:
            # Check number match
            import re
            m = re.search(r"Re = (\d+)", caption)
            if m and m.group(1) in physics_prompt:
                 print("Physics Consistency: PASS")
            else:
                 print("Physics Consistency: FAIL")

if __name__ == "__main__":
    verify_final()
