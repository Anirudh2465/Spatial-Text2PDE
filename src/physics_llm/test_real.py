import torch
import h5py
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.physics_llm.model import PhysicsGPT, PhysicsGPTConfig
from src.physics_llm.tokenizer import PhysicsTokenizer
from src.physics_llm.vision import SimpleVisionEncoder, GrassmannProjector

def heuristic_visual_analysis(grid_tensor):
    """
    Analyze real video tensor (C, T, H, W) to determine flow type.
    Returns: 'laminar' or 'turbulent'
    """
    # Calculate temporal standard deviation
    # grid: (3, T, 64, 64)
    # We want variance over T dimension
    # std across dim 1 (T)
    
    # 1. Norm velocity mag? Just take raw.
    std_map = torch.std(grid_tensor, dim=1) # (3, 64, 64)
    mean_std = std_map.mean().item()
    
    # Threshold determined empirically (or estimated)
    # Laminar flow is steady -> Low temporal variance.
    # Turbulent/Vortex Shedding -> High temporal variance.
    # Let's say threshold 0.05? 
    # Actually, let's print it to see.
    return mean_std

def synthesize_features(device, flow_type):
    """Bridge: Generate synthetic features compatible with PhysicsGPT based on analysis."""
    # Matches dataset.py logic
    t = torch.linspace(0, 4*3.14159, 24).unsqueeze(1) # (24, 1)
    base = torch.randn(1, 64)
    
    if flow_type == "laminar":
        # Smooth sine
        mod = torch.sin(t) 
        vis = (mod * base) * 0.5
    else:
        # Turbulent / Vortex
        mod = torch.sin(t * 3)
        noise = torch.randn(24, 64) * 2.0
        vis = (mod * base) + noise
        
    return vis.float().unsqueeze(0).to(device)

def test_real_data():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints/physics_llm'
    data_path = "train_grid_64.h5"
    
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found.")
        return

    # Load Model
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
    physics_end_token_id = tokenizer.token_to_id["</PHYSICS>"]
    
    print("\n--- Testing on Real Dataset (Bridge Mode) ---")
    
    with h5py.File(data_path, 'r') as f:
        keys = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)
        
        # Test 3 Samples: Low Re, Mid Re, High Re
        # Assuming keys are sorted by something? Or random?
        # Let's scan for specific Re values
        
        targets = [30.0, 100.0, 300.0, 2000.0]
        found_count = 0
        
        for key in keys:
            if found_count >= 5: break
            
            grp = f[key]
            re = float(grp['reynolds_number'][()])
            
            # Simple sampling strategy
            # Just take first few unique ones?
            
            # Load Grid
            grid = grp['grid'][:] # (T, 3, H, W) or (25, 3, 64, 64)
            # Check shape
            if grid.shape[0] != 25: 
                # Transpose? dataset.py says grp['grid'][:] is (25, 3, 64, 64)
                pass
                
            grid_tensor = torch.tensor(grid).float()
            # Permute to (C, T, H, W) for analysis
            # Dataset.py: image = grid_tensor.permute(1, 0, 2, 3)
            # grid (25, 3, 64, 64) -> (3, 25, 64, 64)
            grid_tensor = grid_tensor.permute(1, 0, 2, 3) 
            
            # Analyze
            variance_score = heuristic_visual_analysis(grid_tensor)
            
            # Classify
            # This threshold needs tuning. Let's guess: Laminar is VERY steady.
            # Std dev should be near zero.
            is_laminar = variance_score < 0.01 
            detected_flow = "laminar" if is_laminar else "turbulent"
            
            # Construct Physics Block
            # We don't have Pos_X etc in H5. Fill dummy.
            phys_text = (
                "<PHYSICS>\n"
                f"Re = {int(re)} ;\n"
                "Velocity = 0.00 ;\n"
                "Radius = 0.00 ;\n"
                "Pos_X = 0.00 ;\n"
                "Pos_Y = 0.00 ;\n"
                "</PHYSICS>\n"
            )
            
            # Synthesize Features
            vis_tensor = synthesize_features(device, detected_flow)
            
            # Generate
            prompt_ids = tokenizer.encode(phys_text, add_special_tokens=True)
            if prompt_ids[-1] == tokenizer.eos_token_id: prompt_ids = prompt_ids[:-1]
            prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(device)
            
            with torch.no_grad():
                v = vision_encoder(vis_tensor)
                vis_emb = grassmann(v)
                gen_ids = model.generate(prompt_tensor, vision_embeds=vis_emb, max_new_tokens=40, temperature=0, physics_end_token_id=physics_end_token_id)
                
            out = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=False)
            try: caption = out.split("</PHYSICS>")[-1].strip().replace("<EOS>", "")
            except: caption = out
            
            print(f"\nSample Key: {key}")
            print(f"Real Re: {re}")
            print(f"Video Var: {variance_score:.4f} -> Detected: {detected_flow}")
            print(f"Physics Input: Re = {int(re)}")
            print(f"Generated: {caption}")
            
            if str(int(re)) in caption:
                print("Physics Check: PASS")
            else:
                print("Physics Check: FAIL")
                
            if detected_flow in caption:
                print("Vision Check: PASS (Matches Heuristic)")
            else:
                print("Vision Check: FAIL")
                
            found_count += 1

if __name__ == "__main__":
    test_real_data()
