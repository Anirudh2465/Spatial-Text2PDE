import h5py
import re
import numpy as np
import os

def parse_prompt(prompt_str):
    # Format: 
    # "Fluid passes over a cylinder with a radius of 5.15 and position: 0.34, 0.12. Fluid enters with a velocity of 0.24. The Reynolds number is 230. The flow is transitioning in the wake."
    
    data = {}
    
    # Radius
    m_rad = re.search(r"radius of ([\d\.]+)", prompt_str)
    if m_rad: data['Radius'] = m_rad.group(1)
    
    # Position
    m_pos = re.search(r"position: ([\d\.]+), ([\d\.]+)", prompt_str)
    if m_pos:
        data['Pos_X'] = m_pos.group(1)
        data['Pos_Y'] = m_pos.group(2)
        
    # Velocity
    m_vel = re.search(r"velocity of ([\d\.]+)", prompt_str)
    if m_vel: data['Velocity'] = m_vel.group(1)
    
    # Re (Can also get from dataset)
    m_re = re.search(r"Reynolds number is (\d+)", prompt_str)
    if m_re: data['Re'] = m_re.group(1)
    
    return data

def check_h5():
    file_path = "train_grid_64.h5"
    if not os.path.exists(file_path):
        print("File not found.")
        return
        
    with h5py.File(file_path, 'r') as f:
        keys = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)[:5]
        
        for k in keys:
            prompt = f[k]['prompt'][()]
            if isinstance(prompt, bytes): prompt = prompt.decode('utf-8')
            
            print(f"--- Sample {k} ---")
            print(f"Prompt: {prompt}")
            parsed = parse_prompt(prompt)
            print(f"Parsed: {parsed}")
            
            # Construct Block
            if len(parsed) == 5:
                # Re, Vel, Rad, PosX, PosY
                block = (
                    "<PHYSICS>\n"
                    f"Re = {parsed['Re']} ;\n"
                    f"Velocity = {parsed['Velocity']} ;\n"
                    f"Radius = {parsed['Radius']} ;\n"
                    f"Pos_X = {parsed['Pos_X']} ;\n"
                    f"Pos_Y = {parsed['Pos_Y']} ;\n"
                    "</PHYSICS>"
                )
                print("Block:\n" + block)
            else:
                print("FAILED to parse all fields.")

if __name__ == "__main__":
    check_h5()
