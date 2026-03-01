import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.physics_llm.model import PhysicsGPT, PhysicsGPTConfig
from src.physics_llm.tokenizer import PhysicsTokenizer
from src.physics_llm.dataset_real import RealPhysicsDataset
from src.physics_llm.vision_real import PhysicsViT 
from src.physics_llm.vision import GrassmannProjector

def collate_fn(batch):
    # Dynamic padding for inputs
    # batch is list of dicts
    max_len = max([item['input_ids'].size(0) for item in batch])
    
    input_ids_list = []
    labels_list = []
    vision_list = []
    numeric_mask_list = []
    
    for item in batch:
        l = item['input_ids'].size(0)
        pad_len = max_len - l
        
        # Pad Inputs (0 usually PAD)
        # Using 0 as PAD
        padded_ids = torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)])
        input_ids_list.append(padded_ids)
        
        # Pad Labels (-100)
        padded_labels = torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)])
        labels_list.append(padded_labels)
        
        # Vision (C, T, H, W) -> Stack directly
        vision_list.append(item['vision_tensor'])
        
        # Numeric Mask (0)
        padded_mask = torch.cat([item['numeric_mask'], torch.zeros(pad_len, dtype=torch.float)])
        numeric_mask_list.append(padded_mask)
        
    return {
        'input_ids': torch.stack(input_ids_list),
        'vision_tensor': torch.stack(vision_list),
        'labels': torch.stack(labels_list),
        'numeric_mask': torch.stack(numeric_mask_list)
    }

def train_real():
    batch_size = 4 # Video memory heavy?
    max_epochs = 5 # Retuning
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints/physics_llm'
    
    tokenizer = PhysicsTokenizer()
    dataset = RealPhysicsDataset("train_grid_64.h5", tokenizer)
    
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Model
    # We need to handle potentially longer prompts (real data prompts are wordy)
    config = PhysicsGPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_len=256, # Increased for safety
        n_embd=256,
        n_head=4,
        n_layer=4,
        dropout=0.1
    )
    model = PhysicsGPT(config).to(device)
    
    # Load Phase 3 Checkpoint (LLM + Weights)
    # Ignore vision_encoder weights as we are changing architecture (Simple -> PhysicsViT)
    phase3_path = os.path.join(save_dir, 'phase3_checkpoint.pth')
    if os.path.exists(phase3_path):
        ckpt = torch.load(phase3_path)
        state_dict = ckpt['model']
        
        # Handle position embedding resizing
        pos_weight = state_dict['position_embedding.weight']
        old_len = pos_weight.size(0)
        new_len = 256 # Config max_len
        
        if old_len != new_len:
            print(f"Resizing pos emb from {old_len} to {new_len}")
            # Create new larger embedding
            new_pos = model.position_embedding.weight.data.clone()
            # Copy overlap
            min_len = min(old_len, new_len)
            new_pos[:min_len] = pos_weight[:min_len]
            # Update state dict
            state_dict['position_embedding.weight'] = new_pos
            
        model.load_state_dict(state_dict, strict=False) 
        print("Loaded Phase 3 LLM weights (resized).")
    else:
        print("Warning: Phase 3 weights not found. Starting fresh.")
    
    # Real Vision Encoder
    vision_encoder = PhysicsViT(img_size=64, num_frames=24, embed_dim=256).to(device)
    grassmann = GrassmannProjector(input_dim=256, k_tokens=8, proj_dim=256).to(device)
    
    # Ops
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-5},
        {'params': vision_encoder.parameters(), 'lr': 1e-4}, # Fresh
        {'params': grassmann.parameters(), 'lr': 1e-4}
    ])
    
    physics_end_token_id = tokenizer.token_to_id.get("</PHYSICS>")
    
    model.train()
    
    for epoch in range(max_epochs):
        total_loss = 0
        steps = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            vision_tensor = batch['vision_tensor'].to(device) # (B, C, T, H, W)
            labels = batch['labels'].to(device)
            numeric_mask = batch['numeric_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward Vision
            # PhysicsViT returns (B, 384, 256)
            vis_feat = vision_encoder(vision_tensor)
            
            # Grassmann Projection -> (B, K, 256)
            vis_tokens = grassmann(vis_feat) 
            
            logits, loss = model(
                input_ids,
                vision_embeds=vis_tokens,
                targets=labels,
                numeric_mask=numeric_mask,
                physics_end_token_id=physics_end_token_id
            )
            
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
                
        print(f"Epoch {epoch+1}/{max_epochs} | Loss: {total_loss/steps:.4f}")
        
    # Save Real Checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'vision_encoder': vision_encoder.state_dict(),
        'grassmann': grassmann.state_dict()
    }
    torch.save(checkpoint, os.path.join(save_dir, 'real_model.pth'))
    print("Real Data Training Complete.")

if __name__ == "__main__":
    train_real()
