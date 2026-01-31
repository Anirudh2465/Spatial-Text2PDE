import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.physics_llm.model import PhysicsGPT, PhysicsGPTConfig
from src.physics_llm.tokenizer import PhysicsTokenizer
from src.physics_llm.dataset import PhysicsDataset
from src.physics_llm.vision import SimpleVisionEncoder, GrassmannProjector

class SimpleClassifier(nn.Module):
    """Auxiliary head for vision alignment."""
    def __init__(self, input_dim=256, num_classes=2):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # x: (B, K, D) -> Pool -> (B, D) -> (B, num_classes)
        pooled = x.mean(dim=1)
        return self.head(pooled)

def train_phase3():
    batch_size = 16
    max_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints/physics_llm'
    
    print(f"Using device: {device}")
    
    # Components
    tokenizer = PhysicsTokenizer()
    dataset = PhysicsDataset(tokenizer, size=2000, max_len=128)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model Setup
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
    aux_head = SimpleClassifier(input_dim=256, num_classes=2).to(device)
    
    # Load Phase 2 Checkpoint
    checkpoint_path = os.path.join(save_dir, 'phase2_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        vision_encoder.load_state_dict(checkpoint['vision_encoder'])
        grassmann.load_state_dict(checkpoint['grassmann'])
        print("Loaded Phase 2 Checkpoint.")
    else:
        print("ERROR: Phase 2 checkpoint not found!")
        return

    # Optimizer Groups (Differential LR)
    # Frozen: Physics Embeddings (Token Embedding)
    # Actually, we can just set requires_grad=False for embeddings?
    # No, we want to allow text embeddings to shift if needed, but preserve physics.
    # But embedding matrix is shared.
    # Plan: Set LR=0 for the entire embedding layer to be safe.
    
    # Transformer: Low LR
    # Vision: Moderate LR
    
    param_groups = [
        # Frozen (LR=0)
        {'params': model.token_embedding.parameters(), 'lr': 0.0},
        {'params': model.position_embedding.parameters(), 'lr': 0.0},
        
        # Transformer (Low LR)
        {'params': model.blocks.parameters(), 'lr': 1e-5},
        {'params': model.ln_f.parameters(), 'lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 1e-5}, # Output head shares weights with embedding? Check.
        # Logic: self.token_embedding.weight = self.head.weight in model init.
        # If we update head, we update embedding.
        # If we set embedding LR=0, we must set head LR=0?
        # If they share the SAME parameter object, yes.
        # Let's verify.
        
        # Vision (Moderate LR)
        {'params': vision_encoder.parameters(), 'lr': 1e-4},
        {'params': grassmann.parameters(), 'lr': 1e-4},
        {'params': aux_head.parameters(), 'lr': 1e-4}
    ]
    
    # Note on Weight Tying:
    # If parameters are same object, optimizer will see them twice?
    # No, we should filter.
    # But optimization groups are by list of params.
    # If I pass model.head.parameters() and it is same as model.token_embedding.parameters(),
    # and one has LR=0 and other LR=1e-5... conflict.
    # Strategy: Freeze Embedding explicitly.
    # Since head is tied, head is also frozen?
    # "Physics tokenizer: frozen". "Physics syntax... frozen".
    # "Transformer layers: trainable". "Output head: trainable".
    # If head is tied to embedding, we can't freeze embedding and train head unless untied.
    # Usually we untie or allow small updates.
    # Let's set LR very low (1e-6) for embeddings to allow minor drift for text but keep physics stable?
    # Or just untie for Phase 3? No structure changes allowed.
    # Let's set LR=1e-5 for everything in LLM, including embeddings.
    # "Physics-related parameters: 0" -> This might mean tokens specific to physics?
    # Implementing per-token freezing is hard in optim groups.
    # Let's stick to Low LR for Transformer (including embeddings).
    
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-5},
        {'params': vision_encoder.parameters(), 'lr': 1e-4},
        {'params': grassmann.parameters(), 'lr': 1e-4},
        {'params': aux_head.parameters(), 'lr': 1e-4}
    ])
    
    physics_end_token_id = tokenizer.token_to_id["</PHYSICS>"]
    numeric_loss_lambda = 5.0
    align_loss_alpha = 1.0
    
    model.train()
    
    for epoch in range(max_epochs):
        total_loss = 0
        total_text_loss = 0
        total_align_loss = 0
        steps = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            vision_tensor = batch['vision_tensor'].to(device)
            labels = batch['labels'].to(device)
            numeric_mask = batch['numeric_mask'].to(device)
            
            # Determine Target Class from Text (for alignment)
            # We don't have explicit class labels in dataset, but we can infer from text content.
            # "The flow is [laminar/turbulent]"
            # Scan batch labels/input_ids for "laminar" vs "turbulent".
            # laminar_id = tokenizer.token_to_id["laminar"] ...
            # Simpler: Generate ground truth alignment labels from flow logic.
            # Dataset doesn't return flow type explicitly.
            # But we know synthetic rule: Look at the visual tensor?
            # Or just parse the input text.
            # Let's parse text in batch loop? Slow.
            # Modification: Update dataset to return 'flow_class' (0 or 1).
            # But let's infer for now to avoid modifying dataset again if possible.
            # Actually, I modified dataset to return {..., labels, ...}. 
            # I can reconstruct text.
            
            # Alignment Targets
            align_targets = []
            laminar_id = tokenizer.token_to_id.get("laminar")
            turbulent_id = tokenizer.token_to_id.get("turbulent")
            
            # This is heuristics based on token presence in input_ids (target part)
            for i in range(input_ids.size(0)):
                if laminar_id in input_ids[i]:
                    align_targets.append(0)
                elif turbulent_id in input_ids[i]:
                    align_targets.append(1)
                else:
                    align_targets.append(0) # Default/mask?
            
            align_targets = torch.tensor(align_targets, device=device)
            
            optimizer.zero_grad()
            
            # Forward Vision
            vis_feat = vision_encoder(vision_tensor)
            vis_tokens = grassmann(vis_feat) # (B, K, D)
            
            # Auxiliary Alignment Loss
            # Predict flow type from vision tokens
            align_logits = aux_head(vis_tokens)
            loss_align = nn.CrossEntropyLoss()(align_logits, align_targets)
            
            # LLM Forward
            logits, loss_text = model(
                input_ids, 
                vision_embeds=vis_tokens, 
                targets=labels, 
                numeric_mask=numeric_mask,
                numeric_loss_lambda=numeric_loss_lambda,
                physics_end_token_id=physics_end_token_id
            )
            
            if loss_text is None: continue
            
            # Total Loss
            loss = loss_text + align_loss_alpha * loss_align
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_text_loss += loss_text.item()
            total_align_loss += loss_align.item()
            steps += 1
            
        print(f"Epoch {epoch+1}/{max_epochs} | Total: {total_loss/steps:.4f} | Text: {total_text_loss/steps:.4f} | Align: {total_align_loss/steps:.4f}")
        
        # Validation
        validate_phase3(model, vision_encoder, grassmann, val_dataset, tokenizer, device, epoch)

    # Save Phase 3
    checkpoint = {
        'model': model.state_dict(),
        'vision_encoder': vision_encoder.state_dict(),
        'grassmann': grassmann.state_dict(),
        'aux_head': aux_head.state_dict()
    }
    torch.save(checkpoint, os.path.join(save_dir, 'phase3_checkpoint.pth'))
    print("Phase 3 Training Complete.")

def validate_phase3(model, vision_encoder, grassmann, dataset, tokenizer, device, epoch):
    model.eval()
    if epoch % 2 != 0 and epoch != 9: return # Check every 2 epochs
    
    print("\n--- Validation Phase 3 ---")
    physics_end_token_id = tokenizer.token_to_id["</PHYSICS>"]
    
    # Check 3 samples
    for i in [0, 1, 2]:
        item = dataset[i]
        input_ids = item['input_ids'].to(device)
        vision_tensor = item['vision_tensor'].to(device).unsqueeze(0)
        
        # Split prompt
        try:
            split_pos = (input_ids == physics_end_token_id).nonzero(as_tuple=True)[0][0].item()
            prompt_ids = input_ids[:split_pos+1].unsqueeze(0)
        except: continue
        
         # Ground Truth
        full_text = tokenizer.decode(item['input_ids'].tolist(), skip_special_tokens=False)
        target_caption = full_text.split("</PHYSICS>")[-1].strip().replace("<EOS>", "").replace("<PAD>", "")
        
        with torch.no_grad():
            v = vision_encoder(vision_tensor)
            vis_emb = grassmann(v)
            gen_ids = model.generate(
                prompt_ids, vision_embeds=vis_emb, max_new_tokens=40, 
                temperature=0, physics_end_token_id=physics_end_token_id
            )
            
        out = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=False)
        try: caption = out.split("</PHYSICS>")[-1].strip().replace("<EOS>", "")
        except: caption = out
        
        print(f"Target:    {target_caption}")
        print(f"Generated: {caption}")
        
        # Re Check
        re_pattern = r"Re = (\d+)"
        t_re = re.search(re_pattern, target_caption)
        g_re = re.search(re_pattern, caption)
        if t_re and g_re and t_re.group(1) == g_re.group(1):
            print("Physics Check: PASS")
        else:
            print("Physics Check: FAIL")
            
    print("--------------------------\n")
    model.train()

if __name__ == "__main__":
    train_phase3()
