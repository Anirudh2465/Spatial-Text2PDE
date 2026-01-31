import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add project root to path if needed, though running as module is preferred
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.physics_llm.model import PhysicsGPT, PhysicsGPTConfig
from src.physics_llm.tokenizer import PhysicsTokenizer
from src.physics_llm.dataset import PhysicsDataset

def calculate_numeric_accuracy(logits, targets, numeric_mask):
    # logits: (b, t, v)
    # targets: (b, t) -> shifted in loop usually, but here passed aligned (prediction for next)
    # Actually in the loop:
    # logits (prediction for next step) vs targets (next step)
    
    # helper for the training loop where we have shifting manually or handled
    # Model returns logits (b, t, v).
    # We need to compare logits[:, :-1] argmax with targets[:, 1:]
    
    preds = torch.argmax(logits, dim=-1) # (b, t)
    
    # We need to align with how loss was calculated
    # Loss calculated on shift_logits = logits[:, :-1] and shift_labels = targets[:, 1:]
    
    shift_preds = preds[:, :-1]
    shift_targets = targets[:, 1:]
    shift_mask = numeric_mask[:, 1:]
    
    correct = (shift_preds == shift_targets) & (shift_mask == 1.0)
    total_numeric = shift_mask.sum()
    
    if total_numeric == 0:
        return 1.0 # No numerics to predict?
    
    return correct.sum().float() / total_numeric

def train():
    # settings
    batch_size = 32
    max_epochs = 50
    learning_rate = 3e-4
    numeric_loss_lambda = 5.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'checkpoints/physics_llm'
    os.makedirs(save_dir, exist_ok=True)

    print(f"Using device: {device}")

    # Data & Model
    tokenizer = PhysicsTokenizer()
    dataset = PhysicsDataset(tokenizer, size=2000, max_len=128)
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    config = PhysicsGPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_len=128,
        n_embd=256,
        n_head=4,
        n_layer=4,
        dropout=0.1
    )
    model = PhysicsGPT(config).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training Loop
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        total_numeric_acc = 0
        steps = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            numeric_mask = batch['numeric_mask'].to(device)
            
            optimizer.zero_grad()
            
            logits, loss = model(input_ids, targets=labels, numeric_mask=numeric_mask, numeric_loss_lambda=numeric_loss_lambda)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # monitor numeric accuracy
            with torch.no_grad():
                acc = calculate_numeric_accuracy(logits, labels, numeric_mask)
                total_numeric_acc += acc.item()
            
            steps += 1
            
        avg_loss = total_loss / steps
        avg_acc = total_numeric_acc / steps
        
        print(f"Epoch {epoch+1}/{max_epochs} | Loss: {avg_loss:.4f} | Numeric Acc: {avg_acc:.4f}")
        
        # Validation & Generation Check
        if (epoch + 1) % 5 == 0 or avg_acc > 0.99:
            evaluate(model, val_dataset, tokenizer, device)
            
        if avg_acc > 0.999: # Early stop if perfect
            print("Converged to >99.9% numeric accuracy.")
            break
            
    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, 'phase1_model.pth'))
    print("Training Complete. Model Saved.")

def evaluate(model, dataset, tokenizer, device):
    model.eval()
    print("\n--- Validation Generation ---")
    
    # Pick a few samples
    indices = [0, 1, 2] # just first few
    
    for i in indices:
        if i >= len(dataset): break
        
        # Get raw sample from dataset (using __getitem__ logic but we want just the prompt)
        # We need to construct the prompt manually or find the split point in the input_ids
        
        item = dataset[i]
        input_ids = item['input_ids'].to(device)
        
        # Find split point </PHYSICS>
        end_tag_id = tokenizer.token_to_id["</PHYSICS>"]
        try:
            split_pos = (input_ids == end_tag_id).nonzero(as_tuple=True)[0][0].item()
            prompt_ids = input_ids[:split_pos+1].unsqueeze(0) # include </PHYSICS>
        except IndexError:
            continue
            
        # Ground Truth
        # We need special tokens to find the delimiter
        full_text = tokenizer.decode(item['input_ids'].tolist(), skip_special_tokens=False)
        target_parts = full_text.split("</PHYSICS>")
        if len(target_parts) < 2:
            print(f"DEBUG: </PHYSICS> not found in target. Full text: {full_text[:50]}...")
            continue
            
        target_caption = target_parts[-1].strip()
        # Remove EOS/PAD from target_caption if present
        target_caption = target_caption.replace("<EOS>", "").replace("<PAD>", "").strip()
        
        # Generate
        generated_ids = model.generate(prompt_ids, max_new_tokens=50, temperature=0, top_k=None)
        
        # Truncate at EOS
        if tokenizer.eos_token_id in generated_ids[0]:
            eos_idx = (generated_ids[0] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
            generated_ids = generated_ids[:, :eos_idx]
            
        generated_text_full = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
        generated_parts = generated_text_full.split("</PHYSICS>")
        
        if len(generated_parts) < 2:
             generated_caption = "ERROR_NO_DELIMITER"
        else:
             generated_caption = generated_parts[-1].strip()
             generated_caption = generated_caption.replace("<EOS>", "").replace("<PAD>", "").strip()
        
        print(f"Input Physics: [Hidden]") 
        
        # Extract Re from Prompt for sanity check
        # Naive extraction from text
        prompt_text = tokenizer.decode(prompt_ids[0].tolist())
        re_match = "Re = " in prompt_text
        
        print(f"Target:    {target_caption}")
        print(f"Generated: {generated_caption}")
        
        if target_caption == generated_caption:
            print("Result: PASS Exact Match")
        else:
            print("Result: FAIL Mismatch")
            # Debug split
            if "</PHYSICS>" not in full_text:
                print("DEBUG: </PHYSICS> token missing in decoded text")
            
    print("-----------------------------\n")

if __name__ == '__main__':
    train()
