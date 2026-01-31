import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from tqdm import tqdm

from src.mllm.dataset import PhysicsMLLMDataset
from src.mllm.model import PhysicsMLLM

# Config
DATA_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
TOKENIZER_PATH = "d:/Semester 6/Natural Language Processing/Project 3/src/mllm/mllm_tokenizer.json"
STAT_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl"
SAVE_DIR = "d:/Semester 6/Natural Language Processing/Project 3/checkpoints"

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption_ids'] for item in batch]
    re_nums = torch.stack([item['reynolds_number'] for item in batch])
    pad_id = 0 
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=pad_id)
    return images, captions_padded, re_nums

def train(args):
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device} | End-to-End Mode | + Regression Head")
    
    # 1. Dataset
    dataset = PhysicsMLLMDataset(DATA_PATH, TOKENIZER_PATH, STAT_PATH, num_frames=24)
    
    pad_id = 0 
    if dataset.tokenizer.token_to_id("[PAD]") is not None:
        pad_id = dataset.tokenizer.token_to_id("[PAD]")
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # 2. Model
    model = PhysicsMLLM(
        vocab_size=dataset.tokenizer.get_vocab_size(),
        vision_dim=256,
        llm_dim=256
    ).to(device)
    
    # Load weights if resuming
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from))
    
    print("Training End-to-End (Unfrozen).")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Scaler
    if torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = torch.cuda.amp.GradScaler()
        
    criterion_text = nn.CrossEntropyLoss(ignore_index=pad_id)
    criterion_re = nn.MSELoss()
    
    # Logging Setup
    import csv
    import time
    log_file = os.path.join(SAVE_DIR, "training_log.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Duration(s)', 'Avg_Total_Loss', 'Avg_Text_Loss', 'Avg_Re_MSE'])
    
    model.train()
    
    for epoch in range(args.epochs):
        start_time = time.time()
        total_loss = 0
        total_loss_text = 0
        total_loss_re = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for step, (images, captions, re_truth) in enumerate(pbar):
            images = images.to(device)
            captions = captions.to(device)
            re_truth = re_truth.to(device).unsqueeze(1) # (B, 1)
            
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.autocast('cpu'):
                input_ids = captions[:, :-1]
                targets = captions[:, 1:]
                
                logits, re_pred = model(images, input_ids) 
                
                # Vision Length depends on Projector (K)
                vis_len = model.projector.num_tokens 
                
                text_logits = logits[:, vis_len:, :]
                
                pred_flat = text_logits.reshape(-1, text_logits.size(-1))
                targ_flat = targets.reshape(-1)
                
                loss_text = criterion_text(pred_flat, targ_flat)
                loss_re = criterion_re(re_pred, re_truth)
                
                # Scale Re loss? Re is 0-1000+, MSE will be large.
                # Let's normalize Re locally or just rely on Adam. 
                # Better to scale Re loss down or normalize Re in dataset.
                # For now, let's use a small coefficient because MSE of 100^2 is 10000.
                # If Re is ~100-1000, MSE is huge. 
                # Let's create a normalized Re target solely for training stability if possible,
                # Or just use a small weight like 1e-4. 
                # Let's use 0.001 as start.
                
                loss = loss_text + 0.001 * loss_re
                loss = loss / args.accum_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % args.accum_steps == 0:
                # Gradient Clipping for SVD stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_loss += loss.item() * args.accum_steps
            total_loss_text += loss_text.item() * args.accum_steps
            total_loss_re += loss_re.item() * args.accum_steps
            pbar.set_postfix({'loss': loss.item() * args.accum_steps, 're_mse': loss_re.item()})
            
            if args.dry_run and step >= 5:
                print("Dry run complete.")
                return
        
        avg_loss = total_loss / len(loader)
        avg_text_loss = total_loss_text / len(loader)
        avg_re_mse = total_loss_re / len(loader)
        duration = time.time() - start_time
        
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} | Text: {avg_text_loss:.4f} | Re MSE: {avg_re_mse:.4f} | Time: {duration:.2f}s")
        
        # Log to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{duration:.2f}", f"{avg_loss:.4f}", f"{avg_text_loss:.4f}", f"{avg_re_mse:.4f}"])
        
        # Save checkpont
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"mllm_epoch{epoch+1}.pth"))
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "mllm_last.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Removed --stage since we do End-to-End
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    train(args)
