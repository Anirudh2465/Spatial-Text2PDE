import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from tqdm import tqdm

from src.image_mllm.dataset_image import ImageCylinderDataset
from src.image_mllm.model_image import ImageMLLM

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    numeric_masks = [item['numeric_mask'] for item in batch]
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    numeric_masks_padded = pad_sequence(numeric_masks, batch_first=True, padding_value=0.0)
    
    return {
        'image': images,
        'input_ids': input_ids_padded,
        'labels': labels_padded,
        'numeric_mask': numeric_masks_padded
    }

def train():
    data_path = "train_grid_64.h5"
    if not os.path.exists(data_path):
        data_path = "../../train_grid_64.h5"
        if not os.path.exists(data_path):
            data_path = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
            
    tokenizer_path = "d:/Semester 6/Natural Language Processing/Project 3/src/mllm/mllm_tokenizer.json"
    stat_path = "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl"
    
    batch_size = 8
    accumulation_steps = 2
    epochs = 20
    lr = 2e-5
    numeric_loss_lambda = 5.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    train_dataset = ImageCylinderDataset(data_path, tokenizer_path, stat_path, num_frames=24, split='train')
    val_dataset = ImageCylinderDataset(data_path, tokenizer_path, stat_path, num_frames=24, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
    
    vocab_size = train_dataset.tokenizer.get_vocab_size()
    
    # Larger Model
    model = ImageMLLM(vocab_size=vocab_size, vision_dim=512, llm_dim=512, img_size=64, patch_size=16)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    os.makedirs("checkpoints/image_mllm", exist_ok=True)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i_batch, batch in enumerate(pbar):
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            numeric_mask = batch['numeric_mask'].to(device)
            
            with torch.amp.autocast('cuda'):
                logits, loss = model(
                    image, 
                    input_ids, 
                    targets=labels, 
                    numeric_mask=numeric_mask,
                    numeric_loss_lambda=numeric_loss_lambda
                )
                
            if loss is None or loss.item() == 0.0:
                continue
                
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            
            if (i_batch + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += (loss.item() * accumulation_steps)
            pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
            
        avg_train_loss = epoch_loss / max(1, len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                numeric_mask = batch['numeric_mask'].to(device)
                
                with torch.amp.autocast('cuda'):
                    _, loss = model(
                        image, 
                        input_ids, 
                        targets=labels, 
                        numeric_mask=numeric_mask,
                        numeric_loss_lambda=numeric_loss_lambda
                    )
                if loss is not None and loss.item() != 0.0:
                    val_loss += loss.item()
                    
        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/image_mllm/best_model.pth")
            print(f"Saved new best model with Val Loss: {best_val_loss:.4f}")
            
        torch.save(model.state_dict(), f"checkpoints/image_mllm/epoch_{epoch+1}.pth")
        
        # Validation Demo
        with torch.no_grad():
            item = val_dataset[0] # Try first frame from val set
            test_img = item['image'].unsqueeze(0).to(device)
            sos = val_dataset.tokenizer.token_to_id("[SOS]")
            prompt_ids = torch.tensor([[sos]], device=device)
            
            gen_ids = model.generate(test_img, prompt_ids, max_new_tokens=40)
            gen_text = val_dataset.tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=False)
            print(f"Sample Generation: {gen_text}")

if __name__ == "__main__":
    train()
