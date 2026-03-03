import torch
import torch.nn as nn
from src.image_mllm.dataset_image import ImageCylinderDataset
from src.image_mllm.model_image import ImageMLLM
import random

def evaluate_model():
    data_path = "train_grid_64.h5"
    tokenizer_path = "src/mllm/mllm_tokenizer.json"
    stat_path = "train_normal_stat.pkl"
    checkpoint_path = "checkpoints/image_mllm/best_model.pth" # Using the best validation epoch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Needs to match the model init in train_image.py
    dataset = ImageCylinderDataset(data_path, tokenizer_path, stat_path, num_frames=24, split='test')
    vocab_size = dataset.tokenizer.get_vocab_size()
    
    model = ImageMLLM(vocab_size=vocab_size, vision_dim=512, llm_dim=512, img_size=64, patch_size=16)
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print("Successfully loaded checkpoint.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    model.to(device)
    model.eval()

    print("\n--- Model Evaluation Samples ---")
    
    # Test on a few random frames
    num_samples = 5
    indices = [random.randint(0, len(dataset)-1) for _ in range(num_samples)]
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            print(f"\nEvaluating Sample {i+1} (Dataset Index {idx}):")
            item = dataset[idx]
            
            # Ground truth prompt
            # The original implementation added special tokens for training, let's just reverse the input_ids
            target_ids = item['input_ids'].tolist()
            target_text = dataset.tokenizer.decode(target_ids, skip_special_tokens=True)
            print(f"Target Prompt:    {target_text}")
            
            # Generate from model
            image = item['image'].unsqueeze(0).to(device)
            sos = dataset.tokenizer.token_to_id("[SOS]")
            prompt_ids = torch.tensor([[sos]], device=device)
            
            gen_ids = model.generate(image, prompt_ids, max_new_tokens=80, temperature=0.0) # Greedy for evaluation
            gen_text = dataset.tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
            print(f"Generated Output: {gen_text}")
            
            print("-" * 50)

if __name__ == "__main__":
    evaluate_model()
