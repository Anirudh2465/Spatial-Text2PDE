import torch
from src.image_mllm.dataset_image import ImageCylinderDataset

def test_ds():
    ds = ImageCylinderDataset("train_grid_64.h5", "src/mllm/mllm_tokenizer.json", "train_normal_stat.pkl", num_frames=24, split='train')
    
    print("Testing Train Split Patterns:")
    for i in range(5):
        item = ds[i]
        text = ds.tokenizer.decode(item['input_ids'].tolist(), skip_special_tokens=True)
        print(f"Sample {i}:", text)
        
    print("\nTesting Val Split (Should be static):")
    ds_val = ImageCylinderDataset("train_grid_64.h5", "src/mllm/mllm_tokenizer.json", "train_normal_stat.pkl", num_frames=24, split='val')
    for i in range(2):
        item = ds_val[i]
        text = ds_val.tokenizer.decode(item['input_ids'].tolist(), skip_special_tokens=True)
        print(f"Val {i}:", text)

if __name__ == "__main__":
    test_ds()
