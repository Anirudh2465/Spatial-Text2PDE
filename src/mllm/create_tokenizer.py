import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import h5py
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from src.mllm.label_generator import generate_description

DATA_PATH = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
TOKENIZER_PATH = "d:/Semester 6/Natural Language Processing/Project 3/src/mllm/mllm_tokenizer.json"

def train_tokenizer():
    print("Collecting corpus...")
    corpus = []
    
    # 1. Physics Terms
    physics_terms = [
        "vortex", "laminar", "Reynolds", "shedding", "unsteady", "steady", 
        "turbulence", "fluid", "dynamics", "cylinder", "flow", "wake", 
        "velocity", "pressure", "magnitude", "oscillation", "boundary", "layer",
        "viscosity", "incompressible", "Navier-Stokes", "simulation", "2D", "field"
    ]
    # Repeat important terms to ensure they are tokenized well
    corpus.extend(physics_terms * 10)
    
    # 2. Generate descriptions from dataset
    with h5py.File(DATA_PATH, 'r') as f:
        keys = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)
        # Sample a subset to save time if huge
        for k in keys[:500]: 
            # meta = f[k]['metadata'] # ERROR: Metadata was flattened
            # re = meta['reynolds_number'][()]
            # prompt = meta['prompt'][()]
            
            re = f[k]['reynolds_number'][()]
            prompt = f[k]['prompt'][()]
            
            desc = generate_description({'reynolds_number': re, 'prompt': prompt})
            corpus.append(desc)
            
    print(f"Corpus size: {len(corpus)} sentences.")
    
    # 3. Initialize Tokenizer (ByteLevel BPE)
    # ByteLevel handles whitespace as a character, allowing proper reconstruction
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 4. Trainer
    special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[SEP]", "<IMG>"]
    trainer = trainers.BpeTrainer(
        vocab_size=5000, 
        special_tokens=special_tokens,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel().alphabet()
    )
    
    # 5. Train
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    
    # 6. Post-Processing
    tokenizer.decoder = decoders.ByteLevel()
    
    # Save
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer saved to {TOKENIZER_PATH}")
    
    # Verification
    encoded = tokenizer.encode("Vortex shedding at high Reynolds number.")
    print("Tokens:", encoded.tokens)
    print("IDs:", encoded.ids)

if __name__ == "__main__":
    train_tokenizer()
