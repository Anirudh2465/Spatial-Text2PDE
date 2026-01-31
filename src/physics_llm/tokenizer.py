import re
from typing import List, Dict, Union

class PhysicsTokenizer:
    def __init__(self):
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<PHYSICS>", "</PHYSICS>"]
        self.physics_keys = ["Re", "Velocity", "Radius", "Pos_X", "Pos_Y"]
        self.symbols = ["=", ";", "."]
        self.digits = [str(i) for i in range(10)]
        self.common_words = [
            "The", "flow", "is", "laminar", "turbulent", "with", "Reynolds", "number",
            "at", "position", "radius", "velocity", "and"
        ]
        
        # Build vocabulary
        self.vocab = self.special_tokens + self.digits + self.symbols + self.physics_keys + self.common_words
        
        # Determine unique tokens and assign IDs
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        self.pad_token_id = self.token_to_id["<PAD>"]
        self.unk_token_id = self.token_to_id["<UNK>"]
        self.bos_token_id = self.token_to_id["<BOS>"]
        self.eos_token_id = self.token_to_id["<EOS>"]
        
        # Regex for tokenization
        # Captures: 
        # 1. Physics control tokens
        # 2. Physics keys (specific words)
        # 3. Digits
        # 4. Symbols
        # 5. Words (sequences of alphabets)
        # We need to be careful with order.
        
        # Escape special characters for regex
        escaped_symbols = [re.escape(s) for s in self.symbols]
        control_tokens_pattern = "|".join([re.escape(t) for t in ["<PHYSICS>", "</PHYSICS>"]])
        # Add word boundaries to keys to handle "Re" vs "Reynolds"
        keys_pattern = "|".join([r'\b' + re.escape(k) + r'\b' for k in self.physics_keys])
        
        # Combined pattern
        # We want to match:
        # - Control tokens
        # - Keys (whole word boundary?)
        # - Digits
        # - Symbols
        # - Words
        # - Whitespace (ignored)
        
        self.pattern = re.compile(
            rf'({control_tokens_pattern})|'  # Control tokens
            rf'({keys_pattern})|'            # Physics keys
            r'(\d)|'                         # Single digits
            rf'({"|".join(escaped_symbols)})|' # Symbols
            r'([a-zA-Z]+)|'                  # Words
            r'(\s+)'                         # Whitespace
        )

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        for match in self.pattern.finditer(text):
            group = match.lastgroup
            token = match.group()
            
            # If it's whitespace, skip
            if match.group(6): 
                continue
                
            # If it's a word, checking if it is in vocab is handled in encode
            tokens.append(token)
            
        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = self.tokenize(text)
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
            
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # Basic OOV handling: try to see if it's a known word-like thing or fallback to UNK
                # Since we strictly defined digits and symbols, UNK is mostly for unknown words
                ids.append(self.unk_token_id)
                
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for i in ids:
            token = self.id_to_token.get(i, "<UNK>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
            
        # Join logic is tricky because we split digits and symbols
        # A simple join with space is wrong for "230" -> "2 3 0"
        # We need a custom detokenizer or just simple space joining and post-processing?
        # For now, simple space join. Better: reconstruction logic.
        
        # Simple heuristic reconstruction:
        # - Digits next to digits: no space
        # - Symbol next to digit: might need space or not? "Re = 230"
        # - Let's just return space-separated for now and improve if needed.
        # Actually, the user validation criteria is strict. "Re = 230"
        # My tokenizer produces: ["Re", "=", "2", "3", "0"]
        # Decoding "2 3 0" to "230" logic:
        
        output = ""
        for idx, token in enumerate(tokens):
            if idx == 0:
                output += token
            else:
                prev_token = tokens[idx-1]
                # Rules for adding space
                # No space between digits
                if token.isdigit() and prev_token.isdigit():
                    output += token
                # No space between . and digit (1.20)
                elif token.isdigit() and prev_token == ".":
                    output += token
                elif token == "." and prev_token.isdigit():
                    output += token
                # Space otherwise
                else:
                    output += " " + token
        return output

    def get_vocab_size(self):
        return len(self.vocab)
