import torch
import torch.nn as nn
from src.mllm.vision_encoder import PhysicsViT
from src.mllm.language_decoder import PhysicsGPT

class GrassmannProjector(nn.Module):
    def __init__(self, input_dim, llm_dim, num_tokens=32, mlp_depth=2):
        super().__init__()
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        
        # SVD doesn't have parameters, but we perform feature adaptation
        layers = [nn.Linear(input_dim, llm_dim), nn.GELU()]
        for _ in range(mlp_depth - 1):
            layers.append(nn.Linear(llm_dim, llm_dim))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        x: (B, L, D) -> (B, 384, 256)
        Returns: (B, K, D_llm)
        """
        # SVD
        # x = U S Vh
        # Vh: (B, D, D) (since L > D)
        # We take the top K rows of Vh, which represent the principal feature directions
        
        # SVD can be unstable if singular values are degenerate. 
        # Adding noise or using safe svd is good practice, but standard usually works with float32.
        
        try:
            # full_matrices=False -> U(B, L, D), S(B, D), Vh(B, D, D)
            _, _, Vh = torch.linalg.svd(x.float(), full_matrices=False)
        except RuntimeError: 
            # Fallback for stability (CPU fallback or noise)
            _, _, Vh = torch.linalg.svd(x.float() + 1e-4 * torch.randn_like(x), full_matrices=False)
            
        # Top K feature vectors
        # Vh shape (B, D, D). We want the first K vectors.
        # But wait, these are in D-dim space.
        # We treat them as K tokens.
        
        eigen_tokens = Vh[:, :self.num_tokens, :] # (B, K, D)
        
        # Project to LLM space
        out = self.mlp(eigen_tokens) # (B, K, LLM_Dim)
        
        return out

class PhysicsRegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1) # Scalar prediction
        )
        
    def forward(self, x):
        return self.net(x)

class PhysicsMLLM(nn.Module):
    def __init__(
        self,
        vocab_size=5000,
        vision_dim=256,
        llm_dim=256,
        num_frames=24,
        img_size=64,
        patch_size=16
    ):
        super().__init__()
        
        # 1. Vision Encoder
        self.vision_encoder = PhysicsViT(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            embed_dim=vision_dim
        )
        
        # 2. Projector (Grassmannian)
        self.projector = GrassmannProjector(
            input_dim=vision_dim,
            llm_dim=llm_dim,
            num_tokens=32, # Top-K singular vectors
            mlp_depth=2
        )
        
        # 3. Language Decoder
        self.llm = PhysicsGPT(
            vocab_size=vocab_size,
            embed_dim=llm_dim,
            depth=6,
            num_heads=8
        )
        
        # 4. Regression Head
        self.regression_head = PhysicsRegressionHead(input_dim=llm_dim)
        
        self.llm_dim = llm_dim
        
    def forward(self, image, input_ids):
        # 1. Vision Embeddings
        vis_embeds = self.vision_encoder(image) # (B, 384, Vis_Dim)
        
        # 2. Grassmann Projection
        proj_embeds = self.projector(vis_embeds) # (B, K, LLM_Dim)
        
        # 3. Regression Prediction
        # Pool over K Grassmann tokens to get a global video representation
        vis_pooled = proj_embeds.mean(dim=1) # (B, LLM_Dim)
        re_pred = self.regression_head(vis_pooled) # (B, 1)
        
        # 4. Text Embeddings
        text_embeds = self.llm.token_emb(input_ids) # (B, Seq_Len, LLM_Dim)
        
        # 5. Concatenate (Vision Prefix)
        # Sequence: [Vision Tokens] [Text Tokens]
        combined_embeds = torch.cat([proj_embeds, text_embeds], dim=1) # (B, Total_Len, LLM_Dim)
        
        # 6. Pass through LLM
        logits = self.llm(inputs_embeds=combined_embeds)
        
        return logits, re_pred
        
    @torch.no_grad()
    def generate(self, image, tokenizer, max_new_tokens=50, temperature=0.7, top_k=None):
        """
        Autoregressive Generation.
        Args:
            image: (1, 3, T, H, W)
            tokenizer: Tokenizer instance (for special tokens)
        """
        self.eval()
        
        # 1. Encode Image
        vis_embeds = self.vision_encoder(image)
        proj_embeds = self.projector(vis_embeds) # (1, V, D)
        
        # 2. Start with [SOS]
        sos_id = tokenizer.token_to_id("[SOS]")
        curr_ids = torch.tensor([[sos_id]], device=image.device) # (1, 1)
        
        generated = []
        
        for _ in range(max_new_tokens):
            # Embed current text
            text_embeds = self.llm.token_emb(curr_ids) # (1, L, D)
            
            # Combine
            combined = torch.cat([proj_embeds, text_embeds], dim=1) # (1, V+L, D)
            
            # Forward
            logits = self.llm(inputs_embeds=combined)
            
            # Get last token logits
            next_token_logits = logits[:, -1, :] # (1, Vocab)
            
            # Sampling
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            token_id = next_token.item()
            
            # Stop condition
            if tokenizer.token_to_id("[EOS]") == token_id:
                break
                
            generated.append(token_id)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
        return tokenizer.decode(generated)

if __name__ == "__main__":
    model = PhysicsMLLM()
    print("PhysicsMLLM Initialized")
