import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Rotary Positional Embeddings (RoPE) ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Theta for rotation (10000.0 ^ -(2i/d))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache cos/sin
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len):
        self.cache_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq) # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1) # (seq_len, dim)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        
    def forward(self, q, k):
        # q, k: (B, H, L, D)
        seq_len = q.shape[2]
        if seq_len > self.cache_seq_len:
            self._update_cache(seq_len)
            
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    def _apply_rotary(self, x, cos, sin):
        # Rotate half the dimensions: [-x2, x1, -x4, x3, ...]
        d = x.shape[-1]
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (x_rotated * sin)

# --- Causal Self Attention ---
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        
        # Causal mask buffer
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len))
                                     .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        
        # QKV
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, H, T, D)
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, H, T, D)
        
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.resid_drop(self.proj(y))

# --- Feed Forward ---
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

# --- Transformer Block ---
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# --- PhysicsGPT ---
class PhysicsGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, depth=6, num_heads=8, max_seq_len=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, inputs_embeds=None, input_ids=None):
        """
        Forward pass.
        Can provide either `inputs_embeds` (mixed vision+text) or `input_ids` (text only).
        """
        if inputs_embeds is None and input_ids is not None:
             inputs_embeds = self.token_emb(input_ids)
        
        x = inputs_embeds
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

if __name__ == "__main__":
    # Test
    model = PhysicsGPT(vocab_size=1000, embed_dim=256, depth=6, num_heads=8)
    dummy_ids = torch.randint(0, 1000, (2, 50))
    logits = model(input_ids=dummy_ids)
    print("Logits shape:", logits.shape)
    
    # Test with embeddings
    dummy_emb = torch.randn(2, 60, 256) # 10 img + 50 text
    logits_emb = model(inputs_embeds=dummy_emb)
    print("Logits from embeddings:", logits_emb.shape)
