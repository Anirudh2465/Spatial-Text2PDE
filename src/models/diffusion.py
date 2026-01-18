import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # t: (N,) tensor.
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, context_dim=768, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, kdim=context_dim, vdim=context_dim)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        # Weights show fc1, fc2. Sequential maps to 0, 2 usually (with activation at 1).
        # We might need to name them explicitly if loading strictly, or map.
        
        # adaLN_modulation: Regresses 6 parameters * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, t_emb):
        # x: (N, L, D), c: (N, L_c, D_c), t_emb: (N, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        # Self Attention
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        # Note: Checkpoint keys for attn are `qkv` [3072, 1024].
        # Torch MultiheadAttention uses `in_proj_weight` [3072, 1024]. Matches!
        # But split q,k,v keys?
        # Checkpoint: `attn.qkv.weight`. (Not q, k, v separate).
        # Torch uses one matrix `in_proj_weight`. This works.
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # Cross Attention (No modulation usually on cross attn? Or same modulation?)
        # Keys show `cross_attn.to_q`, `to_k`, `to_v`.
        # Torch MultiheadAttention uses `in_proj_weight` IF kdim=vdim=embed_dim.
        # But here context_dim=768.
        # Torch uses `q_proj_weight`, `k_proj_weight`, `v_proj_weight` separating them.
        # Keys match: `to_q`, `to_k` etc.
        # I need to implement Custom CrossAttention or map Torch Linear layers.
        # Torch MHA does not expose `to_q` etc. 
        # I will implement Custom Attention to be exact.
        
        # ... Wait, if I implement Custom Block, I verify easier.
        pass # To be fleshed out below. Since Torch MHA naming is rigid.

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim, bias=False) # Checkpoint shows to_q weight, no bias? Wait.
        # Keys: cross_attn.to_q.weight (no bias key in dump for to_q? Dump had bias for `attn.qkv.bias` but `cross_attn.to_out.0.bias`.
        # Wait, check dump (Step 491):
        # `cross_attn.to_q.weight` OK.
        # `cross_attn.to_q.bias` MISSING! -> Bias=False.
        # `cross_attn.to_k.weight`. Bias missing.
        # `cross_attn.to_v.weight`. Bias missing.
        # `cross_attn.to_out.0.weight`, `bias`. -> Linear+Bias.
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim)) # Sequential because key is `to_out.0`

    def forward(self, x, context):
        h = self.num_heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape
        # q: (B, N, C) -> (B, Heads, N, Dim)
        q = q.view(x.shape[0], x.shape[1], h, -1).permute(0, 2, 1, 3)
        k = k.view(context.shape[0], context.shape[1], h, -1).permute(0, 2, 1, 3)
        v = v.view(context.shape[0], context.shape[1], h, -1).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.to_out(out)

class DiTBlockFinal(nn.Module):
    def __init__(self, hidden_size, num_heads, context_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.cross_attn = CrossAttention(hidden_size, context_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, hidden_size)
        )
        # Checkpoint: `mlp.fc1`, `mlp.fc2`.
        # Sequential keys: `mlp.0`, `mlp.2`.
        # I will map `mlp.0` -> `mlp.fc1` in loading.
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, t_emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        # Self Attn
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # Cross Attn
        x = x + self.cross_attn(x, c)
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)

    def forward(self, x, t_emb):
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x

class DiffusionTransformer(nn.Module):
    def __init__(self, 
                 input_size=(6, 16, 16), # T, H, W (Latent)
                 patch_size=2, 
                 in_channels=16, 
                 hidden_size=1024, 
                 depth=28, 
                 num_heads=16, 
                 context_dim=768):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.input_size = input_size
        
        # Embedding
        self.x_embedder = nn.ModuleDict() 
        # Checkpoint: `x_embedder.proj.weight`
        self.x_embedder['proj'] = nn.Conv3d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Y Embedder (Text)
        # Checkpoint: `y_embedder.0`, `y_embedder.2`.
        self.y_embedder = nn.Sequential(
            nn.Linear(context_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_size), requires_grad=True) # Matches 512 length
        
        self.blocks = nn.ModuleList([
            DiTBlockFinal(hidden_size, num_heads, context_dim) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels=in_channels) # Out channels = In channels (Velocity prediction)

    def forward(self, x, t, context):
        # x: (B, C, T, H, W)
        # t: (B,)
        # context: (B, L, C_ctx)
        
        # Embed Inputs
        x = self.x_embedder['proj'](x) # (B, Hidden, T', H', W')
        # Flatten
        x = x.flatten(2).transpose(1, 2) # (B, N, Hidden)
        
        # Pos Embed (Add)
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Embed Time/Context
        t_emb = self.t_embedder(t) # (B, Hidden)
        
        # Y embed (Global from context mean? Or context is sequence?)
        # Checkpoint has `y_embedder`.
        # DiT usually conditions on Class Label (Y).
        # But this is Text-to-3D?
        # If `context` is (B, L, 768).
        # `y_embedder` input is 768.
        # Maybe it takes pooled text embedding?
        # Let's assume `context_mean` is passed to `y_embedder` for Modulation?
        # But wait, `blocks` use `adaLN_modulation(t_emb)`. `t_emb` is sum of t and y?
        # Common pattern: `c = t_embed(t) + y_embed(y)`.
        # `adaLN(c)`.
        # `y_embedder` output is `hidden_size`.
        # So we add them.
        
        # Pooled context
        if context.ndim == 3:
            y = context.mean(dim=1) # (B, 768)
        else:
            y = context
            
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb ## Summing time and class/text embeddings
        
        for block in self.blocks:
            x = block(x, context, c) # Pass Sequence `context` for CrossAttn, `c` for Modulation
            
        x = self.final_layer(x, c)
        
        # Unpatchify
        # x: (B, N, Patch^3 * C)
        # Reshape to (B, C, T, H, W)
        B = x.shape[0]
        P = self.patch_size
        C = self.in_channels
        T = self.input_size[0] // P
        H = self.input_size[1] // P
        W = self.input_size[2] // P
        
        x = x.transpose(1, 2).reshape(B, C, P, P, P, T, H, W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(B, C, T*P, H*P, W*P)
        return x

    def load_from_ckpt(self, ckpt_path):
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'state_dict' in sd: sd = sd['state_dict']
        
        # Filter for 'model.diffusion_model'
        new_sd = {}
        for k, v in sd.items():
            if 'model.diffusion_model' in k:
                name = k.replace('model.diffusion_model.', '')
                
                # Maps
                # x_embedder.proj -> matches
                # t_embedder.mlp.0 -> matches Sequential
                # y_embedder.0 -> matches Sequential
                # blocks.0... -> matches
                # blocks.0.mlp.fc1 -> mlp.0
                # blocks.0.mlp.fc2 -> mlp.2
                name = name.replace('mlp.fc1', 'mlp.0').replace('mlp.fc2', 'mlp.2')
                new_sd[name] = v
        
        self.load_state_dict(new_sd, strict=False)
        print("DiT Weights Loaded.")

class DDIMSampler:
    def __init__(self, model, ckpt_path=None, num_steps=1000, ddim_steps=50):
        self.model = model
        self.num_steps = num_steps
        self.ddim_steps = ddim_steps
        self.device = next(model.parameters()).device
        
        # Default schedule (Linear)
        beta_start = 0.0001
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Load from Checkpoint if provided (Overwrites defaults)
        if ckpt_path:
            self.load_schedule(ckpt_path)
            
        # DDIM Timesteps (Uniform spacing)
        c = num_steps // ddim_steps
        self.ddim_timesteps = torch.arange(0, num_steps, c, device=self.device).flip(0) # [980, ..., 0] for 50 steps
    
    def load_schedule(self, ckpt_path):
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'state_dict' in sd: sd = sd['state_dict']
        
        # Check for alphas_cumprod
        # Keys seen: `sqrt_alphas_cumprod` (size 1000)
        if 'sqrt_alphas_cumprod' in sd:
            # alphas_cumprod = sqrt^2
            sqrt_inv = sd['sqrt_alphas_cumprod']
            # Wait, key is `sqrt_alphas_cumprod`. Is it sqrt(alpha_bar)?
            # Yes. so alpha_bar = key**2.
            self.alphas_cumprod = (sqrt_inv.to(self.device) ** 2)
            self.num_steps = len(self.alphas_cumprod)
            print("Loaded alphas_cumprod from checkpoint.")
            
    def refine(self, z_coarse, prompt_emb, strength=0.5):
        # SDEdit:
        # 1. Add noise to z_coarse to timestep t_start
        # 2. Denoise from t_start to 0
        
        batch_size = z_coarse.shape[0]
        
        # t_start index
        t_start_idx = int(strength * len(self.ddim_timesteps))
        # Clamp
        t_start_idx = min(t_start_idx, len(self.ddim_timesteps) - 1)
        
        # Actual timestep
        t_start = self.ddim_timesteps[t_start_idx]
        
        # Get alpha_bar at t_start
        alpha_bar = self.alphas_cumprod[t_start]
        sqrt_alpha = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar)
        
        # Noise
        noise = torch.randn_like(z_coarse)
        z_t = sqrt_alpha * z_coarse + sqrt_one_minus_alpha * noise
        
        # Denoise Loop
        print(f"Refining from t={t_start.item()} (Step {t_start_idx}/{len(self.ddim_timesteps)})...")
        z = z_t
        
        # Iterate from t_start_idx to End
        for i in range(t_start_idx, len(self.ddim_timesteps)):
            t = self.ddim_timesteps[i]
            prev_t = self.ddim_timesteps[i+1] if i < len(self.ddim_timesteps) - 1 else torch.tensor(0, device=self.device)
            
            z = self.p_sample_ddim(z, t, prev_t, prompt_emb)
            
        return z

    @torch.no_grad()
    def p_sample_ddim(self, x_t, t, prev_t, context, eta=0.0):
        # Model Prediction (v-prediction or eps-prediction?)
        # Checkpoint: `final_layer` output matches input.
        # Assuming epsilon-prediction (standard LDM).
        
        t_batch = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
        
        model_output = self.model(x_t, t_batch, context)
        # Assuming model outputs epsilon
        
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod[prev_t]
        
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        
        # Pred x0
        pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * model_output) / torch.sqrt(alpha_bar_t)
        
        # Direction to xt
        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
        
        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1 - alpha_bar_prev - sigma_t**2)
        
        x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + sqrt_one_minus_alpha_bar_prev * model_output + sigma_t * torch.randn_like(x_t)
        return x_prev
