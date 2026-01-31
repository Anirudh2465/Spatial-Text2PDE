import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVisionEncoder(nn.Module):
    """
    Simulates a vision encoder or acts as a placeholder if we use synthetic features.
    Input: (B, T, D_in) 
    Output: (B, T, D_out)
    """
    def __init__(self, input_dim=64, output_dim=256):
        super().__init__()
        # Simple MLP per frame for now, simulating frame processing
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        # x: (B, T, input_dim)
        return self.net(x)

class GrassmannProjector(nn.Module):
    """
    Projects temporal sequence to top-K basis vectors (tokens).
    """
    def __init__(self, input_dim=256, k_tokens=8, proj_dim=256):
        super().__init__()
        self.k = k_tokens
        # Optional projection to match LLM dim if different from input_dim
        self.output_proj = nn.Linear(input_dim, proj_dim) if input_dim != proj_dim else nn.Identity()

    def forward(self, x):
        """
        x: (B, T, D)
        Returns: (B, K, D_proj)
        """
        # SVD on the last two dimensions (T, D)
        # We want to find the Top-K components.
        # Usuaully we treat T as observations and D as features?
        # If we want tokens that represent the "subspace", we typically want the
        # basis vectors in the feature space (V) or temporal coefficients?
        # The prompt says: "Extract top-K temporal basis vectors ... Each basis vector = one vision token"
        # and "Result: K vision tokens".
        
        # If we take V (D x D), the columns are the principal axes in feature space.
        # If we take U (T x T), the columns are the principal axes in time.
        # Since the token must be input to the LLM, it must have dimension D (or proj_dim).
        # So we likely want the V vectors (size D).
        # However, we have K tokens.
        
        # Let's perform SVD: X = U S V^T
        # X: (B, T, D)
        
        # PyTorch SVD returns U, S, V.
        # U: (B, T, T), S: (B, min(T, D)), V: (B, D, D)
        # Note: torch.linalg.svd returns Vh (V transpose) usually.
        
        try:
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        except RuntimeError: 
            # Fallback for stability if needed, or add jitter
            U, S, Vh = torch.linalg.svd(x + 1e-6 * torch.randn_like(x), full_matrices=False)
            
        # Vh is (B, min(T,D), D). Rows of Vh are the eigenvectors of X^T X (Feature correlations)
        # We take the top K rows of Vh.
        
        k = min(self.k, Vh.size(1))
        
        # These are the top K feature basis vectors.
        # Shape: (B, K, D)
        basis_vectors = Vh[:, :k, :] 
        
        # We might want to scale them by singular values to retain magnitude info?
        # "Each basis vector = one vision token".
        # If we just take the direction, we lose importance.
        # Let's scale by S.
        
        # S: (B, min(T,D))
        s_k = S[:, :k].unsqueeze(-1) # (B, K, 1)
        
        weighted_basis = basis_vectors * s_k
        
        # Project to LLM dim
        out = self.output_proj(weighted_basis) # (B, K, Projection)
        
        # If k < self.k, pad?
        if k < self.k:
            pad = torch.zeros(x.size(0), self.k - k, out.size(2), device=x.device)
            out = torch.cat([out, pad], dim=1)
            
        return out
