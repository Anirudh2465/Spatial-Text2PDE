"""
grassmann.py --- Grassmann Manifold Operations for Cylinder Flow

G(k, n): the set of all k-dimensional linear subspaces of R^n.
A point is represented by an n x k orthonormal matrix U (U^T U = I_k).

All functions are fully differentiable via torch.linalg.svd.

NOTE on rank and cylinder flow:
    The input is x in (B, T, M, C) with C=3 channels.
    A per-time-step SVD of F in (M, C) gives at most min(M, C) = 3 modes.
    To get higher-rank (k > C) representations we use the SPATIO-TEMPORAL
    formulation: reshape (T, M, C) -> (M, T*C) before SVD, giving up to
    min(M, T*C) modes. This is equivalent to Proper Orthogonal Decomposition
    applied to the whole temporal trajectory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pure manifold math
# ---------------------------------------------------------------------------

def grassmann_project(F: torch.Tensor, k: int):
    """
    Project a field matrix onto G(k, M) via thin SVD.
    k will be automatically clamped to min(k, min(M, C)).

    Args:
        F:  (..., M, C)  field on M nodes with C channels
        k:  int          number of dominant modes to retain

    Returns:
        U:  (..., M, k_eff)  Grassmann point (dominant spatial modes)
        S:  (..., k_eff)     singular values (mode energy)
        Vh: (..., k_eff, C)  mode mixing over channels
    """
    M, C = F.shape[-2], F.shape[-1]
    k_eff = min(k, M, C)
    U, S, Vh = torch.linalg.svd(F, full_matrices=False)
    return U[..., :k_eff], S[..., :k_eff], Vh[..., :k_eff, :]


def retract_qr(U: torch.Tensor) -> torch.Tensor:
    """
    QR retraction: project an arbitrary matrix onto the Stiefel manifold
    (orthonormal columns) using thin QR decomposition.

    Args:
        U: (..., n, k)  possibly non-orthonormal matrix

    Returns:
        Q: (..., n, k)  columns are orthonormal
    """
    Q, R = torch.linalg.qr(U, mode='reduced')
    # Fix sign ambiguity so diagonal of R is positive
    signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))  # (..., k)
    # Replace zero signs with 1
    signs = signs + (signs == 0).float()
    signs = signs.unsqueeze(-2)  # (..., 1, k)
    return Q * signs


def log_map(Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map: Log_Y(X) -- maps X in G(k,n) to the tangent space at Y.

    Algorithm (Edelman et al.):
        M         = (I - Y Y^T) X (Y^T X)^{-1}
        U, Theta, V^T = SVD(M)
        Log_Y(X) = U * arctan(Theta) * V^T

    Args:
        Y: (..., n, k)  base point (orthonormal columns)
        X: (..., n, k)  target point (orthonormal columns)

    Returns:
        Delta: (..., n, k)  tangent vector at Y
    """
    YtX = Y.transpose(-1, -2) @ X   # (..., k, k)
    # (I - Y Y^T) X = X - Y (Y^T X)
    M = X - Y @ YtX
    # M (Y^T X)^{-1}  -- use pinv for numerical stability
    YtX_inv = torch.linalg.pinv(YtX)
    M = M @ YtX_inv
    # Clamp to avoid numerical issues near zero
    U, Sigma, Vh_m = torch.linalg.svd(M, full_matrices=False)
    arctan_Sigma = torch.arctan(Sigma.clamp(-20.0, 20.0))
    Delta = U * arctan_Sigma.unsqueeze(-2) @ Vh_m
    return Delta


def exp_map(Y: torch.Tensor, Delta: torch.Tensor) -> torch.Tensor:
    """
    Exponential map: Exp_Y(Delta) -- maps tangent vector Delta at Y back to G(k,n).

    Algorithm:
        U, S, V^T = SVD(Delta)
        Exp_Y(Delta) = Y * V * cos(S) * V^T + U * sin(S) * V^T

    Args:
        Y:     (..., n, k)  base point
        Delta: (..., n, k)  tangent vector at Y

    Returns:
        X: (..., n, k)  point on the manifold (orthonormal columns)
    """
    U, Sigma, Vh = torch.linalg.svd(Delta, full_matrices=False)
    cos_S = torch.cos(Sigma)
    sin_S = torch.sin(Sigma)
    V = Vh.transpose(-1, -2)
    term1 = Y @ (V * cos_S.unsqueeze(-2)) @ Vh
    term2 = (U * sin_S.unsqueeze(-2)) @ Vh
    return term1 + term2


def geodesic_distance(Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Geodesic (chordal) distance between two Grassmann points via principal angles.

    cos(theta_i) = sigma_i(Y^T X)
    d(Y, X)      = ||[theta_1, ..., theta_k]||_2

    Args:
        Y: (..., n, k)  point on G(k, n)
        X: (..., n, k)  point on G(k, n)

    Returns:
        dist: (...)  geodesic distance (scalar per batch element)
    """
    M = Y.transpose(-1, -2) @ X  # (..., k, k)
    # svdvals is more numerically stable than svd then taking S
    singular_vals = torch.linalg.svdvals(M)  # (..., k)
    # Tight clamp for acos stability
    singular_vals = singular_vals.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angles = torch.acos(singular_vals)
    return torch.norm(angles, p=2, dim=-1)


def grassmann_mean(points: torch.Tensor, n_iters: int = 10, tol: float = 1e-6) -> torch.Tensor:
    """
    Karcher (Frechet) mean on G(k, n) via iterative gradient descent.

    Args:
        points: (N, n, k)  set of Grassmann points
        n_iters: int        maximum number of iterations
        tol:     float      convergence tolerance

    Returns:
        mean: (n, k)  Karcher mean
    """
    N = points.shape[0]
    mu = points[0].clone()

    for _ in range(n_iters):
        logs = torch.stack([
            log_map(mu.unsqueeze(0), points[i].unsqueeze(0)).squeeze(0)
            for i in range(N)
        ], dim=0)
        grad   = logs.mean(dim=0)
        mu_new = exp_map(mu.unsqueeze(0), grad.unsqueeze(0)).squeeze(0)
        delta  = torch.norm(mu_new - mu)
        mu     = mu_new
        if delta < tol:
            break

    return mu


# ---------------------------------------------------------------------------
# nn.Module wrappers
# ---------------------------------------------------------------------------

class GrassmannTangentMLP(nn.Module):
    """
    Riemannian residual block that applies an MLP in the tangent space at Y.

    Args:
        n:          ambient dimension (mesh nodes)
        k:          subspace dimension (rank)
        hidden_dim: hidden width of the MLP
        step_size:  learnable scale for the tangent update
    """

    def __init__(self, n: int, k: int, hidden_dim: int = 128, step_size: float = 0.1):
        super().__init__()
        self.n = n
        self.k = k
        dim = n * k
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.tau = nn.Parameter(torch.tensor(step_size))

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Y: (B, n, k)  -> Y_new: (B, n, k)
        """
        B = Y.shape[0]
        flat = Y.reshape(B, -1)
        delta_flat = self.mlp(flat)
        delta = delta_flat.reshape(B, self.n, self.k)
        Y_new = exp_map(Y, self.tau * delta)
        Y_new = retract_qr(Y_new)
        return Y_new


class GrassmannProjector(nn.Module):
    """
    nn.Module: spatio-temporal SVD projection of mesh fields.

    Uses the SPATIO-TEMPORAL formulation to allow rank >> C:
        x (B, T, M, C)  ->  reshape each sample to (M, T*C)
                        ->  thin SVD
                        ->  U (B, M, k), S (B, k), context (B, k, T*C)

    Then we split context back into per-time-step S' and Vh' for
    compatibility with ModeEmbedder:
        U:  (B, T, M, k)  -- same U broadcast across T (spatial modes)
        S:  (B, T, k)     -- per-frame energy (from per-frame Vh projection)
        Vh: (B, T, k, C)  -- per-frame mode-channel mixing

    Args:
        rank: int  number of dominant modes k
    """

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, M, C)

        Returns:
            U:  (B, T, M, k)   spatial basis (broadcast across T)
            S:  (B, T, k)      per-frame mode energies
            Vh: (B, T, k, C)   per-frame mode-channel matrix
        """
        B, T, M, C = x.shape
        k_eff = min(self.rank, M, T * C)

        # --- Spatio-temporal matrix: (B, M, T*C)
        F_st = x.permute(0, 2, 1, 3).reshape(B, M, T * C)  # (B, M, T*C)
        U_st, _, _ = torch.linalg.svd(F_st, full_matrices=False)
        U_basis = U_st[:, :, :k_eff]  # (B, M, k_eff)  -- global spatial basis

        # --- Per-frame projections to get per-timestep S and Vh
        # Project each frame's field onto the basis: coeff = U^T x_t
        # x_t: (B, M, C)  -> coeff = (B, k_eff, C),  S_t = ||coeff||
        # x: (B, T, M, C) -> (B, T, M, C)
        U_t = U_basis.unsqueeze(1).expand(B, T, M, k_eff)  # (B, T, M, k)
        # coeff_t = U^T x_t = (B, T, k_eff, C)
        coeff_t = U_t.transpose(-1, -2) @ x  # (B, T, k, C)
        # Singular values per frame: norm across C dimension
        S_t = torch.norm(coeff_t, dim=-1)  # (B, T, k)
        S_t = S_t.clamp(min=1e-8)
        # Normalised Vh per frame: (B, T, k, C)
        Vh_t = coeff_t / S_t.unsqueeze(-1)

        # U is the same spatial basis for all T
        U_out = U_t  # (B, T, M, k_eff)
        return U_out, S_t, Vh_t


class GrassmannReconstructor(nn.Module):
    """
    nn.Module: reconstruct physical fields from Grassmann decomposition.

    x_rec = U * diag(S) * Vh

    Input : U  (B, T, M, k)
            S  (B, T, k)
            Vh (B, T, k, C)
    Output: x_rec  (B, T, M, C)
    """

    def __init__(self):
        super().__init__()

    def forward(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor) -> torch.Tensor:
        US = U * S.unsqueeze(-2)   # (B, T, M, k)
        return US @ Vh              # (B, T, M, C)
