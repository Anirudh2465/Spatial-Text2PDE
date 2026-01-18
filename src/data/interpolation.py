import numpy as np
from scipy.spatial.distance import cdist

def kernel_interpolate(mesh_pos, values, grid_x, grid_y, sigma=0.05):
    """
    Interpolates unstructured mesh values onto a regular grid using Gaussian Kernel Regression.
    
    Args:
        mesh_pos (np.ndarray): (N, 2) array of node coordinates.
        values (np.ndarray): (N,) array of values at nodes.
        grid_x (np.ndarray): (H, W) array of x locations.
        grid_y (np.ndarray): (H, W) array of y locations.
        sigma (float): Bandwidth of the Gaussian kernel.
        
    Returns:
        np.ndarray: (H, W) array of interpolated values.
    """
    # Flatten the grid for vectorized distance calculation
    H, W = grid_x.shape
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1) # (H*W, 2)
    
    # Calculate distances between all grid points and all mesh nodes
    # Shape: (H*W, N)
    dists = cdist(grid_points, mesh_pos)
    
    # Compute Gaussian weights
    # w_ij = exp(-d_ij^2 / sigma^2)
    weights = np.exp(- (dists ** 2) / (sigma ** 2))
    
    # Avoid division by zero
    weight_sum = weights.sum(axis=1)
    weight_sum[weight_sum == 0] = 1.0
    
    # Weighted average
    # (H*W, N) @ (N,) -> (H*W,)
    interpolated_flat = (weights @ values) / weight_sum
    
    return interpolated_flat.reshape(H, W)
