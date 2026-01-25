"""
Adaptive Mesh Refinement for PINNs

Dynamically adds collocation points in regions with high PDE residual.
Improves accuracy where the solution has complex features.

Algorithm:
1. Sample dense candidate points
2. Compute PDE residual at each
3. Select points where |residual| > threshold
4. Add to training set (up to max_new_points limit)

Schedule: Refine every N epochs, decrease threshold over time.
"""

import torch
import numpy as np
from typing import Callable, Tuple, Optional


def sample_sphere_candidates(
    n_candidates: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample candidate points uniformly on unit sphere.
    """
    # Random uniform sampling
    u = torch.rand(n_candidates, device=device)
    v = torch.rand(n_candidates, device=device)
    
    theta = 2 * np.pi * u
    phi = torch.acos(2 * v - 1)
    
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    
    return torch.stack([x, y, z], dim=-1)


def compute_residual(
    model: torch.nn.Module,
    points: torch.Tensor,
    laplacian_fn: Callable,
    source_fn: Callable
) -> torch.Tensor:
    """
    Compute PDE residual at given points.
    
    Args:
        model: neural network
        points: [N, 3] collocation points
        laplacian_fn: function to compute Laplace-Beltrami
        source_fn: source term function
        
    Returns:
        residual: [N] absolute residual values
    """
    points = points.requires_grad_(True)
    
    # Compute Laplacian
    laplacian = laplacian_fn(model, points, create_graph=False)
    
    # Source term
    source = source_fn(points)
    
    # Residual
    residual = torch.abs(laplacian.squeeze() - source.squeeze())
    
    return residual.detach()


def adaptive_refine(
    model: torch.nn.Module,
    current_points: torch.Tensor,
    laplacian_fn: Callable,
    source_fn: Callable,
    threshold: float = 0.1,
    n_candidates: int = 5000,
    max_new_points: int = 500,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, int]:
    """
    Adaptively add collocation points based on residual.
    
    Args:
        model: neural network
        current_points: existing collocation points
        laplacian_fn: Laplace-Beltrami computation function
        source_fn: source term function
        threshold: minimum residual to add point
        n_candidates: number of candidate points to sample
        max_new_points: maximum points to add per refinement
        device: torch device
        
    Returns:
        new_points: updated collocation points
        n_added: number of points added
    """
    model.eval()
    
    with torch.no_grad():
        # Sample candidate points
        candidates = sample_sphere_candidates(n_candidates, device)
        
        # Compute residuals at candidates
        residuals = compute_residual(model, candidates, laplacian_fn, source_fn)
        
        # Find high-residual points
        high_residual_mask = residuals > threshold
        high_residual_points = candidates[high_residual_mask]
        high_residual_values = residuals[high_residual_mask]
        
        n_high = high_residual_points.shape[0]
        
        if n_high == 0:
            return current_points, 0
        
        # Select top max_new_points by residual magnitude
        if n_high > max_new_points:
            _, top_indices = torch.topk(high_residual_values, max_new_points)
            selected_points = high_residual_points[top_indices]
        else:
            selected_points = high_residual_points
        
        n_added = selected_points.shape[0]
        
        # Combine with existing points
        new_points = torch.cat([current_points, selected_points], dim=0)
    
    model.train()
    
    return new_points, n_added


class AdaptiveRefinementScheduler:
    """
    Manages adaptive refinement during training.
    
    Refines at specified intervals with decreasing threshold.
    
    Args:
        initial_threshold: starting residual threshold
        final_threshold: final threshold
        refine_every: epochs between refinements
        n_refinements: total number of refinements
        n_candidates: candidate points per refinement
        max_new_points: max points per refinement
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.1,
        final_threshold: float = 0.01,
        refine_every: int = 1000,
        n_refinements: int = 5,
        n_candidates: int = 5000,
        max_new_points: int = 500
    ):
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.refine_every = refine_every
        self.n_refinements = n_refinements
        self.n_candidates = n_candidates
        self.max_new_points = max_new_points
        
        self.refinement_count = 0
        self.total_added = 0
    
    def get_threshold(self) -> float:
        """Get current threshold based on refinement progress."""
        if self.n_refinements <= 1:
            return self.final_threshold
        
        progress = self.refinement_count / (self.n_refinements - 1)
        threshold = self.initial_threshold + \
                    progress * (self.final_threshold - self.initial_threshold)
        return threshold
    
    def should_refine(self, epoch: int) -> bool:
        """Check if refinement should occur at this epoch."""
        if self.refinement_count >= self.n_refinements:
            return False
        return epoch > 0 and epoch % self.refine_every == 0
    
    def refine(
        self,
        model: torch.nn.Module,
        current_points: torch.Tensor,
        laplacian_fn: Callable,
        source_fn: Callable,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, int]:
        """
        Perform adaptive refinement.
        
        Returns updated points and count of added points.
        """
        threshold = self.get_threshold()
        
        new_points, n_added = adaptive_refine(
            model=model,
            current_points=current_points,
            laplacian_fn=laplacian_fn,
            source_fn=source_fn,
            threshold=threshold,
            n_candidates=self.n_candidates,
            max_new_points=self.max_new_points,
            device=device
        )
        
        self.refinement_count += 1
        self.total_added += n_added
        
        print(f"Refinement {self.refinement_count}/{self.n_refinements}: "
              f"Added {n_added} points (threshold={threshold:.4f}), "
              f"Total points: {new_points.shape[0]}")
        
        return new_points, n_added
