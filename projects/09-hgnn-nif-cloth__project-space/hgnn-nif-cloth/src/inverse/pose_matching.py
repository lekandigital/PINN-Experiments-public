"""
Pose Matching and Inverse Design Optimization.

Ported from Project 04 (ClothGeom-NIF) with enhancements for:
- Temporal models (cloth dynamics)
- Point cloud matching (in addition to SDF volume matching)
- Latent trajectory optimization for animations

Given a target cloth shape (SDF volume or point cloud), optimize a latent
code to reproduce that geometry. This enables:
- Finding latent codes for desired cloth configurations
- Fitting models to real captured cloth data
- Interpolating between matched poses for animation

Usage:
    >>> from src.inverse import PoseMatchingOptimizer, InverseDesignConfig
    >>> 
    >>> config = InverseDesignConfig(num_steps=500, lr=0.01)
    >>> optimizer = PoseMatchingOptimizer(model, config)
    >>> 
    >>> # Match to target point cloud
    >>> result = optimizer.match_pose(target_points, target_normals)
    >>> optimized_latent = result['latent']
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import logging

# Add repository root for imports
repo_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logger = logging.getLogger(__name__)


@dataclass
class InverseDesignConfig:
    """Configuration for inverse design optimization.
    
    Attributes:
        num_steps: Number of optimization steps. Default: 500.
        lr: Learning rate. Default: 0.02.
        num_points: Query points per iteration. Default: 8192.
        surface_ratio: Fraction of points near surface. Default: 0.5.
        latent_reg: Latent regularization weight. Default: 0.001.
        surface_weight: Extra weight for near-surface points. Default: 0.5.
        normal_weight: Weight for normal alignment loss. Default: 0.1.
        optimizer: Optimizer type ('adam', 'sgd', 'lbfgs'). Default: 'adam'.
        scheduler: LR scheduler ('cosine', 'step', 'none'). Default: 'cosine'.
        log_every: Log progress every N steps. Default: 50.
        early_stop_patience: Stop if no improvement for N steps. Default: 100.
        convergence_threshold: Stop if loss change below this. Default: 1e-6.
    """
    num_steps: int = 500
    lr: float = 0.02
    num_points: int = 8192
    surface_ratio: float = 0.5
    latent_reg: float = 0.001
    surface_weight: float = 0.5
    normal_weight: float = 0.1
    optimizer: str = 'adam'
    scheduler: str = 'cosine'
    log_every: int = 50
    early_stop_patience: int = 100
    convergence_threshold: float = 1e-6


class InverseDesignOptimizer:
    """
    Optimize latent codes to match target SDF volumes.
    
    This is the original approach from P04 - matching against an SDF volume
    using importance sampling near the surface.
    
    Args:
        model: Trained NIF model with interface: model(coords, latent) -> (sdf, var)
        device: Torch device.
        config: Optimization configuration.
    
    Example:
        >>> optimizer = InverseDesignOptimizer(model)
        >>> latent, losses = optimizer.optimize(target_sdf_volume)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        config: Optional[InverseDesignConfig] = None
    ):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config or InverseDesignConfig()
        
        self.model.to(self.device)
        self.model.eval()
    
    def create_query_points(
        self,
        resolution: int,
        num_points: int,
        surface_ratio: float,
        target_sdf: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """Create query points for SDF evaluation with importance sampling."""
        if target_sdf is not None and surface_ratio > 0:
            num_surface = int(num_points * surface_ratio)
            num_uniform = num_points - num_surface
            
            # Find near-surface voxels
            surface_mask = np.abs(target_sdf) < 0.1
            surface_indices = np.argwhere(surface_mask)
            
            if len(surface_indices) > num_surface:
                idx = np.random.choice(len(surface_indices), num_surface, replace=False)
                surface_indices = surface_indices[idx]
            
            coords = []
            
            # Surface points with jitter
            for voxel_idx in surface_indices:
                coord = (voxel_idx / (resolution - 1)) * 2 - 1
                coord = coord + np.random.normal(0, 0.02, 3)
                coord = np.clip(coord, -1, 1)
                coords.append(coord)
            
            # Uniform points
            uniform_pts = np.random.uniform(-1, 1, (num_uniform, 3))
            coords.extend(uniform_pts.tolist())
            
            return torch.tensor(np.array(coords), dtype=torch.float32, device=self.device)
        else:
            return torch.rand(num_points, 3, device=self.device) * 2 - 1
    
    def sample_target_sdf(
        self,
        target_sdf: np.ndarray,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """Sample SDF values at given coordinates using trilinear interpolation."""
        resolution = target_sdf.shape[0]
        
        target_tensor = torch.tensor(target_sdf, dtype=torch.float32, device=self.device)
        target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        
        sample_coords = coords.view(1, 1, 1, -1, 3)
        
        sampled = F.grid_sample(
            target_tensor,
            sample_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return sampled.view(-1)
    
    def optimize(
        self,
        target_sdf: np.ndarray,
        init_latent: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Optimize latent code to match target SDF volume.
        
        Args:
            target_sdf: Target SDF volume [D, H, W]
            init_latent: Initial latent code (random if None)
            time: Optional time value for temporal models
            verbose: Print progress
        
        Returns:
            optimized_latent: Best latent code found
            losses: List of loss values during optimization
        """
        cfg = self.config
        latent_dim = self.model.latent_dim
        resolution = target_sdf.shape[0]
        
        # Initialize latent
        if init_latent is None:
            latent = torch.randn(1, latent_dim, device=self.device) * 0.1
        else:
            latent = init_latent.clone().to(self.device)
        
        latent.requires_grad_(True)
        
        # Setup optimizer
        if cfg.optimizer == 'adam':
            optimizer = optim.Adam([latent], lr=cfg.lr)
        elif cfg.optimizer == 'sgd':
            optimizer = optim.SGD([latent], lr=cfg.lr, momentum=0.9)
        elif cfg.optimizer == 'lbfgs':
            optimizer = optim.LBFGS([latent], lr=cfg.lr, max_iter=20)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
        
        # Setup scheduler
        if cfg.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, cfg.num_steps, eta_min=cfg.lr * 0.01
            )
        elif cfg.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        else:
            scheduler = None
        
        # Optimization loop
        losses = []
        best_loss = float('inf')
        best_latent = latent.clone()
        no_improvement_count = 0
        
        iterator = tqdm(range(cfg.num_steps), desc="Optimizing") if verbose else range(cfg.num_steps)
        
        for step in iterator:
            def closure():
                optimizer.zero_grad()
                
                # Sample query points
                coords = self.create_query_points(
                    resolution, cfg.num_points, cfg.surface_ratio, target_sdf
                )
                
                # Get target values
                target_values = self.sample_target_sdf(target_sdf, coords)
                
                # Predict SDF
                latent_expanded = latent.expand(len(coords), -1)
                
                if time is not None:
                    time_expanded = time.expand(len(coords), -1)
                    pred_sdf, _ = self.model(coords, latent_expanded, time_expanded)
                else:
                    pred_sdf, _ = self.model(coords, latent_expanded)
                
                pred_sdf = pred_sdf.squeeze()
                
                # Compute losses
                sdf_loss = F.mse_loss(pred_sdf, target_values)
                
                # Surface-weighted loss
                surface_weight = torch.exp(-10 * torch.abs(target_values))
                weighted_loss = (surface_weight * (pred_sdf - target_values) ** 2).mean()
                
                # Latent regularization
                latent_reg = cfg.latent_reg * (latent ** 2).mean()
                
                loss = sdf_loss + cfg.surface_weight * weighted_loss + latent_reg
                loss.backward()
                
                return loss
            
            if cfg.optimizer == 'lbfgs':
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            loss_val = loss.item()
            losses.append(loss_val)
            
            # Track best
            if loss_val < best_loss:
                best_loss = loss_val
                best_latent = latent.clone().detach()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= cfg.early_stop_patience:
                if verbose:
                    logger.info(f"Early stopping at step {step}")
                break
            
            # Convergence check
            if len(losses) > 10:
                recent_change = abs(losses[-1] - losses[-10]) / (abs(losses[-10]) + 1e-8)
                if recent_change < cfg.convergence_threshold:
                    if verbose:
                        logger.info(f"Converged at step {step}")
                    break
            
            if verbose and (step + 1) % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0] if scheduler else cfg.lr
                logger.info(f"Step {step+1}: loss={loss_val:.6f}, lr={lr_now:.6f}")
        
        return best_latent, losses


class PoseMatchingOptimizer:
    """
    Optimize latent codes to match target point clouds.
    
    This approach matches against surface point clouds rather than SDF volumes,
    which is more flexible for real captured data.
    
    Args:
        model: Trained NIF model
        device: Torch device
        config: Optimization configuration
    
    Example:
        >>> optimizer = PoseMatchingOptimizer(model)
        >>> result = optimizer.match_pose(target_points, target_normals)
        >>> print(f"Final loss: {result['final_loss']:.6f}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        config: Optional[InverseDesignConfig] = None
    ):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config or InverseDesignConfig()
        
        self.model.to(self.device)
        self.model.eval()
    
    def match_pose(
        self,
        target_points: torch.Tensor,
        target_normals: Optional[torch.Tensor] = None,
        init_latent: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Optimize latent to match target point cloud.
        
        Args:
            target_points: [N, 3] target surface points
            target_normals: [N, 3] optional target normals
            init_latent: Starting latent code
            time: Optional time value for temporal models
            verbose: Print progress
        
        Returns:
            Dict with:
            - 'latent': Optimized latent code
            - 'loss_history': List of loss values
            - 'final_loss': Final loss value
        """
        cfg = self.config
        latent_dim = self.model.latent_dim
        
        target_points = target_points.to(self.device)
        if target_normals is not None:
            target_normals = target_normals.to(self.device)
            target_normals = target_normals / (target_normals.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Initialize latent
        if init_latent is None:
            latent = torch.randn(1, latent_dim, device=self.device) * 0.1
        else:
            latent = init_latent.clone().to(self.device)
        
        latent = nn.Parameter(latent)
        
        # Setup optimizer
        optimizer = optim.Adam([latent], lr=cfg.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.num_steps, eta_min=cfg.lr * 0.01
        )
        
        # Optimization loop
        losses = []
        best_loss = float('inf')
        best_latent = latent.data.clone()
        
        iterator = tqdm(range(cfg.num_steps), desc="Matching pose") if verbose else range(cfg.num_steps)
        
        for step in iterator:
            optimizer.zero_grad()
            
            # Sample subset of points
            n_sample = min(cfg.num_points, len(target_points))
            idx = torch.randperm(len(target_points))[:n_sample]
            pts = target_points[idx]
            norms = target_normals[idx] if target_normals is not None else None
            
            # Prepare inputs
            latent_expanded = latent.expand(n_sample, -1)
            
            # Get SDF and gradient
            pts_grad = pts.requires_grad_(True)
            
            if time is not None:
                time_expanded = time.expand(n_sample, -1)
                sdf, _ = self.model(pts_grad, latent_expanded, time_expanded, return_variance=False)
            else:
                sdf, _ = self.model(pts_grad, latent_expanded, return_variance=False)
            
            # Compute gradient for normal alignment
            gradient = torch.autograd.grad(
                sdf, pts_grad,
                grad_outputs=torch.ones_like(sdf),
                create_graph=True
            )[0]
            
            # Position loss: SDF should be ~0 at surface
            position_loss = (sdf ** 2).mean()
            
            # Normal loss
            if norms is not None:
                grad_normalized = gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-8)
                normal_loss = (1 - (grad_normalized * norms).sum(dim=-1)).mean()
            else:
                normal_loss = torch.tensor(0.0, device=self.device)
            
            # Latent regularization
            latent_reg = cfg.latent_reg * (latent ** 2).mean()
            
            # Total loss
            loss = position_loss + cfg.normal_weight * normal_loss + latent_reg
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_val = loss.item()
            losses.append(loss_val)
            
            if loss_val < best_loss:
                best_loss = loss_val
                best_latent = latent.data.clone()
            
            if verbose and (step + 1) % cfg.log_every == 0:
                logger.info(f"Step {step+1}: loss={loss_val:.6f}, pos={position_loss.item():.6f}")
        
        return {
            'latent': best_latent,
            'loss_history': losses,
            'final_loss': best_loss,
        }
    
    def interpolate_poses(
        self,
        latent_start: torch.Tensor,
        latent_end: torch.Tensor,
        n_frames: int = 30,
    ) -> List[torch.Tensor]:
        """
        Linear interpolation between two latent codes.
        
        Args:
            latent_start: Starting latent code
            latent_end: Ending latent code
            n_frames: Number of interpolation frames
        
        Returns:
            List of interpolated latent codes
        """
        latents = []
        for i in range(n_frames):
            alpha = i / (n_frames - 1)
            latent = (1 - alpha) * latent_start + alpha * latent_end
            latents.append(latent)
        return latents


def interpolate_latents(
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
    n_steps: int = 10,
    method: str = 'linear'
) -> List[torch.Tensor]:
    """
    Interpolate between two latent codes.
    
    Args:
        latent_a: First latent code
        latent_b: Second latent code
        n_steps: Number of interpolation steps
        method: 'linear' or 'spherical'
    
    Returns:
        List of interpolated latent codes
    """
    latents = []
    
    for i in range(n_steps):
        t = i / (n_steps - 1) if n_steps > 1 else 0.0
        
        if method == 'linear':
            latent = (1 - t) * latent_a + t * latent_b
        elif method == 'spherical':
            # Spherical linear interpolation (slerp)
            a_norm = latent_a / (latent_a.norm() + 1e-8)
            b_norm = latent_b / (latent_b.norm() + 1e-8)
            
            dot = (a_norm * b_norm).sum()
            dot = torch.clamp(dot, -1.0, 1.0)
            
            theta = torch.acos(dot)
            sin_theta = torch.sin(theta)
            
            if sin_theta.abs() < 1e-6:
                latent = (1 - t) * latent_a + t * latent_b
            else:
                latent = (torch.sin((1 - t) * theta) / sin_theta) * latent_a + \
                        (torch.sin(t * theta) / sin_theta) * latent_b
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        latents.append(latent)
    
    return latents
