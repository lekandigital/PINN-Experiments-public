"""
Sphere PINN Trainer

Trains physics-informed neural networks for PDEs on the unit sphere.
Implements the Laplace-Beltrami operator using automatic differentiation.

Key formula for unit sphere:
    Δ_S u = Δu - n^T H n - 2(n·∇u)
    
where n = x for unit sphere, H is the Hessian of u.
Since n·∇(n·∇u) = n·∇u + n^T H n on unit sphere, this simplifies to:
    Δ_S u = Δu - n·∇(n·∇u) - n·∇u

For test solution u(x,y,z) = xy (l=2 spherical harmonic):
    Δ_S(xy) = -6xy
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, Tuple, Dict
import time


class SphereMLP(nn.Module):
    """
    Simple MLP for scalar field on sphere.
    
    Args:
        in_dim: input dimension (3 for Cartesian coords)
        out_dim: output dimension (1 for scalar)
        hidden_dim: hidden layer size
        num_layers: number of hidden layers
        activation: activation function
    """
    
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        activation: str = 'tanh'
    ):
        super().__init__()
        
        if activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'silu':
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act_fn())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())
        
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_laplace_beltrami_sphere(
    model: nn.Module,
    points: torch.Tensor,
    create_graph: bool = True
) -> torch.Tensor:
    """
    Compute Laplace-Beltrami operator on unit sphere using autograd.
    
    For a surface with mean curvature H:
        Δ_S u = Δu - 2H(n·∇u) - (n·∇)(n·∇u)
    
    For unit sphere: H = 1, n = x
    
    Args:
        model: neural network u(x, y, z) -> scalar
        points: [N, 3] points on unit sphere (requires_grad=True)
        create_graph: whether to create graph for higher-order gradients
        
    Returns:
        laplacian: [N, 1] Laplace-Beltrami of u at each point
    """
    # Ensure points require gradients
    if not points.requires_grad:
        points = points.clone().requires_grad_(True)
    
    # Forward pass
    u = model(points)  # [N, 1]
    
    # Compute gradient ∇u
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=points,
        grad_outputs=torch.ones_like(u),
        create_graph=create_graph,
        retain_graph=True
    )[0]  # [N, 3]
    
    # Compute Euclidean Laplacian Δu = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²
    laplacian_euclidean = torch.zeros_like(u)
    
    for i in range(3):
        grad_u_i = grad_u[:, i:i+1]  # [N, 1]
        grad2_u_i = torch.autograd.grad(
            outputs=grad_u_i,
            inputs=points,
            grad_outputs=torch.ones_like(grad_u_i),
            create_graph=create_graph,
            retain_graph=True
        )[0]  # [N, 3]
        laplacian_euclidean += grad2_u_i[:, i:i+1]
    
    # Normal vector n = x for unit sphere
    n = points  # [N, 3]
    
    # Normal derivative: n·∇u
    n_dot_grad_u = torch.sum(n * grad_u, dim=1, keepdim=True)  # [N, 1]
    
    # Second normal derivative: (n·∇)(n·∇u)
    # This is the directional derivative of (n·∇u) in direction n
    grad_n_dot_grad_u = torch.autograd.grad(
        outputs=n_dot_grad_u,
        inputs=points,
        grad_outputs=torch.ones_like(n_dot_grad_u),
        create_graph=create_graph,
        retain_graph=True
    )[0]  # [N, 3]
    
    n_dot_grad_n_dot_grad_u = torch.sum(n * grad_n_dot_grad_u, dim=1, keepdim=True)  # [N, 1]
    
    # Key identity: n·∇(n·∇u) = n·∇u + n^T H n (because ∂n_j/∂x_i = δ_ij on unit sphere)
    # Therefore: n^T H n = n·∇(n·∇u) - n·∇u
    #
    # Correct formula: Δ_S u = Δu - n^T H n - 2(n·∇u)
    #                       = Δu - (n·∇(n·∇u) - n·∇u) - 2(n·∇u)
    #                       = Δu - n·∇(n·∇u) + n·∇u - 2(n·∇u)
    #                       = Δu - n·∇(n·∇u) - n·∇u
    laplacian_beltrami = laplacian_euclidean - n_dot_grad_n_dot_grad_u - n_dot_grad_u
    
    return laplacian_beltrami


def sample_sphere_torch(n_points: int, device: str = 'cpu') -> torch.Tensor:
    """
    Sample points uniformly on unit sphere using Fibonacci spiral.
    
    Args:
        n_points: number of points
        device: torch device
        
    Returns:
        points: [n_points, 3] on unit sphere
    """
    indices = torch.arange(n_points, dtype=torch.float32, device=device) + 0.5
    
    phi = torch.acos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    
    return torch.stack([x, y, z], dim=-1)


class SpherePINNTrainer:
    """
    Trainer for PINN on unit sphere.
    
    Trains to minimize PDE residual + supervision loss:
        Loss = mean((Δ_S u - f)²) + λ * mean((u - u_true)²)
        
    where f is the known source term and u_true is supervision data.
    
    Args:
        model: neural network
        source_fn: callable (points) -> source term f
        device: torch device
        lr: learning rate
        use_amp: use automatic mixed precision
        supervision_weight: weight for supervision loss (λ)
    """
    
    def __init__(
        self,
        model: nn.Module,
        source_fn: Callable,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 1e-3,
        use_amp: bool = False,
        supervision_weight: float = 1.0
    ):
        self.model = model.to(device)
        self.source_fn = source_fn
        self.device = device
        self.lr = lr
        self.use_amp = use_amp
        self.supervision_weight = supervision_weight
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.history = {
            'loss': [],
            'l2_error': [],
            'epoch_time': []
        }
    
    def compute_loss(
        self,
        points: torch.Tensor,
        source: torch.Tensor,
        supervision_points: torch.Tensor = None,
        supervision_values: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute PDE residual loss + optional supervision loss.
        
        Loss = mean((Δ_S u - f)²) + λ * mean((u - u_true)²)
        """
        laplacian = compute_laplace_beltrami_sphere(self.model, points)
        residual = laplacian - source
        pde_loss = torch.mean(residual ** 2)
        
        # Add supervision loss if provided
        if supervision_points is not None and supervision_values is not None:
            pred = self.model(supervision_points)
            supervision_loss = torch.mean((pred - supervision_values) ** 2)
            loss = pde_loss + self.supervision_weight * supervision_loss
        else:
            loss = pde_loss
        
        return loss
    
    def compute_l2_error(
        self,
        points: torch.Tensor,
        true_solution: torch.Tensor
    ) -> float:
        """Compute L2 error vs ground truth."""
        with torch.no_grad():
            pred = self.model(points)
            error = torch.sqrt(torch.mean((pred - true_solution) ** 2))
        return error.item()
    
    def train_epoch(
        self,
        points: torch.Tensor,
        supervision_points: torch.Tensor = None,
        supervision_values: torch.Tensor = None
    ) -> float:
        """Train for one epoch on given collocation points."""
        self.model.train()
        
        points = points.to(self.device).requires_grad_(True)
        source = self.source_fn(points).to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(points, source, supervision_points, supervision_values)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.compute_loss(points, source, supervision_points, supervision_values)
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def train(
        self,
        n_points: int = 1000,
        n_epochs: int = 100,
        log_every: int = 20,
        true_solution_fn: Optional[Callable] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 50,
        n_supervision: int = 100
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            n_points: number of collocation points
            n_epochs: training epochs
            log_every: logging frequency
            true_solution_fn: ground truth for L2 error and supervision
            checkpoint_path: where to save checkpoints
            checkpoint_every: checkpoint frequency
            n_supervision: number of supervision points (for uniqueness)
            
        Returns:
            training history
        """
        # Sample collocation points
        points = sample_sphere_torch(n_points, self.device)
        
        # Setup supervision points if ground truth available
        supervision_points = None
        supervision_values = None
        if true_solution_fn is not None:
            # Sample separate supervision points
            supervision_points = sample_sphere_torch(n_supervision, self.device)
            supervision_values = true_solution_fn(supervision_points)
        
        # Get ground truth for L2 error evaluation
        if true_solution_fn is not None:
            true_solution = true_solution_fn(points)
        
        print(f"Training PINN on sphere ({n_points} points, {n_epochs} epochs)")
        print(f"Device: {self.device}, AMP: {self.use_amp}")
        print("-" * 50)
        
        initial_loss = None
        
        for epoch in range(n_epochs):
            start_time = time.time()
            
            loss = self.train_epoch(points, supervision_points, supervision_values)
            
            epoch_time = time.time() - start_time
            
            self.history['loss'].append(loss)
            self.history['epoch_time'].append(epoch_time)
            
            if initial_loss is None:
                initial_loss = loss
            
            # Compute L2 error
            if true_solution_fn is not None:
                l2_error = self.compute_l2_error(points, true_solution)
                self.history['l2_error'].append(l2_error)
            else:
                l2_error = None
            
            # Logging
            if (epoch + 1) % log_every == 0 or epoch == 0:
                msg = f"Epoch {epoch+1:4d}/{n_epochs} | Loss: {loss:.6f}"
                if l2_error is not None:
                    msg += f" | L2 Error: {l2_error:.6f}"
                msg += f" | Time: {epoch_time:.3f}s"
                print(msg)
            
            # Checkpointing
            if checkpoint_path and (epoch + 1) % checkpoint_every == 0:
                self.save_checkpoint(checkpoint_path)
        
        print("-" * 50)
        print(f"Training complete!")
        print(f"Initial loss: {initial_loss:.6f} -> Final loss: {self.history['loss'][-1]:.6f}")
        if self.history['l2_error']:
            print(f"Final L2 error: {self.history['l2_error'][-1]:.6f}")
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from {path}")
