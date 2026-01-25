"""
Mesh PINN Trainer

Trains physics-informed neural networks on triangle mesh domains.
Uses DEC operators for discrete Laplacian computation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, Dict
import time

from ..layers.dec_operators import DECLaplacian, build_dec_operators


class MeshMLP(nn.Module):
    """
    MLP for scalar/vector field on mesh.
    Takes 3D coordinates, outputs field values.
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
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeshPINNTrainer:
    """
    Trainer for PINN on triangle mesh.
    
    Uses DEC Laplacian for PDE residual computation.
    Supports both Dirichlet and Neumann boundary conditions.
    
    Args:
        model: neural network
        vertices: mesh vertices (numpy)
        faces: mesh faces (numpy)
        boundary_mask: boolean mask for boundary vertices
        source_fn: source term function
        bc_type: 'dirichlet' or 'neumann'
        bc_value_fn: boundary condition value function
        device: torch device
        lr: learning rate
    """
    
    def __init__(
        self,
        model: nn.Module,
        vertices: np.ndarray,
        faces: np.ndarray,
        boundary_mask: np.ndarray,
        source_fn: Callable,
        bc_type: str = 'dirichlet',
        bc_value_fn: Optional[Callable] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 1e-3,
        lambda_bc: float = 100.0  # BC weight in loss
    ):
        self.model = model.to(device)
        self.device = device
        self.source_fn = source_fn
        self.bc_type = bc_type
        self.bc_value_fn = bc_value_fn
        self.lambda_bc = lambda_bc
        
        # Build DEC Laplacian
        self.laplacian = DECLaplacian(vertices, faces, normalized=True, device=device)
        
        # Store mesh data
        self.vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
        self.boundary_mask = torch.tensor(boundary_mask, device=device)
        self.interior_mask = ~self.boundary_mask
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        self.history = {
            'loss': [],
            'pde_loss': [],
            'bc_loss': [],
            'l2_error': []
        }
    
    def compute_pde_loss(self) -> torch.Tensor:
        """
        Compute PDE residual loss on interior vertices.
        
        Loss = mean((L*u - f)² for interior vertices)
        """
        u = self.model(self.vertices).squeeze()  # [N]
        
        # Apply DEC Laplacian
        Lu = self.laplacian(u)  # [N]
        
        # Source term
        f = self.source_fn(self.vertices).squeeze()  # [N]
        
        # Residual on interior only
        residual = (Lu - f)[self.interior_mask]
        
        return torch.mean(residual ** 2)
    
    def compute_bc_loss(self) -> torch.Tensor:
        """
        Compute boundary condition loss.
        """
        if not self.boundary_mask.any():
            return torch.tensor(0.0, device=self.device)
        
        boundary_vertices = self.vertices[self.boundary_mask]
        u_boundary = self.model(boundary_vertices).squeeze()
        
        if self.bc_type == 'dirichlet':
            # u = g on boundary
            if self.bc_value_fn is not None:
                g = self.bc_value_fn(boundary_vertices).squeeze()
            else:
                g = torch.zeros_like(u_boundary)
            
            bc_loss = torch.mean((u_boundary - g) ** 2)
        
        elif self.bc_type == 'neumann':
            # ∂u/∂n = h on boundary (would need normal derivative)
            # Simplified: just penalize deviation from BC values
            if self.bc_value_fn is not None:
                h = self.bc_value_fn(boundary_vertices).squeeze()
            else:
                h = torch.zeros_like(u_boundary)
            bc_loss = torch.mean((u_boundary - h) ** 2)
        
        else:
            bc_loss = torch.tensor(0.0, device=self.device)
        
        return bc_loss
    
    def compute_loss(self) -> tuple:
        """Compute total loss = PDE loss + lambda * BC loss."""
        pde_loss = self.compute_pde_loss()
        bc_loss = self.compute_bc_loss()
        total_loss = pde_loss + self.lambda_bc * bc_loss
        return total_loss, pde_loss, bc_loss
    
    def train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, pde_loss, bc_loss = self.compute_loss()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), pde_loss.item(), bc_loss.item()
    
    def compute_l2_error(self, true_solution_fn: Callable) -> float:
        """Compute L2 error vs ground truth."""
        with torch.no_grad():
            pred = self.model(self.vertices).squeeze()
            true = true_solution_fn(self.vertices).squeeze()
            error = torch.sqrt(torch.mean((pred - true) ** 2))
        return error.item()
    
    def train(
        self,
        n_epochs: int = 1000,
        log_every: int = 100,
        true_solution_fn: Optional[Callable] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 500
    ) -> Dict:
        """
        Full training loop.
        """
        print(f"Training Mesh PINN ({self.vertices.shape[0]} vertices, {n_epochs} epochs)")
        print(f"Device: {self.device}")
        print("-" * 50)
        
        for epoch in range(n_epochs):
            loss, pde_loss, bc_loss = self.train_epoch()
            
            self.history['loss'].append(loss)
            self.history['pde_loss'].append(pde_loss)
            self.history['bc_loss'].append(bc_loss)
            
            if true_solution_fn is not None:
                l2_error = self.compute_l2_error(true_solution_fn)
                self.history['l2_error'].append(l2_error)
            else:
                l2_error = None
            
            if (epoch + 1) % log_every == 0 or epoch == 0:
                msg = f"Epoch {epoch+1:4d}/{n_epochs} | Loss: {loss:.6f} | PDE: {pde_loss:.6f} | BC: {bc_loss:.6f}"
                if l2_error is not None:
                    msg += f" | L2: {l2_error:.6f}"
                print(msg)
            
            if checkpoint_path and (epoch + 1) % checkpoint_every == 0:
                torch.save(self.model.state_dict(), checkpoint_path)
        
        print("-" * 50)
        print(f"Training complete! Final loss: {self.history['loss'][-1]:.6f}")
        
        return self.history
