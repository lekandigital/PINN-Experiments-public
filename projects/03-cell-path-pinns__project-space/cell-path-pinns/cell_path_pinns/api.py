"""
CellPathModel API Wrapper

High-level interface for training and using Cell-Path PINNs.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Tuple, Union, List, Dict, Any
import warnings

from .models import PathNet, PotentialNet, get_device
from .losses import compute_losses
from .data_utils import normalize_trajectory, denormalize_trajectory


class CellPathModel:
    """
    High-level API for Cell-Path PINNs trajectory prediction.
    
    Example:
        >>> from cell_path_pinns import CellPathModel, generate_synthetic_trajectory
        >>> 
        >>> # Generate data
        >>> t, x, y = generate_synthetic_trajectory(n_steps=200)
        >>> 
        >>> # Train model
        >>> model = CellPathModel(device='cuda')
        >>> model.fit(t, x, y, epochs=200)
        >>> 
        >>> # Predict
        >>> t_pred = np.linspace(0, 20, 100)
        >>> xy_pred = model.predict_paths(t_pred)
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        device: Optional[str] = None,
        lambda_data: float = 1.0,
        lambda_geo: float = 0.1,
        lambda_chem: float = 1.0,
        target_speed: float = 1.0,
        use_amp: bool = False
    ):
        """
        Initialize CellPathModel.

        Args:
            hidden_dim: Hidden layer size for both networks
            device: 'cuda', 'cpu', or None for auto-detect
            lambda_data: Weight for data fitting loss
            lambda_geo: Weight for geodesic (constant-speed) loss
            lambda_chem: Weight for chemotactic loss
            target_speed: Target speed for geodesic constraint
            use_amp: Enable automatic mixed precision training (requires CUDA)
        """
        self.device = get_device(device)
        self.hidden_dim = hidden_dim

        # Loss weights
        self.lambda_data = lambda_data
        self.lambda_geo = lambda_geo
        self.lambda_chem = lambda_chem
        self.target_speed = target_speed

        # AMP support
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Initialize networks
        self.path_net = PathNet(hidden_dim=hidden_dim).to(self.device)
        self.potential_net = PotentialNet(hidden_dim=hidden_dim).to(self.device)

        # Normalization parameters (set during fit)
        self.norm_params = None

        # Training history
        self.history = {
            'loss': [],
            'loss_data': [],
            'loss_geo': [],
            'loss_chem': [],
            'lr': [],
        }

        # Training state
        self._is_fitted = False
    
    def fit(
        self,
        t_data: np.ndarray,
        x_data: np.ndarray,
        y_data: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
        verbose: bool = True,
        print_every: int = 20,
        normalize: bool = True,
        grad_clip: Optional[float] = 1.0,
        use_scheduler: bool = False,
        T_max: Optional[int] = None,
        save_checkpoints: bool = False,
        checkpoint_dir: str = 'results/checkpoints',
        checkpoint_every: int = 50
    ) -> 'CellPathModel':
        """
        Train model on trajectory data.

        Args:
            t_data: Time points array
            x_data: X positions array
            y_data: Y positions array
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Print training progress
            print_every: Print frequency
            normalize: Normalize data to [0,1] range
            grad_clip: Gradient clipping norm (None to disable)
            use_scheduler: Use cosine annealing LR scheduler
            T_max: Scheduler period (defaults to epochs if None)
            save_checkpoints: Save model checkpoints during training
            checkpoint_dir: Directory to save checkpoints
            checkpoint_every: Save checkpoint every N epochs

        Returns:
            self (for method chaining)
        """
        # Flatten inputs
        t_data = np.asarray(t_data).flatten()
        x_data = np.asarray(x_data).flatten()
        y_data = np.asarray(y_data).flatten()

        # Normalize if requested
        if normalize:
            t_norm, x_norm, y_norm, self.norm_params = normalize_trajectory(
                t_data, x_data, y_data
            )
        else:
            t_norm, x_norm, y_norm = t_data, x_data, y_data
            self.norm_params = None

        # Convert to tensors
        t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1).to(self.device)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).view(-1, 1).to(self.device)
        y_tensor = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1).to(self.device)

        # Enable gradients for time tensor
        t_tensor.requires_grad_(True)

        # Get parameters
        parameters = list(self.path_net.parameters()) + list(self.potential_net.parameters())

        # Optimizer
        optimizer = torch.optim.Adam(parameters, lr=lr)

        # Learning rate scheduler
        scheduler = None
        if use_scheduler:
            T_max = T_max or epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=lr / 100)

        # Create checkpoint directory if needed
        if save_checkpoints:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Training loop
        self.path_net.train()
        self.potential_net.train()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    total_loss, loss_data, loss_geo, loss_chem = compute_losses(
                        self.path_net,
                        self.potential_net,
                        t_tensor,
                        x_tensor,
                        y_tensor,
                        lambda_data=self.lambda_data,
                        lambda_geo=self.lambda_geo,
                        lambda_chem=self.lambda_chem,
                        target_speed=self.target_speed
                    )

                # Backward with gradient scaling
                self.scaler.scale(total_loss).backward()

                # Unscale before clipping
                self.scaler.unscale_(optimizer)

                # Gradient clipping
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(parameters, grad_clip)

                # Optimizer step with scaler
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                total_loss, loss_data, loss_geo, loss_chem = compute_losses(
                    self.path_net,
                    self.potential_net,
                    t_tensor,
                    x_tensor,
                    y_tensor,
                    lambda_data=self.lambda_data,
                    lambda_geo=self.lambda_geo,
                    lambda_chem=self.lambda_chem,
                    target_speed=self.target_speed
                )

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(parameters, grad_clip)

                optimizer.step()

            # Update scheduler
            if scheduler is not None:
                scheduler.step()
                self.history['lr'].append(scheduler.get_last_lr()[0])
            else:
                self.history['lr'].append(lr)

            # Record history
            self.history['loss'].append(total_loss.item())
            self.history['loss_data'].append(loss_data.item())
            self.history['loss_geo'].append(loss_geo.item())
            self.history['loss_chem'].append(loss_chem.item())

            # Save checkpoint
            if save_checkpoints and (epoch + 1) % checkpoint_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1:04d}.pt')
                self.save(checkpoint_path)
                if verbose:
                    print(f"  Saved checkpoint: {checkpoint_path}")

            # Print progress
            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                lr_str = f", lr={self.history['lr'][-1]:.2e}" if use_scheduler else ""
                print(f"Epoch {epoch:4d}: Loss = {total_loss.item():.6f} "
                      f"(data={loss_data.item():.4f}, geo={loss_geo.item():.4f}, "
                      f"chem={loss_chem.item():.4f}{lr_str})")

        self._is_fitted = True
        return self
    
    def predict_paths(
        self,
        t_sequence: np.ndarray
    ) -> np.ndarray:
        """
        Forecast positions for given time points.
        
        Args:
            t_sequence: Array-like of time points
            
        Returns:
            numpy array of shape (len(t_sequence), 2) with (x, y) positions
        """
        if not self._is_fitted:
            warnings.warn("Model has not been fitted. Predictions may be random.")
        
        self.path_net.eval()
        
        # Flatten input
        t_sequence = np.asarray(t_sequence).flatten()
        
        # Normalize time if we have normalization params
        if self.norm_params is not None:
            t_norm = (t_sequence - self.norm_params['t_min']) / self.norm_params['t_range']
        else:
            t_norm = t_sequence
        
        # Predict
        with torch.no_grad():
            t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1).to(self.device)
            xy_pred = self.path_net(t_tensor).cpu().numpy()
        
        # Denormalize if needed
        if self.norm_params is not None:
            xy_pred[:, 0] = xy_pred[:, 0] * self.norm_params['x_range'] + self.norm_params['x_min']
            xy_pred[:, 1] = xy_pred[:, 1] * self.norm_params['y_range'] + self.norm_params['y_min']
        
        return xy_pred
    
    def get_potential_field(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        t: float,
        resolution: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Visualize the learned nutrient field at a given time.
        
        Args:
            x_range: (x_min, x_max) for visualization
            y_range: (y_min, y_max) for visualization
            t: Time point
            resolution: Grid resolution
            
        Returns:
            Tuple of (X_grid, Y_grid, U_values) for plotting with plt.contourf
        """
        self.potential_net.eval()
        
        # Create grid
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Flatten for network input
        x_flat = X.flatten()
        y_flat = Y.flatten()
        t_flat = np.full_like(x_flat, t)
        
        # Normalize if needed
        if self.norm_params is not None:
            x_norm = (x_flat - self.norm_params['x_min']) / self.norm_params['x_range']
            y_norm = (y_flat - self.norm_params['y_min']) / self.norm_params['y_range']
            t_norm = (t_flat - self.norm_params['t_min']) / self.norm_params['t_range']
        else:
            x_norm, y_norm, t_norm = x_flat, y_flat, t_flat
        
        # Compute potential
        with torch.no_grad():
            x_tensor = torch.tensor(x_norm, dtype=torch.float32).view(-1, 1).to(self.device)
            y_tensor = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1).to(self.device)
            t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1).to(self.device)
            
            U = self.potential_net(x_tensor, y_tensor, t_tensor).cpu().numpy()
        
        # Reshape to grid
        U = U.reshape(resolution, resolution)
        
        return X, Y, U
    
    def save(self, path: str):
        """Save model to file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Convert norm_params numpy values to Python floats for safe loading
        norm_params_safe = None
        if self.norm_params is not None:
            norm_params_safe = {k: float(v) for k, v in self.norm_params.items()}

        torch.save({
            'path_net_state': self.path_net.state_dict(),
            'potential_net_state': self.potential_net.state_dict(),
            'hidden_dim': self.hidden_dim,
            'norm_params': norm_params_safe,
            'lambda_data': self.lambda_data,
            'lambda_geo': self.lambda_geo,
            'lambda_chem': self.lambda_chem,
            'target_speed': self.target_speed,
            'use_amp': self.use_amp,
            'history': {k: [float(v) for v in vals] for k, vals in self.history.items()},
        }, path)

    def load(self, path: str):
        """Load model from file."""
        # Use weights_only=False for backward compatibility with numpy-containing checkpoints
        # This is safe for our own checkpoints
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Recreate networks if hidden_dim changed
        if checkpoint['hidden_dim'] != self.hidden_dim:
            self.hidden_dim = checkpoint['hidden_dim']
            self.path_net = PathNet(hidden_dim=self.hidden_dim).to(self.device)
            self.potential_net = PotentialNet(hidden_dim=self.hidden_dim).to(self.device)

        self.path_net.load_state_dict(checkpoint['path_net_state'])
        self.potential_net.load_state_dict(checkpoint['potential_net_state'])
        self.norm_params = checkpoint['norm_params']
        self.lambda_data = checkpoint['lambda_data']
        self.lambda_geo = checkpoint['lambda_geo']
        self.lambda_chem = checkpoint['lambda_chem']
        self.target_speed = checkpoint['target_speed']
        self.use_amp = checkpoint.get('use_amp', False)
        self.history = checkpoint['history']
        self._is_fitted = True

        # Recreate scaler if AMP is enabled
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        return self
    
    def get_training_history(self) -> dict:
        """Return training history dictionary."""
        return self.history.copy()
    
    def compute_mse(
        self,
        t_data: np.ndarray,
        x_data: np.ndarray,
        y_data: np.ndarray
    ) -> float:
        """
        Compute mean squared error on given data.
        
        Args:
            t_data: Time points
            x_data: True x positions
            y_data: True y positions
            
        Returns:
            MSE value
        """
        xy_pred = self.predict_paths(t_data)
        x_pred = xy_pred[:, 0]
        y_pred = xy_pred[:, 1]
        
        mse = np.mean((x_pred - x_data)**2 + (y_pred - y_data)**2)
        return mse
