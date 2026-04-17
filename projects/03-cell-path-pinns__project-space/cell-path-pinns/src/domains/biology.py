"""
Biology Domain Adapter: Microbe Chemotaxis.

Models microbe trajectories as geodesics on nutrient concentration fields.
The original Cell-Path-PINNs domain, now refactored to use the core abstractions.

Key Physics:
- Microbes move toward high nutrient concentration (chemotaxis)
- Movement follows gradient of nutrient field
- Constant speed constraint (geodesic motion)

Potential Convention:
- φ = -log(concentration) so gradient descent → concentration ascent
- Low potential = high nutrients = attractive region
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn

from ..core.potential_field import PotentialFieldBase, LearnedPotentialField
from ..core.geodesic_loss import GeodesicLoss, GradientFollowingLoss


class NutrientField(PotentialFieldBase):
    """
    Nutrient concentration field for chemotaxis modeling.
    
    The potential is defined as φ = -log(c) where c is nutrient concentration.
    This means:
    - High concentration → low potential → attractive
    - Gradient descent on φ = gradient ascent on concentration
    
    Architecture: (x, y, t) → hidden → hidden → concentration → -log(c)
    """
    
    def __init__(
        self,
        spatial_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        time_dependent: bool = True,
        min_concentration: float = 1e-6,
    ):
        """
        Initialize nutrient field.
        
        Args:
            spatial_dim: Spatial dimension (typically 2 for petri dish)
            hidden_dim: Hidden layer width
            num_layers: Number of hidden layers
            time_dependent: Whether concentration varies with time
            min_concentration: Minimum concentration (prevents log(0))
        """
        super().__init__(spatial_dim=spatial_dim, time_dependent=time_dependent)
        
        self.hidden_dim = hidden_dim
        self.min_concentration = min_concentration
        
        # Input: spatial coords + time (if time-dependent)
        input_dim = spatial_dim + (1 if time_dependent else 0)
        
        # Build concentration network
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Softplus())  # Ensure positive concentration
        
        self.concentration_net = nn.Sequential(*layers)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with He initialization for ReLU."""
        for m in self.concentration_net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def concentration(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute nutrient concentration at positions.
        
        Args:
            x: Positions [batch, spatial_dim]
            t: Optional time [batch, 1]
            
        Returns:
            Concentration [batch, 1] (always positive)
        """
        if self.time_dependent:
            if t is None:
                raise ValueError("Time required for time-dependent nutrient field")
            inputs = torch.cat([x, t], dim=-1)
        else:
            inputs = x
        
        c = self.concentration_net(inputs)
        return c + self.min_concentration
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute potential φ = -log(concentration).
        
        Low potential = high concentration = attractive.
        """
        c = self.concentration(x, t)
        return -torch.log(c)
    
    def concentration_gradient(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute gradient of concentration ∇c.
        
        This is what microbes actually sense and follow.
        """
        x = x.requires_grad_(True)
        c = self.concentration(x, t)
        
        grad_c = torch.autograd.grad(
            c.sum(), x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        return grad_c


class ChemotaxisLoss(GeodesicLoss):
    """
    Chemotaxis-specific loss function.
    
    Extends GeodesicLoss with:
    - Chemotactic alignment: velocity should follow concentration gradient
    - Nutrient-aware weighting: higher loss weight in nutrient-rich regions
    
    Total loss:
        L = λ_speed * L_speed + λ_boundary * L_boundary + 
            λ_chemotaxis * L_chemotaxis + λ_data * L_data
    """
    
    def __init__(
        self,
        nutrient_field: NutrientField,
        lambda_speed: float = 0.1,
        lambda_boundary: float = 10.0,
        lambda_chemotaxis: float = 1.0,
        lambda_data: float = 1.0,
        target_speed: float = 1.0,
    ):
        """
        Initialize chemotaxis loss.
        
        Args:
            nutrient_field: The nutrient field to follow
            lambda_speed: Weight for constant speed constraint
            lambda_boundary: Weight for boundary conditions
            lambda_chemotaxis: Weight for chemotactic alignment
            lambda_data: Weight for data fitting
            target_speed: Target microbe speed
        """
        # Note: gradient_direction="descent" on potential = "ascent" on concentration
        super().__init__(
            potential_field=nutrient_field,
            lambda_speed=lambda_speed,
            lambda_boundary=lambda_boundary,
            lambda_gradient=lambda_chemotaxis,  # Chemotaxis = gradient following
            lambda_curvature=0.0,  # Microbes don't have curvature constraints
            lambda_data=lambda_data,
            target_speed=target_speed,
            gradient_direction="descent",  # Descent on potential = ascent on concentration
        )
        
        self.nutrient_field = nutrient_field
        self.lambda_chemotaxis = lambda_chemotaxis


class BiologyAdapter:
    """
    High-level adapter for biology/chemotaxis domain.
    
    Provides:
    - Pre-configured models for microbe tracking
    - Data loading utilities for microscopy data
    - Domain-specific evaluation metrics
    
    Example:
        >>> adapter = BiologyAdapter(hidden_dim=64)
        >>> model, loss_fn = adapter.create_model()
        >>> trajectory = adapter.predict(model, start, end, n_steps=100)
    """
    
    def __init__(
        self,
        spatial_dim: int = 2,
        hidden_dim: int = 64,
        time_dependent: bool = True,
    ):
        """
        Initialize biology adapter.
        
        Args:
            spatial_dim: Spatial dimension (usually 2 for microscopy)
            hidden_dim: Hidden layer width for networks
            time_dependent: Whether nutrient field varies with time
        """
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.time_dependent = time_dependent
    
    def create_model(
        self,
        lambda_speed: float = 0.1,
        lambda_boundary: float = 10.0,
        lambda_chemotaxis: float = 1.0,
        target_speed: float = 1.0,
    ) -> Tuple[nn.Module, ChemotaxisLoss]:
        """
        Create trajectory model and loss function.
        
        Returns:
            Tuple of (TrajectoryPINN, ChemotaxisLoss)
        """
        from ..core.trajectory_pinn import TrajectoryPINN
        
        # Create networks
        nutrient_field = NutrientField(
            spatial_dim=self.spatial_dim,
            hidden_dim=self.hidden_dim,
            time_dependent=self.time_dependent,
        )
        
        # Context: start (2D) + end (2D) = 4D for 2D trajectories
        trajectory_net = TrajectoryPINN(
            spatial_dim=self.spatial_dim,
            hidden_dim=self.hidden_dim,
            context_dim=self.spatial_dim * 2,  # start + end
        )
        
        # Loss function
        loss_fn = ChemotaxisLoss(
            nutrient_field=nutrient_field,
            lambda_speed=lambda_speed,
            lambda_boundary=lambda_boundary,
            lambda_chemotaxis=lambda_chemotaxis,
            target_speed=target_speed,
        )
        
        # Combine into single module
        class ChemotaxisModel(nn.Module):
            def __init__(self, traj_net, nutr_field):
                super().__init__()
                self.trajectory_net = traj_net
                self.nutrient_field = nutr_field
            
            def forward(self, t, context=None, start=None, end=None):
                if context is None and start is not None and end is not None:
                    context = torch.cat([start, end], dim=-1)
                return self.trajectory_net(t, context)
            
            def predict_trajectory(self, start, end, n_steps=100):
                t = torch.linspace(0, 1, n_steps, device=start.device).unsqueeze(-1)
                context = torch.cat([start, end]).unsqueeze(0).expand(n_steps, -1)
                return self.trajectory_net(t, context)
        
        model = ChemotaxisModel(trajectory_net, nutrient_field)
        
        return model, loss_fn
    
    def create_synthetic_data(
        self,
        n_trajectories: int = 100,
        n_steps: int = 50,
        target: Tuple[float, float] = (0.5, 0.5),
        noise_scale: float = 0.1,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate synthetic chemotaxis trajectories.
        
        Args:
            n_trajectories: Number of trajectories
            n_steps: Steps per trajectory
            target: Nutrient source location
            noise_scale: Movement noise
            seed: Random seed
            
        Returns:
            Dictionary with 'positions', 'times', 'starts', 'ends'
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        trajectories = []
        
        for _ in range(n_trajectories):
            # Random start
            start = torch.rand(2)
            
            # Initialize
            pos = start.clone()
            positions = [pos.clone()]
            
            target_t = torch.tensor(target)
            
            for _ in range(n_steps - 1):
                # Direction to target (chemotaxis)
                direction = target_t - pos
                direction = direction / (torch.norm(direction) + 1e-8)
                
                # Add noise
                noise = torch.randn(2) * noise_scale
                
                # Update
                step = 0.02 * (direction + noise)
                pos = pos + step
                positions.append(pos.clone())
            
            trajectories.append(torch.stack(positions))
        
        positions = torch.stack(trajectories)  # [n_traj, n_steps, 2]
        times = torch.linspace(0, 1, n_steps).unsqueeze(0).expand(n_trajectories, -1)
        
        return {
            'positions': positions,
            'times': times,
            'starts': positions[:, 0],
            'ends': positions[:, -1],
        }
