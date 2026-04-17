"""
Migration Domain Adapter: Animal Movement on Environmental Gradients.

Models animal migration trajectories (birds, fish, mammals) as geodesics
on multi-factor environmental fields.

Key Factors:
- Temperature gradients (thermal soaring for birds)
- Magnetic field (navigation cues)
- Food availability (foraging pressure)
- Predator density (avoidance)
- Wind/current patterns (energy efficiency)

Potential Convention:
- Favorable conditions → low potential → preferred
- Multiple factors combined with learned weights
"""

from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn

from ..core.potential_field import PotentialFieldBase, ComposedPotentialField
from ..core.geodesic_loss import GeodesicLoss


class SingleFactorField(PotentialFieldBase):
    """
    Single environmental factor as potential field.
    
    Maps (lat, lon, time) → scalar potential for one factor.
    """
    
    def __init__(
        self,
        factor_name: str,
        spatial_dim: int = 2,
        hidden_dim: int = 32,
        time_dependent: bool = True,
    ):
        """
        Initialize single factor field.
        
        Args:
            factor_name: Name of the factor (for identification)
            spatial_dim: 2 for lat/lon, 3 for lat/lon/altitude
            hidden_dim: Network hidden dimension
            time_dependent: Whether factor varies with time/season
        """
        super().__init__(spatial_dim=spatial_dim, time_dependent=time_dependent)
        
        self.factor_name = factor_name
        input_dim = spatial_dim + (1 if time_dependent else 0)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.time_dependent:
            if t is None:
                raise ValueError(f"Time required for time-dependent {self.factor_name} field")
            inputs = torch.cat([x, t], dim=-1)
        else:
            inputs = x
        return self.net(inputs)


class EnvironmentalField(PotentialFieldBase):
    """
    Multi-factor environmental potential field.
    
    Combines multiple environmental factors:
        φ_total = Σ w_i * φ_i(x, t)
    
    Where w_i are learned (or fixed) weights and φ_i are individual factors.
    """
    
    def __init__(
        self,
        factors: List[str] = None,
        spatial_dim: int = 2,
        hidden_dim: int = 32,
        time_dependent: bool = True,
        learnable_weights: bool = True,
    ):
        """
        Initialize multi-factor environmental field.
        
        Args:
            factors: List of factor names. Default: ["temperature", "food", "predators"]
            spatial_dim: Spatial dimension
            hidden_dim: Hidden dimension per factor
            time_dependent: Whether factors vary with time
            learnable_weights: Whether factor weights are learnable
        """
        super().__init__(spatial_dim=spatial_dim, time_dependent=time_dependent)
        
        if factors is None:
            factors = ["temperature", "food", "predators"]
        
        self.factors = factors
        self.learnable_weights = learnable_weights
        
        # Create network for each factor
        self.factor_nets = nn.ModuleDict({
            f: SingleFactorField(f, spatial_dim, hidden_dim, time_dependent)
            for f in factors
        })
        
        # Combination weights
        n_factors = len(factors)
        if learnable_weights:
            self.weights = nn.Parameter(torch.ones(n_factors) / n_factors)
        else:
            self.register_buffer('weights', torch.ones(n_factors) / n_factors)
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute weighted sum of factor potentials."""
        # Compute each factor's potential
        potentials = []
        for factor_name in self.factors:
            phi = self.factor_nets[factor_name](x, t)
            potentials.append(phi)
        
        potentials = torch.cat(potentials, dim=-1)  # [batch, n_factors]
        
        # Weighted sum
        if self.learnable_weights:
            w = torch.softmax(self.weights, dim=0)
        else:
            w = self.weights
        
        return (potentials * w).sum(dim=-1, keepdim=True)
    
    def get_factor_contributions(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get individual factor potentials for analysis."""
        contributions = {}
        for factor_name in self.factors:
            contributions[factor_name] = self.factor_nets[factor_name](x, t)
        return contributions


class MigrationLoss(GeodesicLoss):
    """
    Migration-specific loss function.
    
    Additional constraints for migration modeling:
    - Seasonal timing (migrations happen at specific times)
    - Stopover sites (paths should pass near known rest stops)
    - Energy budget (limited fat reserves)
    """
    
    def __init__(
        self,
        environmental_field: EnvironmentalField,
        lambda_speed: float = 0.1,
        lambda_boundary: float = 10.0,
        lambda_environment: float = 1.0,
        lambda_stopover: float = 1.0,
        lambda_data: float = 1.0,
        target_speed: float = 1.0,
        stopover_sites: Optional[torch.Tensor] = None,
        stopover_threshold: float = 0.1,
    ):
        """
        Initialize migration loss.
        
        Args:
            environmental_field: Multi-factor environment
            lambda_speed: Constant speed weight
            lambda_boundary: Start/end weight
            lambda_environment: Environment-following weight
            lambda_stopover: Stopover proximity weight
            lambda_data: Data fitting weight
            target_speed: Target migration speed
            stopover_sites: Known stopover locations [n_sites, dim]
            stopover_threshold: Distance threshold for stopover
        """
        super().__init__(
            potential_field=environmental_field,
            lambda_speed=lambda_speed,
            lambda_boundary=lambda_boundary,
            lambda_gradient=lambda_environment,
            lambda_curvature=0.0,  # Animals don't have strict curvature limits
            lambda_data=lambda_data,
            target_speed=target_speed,
            gradient_direction="descent",
        )
        
        self.lambda_stopover = lambda_stopover
        self.stopover_sites = stopover_sites
        self.stopover_threshold = stopover_threshold
    
    def stopover_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute stopover proximity loss.
        
        Encourages paths to pass near known stopover sites.
        """
        if self.stopover_sites is None or self.lambda_stopover == 0:
            return torch.tensor(0.0, device=positions.device)
        
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        
        # Minimum distance from path to each stopover
        # positions: [batch, time, dim]
        # stopover_sites: [n_sites, dim]
        batch_size = positions.shape[0]
        
        losses = []
        for b in range(batch_size):
            # Distance from each path point to each stopover
            distances = torch.cdist(positions[b], self.stopover_sites)  # [time, n_sites]
            
            # Minimum distance to each stopover over the trajectory
            min_distances = distances.min(dim=0).values  # [n_sites]
            
            # Penalize distances above threshold
            excess = torch.relu(min_distances - self.stopover_threshold)
            losses.append(excess.mean())
        
        return torch.stack(losses).mean()
    
    def forward(
        self,
        positions: torch.Tensor,
        times: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        positions_true: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute migration loss including stopover term."""
        total, losses = super().forward(positions, times, start, end, positions_true)
        
        # Add stopover loss
        if self.lambda_stopover > 0 and self.stopover_sites is not None:
            stopover = self.stopover_loss(positions)
            losses['stopover'] = stopover
            total = total + self.lambda_stopover * stopover
            losses['total'] = total
        
        return total, losses


class MigrationAdapter:
    """
    High-level adapter for migration modeling.
    
    Provides:
    - Multi-factor environmental models
    - Stopover site integration
    - Seasonal timing constraints
    
    Example:
        >>> adapter = MigrationAdapter(factors=["temperature", "magnetic", "food"])
        >>> adapter.add_stopover(lat=45.0, lon=-75.0)
        >>> model, loss = adapter.create_model()
    """
    
    def __init__(
        self,
        spatial_dim: int = 2,
        factors: List[str] = None,
        hidden_dim: int = 32,
        time_dependent: bool = True,
    ):
        """
        Initialize migration adapter.
        
        Args:
            spatial_dim: 2 (lat/lon) or 3 (lat/lon/altitude)
            factors: Environmental factors to model
            hidden_dim: Network hidden dimension
            time_dependent: Whether environment varies with time
        """
        self.spatial_dim = spatial_dim
        self.factors = factors or ["temperature", "food", "predators"]
        self.hidden_dim = hidden_dim
        self.time_dependent = time_dependent
        
        self._stopovers = []
    
    def add_stopover(self, *coords, threshold: float = 0.1):
        """Add stopover site."""
        self._stopovers.append((coords, threshold))
    
    def create_model(
        self,
        lambda_speed: float = 0.1,
        lambda_boundary: float = 10.0,
        lambda_environment: float = 1.0,
        lambda_stopover: float = 1.0,
        target_speed: float = 1.0,
    ) -> Tuple[nn.Module, MigrationLoss]:
        """Create migration model and loss."""
        from ..core.trajectory_pinn import TrajectoryPINN
        
        # Create environmental field
        env_field = EnvironmentalField(
            factors=self.factors,
            spatial_dim=self.spatial_dim,
            hidden_dim=self.hidden_dim,
            time_dependent=self.time_dependent,
        )
        
        # Trajectory network
        trajectory_net = TrajectoryPINN(
            spatial_dim=self.spatial_dim,
            hidden_dim=64,
            context_dim=self.spatial_dim * 2,  # start + end
        )
        
        # Stopover sites tensor
        stopovers = None
        threshold = 0.1
        if self._stopovers:
            stopovers = torch.tensor([s[0] for s in self._stopovers], dtype=torch.float32)
            threshold = self._stopovers[0][1]
        
        # Loss function
        loss_fn = MigrationLoss(
            environmental_field=env_field,
            lambda_speed=lambda_speed,
            lambda_boundary=lambda_boundary,
            lambda_environment=lambda_environment,
            lambda_stopover=lambda_stopover,
            target_speed=target_speed,
            stopover_sites=stopovers,
            stopover_threshold=threshold,
        )
        
        # Combined model
        class MigrationModel(nn.Module):
            def __init__(self, traj_net, env):
                super().__init__()
                self.trajectory_net = traj_net
                self.environmental_field = env
            
            def forward(self, t, context=None, start=None, end=None):
                if context is None and start is not None and end is not None:
                    if start.dim() == 1:
                        start = start.unsqueeze(0)
                    if end.dim() == 1:
                        end = end.unsqueeze(0)
                    context = torch.cat([start, end], dim=-1)
                    context = context.expand(t.shape[0], -1)
                return self.trajectory_net(t, context)
        
        model = MigrationModel(trajectory_net, env_field)
        
        return model, loss_fn
