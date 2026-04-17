"""
Finance Domain Adapter: Agent Trajectories in Profit/Risk Space.

Models financial agent (trader, portfolio) movement through state space
as geodesics on profit/risk landscapes.

Key Concepts:
- State space dimensions: risk exposure, leverage, sector allocation, etc.
- Potential = -profit + λ*risk (risk-adjusted returns)
- Gradient descent on potential = maximizing risk-adjusted profit
- Constraints: transaction costs, position limits

Potential Convention:
- Low potential = high risk-adjusted profit = desirable
- Risk aversion parameter controls risk/return tradeoff
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn

from ..core.potential_field import PotentialFieldBase
from ..core.geodesic_loss import GeodesicLoss


class ProfitLandscape(PotentialFieldBase):
    """
    Profit/risk landscape as potential field.
    
    The potential combines expected profit and risk:
        φ(x) = -E[profit(x)] + λ * Risk(x)
    
    Where:
    - x is position in strategy space
    - λ is risk aversion parameter
    - Low φ = high expected risk-adjusted return
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 64,
        risk_aversion: float = 0.5,
        time_dependent: bool = True,
    ):
        """
        Initialize profit landscape.
        
        Args:
            state_dim: Dimension of strategy/state space
            hidden_dim: Network hidden dimension
            risk_aversion: λ parameter (higher = more risk-averse)
            time_dependent: Whether market conditions vary with time
        """
        super().__init__(spatial_dim=state_dim, time_dependent=time_dependent)
        
        self.state_dim = state_dim
        self.risk_aversion = risk_aversion
        
        input_dim = state_dim + (1 if time_dependent else 0)
        
        # Profit network (expected return)
        self.profit_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Risk network (variance, VaR, etc.)
        self.risk_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Risk is always non-negative
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for net in [self.profit_net, self.risk_net]:
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    nn.init.zeros_(m.bias)
    
    def profit(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute expected profit at state x."""
        if self.time_dependent:
            inputs = torch.cat([x, t], dim=-1)
        else:
            inputs = x
        return self.profit_net(inputs)
    
    def risk(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute risk at state x."""
        if self.time_dependent:
            inputs = torch.cat([x, t], dim=-1)
        else:
            inputs = x
        return self.risk_net(inputs)
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute potential φ = -profit + λ*risk.
        
        Minimizing this potential maximizes risk-adjusted profit.
        """
        profit = self.profit(x, t)
        risk = self.risk(x, t)
        return -profit + self.risk_aversion * risk
    
    def sharpe_ratio(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Sharpe-like ratio: profit / risk."""
        profit = self.profit(x, t)
        risk = self.risk(x, t) + 1e-8
        return profit / risk


class TransactionCostLoss(nn.Module):
    """
    Penalizes high transaction costs (rapid state changes).
    
    In finance, changing positions quickly incurs costs.
    This encourages smoother trajectories through state space.
    """
    
    def __init__(self, cost_rate: float = 0.01):
        """
        Initialize transaction cost loss.
        
        Args:
            cost_rate: Cost per unit of position change
        """
        super().__init__()
        self.cost_rate = cost_rate
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute total transaction costs along trajectory.
        
        Args:
            positions: [batch, time, dim] state trajectory
        """
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        
        # Total variation (sum of absolute changes)
        changes = torch.abs(positions[:, 1:] - positions[:, :-1])  # [batch, time-1, dim]
        total_change = changes.sum(dim=(-1, -2))  # [batch]
        
        return self.cost_rate * total_change.mean()


class FinanceLoss(GeodesicLoss):
    """
    Finance-specific loss function.
    
    Combines:
    - Profit-seeking (gradient descent on potential)
    - Transaction costs (penalize rapid changes)
    - Position constraints (stay within limits)
    """
    
    def __init__(
        self,
        profit_landscape: ProfitLandscape,
        lambda_speed: float = 0.1,
        lambda_boundary: float = 10.0,
        lambda_profit: float = 1.0,
        lambda_transaction: float = 0.5,
        lambda_data: float = 0.0,
        target_speed: float = 1.0,
        transaction_cost_rate: float = 0.01,
    ):
        """
        Initialize finance loss.
        
        Args:
            profit_landscape: Profit/risk landscape
            lambda_speed: Constant speed weight (smooth movement)
            lambda_boundary: Start/end weight
            lambda_profit: Profit-seeking weight
            lambda_transaction: Transaction cost weight
            lambda_data: Data fitting weight
            target_speed: Target state change rate
            transaction_cost_rate: Cost per unit position change
        """
        super().__init__(
            potential_field=profit_landscape,
            lambda_speed=lambda_speed,
            lambda_boundary=lambda_boundary,
            lambda_gradient=lambda_profit,
            lambda_curvature=0.0,
            lambda_data=lambda_data,
            target_speed=target_speed,
            gradient_direction="descent",
        )
        
        self.lambda_transaction = lambda_transaction
        self.transaction_loss = TransactionCostLoss(transaction_cost_rate)
    
    def forward(
        self,
        positions: torch.Tensor,
        times: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        positions_true: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute finance loss with transaction costs."""
        total, losses = super().forward(positions, times, start, end, positions_true)
        
        # Add transaction cost
        if self.lambda_transaction > 0:
            transaction = self.transaction_loss(positions)
            losses['transaction'] = transaction
            total = total + self.lambda_transaction * transaction
            losses['total'] = total
        
        return total, losses


class FinanceAdapter:
    """
    High-level adapter for financial trajectory modeling.
    
    Example:
        >>> adapter = FinanceAdapter(state_dim=4, risk_aversion=0.5)
        >>> model, loss = adapter.create_model()
        >>> # State: [equity_exposure, bond_exposure, cash, leverage]
        >>> trajectory = model.plan_trajectory(start_state, target_state)
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 64,
        risk_aversion: float = 0.5,
        time_dependent: bool = True,
    ):
        """
        Initialize finance adapter.
        
        Args:
            state_dim: Dimension of strategy space
            hidden_dim: Network hidden dimension
            risk_aversion: Risk aversion parameter
            time_dependent: Whether market varies with time
        """
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.risk_aversion = risk_aversion
        self.time_dependent = time_dependent
    
    def create_model(
        self,
        lambda_speed: float = 0.1,
        lambda_boundary: float = 10.0,
        lambda_profit: float = 1.0,
        lambda_transaction: float = 0.5,
        target_speed: float = 1.0,
    ) -> Tuple[nn.Module, FinanceLoss]:
        """Create financial trajectory model and loss."""
        from ..core.trajectory_pinn import TrajectoryPINN
        
        # Create profit landscape
        profit_landscape = ProfitLandscape(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            risk_aversion=self.risk_aversion,
            time_dependent=self.time_dependent,
        )
        
        # Trajectory network
        trajectory_net = TrajectoryPINN(
            spatial_dim=self.state_dim,
            hidden_dim=64,
            context_dim=self.state_dim * 2,  # start + target state
        )
        
        # Loss function
        loss_fn = FinanceLoss(
            profit_landscape=profit_landscape,
            lambda_speed=lambda_speed,
            lambda_boundary=lambda_boundary,
            lambda_profit=lambda_profit,
            lambda_transaction=lambda_transaction,
            target_speed=target_speed,
        )
        
        # Combined model
        class FinanceModel(nn.Module):
            def __init__(self, traj_net, landscape):
                super().__init__()
                self.trajectory_net = traj_net
                self.profit_landscape = landscape
            
            def forward(self, t, context=None, start=None, end=None):
                if context is None and start is not None and end is not None:
                    if start.dim() == 1:
                        start = start.unsqueeze(0)
                    if end.dim() == 1:
                        end = end.unsqueeze(0)
                    context = torch.cat([start, end], dim=-1)
                    context = context.expand(t.shape[0], -1)
                return self.trajectory_net(t, context)
            
            def plan_trajectory(self, start, target, n_steps=100):
                """Plan optimal trajectory through state space."""
                if not isinstance(start, torch.Tensor):
                    start = torch.tensor(start, dtype=torch.float32)
                if not isinstance(target, torch.Tensor):
                    target = torch.tensor(target, dtype=torch.float32)
                
                t = torch.linspace(0, 1, n_steps).unsqueeze(-1)
                context = torch.cat([start, target]).unsqueeze(0).expand(n_steps, -1)
                return self.trajectory_net(t, context)
        
        model = FinanceModel(trajectory_net, profit_landscape)
        
        return model, loss_fn
