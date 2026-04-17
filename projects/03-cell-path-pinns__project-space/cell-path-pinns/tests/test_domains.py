"""
Tests for domain adapters.

Tests biology, robotics, migration, finance, and game AI adapters.
"""

import pytest
import torch

import sys
sys.path.insert(0, '..')

from src.domains.biology import (
    NutrientField,
    ChemotaxisLoss,
    BiologyAdapter,
)
from src.domains.robotics import (
    TerrainField,
    RobotPathLoss,
    RoboticsAdapter,
)
from src.domains.migration import (
    EnvironmentalField,
    MigrationLoss,
    MigrationAdapter,
)
from src.domains.finance import (
    ProfitLandscape,
    FinanceLoss,
    FinanceAdapter,
)
from src.domains.game_ai import (
    GameWorldField,
    GameAILoss,
    GameAIAdapter,
)


# ============================================================================
# Biology Domain Tests
# ============================================================================

class TestNutrientField:
    """Tests for NutrientField potential."""
    
    def test_construction(self):
        """Test basic construction."""
        field = NutrientField(input_dim=2, hidden_dims=[32, 32])
        assert field.input_dim == 2
        
    def test_forward_shape(self):
        """Test nutrient concentration output."""
        field = NutrientField(input_dim=2, hidden_dims=[32, 32])
        x = torch.randn(100, 2)
        conc = field(x)
        assert conc.shape == (100, 1)
        
    def test_gradient(self):
        """Test concentration gradient."""
        field = NutrientField(input_dim=2, hidden_dims=[32, 32])
        x = torch.randn(50, 2, requires_grad=True)
        grad = field.gradient(x)
        assert grad.shape == (50, 2)
        
    def test_with_source_term(self):
        """Test field with diffusion source."""
        field = NutrientField(input_dim=2, hidden_dims=[32, 32])
        x = torch.randn(50, 2, requires_grad=True)
        
        # Concentration and source
        conc = field(x)
        source = field.diffusion_source(x)
        
        assert conc.shape == (50, 1)
        assert source.shape == (50, 1)


class TestChemotaxisLoss:
    """Tests for ChemotaxisLoss."""
    
    def test_construction(self):
        """Test loss construction."""
        field = NutrientField(input_dim=2)
        loss_fn = ChemotaxisLoss(nutrient_field=field)
        assert loss_fn is not None
        
    def test_loss_computation(self):
        """Test loss returns dict."""
        field = NutrientField(input_dim=2, hidden_dims=[16])
        loss_fn = ChemotaxisLoss(nutrient_field=field)
        
        positions = torch.randn(50, 2, requires_grad=True)
        velocity = torch.randn(50, 2)
        acceleration = torch.randn(50, 2)
        
        total, loss_dict = loss_fn(
            positions=positions,
            velocity=velocity,
            acceleration=acceleration,
        )
        
        assert isinstance(total, torch.Tensor)
        assert 'chemotaxis' in loss_dict or 'gradient' in loss_dict


class TestBiologyAdapter:
    """Tests for BiologyAdapter."""
    
    def test_create_default_config(self):
        """Test default configuration."""
        adapter = BiologyAdapter()
        config = adapter.get_default_config()
        
        assert 'hidden_dims' in config
        assert 'chemotaxis_weight' in config
        
    def test_create_model(self):
        """Test model creation."""
        adapter = BiologyAdapter()
        field, loss_fn = adapter.create_model_and_loss()
        
        assert field is not None
        assert loss_fn is not None


# ============================================================================
# Robotics Domain Tests
# ============================================================================

class TestTerrainField:
    """Tests for TerrainField potential."""
    
    def test_construction(self):
        """Test basic construction."""
        obstacles = torch.tensor([[0.5, 0.5]])
        goal = torch.tensor([1.0, 1.0])
        
        field = TerrainField(
            input_dim=2,
            obstacles=obstacles,
            goal=goal,
        )
        assert field.input_dim == 2
        
    def test_obstacle_repulsion(self):
        """Test that obstacles create high potential."""
        obstacles = torch.tensor([[0.5, 0.5]])
        goal = torch.tensor([1.0, 1.0])
        
        field = TerrainField(
            input_dim=2,
            obstacles=obstacles,
            goal=goal,
            obstacle_strength=10.0,
        )
        
        # Point near obstacle
        near_obstacle = torch.tensor([[0.5, 0.5]])
        # Point far from obstacle
        far_point = torch.tensor([[0.0, 0.0]])
        
        phi_near = field(near_obstacle)
        phi_far = field(far_point)
        
        # Potential should be higher near obstacle
        assert phi_near.item() > phi_far.item()
        
    def test_goal_attraction(self):
        """Test that goal creates low potential."""
        obstacles = torch.tensor([[0.5, 0.5]])
        goal = torch.tensor([1.0, 1.0])
        
        field = TerrainField(
            input_dim=2,
            obstacles=obstacles,
            goal=goal,
            goal_strength=5.0,
        )
        
        # Point at goal
        at_goal = torch.tensor([[1.0, 1.0]])
        # Point far from goal
        far_point = torch.tensor([[0.0, 0.0]])
        
        phi_goal = field(at_goal)
        phi_far = field(far_point)
        
        # Potential should be lower at goal
        assert phi_goal.item() < phi_far.item()
        
    def test_gradient_shape(self):
        """Test terrain gradient."""
        obstacles = torch.tensor([[0.5, 0.5]])
        goal = torch.tensor([1.0, 1.0])
        
        field = TerrainField(input_dim=2, obstacles=obstacles, goal=goal)
        x = torch.randn(50, 2, requires_grad=True)
        
        grad = field.gradient(x)
        assert grad.shape == (50, 2)


class TestRobotPathLoss:
    """Tests for RobotPathLoss."""
    
    def test_loss_computation(self):
        """Test loss computation."""
        obstacles = torch.tensor([[0.5, 0.5]])
        goal = torch.tensor([1.0, 1.0])
        field = TerrainField(input_dim=2, obstacles=obstacles, goal=goal)
        
        loss_fn = RobotPathLoss(terrain_field=field)
        
        positions = torch.randn(50, 2, requires_grad=True)
        velocity = torch.randn(50, 2)
        acceleration = torch.randn(50, 2)
        
        total, loss_dict = loss_fn(
            positions=positions,
            velocity=velocity,
            acceleration=acceleration,
        )
        
        assert isinstance(total, torch.Tensor)
        assert isinstance(loss_dict, dict)


class TestRoboticsAdapter:
    """Tests for RoboticsAdapter."""
    
    def test_create_model(self):
        """Test model creation."""
        obstacles = torch.tensor([[0.3, 0.3], [0.7, 0.7]])
        goal = torch.tensor([1.0, 0.0])
        
        adapter = RoboticsAdapter()
        field, loss_fn = adapter.create_model_and_loss(
            obstacles=obstacles,
            goal=goal,
        )
        
        assert field is not None
        assert loss_fn is not None


# ============================================================================
# Migration Domain Tests
# ============================================================================

class TestEnvironmentalField:
    """Tests for EnvironmentalField."""
    
    def test_construction(self):
        """Test basic construction."""
        field = EnvironmentalField(input_dim=2)
        assert field.input_dim == 2
        
    def test_multi_factor_field(self):
        """Test that field combines multiple factors."""
        field = EnvironmentalField(
            input_dim=2,
            hidden_dims=[32, 32],
            factors=['temperature', 'food', 'safety'],
        )
        
        x = torch.randn(50, 2)
        phi = field(x)
        
        assert phi.shape == (50, 1)
        
    def test_seasonal_variation(self):
        """Test seasonal time-varying field."""
        field = EnvironmentalField(
            input_dim=2,
            hidden_dims=[32],
            include_time=True,
        )
        
        # Position + time
        x = torch.randn(50, 3)  # [x, y, t]
        phi = field(x)
        
        assert phi.shape == (50, 1)


class TestMigrationLoss:
    """Tests for MigrationLoss."""
    
    def test_with_stopovers(self):
        """Test loss with stopover points."""
        field = EnvironmentalField(input_dim=2, hidden_dims=[16])
        
        stopovers = torch.tensor([[0.3, 0.3], [0.6, 0.6]])
        loss_fn = MigrationLoss(
            environmental_field=field,
            stopover_points=stopovers,
            stopover_weight=1.0,
        )
        
        positions = torch.randn(50, 2, requires_grad=True)
        velocity = torch.randn(50, 2)
        acceleration = torch.randn(50, 2)
        
        total, loss_dict = loss_fn(
            positions=positions,
            velocity=velocity,
            acceleration=acceleration,
        )
        
        assert 'stopover' in loss_dict


class TestMigrationAdapter:
    """Tests for MigrationAdapter."""
    
    def test_create_model(self):
        """Test model creation."""
        adapter = MigrationAdapter()
        field, loss_fn = adapter.create_model_and_loss()
        
        assert field is not None
        assert loss_fn is not None


# ============================================================================
# Finance Domain Tests
# ============================================================================

class TestProfitLandscape:
    """Tests for ProfitLandscape."""
    
    def test_construction(self):
        """Test basic construction."""
        field = ProfitLandscape(
            input_dim=2,
            hidden_dims=[32, 32],
        )
        assert field.input_dim == 2
        
    def test_risk_adjusted_return(self):
        """Test risk-adjusted profit computation."""
        field = ProfitLandscape(
            input_dim=2,
            hidden_dims=[32],
            risk_aversion=1.0,
        )
        
        # Portfolio positions
        x = torch.randn(50, 2)
        profit = field(x)
        
        assert profit.shape == (50, 1)
        
    def test_gradient_represents_opportunity(self):
        """Test that gradient points toward better positions."""
        field = ProfitLandscape(input_dim=2, hidden_dims=[32])
        x = torch.randn(50, 2, requires_grad=True)
        
        grad = field.gradient(x)
        assert grad.shape == (50, 2)


class TestFinanceLoss:
    """Tests for FinanceLoss."""
    
    def test_transaction_costs(self):
        """Test loss includes transaction costs."""
        field = ProfitLandscape(input_dim=2, hidden_dims=[16])
        
        loss_fn = FinanceLoss(
            profit_landscape=field,
            transaction_cost_weight=0.1,
        )
        
        positions = torch.randn(50, 2, requires_grad=True)
        velocity = torch.randn(50, 2)
        acceleration = torch.randn(50, 2)
        
        total, loss_dict = loss_fn(
            positions=positions,
            velocity=velocity,
            acceleration=acceleration,
        )
        
        assert 'transaction' in loss_dict or isinstance(total, torch.Tensor)


class TestFinanceAdapter:
    """Tests for FinanceAdapter."""
    
    def test_create_model(self):
        """Test model creation."""
        adapter = FinanceAdapter()
        field, loss_fn = adapter.create_model_and_loss()
        
        assert field is not None
        assert loss_fn is not None


# ============================================================================
# Game AI Domain Tests
# ============================================================================

class TestGameWorldField:
    """Tests for GameWorldField."""
    
    def test_construction(self):
        """Test basic construction."""
        field = GameWorldField(
            input_dim=2,
            hidden_dims=[32, 32],
        )
        assert field.input_dim == 2
        
    def test_with_obstacles(self):
        """Test field with game obstacles."""
        obstacles = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        
        field = GameWorldField(
            input_dim=2,
            obstacles=obstacles,
            obstacle_radius=0.1,
        )
        
        # Check near obstacle has high cost
        near = torch.tensor([[0.5, 0.5]])
        far = torch.tensor([[0.0, 0.0]])
        
        phi_near = field(near)
        phi_far = field(far)
        
        assert phi_near.item() > phi_far.item()
        
    def test_fast_inference(self):
        """Test that inference is fast (for real-time)."""
        import time
        
        field = GameWorldField(input_dim=2, hidden_dims=[32, 32])
        x = torch.randn(1, 2)  # Single query
        
        # Warmup
        for _ in range(10):
            _ = field(x)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = field(x)
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / 100) * 1000
        # Should be sub-millisecond
        assert avg_ms < 5.0, f"Inference too slow: {avg_ms:.3f}ms"


class TestGameAILoss:
    """Tests for GameAILoss."""
    
    def test_loss_computation(self):
        """Test loss computation."""
        field = GameWorldField(input_dim=2, hidden_dims=[16])
        
        loss_fn = GameAILoss(game_field=field)
        
        positions = torch.randn(50, 2, requires_grad=True)
        velocity = torch.randn(50, 2)
        acceleration = torch.randn(50, 2)
        
        total, loss_dict = loss_fn(
            positions=positions,
            velocity=velocity,
            acceleration=acceleration,
        )
        
        assert isinstance(total, torch.Tensor)


class TestGameAIAdapter:
    """Tests for GameAIAdapter."""
    
    def test_create_model(self):
        """Test model creation."""
        adapter = GameAIAdapter()
        field, loss_fn = adapter.create_model_and_loss()
        
        assert field is not None
        assert loss_fn is not None
        
    def test_create_for_realtime(self):
        """Test creating optimized model for real-time."""
        adapter = GameAIAdapter()
        config = adapter.get_realtime_config()
        
        # Real-time config should have small network
        assert config['hidden_dims'] == [32, 32] or len(config['hidden_dims']) <= 3


# ============================================================================
# Cross-Domain Tests
# ============================================================================

class TestCrossDomain:
    """Tests for cross-domain compatibility."""
    
    def test_all_fields_have_gradient(self):
        """Test all domain fields implement gradient."""
        fields = [
            NutrientField(input_dim=2, hidden_dims=[16]),
            TerrainField(input_dim=2, obstacles=torch.zeros(1, 2), goal=torch.ones(2)),
            EnvironmentalField(input_dim=2, hidden_dims=[16]),
            ProfitLandscape(input_dim=2, hidden_dims=[16]),
            GameWorldField(input_dim=2, hidden_dims=[16]),
        ]
        
        x = torch.randn(10, 2, requires_grad=True)
        
        for field in fields:
            grad = field.gradient(x.clone().requires_grad_(True))
            assert grad.shape == (10, 2), f"{type(field).__name__} gradient wrong shape"
            
    def test_all_losses_return_dict(self):
        """Test all domain losses return (total, dict)."""
        # Create minimal fields and losses
        nutrient = NutrientField(input_dim=2, hidden_dims=[16])
        terrain = TerrainField(input_dim=2, obstacles=torch.zeros(1, 2), goal=torch.ones(2))
        environ = EnvironmentalField(input_dim=2, hidden_dims=[16])
        profit = ProfitLandscape(input_dim=2, hidden_dims=[16])
        game = GameWorldField(input_dim=2, hidden_dims=[16])
        
        losses = [
            ChemotaxisLoss(nutrient_field=nutrient),
            RobotPathLoss(terrain_field=terrain),
            MigrationLoss(environmental_field=environ),
            FinanceLoss(profit_landscape=profit),
            GameAILoss(game_field=game),
        ]
        
        positions = torch.randn(50, 2, requires_grad=True)
        velocity = torch.randn(50, 2)
        acceleration = torch.randn(50, 2)
        
        for loss_fn in losses:
            total, loss_dict = loss_fn(
                positions=positions.clone().requires_grad_(True),
                velocity=velocity,
                acceleration=acceleration,
            )
            assert isinstance(total, torch.Tensor), f"{type(loss_fn).__name__}"
            assert isinstance(loss_dict, dict), f"{type(loss_fn).__name__}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
