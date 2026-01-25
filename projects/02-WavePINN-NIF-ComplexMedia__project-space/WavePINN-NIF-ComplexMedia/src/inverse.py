"""
Inverse problem solver for wave speed tomography.

Given observed wavefield data at sensor locations,
recover the unknown wave speed field c(x).
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple, Optional, Callable
from functools import partial

from .model import create_model, count_parameters
from .physics_loss import WavePDELoss


class InverseTomography:
    """
    Inverse problem: Learn c(x) from observed wavefield data.
    
    Given:
    - Sensor measurements u_obs(x_sensors, t)
    - Source wavelet
    - Domain geometry
    
    Recover:
    - Velocity field c(x)
    """
    
    def __init__(self,
                 config: Dict,
                 sensor_positions: jnp.ndarray,
                 observed_data: jnp.ndarray,
                 source_function: Callable):
        """
        Initialize inverse problem.
        
        Args:
            config: Configuration dictionary
            sensor_positions: Sensor locations (n_sensors, ndim)
            observed_data: Observed waveforms (n_sensors, n_times)
            source_function: Source wavelet function f(t) -> amplitude
        """
        self.config = config
        self.sensor_positions = sensor_positions
        self.observed_data = observed_data
        self.source_function = source_function
        
        self.n_sensors = len(sensor_positions)
        self.ndim = sensor_positions.shape[1]
        
        # Create model
        self.model = create_model()
        
        # Initialize parameters
        self.rng = jax.random.PRNGKey(config.get('seed', 42))
        self.rng, init_rng = jax.random.split(self.rng)
        
        dummy_input = jnp.zeros((1, self.ndim + 1))
        self.params = self.model.init(init_rng, dummy_input)
        
        print(f"Inverse model initialized: {count_parameters(self.params):,} parameters")
        
        # Optimizer
        lr = config.get('learning_rate', 1e-3)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)
        
        # Loss weights
        self.lambda_data = config.get('lambda_data', 10.0)
        self.lambda_pde = config.get('lambda_pde', 1.0)
        self.lambda_smooth = config.get('lambda_smooth', 0.1)  # Regularization
    
    def data_misfit(self,
                    params: Dict,
                    rng: jax.Array,
                    time_points: jnp.ndarray) -> jnp.ndarray:
        """
        Compute misfit between predicted and observed data.
        
        Args:
            params: Model parameters
            rng: Random key
            time_points: Time values (n_times,)
            
        Returns:
            Data misfit loss
        """
        n_times = len(time_points)
        
        # Create sensor-time coordinates
        # Shape: (n_sensors * n_times, ndim + 1)
        sensor_tile = jnp.tile(self.sensor_positions, (n_times, 1))
        time_tile = jnp.repeat(time_points, self.n_sensors)[:, None]
        coords = jnp.concatenate([sensor_tile, time_tile], axis=1)
        
        # Predict wavefield at sensor locations
        u_pred = self.model.apply(params, rng, coords, return_media=False)
        u_pred = u_pred.reshape(n_times, self.n_sensors).T  # (n_sensors, n_times)
        
        # Observed data should have same shape
        misfit = jnp.mean((u_pred - self.observed_data)**2)
        
        return misfit
    
    def pde_loss(self,
                 params: Dict,
                 rng: jax.Array,
                 x_collocation: jnp.ndarray) -> jnp.ndarray:
        """
        Compute PDE residual loss at collocation points.
        
        Args:
            params: Model parameters
            rng: Random key
            x_collocation: Interior points (batch, ndim+1)
            
        Returns:
            PDE residual loss
        """
        loss_computer = WavePDELoss(
            model_apply=self.model.apply,
            lambda_pde=1.0,
            lambda_bc=0.0,
            lambda_ic=0.0,
            ndim=self.ndim
        )
        
        residuals = loss_computer.pde_residual(params, rng, x_collocation)
        return jnp.mean(residuals**2)
    
    def smoothness_regularization(self,
                                  params: Dict,
                                  rng: jax.Array,
                                  x_spatial: jnp.ndarray) -> jnp.ndarray:
        """
        Regularization to encourage smooth velocity fields.
        
        Penalizes large gradients in c(x).
        
        Args:
            params: Model parameters
            rng: Random key
            x_spatial: Spatial points (batch, ndim)
            
        Returns:
            Smoothness regularization loss
        """
        def c_at_point(x_single):
            # Append dummy time
            x_full = jnp.concatenate([x_single, jnp.zeros(1)])
            _, c = self.model.apply(params, rng, x_full[None, :], return_media=True)
            return c[0, 0]
        
        def grad_c(x_single):
            return jax.grad(c_at_point)(x_single)
        
        # Compute gradient magnitude at each point
        grads = jax.vmap(grad_c)(x_spatial)
        grad_magnitudes = jnp.sum(grads**2, axis=-1)
        
        return jnp.mean(grad_magnitudes)
    
    def total_loss(self,
                   params: Dict,
                   rng: jax.Array,
                   data_batch: Dict) -> Tuple[jnp.ndarray, Dict]:
        """
        Total inverse problem loss.
        
        Args:
            params: Model parameters
            rng: Random key
            data_batch: Dictionary with 'time_points', 'collocation', 'spatial'
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        
        # Data misfit
        losses['loss_data'] = self.data_misfit(
            params, rng, data_batch['time_points']
        )
        
        # PDE constraint
        losses['loss_pde'] = self.pde_loss(
            params, rng, data_batch['collocation']
        )
        
        # Smoothness regularization
        losses['loss_smooth'] = self.smoothness_regularization(
            params, rng, data_batch['spatial']
        )
        
        # Weighted total
        total = (
            self.lambda_data * losses['loss_data'] +
            self.lambda_pde * losses['loss_pde'] +
            self.lambda_smooth * losses['loss_smooth']
        )
        losses['loss_total'] = total
        
        return total, losses
    
    @partial(jax.jit, static_argnums=(0,))
    def train_step(self,
                   params: Dict,
                   opt_state: any,
                   rng: jax.Array,
                   data_batch: Dict) -> Tuple[Dict, any, Dict]:
        """JIT-compiled training step."""
        def loss_fn(p):
            return self.total_loss(p, rng, data_batch)
        
        (loss, loss_dict), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss_dict
    
    def solve(self,
              n_iterations: int,
              x_collocation: jnp.ndarray,
              time_points: jnp.ndarray,
              verbose: bool = True) -> Dict:
        """
        Solve the inverse problem.
        
        Args:
            n_iterations: Number of optimization iterations
            x_collocation: Collocation points (batch, ndim+1)
            time_points: Time sampling points
            verbose: Print progress
            
        Returns:
            Final parameters
        """
        from tqdm import tqdm
        
        # Spatial points (for regularization)
        x_spatial = x_collocation[:, :-1]  # Remove time
        
        data_batch = {
            'time_points': time_points,
            'collocation': x_collocation,
            'spatial': x_spatial,
        }
        
        if verbose:
            print(f"\nSolving inverse problem for {n_iterations} iterations...")
        
        iterator = tqdm(range(n_iterations)) if verbose else range(n_iterations)
        
        for i in iterator:
            self.rng, step_rng = jax.random.split(self.rng)
            
            self.params, self.opt_state, loss_dict = self.train_step(
                self.params, self.opt_state, step_rng, data_batch
            )
            
            if verbose and i % 100 == 0:
                losses_str = ", ".join(f"{k}={float(v):.4f}" for k, v in loss_dict.items())
                tqdm.write(f"Iter {i}: {losses_str}")
        
        return self.params
    
    def get_velocity_field(self,
                           grid_points: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate learned velocity field on a grid.
        
        Args:
            grid_points: Spatial grid (n_points, ndim)
            
        Returns:
            Velocity values (n_points,)
        """
        # Append dummy time dimension
        n_points = len(grid_points)
        coords = jnp.concatenate([
            grid_points,
            jnp.zeros((n_points, 1))
        ], axis=1)
        
        self.rng, pred_rng = jax.random.split(self.rng)
        _, c = self.model.apply(self.params, pred_rng, coords, return_media=True)
        
        return c.squeeze()


def synthetic_inverse_test():
    """Test inverse solver with synthetic data."""
    print("\n" + "=" * 60)
    print("Synthetic Inverse Problem Test")
    print("=" * 60)
    
    # True velocity model (simple example)
    def true_velocity(x):
        """Simple layered model: c = 2 + x[1]"""
        return 2.0 + x[:, 1]
    
    # Generate "observed" data at sensors
    # In practice, this would come from real measurements or FD simulation
    n_sensors = 10
    n_times = 50
    
    key = jax.random.PRNGKey(123)
    sensor_positions = jax.random.uniform(key, (n_sensors, 2))  # 2D
    time_points = jnp.linspace(0, 1.0, n_times)
    
    # Synthetic observations (simplified)
    observed_data = jax.random.normal(key, (n_sensors, n_times)) * 0.1
    
    # Source function
    def ricker_source(t, f0=10.0):
        t_shift = 0.1
        tau = t - t_shift
        return (1 - 2*(jnp.pi*f0*tau)**2) * jnp.exp(-(jnp.pi*f0*tau)**2)
    
    # Create inverse solver
    config = {
        'seed': 42,
        'learning_rate': 1e-3,
        'lambda_data': 10.0,
        'lambda_pde': 1.0,
        'lambda_smooth': 0.1,
    }
    
    solver = InverseTomography(
        config=config,
        sensor_positions=sensor_positions,
        observed_data=observed_data,
        source_function=ricker_source
    )
    
    # Generate collocation points
    key = jax.random.PRNGKey(456)
    x_collocation = jax.random.uniform(key, (500, 3))  # (x, y, t)
    
    # Solve (quick test with few iterations)
    params = solver.solve(
        n_iterations=50,
        x_collocation=x_collocation,
        time_points=time_points,
        verbose=True
    )
    
    # Evaluate velocity field
    grid = jax.random.uniform(key, (100, 2))
    c_learned = solver.get_velocity_field(grid)
    
    print(f"\nLearned velocity range: [{float(c_learned.min()):.3f}, {float(c_learned.max()):.3f}]")
    print("✅ Inverse test completed!")
    
    return solver


if __name__ == "__main__":
    synthetic_inverse_test()
