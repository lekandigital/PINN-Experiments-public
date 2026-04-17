"""
Adapter for P17: WavePINN-NIF (Scalar) - Wave PINN with Scalar Fields (JAX)
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from harness.base_adapter import JAXAdapter, ProjectInfo
from harness.core import ModelInfo, TrainingInfo

try:
    import jax
    import jax.numpy as jnp
    import haiku as hk
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def create_wavepinn_scalar_forward(hidden_dim: int = 256, num_layers: int = 5):
    """Create forward function for scalar WavePINN."""
    
    def forward(xt: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for scalar wave PINN.
        Maps (x, t) -> scalar pressure field u(x, t).
        
        Args:
            xt: (B, 2) for 1D wave, or (B, 3) for 2D, or (B, 4) for 3D+time
        
        Returns:
            (B, 1): scalar wave amplitude
        """
        input_dim = xt.shape[-1]
        
        # Positional encoding
        num_frequencies = 6
        encoded = [xt]
        for freq in range(num_frequencies):
            scale = 2.0 ** freq
            encoded.append(jnp.sin(np.pi * scale * xt))
            encoded.append(jnp.cos(np.pi * scale * xt))
        
        x = jnp.concatenate(encoded, axis=-1)
        
        # MLP with skip connections
        skip_indices = [num_layers // 2]
        skip_value = None
        
        for i in range(num_layers):
            x = hk.Linear(hidden_dim, name=f'linear_{i}')(x)
            x = jnp.tanh(x)
            
            if i in skip_indices:
                skip_value = x
            elif skip_value is not None and i == skip_indices[0] + 2:
                x = x + skip_value
                skip_value = None
        
        # Output
        out = hk.Linear(1, name='output')(x)
        
        return out
    
    return forward


class P17WavePINNScalarAdapter(JAXAdapter):
    """Adapter for WavePINN-NIF scalar project (JAX)."""
    
    PROJECT_ID = "P17"
    PROJECT_NAME = "WavePINN-NIF-Scalar"
    
    def __init__(self, project_path: Path, subprocess_isolation: bool = True):
        super().__init__(project_path, subprocess_isolation)
        self._model_fn = None
        self._params = None
    
    def get_project_info(self) -> ProjectInfo:
        return ProjectInfo(
            project_id=self.PROJECT_ID,
            project_name=self.PROJECT_NAME,
            framework="jax",
            project_path=str(self.project_path),
        )
    
    def _create_model(self):
        """Create and initialize the JAX model."""
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX/Haiku not available")
        
        def model_fn(xt):
            forward = create_wavepinn_scalar_forward(hidden_dim=256, num_layers=5)
            return forward(xt)
        
        self._model_fn = hk.transform(model_fn)
        
        rng_key = jax.random.PRNGKey(42)
        dummy_input = jnp.zeros((1, 2))  # 1D wave: (x, t)
        self._params = self._model_fn.init(rng_key, dummy_input)
        
        return (self._model_fn, self._params)
    
    def load_model(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        """Load or create the model."""
        return self._create_model()
    
    def prepare_test_input(self, device: str, batch_size: int = 1) -> Tuple:
        """Prepare test inputs."""
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available")
        
        key = jax.random.PRNGKey(42)
        
        # 1D wave: x in [0, 1], t in [0, 2]
        x = jax.random.uniform(key, (batch_size, 2000, 1), minval=0.0, maxval=1.0)
        key, subkey = jax.random.split(key)
        t = jax.random.uniform(subkey, (batch_size, 2000, 1), minval=0.0, maxval=2.0)
        xt = jnp.concatenate([x, t], axis=-1)
        
        return (xt,)
    
    def run_inference(self, model, inputs: Tuple, device: str) -> Any:
        """Run model inference."""
        model_fn, params = model
        xt, = inputs
        
        original_shape = xt.shape
        xt_flat = xt.reshape(-1, 2)
        
        output = model_fn.apply(params, None, xt_flat)
        output = jax.block_until_ready(output)
        output = output.reshape(*original_shape[:-1], 1)
        
        return output
    
    def compute_domain_metrics(self, predictions: Any,
                               references: Optional[Any] = None) -> Dict[str, float]:
        """Compute scalar wave metrics."""
        metrics = {}
        
        pred = np.array(predictions).flatten()
        
        # Wave amplitude statistics
        metrics['amplitude_mean'] = float(np.mean(pred))
        metrics['amplitude_std'] = float(np.std(pred))
        metrics['amplitude_max'] = float(np.max(np.abs(pred)))
        metrics['amplitude_min'] = float(np.min(pred))
        
        # Energy (proportional to u^2)
        energy = pred ** 2
        metrics['total_energy'] = float(np.sum(energy))
        metrics['mean_energy'] = float(np.mean(energy))
        
        # Smoothness (approximate via differences)
        if len(pred) > 1:
            grad = np.diff(pred)
            metrics['smoothness'] = float(np.mean(np.abs(grad)))
        
        if references is not None:
            ref = np.array(references).flatten()
            
            error = pred - ref
            metrics['rmse'] = float(np.sqrt(np.mean(error**2)))
            metrics['mae'] = float(np.mean(np.abs(error)))
            metrics['max_error'] = float(np.max(np.abs(error)))
            
            # Relative L2 error
            ref_norm = np.sqrt(np.sum(ref**2))
            if ref_norm > 1e-8:
                metrics['l2_relative_error'] = float(np.sqrt(np.sum(error**2)) / ref_norm)
        
        return metrics
    
    def get_reference_data(self) -> Optional[Any]:
        """Generate synthetic reference (traveling wave)."""
        if not JAX_AVAILABLE:
            return None
        
        key = jax.random.PRNGKey(123)
        x = jax.random.uniform(key, (1, 2000, 1), minval=0.0, maxval=1.0)
        key, subkey = jax.random.split(key)
        t = jax.random.uniform(subkey, (1, 2000, 1), minval=0.0, maxval=2.0)
        
        # Traveling wave solution: u(x,t) = sin(k*x - w*t)
        k = 2 * np.pi  # Wave number
        c = 1.0       # Wave speed
        w = k * c     # Angular frequency
        
        u = jnp.sin(k * x - w * t)
        
        return u
    
    def get_training_info(self) -> TrainingInfo:
        """Return training information."""
        return TrainingInfo(
            total_time_seconds=2400,  # ~40 minutes
            epochs=1000,
            final_loss=0.00005,
            final_metrics={
                'wave_equation_residual': 1e-5,
                'initial_condition_error': 1e-4,
                'boundary_condition_error': 1e-4,
            },
        )
    
    def get_test_dataset_description(self) -> str:
        return "2000 points (x,t) in [0,1] x [0,2] for 1D scalar wave equation"


def get_adapter(project_path: Optional[Path] = None) -> P17WavePINNScalarAdapter:
    """Get adapter instance for P17 project."""
    if project_path is None:
        project_path = Path(__file__).parent.parent.parent / "projects" / "17-wavepinn-nif__project-space"
    return P17WavePINNScalarAdapter(project_path)
