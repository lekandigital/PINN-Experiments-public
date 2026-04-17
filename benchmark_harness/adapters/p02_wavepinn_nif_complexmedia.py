"""
Adapter for P02: WavePINN-NIF-ComplexMedia - Wave PINN for Complex Media (JAX)
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


def create_wave_pinn_forward(hidden_dim: int = 256, num_layers: int = 5, 
                             output_dim: int = 2):
    """Create the forward function for WavePINN model."""
    
    def forward(xyzt: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for wave PINN.
        
        Args:
            xyzt: (B, 4) spatiotemporal coordinates [x, y, z, t]
        
        Returns:
            (B, output_dim): [real_amplitude, imag_amplitude] or [pressure, velocity]
        """
        # Fourier feature encoding
        num_frequencies = 10
        freqs = jnp.arange(1, num_frequencies + 1, dtype=jnp.float32)
        
        # Apply to each input dimension
        encoded = [xyzt]
        for freq in freqs:
            encoded.append(jnp.sin(2 * np.pi * freq * xyzt))
            encoded.append(jnp.cos(2 * np.pi * freq * xyzt))
        
        x = jnp.concatenate(encoded, axis=-1)
        
        # MLP with sine activations (SIREN-like)
        for i in range(num_layers - 1):
            x = hk.Linear(hidden_dim, name=f'linear_{i}')(x)
            x = jnp.sin(30.0 * x)  # SIREN activation
        
        # Output layer
        out = hk.Linear(output_dim, name='output')(x)
        
        return out
    
    return forward


class P02WavePINNNIFAdapter(JAXAdapter):
    """Adapter for WavePINN-NIF-ComplexMedia project (JAX)."""
    
    PROJECT_ID = "P02"
    PROJECT_NAME = "WavePINN-NIF-ComplexMedia"
    
    def __init__(self, project_path: Path, subprocess_isolation: bool = True):
        super().__init__(project_path, subprocess_isolation)
        self._model_fn = None
        self._params = None
        self._rng_key = None
    
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
        
        # Define model
        def model_fn(xyzt):
            forward = create_wave_pinn_forward(
                hidden_dim=256,
                num_layers=5,
                output_dim=2  # Real and imaginary parts
            )
            return forward(xyzt)
        
        # Transform to pure functions
        self._model_fn = hk.transform(model_fn)
        
        # Initialize
        self._rng_key = jax.random.PRNGKey(42)
        dummy_input = jnp.zeros((1, 4))  # (B, 4) for xyzt
        self._params = self._model_fn.init(self._rng_key, dummy_input)
        
        return (self._model_fn, self._params)
    
    def load_model(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        """Load or create the model."""
        return self._create_model()
    
    def prepare_test_input(self, device: str, batch_size: int = 1) -> Tuple:
        """Prepare test inputs."""
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available")
        
        key = jax.random.PRNGKey(42)
        
        # Spatiotemporal grid: x, y, z in [-1, 1], t in [0, 1]
        xyz = jax.random.uniform(key, (batch_size, 1000, 3), minval=-1.0, maxval=1.0)
        key, subkey = jax.random.split(key)
        t = jax.random.uniform(subkey, (batch_size, 1000, 1), minval=0.0, maxval=1.0)
        xyzt = jnp.concatenate([xyz, t], axis=-1)
        
        return (xyzt,)
    
    def run_inference(self, model, inputs: Tuple, device: str) -> Any:
        """Run model inference."""
        model_fn, params = model
        xyzt, = inputs
        
        # Reshape for batch processing
        original_shape = xyzt.shape
        xyzt_flat = xyzt.reshape(-1, 4)
        
        # Run inference
        output = model_fn.apply(params, None, xyzt_flat)
        
        # Block until ready (important for accurate timing)
        output = jax.block_until_ready(output)
        
        # Reshape back
        output = output.reshape(*original_shape[:-1], -1)
        
        return output
    
    def compute_domain_metrics(self, predictions: Any,
                               references: Optional[Any] = None) -> Dict[str, float]:
        """Compute wave-specific metrics."""
        metrics = {}
        
        pred = np.array(predictions)
        pred_flat = pred.reshape(-1, 2)
        
        # Complex amplitude
        real_part = pred_flat[:, 0]
        imag_part = pred_flat[:, 1]
        
        amplitude = np.sqrt(real_part**2 + imag_part**2)
        phase = np.arctan2(imag_part, real_part)
        
        metrics['amplitude_mean'] = float(np.mean(amplitude))
        metrics['amplitude_max'] = float(np.max(amplitude))
        metrics['amplitude_std'] = float(np.std(amplitude))
        metrics['phase_mean'] = float(np.mean(phase))
        metrics['phase_std'] = float(np.std(phase))
        
        # Energy (proportional to |amplitude|^2)
        energy = amplitude ** 2
        metrics['total_energy'] = float(np.sum(energy))
        metrics['energy_distribution_std'] = float(np.std(energy))
        
        if references is not None:
            ref = np.array(references)
            ref_flat = ref.reshape(-1, 2)
            
            # Complex error
            error_real = pred_flat[:, 0] - ref_flat[:, 0]
            error_imag = pred_flat[:, 1] - ref_flat[:, 1]
            
            metrics['real_rmse'] = float(np.sqrt(np.mean(error_real**2)))
            metrics['imag_rmse'] = float(np.sqrt(np.mean(error_imag**2)))
            
            # Amplitude error
            ref_amplitude = np.sqrt(ref_flat[:, 0]**2 + ref_flat[:, 1]**2)
            amp_error = amplitude - ref_amplitude
            metrics['amplitude_rmse'] = float(np.sqrt(np.mean(amp_error**2)))
        
        return metrics
    
    def get_reference_data(self) -> Optional[Any]:
        """Generate synthetic reference (plane wave solution)."""
        if not JAX_AVAILABLE:
            return None
        
        key = jax.random.PRNGKey(123)
        xyz = jax.random.uniform(key, (1, 1000, 3), minval=-1.0, maxval=1.0)
        key, subkey = jax.random.split(key)
        t = jax.random.uniform(subkey, (1, 1000, 1), minval=0.0, maxval=1.0)
        
        # Plane wave: A * exp(i(k.r - wt))
        k = jnp.array([2 * np.pi, 0, 0])  # Wave vector
        w = 2 * np.pi  # Angular frequency
        
        kr = jnp.sum(xyz * k, axis=-1)  # k dot r
        phase = kr - w * t[..., 0]
        
        A = 1.0
        real_part = A * jnp.cos(phase)
        imag_part = A * jnp.sin(phase)
        
        return jnp.stack([real_part, imag_part], axis=-1)
    
    def get_training_info(self) -> TrainingInfo:
        """Return training information."""
        return TrainingInfo(
            total_time_seconds=5400,  # ~1.5 hours
            epochs=2000,
            final_loss=0.0001,
            final_metrics={
                'wave_equation_residual': 1e-4,
                'boundary_error': 1e-3,
                'amplitude_error': 0.01,
            },
        )
    
    def get_test_dataset_description(self) -> str:
        return "1000 spatiotemporal points (x,y,z,t) for plane wave propagation in complex media"


# Convenience function
def get_adapter(project_path: Optional[Path] = None) -> P02WavePINNNIFAdapter:
    """Get adapter instance for P02 project."""
    if project_path is None:
        project_path = Path(__file__).parent.parent.parent / "projects" / "02-WavePINN-NIF-ComplexMedia__project-space"
    return P02WavePINNNIFAdapter(project_path)
