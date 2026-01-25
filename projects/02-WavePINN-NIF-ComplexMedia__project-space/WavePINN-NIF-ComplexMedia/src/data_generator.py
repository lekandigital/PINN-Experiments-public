"""
Synthetic data generator for WavePINN-NIF.

Generates heterogeneous 2D/3D acoustic media with random slowness maps,
collocation points, and validation sets. Exports to HDF5 format.
"""

import jax
import jax.numpy as jnp
import numpy as np
import h5py
from typing import Tuple, Dict, Optional, List, Callable
from functools import partial


class SyntheticWaveData:
    """Generate synthetic wave propagation scenarios for PINN training."""
    
    def __init__(self, 
                 domain_size: Tuple[float, ...] = (1.0, 1.0),
                 n_collocation: int = 10000,
                 n_boundary: int = 1000,
                 n_initial: int = 1000,
                 t_max: float = 1.0,
                 seed: int = 42):
        """
        Initialize the synthetic data generator.
        
        Args:
            domain_size: Size of spatial domain, e.g., (Lx, Ly) for 2D
            n_collocation: Number of interior collocation points
            n_boundary: Number of boundary points per face
            n_initial: Number of initial condition points
            t_max: Maximum simulation time
            seed: Random seed for reproducibility
        """
        self.domain_size = domain_size
        self.ndim = len(domain_size)
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial
        self.t_max = t_max
        self.key = jax.random.PRNGKey(seed)
        
        # Validate dimensions
        assert self.ndim in [2, 3], f"Only 2D and 3D supported, got {self.ndim}D"
    
    def _split_key(self) -> jax.Array:
        """Split and update the random key."""
        self.key, subkey = jax.random.split(self.key)
        return subkey
    
    def generate_layered_medium(self, 
                                n_layers: int = 5,
                                c_min: float = 1.0,
                                c_max: float = 5.0,
                                grid_size: int = 128) -> jnp.ndarray:
        """
        Create a layered velocity model (horizontal layers).
        
        Args:
            n_layers: Number of horizontal layers
            c_min: Minimum wave speed
            c_max: Maximum wave speed
            grid_size: Resolution of the velocity grid
            
        Returns:
            2D array of wave speeds c(x, y)
        """
        subkey = self._split_key()
        
        # Generate random layer velocities
        layer_velocities = jax.random.uniform(
            subkey, 
            shape=(n_layers,),
            minval=c_min,
            maxval=c_max
        )
        
        # Create grid
        if self.ndim == 2:
            y = jnp.linspace(0, self.domain_size[1], grid_size)
            layer_boundaries = jnp.linspace(0, self.domain_size[1], n_layers + 1)
            
            # Assign velocities to each layer
            c_profile = jnp.zeros(grid_size)
            for i in range(n_layers):
                mask = (y >= layer_boundaries[i]) & (y < layer_boundaries[i + 1])
                c_profile = jnp.where(mask, layer_velocities[i], c_profile)
            
            # Extend to 2D (constant in x)
            c_field = jnp.tile(c_profile, (grid_size, 1))
        else:
            # 3D: layers in z direction
            z = jnp.linspace(0, self.domain_size[2], grid_size)
            layer_boundaries = jnp.linspace(0, self.domain_size[2], n_layers + 1)
            
            c_profile = jnp.zeros(grid_size)
            for i in range(n_layers):
                mask = (z >= layer_boundaries[i]) & (z < layer_boundaries[i + 1])
                c_profile = jnp.where(mask, layer_velocities[i], c_profile)
            
            c_field = jnp.tile(c_profile, (grid_size, grid_size, 1))
        
        return c_field
    
    def generate_random_medium(self, 
                               correlation_length: float = 0.1,
                               c_mean: float = 3.0,
                               c_std: float = 1.0,
                               c_min: float = 1.0,
                               c_max: float = 5.0,
                               grid_size: int = 128) -> jnp.ndarray:
        """
        Create a random heterogeneous medium using Gaussian random fields.
        
        Uses a simple spectral method to generate correlated random fields.
        
        Args:
            correlation_length: Spatial correlation length
            c_mean: Mean wave speed
            c_std: Standard deviation of wave speed
            c_min: Minimum wave speed (clipped)
            c_max: Maximum wave speed (clipped)
            grid_size: Resolution of the velocity grid
            
        Returns:
            ND array of wave speeds c(x, y, [z])
        """
        subkey = self._split_key()
        
        if self.ndim == 2:
            shape = (grid_size, grid_size)
            # Generate white noise in frequency domain
            noise_real = jax.random.normal(subkey, shape=shape)
            subkey = self._split_key()
            noise_imag = jax.random.normal(subkey, shape=shape)
            noise_freq = noise_real + 1j * noise_imag
            
            # Create Gaussian filter in frequency domain
            kx = jnp.fft.fftfreq(grid_size, d=self.domain_size[0]/grid_size)
            ky = jnp.fft.fftfreq(grid_size, d=self.domain_size[1]/grid_size)
            KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
            K2 = KX**2 + KY**2
            
            # Gaussian spectrum with correlation length
            sigma_k = 1.0 / (2 * jnp.pi * correlation_length)
            spectrum = jnp.exp(-K2 / (2 * sigma_k**2))
            
            # Apply filter and inverse FFT
            filtered_freq = noise_freq * spectrum
            c_field = jnp.real(jnp.fft.ifft2(filtered_freq))
            
        else:  # 3D
            shape = (grid_size, grid_size, grid_size)
            noise_real = jax.random.normal(subkey, shape=shape)
            subkey = self._split_key()
            noise_imag = jax.random.normal(subkey, shape=shape)
            noise_freq = noise_real + 1j * noise_imag
            
            kx = jnp.fft.fftfreq(grid_size, d=self.domain_size[0]/grid_size)
            ky = jnp.fft.fftfreq(grid_size, d=self.domain_size[1]/grid_size)
            kz = jnp.fft.fftfreq(grid_size, d=self.domain_size[2]/grid_size)
            KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing='ij')
            K2 = KX**2 + KY**2 + KZ**2
            
            sigma_k = 1.0 / (2 * jnp.pi * correlation_length)
            spectrum = jnp.exp(-K2 / (2 * sigma_k**2))
            
            filtered_freq = noise_freq * spectrum
            c_field = jnp.real(jnp.fft.ifftn(filtered_freq))
        
        # Normalize and scale to desired range
        c_field = (c_field - c_field.mean()) / (c_field.std() + 1e-8)
        c_field = c_mean + c_std * c_field
        c_field = jnp.clip(c_field, c_min, c_max)
        
        return c_field
    
    def generate_homogeneous_medium(self, 
                                    c_value: float = 3.0,
                                    grid_size: int = 128) -> jnp.ndarray:
        """Create a homogeneous medium with constant wave speed."""
        if self.ndim == 2:
            return jnp.full((grid_size, grid_size), c_value)
        else:
            return jnp.full((grid_size, grid_size, grid_size), c_value)
    
    def ricker_wavelet(self, t: jnp.ndarray, f0: float = 10.0) -> jnp.ndarray:
        """
        Ricker wavelet (Mexican hat) source function.
        
        w(t) = (1 - 2π²f₀²(t-tₛ)²) exp(-π²f₀²(t-tₛ)²)
        
        Args:
            t: Time array
            f0: Dominant frequency (Hz)
            
        Returns:
            Wavelet amplitude at times t
        """
        t_shift = 1.0 / f0  # Shift to ensure causality
        tau = t - t_shift
        arg = (jnp.pi * f0 * tau) ** 2
        return (1.0 - 2.0 * arg) * jnp.exp(-arg)
    
    def gaussian_source(self, 
                        t: jnp.ndarray, 
                        t0: float = 0.1, 
                        sigma: float = 0.05) -> jnp.ndarray:
        """
        Gaussian pulse source function.
        
        Args:
            t: Time array
            t0: Center time of the pulse
            sigma: Width of the Gaussian
            
        Returns:
            Source amplitude at times t
        """
        return jnp.exp(-((t - t0) ** 2) / (2 * sigma ** 2))
    
    def sample_collocation_points(self) -> Dict[str, jnp.ndarray]:
        """
        Sample interior, boundary, and initial condition points.
        
        Returns:
            Dictionary with keys 'interior', 'boundary', 'initial'
            containing coordinate arrays of shape (n_points, ndim+1)
        """
        # Interior collocation points (x, y, [z], t)
        subkey = self._split_key()
        
        if self.ndim == 2:
            minval = jnp.array([0.0, 0.0, 0.0])
            maxval = jnp.array([self.domain_size[0], self.domain_size[1], self.t_max])
        else:
            minval = jnp.array([0.0, 0.0, 0.0, 0.0])
            maxval = jnp.array([*self.domain_size, self.t_max])
        
        interior = jax.random.uniform(
            subkey, 
            shape=(self.n_collocation, self.ndim + 1),
            minval=minval,
            maxval=maxval
        )
        
        # Boundary points - sample each face
        boundary_points = self._sample_boundary_points()
        
        # Initial condition points (t=0)
        initial_points = self._sample_initial_points()
        
        return {
            'interior': interior,
            'boundary': boundary_points,
            'initial': initial_points
        }
    
    def _sample_boundary_points(self) -> Dict[str, jnp.ndarray]:
        """Sample points on each boundary face."""
        boundaries = {}
        
        if self.ndim == 2:
            # Left boundary (x=0)
            subkey = self._split_key()
            y_t = jax.random.uniform(subkey, (self.n_boundary, 2),
                                     minval=jnp.array([0.0, 0.0]),
                                     maxval=jnp.array([self.domain_size[1], self.t_max]))
            boundaries['left'] = jnp.concatenate([
                jnp.zeros((self.n_boundary, 1)),  # x=0
                y_t
            ], axis=1)
            
            # Right boundary (x=Lx)
            subkey = self._split_key()
            y_t = jax.random.uniform(subkey, (self.n_boundary, 2),
                                     minval=jnp.array([0.0, 0.0]),
                                     maxval=jnp.array([self.domain_size[1], self.t_max]))
            boundaries['right'] = jnp.concatenate([
                jnp.full((self.n_boundary, 1), self.domain_size[0]),  # x=Lx
                y_t
            ], axis=1)
            
            # Bottom boundary (y=0)
            subkey = self._split_key()
            x_t = jax.random.uniform(subkey, (self.n_boundary, 2),
                                     minval=jnp.array([0.0, 0.0]),
                                     maxval=jnp.array([self.domain_size[0], self.t_max]))
            boundaries['bottom'] = jnp.concatenate([
                x_t[:, :1],  # x
                jnp.zeros((self.n_boundary, 1)),  # y=0
                x_t[:, 1:]   # t
            ], axis=1)
            
            # Top boundary (y=Ly)
            subkey = self._split_key()
            x_t = jax.random.uniform(subkey, (self.n_boundary, 2),
                                     minval=jnp.array([0.0, 0.0]),
                                     maxval=jnp.array([self.domain_size[0], self.t_max]))
            boundaries['top'] = jnp.concatenate([
                x_t[:, :1],  # x
                jnp.full((self.n_boundary, 1), self.domain_size[1]),  # y=Ly
                x_t[:, 1:]   # t
            ], axis=1)
        
        else:  # 3D
            # Similar structure for 6 faces
            face_names = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
            for i, name in enumerate(face_names):
                subkey = self._split_key()
                # Sample 3D coordinates (2 spatial + time)
                coords = jax.random.uniform(subkey, (self.n_boundary, 3),
                                           minval=jnp.array([0.0, 0.0, 0.0]),
                                           maxval=jnp.array([*self.domain_size[:2], self.t_max]))
                
                if 'x' in name:
                    val = 0.0 if 'min' in name else self.domain_size[0]
                    boundaries[name] = jnp.concatenate([
                        jnp.full((self.n_boundary, 1), val),
                        coords
                    ], axis=1)
                elif 'y' in name:
                    val = 0.0 if 'min' in name else self.domain_size[1]
                    boundaries[name] = jnp.concatenate([
                        coords[:, :1],
                        jnp.full((self.n_boundary, 1), val),
                        coords[:, 1:]
                    ], axis=1)
                else:  # z
                    val = 0.0 if 'min' in name else self.domain_size[2]
                    boundaries[name] = jnp.concatenate([
                        coords[:, :2],
                        jnp.full((self.n_boundary, 1), val),
                        coords[:, 2:]
                    ], axis=1)
        
        return boundaries
    
    def _sample_initial_points(self) -> jnp.ndarray:
        """Sample initial condition points at t=0."""
        subkey = self._split_key()
        
        if self.ndim == 2:
            spatial = jax.random.uniform(subkey, (self.n_initial, 2),
                                        minval=jnp.array([0.0, 0.0]),
                                        maxval=jnp.array(self.domain_size))
            # Append t=0
            return jnp.concatenate([
                spatial,
                jnp.zeros((self.n_initial, 1))
            ], axis=1)
        else:
            spatial = jax.random.uniform(subkey, (self.n_initial, 3),
                                        minval=jnp.array([0.0, 0.0, 0.0]),
                                        maxval=jnp.array(self.domain_size))
            return jnp.concatenate([
                spatial,
                jnp.zeros((self.n_initial, 1))
            ], axis=1)
    
    def create_initial_condition(self,
                                 source_position: Tuple[float, ...],
                                 source_width: float = 0.1,
                                 amplitude: float = 1.0) -> Callable:
        """
        Create a Gaussian initial condition centered at source_position.
        
        Args:
            source_position: (x, y, [z]) position of source
            source_width: Width of Gaussian
            amplitude: Peak amplitude
            
        Returns:
            Function u0(x) giving initial displacement
        """
        source_pos = jnp.array(source_position)
        
        def u0(x_spatial):
            r2 = jnp.sum((x_spatial - source_pos) ** 2, axis=-1)
            return amplitude * jnp.exp(-r2 / (2 * source_width ** 2))
        
        return u0
    
    def save_to_hdf5(self, filename: str, data: Dict, 
                    velocity_field: Optional[jnp.ndarray] = None):
        """
        Export dataset to HDF5 format.
        
        Args:
            filename: Output file path
            data: Dictionary of collocation points
            velocity_field: Optional velocity model grid
        """
        with h5py.File(filename, 'w') as f:
            # Save collocation points
            for key, value in data.items():
                if isinstance(value, dict):
                    grp = f.create_group(key)
                    for subkey, subvalue in value.items():
                        grp.create_dataset(subkey, data=np.array(subvalue))
                else:
                    f.create_dataset(key, data=np.array(value))
            
            # Save velocity field if provided
            if velocity_field is not None:
                f.create_dataset('velocity_field', data=np.array(velocity_field))
            
            # Metadata
            f.attrs['domain_size'] = self.domain_size
            f.attrs['ndim'] = self.ndim
            f.attrs['t_max'] = self.t_max
            f.attrs['n_collocation'] = self.n_collocation
            f.attrs['n_boundary'] = self.n_boundary
            f.attrs['n_initial'] = self.n_initial
    
    @staticmethod
    def load_from_hdf5(filename: str) -> Dict:
        """Load dataset from HDF5 format."""
        data = {}
        with h5py.File(filename, 'r') as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    data[key] = {}
                    for subkey in f[key].keys():
                        data[key][subkey] = jnp.array(f[key][subkey][:])
                else:
                    data[key] = jnp.array(f[key][:])
            
            # Load attributes
            data['metadata'] = dict(f.attrs)
        
        return data


def create_data_batch(data: Dict, batch_size: int, key: jax.Array) -> Dict:
    """
    Create a random batch from the full dataset.
    
    Args:
        data: Full dataset dictionary
        batch_size: Number of samples per batch
        key: JAX random key
        
    Returns:
        Batched data dictionary
    """
    batch = {}
    
    # Sample interior points
    key, subkey = jax.random.split(key)
    n_interior = len(data['interior'])
    idx = jax.random.choice(subkey, n_interior, shape=(min(batch_size, n_interior),), replace=False)
    batch['interior'] = data['interior'][idx]
    
    # Sample from each boundary
    if isinstance(data['boundary'], dict):
        batch['boundary'] = {}
        for face, points in data['boundary'].items():
            key, subkey = jax.random.split(key)
            n_face = len(points)
            n_sample = min(batch_size // 4, n_face)
            idx = jax.random.choice(subkey, n_face, shape=(n_sample,), replace=False)
            batch['boundary'][face] = points[idx]
    
    # Sample initial points
    key, subkey = jax.random.split(key)
    n_initial = len(data['initial'])
    idx = jax.random.choice(subkey, n_initial, shape=(min(batch_size // 2, n_initial),), replace=False)
    batch['initial'] = data['initial'][idx]
    
    return batch


if __name__ == "__main__":
    # Quick test
    print("Testing SyntheticWaveData...")
    
    data_gen = SyntheticWaveData(
        domain_size=(1.0, 1.0),
        n_collocation=1000,
        n_boundary=200,
        n_initial=200,
        t_max=1.0
    )
    
    # Generate data
    data = data_gen.sample_collocation_points()
    print(f"Interior points: {data['interior'].shape}")
    print(f"Initial points: {data['initial'].shape}")
    print(f"Boundary faces: {list(data['boundary'].keys())}")
    
    # Generate velocity model
    c_layered = data_gen.generate_layered_medium(n_layers=5)
    print(f"Layered velocity field: {c_layered.shape}")
    
    c_random = data_gen.generate_random_medium(correlation_length=0.1)
    print(f"Random velocity field: {c_random.shape}")
    
    # Test wavelet
    t = jnp.linspace(0, 0.5, 100)
    wavelet = data_gen.ricker_wavelet(t, f0=10.0)
    print(f"Ricker wavelet range: [{wavelet.min():.4f}, {wavelet.max():.4f}]")
    
    print("✓ All data generation tests passed!")
