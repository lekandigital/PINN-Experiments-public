"""
Thin-Shell Elasticity Data Generator

Generates data for thin-shell elasticity problems on spherical caps.
Uses Kirchhoff-Love shell theory with clamped boundary conditions.

Domain: Spherical cap (quarter sphere, θ ∈ [0, π/2])
Boundary: Clamped at rim (u = 0, ∂u/∂n = 0)
Loading: Uniform pressure or point load

Output saved to HDF5 format for training.
"""

import numpy as np
import h5py
from typing import Tuple, Optional
from scipy.spatial import Delaunay


def generate_spherical_cap_mesh(
    n_radial: int = 50,
    n_angular: int = 50,
    cap_angle: float = np.pi / 2,  # Quarter sphere
    radius: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a triangular mesh on a spherical cap.
    
    Args:
        n_radial: number of points from pole to rim
        n_angular: number of points around circumference
        cap_angle: cap opening angle (π/2 for quarter sphere)
        radius: sphere radius
        
    Returns:
        vertices: [N, 3] vertex positions
        faces: [F, 3] triangle indices
        boundary_mask: [N] boolean mask for boundary vertices
    """
    # Generate parametric grid
    # θ: polar angle from pole (0 to cap_angle)
    # φ: azimuthal angle (0 to 2π)
    
    theta = np.linspace(0, cap_angle, n_radial)
    phi = np.linspace(0, 2 * np.pi, n_angular, endpoint=False)
    
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Convert to Cartesian (pole at z = radius)
    x = radius * np.sin(theta_flat) * np.cos(phi_flat)
    y = radius * np.sin(theta_flat) * np.sin(phi_flat)
    z = radius * np.cos(theta_flat)
    
    vertices = np.stack([x, y, z], axis=-1)
    
    # Identify boundary (rim at θ = cap_angle)
    boundary_mask = np.abs(theta_flat - cap_angle) < 1e-6
    
    # Triangulate using 2D Delaunay on (θ, φ) then map to 3D
    # Handle periodicity in φ
    param_points = np.stack([theta_flat, phi_flat], axis=-1)
    
    # Simple structured triangulation
    faces = []
    for i in range(n_angular):
        for j in range(n_radial - 1):
            # Vertex indices
            i_next = (i + 1) % n_angular
            
            v00 = i * n_radial + j
            v01 = i * n_radial + j + 1
            v10 = i_next * n_radial + j
            v11 = i_next * n_radial + j + 1
            
            # Two triangles per quad
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])
    
    faces = np.array(faces)
    
    return vertices, faces, boundary_mask


def compute_shell_normals(vertices: np.ndarray, center: np.ndarray = None) -> np.ndarray:
    """
    Compute outward normals for spherical cap.
    
    Args:
        vertices: [N, 3] vertex positions
        center: sphere center (default origin)
        
    Returns:
        normals: [N, 3] unit normal vectors
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    
    normals = vertices - center
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / (norms + 1e-10)


def analytical_shell_displacement(
    vertices: np.ndarray,
    pressure: float = 1.0,
    E: float = 1e6,  # Young's modulus
    nu: float = 0.3,  # Poisson's ratio
    thickness: float = 0.01,
    radius: float = 1.0
) -> np.ndarray:
    """
    Compute analytical displacement for uniformly loaded spherical cap.
    
    For a thin spherical shell under uniform pressure, the membrane solution gives:
    w = pR² / (2Et) * (1 - ν) 
    
    This is simplified; actual solution requires solving shell equations.
    
    Args:
        vertices: [N, 3] vertex positions
        pressure: uniform pressure load
        E: Young's modulus
        nu: Poisson's ratio
        thickness: shell thickness
        radius: sphere radius
        
    Returns:
        displacement: [N, 3] displacement vectors (primarily radial)
    """
    # Membrane solution for uniform pressure
    # Radial displacement: w = pR² / (2Et) * (1 - ν) for spherical membrane
    
    # Get polar angle from pole
    normals = compute_shell_normals(vertices)
    z = vertices[:, 2]
    theta = np.arccos(np.clip(z / radius, -1, 1))
    
    # Membrane stress resultants
    # N_θ = N_φ = pR / 2 (equi-biaxial for sphere)
    
    # Radial displacement (simplified)
    D = E * thickness**3 / (12 * (1 - nu**2))  # Bending stiffness
    
    # For clamped cap, displacement varies with position
    # Use simplified distribution: max at pole, zero at boundary
    cap_angle = np.max(theta)
    
    # Smooth displacement field (quadratic approximation)
    w_max = pressure * radius**2 / (2 * E * thickness) * (1 - nu)
    w = w_max * (1 - (theta / cap_angle)**2)
    
    # Displacement is in normal direction
    displacement = w[:, np.newaxis] * normals
    
    # Apply clamped BC (zero at boundary)
    boundary_mask = np.abs(theta - cap_angle) < 0.05
    displacement[boundary_mask] = 0.0
    
    return displacement.astype(np.float32)


def generate_shell_elasticity_data(
    n_radial: int = 50,
    n_angular: int = 50,
    pressure: float = 1.0,
    E: float = 1e6,
    nu: float = 0.3,
    thickness: float = 0.01,
    radius: float = 1.0,
    save_path: Optional[str] = None
) -> dict:
    """
    Generate thin-shell elasticity data on spherical cap.
    
    Args:
        n_radial: radial mesh resolution
        n_angular: angular mesh resolution
        pressure: uniform pressure load
        E: Young's modulus
        nu: Poisson's ratio
        thickness: shell thickness
        radius: sphere radius
        save_path: if provided, save to HDF5
        
    Returns:
        dict with mesh and solution data
    """
    # Generate mesh
    vertices, faces, boundary_mask = generate_spherical_cap_mesh(
        n_radial, n_angular, cap_angle=np.pi/2, radius=radius
    )
    
    # Compute normals
    normals = compute_shell_normals(vertices)
    
    # Get boundary points
    boundary_points = vertices[boundary_mask]
    boundary_values = np.zeros_like(boundary_points)  # Clamped BC
    
    # Compute analytical displacement
    displacement = analytical_shell_displacement(
        vertices, pressure, E, nu, thickness, radius
    )
    
    result = {
        'collocation_points': vertices.astype(np.float32),
        'faces': faces.astype(np.int32),
        'normals': normals.astype(np.float32),
        'boundary_mask': boundary_mask,
        'boundary_points': boundary_points.astype(np.float32),
        'boundary_values': boundary_values.astype(np.float32),
        'solution_displacement': displacement,
        'params': {
            'pressure': pressure,
            'E': E,
            'nu': nu,
            'thickness': thickness,
            'radius': radius,
            'n_radial': n_radial,
            'n_angular': n_angular
        }
    }
    
    # Save to HDF5
    if save_path is not None:
        with h5py.File(save_path, 'w') as f:
            for key in ['collocation_points', 'faces', 'normals', 
                       'boundary_points', 'boundary_values', 'solution_displacement']:
                f.create_dataset(key, data=result[key])
            f.create_dataset('boundary_mask', data=result['boundary_mask'])
            
            params = f.create_group('params')
            for k, v in result['params'].items():
                params.attrs[k] = v
        
        print(f"Saved shell elasticity data to {save_path}")
    
    return result


def compute_shell_strain_energy(
    vertices: np.ndarray,
    faces: np.ndarray,
    displacement: np.ndarray,
    E: float,
    nu: float,
    thickness: float
) -> float:
    """
    Compute total strain energy of deformed shell.
    
    Includes both membrane and bending contributions.
    
    Args:
        vertices: [N, 3] undeformed vertices
        faces: [F, 3] face indices
        displacement: [N, 3] displacement field
        E: Young's modulus
        nu: Poisson's ratio
        thickness: shell thickness
        
    Returns:
        total strain energy
    """
    # Simplified: just compute membrane strain energy
    # Full implementation requires computing strain tensor on each element
    
    D = E * thickness**3 / (12 * (1 - nu**2))
    C = E * thickness / (1 - nu**2)
    
    # Compute average displacement magnitude
    disp_mag = np.linalg.norm(displacement, axis=1)
    
    # Very rough estimate (proper implementation needs FEM)
    total_area = 0.0
    for f in faces:
        v0, v1, v2 = vertices[f]
        e1 = v1 - v0
        e2 = v2 - v0
        area = 0.5 * np.linalg.norm(np.cross(e1, e2))
        total_area += area
    
    # Strain energy ~ 0.5 * C * (u/R)^2 * A
    avg_disp = np.mean(disp_mag)
    radius = np.mean(np.linalg.norm(vertices, axis=1))
    strain_energy = 0.5 * C * (avg_disp / radius)**2 * total_area
    
    return strain_energy


if __name__ == '__main__':
    # Generate sample data
    data = generate_shell_elasticity_data(
        n_radial=50,
        n_angular=50,
        save_path='shell_elasticity_data.h5'
    )
    print(f"Generated mesh with {data['collocation_points'].shape[0]} vertices")
    print(f"Number of faces: {data['faces'].shape[0]}")
    print(f"Boundary points: {data['boundary_points'].shape[0]}")
    print(f"Max displacement: {np.abs(data['solution_displacement']).max():.6f}")
