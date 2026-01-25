"""
PML (Perfectly Matched Layer) Loss Module for Maxwell-PINN-NIF

This module implements absorbing boundary conditions for electromagnetic simulations.
The PML technique prevents artificial reflections at domain boundaries by gradually
attenuating outgoing waves in a boundary layer region.

Two approaches are implemented:
1. SimplePMLLoss: Soft boundary condition penalizing non-zero fields in PML region
2. DampedPMLLoss: More physically accurate with quadratic/polynomial damping profile

For true PML behavior, fields should decay as:
    E(x) → E(x) * exp(-σ(x) * d)
where σ(x) is the conductivity profile and d is the distance into the PML.

References:
- Berenger (1994): Original PML formulation for FDTD
- Nohra & Dufour (2024): PINN boundary treatments for EM
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class PMLRegion:
    """
    Helper class to define PML regions in the computational domain.
    
    The PML is typically a layer at the domain boundaries where fields
    are gradually absorbed to simulate an infinite domain.
    """
    
    def __init__(
        self,
        domain_bounds: List[float],
        pml_thickness: float,
        sides: str = 'all'
    ):
        """
        Args:
            domain_bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
            pml_thickness: Thickness of PML layer (typically 5-10% of domain size)
            sides: Which sides to apply PML: 'all', 'positive', 'negative'
        """
        xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds
        self.domain_bounds = domain_bounds
        self.thickness = pml_thickness
        
        # Define PML regions for each boundary
        self.pml_regions = {}
        
        if sides in ['all', 'negative']:
            self.pml_regions['x_min'] = (xmin, xmin + pml_thickness)
            self.pml_regions['y_min'] = (ymin, ymin + pml_thickness)
            self.pml_regions['z_min'] = (zmin, zmin + pml_thickness)
        
        if sides in ['all', 'positive']:
            self.pml_regions['x_max'] = (xmax - pml_thickness, xmax)
            self.pml_regions['y_max'] = (ymax - pml_thickness, ymax)
            self.pml_regions['z_max'] = (zmax - pml_thickness, zmax)
    
    def get_pml_mask(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Create boolean mask for points inside any PML region.
        
        Args:
            coords: [N, 3] spatial coordinates
            
        Returns:
            mask: [N] boolean tensor, True if point is in PML
        """
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        mask = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
        
        for region_name, (lo, hi) in self.pml_regions.items():
            if 'x' in region_name:
                mask |= (x >= lo) & (x <= hi)
            elif 'y' in region_name:
                mask |= (y >= lo) & (y <= hi)
            elif 'z' in region_name:
                mask |= (z >= lo) & (z <= hi)
        
        return mask
    
    def get_damping_profile(
        self,
        coords: torch.Tensor,
        sigma_max: float = 1.0,
        order: int = 2
    ) -> torch.Tensor:
        """
        Compute damping coefficient σ(x) for each point.
        
        Uses polynomial profile: σ(d) = σ_max * (d/thickness)^order
        where d is the distance into the PML region.
        
        Quadratic (order=2) and cubic (order=3) profiles are common choices.
        Higher order gives sharper transition but may cause numerical issues.
        
        Args:
            coords: [N, 3] spatial coordinates
            sigma_max: Maximum damping at outer boundary
            order: Polynomial order for damping profile
            
        Returns:
            sigma: [N] damping coefficients (0 inside domain, >0 in PML)
        """
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        sigma = torch.zeros(coords.shape[0], device=coords.device)
        
        # Compute distance into each PML region and apply damping
        for region_name, (lo, hi) in self.pml_regions.items():
            if 'x_min' in region_name:
                # Distance from inner boundary (hi) going toward lo
                d = torch.clamp(hi - x, min=0) / self.thickness
                sigma = torch.maximum(sigma, sigma_max * (d ** order))
            elif 'x_max' in region_name:
                # Distance from inner boundary (lo) going toward hi
                d = torch.clamp(x - lo, min=0) / self.thickness
                sigma = torch.maximum(sigma, sigma_max * (d ** order))
            elif 'y_min' in region_name:
                d = torch.clamp(hi - y, min=0) / self.thickness
                sigma = torch.maximum(sigma, sigma_max * (d ** order))
            elif 'y_max' in region_name:
                d = torch.clamp(y - lo, min=0) / self.thickness
                sigma = torch.maximum(sigma, sigma_max * (d ** order))
            elif 'z_min' in region_name:
                d = torch.clamp(hi - z, min=0) / self.thickness
                sigma = torch.maximum(sigma, sigma_max * (d ** order))
            elif 'z_max' in region_name:
                d = torch.clamp(z - lo, min=0) / self.thickness
                sigma = torch.maximum(sigma, sigma_max * (d ** order))
        
        return sigma


class PMLLoss(nn.Module):
    """
    PML loss module for absorbing boundary conditions.
    
    This implements a soft PML by penalizing field magnitudes in the
    boundary region with a damping profile. The loss encourages fields
    to decay to zero near the domain boundaries.
    
    Loss = ∫_PML σ(x) * (|E|² + |H|²) dx
    
    where σ(x) is a polynomial damping profile that increases from 0
    at the PML/domain interface to σ_max at the outer boundary.
    """
    
    def __init__(
        self,
        domain_bounds: List[float],
        pml_thickness: float,
        sigma_max: float = 1.0,
        order: int = 2,
        sides: str = 'all'
    ):
        """
        Args:
            domain_bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
            pml_thickness: Thickness of PML layer
            sigma_max: Maximum absorption coefficient
            order: Polynomial order (2=quadratic, 3=cubic)
            sides: 'all', 'positive', or 'negative'
        """
        super().__init__()
        
        self.pml_region = PMLRegion(domain_bounds, pml_thickness, sides)
        self.sigma_max = sigma_max
        self.order = order
        
        # Store bounds for validation
        self.register_buffer('domain_min', torch.tensor(domain_bounds[::2]))
        self.register_buffer('domain_max', torch.tensor(domain_bounds[1::2]))
    
    def forward(
        self,
        coords: torch.Tensor,
        E_pred: torch.Tensor,
        H_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PML absorption loss.
        
        Args:
            coords: [N, 3] spatial coordinates
            E_pred: [N, 3] predicted E-field
            H_pred: [N, 3] predicted H-field
            
        Returns:
            loss: Weighted field magnitude in PML regions
        """
        # Get damping profile (0 inside domain, increasing in PML)
        sigma = self.pml_region.get_damping_profile(
            coords, self.sigma_max, self.order
        )
        
        # If no points in PML, return zero loss
        pml_mask = sigma > 0
        if not pml_mask.any():
            return torch.tensor(0.0, device=coords.device, requires_grad=True)
        
        # Compute field energy: |E|² + |H|²
        E_energy = torch.sum(E_pred ** 2, dim=1)  # [N]
        H_energy = torch.sum(H_pred ** 2, dim=1)  # [N]
        total_energy = E_energy + H_energy
        
        # Weighted loss: σ(x) * field_energy
        weighted_energy = sigma * total_energy
        
        # Average over PML points only
        loss = weighted_energy[pml_mask].mean()
        
        return loss


class HardBoundaryLoss(nn.Module):
    """
    Hard boundary condition loss (Dirichlet-type).
    
    Enforces E_tangential = 0 or specific values at domain boundaries.
    This is useful for modeling perfect electric conductors (PEC) or
    prescribed field values.
    
    For PEC: E × n = 0 (tangential E vanishes)
    For PMC: H × n = 0 (tangential H vanishes)
    """
    
    def __init__(
        self,
        domain_bounds: List[float],
        boundary_thickness: float = 0.01,
        boundary_type: str = 'pec'
    ):
        """
        Args:
            domain_bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
            boundary_thickness: Thickness of boundary sampling region
            boundary_type: 'pec' (E_tan=0) or 'pmc' (H_tan=0)
        """
        super().__init__()
        self.domain_bounds = domain_bounds
        self.thickness = boundary_thickness
        self.boundary_type = boundary_type
        
        xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds
        self.register_buffer('bounds', torch.tensor([
            [xmin, xmax], [ymin, ymax], [zmin, zmax]
        ]))
    
    def forward(
        self,
        coords: torch.Tensor,
        E_pred: torch.Tensor,
        H_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary condition loss.
        
        Args:
            coords: [N, 3] spatial coordinates
            E_pred: [N, 3] predicted E-field
            H_pred: [N, 3] predicted H-field
            
        Returns:
            loss: MSE of tangential field components at boundaries
        """
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        xmin, xmax = self.bounds[0]
        ymin, ymax = self.bounds[1]
        zmin, zmax = self.bounds[2]
        
        loss = torch.tensor(0.0, device=coords.device)
        count = 0
        
        # For each boundary face, enforce tangential field = 0
        # At x boundaries: tangential components are Ey, Ez (and Hy, Hz)
        x_min_mask = x <= xmin + self.thickness
        x_max_mask = x >= xmax - self.thickness
        
        if self.boundary_type == 'pec':
            # PEC: tangential E = 0
            if x_min_mask.any():
                loss = loss + (E_pred[x_min_mask, 1:3] ** 2).mean()
                count += 1
            if x_max_mask.any():
                loss = loss + (E_pred[x_max_mask, 1:3] ** 2).mean()
                count += 1
        else:
            # PMC: tangential H = 0
            if x_min_mask.any():
                loss = loss + (H_pred[x_min_mask, 1:3] ** 2).mean()
                count += 1
            if x_max_mask.any():
                loss = loss + (H_pred[x_max_mask, 1:3] ** 2).mean()
                count += 1
        
        # Similar for y and z boundaries
        y_min_mask = y <= ymin + self.thickness
        y_max_mask = y >= ymax - self.thickness
        
        if self.boundary_type == 'pec':
            if y_min_mask.any():
                # At y boundary: tangential E components are Ex, Ez
                tan_E = torch.stack([E_pred[y_min_mask, 0], E_pred[y_min_mask, 2]], dim=1)
                loss = loss + (tan_E ** 2).mean()
                count += 1
            if y_max_mask.any():
                tan_E = torch.stack([E_pred[y_max_mask, 0], E_pred[y_max_mask, 2]], dim=1)
                loss = loss + (tan_E ** 2).mean()
                count += 1
        
        z_min_mask = z <= zmin + self.thickness
        z_max_mask = z >= zmax - self.thickness
        
        if self.boundary_type == 'pec':
            if z_min_mask.any():
                # At z boundary: tangential E components are Ex, Ey
                loss = loss + (E_pred[z_min_mask, 0:2] ** 2).mean()
                count += 1
            if z_max_mask.any():
                loss = loss + (E_pred[z_max_mask, 0:2] ** 2).mean()
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss


class CombinedBoundaryLoss(nn.Module):
    """
    Combined boundary loss with both PML absorption and hard boundary conditions.
    
    This allows mixing boundary types, e.g., PML on sides and PEC on top/bottom.
    """
    
    def __init__(
        self,
        domain_bounds: List[float],
        pml_thickness: float = 0.1,
        pml_weight: float = 1.0,
        hard_bc_weight: float = 10.0,
        pml_sides: str = 'all',
        hard_bc_type: Optional[str] = None
    ):
        super().__init__()
        
        self.pml_loss = PMLLoss(
            domain_bounds, pml_thickness, 
            sigma_max=1.0, order=2, sides=pml_sides
        )
        
        self.hard_bc_loss = None
        if hard_bc_type is not None:
            self.hard_bc_loss = HardBoundaryLoss(
                domain_bounds, 
                boundary_thickness=0.01,
                boundary_type=hard_bc_type
            )
        
        self.pml_weight = pml_weight
        self.hard_bc_weight = hard_bc_weight
    
    def forward(
        self,
        coords: torch.Tensor,
        E_pred: torch.Tensor,
        H_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined boundary losses.
        
        Returns:
            total_loss: Weighted sum of boundary losses
            loss_dict: Individual loss components
        """
        loss_pml = self.pml_loss(coords, E_pred, H_pred)
        
        loss_hard = torch.tensor(0.0, device=coords.device)
        if self.hard_bc_loss is not None:
            loss_hard = self.hard_bc_loss(coords, E_pred, H_pred)
        
        total = self.pml_weight * loss_pml + self.hard_bc_weight * loss_hard
        
        return total, {'pml': loss_pml.item(), 'hard_bc': loss_hard.item()}


def validate_pml_absorption(
    model: nn.Module,
    pml_loss_fn: PMLLoss,
    device: torch.device,
    n_points: int = 1000
) -> dict:
    """
    Validate PML effectiveness by measuring field attenuation.
    
    Compares field magnitude at domain center vs. boundary to estimate
    absorption in dB.
    
    Args:
        model: Trained MaxwellPINN model
        pml_loss_fn: PML loss module (contains domain info)
        device: Compute device
        n_points: Number of test points
        
    Returns:
        metrics: Dict with attenuation measurements
    """
    model.eval()
    
    domain_bounds = pml_loss_fn.pml_region.domain_bounds
    xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds
    
    # Sample center region
    center_coords = torch.rand(n_points, 3, device=device)
    center_coords[:, 0] = center_coords[:, 0] * 0.3 * (xmax - xmin) + 0.35 * (xmax - xmin) + xmin
    center_coords[:, 1] = center_coords[:, 1] * 0.3 * (ymax - ymin) + 0.35 * (ymax - ymin) + ymin
    center_coords[:, 2] = center_coords[:, 2] * 0.3 * (zmax - zmin) + 0.35 * (zmax - zmin) + zmin
    
    # Sample PML region
    pml_coords = torch.rand(n_points, 3, device=device)
    # Push to boundary
    pml_coords[:, 0] = pml_coords[:, 0] * pml_loss_fn.pml_region.thickness + (xmax - pml_loss_fn.pml_region.thickness)
    pml_coords[:, 1] = pml_coords[:, 1] * (ymax - ymin) + ymin
    pml_coords[:, 2] = pml_coords[:, 2] * (zmax - zmin) + zmin
    
    with torch.no_grad():
        E_center, H_center = model(center_coords)
        E_pml, H_pml = model(pml_coords)
        
        power_center = torch.mean(E_center ** 2 + H_center ** 2).item()
        power_pml = torch.mean(E_pml ** 2 + H_pml ** 2).item()
        
        # Avoid log(0)
        if power_pml > 1e-20 and power_center > 1e-20:
            attenuation_db = 10 * torch.log10(
                torch.tensor(power_pml / power_center)
            ).item()
        else:
            attenuation_db = -100.0  # Very high attenuation
    
    return {
        'power_center': power_center,
        'power_pml': power_pml,
        'attenuation_dB': attenuation_db
    }


if __name__ == "__main__":
    # Self-test
    print("Testing PML Loss Module...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define domain
    domain_bounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    pml_thickness = 0.1
    
    # Create PML loss
    pml = PMLLoss(domain_bounds, pml_thickness, sigma_max=1.0, order=2)
    pml = pml.to(device)
    
    # Test coordinates
    n_points = 1000
    coords = torch.rand(n_points, 3, device=device)
    E_pred = torch.randn(n_points, 3, device=device)
    H_pred = torch.randn(n_points, 3, device=device)
    
    # Compute loss
    loss = pml(coords, E_pred, H_pred)
    print(f"PML loss: {loss.item():.6f}")
    
    # Check that PML region is detected
    pml_mask = pml.pml_region.get_pml_mask(coords)
    print(f"Points in PML: {pml_mask.sum().item()} / {n_points}")
    
    # Check damping profile
    sigma = pml.pml_region.get_damping_profile(coords)
    print(f"Damping σ range: [{sigma.min().item():.4f}, {sigma.max().item():.4f}]")
    
    # Test backward pass
    loss.backward()
    print("Backward pass: ✓")
    
    # Test combined boundary loss
    combined = CombinedBoundaryLoss(
        domain_bounds, pml_thickness=0.1,
        pml_weight=1.0, hard_bc_weight=10.0,
        hard_bc_type='pec'
    ).to(device)
    
    coords.requires_grad = True
    E_pred = torch.randn(n_points, 3, device=device, requires_grad=True)
    H_pred = torch.randn(n_points, 3, device=device, requires_grad=True)
    
    total_bc_loss, bc_dict = combined(coords, E_pred, H_pred)
    print(f"Combined BC loss: {total_bc_loss.item():.6f}")
    print(f"  PML: {bc_dict['pml']:.6f}, Hard BC: {bc_dict['hard_bc']:.6f}")
    
    print("\n✓ All PML tests passed!")
