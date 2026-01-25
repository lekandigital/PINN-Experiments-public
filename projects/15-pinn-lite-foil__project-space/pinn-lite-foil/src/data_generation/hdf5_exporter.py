"""
HDF5 Dataset Exporter
Parses OpenFOAM results and exports to HDF5 format for PINN training.

Usage:
    python hdf5_exporter.py --input data/processed/ --output data/processed/dataset.h5
"""

import argparse
import os
import re
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FlowFieldData:
    """Container for flow field data from a single simulation."""
    x: np.ndarray          # x-coordinates
    y: np.ndarray          # y-coordinates  
    u: np.ndarray          # x-velocity
    v: np.ndarray          # y-velocity
    p: np.ndarray          # pressure
    aoa: float             # angle of attack
    reynolds: float        # Reynolds number
    airfoil_params: Dict   # NACA parameters (m, p, t)
    cl: Optional[float]    # lift coefficient
    cd: Optional[float]    # drag coefficient


def parse_openfoam_vector_field(filepath: Path) -> np.ndarray:
    """Parse OpenFOAM vector field file (e.g., U)."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find internal field data
    match = re.search(r'internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\);', 
                      content, re.DOTALL)
    
    if not match:
        # Try uniform field
        match = re.search(r'internalField\s+uniform\s+\(([\d\s.\-e]+)\)', content)
        if match:
            values = [float(x) for x in match.group(1).split()]
            return np.array([values])
        raise ValueError(f"Could not parse vector field: {filepath}")
    
    n_cells = int(match.group(1))
    data_str = match.group(2)
    
    # Parse vectors
    vectors = []
    for vec_match in re.finditer(r'\(([\d.\-e\s]+)\)', data_str):
        values = [float(x) for x in vec_match.group(1).split()]
        vectors.append(values)
    
    return np.array(vectors)


def parse_openfoam_scalar_field(filepath: Path) -> np.ndarray:
    """Parse OpenFOAM scalar field file (e.g., p)."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find internal field data
    match = re.search(r'internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\);',
                      content, re.DOTALL)
    
    if not match:
        # Try uniform field
        match = re.search(r'internalField\s+uniform\s+([\d.\-e]+)', content)
        if match:
            return np.array([float(match.group(1))])
        raise ValueError(f"Could not parse scalar field: {filepath}")
    
    n_cells = int(match.group(1))
    data_str = match.group(2)
    
    # Parse scalars
    values = [float(x) for x in data_str.split()]
    
    return np.array(values)


def parse_openfoam_points(filepath: Path) -> np.ndarray:
    """Parse OpenFOAM points file for cell centers."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find points data
    match = re.search(r'(\d+)\s*\((.*?)\)', content, re.DOTALL)
    
    if not match:
        raise ValueError(f"Could not parse points: {filepath}")
    
    data_str = match.group(2)
    
    # Parse points
    points = []
    for pt_match in re.finditer(r'\(([\d.\-e\s]+)\)', data_str):
        values = [float(x) for x in pt_match.group(1).split()]
        points.append(values)
    
    return np.array(points)


def get_cell_centers(case_dir: Path, time_dir: str) -> np.ndarray:
    """Get cell center coordinates from OpenFOAM case."""
    # First try to find cellCentres in postProcessing
    cell_centers_file = case_dir / time_dir / 'C'
    
    if not cell_centers_file.exists():
        # Try running postProcess to generate cell centers
        # For now, estimate from mesh
        points_file = case_dir / 'constant' / 'polyMesh' / 'points'
        if points_file.exists():
            return parse_openfoam_points(points_file)
    
    return parse_openfoam_vector_field(cell_centers_file)


def parse_case_directory(case_dir: Path) -> Optional[FlowFieldData]:
    """
    Parse a single OpenFOAM case directory.
    
    Args:
        case_dir: Path to OpenFOAM case directory
    
    Returns:
        FlowFieldData object or None if parsing fails
    """
    # Find latest time directory
    time_dirs = []
    for item in os.listdir(case_dir):
        try:
            t = float(item)
            if t > 0:
                time_dirs.append(item)
        except ValueError:
            continue
    
    if not time_dirs:
        return None
    
    latest_time = max(time_dirs, key=float)
    time_dir = case_dir / latest_time
    
    try:
        # Parse velocity field
        u_file = time_dir / 'U'
        if not u_file.exists():
            return None
        
        velocity = parse_openfoam_vector_field(u_file)
        u = velocity[:, 0]
        v = velocity[:, 1]
        
        # Parse pressure field
        p_file = time_dir / 'p'
        if not p_file.exists():
            return None
        
        pressure = parse_openfoam_scalar_field(p_file)
        
        # Get cell centers
        # For simplicity, use mesh points (would need proper cell center calculation)
        points_file = case_dir / 'constant' / 'polyMesh' / 'points'
        if points_file.exists():
            points = parse_openfoam_points(points_file)
            x = points[:len(pressure), 0]
            y = points[:len(pressure), 1]
        else:
            # Generate placeholder coordinates
            n_points = len(pressure)
            x = np.zeros(n_points)
            y = np.zeros(n_points)
        
        # Extract case parameters from directory name
        case_name = case_dir.name
        aoa = 0.0
        reynolds = 1e5
        
        # Parse AoA from case name (e.g., "Re1e+05_AoAp5.0")
        aoa_match = re.search(r'AoA([mp])([\d.]+)', case_name)
        if aoa_match:
            aoa = float(aoa_match.group(2))
            if aoa_match.group(1) == 'm':
                aoa = -aoa
        
        re_match = re.search(r'Re([\d.e+]+)', case_name)
        if re_match:
            reynolds = float(re_match.group(1))
        
        # Load force coefficients if available
        cl, cd = None, None
        force_file = case_dir / 'postProcessing' / 'forceCoeffs' / '0' / 'coefficient.dat'
        if force_file.exists():
            with open(force_file) as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].split()
                    if len(last_line) >= 4 and not last_line[0].startswith('#'):
                        cd = float(last_line[2])
                        cl = float(last_line[3])
        
        return FlowFieldData(
            x=x,
            y=y,
            u=u[:len(x)],
            v=v[:len(x)],
            p=pressure[:len(x)],
            aoa=aoa,
            reynolds=reynolds,
            airfoil_params={},
            cl=cl,
            cd=cd
        )
    
    except Exception as e:
        print(f"  Error parsing {case_dir}: {e}")
        return None


def create_synthetic_dataset(
    n_airfoils: int = 50,
    n_aoa: int = 5,
    n_points: int = 1000,
    seed: int = 42
) -> List[FlowFieldData]:
    """
    Create synthetic dataset for testing (no OpenFOAM required).
    
    This generates approximate flow fields based on analytical solutions
    and empirical correlations for training data.
    
    Args:
        n_airfoils: Number of airfoil configurations
        n_aoa: Number of angles of attack per airfoil
        n_points: Number of sample points per case
        seed: Random seed
    
    Returns:
        List of FlowFieldData objects
    """
    np.random.seed(seed)
    
    datasets = []
    
    # Generate NACA parameters
    m_vals = np.random.uniform(0.01, 0.04, n_airfoils)
    p_vals = np.random.uniform(0.2, 0.6, n_airfoils)
    t_vals = np.random.uniform(0.08, 0.18, n_airfoils)
    
    aoa_range = np.linspace(-5, 15, n_aoa)
    reynolds = 1e5
    u_inf = 10.0  # m/s
    
    for i in range(n_airfoils):
        m, p, t = m_vals[i], p_vals[i], t_vals[i]
        
        for aoa in aoa_range:
            # Generate sample points in domain
            # Focus on near-airfoil region
            x = np.random.uniform(-0.5, 2.0, n_points)
            y = np.random.uniform(-0.5, 0.5, n_points)
            
            # Approximate flow field (simplified potential flow + boundary layer effects)
            aoa_rad = np.radians(aoa)
            
            # Base uniform flow
            u_base = u_inf * np.cos(aoa_rad)
            v_base = u_inf * np.sin(aoa_rad)
            
            # Distance from airfoil center
            r = np.sqrt((x - 0.5)**2 + y**2)
            r = np.maximum(r, 0.1)  # Avoid singularities
            
            # Perturbation based on distance (simplified)
            perturbation = 0.5 * t * np.exp(-r / 0.2)
            
            # Add camber effect
            camber_effect = m * np.exp(-((x - p)**2) / 0.1) * np.sign(y)
            
            u = u_base * (1 - perturbation) + 0.1 * np.random.randn(n_points)
            v = v_base + u_inf * camber_effect + 0.05 * np.random.randn(n_points)
            
            # Pressure coefficient (Bernoulli-like)
            speed = np.sqrt(u**2 + v**2)
            p = 0.5 * (u_inf**2 - speed**2) + 0.01 * np.random.randn(n_points)
            
            # Approximate Cl, Cd (thin airfoil theory + empirical)
            cl_theory = 2 * np.pi * aoa_rad + 2 * np.pi * m
            cl = cl_theory + 0.1 * np.random.randn()
            cd = 0.01 + 0.05 * aoa_rad**2 + 0.02 * t + 0.005 * np.random.randn()
            
            data = FlowFieldData(
                x=x,
                y=y,
                u=u,
                v=v,
                p=p,
                aoa=float(aoa),
                reynolds=reynolds,
                airfoil_params={'m': float(m), 'p': float(p), 't': float(t)},
                cl=float(cl),
                cd=float(cd)
            )
            datasets.append(data)
    
    return datasets


def export_to_hdf5(
    datasets: List[FlowFieldData],
    output_path: Path,
    compression: str = 'gzip',
    train_split: float = 0.8
) -> None:
    """
    Export flow field data to HDF5 format.
    
    Args:
        datasets: List of FlowFieldData objects
        output_path: Output HDF5 file path
        compression: Compression algorithm ('gzip', 'lzf', or None)
        train_split: Fraction of data for training
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split data
    np.random.shuffle(datasets)
    n_train = int(len(datasets) * train_split)
    train_data = datasets[:n_train]
    test_data = datasets[n_train:]
    
    with h5py.File(output_path, 'w') as f:
        # Store metadata
        f.attrs['n_samples'] = len(datasets)
        f.attrs['n_train'] = len(train_data)
        f.attrs['n_test'] = len(test_data)
        f.attrs['train_split'] = train_split
        
        # Create train and test groups
        for split_name, split_data in [('train', train_data), ('test', test_data)]:
            grp = f.create_group(split_name)
            
            for i, data in enumerate(split_data):
                case_grp = grp.create_group(f'case_{i:04d}')
                
                # Store arrays
                case_grp.create_dataset('x', data=data.x, compression=compression)
                case_grp.create_dataset('y', data=data.y, compression=compression)
                case_grp.create_dataset('u', data=data.u, compression=compression)
                case_grp.create_dataset('v', data=data.v, compression=compression)
                case_grp.create_dataset('p', data=data.p, compression=compression)
                
                # Store scalars as attributes
                case_grp.attrs['aoa'] = data.aoa
                case_grp.attrs['reynolds'] = data.reynolds
                
                if data.cl is not None:
                    case_grp.attrs['cl'] = data.cl
                if data.cd is not None:
                    case_grp.attrs['cd'] = data.cd
                
                # Store airfoil parameters
                if data.airfoil_params:
                    params_grp = case_grp.create_group('airfoil_params')
                    for key, val in data.airfoil_params.items():
                        params_grp.attrs[key] = val
    
    print(f"✓ Exported {len(datasets)} cases to {output_path}")
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")


def load_from_hdf5(
    filepath: Path,
    split: str = 'train'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        split: 'train' or 'test'
    
    Returns:
        Tuple of (x, y, aoa, u, v, p) arrays
    """
    all_x, all_y, all_aoa = [], [], []
    all_u, all_v, all_p = [], [], []
    
    with h5py.File(filepath, 'r') as f:
        grp = f[split]
        
        for case_name in grp.keys():
            case = grp[case_name]
            
            x = case['x'][:]
            y = case['y'][:]
            u = case['u'][:]
            v = case['v'][:]
            p = case['p'][:]
            aoa = np.full_like(x, case.attrs['aoa'])
            
            all_x.append(x)
            all_y.append(y)
            all_aoa.append(aoa)
            all_u.append(u)
            all_v.append(v)
            all_p.append(p)
    
    return (
        np.concatenate(all_x),
        np.concatenate(all_y),
        np.concatenate(all_aoa),
        np.concatenate(all_u),
        np.concatenate(all_v),
        np.concatenate(all_p)
    )


def main():
    parser = argparse.ArgumentParser(
        description='Export OpenFOAM results to HDF5 dataset'
    )
    parser.add_argument(
        '--input', type=str,
        help='Input directory containing OpenFOAM cases'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output HDF5 file path'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Generate synthetic dataset (no OpenFOAM required)'
    )
    parser.add_argument(
        '--n_airfoils', type=int, default=50,
        help='Number of airfoils for synthetic dataset'
    )
    parser.add_argument(
        '--n_aoa', type=int, default=5,
        help='Number of AoA per airfoil for synthetic dataset'
    )
    parser.add_argument(
        '--n_points', type=int, default=1000,
        help='Number of sample points per case'
    )
    parser.add_argument(
        '--train_split', type=float, default=0.8,
        help='Fraction of data for training'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    if args.synthetic:
        print(f"Generating synthetic dataset...")
        print(f"  Airfoils: {args.n_airfoils}")
        print(f"  AoA per airfoil: {args.n_aoa}")
        print(f"  Points per case: {args.n_points}")
        
        datasets = create_synthetic_dataset(
            n_airfoils=args.n_airfoils,
            n_aoa=args.n_aoa,
            n_points=args.n_points,
            seed=args.seed
        )
    else:
        if not args.input:
            raise ValueError("--input required when not using --synthetic")
        
        input_dir = Path(args.input)
        print(f"Parsing OpenFOAM results from {input_dir}...")
        
        datasets = []
        
        # Find all airfoil directories
        for airfoil_dir in input_dir.iterdir():
            if not airfoil_dir.is_dir():
                continue
            
            # Find all case directories within airfoil directory
            for case_dir in airfoil_dir.iterdir():
                if not case_dir.is_dir():
                    continue
                
                # Check if this looks like an OpenFOAM case
                if not (case_dir / 'system').exists():
                    continue
                
                print(f"  Processing: {case_dir.name}")
                data = parse_case_directory(case_dir)
                
                if data is not None:
                    # Try to get airfoil parameters from metadata
                    metadata_file = airfoil_dir / 'metadata.json'
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                            # Match airfoil by directory name
                            for sample in metadata.get('samples', []):
                                if sample['designation'] in airfoil_dir.name:
                                    data.airfoil_params = {
                                        'm': sample['m'],
                                        'p': sample['p'],
                                        't': sample['t']
                                    }
                                    break
                    
                    datasets.append(data)
        
        if not datasets:
            print("No valid OpenFOAM cases found. Generating synthetic dataset instead...")
            datasets = create_synthetic_dataset(
                n_airfoils=args.n_airfoils,
                n_aoa=args.n_aoa,
                n_points=args.n_points,
                seed=args.seed
            )
    
    export_to_hdf5(
        datasets=datasets,
        output_path=Path(args.output),
        train_split=args.train_split
    )


if __name__ == '__main__':
    main()
