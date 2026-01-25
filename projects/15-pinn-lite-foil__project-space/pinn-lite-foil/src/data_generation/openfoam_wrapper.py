"""
OpenFOAM CFD Simulation Wrapper
Automates OpenFOAM case setup and execution for airfoil simulations.

Usage:
    python openfoam_wrapper.py --input data/raw/ --output data/processed/
"""

import argparse
import subprocess
import shutil
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


class OpenFOAMCase:
    """Manages OpenFOAM case setup and execution for 2D airfoil simulations."""
    
    def __init__(
        self,
        case_dir: Path,
        airfoil_file: Path,
        aoa: float = 0.0,
        reynolds: float = 1e5,
        chord: float = 1.0,
        nu: float = 1e-5
    ):
        """
        Initialize OpenFOAM case.
        
        Args:
            case_dir: Directory for OpenFOAM case
            airfoil_file: Path to airfoil .dat file
            aoa: Angle of attack in degrees
            reynolds: Reynolds number
            chord: Chord length
            nu: Kinematic viscosity
        """
        self.case_dir = Path(case_dir)
        self.airfoil_file = Path(airfoil_file)
        self.aoa = aoa
        self.reynolds = reynolds
        self.chord = chord
        self.nu = nu
        
        # Calculate inlet velocity from Reynolds number
        self.u_inf = self.reynolds * self.nu / self.chord
        
        # Velocity components based on AoA
        aoa_rad = np.radians(self.aoa)
        self.u_x = self.u_inf * np.cos(aoa_rad)
        self.u_y = self.u_inf * np.sin(aoa_rad)
        
    def create_directory_structure(self) -> None:
        """Create OpenFOAM case directory structure."""
        dirs = ['0', 'constant', 'system']
        for d in dirs:
            (self.case_dir / d).mkdir(parents=True, exist_ok=True)
    
    def write_control_dict(self) -> None:
        """Write system/controlDict file."""
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         2000;
deltaT          1;
writeControl    timeStep;
writeInterval   500;
purgeWrite      2;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    fieldAverage
    {{
        type            fieldAverage;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        fields
        (
            U
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
            p
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
        );
    }}
    
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        
        patches         (airfoil);
        rho             rhoInf;
        rhoInf          1;
        liftDir         ({-np.sin(np.radians(self.aoa)):.6f} {np.cos(np.radians(self.aoa)):.6f} 0);
        dragDir         ({np.cos(np.radians(self.aoa)):.6f} {np.sin(np.radians(self.aoa)):.6f} 0);
        CofR            (0.25 0 0);
        pitchAxis       (0 0 1);
        magUInf         {self.u_inf:.6f};
        lRef            {self.chord:.6f};
        Aref            {self.chord:.6f};
    }}
}}
"""
        with open(self.case_dir / 'system' / 'controlDict', 'w') as f:
            f.write(content)
    
    def write_fv_schemes(self) -> None:
        """Write system/fvSchemes file."""
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
"""
        with open(self.case_dir / 'system' / 'fvSchemes', 'w') as f:
            f.write(content)
    
    def write_fv_solution(self) -> None:
        """Write system/fvSolution file."""
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        smoother        DICGaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }

    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    
    residualControl
    {
        p               1e-6;
        U               1e-6;
        "(k|omega)"     1e-6;
    }
}

relaxationFactors
{
    equations
    {
        U               0.7;
        p               0.3;
        k               0.5;
        omega           0.5;
    }
}
"""
        with open(self.case_dir / 'system' / 'fvSolution', 'w') as f:
            f.write(content)
    
    def write_block_mesh_dict(self) -> None:
        """Write system/blockMeshDict for far-field mesh."""
        # Domain size (chord lengths)
        x_min, x_max = -10, 20
        y_min, y_max = -10, 10
        
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale   {self.chord};

vertices
(
    ({x_min} {y_min} -0.5)
    ({x_max} {y_min} -0.5)
    ({x_max} {y_max} -0.5)
    ({x_min} {y_max} -0.5)
    ({x_min} {y_min}  0.5)
    ({x_max} {y_min}  0.5)
    ({x_max} {y_max}  0.5)
    ({x_min} {y_max}  0.5)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (100 80 1) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }}
    top
    {{
        type patch;
        faces
        (
            (3 7 6 2)
        );
    }}
    bottom
    {{
        type patch;
        faces
        (
            (0 1 5 4)
        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);
"""
        with open(self.case_dir / 'system' / 'blockMeshDict', 'w') as f:
            f.write(content)
    
    def write_snappy_hex_mesh_dict(self) -> None:
        """Write system/snappyHexMeshDict for airfoil refinement."""
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}

castellatedMesh true;
snap            true;
addLayers       true;

geometry
{
    airfoil.stl
    {
        type triSurfaceMesh;
        name airfoil;
    }
}

castellatedMeshControls
{
    maxLocalCells   100000;
    maxGlobalCells  2000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    
    features
    (
        {
            file "airfoil.eMesh";
            level 4;
        }
    );
    
    refinementSurfaces
    {
        airfoil
        {
            level (4 4);
            patchInfo
            {
                type wall;
            }
        }
    }
    
    resolveFeatureAngle 30;
    
    refinementRegions
    {
    }
    
    locationInMesh (0.5 0.5 0);
    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      100;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes   true;
    layers
    {
        airfoil
        {
            nSurfaceLayers 5;
        }
    }
    
    expansionRatio  1.2;
    finalLayerThickness 0.5;
    minThickness    0.1;
    nGrow           0;
    featureAngle    60;
    nRelaxIter      5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals  3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter      50;
}

meshQualityControls
{
    maxNonOrtho     65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave      80;
    minVol          1e-13;
    minTetQuality   -1e30;
    minArea         -1;
    minTwist        0.02;
    minDeterminant  0.001;
    minFaceWeight   0.05;
    minVolRatio     0.01;
    minTriangleTwist -1;
    nSmoothScale    4;
    errorReduction  0.75;
}

writeFlags
(
    scalarLevels
    layerSets
    layerFields
);

mergeTolerance  1e-6;
"""
        with open(self.case_dir / 'system' / 'snappyHexMeshDict', 'w') as f:
            f.write(content)
    
    def write_transport_properties(self) -> None:
        """Write constant/transportProperties file."""
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}

transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] {self.nu};
"""
        with open(self.case_dir / 'constant' / 'transportProperties', 'w') as f:
            f.write(content)
    
    def write_turbulence_properties(self) -> None:
        """Write constant/turbulenceProperties for k-omega SST."""
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}

simulationType  RAS;

RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
"""
        with open(self.case_dir / 'constant' / 'turbulenceProperties', 'w') as f:
            f.write(content)
    
    def write_initial_conditions(self) -> None:
        """Write initial condition files in 0/ directory."""
        # Pressure
        p_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    top
    {{
        type            zeroGradient;
    }}
    bottom
    {{
        type            zeroGradient;
    }}
    airfoil
    {{
        type            zeroGradient;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
        with open(self.case_dir / '0' / 'p', 'w') as f:
            f.write(p_content)
        
        # Velocity
        u_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({self.u_x:.6f} {self.u_y:.6f} 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({self.u_x:.6f} {self.u_y:.6f} 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    top
    {{
        type            fixedValue;
        value           uniform ({self.u_x:.6f} {self.u_y:.6f} 0);
    }}
    bottom
    {{
        type            fixedValue;
        value           uniform ({self.u_x:.6f} {self.u_y:.6f} 0);
    }}
    airfoil
    {{
        type            noSlip;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
        with open(self.case_dir / '0' / 'U', 'w') as f:
            f.write(u_content)
        
        # Turbulent kinetic energy
        # k = 1.5 * (I * U)^2, I = 0.01 (1% turbulence intensity)
        k_val = 1.5 * (0.01 * self.u_inf)**2
        k_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {k_val:.6e};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {k_val:.6e};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    top
    {{
        type            fixedValue;
        value           uniform {k_val:.6e};
    }}
    bottom
    {{
        type            fixedValue;
        value           uniform {k_val:.6e};
    }}
    airfoil
    {{
        type            kqRWallFunction;
        value           uniform {k_val:.6e};
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
        with open(self.case_dir / '0' / 'k', 'w') as f:
            f.write(k_content)
        
        # Specific dissipation rate
        # omega = k / (nu_t), nu_t ~ 10 * nu
        omega_val = k_val / (10 * self.nu)
        omega_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      omega;
}}

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform {omega_val:.6e};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {omega_val:.6e};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    top
    {{
        type            fixedValue;
        value           uniform {omega_val:.6e};
    }}
    bottom
    {{
        type            fixedValue;
        value           uniform {omega_val:.6e};
    }}
    airfoil
    {{
        type            omegaWallFunction;
        value           uniform {omega_val:.6e};
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
        with open(self.case_dir / '0' / 'omega', 'w') as f:
            f.write(omega_content)
        
        # Turbulent viscosity
        nut_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}}

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform {10 * self.nu:.6e};

boundaryField
{{
    inlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    outlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    top
    {{
        type            calculated;
        value           uniform 0;
    }}
    bottom
    {{
        type            calculated;
        value           uniform 0;
    }}
    airfoil
    {{
        type            nutkWallFunction;
        value           uniform 0;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
        with open(self.case_dir / '0' / 'nut', 'w') as f:
            f.write(nut_content)
    
    def setup_case(self) -> None:
        """Set up complete OpenFOAM case."""
        print(f"  Setting up case: AoA={self.aoa}°, Re={self.reynolds:.0e}")
        
        self.create_directory_structure()
        self.write_control_dict()
        self.write_fv_schemes()
        self.write_fv_solution()
        self.write_block_mesh_dict()
        self.write_snappy_hex_mesh_dict()
        self.write_transport_properties()
        self.write_turbulence_properties()
        self.write_initial_conditions()
        
        # Copy airfoil geometry
        shutil.copy(self.airfoil_file, self.case_dir / 'constant' / 'triSurface' / 'airfoil.stl')
    
    def run_mesh(self) -> bool:
        """Generate mesh using blockMesh and snappyHexMesh."""
        try:
            # Run blockMesh
            subprocess.run(
                ['blockMesh'],
                cwd=self.case_dir,
                check=True,
                capture_output=True
            )
            
            # Run surfaceFeatureExtract
            subprocess.run(
                ['surfaceFeatureExtract'],
                cwd=self.case_dir,
                check=True,
                capture_output=True
            )
            
            # Run snappyHexMesh
            subprocess.run(
                ['snappyHexMesh', '-overwrite'],
                cwd=self.case_dir,
                check=True,
                capture_output=True
            )
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"  Mesh generation failed: {e}")
            return False
    
    def run_simulation(self) -> bool:
        """Run simpleFoam solver."""
        try:
            result = subprocess.run(
                ['simpleFoam'],
                cwd=self.case_dir,
                check=True,
                capture_output=True,
                timeout=3600  # 1 hour timeout
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"  Simulation failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            print(f"  Simulation timed out")
            return False
    
    def extract_results(self) -> Optional[Dict]:
        """Extract results from completed simulation."""
        # Find latest time directory
        time_dirs = [
            d for d in os.listdir(self.case_dir) 
            if d.replace('.', '').isdigit() and float(d) > 0
        ]
        
        if not time_dirs:
            return None
        
        latest_time = max(time_dirs, key=float)
        time_dir = self.case_dir / latest_time
        
        # Read force coefficients
        force_file = self.case_dir / 'postProcessing' / 'forceCoeffs' / '0' / 'coefficient.dat'
        
        results = {
            'aoa': self.aoa,
            'reynolds': self.reynolds,
            'u_inf': self.u_inf,
            'time_dir': str(time_dir),
            'converged': True
        }
        
        if force_file.exists():
            with open(force_file) as f:
                lines = f.readlines()
                if lines:
                    # Get last line (final coefficients)
                    last_line = lines[-1].split()
                    if len(last_line) >= 4:
                        results['cl'] = float(last_line[3])
                        results['cd'] = float(last_line[2])
        
        return results


def convert_dat_to_stl(dat_file: Path, stl_file: Path, span: float = 1.0) -> None:
    """Convert 2D airfoil .dat file to 3D STL for OpenFOAM."""
    # Read airfoil coordinates
    with open(dat_file) as f:
        lines = f.readlines()[1:]  # Skip header
    
    coords = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            coords.append([float(parts[0]), float(parts[1])])
    
    coords = np.array(coords)
    n_points = len(coords)
    
    # Create STL triangles by extruding in z-direction
    with open(stl_file, 'w') as f:
        f.write("solid airfoil\n")
        
        half_span = span / 2
        
        for i in range(n_points - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            
            # Front face triangle 1
            f.write(f"  facet normal 0 0 -1\n")
            f.write(f"    outer loop\n")
            f.write(f"      vertex {x0} {y0} {-half_span}\n")
            f.write(f"      vertex {x1} {y1} {-half_span}\n")
            f.write(f"      vertex {x0} {y0} {half_span}\n")
            f.write(f"    endloop\n")
            f.write(f"  endfacet\n")
            
            # Front face triangle 2
            f.write(f"  facet normal 0 0 -1\n")
            f.write(f"    outer loop\n")
            f.write(f"      vertex {x1} {y1} {-half_span}\n")
            f.write(f"      vertex {x1} {y1} {half_span}\n")
            f.write(f"      vertex {x0} {y0} {half_span}\n")
            f.write(f"    endloop\n")
            f.write(f"  endfacet\n")
        
        f.write("endsolid airfoil\n")


def run_simulation_sweep(
    airfoil_file: Path,
    output_dir: Path,
    aoa_range: List[float],
    reynolds_range: List[float],
    nu: float = 1e-5
) -> List[Dict]:
    """
    Run simulation sweep over AoA and Reynolds number.
    
    Args:
        airfoil_file: Path to airfoil .dat file
        output_dir: Output directory for cases
        aoa_range: List of angles of attack
        reynolds_range: List of Reynolds numbers
        nu: Kinematic viscosity
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Convert to STL
    stl_file = output_dir / 'airfoil.stl'
    convert_dat_to_stl(airfoil_file, stl_file)
    
    for re in reynolds_range:
        for aoa in aoa_range:
            case_name = f"Re{re:.0e}_AoA{aoa:+.1f}".replace('+', 'p').replace('-', 'm')
            case_dir = output_dir / case_name
            
            # Create and run case
            case = OpenFOAMCase(
                case_dir=case_dir,
                airfoil_file=stl_file,
                aoa=aoa,
                reynolds=re,
                nu=nu
            )
            
            case.setup_case()
            
            if case.run_mesh() and case.run_simulation():
                result = case.extract_results()
                if result:
                    results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run OpenFOAM CFD simulations for airfoils'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Input directory containing airfoil .dat files'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output directory for simulation results'
    )
    parser.add_argument(
        '--aoa_min', type=float, default=-5.0,
        help='Minimum angle of attack (degrees)'
    )
    parser.add_argument(
        '--aoa_max', type=float, default=15.0,
        help='Maximum angle of attack (degrees)'
    )
    parser.add_argument(
        '--aoa_step', type=float, default=5.0,
        help='Angle of attack step (degrees)'
    )
    parser.add_argument(
        '--re_min', type=float, default=1e5,
        help='Minimum Reynolds number'
    )
    parser.add_argument(
        '--re_max', type=float, default=5e5,
        help='Maximum Reynolds number'
    )
    parser.add_argument(
        '--re_steps', type=int, default=3,
        help='Number of Reynolds number steps'
    )
    parser.add_argument(
        '--nu', type=float, default=1e-5,
        help='Kinematic viscosity'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Set up cases without running simulations'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate AoA and Re ranges
    aoa_range = np.arange(args.aoa_min, args.aoa_max + args.aoa_step, args.aoa_step).tolist()
    re_range = np.logspace(np.log10(args.re_min), np.log10(args.re_max), args.re_steps).tolist()
    
    print(f"AoA range: {aoa_range}")
    print(f"Re range: {[f'{r:.1e}' for r in re_range]}")
    
    # Find airfoil files
    airfoil_files = list(input_dir.glob('*.dat'))
    print(f"\nFound {len(airfoil_files)} airfoil files")
    
    all_results = []
    
    for airfoil_file in airfoil_files:
        print(f"\nProcessing: {airfoil_file.stem}")
        airfoil_output = output_dir / airfoil_file.stem
        
        if args.dry_run:
            # Just set up cases
            stl_file = airfoil_output / 'airfoil.stl'
            airfoil_output.mkdir(parents=True, exist_ok=True)
            (airfoil_output / 'constant' / 'triSurface').mkdir(parents=True, exist_ok=True)
            convert_dat_to_stl(airfoil_file, stl_file)
            
            for re in re_range:
                for aoa in aoa_range:
                    case_name = f"Re{re:.0e}_AoA{aoa:+.1f}".replace('+', 'p').replace('-', 'm')
                    case_dir = airfoil_output / case_name
                    
                    case = OpenFOAMCase(
                        case_dir=case_dir,
                        airfoil_file=stl_file,
                        aoa=aoa,
                        reynolds=re,
                        nu=args.nu
                    )
                    case.setup_case()
                    print(f"  Set up: {case_name}")
        else:
            results = run_simulation_sweep(
                airfoil_file=airfoil_file,
                output_dir=airfoil_output,
                aoa_range=aoa_range,
                reynolds_range=re_range,
                nu=args.nu
            )
            all_results.extend(results)
    
    # Save results summary
    if all_results:
        with open(output_dir / 'simulation_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Saved {len(all_results)} results to {output_dir / 'simulation_results.json'}")


if __name__ == '__main__':
    main()
