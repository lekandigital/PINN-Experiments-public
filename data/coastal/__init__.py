"""
Shared Coastal Data Infrastructure for PINN-Experiments.

This module provides real-world NOAA data access for coastal physics projects:
- Project 01 (GeoPINN-Manifold): Coastal manifold geometry
- Project 02 (WavePINN-NIF-ComplexMedia): Ocean sound speed profiles
- Project 06 (CoastFlow-GNN): Bathymetry, tides, waves, mesh generation
- Project 16 (SurfPINN): Dual Eulerian/Lagrangian coastal data
- Project 17 (WavePINN-NIF-Scalar): Wave transformation data

Data Sources:
- NOAA CUDEM: Coastal bathymetry/topography
- NOAA CO-OPS: Tide gauge and current observations
- NOAA NDBC: Wave buoy measurements
- World Ocean Atlas (WOA): Temperature, salinity, sound speed
- Natural Earth: Coastline geometry

Usage:
    from data.coastal import CoastalDataManager, REGIONS
    
    # Quick start - get complete data bundle
    manager = CoastalDataManager(cache_dir="./coastal_cache")
    bundle = manager.get_bundle(
        region="chesapeake_bay",
        start_date="2023-01-01",
        end_date="2023-01-31",
    )
    
    # Access components
    mesh = bundle.mesh           # Computational mesh for FEM/GNN
    pyg_data = mesh.to_pyg_data()  # PyTorch Geometric format
    bathymetry = bundle.bathymetry
    tidal_forcing = bundle.tidal_forcing
    validation = bundle.validation
    
    # Or use individual downloaders
    from data.coastal.downloaders import CUDEMDownloader, COOPSDownloader
    
    cudem = CUDEMDownloader()
    bathy = cudem.download(bbox)
"""

# Core config and datatypes
from .config import (
    BoundingBox,
    RegionConfig,
    REGIONS,
    get_region_config,
    get_utm_zone,
    get_utm_epsg,
)
from .datatypes import (
    BathymetryGrid,
    TideStation,
    HarmonicConstituent,
    WaterLevelTimeSeries,
    CurrentTimeSeries,
    WaveBuoy,
    WaveBuoyTimeSeries,
    OceanProfile,
    OceanProfileGrid,
    Shoreline,
    CoastalMesh,
    TidalForcing,
    WaveForcing,
    WindForcing,
    ValidationPoint,
    ValidationDataset,
    CoastalDataBundle,
)

# Unified manager
from .manager import CoastalDataManager, get_coastal_data

# Downloaders
from .downloaders import (
    COOPSDownloader,
    NDBCDownloader,
    CUDEMDownloader,
    WOADownloader,
    ShorelineDownloader,
)

# Processors
from .processors import (
    CoastalMeshGenerator,
    MeshParameters,
    ForcingBuilder,
    ForcingParameters,
    ValidationBuilder,
    ValidationParameters,
)

__version__ = "0.1.0"
__all__ = [
    # Config
    "BoundingBox",
    "RegionConfig",
    "REGIONS",
    "get_region_config",
    "get_utm_zone",
    "get_utm_epsg",
    # Datatypes
    "BathymetryGrid",
    "TideStation",
    "HarmonicConstituent",
    "WaterLevelTimeSeries",
    "CurrentTimeSeries",
    "WaveBuoy",
    "WaveBuoyTimeSeries",
    "OceanProfile",
    "OceanProfileGrid",
    "Shoreline",
    "CoastalMesh",
    "TidalForcing",
    "WaveForcing",
    "WindForcing",
    "ValidationPoint",
    "ValidationDataset",
    "CoastalDataBundle",
    # Manager
    "CoastalDataManager",
    "get_coastal_data",
    # Downloaders
    "COOPSDownloader",
    "NDBCDownloader",
    "CUDEMDownloader",
    "WOADownloader",
    "ShorelineDownloader",
    # Processors
    "CoastalMeshGenerator",
    "MeshParameters",
    "ForcingBuilder",
    "ForcingParameters",
    "ValidationBuilder",
    "ValidationParameters",
]
