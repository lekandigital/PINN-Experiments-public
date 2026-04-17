# Coastal Data Infrastructure

Shared data infrastructure providing real-world NOAA data for coastal physics-informed ML projects.

## Update (April 17, 2026)

- Downloader coverage now spans CO-OPS, NDBC, CUDEM, shoreline, and WOA sources.
- Processors and integration helpers were expanded for mesh generation, forcing assembly, validation bundles, and project-06 handoff.
- This package now acts as the shared coastal data layer for projects 01, 02, 06, 16, and 17.

## Quick Start

```python
from data.coastal import CoastalDataManager, REGIONS

# Initialize manager with cache directory
manager = CoastalDataManager(cache_dir="./coastal_cache")

# Get complete data bundle for Chesapeake Bay
bundle = manager.get_bundle(
    region="chesapeake_bay",
    start_date="2023-01-01",
    end_date="2023-01-31",
)

# Access individual components
mesh = bundle.mesh                    # Unstructured triangular mesh
bathymetry = bundle.bathymetry        # Elevation grid
tidal_forcing = bundle.tidal_forcing  # Harmonic constituents
validation = bundle.validation        # Observation datasets

# Convert to PyTorch Geometric format for GNNs
pyg_data = mesh.to_pyg_data()
```

## Supported Projects

| Project | Description | Primary Data |
|---------|-------------|--------------|
| 01 - GeoPINN-Manifold | Coastal manifold geometry | Shoreline, bathymetry |
| 02 - WavePINN-NIF-ComplexMedia | Ocean acoustics | Sound speed profiles (WOA) |
| 06 - CoastFlow-GNN | Coastal flow prediction | Bathymetry, tides, waves, mesh |
| 16 - SurfPINN | Surface dynamics | Bathymetry, waves |
| 17 - WavePINN-NIF-Scalar | Wave transformation | Wave buoys, bathymetry |

## Available Regions

```python
from data.coastal import REGIONS

# Predefined regions (BoundingBox objects):
# - chesapeake_bay      - Full Chesapeake Bay
# - chesapeake_mouth    - Bay mouth / CBBT area
# - outer_banks_nc      - North Carolina Outer Banks
# - galveston_bay_tx    - Galveston Bay, Texas
# - sf_bay_ca           - San Francisco Bay
# - puget_sound_wa      - Puget Sound, Washington
# - delaware_bay        - Delaware Bay
# - mobile_bay_al       - Mobile Bay, Alabama
# - tampa_bay_fl        - Tampa Bay, Florida
# - long_island_sound   - Long Island Sound
# - test_tiny           - Small test region (fast)
```

## Data Sources

### NOAA CUDEM (Bathymetry)
- Resolution: 1/9 arc-second (~3m) to 1 arc-second (~30m)
- Coverage: US coastal waters
- Fallback: ETOPO 2022 global dataset

### NOAA CO-OPS (Tides & Currents)
- Water level observations
- Tidal predictions
- Harmonic constituents (37 constituents)
- Current measurements

### NOAA NDBC (Waves)
- Significant wave height
- Peak/dominant period
- Wave direction
- Wind speed/direction

### World Ocean Atlas (WOA)
- Temperature profiles
- Salinity profiles
- Derived sound speed (Mackenzie 1981)

### Natural Earth (Coastlines)
- Resolutions: 10m, 50m, 110m
- Land/water boundaries

## Module Structure

```
data/coastal/
├── __init__.py          # Main package exports
├── config.py            # Region definitions, BoundingBox
├── datatypes.py         # Data classes (Mesh, Grid, etc.)
├── manager.py           # CoastalDataManager orchestrator
├── downloaders/
│   ├── coops.py         # NOAA CO-OPS tide/current
│   ├── ndbc.py          # NOAA NDBC wave buoys
│   ├── cudem.py         # NOAA CUDEM bathymetry
│   ├── woa.py           # World Ocean Atlas T/S
│   └── shoreline.py     # Coastline geometry
├── processors/
│   ├── mesh_generator.py      # Triangular mesh generation
│   ├── forcing_builder.py     # Tidal/wave forcing
│   ├── validation_builder.py  # Observation matching
│   └── coordinate_utils.py    # UTM/lat-lon conversion
└── integration/
    └── project06_coastflow_gnn.py  # Example integration
```

## Requirements

### Required
- Python ≥ 3.9
- numpy
- requests

### Optional
- scipy (mesh generation, interpolation)
- xarray (OPeNDAP access for WOA/CUDEM)
- netCDF4 (alternative to xarray)
- torch, torch_geometric (PyG Data conversion)
- pyyaml (YAML config loading)
- shapely (shoreline processing)

## Individual Downloaders

```python
from data.coastal import BoundingBox
from data.coastal.downloaders import (
    CUDEMDownloader,
    COOPSDownloader, 
    NDBCDownloader,
    WOADownloader,
    ShorelineDownloader,
)

bbox = BoundingBox(lat_min=36.8, lat_max=37.3, lon_min=-76.3, lon_max=-75.9)

# Bathymetry
cudem = CUDEMDownloader(cache_dir="./cache/cudem")
bathymetry = cudem.download(bbox, resolution=1/360)  # ~3m

# Tide stations
coops = COOPSDownloader(cache_dir="./cache/coops")
stations = coops.discover_stations(bbox)
water_levels = coops.download_water_levels("8638610", "2023-01-01", "2023-01-31")

# Wave buoys
ndbc = NDBCDownloader(cache_dir="./cache/ndbc")
buoys = ndbc.discover_stations(bbox)
waves = ndbc.download_range("44014", 2022, 2023)

# Ocean profiles (T, S, sound speed)
woa = WOADownloader(cache_dir="./cache/woa")
profiles = woa.download_profiles(bbox)

# Coastline
shoreline = ShorelineDownloader(cache_dir="./cache/shore")
coast = shoreline.download(bbox, resolution="10m")
```

## Mesh Generation

```python
from data.coastal.processors import CoastalMeshGenerator, MeshParameters

params = MeshParameters(
    min_edge_length=200.0,     # 200m near shore
    max_edge_length=3000.0,    # 3km offshore
    depth_grading_factor=0.15, # Refine in shallow water
    shore_grading_factor=0.1,  # Refine near coast
)

generator = CoastalMeshGenerator(params)
mesh = generator.generate(bathymetry, shoreline)

# Convert to PyG Data for GNN training
pyg_data = mesh.to_pyg_data()
print(f"Nodes: {pyg_data.num_nodes}, Edges: {pyg_data.edge_index.shape[1]}")
```

## Forcing Fields

```python
from data.coastal.processors import ForcingBuilder

builder = ForcingBuilder()

# Tidal forcing from harmonic constituents
tidal = builder.build_tidal_forcing(
    stations=tide_stations,
    mesh=mesh,
    times=np.arange(0, 24*30, 1),  # hours
)

# Wave forcing from buoy observations  
wave = builder.build_wave_forcing(
    buoys=wave_buoys,
    time_series=wave_data,
    mesh=mesh,
    times=timestamps,
)
```

## Validation

```python
from data.coastal.processors import ValidationBuilder

builder = ValidationBuilder()
val_data = builder.build_from_tide_gauges(
    stations=tide_stations,
    water_levels=water_level_timeseries,
    mesh=mesh,
)

# Compare model predictions to observations
metrics = builder.compute_validation_metrics(
    validation=val_data,
    predictions=model_output,
    prediction_times=time_array,
)
print(f"RMSE: {metrics['rmse']:.3f} m")
print(f"Skill: {metrics['skill']:.3f}")
```

## Saving/Loading Bundles

```python
# Save bundle to disk
manager.save_bundle(bundle, "./saved_bundles/chesapeake_jan2023")

# Load later
loaded = manager.load_bundle("./saved_bundles/chesapeake_jan2023")
```

## API Rate Limits

The downloaders respect NOAA API rate limits:
- CO-OPS: 0.5 second between requests
- NDBC: 1.0 second between requests
- All: Exponential backoff on 429/503 errors

## Caching

Downloaded data is cached by default:
```python
manager = CoastalDataManager(
    cache_dir="./coastal_cache",
    use_cache=True,  # Default
)
```

To force re-download:
```python
manager = CoastalDataManager(use_cache=False)
```
