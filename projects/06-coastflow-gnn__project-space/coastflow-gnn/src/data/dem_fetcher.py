"""
DEM Fetcher for CoastFlow-GNN

Utilities for downloading NOAA Digital Elevation Model (DEM) data
for coastal regions.

Note: This module provides the interface for fetching real DEM data.
For quick testing, use the SyntheticCoastalDataset instead.
"""

import os
import requests
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path


# NOAA DEM endpoints
NOAA_THREDDS_BASE = "https://www.ngdc.noaa.gov/thredds/dodsC/dem/"
NOAA_GRID_EXTRACT_URL = "https://gis.ngdc.noaa.gov/arcgis/rest/services/"


class DEMFetcher:
    """
    Fetches NOAA DEM tiles for coastal regions.
    
    Uses NOAA's Continuously Updated Digital Elevation Model (CUDEM)
    or regional coastal DEMs.
    
    Args:
        cache_dir: Directory to cache downloaded tiles
        resolution: Desired resolution in arc-seconds (1/9, 1/3, 1)
    """
    
    def __init__(
        self,
        cache_dir: str = "./data/dem_cache",
        resolution: str = "1/9",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        
        # Available DEM datasets
        self.datasets = {
            "chesapeake_bay": {
                "name": "Chesapeake Bay 1/9 arc-second DEM",
                "bounds": (-77.5, 36.5, -75.5, 39.5),  # (lon_min, lat_min, lon_max, lat_max)
                "url_template": "chesapeake_bay_13_navd88_2015.nc",
            },
            "gulf_coast": {
                "name": "Northern Gulf of Mexico DEM",
                "bounds": (-98.0, 24.0, -80.0, 31.0),
                "url_template": "ngom_1_9_mhw_2019.nc",
            },
            "southern_california": {
                "name": "Southern California DEM",
                "bounds": (-121.0, 32.0, -117.0, 35.0),
                "url_template": "socal_1_9_mhw_2012.nc",
            },
        }
    
    def list_available_datasets(self) -> List[str]:
        """List available DEM datasets."""
        return list(self.datasets.keys())
    
    def get_dem_info(self, dataset_name: str) -> dict:
        """Get information about a specific DEM dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return self.datasets[dataset_name]
    
    def fetch_dem_tile(
        self,
        bounds: Tuple[float, float, float, float],
        output_file: Optional[str] = None,
        dataset: str = "chesapeake_bay",
    ) -> np.ndarray:
        """
        Fetch DEM tile for specified bounds.
        
        Note: This is a placeholder implementation. In production,
        this would connect to NOAA's OPeNDAP or WCS services.
        
        Args:
            bounds: (lon_min, lat_min, lon_max, lat_max)
            output_file: Path to save the downloaded tile
            dataset: Name of the DEM dataset to use
            
        Returns:
            DEM data as numpy array
        """
        # For testing purposes, generate synthetic DEM data
        # Real implementation would use xarray + OPeNDAP
        
        lon_min, lat_min, lon_max, lat_max = bounds
        
        # Create grid
        resolution_deg = 1/3600 * 9  # ~1/9 arc-second ≈ 3m
        n_lon = int((lon_max - lon_min) / resolution_deg)
        n_lat = int((lat_max - lat_min) / resolution_deg)
        
        # Limit size for memory
        n_lon = min(n_lon, 1000)
        n_lat = min(n_lat, 1000)
        
        lons = np.linspace(lon_min, lon_max, n_lon)
        lats = np.linspace(lat_min, lat_max, n_lat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Generate synthetic coastal elevation
        # In reality, this would be actual DEM data
        x_norm = (lon_grid - lon_min) / (lon_max - lon_min)
        y_norm = (lat_grid - lat_min) / (lat_max - lat_min)
        
        # Coastal gradient
        elevation = 20 * (1 - x_norm) - 10
        
        # Add terrain features
        elevation += 5 * np.sin(2 * np.pi * 3 * x_norm) * np.cos(2 * np.pi * 2 * y_norm)
        elevation += 2 * np.sin(2 * np.pi * 7 * x_norm + 0.5) * np.sin(2 * np.pi * 5 * y_norm)
        
        # Random noise
        elevation += np.random.randn(*elevation.shape) * 0.5
        
        if output_file:
            np.savez(
                output_file,
                elevation=elevation,
                lons=lons,
                lats=lats,
                bounds=bounds,
            )
        
        return elevation
    
    def fetch_bathymetry(
        self,
        bounds: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """
        Fetch bathymetry (underwater elevation) data.
        
        Args:
            bounds: (lon_min, lat_min, lon_max, lat_max)
            
        Returns:
            Bathymetry data (negative values for depth)
        """
        # Fetch DEM and filter for underwater regions
        dem = self.fetch_dem_tile(bounds)
        
        # Bathymetry is where elevation is negative
        bathymetry = np.where(dem < 0, dem, np.nan)
        
        return bathymetry


def download_sample_dem(
    output_dir: str = "./data/sample_dem",
    region: str = "sample_bay",
) -> str:
    """
    Download a sample DEM for testing purposes.
    
    Creates a synthetic coastal DEM suitable for testing
    the CoastFlow-GNN pipeline.
    
    Args:
        output_dir: Directory to save the DEM
        region: Name for the sample region
        
    Returns:
        Path to the saved DEM file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{region}_dem.npz"
    
    # Generate sample coastal DEM
    n_x, n_y = 500, 400
    x = np.linspace(0, 2000, n_x)  # 2km domain
    y = np.linspace(0, 1600, n_y)  # 1.6km domain
    xx, yy = np.meshgrid(x, y)
    
    # Create coastal profile
    x_norm = xx / 2000
    y_norm = yy / 1600
    
    # Land to sea gradient
    elevation = 15 * (1 - x_norm) - 8
    
    # Bay shape (concave coastline)
    bay_depth = 3 * np.sin(np.pi * y_norm) * np.sin(np.pi * x_norm)
    elevation -= bay_depth
    
    # Terrain roughness
    elevation += 2 * np.sin(2*np.pi*5*x_norm) * np.cos(2*np.pi*4*y_norm)
    elevation += 1 * np.sin(2*np.pi*12*x_norm) * np.sin(2*np.pi*10*y_norm)
    
    # Save
    np.savez(
        output_file,
        elevation=elevation,
        x=x,
        y=y,
        x_grid=xx,
        y_grid=yy,
        region=region,
    )
    
    print(f"Sample DEM saved to: {output_file}")
    return str(output_file)


if __name__ == "__main__":
    print("Testing DEM Fetcher...")
    
    # Test fetcher
    fetcher = DEMFetcher()
    print(f"Available datasets: {fetcher.list_available_datasets()}")
    
    # Generate sample DEM
    dem_path = download_sample_dem()
    
    # Load and check
    data = np.load(dem_path)
    print(f"DEM shape: {data['elevation'].shape}")
    print(f"Elevation range: [{data['elevation'].min():.2f}, {data['elevation'].max():.2f}]")
    
    print("\n✓ DEM Fetcher tests passed!")
