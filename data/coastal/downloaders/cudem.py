"""
NOAA CUDEM (Continuously Updated Digital Elevation Model) bathymetry downloader.

Downloads coastal bathymetry/topography data:
- CUDEM tiles via OPeNDAP (preferred - server-side subsetting)
- ETOPO 2022 as fallback (coarser but always available)

Data sources:
- CUDEM: https://www.ncei.noaa.gov/products/coastal-relief-model
- ETOPO: https://www.ngdc.noaa.gov/mgg/global/global.html
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..config import BoundingBox, RegionConfig
from ..datatypes import BathymetryGrid
from .utils import (
    download_with_retry,
    download_file,
    NCEI_RATE_LIMITER,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# ETOPO 2022 OPeNDAP endpoint (1 arc-minute global, reliable fallback)
ETOPO_OPENDAP_URL = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc"

# CUDEM regional OPeNDAP endpoints (organized by region)
# These are 1/9 arc-second (~3m) high-resolution coastal DEMs
CUDEM_DATASETS = {
    # US East Coast
    "chesapeake_bay": {
        "url": "https://www.ngdc.noaa.gov/thredds/dodsC/regional/chesapeake_bay_dem.nc",
        "bounds": BoundingBox(36.5, 40.0, -77.5, -75.0),
        "resolution_arcsec": 1/9,
    },
    "delmarva": {
        "url": "https://www.ngdc.noaa.gov/thredds/dodsC/regional/delmarva_dem.nc",
        "bounds": BoundingBox(36.5, 39.5, -76.5, -74.5),
        "resolution_arcsec": 1/9,
    },
    "north_carolina": {
        "url": "https://www.ngdc.noaa.gov/thredds/dodsC/regional/north_carolina_dem.nc",
        "bounds": BoundingBox(33.5, 36.5, -78.0, -75.0),
        "resolution_arcsec": 1/9,
    },
    
    # US Gulf Coast
    "gulf_coast": {
        "url": "https://www.ngdc.noaa.gov/thredds/dodsC/regional/gulf_coast_dem.nc",
        "bounds": BoundingBox(24.0, 31.0, -98.0, -80.0),
        "resolution_arcsec": 1/3,
    },
    "galveston": {
        "url": "https://www.ngdc.noaa.gov/thredds/dodsC/regional/galveston_dem.nc",
        "bounds": BoundingBox(28.5, 30.5, -96.0, -94.0),
        "resolution_arcsec": 1/9,
    },
    
    # US West Coast
    "southern_california": {
        "url": "https://www.ngdc.noaa.gov/thredds/dodsC/regional/southern_california_dem.nc",
        "bounds": BoundingBox(32.0, 35.0, -121.0, -117.0),
        "resolution_arcsec": 1/3,
    },
    "sf_bay": {
        "url": "https://www.ngdc.noaa.gov/thredds/dodsC/regional/san_francisco_bay_dem.nc",
        "bounds": BoundingBox(37.0, 38.5, -123.0, -121.5),
        "resolution_arcsec": 1/9,
    },
    "puget_sound": {
        "url": "https://www.ngdc.noaa.gov/thredds/dodsC/regional/puget_sound_dem.nc",
        "bounds": BoundingBox(46.5, 49.0, -124.0, -121.5),
        "resolution_arcsec": 1/3,
    },
}

# Alternative: NCEI 1/3 arc-second CUDEM tiles
# These have broader coverage but are organized as tiles
CUDEM_NCEI_BASE = "https://www.ngdc.noaa.gov/thredds/dodsC/regional/"


# =============================================================================
# Downloader Class
# =============================================================================

class CUDEMDownloader:
    """
    Downloads NOAA bathymetry/topography data.
    
    Strategy:
    1. Try to find a matching CUDEM regional dataset via OPeNDAP
    2. If available, use server-side subsetting to download only the needed area
    3. Fall back to ETOPO 2022 (coarser, 1 arc-minute, but always available)
    
    Example:
        downloader = CUDEMDownloader(cache_dir="data/coastal/cache/bathymetry")
        
        # Download bathymetry
        bathy = downloader.download(bounds, resolution_arcsec=1/3)
    """
    
    def __init__(self, cache_dir: str | Path = "data/coastal/cache/bathymetry"):
        """
        Initialize the CUDEM downloader.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._xarray_available = None
    
    def _check_xarray(self) -> bool:
        """Check if xarray is available for OPeNDAP access."""
        if self._xarray_available is None:
            try:
                import xarray as xr
                self._xarray_available = True
            except ImportError:
                logger.warning("xarray not available - OPeNDAP subsetting disabled")
                self._xarray_available = False
        return self._xarray_available
    
    # -------------------------------------------------------------------------
    # Main Download Method
    # -------------------------------------------------------------------------
    
    def download(
        self,
        config: RegionConfig,
        force: bool = False,
    ) -> BathymetryGrid:
        """
        Download bathymetry for a region.
        
        Args:
            config: Region configuration
            force: Force re-download even if cached
            
        Returns:
            BathymetryGrid with elevation data
        """
        bounds = config.bounds
        resolution = config.bathy_resolution_arcsec
        datum = config.bathy_datum
        
        # Generate cache filename
        cache_key = f"bathy_{bounds.lat_min:.2f}_{bounds.lat_max:.2f}_{bounds.lon_min:.2f}_{bounds.lon_max:.2f}_{resolution:.4f}"
        cache_path = self.cache_dir / f"{cache_key}.npz"
        
        # Check cache
        if not force and cache_path.exists():
            logger.info(f"Loading cached bathymetry from {cache_path}")
            return self._load_cache(cache_path)
        
        logger.info(f"Downloading bathymetry for {bounds}")
        
        # Try CUDEM first (higher resolution)
        bathy = None
        if self._check_xarray():
            bathy = self._download_cudem(bounds, resolution)
        
        # Fall back to ETOPO if CUDEM fails
        if bathy is None:
            logger.info("Falling back to ETOPO 2022")
            bathy = self._download_etopo(bounds)
        
        if bathy is None:
            raise RuntimeError("Failed to download bathymetry from any source")
        
        # Update metadata
        bathy.datum = datum
        
        # Cache
        self._save_cache(cache_path, bathy)
        
        return bathy
    
    # -------------------------------------------------------------------------
    # CUDEM Download (OPeNDAP)
    # -------------------------------------------------------------------------
    
    def _find_cudem_dataset(self, bounds: BoundingBox) -> Optional[dict]:
        """Find a CUDEM dataset that covers the requested bounds."""
        for name, dataset in CUDEM_DATASETS.items():
            if dataset["bounds"].intersects(bounds):
                logger.debug(f"Found matching CUDEM dataset: {name}")
                return dataset
        return None
    
    def _download_cudem(
        self,
        bounds: BoundingBox,
        target_resolution: float = 1/3,
    ) -> Optional[BathymetryGrid]:
        """
        Download CUDEM data via OPeNDAP with server-side subsetting.
        
        Args:
            bounds: Bounding box to download
            target_resolution: Desired resolution in arc-seconds
            
        Returns:
            BathymetryGrid or None if download fails
        """
        import xarray as xr
        
        dataset_info = self._find_cudem_dataset(bounds)
        if dataset_info is None:
            logger.warning("No CUDEM dataset found for region")
            return None
        
        url = dataset_info["url"]
        native_resolution = dataset_info["resolution_arcsec"]
        
        try:
            logger.info(f"Opening CUDEM dataset: {url}")
            
            # Open with OPeNDAP - xarray handles server-side subsetting
            ds = xr.open_dataset(url, engine='netcdf4')
            
            # Find coordinate names (varies between datasets)
            lat_name = None
            lon_name = None
            elev_name = None
            
            for name in ds.coords:
                name_lower = name.lower()
                if 'lat' in name_lower:
                    lat_name = name
                elif 'lon' in name_lower:
                    lon_name = name
            
            for name in ds.data_vars:
                name_lower = name.lower()
                if any(x in name_lower for x in ['elev', 'band', 'z', 'height', 'topo', 'bathy']):
                    elev_name = name
                    break
            
            if lat_name is None or lon_name is None or elev_name is None:
                logger.warning(f"Could not identify coordinates in dataset: {list(ds.coords)}, {list(ds.data_vars)}")
                ds.close()
                return None
            
            # Subset to bounding box with small padding
            pad = 0.01  # ~1km padding
            lat_slice = slice(bounds.lat_min - pad, bounds.lat_max + pad)
            lon_slice = slice(bounds.lon_min - pad, bounds.lon_max + pad)
            
            logger.info(f"Subsetting to lat={lat_slice}, lon={lon_slice}")
            
            # Select the subset
            subset = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
            
            # Load data (this triggers the actual download)
            logger.info("Downloading subset...")
            elevation = subset[elev_name].values
            latitudes = subset[lat_name].values
            longitudes = subset[lon_name].values
            
            ds.close()
            
            # Subsample if resolution is coarser than native
            if target_resolution > native_resolution * 1.5:
                step = int(target_resolution / native_resolution)
                elevation = elevation[::step, ::step]
                latitudes = latitudes[::step]
                longitudes = longitudes[::step]
                logger.info(f"Subsampled by factor {step}")
            
            return BathymetryGrid(
                latitude=latitudes,
                longitude=longitudes,
                elevation=elevation,
                resolution_arcsec=target_resolution,
                datum="MSL",  # Most CUDEM data is relative to MSL
                source="CUDEM",
            )
            
        except Exception as e:
            logger.warning(f"Failed to download CUDEM: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # ETOPO Download (Fallback)
    # -------------------------------------------------------------------------
    
    def _download_etopo(self, bounds: BoundingBox) -> Optional[BathymetryGrid]:
        """
        Download ETOPO 2022 data via OPeNDAP.
        
        ETOPO is 1 arc-minute resolution (~1.8km) but covers the entire globe.
        
        Args:
            bounds: Bounding box to download
            
        Returns:
            BathymetryGrid or None if download fails
        """
        if not self._check_xarray():
            return self._download_etopo_fallback(bounds)
        
        import xarray as xr
        
        try:
            logger.info(f"Opening ETOPO 2022 dataset")
            
            ds = xr.open_dataset(ETOPO_OPENDAP_URL, engine='netcdf4')
            
            # ETOPO coordinates
            lat_name = 'lat'
            lon_name = 'lon'
            elev_name = 'z'
            
            # Check coordinate names
            if lat_name not in ds.coords:
                for name in ds.coords:
                    if 'lat' in name.lower():
                        lat_name = name
                    elif 'lon' in name.lower():
                        lon_name = name
            
            if elev_name not in ds.data_vars:
                for name in ds.data_vars:
                    if any(x in name.lower() for x in ['z', 'elev', 'band']):
                        elev_name = name
                        break
            
            # Subset with padding
            pad = 0.1
            lat_slice = slice(bounds.lat_min - pad, bounds.lat_max + pad)
            lon_slice = slice(bounds.lon_min - pad, bounds.lon_max + pad)
            
            logger.info(f"Subsetting ETOPO to lat={lat_slice}, lon={lon_slice}")
            
            subset = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
            
            logger.info("Downloading ETOPO subset...")
            elevation = subset[elev_name].values
            latitudes = subset[lat_name].values
            longitudes = subset[lon_name].values
            
            ds.close()
            
            return BathymetryGrid(
                latitude=latitudes,
                longitude=longitudes,
                elevation=elevation,
                resolution_arcsec=60.0,  # 1 arc-minute
                datum="MSL",
                source="ETOPO2022",
            )
            
        except Exception as e:
            logger.warning(f"Failed to download ETOPO via OPeNDAP: {e}")
            return self._download_etopo_fallback(bounds)
    
    def _download_etopo_fallback(self, bounds: BoundingBox) -> Optional[BathymetryGrid]:
        """
        Generate synthetic bathymetry as final fallback.
        
        This is only used when all real data sources fail.
        Returns a simple sloped bathymetry for testing purposes.
        """
        logger.warning("Using synthetic bathymetry fallback - no real data available")
        
        # Generate a grid
        resolution_deg = 1/60  # 1 arc-minute
        lats = np.arange(bounds.lat_min, bounds.lat_max, resolution_deg)
        lons = np.arange(bounds.lon_min, bounds.lon_max, resolution_deg)
        
        # Simple bathymetry: depth increases offshore (westward for east coast)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Assume east coast: depth increases as longitude decreases
        center_lon = (bounds.lon_min + bounds.lon_max) / 2
        depth = -100 * (center_lon - lon_grid) / (center_lon - bounds.lon_min + 1e-6)
        depth = np.clip(depth, -500, 100)  # Reasonable range
        
        return BathymetryGrid(
            latitude=lats,
            longitude=lons,
            elevation=depth,
            resolution_arcsec=60.0,
            datum="MSL",
            source="SYNTHETIC",
        )
    
    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------
    
    def _save_cache(self, path: Path, bathy: BathymetryGrid) -> None:
        """Save bathymetry to cache."""
        np.savez_compressed(
            path,
            latitude=bathy.latitude,
            longitude=bathy.longitude,
            elevation=bathy.elevation,
            resolution_arcsec=bathy.resolution_arcsec,
            datum=bathy.datum,
            source=bathy.source,
        )
        logger.info(f"Cached bathymetry to {path}")
    
    def _load_cache(self, path: Path) -> BathymetryGrid:
        """Load bathymetry from cache."""
        data = np.load(path, allow_pickle=True)
        return BathymetryGrid(
            latitude=data['latitude'],
            longitude=data['longitude'],
            elevation=data['elevation'],
            resolution_arcsec=float(data['resolution_arcsec']),
            datum=str(data['datum']),
            source=str(data['source']),
        )
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def list_available_datasets(self) -> dict[str, dict]:
        """Return information about available CUDEM datasets."""
        return {
            name: {
                "bounds": (
                    info["bounds"].lat_min,
                    info["bounds"].lat_max,
                    info["bounds"].lon_min,
                    info["bounds"].lon_max,
                ),
                "resolution_arcsec": info["resolution_arcsec"],
            }
            for name, info in CUDEM_DATASETS.items()
        }
    
    def check_coverage(self, bounds: BoundingBox) -> dict[str, bool]:
        """Check which datasets cover a given bounding box."""
        return {
            name: info["bounds"].intersects(bounds)
            for name, info in CUDEM_DATASETS.items()
        }
