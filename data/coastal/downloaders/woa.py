"""
World Ocean Atlas (WOA) temperature, salinity, and sound speed downloader.

Downloads climatological ocean data:
- Temperature profiles
- Salinity profiles
- Computes sound speed using Mackenzie (1981) equation

Data source: https://www.ncei.noaa.gov/products/world-ocean-atlas
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import BoundingBox, RegionConfig
from ..datatypes import OceanProfile, OceanProfileGrid
from .utils import (
    NCEI_RATE_LIMITER,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# WOA23 OPeNDAP endpoints (NCEI Thredds)
WOA_THREDDS_BASE = "https://www.ncei.noaa.gov/thredds-ocean/dodsC/ncei/woa"

# File naming convention:
# woa23_decav91C0_t00_04.nc = temperature, decadal average 1991-2020, annual (00), 0.25° (04)
# Seasons: 00=annual, 13=winter, 14=spring, 15=summer, 16=fall

WOA_SEASON_CODES = {
    "annual": "00",
    "winter": "13",  # Jan-Mar
    "spring": "14",  # Apr-Jun
    "summer": "15",  # Jul-Sep
    "fall": "16",    # Oct-Dec
}

WOA_RESOLUTION_CODES = {
    "1": "01",    # 1 degree
    "0.25": "04", # 0.25 degree (quarter degree)
}

# Standard WOA depth levels (102 levels from 0 to 5500m)
WOA_STANDARD_DEPTHS = np.array([
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
    500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200,
    1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850,
    1900, 1950, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000,
    3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300,
    4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500
])


# =============================================================================
# Sound Speed Equations
# =============================================================================

def mackenzie_sound_speed(
    temperature_c: np.ndarray,
    salinity_psu: np.ndarray,
    depth_m: np.ndarray,
) -> np.ndarray:
    """
    Calculate sound speed using Mackenzie (1981) equation.
    
    c = 1448.96 + 4.591*T - 5.304e-2*T² + 2.374e-4*T³
      + 1.340*(S - 35) + 1.630e-2*D + 1.675e-7*D²
      - 1.025e-2*T*(S - 35) - 7.139e-13*T*D³
    
    Valid ranges:
        Temperature: 2 < T < 30°C
        Salinity: 25 < S < 40 PSU
        Depth: 0 < D < 8000m
    
    Accuracy: ±0.07 m/s
    
    Args:
        temperature_c: Temperature in degrees Celsius
        salinity_psu: Salinity in PSU (Practical Salinity Units)
        depth_m: Depth in meters
        
    Returns:
        Sound speed in m/s
    """
    T = np.asarray(temperature_c)
    S = np.asarray(salinity_psu)
    D = np.asarray(depth_m)
    
    c = (1448.96 
         + 4.591 * T 
         - 5.304e-2 * T**2 
         + 2.374e-4 * T**3
         + 1.340 * (S - 35) 
         + 1.630e-2 * D 
         + 1.675e-7 * D**2
         - 1.025e-2 * T * (S - 35) 
         - 7.139e-13 * T * D**3)
    
    return c


def seawater_density(
    temperature_c: np.ndarray,
    salinity_psu: np.ndarray,
    depth_m: np.ndarray,
) -> np.ndarray:
    """
    Calculate seawater density using simplified UNESCO equation.
    
    Args:
        temperature_c: Temperature in degrees Celsius
        salinity_psu: Salinity in PSU
        depth_m: Depth in meters
        
    Returns:
        Density in kg/m³
    """
    T = np.asarray(temperature_c)
    S = np.asarray(salinity_psu)
    
    # Simplified equation (ignoring pressure effects for shallow water)
    # Full UNESCO EOS-80 would be more accurate but more complex
    rho = (999.842594 
           + 6.793952e-2 * T 
           - 9.095290e-3 * T**2
           + 1.001685e-4 * T**3 
           - 1.120083e-6 * T**4
           + 6.536336e-9 * T**5
           + S * (0.824493 
                  - 4.0899e-3 * T 
                  + 7.6438e-5 * T**2
                  - 8.2467e-7 * T**3 
                  + 5.3875e-9 * T**4)
           + S**1.5 * (-5.72466e-3 
                       + 1.0227e-4 * T 
                       - 1.6546e-6 * T**2)
           + 4.8314e-4 * S**2)
    
    return rho


# =============================================================================
# Downloader Class
# =============================================================================

class WOADownloader:
    """
    Downloads World Ocean Atlas temperature and salinity data.
    
    Computes derived sound speed profiles using the Mackenzie equation.
    
    Example:
        downloader = WOADownloader(cache_dir="data/coastal/cache/ocean")
        
        # Download profiles for a region
        profiles = downloader.download_profiles(bounds, resolution="0.25")
        
        # Extract a single profile
        profile = profiles.extract_profile(lat=37.0, lon=-76.0)
    """
    
    def __init__(self, cache_dir: str | Path = "data/coastal/cache/ocean"):
        """
        Initialize the WOA downloader.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._xarray_available = None
    
    def _check_xarray(self) -> bool:
        """Check if xarray is available."""
        if self._xarray_available is None:
            try:
                import xarray as xr
                self._xarray_available = True
            except ImportError:
                logger.warning("xarray not available for WOA download")
                self._xarray_available = False
        return self._xarray_available
    
    # -------------------------------------------------------------------------
    # Main Download Methods
    # -------------------------------------------------------------------------
    
    def download_profiles(
        self,
        config: RegionConfig,
        force: bool = False,
    ) -> OceanProfileGrid:
        """
        Download temperature and salinity profiles for a region.
        
        Args:
            config: Region configuration
            force: Force re-download
            
        Returns:
            OceanProfileGrid with temperature, salinity, and computed sound speed
        """
        bounds = config.bounds
        resolution = config.woa_resolution
        season = config.woa_season
        
        # Cache key
        cache_key = f"woa_{bounds.lat_min:.1f}_{bounds.lat_max:.1f}_{bounds.lon_min:.1f}_{bounds.lon_max:.1f}_{resolution}_{season}"
        cache_path = self.cache_dir / f"{cache_key}.npz"
        
        # Check cache
        if not force and cache_path.exists():
            logger.info(f"Loading cached WOA data from {cache_path}")
            return self._load_cache(cache_path)
        
        logger.info(f"Downloading WOA {season} profiles at {resolution}° resolution")
        
        if self._check_xarray():
            grid = self._download_opendap(bounds, resolution, season)
        else:
            grid = self._generate_synthetic_profiles(bounds, season)
        
        if grid is None:
            grid = self._generate_synthetic_profiles(bounds, season)
        
        # Cache
        self._save_cache(cache_path, grid)
        
        return grid
    
    def _download_opendap(
        self,
        bounds: BoundingBox,
        resolution: str,
        season: str,
    ) -> Optional[OceanProfileGrid]:
        """
        Download WOA data via OPeNDAP.
        
        Args:
            bounds: Geographic bounding box
            resolution: "1" or "0.25" degrees
            season: "annual", "winter", "spring", "summer", "fall"
            
        Returns:
            OceanProfileGrid or None if download fails
        """
        import xarray as xr
        
        season_code = WOA_SEASON_CODES.get(season, "00")
        res_code = WOA_RESOLUTION_CODES.get(resolution, "04")
        
        # Build URLs for temperature and salinity
        temp_url = f"{WOA_THREDDS_BASE}/temperature/decav91C0/{resolution}/woa23_decav91C0_t{season_code}_{res_code}.nc"
        sal_url = f"{WOA_THREDDS_BASE}/salinity/decav91C0/{resolution}/woa23_decav91C0_s{season_code}_{res_code}.nc"
        
        try:
            logger.info(f"Opening WOA temperature: {temp_url}")
            ds_temp = xr.open_dataset(temp_url, engine='netcdf4')
            
            logger.info(f"Opening WOA salinity: {sal_url}")
            ds_sal = xr.open_dataset(sal_url, engine='netcdf4')
            
            # Find coordinate names
            lat_name = 'lat'
            lon_name = 'lon'
            depth_name = 'depth'
            
            for name in ds_temp.coords:
                name_lower = name.lower()
                if 'lat' in name_lower:
                    lat_name = name
                elif 'lon' in name_lower:
                    lon_name = name
                elif 'depth' in name_lower:
                    depth_name = name
            
            # Find variable names
            temp_var = None
            sal_var = None
            
            for name in ds_temp.data_vars:
                if 't_an' in name.lower() or 'temperature' in name.lower():
                    temp_var = name
                    break
            
            for name in ds_sal.data_vars:
                if 's_an' in name.lower() or 'salinity' in name.lower():
                    sal_var = name
                    break
            
            if temp_var is None or sal_var is None:
                logger.warning(f"Could not find temperature/salinity variables")
                ds_temp.close()
                ds_sal.close()
                return None
            
            # Subset to region
            pad = 0.5  # Half degree padding
            lat_slice = slice(bounds.lat_min - pad, bounds.lat_max + pad)
            lon_slice = slice(bounds.lon_min - pad, bounds.lon_max + pad)
            
            logger.info(f"Subsetting WOA data to {lat_slice}, {lon_slice}")
            
            temp_subset = ds_temp.sel({lat_name: lat_slice, lon_name: lon_slice})
            sal_subset = ds_sal.sel({lat_name: lat_slice, lon_name: lon_slice})
            
            # Load data
            logger.info("Loading temperature data...")
            temperature = temp_subset[temp_var].values  # (depth, lat, lon)
            latitudes = temp_subset[lat_name].values
            longitudes = temp_subset[lon_name].values
            depths = temp_subset[depth_name].values
            
            logger.info("Loading salinity data...")
            salinity = sal_subset[sal_var].values  # (depth, lat, lon)
            
            ds_temp.close()
            ds_sal.close()
            
            # Transpose to (lat, lon, depth) for consistency
            temperature = np.transpose(temperature, (1, 2, 0))
            salinity = np.transpose(salinity, (1, 2, 0))
            
            # Handle fill values
            temperature = np.where(temperature > 50, np.nan, temperature)
            salinity = np.where(salinity > 50, np.nan, salinity)
            
            # Compute sound speed
            logger.info("Computing sound speed profiles...")
            depth_3d = np.broadcast_to(
                depths[np.newaxis, np.newaxis, :],
                temperature.shape
            )
            sound_speed = mackenzie_sound_speed(temperature, salinity, depth_3d)
            
            return OceanProfileGrid(
                latitudes=latitudes,
                longitudes=longitudes,
                depths_m=depths,
                temperature=temperature,
                salinity=salinity,
                sound_speed=sound_speed,
                season=season,
            )
            
        except Exception as e:
            logger.warning(f"Failed to download WOA data: {e}")
            return None
    
    def _generate_synthetic_profiles(
        self,
        bounds: BoundingBox,
        season: str,
    ) -> OceanProfileGrid:
        """
        Generate synthetic ocean profiles for testing.
        
        Creates idealized profiles with:
        - Surface mixed layer
        - Thermocline
        - Deep isothermal layer
        """
        logger.warning("Generating synthetic WOA profiles - no real data available")
        
        # Grid
        resolution_deg = 0.25
        lats = np.arange(bounds.lat_min, bounds.lat_max + resolution_deg, resolution_deg)
        lons = np.arange(bounds.lon_min, bounds.lon_max + resolution_deg, resolution_deg)
        
        # Use subset of standard depths (0-1000m for coastal)
        depths = WOA_STANDARD_DEPTHS[WOA_STANDARD_DEPTHS <= 1000]
        
        n_lat = len(lats)
        n_lon = len(lons)
        n_depth = len(depths)
        
        # Seasonal surface temperature adjustments
        seasonal_adjustment = {
            "annual": 0,
            "winter": -5,
            "spring": 0,
            "summer": 5,
            "fall": 0,
        }.get(season, 0)
        
        # Generate temperature profile (typical mid-latitude structure)
        temperature = np.zeros((n_lat, n_lon, n_depth))
        salinity = np.zeros((n_lat, n_lon, n_depth))
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # Surface temperature varies with latitude
                T_surface = 25 - 0.5 * abs(lat - 25) + seasonal_adjustment
                T_deep = 4  # Deep water temperature
                
                # Thermocline depth and strength
                thermocline_depth = 100 + 20 * np.sin(np.radians(lat))
                
                # Temperature profile
                for k, d in enumerate(depths):
                    if d < 50:
                        # Mixed layer
                        temperature[i, j, k] = T_surface
                    elif d < thermocline_depth + 200:
                        # Thermocline
                        frac = (d - 50) / (thermocline_depth + 150)
                        temperature[i, j, k] = T_surface - frac * (T_surface - T_deep)
                    else:
                        # Deep layer
                        temperature[i, j, k] = T_deep
                
                # Salinity profile (less variation)
                S_surface = 35 + 0.1 * np.sin(np.radians(lon))
                salinity[i, j, :] = S_surface - 0.5 * (1 - np.exp(-depths / 500))
        
        # Compute sound speed
        depth_3d = np.broadcast_to(
            depths[np.newaxis, np.newaxis, :],
            temperature.shape
        )
        sound_speed = mackenzie_sound_speed(temperature, salinity, depth_3d)
        
        return OceanProfileGrid(
            latitudes=lats,
            longitudes=lons,
            depths_m=depths,
            temperature=temperature,
            salinity=salinity,
            sound_speed=sound_speed,
            season=season,
        )
    
    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------
    
    def _save_cache(self, path: Path, grid: OceanProfileGrid) -> None:
        """Save profile grid to cache."""
        np.savez_compressed(
            path,
            latitudes=grid.latitudes,
            longitudes=grid.longitudes,
            depths_m=grid.depths_m,
            temperature=grid.temperature,
            salinity=grid.salinity,
            sound_speed=grid.sound_speed,
            season=grid.season,
        )
        logger.info(f"Cached WOA data to {path}")
    
    def _load_cache(self, path: Path) -> OceanProfileGrid:
        """Load profile grid from cache."""
        data = np.load(path)
        return OceanProfileGrid(
            latitudes=data['latitudes'],
            longitudes=data['longitudes'],
            depths_m=data['depths_m'],
            temperature=data['temperature'],
            salinity=data['salinity'],
            sound_speed=data['sound_speed'],
            season=str(data['season']),
        )
    
    # -------------------------------------------------------------------------
    # Profile Extraction
    # -------------------------------------------------------------------------
    
    def extract_profile_at_location(
        self,
        grid: OceanProfileGrid,
        lat: float,
        lon: float,
    ) -> OceanProfile:
        """
        Extract a single vertical profile from a grid.
        
        Args:
            grid: OceanProfileGrid to extract from
            lat: Latitude of desired profile
            lon: Longitude of desired profile
            
        Returns:
            OceanProfile at the specified location
        """
        return grid.extract_profile(lat, lon)
    
    def compute_sound_speed_transect(
        self,
        grid: OceanProfileGrid,
        start: tuple[float, float],
        end: tuple[float, float],
        n_points: int = 50,
    ) -> dict:
        """
        Extract a 2D sound speed transect (range-depth slice).
        
        Args:
            grid: OceanProfileGrid
            start: (lat, lon) start point
            end: (lat, lon) end point
            n_points: Number of points along transect
            
        Returns:
            Dict with ranges, depths, and sound_speed arrays
        """
        # Interpolate along transect
        lats = np.linspace(start[0], end[0], n_points)
        lons = np.linspace(start[1], end[1], n_points)
        
        # Compute range from start
        from ..config import great_circle_distance
        ranges_km = np.array([
            great_circle_distance(start[0], start[1], lat, lon)
            for lat, lon in zip(lats, lons)
        ])
        
        # Extract profiles along transect
        profiles = [grid.extract_profile(lat, lon) for lat, lon in zip(lats, lons)]
        
        # Build 2D array
        depths = profiles[0].depths_m
        sound_speed_2d = np.array([p.sound_speed_m_s for p in profiles]).T  # (depth, range)
        
        return {
            'ranges_km': ranges_km,
            'ranges_m': ranges_km * 1000,
            'depths_m': depths,
            'sound_speed': sound_speed_2d,
            'transect_lats': lats,
            'transect_lons': lons,
        }
