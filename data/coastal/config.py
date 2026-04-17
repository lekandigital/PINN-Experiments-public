"""
Geographic region configuration for coastal data infrastructure.

Defines bounding boxes, region presets, and download configuration
for NOAA data sources (CUDEM, CO-OPS, NDBC, WOA, shoreline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# yaml is optional - only needed for loading region configs from files
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class BoundingBox:
    """
    Geographic bounding box in WGS84 coordinates (EPSG:4326).
    
    Attributes:
        lat_min: Southern boundary (degrees north, -90 to 90)
        lat_max: Northern boundary (degrees north, -90 to 90)
        lon_min: Western boundary (degrees east, -180 to 180)
        lon_max: Eastern boundary (degrees east, -180 to 180)
    """
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    
    def __post_init__(self):
        """Validate bounds."""
        if not (-90 <= self.lat_min <= 90):
            raise ValueError(f"lat_min must be in [-90, 90], got {self.lat_min}")
        if not (-90 <= self.lat_max <= 90):
            raise ValueError(f"lat_max must be in [-90, 90], got {self.lat_max}")
        if not (-180 <= self.lon_min <= 180):
            raise ValueError(f"lon_min must be in [-180, 180], got {self.lon_min}")
        if not (-180 <= self.lon_max <= 180):
            raise ValueError(f"lon_max must be in [-180, 180], got {self.lon_max}")
        if self.lat_min >= self.lat_max:
            raise ValueError(f"lat_min ({self.lat_min}) must be less than lat_max ({self.lat_max})")
        if self.lon_min >= self.lon_max:
            raise ValueError(f"lon_min ({self.lon_min}) must be less than lon_max ({self.lon_max})")
    
    def contains(self, lat: float, lon: float) -> bool:
        """Check if a point is within the bounding box."""
        return (self.lat_min <= lat <= self.lat_max and 
                self.lon_min <= lon <= self.lon_max)
    
    def area_sq_km(self) -> float:
        """
        Approximate area in square kilometers.
        
        Uses spherical Earth approximation with latitude-dependent longitude scaling.
        """
        EARTH_RADIUS_KM = 6371.0
        
        # Convert to radians
        lat_min_rad = math.radians(self.lat_min)
        lat_max_rad = math.radians(self.lat_max)
        lon_min_rad = math.radians(self.lon_min)
        lon_max_rad = math.radians(self.lon_max)
        
        # Area on a sphere: R² * |sin(lat1) - sin(lat2)| * |lon2 - lon1|
        area = (EARTH_RADIUS_KM ** 2) * \
               abs(math.sin(lat_max_rad) - math.sin(lat_min_rad)) * \
               abs(lon_max_rad - lon_min_rad)
        
        return area
    
    def pad(self, degrees: float) -> BoundingBox:
        """
        Expand the bounding box by a margin in all directions.
        
        Args:
            degrees: Padding in degrees (positive expands, negative contracts)
            
        Returns:
            New BoundingBox with expanded bounds (clamped to valid ranges)
        """
        return BoundingBox(
            lat_min=max(-90.0, self.lat_min - degrees),
            lat_max=min(90.0, self.lat_max + degrees),
            lon_min=max(-180.0, self.lon_min - degrees),
            lon_max=min(180.0, self.lon_max + degrees),
        )
    
    def center(self) -> tuple[float, float]:
        """Return the center point (lat, lon) of the bounding box."""
        return (
            (self.lat_min + self.lat_max) / 2,
            (self.lon_min + self.lon_max) / 2,
        )
    
    def width_km(self) -> float:
        """Approximate east-west extent in kilometers at the center latitude."""
        center_lat = (self.lat_min + self.lat_max) / 2
        km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))
        return (self.lon_max - self.lon_min) * km_per_deg_lon
    
    def height_km(self) -> float:
        """Approximate north-south extent in kilometers."""
        km_per_deg_lat = 111.0  # Roughly constant
        return (self.lat_max - self.lat_min) * km_per_deg_lat
    
    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return as (lat_min, lat_max, lon_min, lon_max) tuple."""
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)
    
    def to_wsen(self) -> tuple[float, float, float, float]:
        """Return as (west, south, east, north) tuple - common GIS convention."""
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)
    
    def intersects(self, other: BoundingBox) -> bool:
        """Check if this bounding box intersects another."""
        return not (self.lat_max < other.lat_min or
                    self.lat_min > other.lat_max or
                    self.lon_max < other.lon_min or
                    self.lon_min > other.lon_max)
    
    def intersection(self, other: BoundingBox) -> Optional[BoundingBox]:
        """Return the intersection of two bounding boxes, or None if they don't intersect."""
        if not self.intersects(other):
            return None
        return BoundingBox(
            lat_min=max(self.lat_min, other.lat_min),
            lat_max=min(self.lat_max, other.lat_max),
            lon_min=max(self.lon_min, other.lon_min),
            lon_max=min(self.lon_max, other.lon_max),
        )
    
    def __repr__(self) -> str:
        return (f"BoundingBox(lat=[{self.lat_min:.4f}, {self.lat_max:.4f}], "
                f"lon=[{self.lon_min:.4f}, {self.lon_max:.4f}])")


# =============================================================================
# Predefined Regions
# =============================================================================

REGIONS: dict[str, BoundingBox] = {
    # Primary test region - Chesapeake Bay (full bay)
    "chesapeake_bay": BoundingBox(
        lat_min=36.8,
        lat_max=39.5,
        lon_min=-76.6,
        lon_max=-75.7,
    ),
    
    # Detailed sub-region: Chesapeake Bay mouth + CBBT area
    "chesapeake_mouth": BoundingBox(
        lat_min=36.8,
        lat_max=37.3,
        lon_min=-76.3,
        lon_max=-75.9,
    ),
    
    # Outer Banks, North Carolina
    "outer_banks_nc": BoundingBox(
        lat_min=34.5,
        lat_max=36.5,
        lon_min=-76.5,
        lon_max=-75.3,
    ),
    
    # Galveston Bay, Texas
    "galveston_bay_tx": BoundingBox(
        lat_min=29.2,
        lat_max=29.8,
        lon_min=-95.2,
        lon_max=-94.5,
    ),
    
    # San Francisco Bay, California
    "sf_bay_ca": BoundingBox(
        lat_min=37.4,
        lat_max=38.2,
        lon_min=-122.6,
        lon_max=-122.0,
    ),
    
    # Puget Sound, Washington
    "puget_sound_wa": BoundingBox(
        lat_min=47.0,
        lat_max=48.5,
        lon_min=-123.0,
        lon_max=-122.2,
    ),
    
    # Delaware Bay
    "delaware_bay": BoundingBox(
        lat_min=38.5,
        lat_max=39.8,
        lon_min=-75.6,
        lon_max=-74.8,
    ),
    
    # Mobile Bay, Alabama
    "mobile_bay_al": BoundingBox(
        lat_min=30.2,
        lat_max=30.8,
        lon_min=-88.2,
        lon_max=-87.8,
    ),
    
    # Tampa Bay, Florida
    "tampa_bay_fl": BoundingBox(
        lat_min=27.4,
        lat_max=28.1,
        lon_min=-82.9,
        lon_max=-82.3,
    ),
    
    # Long Island Sound
    "long_island_sound": BoundingBox(
        lat_min=40.8,
        lat_max=41.4,
        lon_min=-73.8,
        lon_max=-72.0,
    ),
    
    # Tiny test region for unit tests (near CBBT tide station)
    "test_tiny": BoundingBox(
        lat_min=36.95,
        lat_max=37.05,
        lon_min=-76.15,
        lon_max=-76.05,
    ),
}


# =============================================================================
# Region Configuration
# =============================================================================

@dataclass
class RegionConfig:
    """
    Complete configuration for downloading and processing data for one region.
    
    This is the main configuration object passed to CoastalDataManager.
    """
    name: str
    bounds: BoundingBox
    
    # ----- Bathymetry Configuration -----
    bathy_resolution_arcsec: float = 1.0  # 1/9 (~3m), 1/3 (~10m), or 1 (~30m) arc-second
    bathy_datum: str = "MSL"  # Vertical datum: "MSL", "NAVD88", "MLLW"
    
    # ----- Tide Station Configuration -----
    tide_stations: Optional[list[str]] = None  # Station IDs, or None for auto-discover
    tide_date_range: tuple[str, str] = ("20240101", "20241231")  # YYYYMMDD format
    tide_products: list[str] = field(default_factory=lambda: [
        "water_level",
        "predictions",
        "harmonic_constituents",
        "datums",
    ])
    
    # ----- Wave Buoy Configuration -----
    wave_stations: Optional[list[str]] = None  # Station IDs, or None for auto-discover
    wave_date_range: tuple[str, str] = ("20240101", "20241231")  # YYYYMMDD format
    wave_buoy_padding_deg: float = 2.0  # Padding for buoy discovery (buoys are offshore)
    
    # ----- Ocean Profile Configuration -----
    woa_resolution: str = "0.25"  # "1" or "0.25" degree grid
    woa_season: str = "annual"  # "annual", "winter", "spring", "summer", "fall"
    
    # ----- Shoreline Configuration -----
    shoreline_resolution: str = "high"  # "crude", "low", "intermediate", "high", "full"
    
    # ----- Cache Configuration -----
    cache_dir: str = "data/coastal/cache"
    force_redownload: bool = False
    
    # ----- Processing Configuration -----
    mesh_min_resolution_m: float = 100.0  # Finest mesh resolution near coast
    mesh_max_resolution_m: float = 2000.0  # Coarsest mesh resolution offshore
    mesh_refinement_depth_m: float = 20.0  # Refine where depth < this value
    
    def __post_init__(self):
        """Validate configuration."""
        if self.bathy_resolution_arcsec not in [1/9, 1/3, 1.0]:
            # Allow approximate values
            valid = [1/9, 1/3, 1.0]
            closest = min(valid, key=lambda x: abs(x - self.bathy_resolution_arcsec))
            if abs(closest - self.bathy_resolution_arcsec) > 0.01:
                raise ValueError(
                    f"bathy_resolution_arcsec must be ~1/9, ~1/3, or ~1, "
                    f"got {self.bathy_resolution_arcsec}"
                )
        
        if self.bathy_datum not in ["MSL", "NAVD88", "MLLW", "MHHW", "MHW", "MLW"]:
            raise ValueError(f"Invalid bathy_datum: {self.bathy_datum}")
        
        if self.woa_resolution not in ["1", "0.25"]:
            raise ValueError(f"woa_resolution must be '1' or '0.25', got {self.woa_resolution}")
        
        valid_seasons = ["annual", "winter", "spring", "summer", "fall"]
        if self.woa_season not in valid_seasons:
            raise ValueError(f"woa_season must be one of {valid_seasons}, got {self.woa_season}")
        
        valid_shoreline = ["crude", "low", "intermediate", "high", "full"]
        if self.shoreline_resolution not in valid_shoreline:
            raise ValueError(
                f"shoreline_resolution must be one of {valid_shoreline}, "
                f"got {self.shoreline_resolution}"
            )
    
    @property
    def bbox(self) -> BoundingBox:
        """Alias for bounds for API consistency."""
        return self.bounds
    
    @property
    def bathymetry_resolution(self) -> float:
        """Get bathymetry resolution in degrees (for CUDEM downloader)."""
        # Convert arc-seconds to degrees
        return self.bathy_resolution_arcsec / 3600.0
    
    @property
    def min_mesh_edge(self) -> float:
        """Alias for mesh_min_resolution_m."""
        return self.mesh_min_resolution_m
    
    @property
    def max_mesh_edge(self) -> float:
        """Alias for mesh_max_resolution_m."""
        return self.mesh_max_resolution_m
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> RegionConfig:
        """Load configuration from a YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required to load YAML configs. Install with: pip install pyyaml")
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        # Parse bounding box
        bounds_data = data.pop("bounds")
        bounds = BoundingBox(**bounds_data)
        
        return cls(bounds=bounds, **data)
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required to save YAML configs. Install with: pip install pyyaml")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "name": self.name,
            "bounds": {
                "lat_min": self.bounds.lat_min,
                "lat_max": self.bounds.lat_max,
                "lon_min": self.bounds.lon_min,
                "lon_max": self.bounds.lon_max,
            },
            "bathy_resolution_arcsec": self.bathy_resolution_arcsec,
            "bathy_datum": self.bathy_datum,
            "tide_stations": self.tide_stations,
            "tide_date_range": list(self.tide_date_range),
            "tide_products": self.tide_products,
            "wave_stations": self.wave_stations,
            "wave_date_range": list(self.wave_date_range),
            "wave_buoy_padding_deg": self.wave_buoy_padding_deg,
            "woa_resolution": self.woa_resolution,
            "woa_season": self.woa_season,
            "shoreline_resolution": self.shoreline_resolution,
            "cache_dir": self.cache_dir,
            "force_redownload": self.force_redownload,
            "mesh_min_resolution_m": self.mesh_min_resolution_m,
            "mesh_max_resolution_m": self.mesh_max_resolution_m,
            "mesh_refinement_depth_m": self.mesh_refinement_depth_m,
        }
        
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def get_cache_path(self, subdir: str, filename: str) -> Path:
        """Get the full cache path for a file."""
        cache_path = Path(self.cache_dir) / subdir / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path
    
    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            f"Region: {self.name}",
            f"  Bounds: {self.bounds}",
            f"  Area: {self.bounds.area_sq_km():.1f} km²",
            f"  Size: {self.bounds.width_km():.1f} km (E-W) × {self.bounds.height_km():.1f} km (N-S)",
            f"  Bathymetry: {self.bathy_resolution_arcsec:.4f} arcsec, datum={self.bathy_datum}",
            f"  Tide date range: {self.tide_date_range[0]} to {self.tide_date_range[1]}",
            f"  Wave date range: {self.wave_date_range[0]} to {self.wave_date_range[1]}",
            f"  WOA: {self.woa_resolution}° resolution, {self.woa_season} season",
            f"  Shoreline: {self.shoreline_resolution} resolution",
            f"  Cache: {self.cache_dir}",
        ]
        return "\n".join(lines)


# =============================================================================
# Utility Functions
# =============================================================================

def get_utm_zone(lon: float, lat: float) -> tuple[int, str]:
    """
    Determine the UTM zone for a given longitude/latitude.
    
    Returns:
        Tuple of (zone_number, hemisphere) where hemisphere is 'N' or 'S'
    """
    zone_number = int((lon + 180) / 6) + 1
    hemisphere = 'N' if lat >= 0 else 'S'
    return zone_number, hemisphere


def get_utm_epsg(lon: float, lat: float) -> int:
    """
    Get the EPSG code for the UTM zone containing a point.
    
    Returns:
        EPSG code (e.g., 32618 for UTM zone 18N)
    """
    zone, hemisphere = get_utm_zone(lon, lat)
    if hemisphere == 'N':
        return 32600 + zone
    else:
        return 32700 + zone


def great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points in kilometers.
    
    Uses the Haversine formula.
    """
    EARTH_RADIUS_KM = 6371.0
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return EARTH_RADIUS_KM * c


def get_region_config(region_name: str) -> RegionConfig:
    """
    Get a predefined region configuration by name.
    
    Args:
        region_name: Name of the region (e.g., 'chesapeake_bay')
        
    Returns:
        RegionConfig for the specified region
        
    Raises:
        KeyError: If region_name is not found in REGIONS
    """
    if region_name not in REGIONS:
        available = list(REGIONS.keys())
        raise KeyError(
            f"Unknown region '{region_name}'. Available regions: {available}"
        )
    bbox = REGIONS[region_name]
    return RegionConfig(name=region_name, bounds=bbox)
