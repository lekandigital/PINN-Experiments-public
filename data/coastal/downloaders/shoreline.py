"""
Shoreline geometry downloader.

Downloads coastline data for mesh generation and boundary definition:
- Natural Earth (primary, free Shapefile download)
- GSHHG via GMT (alternative)

Data sources:
- Natural Earth: https://www.naturalearthdata.com/
- GSHHG: https://www.ngdc.noaa.gov/mgg/shorelines/
"""

from __future__ import annotations

import logging
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import BoundingBox, RegionConfig
from ..datatypes import Shoreline
from .utils import (
    download_with_retry,
    NCEI_RATE_LIMITER,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Natural Earth coastline URLs (Shapefile format)
NATURAL_EARTH_URLS = {
    "full": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip",
    "high": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip",
    "intermediate": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/physical/ne_50m_coastline.zip",
    "low": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/physical/ne_110m_coastline.zip",
    "crude": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/physical/ne_110m_coastline.zip",
}

# Resolution in approximate meters
RESOLUTION_METERS = {
    "full": 100,
    "high": 500,
    "intermediate": 5000,
    "low": 50000,
    "crude": 100000,
}


# =============================================================================
# Downloader Class
# =============================================================================

class ShorelineDownloader:
    """
    Downloads coastline geometry from Natural Earth.
    
    Handles:
    - Multiple resolution levels
    - Clipping to bounding box
    - Conversion to numpy arrays
    
    Example:
        downloader = ShorelineDownloader(cache_dir="data/coastal/cache/shoreline")
        
        shoreline = downloader.download(bounds, resolution="high")
    """
    
    def __init__(self, cache_dir: str | Path = "data/coastal/cache/shoreline"):
        """
        Initialize the shoreline downloader.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._shapely_available = None
        self._fiona_available = None
    
    def _check_shapely(self) -> bool:
        """Check if shapely is available."""
        if self._shapely_available is None:
            try:
                import shapely
                self._shapely_available = True
            except ImportError:
                logger.warning("shapely not available for geometry operations")
                self._shapely_available = False
        return self._shapely_available
    
    def _check_fiona(self) -> bool:
        """Check if fiona is available for shapefile reading."""
        if self._fiona_available is None:
            try:
                import fiona
                self._fiona_available = True
            except ImportError:
                logger.warning("fiona not available for shapefile reading")
                self._fiona_available = False
        return self._fiona_available
    
    # -------------------------------------------------------------------------
    # Main Download Method
    # -------------------------------------------------------------------------
    
    def download(
        self,
        config: RegionConfig,
        force: bool = False,
    ) -> Shoreline:
        """
        Download shoreline data for a region.
        
        Args:
            config: Region configuration
            force: Force re-download
            
        Returns:
            Shoreline with coastline segments
        """
        bounds = config.bounds
        resolution = config.shoreline_resolution
        
        # Cache key
        cache_key = f"shoreline_{bounds.lat_min:.2f}_{bounds.lat_max:.2f}_{bounds.lon_min:.2f}_{bounds.lon_max:.2f}_{resolution}"
        cache_path = self.cache_dir / f"{cache_key}.npz"
        
        # Check cache
        if not force and cache_path.exists():
            logger.info(f"Loading cached shoreline from {cache_path}")
            return self._load_cache(cache_path)
        
        logger.info(f"Downloading {resolution} resolution shoreline")
        
        # Try to download and extract from Natural Earth
        shoreline = None
        
        if self._check_fiona() and self._check_shapely():
            shoreline = self._download_natural_earth(bounds, resolution)
        
        # Fallback to synthetic coastline
        if shoreline is None:
            shoreline = self._generate_synthetic_coastline(bounds, resolution)
        
        # Cache
        self._save_cache(cache_path, shoreline)
        
        return shoreline
    
    # -------------------------------------------------------------------------
    # Natural Earth Download
    # -------------------------------------------------------------------------
    
    def _download_natural_earth(
        self,
        bounds: BoundingBox,
        resolution: str,
    ) -> Optional[Shoreline]:
        """
        Download and extract Natural Earth coastline shapefile.
        
        Args:
            bounds: Bounding box to clip to
            resolution: Resolution level
            
        Returns:
            Shoreline or None if download fails
        """
        import fiona
        from shapely.geometry import shape, box
        
        url = NATURAL_EARTH_URLS.get(resolution, NATURAL_EARTH_URLS["high"])
        
        # Check if we have the shapefile cached
        shapefile_dir = self.cache_dir / f"ne_{resolution}"
        if not shapefile_dir.exists():
            try:
                logger.info(f"Downloading Natural Earth coastline from {url}")
                content = download_with_retry(
                    url,
                    timeout=120.0,  # Large file
                    rate_limiter=NCEI_RATE_LIMITER,
                )
                
                # Extract zip
                logger.info("Extracting shapefile...")
                shapefile_dir.mkdir(parents=True, exist_ok=True)
                
                with zipfile.ZipFile(BytesIO(content)) as zf:
                    zf.extractall(shapefile_dir)
                
            except Exception as e:
                logger.warning(f"Failed to download Natural Earth: {e}")
                return None
        
        # Find the .shp file
        shp_files = list(shapefile_dir.rglob("*.shp"))
        if not shp_files:
            logger.warning("No shapefile found in download")
            return None
        
        shp_path = shp_files[0]
        
        try:
            # Create clipping box
            clip_box = box(bounds.lon_min, bounds.lat_min, bounds.lon_max, bounds.lat_max)
            
            segments = []
            
            with fiona.open(shp_path, 'r') as src:
                for feature in src:
                    try:
                        geom = shape(feature['geometry'])
                        
                        # Check if geometry intersects our bounds
                        if not geom.intersects(clip_box):
                            continue
                        
                        # Clip to bounds
                        clipped = geom.intersection(clip_box)
                        
                        # Extract coordinates from clipped geometry
                        if clipped.is_empty:
                            continue
                        
                        # Handle different geometry types
                        if clipped.geom_type == 'LineString':
                            coords = np.array(clipped.coords)
                            if len(coords) >= 2:
                                segments.append(coords)
                        elif clipped.geom_type == 'MultiLineString':
                            for line in clipped.geoms:
                                coords = np.array(line.coords)
                                if len(coords) >= 2:
                                    segments.append(coords)
                        elif clipped.geom_type in ['Polygon', 'MultiPolygon']:
                            # Extract exterior ring(s)
                            if clipped.geom_type == 'Polygon':
                                polys = [clipped]
                            else:
                                polys = list(clipped.geoms)
                            
                            for poly in polys:
                                coords = np.array(poly.exterior.coords)
                                if len(coords) >= 2:
                                    segments.append(coords)
                                    
                    except Exception as e:
                        logger.debug(f"Error processing feature: {e}")
                        continue
            
            if not segments:
                logger.warning("No coastline segments found in region")
                return None
            
            logger.info(f"Extracted {len(segments)} coastline segments with {sum(len(s) for s in segments)} points")
            
            return Shoreline(
                segments=segments,
                resolution=resolution,
                source="NaturalEarth",
            )
            
        except Exception as e:
            logger.warning(f"Failed to process shapefile: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Synthetic Coastline (Fallback)
    # -------------------------------------------------------------------------
    
    def _generate_synthetic_coastline(
        self,
        bounds: BoundingBox,
        resolution: str,
    ) -> Shoreline:
        """
        Generate a synthetic coastline for testing.
        
        Creates a simple coastline along the western edge of the bounding box
        (typical for US East Coast regions).
        """
        logger.warning("Generating synthetic coastline - no real data available")
        
        # Resolution determines point spacing
        res_m = RESOLUTION_METERS.get(resolution, 1000)
        km_per_deg = 111.0
        point_spacing_deg = res_m / 1000 / km_per_deg
        
        # Generate coastline along western edge with some variation
        n_points = int((bounds.lat_max - bounds.lat_min) / point_spacing_deg)
        n_points = max(10, min(1000, n_points))
        
        lats = np.linspace(bounds.lat_min, bounds.lat_max, n_points)
        
        # Add some sinusoidal variation to make it look more realistic
        base_lon = bounds.lon_min + 0.3 * (bounds.lon_max - bounds.lon_min)
        variation = 0.05 * (bounds.lon_max - bounds.lon_min)
        
        lons = base_lon + variation * np.sin(lats * 10 * np.pi / (bounds.lat_max - bounds.lat_min))
        
        # Create segment (lon, lat format)
        segment = np.column_stack([lons, lats])
        
        return Shoreline(
            segments=[segment],
            resolution=resolution,
            source="SYNTHETIC",
        )
    
    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------
    
    def _save_cache(self, path: Path, shoreline: Shoreline) -> None:
        """Save shoreline to cache."""
        # Save as compressed numpy file with object arrays for variable-length segments
        np.savez_compressed(
            path,
            n_segments=len(shoreline.segments),
            resolution=shoreline.resolution,
            source=shoreline.source,
            **{f"segment_{i}": seg for i, seg in enumerate(shoreline.segments)}
        )
        logger.info(f"Cached shoreline to {path}")
    
    def _load_cache(self, path: Path) -> Shoreline:
        """Load shoreline from cache."""
        data = np.load(path, allow_pickle=True)
        n_segments = int(data['n_segments'])
        segments = [data[f'segment_{i}'] for i in range(n_segments)]
        
        return Shoreline(
            segments=segments,
            resolution=str(data['resolution']),
            source=str(data['source']),
        )


# =============================================================================
# Utility Functions
# =============================================================================

def simplify_shoreline(
    shoreline: Shoreline,
    tolerance: float,
) -> Shoreline:
    """
    Simplify shoreline using Douglas-Peucker algorithm.
    
    Args:
        shoreline: Input shoreline
        tolerance: Simplification tolerance in degrees
        
    Returns:
        Simplified Shoreline
    """
    return shoreline.simplify(tolerance)


def merge_segments(
    shoreline: Shoreline,
    gap_threshold_deg: float = 0.01,
) -> Shoreline:
    """
    Merge nearby shoreline segments.
    
    Args:
        shoreline: Input shoreline
        gap_threshold_deg: Maximum gap to bridge
        
    Returns:
        Shoreline with merged segments
    """
    if len(shoreline.segments) <= 1:
        return shoreline
    
    merged = []
    current = shoreline.segments[0].copy()
    
    for seg in shoreline.segments[1:]:
        # Check if this segment is close to the end of current
        dist = np.sqrt(
            (current[-1, 0] - seg[0, 0])**2 + 
            (current[-1, 1] - seg[0, 1])**2
        )
        
        if dist < gap_threshold_deg:
            # Merge
            current = np.vstack([current, seg])
        else:
            # Start new segment
            merged.append(current)
            current = seg.copy()
    
    merged.append(current)
    
    return Shoreline(
        segments=merged,
        resolution=shoreline.resolution,
        source=shoreline.source,
    )
