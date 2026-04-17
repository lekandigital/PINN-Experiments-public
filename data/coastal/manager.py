"""
Unified coastal data manager.

Orchestrates all downloaders and processors to provide complete
coastal data bundles for physics-informed ML projects.
"""

from __future__ import annotations

import logging
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Union

import numpy as np

from .config import (
    BoundingBox,
    RegionConfig,
    REGIONS,
    get_region_config,
)
from .datatypes import (
    BathymetryGrid,
    TideStation,
    WaterLevelTimeSeries,
    WaveBuoy,
    WaveBuoyTimeSeries,
    OceanProfileGrid,
    Shoreline,
    CoastalMesh,
    TidalForcing,
    WaveForcing,
    ValidationDataset,
    CoastalDataBundle,
)
from .downloaders import (
    COOPSDownloader,
    NDBCDownloader,
    CUDEMDownloader,
    WOADownloader,
    ShorelineDownloader,
)
from .processors import (
    CoastalMeshGenerator,
    MeshParameters,
    ForcingBuilder,
    ForcingParameters,
    ValidationBuilder,
    ValidationParameters,
)

logger = logging.getLogger(__name__)


class CoastalDataManager:
    """
    Unified manager for coastal data acquisition and processing.
    
    Provides a single interface for downloading all coastal data types,
    generating meshes, building forcing fields, and creating validation
    datasets for physics-informed ML projects.
    
    Example:
        manager = CoastalDataManager(cache_dir="./data/cache")
        
        # Get a complete data bundle for a region
        bundle = manager.get_bundle(
            region="chesapeake_bay",
            start_date="2023-01-01",
            end_date="2023-01-31",
        )
        
        # Use individual components
        bathymetry = manager.get_bathymetry("chesapeake_bay")
        mesh = manager.get_mesh("chesapeake_bay")
        tidal_forcing = manager.get_tidal_forcing("chesapeake_bay", mesh)
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
    ):
        """
        Initialize the coastal data manager.
        
        Args:
            cache_dir: Directory for caching downloaded data
            use_cache: Whether to use cached data when available
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./coastal_cache")
        self.use_cache = use_cache
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize downloaders
        self.coops = COOPSDownloader(cache_dir=self.cache_dir / "coops")
        self.ndbc = NDBCDownloader(cache_dir=self.cache_dir / "ndbc")
        self.cudem = CUDEMDownloader(cache_dir=self.cache_dir / "cudem")
        self.woa = WOADownloader(cache_dir=self.cache_dir / "woa")
        self.shoreline = ShorelineDownloader(cache_dir=self.cache_dir / "shoreline")
        
        # Initialize processors
        self.mesh_generator = CoastalMeshGenerator()
        self.forcing_builder = ForcingBuilder()
        self.validation_builder = ValidationBuilder()
        
        # Cache for loaded data
        self._bathymetry_cache: Dict[str, BathymetryGrid] = {}
        self._mesh_cache: Dict[str, CoastalMesh] = {}
        self._shoreline_cache: Dict[str, Shoreline] = {}
    
    def get_region_config(
        self,
        region: Union[str, RegionConfig, BoundingBox],
    ) -> RegionConfig:
        """
        Get or create a RegionConfig from various input types.
        
        Args:
            region: Region name, RegionConfig, or BoundingBox
            
        Returns:
            RegionConfig instance
        """
        if isinstance(region, str):
            return get_region_config(region)
        elif isinstance(region, BoundingBox):
            return RegionConfig(
                name="custom",
                bounds=region,
            )
        else:
            return region
    
    def get_bathymetry(
        self,
        region: Union[str, RegionConfig, BoundingBox],
        resolution: Optional[float] = None,
    ) -> BathymetryGrid:
        """
        Get bathymetry data for a region.
        
        Args:
            region: Region specification
            resolution: Optional resolution override in degrees
            
        Returns:
            BathymetryGrid with elevation data
        """
        config = self.get_region_config(region)
        cache_key = f"{config.name}_{resolution or 'default'}"
        
        if cache_key in self._bathymetry_cache:
            return self._bathymetry_cache[cache_key]
        
        logger.info(f"Downloading bathymetry for {config.name}...")
        
        bathymetry = self.cudem.download(
            config.bbox,
            resolution=resolution or config.bathymetry_resolution,
        )
        
        self._bathymetry_cache[cache_key] = bathymetry
        return bathymetry
    
    def get_shoreline(
        self,
        region: Union[str, RegionConfig, BoundingBox],
        resolution: str = "10m",
    ) -> Shoreline:
        """
        Get shoreline data for a region.
        
        Args:
            region: Region specification
            resolution: Shoreline resolution ('10m', '50m', '110m')
            
        Returns:
            Shoreline with coastal geometry
        """
        config = self.get_region_config(region)
        cache_key = f"{config.name}_{resolution}"
        
        if cache_key in self._shoreline_cache:
            return self._shoreline_cache[cache_key]
        
        logger.info(f"Downloading shoreline for {config.name}...")
        
        shoreline = self.shoreline.download(
            config.bbox,
            resolution=resolution,
        )
        
        self._shoreline_cache[cache_key] = shoreline
        return shoreline
    
    def get_mesh(
        self,
        region: Union[str, RegionConfig, BoundingBox],
        params: Optional[MeshParameters] = None,
        bathymetry: Optional[BathymetryGrid] = None,
        shoreline: Optional[Shoreline] = None,
    ) -> CoastalMesh:
        """
        Get or generate a computational mesh for a region.
        
        Args:
            region: Region specification
            params: Optional mesh parameters
            bathymetry: Optional pre-downloaded bathymetry
            shoreline: Optional pre-downloaded shoreline
            
        Returns:
            CoastalMesh ready for simulations
        """
        config = self.get_region_config(region)
        
        # Use cached mesh if available and no custom params
        if params is None and config.name in self._mesh_cache:
            return self._mesh_cache[config.name]
        
        # Get bathymetry and shoreline if not provided
        if bathymetry is None:
            bathymetry = self.get_bathymetry(region)
        
        if shoreline is None:
            shoreline = self.get_shoreline(region)
        
        # Generate mesh
        logger.info(f"Generating mesh for {config.name}...")
        
        if params is None:
            params = MeshParameters(
                min_edge_length=config.min_mesh_edge,
                max_edge_length=config.max_mesh_edge,
            )
        
        mesh = self.mesh_generator.generate(bathymetry, shoreline, config)
        
        if params is None:  # Only cache default mesh
            self._mesh_cache[config.name] = mesh
        
        return mesh
    
    def get_tide_stations(
        self,
        region: Union[str, RegionConfig, BoundingBox],
    ) -> List[TideStation]:
        """
        Discover tide stations in a region.
        
        Args:
            region: Region specification
            
        Returns:
            List of TideStation metadata
        """
        config = self.get_region_config(region)
        
        logger.info(f"Discovering tide stations in {config.name}...")
        return self.coops.discover_stations(config.bbox)
    
    def get_wave_buoys(
        self,
        region: Union[str, RegionConfig, BoundingBox],
    ) -> List[WaveBuoy]:
        """
        Discover wave buoys in a region.
        
        Args:
            region: Region specification
            
        Returns:
            List of WaveBuoy metadata
        """
        config = self.get_region_config(region)
        
        logger.info(f"Discovering wave buoys in {config.name}...")
        return self.ndbc.discover_stations(config.bbox)
    
    def get_water_levels(
        self,
        stations: List[TideStation],
        start_date: str,
        end_date: str,
    ) -> Dict[str, WaterLevelTimeSeries]:
        """
        Download water level data from tide stations.
        
        Args:
            stations: List of tide stations
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dict mapping station_id to WaterLevelTimeSeries
        """
        logger.info(f"Downloading water levels for {len(stations)} stations...")
        
        result = {}
        for station in stations:
            try:
                wl = self.coops.download_water_levels(
                    station.station_id,
                    start_date,
                    end_date,
                )
                if wl is not None:
                    result[station.station_id] = wl
            except Exception as e:
                logger.warning(f"Failed to download water levels for {station.station_id}: {e}")
        
        logger.info(f"  Downloaded data for {len(result)} stations")
        return result
    
    def get_wave_data(
        self,
        buoys: List[WaveBuoy],
        start_year: int,
        end_year: int,
    ) -> Dict[str, WaveBuoyTimeSeries]:
        """
        Download wave buoy data.
        
        Args:
            buoys: List of wave buoys
            start_year: Start year
            end_year: End year (inclusive)
            
        Returns:
            Dict mapping station_id to WaveBuoyTimeSeries
        """
        logger.info(f"Downloading wave data for {len(buoys)} buoys...")
        
        result = {}
        for buoy in buoys:
            try:
                wd = self.ndbc.download_range(
                    buoy.station_id,
                    start_year,
                    end_year,
                )
                if wd is not None:
                    result[buoy.station_id] = wd
            except Exception as e:
                logger.warning(f"Failed to download wave data for {buoy.station_id}: {e}")
        
        logger.info(f"  Downloaded data for {len(result)} buoys")
        return result
    
    def get_ocean_profiles(
        self,
        region: Union[str, RegionConfig, BoundingBox],
        variables: Optional[List[str]] = None,
    ) -> OceanProfileGrid:
        """
        Download ocean temperature/salinity profiles.
        
        Args:
            region: Region specification
            variables: Variables to download (default: temperature, salinity)
            
        Returns:
            OceanProfileGrid with oceanographic data
        """
        config = self.get_region_config(region)
        
        logger.info(f"Downloading ocean profiles for {config.name}...")
        
        return self.woa.download_profiles(
            config.bbox,
            variables=variables or ['temperature', 'salinity'],
        )
    
    def get_tidal_forcing(
        self,
        region: Union[str, RegionConfig, BoundingBox],
        mesh: CoastalMesh,
        times: Optional[np.ndarray] = None,
    ) -> TidalForcing:
        """
        Build tidal forcing for a mesh.
        
        Args:
            region: Region specification
            mesh: Coastal mesh
            times: Optional time array (hours from reference)
            
        Returns:
            TidalForcing with harmonic constituents
        """
        config = self.get_region_config(region)
        
        # Get tide stations with harmonics
        stations = self.get_tide_stations(region)
        
        # Download harmonics for each station
        for station in stations:
            self.coops.download_station_metadata(station.station_id, station)
        
        # Build forcing
        return self.forcing_builder.build_tidal_forcing(
            stations, mesh, times
        )
    
    def get_wave_forcing(
        self,
        region: Union[str, RegionConfig, BoundingBox],
        mesh: CoastalMesh,
        start_year: int,
        end_year: int,
        times: np.ndarray,
    ) -> WaveForcing:
        """
        Build wave forcing for a mesh.
        
        Args:
            region: Region specification
            mesh: Coastal mesh
            start_year: Start year for buoy data
            end_year: End year for buoy data
            times: Target time array (Unix timestamps)
            
        Returns:
            WaveForcing with wave parameters at mesh nodes
        """
        # Get wave buoys and data
        buoys = self.get_wave_buoys(region)
        wave_data = self.get_wave_data(buoys, start_year, end_year)
        
        # Build forcing
        return self.forcing_builder.build_wave_forcing(
            buoys, wave_data, mesh, times
        )
    
    def get_validation_data(
        self,
        region: Union[str, RegionConfig, BoundingBox],
        mesh: CoastalMesh,
        start_date: str,
        end_date: str,
    ) -> Dict[str, ValidationDataset]:
        """
        Build validation datasets for a region.
        
        Args:
            region: Region specification
            mesh: Coastal mesh
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dict mapping variable name to ValidationDataset
        """
        # Get observations
        tide_stations = self.get_tide_stations(region)
        water_levels = self.get_water_levels(tide_stations, start_date, end_date)
        
        wave_buoys = self.get_wave_buoys(region)
        start_year = int(start_date.split('-')[0])
        end_year = int(end_date.split('-')[0])
        wave_data = self.get_wave_data(wave_buoys, start_year, end_year)
        
        # Build validation datasets
        return self.validation_builder.build_combined(
            tide_stations, water_levels,
            wave_buoys, wave_data,
            mesh,
        )
    
    def get_bundle(
        self,
        region: Union[str, RegionConfig, BoundingBox],
        start_date: str,
        end_date: str,
        include_waves: bool = True,
        include_ocean_profiles: bool = False,
        mesh_params: Optional[MeshParameters] = None,
    ) -> CoastalDataBundle:
        """
        Get a complete data bundle for a region.
        
        Downloads all required data and builds a complete bundle
        ready for use in physics-informed ML projects.
        
        Args:
            region: Region specification
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_waves: Whether to include wave data
            include_ocean_profiles: Whether to include T/S profiles
            mesh_params: Optional mesh generation parameters
            
        Returns:
            CoastalDataBundle with all data components
        """
        config = self.get_region_config(region)
        
        logger.info(f"Building complete data bundle for {config.name}...")
        logger.info(f"  Period: {start_date} to {end_date}")
        
        # 1. Get bathymetry and shoreline
        bathymetry = self.get_bathymetry(region)
        shoreline = self.get_shoreline(region)
        
        # 2. Generate mesh
        mesh = self.get_mesh(region, mesh_params, bathymetry, shoreline)
        
        # 3. Get tide data
        tide_stations = self.get_tide_stations(region)
        water_levels = self.get_water_levels(tide_stations, start_date, end_date)
        
        # 4. Build tidal forcing
        tidal_forcing = self.get_tidal_forcing(region, mesh)
        
        # 5. Get wave data if requested
        wave_buoys = None
        wave_forcing = None
        if include_waves:
            wave_buoys = self.get_wave_buoys(region)
            # Skip wave forcing for now (requires time array)
        
        # 6. Get ocean profiles if requested
        ocean_profiles = None
        if include_ocean_profiles:
            ocean_profiles = self.get_ocean_profiles(region)
        
        # 7. Build validation data
        validation = self.get_validation_data(region, mesh, start_date, end_date)
        
        bundle = CoastalDataBundle(
            region=config,
            bathymetry=bathymetry,
            shoreline=shoreline,
            mesh=mesh,
            tide_stations=tide_stations,
            water_levels=water_levels,
            wave_buoys=wave_buoys or [],
            tidal_forcing=tidal_forcing,
            wave_forcing=wave_forcing,
            validation=validation,
            ocean_profiles=ocean_profiles,
        )
        
        logger.info(f"Bundle complete: {mesh.n_nodes} nodes, "
                   f"{len(tide_stations)} tide gauges, "
                   f"{len(wave_buoys or [])} wave buoys")
        
        return bundle
    
    def save_bundle(
        self,
        bundle: CoastalDataBundle,
        path: Union[str, Path],
        format: str = "npz",
    ) -> None:
        """
        Save a data bundle to disk.
        
        Args:
            bundle: CoastalDataBundle to save
            path: Output path (directory will be created)
            format: Output format ('npz' or 'hdf5')
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving bundle to {path}...")
        
        # Save mesh as NPZ
        if bundle.mesh is not None:
            np.savez(
                path / "mesh.npz",
                vertices=bundle.mesh.vertices,
                triangles=bundle.mesh.triangles,
                edges=bundle.mesh.edges,
                depths=bundle.mesh.depths,
                boundary_nodes=bundle.mesh.boundary_nodes,
                boundary_types=bundle.mesh.boundary_types,
                distance_to_shore=bundle.mesh.distance_to_shore,
            )
        
        # Save bathymetry
        if bundle.bathymetry is not None:
            np.savez(
                path / "bathymetry.npz",
                elevation=bundle.bathymetry.elevation,
                lon=bundle.bathymetry.lon,
                lat=bundle.bathymetry.lat,
            )
        
        # Save tidal forcing
        if bundle.tidal_forcing is not None:
            np.savez(
                path / "tidal_forcing.npz",
                amplitudes=bundle.tidal_forcing.amplitudes,
                phases=bundle.tidal_forcing.phases,
                constituent_names=bundle.tidal_forcing.constituent_names,
            )
        
        # Save metadata as JSON
        metadata = {
            "region": bundle.region.name,
            "bbox": {
                "lon_min": bundle.region.bbox.lon_min,
                "lon_max": bundle.region.bbox.lon_max,
                "lat_min": bundle.region.bbox.lat_min,
                "lat_max": bundle.region.bbox.lat_max,
            },
            "n_tide_stations": len(bundle.tide_stations),
            "n_wave_buoys": len(bundle.wave_buoys) if bundle.wave_buoys else 0,
            "created": datetime.now().isoformat(),
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  Saved to {path}")
    
    def load_bundle(
        self,
        path: Union[str, Path],
    ) -> CoastalDataBundle:
        """
        Load a data bundle from disk.
        
        Args:
            path: Path to saved bundle directory
            
        Returns:
            CoastalDataBundle
        """
        path = Path(path)
        
        logger.info(f"Loading bundle from {path}...")
        
        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        
        # Reconstruct region config
        bbox = BoundingBox(**metadata["bbox"])
        region = RegionConfig(name=metadata["region"], bounds=bbox)
        
        # Load mesh
        mesh_data = np.load(path / "mesh.npz")
        mesh = CoastalMesh(
            vertices=mesh_data["vertices"],
            triangles=mesh_data["triangles"],
            edges=mesh_data["edges"],
            depths=mesh_data["depths"],
            boundary_nodes=mesh_data["boundary_nodes"],
            boundary_types=mesh_data["boundary_types"],
            distance_to_shore=mesh_data.get("distance_to_shore"),
        )
        
        # Load bathymetry
        bathy_data = np.load(path / "bathymetry.npz")
        bathymetry = BathymetryGrid(
            elevation=bathy_data["elevation"],
            lon=bathy_data["lon"],
            lat=bathy_data["lat"],
            bbox=bbox,
        )
        
        # Load tidal forcing if present
        tidal_forcing = None
        if (path / "tidal_forcing.npz").exists():
            tf_data = np.load(path / "tidal_forcing.npz", allow_pickle=True)
            tidal_forcing = TidalForcing(
                node_indices=np.arange(mesh.n_nodes),
                amplitudes=tf_data["amplitudes"],
                phases=tf_data["phases"],
                constituent_names=list(tf_data["constituent_names"]),
            )
        
        return CoastalDataBundle(
            region=region,
            bathymetry=bathymetry,
            mesh=mesh,
            tide_stations=[],
            water_levels={},
            wave_buoys=[],
            tidal_forcing=tidal_forcing,
        )
    
    def clear_cache(self) -> None:
        """Clear all in-memory caches."""
        self._bathymetry_cache.clear()
        self._mesh_cache.clear()
        self._shoreline_cache.clear()
        logger.info("Cleared in-memory caches")


# Module-level convenience function
def get_coastal_data(
    region: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[str] = None,
) -> CoastalDataBundle:
    """
    Convenience function to get a complete coastal data bundle.
    
    Args:
        region: Region name (e.g., 'chesapeake_bay')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        cache_dir: Optional cache directory
        
    Returns:
        CoastalDataBundle ready for ML projects
    """
    manager = CoastalDataManager(cache_dir=cache_dir)
    return manager.get_bundle(region, start_date, end_date)
