"""
Data type definitions for coastal data infrastructure.

All dataclasses representing coastal data: bathymetry, tides, waves,
ocean profiles, shorelines, meshes, forcing, and validation datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import BoundingBox


# =============================================================================
# Bathymetry Data Types
# =============================================================================

@dataclass
class BathymetryGrid:
    """
    Bathymetry/topography data on a regular lat/lon grid.
    
    Elevation values are in meters relative to a vertical datum.
    Negative values = underwater (bathymetry), positive values = land (topography).
    """
    latitude: np.ndarray  # (Nlat,) - latitude values in degrees north
    longitude: np.ndarray  # (Nlon,) - longitude values in degrees east
    elevation: np.ndarray  # (Nlat, Nlon) - elevation in meters
    resolution_arcsec: float  # Grid resolution in arc-seconds
    datum: str  # Vertical datum (e.g., "NAVD88", "MSL", "MLLW")
    source: str  # Data source identifier (e.g., "CUDEM", "ETOPO")
    crs: str = "EPSG:4326"  # Coordinate reference system
    
    @property
    def land_mask(self) -> np.ndarray:
        """Boolean mask where True indicates land (elevation > 0)."""
        return self.elevation > 0
    
    @property
    def water_mask(self) -> np.ndarray:
        """Boolean mask where True indicates water (elevation <= 0)."""
        return self.elevation <= 0
    
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (lat_min, lat_max, lon_min, lon_max)."""
        return (
            float(self.latitude.min()),
            float(self.latitude.max()),
            float(self.longitude.min()),
            float(self.longitude.max()),
        )
    
    @property
    def shape(self) -> tuple[int, int]:
        """Return (Nlat, Nlon) grid shape."""
        return self.elevation.shape
    
    def depth_at(self, lat: float, lon: float) -> float:
        """
        Get interpolated depth at a point (positive down for water).
        
        Returns:
            Depth in meters (positive for underwater, negative for land)
        """
        from scipy.interpolate import RegularGridInterpolator
        
        interp = RegularGridInterpolator(
            (self.latitude, self.longitude),
            -self.elevation,  # Negate so positive = underwater depth
            method='linear',
            bounds_error=False,
            fill_value=np.nan,
        )
        return float(interp([[lat, lon]])[0])
    
    def to_xarray(self):
        """Convert to xarray DataArray."""
        try:
            import xarray as xr
            return xr.DataArray(
                self.elevation,
                dims=['latitude', 'longitude'],
                coords={
                    'latitude': self.latitude,
                    'longitude': self.longitude,
                },
                attrs={
                    'units': 'meters',
                    'datum': self.datum,
                    'source': self.source,
                    'resolution_arcsec': self.resolution_arcsec,
                },
            )
        except ImportError:
            raise ImportError("xarray is required for to_xarray()")
    
    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"BathymetryGrid: {self.shape[0]}×{self.shape[1]} grid\n"
            f"  Lat: [{self.latitude.min():.4f}, {self.latitude.max():.4f}]\n"
            f"  Lon: [{self.longitude.min():.4f}, {self.longitude.max():.4f}]\n"
            f"  Elevation: [{self.elevation.min():.1f}, {self.elevation.max():.1f}] m\n"
            f"  Resolution: {self.resolution_arcsec:.4f} arcsec\n"
            f"  Datum: {self.datum}, Source: {self.source}\n"
            f"  Land coverage: {100*self.land_mask.mean():.1f}%"
        )


# =============================================================================
# Tide/Current Data Types
# =============================================================================

@dataclass
class HarmonicConstituent:
    """
    Single tidal harmonic constituent.
    
    Tides are the sum of harmonic constituents:
    η(t) = Σ A_k * cos(ω_k * t - φ_k)
    """
    name: str  # e.g., "M2", "S2", "K1", "O1"
    amplitude_m: float  # Amplitude in meters
    phase_deg: float  # Phase in degrees (Greenwich epoch)
    speed_deg_hr: float  # Angular speed in degrees per hour
    description: str = ""  # e.g., "Principal lunar semidiurnal"
    
    def evaluate(self, t_hours: np.ndarray) -> np.ndarray:
        """
        Evaluate this constituent's contribution at given times.
        
        Args:
            t_hours: Time in hours since epoch
            
        Returns:
            Water level contribution in meters
        """
        phase_rad = np.radians(self.phase_deg)
        speed_rad_hr = np.radians(self.speed_deg_hr)
        return self.amplitude_m * np.cos(speed_rad_hr * t_hours - phase_rad)


# Standard tidal constituent speeds (degrees per hour)
TIDAL_CONSTITUENT_SPEEDS = {
    "M2": 28.984104,   # Principal lunar semidiurnal
    "S2": 30.000000,   # Principal solar semidiurnal
    "N2": 28.439730,   # Larger lunar elliptic semidiurnal
    "K2": 30.082137,   # Lunisolar semidiurnal
    "K1": 15.041069,   # Lunisolar diurnal
    "O1": 13.943035,   # Principal lunar diurnal
    "P1": 14.958931,   # Principal solar diurnal
    "Q1": 13.398661,   # Larger lunar elliptic diurnal
    "M4": 57.968208,   # Shallow water overtide of M2
    "M6": 86.952313,   # Shallow water overtide of M2
    "S4": 60.000000,   # Shallow water overtide of S2
    "MS4": 58.984104,  # Shallow water quarter diurnal
}


@dataclass
class TideStation:
    """
    Tide gauge station with metadata and optional data.
    """
    station_id: str
    name: str
    latitude: float
    longitude: float
    state: str = ""
    
    # Capabilities
    has_water_level: bool = True
    has_currents: bool = False
    has_met: bool = False  # Meteorological data
    
    # Datum offsets (meters relative to station datum)
    # e.g., {"MHHW": 0.82, "MHW": 0.71, "MSL": 0.37, "MLW": 0.03, "MLLW": 0.0}
    datums: dict[str, float] = field(default_factory=dict)
    
    # Harmonic constituents
    harmonics: list[HarmonicConstituent] = field(default_factory=list)
    
    def predict_tide(
        self,
        t_hours: np.ndarray,
        datum: str = "MSL",
    ) -> np.ndarray:
        """
        Predict tidal water level from harmonic constituents.
        
        Args:
            t_hours: Time in hours since epoch
            datum: Vertical datum to reference predictions to
            
        Returns:
            Predicted water level in meters relative to datum
        """
        if not self.harmonics:
            raise ValueError(f"No harmonic constituents available for station {self.station_id}")
        
        # Sum all constituent contributions
        prediction = np.zeros_like(t_hours, dtype=np.float64)
        for constituent in self.harmonics:
            prediction += constituent.evaluate(t_hours)
        
        # Adjust for datum offset if available
        if datum in self.datums and "MSL" in self.datums:
            prediction += self.datums["MSL"] - self.datums.get(datum, 0)
        
        return prediction
    
    def summary(self) -> str:
        """Return human-readable summary."""
        caps = []
        if self.has_water_level:
            caps.append("water_level")
        if self.has_currents:
            caps.append("currents")
        if self.has_met:
            caps.append("met")
        
        return (
            f"TideStation {self.station_id}: {self.name}\n"
            f"  Location: ({self.latitude:.4f}, {self.longitude:.4f}), {self.state}\n"
            f"  Capabilities: {', '.join(caps)}\n"
            f"  Harmonics: {len(self.harmonics)} constituents\n"
            f"  Datums: {list(self.datums.keys())}"
        )


@dataclass
class WaterLevelTimeSeries:
    """
    Observed water level time series from a tide station.
    """
    station_id: str
    times: np.ndarray  # (N,) - timestamps as numpy datetime64
    water_level_m: np.ndarray  # (N,) - water level in meters relative to datum
    datum: str  # Vertical datum
    quality_flags: Optional[np.ndarray] = None  # (N,) - quality flags per observation
    
    @property
    def time_hours(self) -> np.ndarray:
        """Time in hours since first observation."""
        dt = (self.times - self.times[0]).astype('timedelta64[s]').astype(float)
        return dt / 3600.0
    
    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask where True indicates valid (non-NaN) observations."""
        return ~np.isnan(self.water_level_m)
    
    def resample(self, interval_minutes: int = 60) -> WaterLevelTimeSeries:
        """Resample to a regular time interval."""
        # Simple implementation - could be improved with proper resampling
        from scipy.interpolate import interp1d
        
        valid = self.valid_mask
        if not valid.any():
            return self
        
        # Create interpolator from valid data
        t_valid = self.time_hours[valid]
        wl_valid = self.water_level_m[valid]
        interp = interp1d(t_valid, wl_valid, kind='linear', bounds_error=False, fill_value=np.nan)
        
        # New regular time grid
        interval_hours = interval_minutes / 60.0
        t_new = np.arange(t_valid.min(), t_valid.max(), interval_hours)
        wl_new = interp(t_new)
        
        # Convert back to datetime64
        start_time = self.times[valid][0]
        times_new = start_time + (t_new * 3600).astype('timedelta64[s]')
        
        return WaterLevelTimeSeries(
            station_id=self.station_id,
            times=times_new,
            water_level_m=wl_new,
            datum=self.datum,
            quality_flags=None,
        )
    
    def summary(self) -> str:
        """Return human-readable summary."""
        valid = self.valid_mask
        return (
            f"WaterLevelTimeSeries: Station {self.station_id}\n"
            f"  Time range: {self.times[0]} to {self.times[-1]}\n"
            f"  Samples: {len(self.times)} ({valid.sum()} valid)\n"
            f"  Water level: [{np.nanmin(self.water_level_m):.2f}, "
            f"{np.nanmax(self.water_level_m):.2f}] m ({self.datum})"
        )


@dataclass
class CurrentTimeSeries:
    """
    Observed current velocity time series from a station.
    """
    station_id: str
    times: np.ndarray  # (N,) datetime64
    speed_m_s: np.ndarray  # (N,) - current speed in m/s
    direction_deg: np.ndarray  # (N,) - current direction in degrees (from)
    
    @property
    def u_m_s(self) -> np.ndarray:
        """East-west velocity component (positive = eastward)."""
        direction_rad = np.radians(self.direction_deg)
        return self.speed_m_s * np.sin(direction_rad)
    
    @property
    def v_m_s(self) -> np.ndarray:
        """North-south velocity component (positive = northward)."""
        direction_rad = np.radians(self.direction_deg)
        return self.speed_m_s * np.cos(direction_rad)
    
    def summary(self) -> str:
        return (
            f"CurrentTimeSeries: Station {self.station_id}\n"
            f"  Time range: {self.times[0]} to {self.times[-1]}\n"
            f"  Samples: {len(self.times)}\n"
            f"  Speed: [{np.nanmin(self.speed_m_s):.2f}, {np.nanmax(self.speed_m_s):.2f}] m/s"
        )


# =============================================================================
# Wave Buoy Data Types
# =============================================================================

@dataclass
class WaveBuoy:
    """
    Wave buoy station metadata.
    """
    station_id: str
    name: str
    latitude: float
    longitude: float
    water_depth_m: Optional[float] = None
    
    def summary(self) -> str:
        depth_str = f"{self.water_depth_m:.0f}m" if self.water_depth_m else "unknown"
        return (
            f"WaveBuoy {self.station_id}: {self.name}\n"
            f"  Location: ({self.latitude:.4f}, {self.longitude:.4f})\n"
            f"  Water depth: {depth_str}"
        )


@dataclass
class WaveBuoyTimeSeries:
    """
    Wave measurements from a buoy.
    """
    station_id: str
    times: np.ndarray  # (N,) datetime64
    significant_wave_height_m: np.ndarray  # (N,) - Hs
    dominant_period_s: np.ndarray  # (N,) - Tp
    average_period_s: np.ndarray  # (N,) - Ta
    mean_direction_deg: np.ndarray  # (N,) - MWD
    wind_speed_m_s: np.ndarray  # (N,)
    wind_direction_deg: np.ndarray  # (N,)
    water_temp_c: np.ndarray  # (N,)
    
    @property
    def valid_wave_mask(self) -> np.ndarray:
        """Boolean mask where wave height is valid."""
        return ~np.isnan(self.significant_wave_height_m)
    
    def summary(self) -> str:
        valid = self.valid_wave_mask
        return (
            f"WaveBuoyTimeSeries: Station {self.station_id}\n"
            f"  Time range: {self.times[0]} to {self.times[-1]}\n"
            f"  Samples: {len(self.times)} ({valid.sum()} valid waves)\n"
            f"  Hs: [{np.nanmin(self.significant_wave_height_m):.2f}, "
            f"{np.nanmax(self.significant_wave_height_m):.2f}] m\n"
            f"  Tp: [{np.nanmin(self.dominant_period_s):.1f}, "
            f"{np.nanmax(self.dominant_period_s):.1f}] s"
        )


# =============================================================================
# Ocean Profile Data Types
# =============================================================================

@dataclass
class OceanProfile:
    """
    Vertical profile of ocean properties at one location.
    """
    latitude: float
    longitude: float
    depths_m: np.ndarray  # (D,) - depth levels in meters
    temperature_c: np.ndarray  # (D,) - temperature at each depth
    salinity_psu: np.ndarray  # (D,) - salinity at each depth
    sound_speed_m_s: np.ndarray  # (D,) - computed sound speed
    density_kg_m3: Optional[np.ndarray] = None  # (D,) - computed density
    season: str = "annual"
    
    @property
    def sofar_depth_m(self) -> float:
        """Depth of the SOFAR channel (sound speed minimum)."""
        if len(self.sound_speed_m_s) == 0:
            return np.nan
        min_idx = np.argmin(self.sound_speed_m_s)
        return float(self.depths_m[min_idx])
    
    def sound_speed_at(self, depth: float) -> float:
        """Interpolate sound speed at a specific depth."""
        return float(np.interp(depth, self.depths_m, self.sound_speed_m_s))
    
    def summary(self) -> str:
        return (
            f"OceanProfile at ({self.latitude:.4f}, {self.longitude:.4f})\n"
            f"  Depth range: {self.depths_m.min():.0f} - {self.depths_m.max():.0f} m "
            f"({len(self.depths_m)} levels)\n"
            f"  Temperature: [{self.temperature_c.min():.1f}, {self.temperature_c.max():.1f}] °C\n"
            f"  Salinity: [{self.salinity_psu.min():.1f}, {self.salinity_psu.max():.1f}] PSU\n"
            f"  Sound speed: [{self.sound_speed_m_s.min():.1f}, {self.sound_speed_m_s.max():.1f}] m/s\n"
            f"  SOFAR depth: {self.sofar_depth_m:.0f} m\n"
            f"  Season: {self.season}"
        )


@dataclass
class OceanProfileGrid:
    """
    3D grid of ocean properties over a region.
    """
    latitudes: np.ndarray  # (Nlat,)
    longitudes: np.ndarray  # (Nlon,)
    depths_m: np.ndarray  # (D,)
    temperature: np.ndarray  # (Nlat, Nlon, D)
    salinity: np.ndarray  # (Nlat, Nlon, D)
    sound_speed: np.ndarray  # (Nlat, Nlon, D)
    season: str = "annual"
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """Return (Nlat, Nlon, D) shape."""
        return self.temperature.shape
    
    def extract_profile(self, lat: float, lon: float) -> OceanProfile:
        """Extract a single vertical profile via bilinear interpolation."""
        from scipy.interpolate import RegularGridInterpolator
        
        # Find nearest lat/lon indices for metadata
        lat_idx = np.argmin(np.abs(self.latitudes - lat))
        lon_idx = np.argmin(np.abs(self.longitudes - lon))
        
        # Interpolate each variable
        profiles = {}
        for name, data in [('temperature', self.temperature),
                          ('salinity', self.salinity),
                          ('sound_speed', self.sound_speed)]:
            # Create 2D interpolator for each depth level
            profile = np.zeros(len(self.depths_m))
            for d_idx, depth in enumerate(self.depths_m):
                interp = RegularGridInterpolator(
                    (self.latitudes, self.longitudes),
                    data[:, :, d_idx],
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan,
                )
                profile[d_idx] = interp([[lat, lon]])[0]
            profiles[name] = profile
        
        return OceanProfile(
            latitude=lat,
            longitude=lon,
            depths_m=self.depths_m.copy(),
            temperature_c=profiles['temperature'],
            salinity_psu=profiles['salinity'],
            sound_speed_m_s=profiles['sound_speed'],
            season=self.season,
        )
    
    def summary(self) -> str:
        return (
            f"OceanProfileGrid: {self.shape[0]}×{self.shape[1]}×{self.shape[2]}\n"
            f"  Lat: [{self.latitudes.min():.2f}, {self.latitudes.max():.2f}]\n"
            f"  Lon: [{self.longitudes.min():.2f}, {self.longitudes.max():.2f}]\n"
            f"  Depth: [{self.depths_m.min():.0f}, {self.depths_m.max():.0f}] m\n"
            f"  Season: {self.season}"
        )


# =============================================================================
# Shoreline Data Types
# =============================================================================

@dataclass
class Shoreline:
    """
    Coastline geometry for a region.
    """
    segments: list[np.ndarray]  # List of (Ni, 2) arrays [lon, lat] polylines
    resolution: str  # "crude", "low", "intermediate", "high", "full"
    source: str = "GSHHG"  # Data source
    
    @property
    def total_points(self) -> int:
        """Total number of vertices across all segments."""
        return sum(seg.shape[0] for seg in self.segments)
    
    @property
    def total_segments(self) -> int:
        """Number of polyline segments."""
        return len(self.segments)
    
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (lat_min, lat_max, lon_min, lon_max)."""
        if not self.segments:
            return (np.nan, np.nan, np.nan, np.nan)
        
        all_points = np.vstack(self.segments)
        return (
            float(all_points[:, 1].min()),  # lat_min
            float(all_points[:, 1].max()),  # lat_max
            float(all_points[:, 0].min()),  # lon_min
            float(all_points[:, 0].max()),  # lon_max
        )
    
    def simplify(self, tolerance: float) -> Shoreline:
        """
        Reduce vertex count using Douglas-Peucker algorithm.
        
        Args:
            tolerance: Simplification tolerance in degrees
        """
        try:
            from shapely.geometry import LineString
            
            simplified_segments = []
            for seg in self.segments:
                if len(seg) < 3:
                    simplified_segments.append(seg)
                    continue
                
                line = LineString(seg)
                simplified = line.simplify(tolerance, preserve_topology=True)
                simplified_segments.append(np.array(simplified.coords))
            
            return Shoreline(
                segments=simplified_segments,
                resolution=self.resolution,
                source=self.source,
            )
        except ImportError:
            # Fallback: simple point decimation
            simplified_segments = []
            keep_every = max(1, int(tolerance * 100))
            for seg in self.segments:
                simplified_segments.append(seg[::keep_every])
            
            return Shoreline(
                segments=simplified_segments,
                resolution=self.resolution,
                source=self.source,
            )
    
    def to_mesh_boundary(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert to boundary constraint edges for Delaunay triangulation.
        
        Returns:
            Tuple of (vertices, edges) where:
            - vertices: (N, 2) array of [lon, lat] coordinates
            - edges: (M, 2) array of vertex index pairs defining boundary edges
        """
        if not self.segments:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
        
        vertices_list = []
        edges_list = []
        vertex_offset = 0
        
        for seg in self.segments:
            n_points = seg.shape[0]
            vertices_list.append(seg)
            
            # Create edges connecting consecutive points
            for i in range(n_points - 1):
                edges_list.append([vertex_offset + i, vertex_offset + i + 1])
            
            vertex_offset += n_points
        
        vertices = np.vstack(vertices_list)
        edges = np.array(edges_list) if edges_list else np.array([]).reshape(0, 2)
        
        return vertices, edges.astype(np.int32)
    
    def summary(self) -> str:
        return (
            f"Shoreline ({self.resolution} resolution)\n"
            f"  Segments: {self.total_segments}\n"
            f"  Total points: {self.total_points}\n"
            f"  Source: {self.source}"
        )


# =============================================================================
# Mesh Data Types
# =============================================================================

@dataclass
class CoastalMesh:
    """
    Unstructured triangular mesh with coastal attributes.
    
    Compatible with:
    - PyTorch Geometric (via to_pyg_data())
    - NetworkX (via to_networkx())
    - Project 06 CoastFlow-GNN format
    """
    # Core geometry
    vertices: np.ndarray  # (V, 2) - lon, lat positions
    vertices_xy: np.ndarray  # (V, 2) - projected x, y in meters
    faces: np.ndarray  # (F, 3) - triangle vertex indices
    edges: np.ndarray  # (E, 2) - edge vertex indices
    
    # Node attributes
    depth: np.ndarray  # (V,) - bathymetric depth in meters (positive = underwater)
    distance_to_shore: np.ndarray  # (V,) - distance to nearest shoreline in meters
    
    # Edge attributes
    edge_length: np.ndarray  # (E,) - edge length in meters
    edge_orientation: np.ndarray  # (E,) - edge orientation in radians
    
    # Boundary classification
    open_boundary_edges: np.ndarray  # indices into edges array
    land_boundary_edges: np.ndarray  # indices into edges array
    open_boundary_nodes: np.ndarray  # indices into vertices
    land_boundary_nodes: np.ndarray  # indices into vertices
    
    # Metadata
    projection: str = ""  # e.g., "UTM zone 18N"
    crs: str = "EPSG:4326"
    
    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]
    
    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]
    
    @property
    def num_edges(self) -> int:
        return self.edges.shape[0]
    
    def to_pyg_data(self):
        """
        Convert to PyTorch Geometric Data object.
        
        Format matches Project 06 (CoastFlow-GNN) expectations:
        - x: [N, 6] node features (x, y, z, depth, wind_u, wind_v)
        - edge_index: [2, E] edge connectivity
        - pos: [N, 3] node positions
        - boundary_mask, boundary_type for boundary conditions
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("torch and torch_geometric are required for to_pyg_data()")
        
        # Node features: [x, y, z=0, depth, wind_u=0, wind_v=0]
        x = np.zeros((self.num_vertices, 6), dtype=np.float32)
        x[:, 0] = self.vertices_xy[:, 0]  # x position (meters)
        x[:, 1] = self.vertices_xy[:, 1]  # y position (meters)
        x[:, 2] = 0.0  # z (surface)
        x[:, 3] = self.depth  # bathymetric depth
        # wind_u, wind_v left as 0 - to be filled by forcing
        
        # Node positions
        pos = np.zeros((self.num_vertices, 3), dtype=np.float32)
        pos[:, 0] = self.vertices_xy[:, 0]
        pos[:, 1] = self.vertices_xy[:, 1]
        pos[:, 2] = -self.depth  # elevation (negative for underwater)
        
        # Edge index (undirected: add both directions)
        edge_index = np.vstack([
            np.concatenate([self.edges[:, 0], self.edges[:, 1]]),
            np.concatenate([self.edges[:, 1], self.edges[:, 0]]),
        ])
        
        # Boundary masks
        boundary_mask = np.zeros(self.num_vertices, dtype=bool)
        boundary_mask[self.open_boundary_nodes] = True
        boundary_mask[self.land_boundary_nodes] = True
        
        # Boundary types: 0=interior, 1=land (wall), 4=open (outlet)
        boundary_type = np.zeros(self.num_vertices, dtype=np.int32)
        boundary_type[self.land_boundary_nodes] = 1
        boundary_type[self.open_boundary_nodes] = 4
        
        return Data(
            x=torch.tensor(x),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            pos=torch.tensor(pos),
            boundary_mask=torch.tensor(boundary_mask),
            boundary_type=torch.tensor(boundary_type),
            # Additional attributes
            vertices_lonlat=torch.tensor(self.vertices, dtype=torch.float32),
            faces=torch.tensor(self.faces, dtype=torch.long),
            depth=torch.tensor(self.depth, dtype=torch.float32),
            edge_length=torch.tensor(self.edge_length, dtype=torch.float32),
        )
    
    def to_networkx(self):
        """Convert to NetworkX graph."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required for to_networkx()")
        
        G = nx.Graph()
        
        # Add nodes with attributes
        for i in range(self.num_vertices):
            G.add_node(
                i,
                lon=self.vertices[i, 0],
                lat=self.vertices[i, 1],
                x=self.vertices_xy[i, 0],
                y=self.vertices_xy[i, 1],
                depth=self.depth[i],
                distance_to_shore=self.distance_to_shore[i],
            )
        
        # Add edges with attributes
        for e_idx, (i, j) in enumerate(self.edges):
            G.add_edge(
                i, j,
                length=self.edge_length[e_idx],
                orientation=self.edge_orientation[e_idx],
            )
        
        return G
    
    def to_jax_arrays(self) -> dict[str, Any]:
        """
        Convert to JAX-compatible numpy arrays.
        
        For Projects 02, 16, 17 that use JAX/Haiku.
        """
        return {
            'vertices': self.vertices.astype(np.float32),
            'vertices_xy': self.vertices_xy.astype(np.float32),
            'faces': self.faces.astype(np.int32),
            'edges': self.edges.astype(np.int32),
            'depth': self.depth.astype(np.float32),
            'distance_to_shore': self.distance_to_shore.astype(np.float32),
            'edge_length': self.edge_length.astype(np.float32),
            'edge_orientation': self.edge_orientation.astype(np.float32),
            'open_boundary_nodes': self.open_boundary_nodes.astype(np.int32),
            'land_boundary_nodes': self.land_boundary_nodes.astype(np.int32),
        }
    
    def summary(self) -> str:
        return (
            f"CoastalMesh\n"
            f"  Vertices: {self.num_vertices}\n"
            f"  Faces: {self.num_faces}\n"
            f"  Edges: {self.num_edges}\n"
            f"  Depth range: [{self.depth.min():.1f}, {self.depth.max():.1f}] m\n"
            f"  Open boundary nodes: {len(self.open_boundary_nodes)}\n"
            f"  Land boundary nodes: {len(self.land_boundary_nodes)}\n"
            f"  Projection: {self.projection}"
        )


# =============================================================================
# Forcing Data Types
# =============================================================================

@dataclass
class TidalForcing:
    """
    Time-varying water level boundary condition for model forcing.
    """
    times: np.ndarray  # (T,) - datetime64 timestamps
    boundary_node_indices: np.ndarray  # (Nb,) - mesh node indices on open boundary
    water_level: np.ndarray  # (T, Nb) - water level at each boundary node
    datum: str = "MSL"
    
    # Harmonic decomposition for frequency-domain models
    harmonics_per_node: Optional[list[list[HarmonicConstituent]]] = None
    
    @property
    def time_hours(self) -> np.ndarray:
        """Time in hours since first timestamp."""
        dt = (self.times - self.times[0]).astype('timedelta64[s]').astype(float)
        return dt / 3600.0
    
    def summary(self) -> str:
        return (
            f"TidalForcing\n"
            f"  Time: {self.times[0]} to {self.times[-1]} ({len(self.times)} steps)\n"
            f"  Boundary nodes: {len(self.boundary_node_indices)}\n"
            f"  Water level range: [{self.water_level.min():.2f}, {self.water_level.max():.2f}] m"
        )


@dataclass
class WaveForcing:
    """
    Wave conditions at the offshore boundary for model forcing.
    """
    times: np.ndarray  # (T,) datetime64
    boundary_node_indices: np.ndarray  # (Nb,)
    wave_height: np.ndarray  # (T, Nb) - significant wave height in m
    wave_period: np.ndarray  # (T, Nb) - peak period in s
    wave_direction: np.ndarray  # (T, Nb) - mean direction in degrees
    
    def summary(self) -> str:
        return (
            f"WaveForcing\n"
            f"  Time: {self.times[0]} to {self.times[-1]} ({len(self.times)} steps)\n"
            f"  Boundary nodes: {len(self.boundary_node_indices)}\n"
            f"  Hs range: [{self.wave_height.min():.2f}, {self.wave_height.max():.2f}] m"
        )


@dataclass
class WindForcing:
    """
    Spatially-varying wind field over the model domain.
    """
    times: np.ndarray  # (T,) datetime64
    wind_u: np.ndarray  # (T, V) - east-west component at each mesh node
    wind_v: np.ndarray  # (T, V) - north-south component at each mesh node
    
    @property
    def wind_speed(self) -> np.ndarray:
        """Wind speed magnitude (T, V)."""
        return np.sqrt(self.wind_u**2 + self.wind_v**2)
    
    @property
    def wind_direction(self) -> np.ndarray:
        """Wind direction in degrees (from) (T, V)."""
        return np.degrees(np.arctan2(self.wind_u, self.wind_v)) % 360
    
    def summary(self) -> str:
        return (
            f"WindForcing\n"
            f"  Time: {self.times[0]} to {self.times[-1]} ({len(self.times)} steps)\n"
            f"  Nodes: {self.wind_u.shape[1]}\n"
            f"  Speed range: [{self.wind_speed.min():.1f}, {self.wind_speed.max():.1f}] m/s"
        )


# =============================================================================
# Validation Data Types
# =============================================================================

@dataclass
class ValidationPoint:
    """
    Single validation point matching observation station to mesh node.
    """
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    mesh_node_index: int
    distance_to_node_m: float
    observation_type: str  # "water_level", "current", "wave"
    observations: Any  # WaterLevelTimeSeries / CurrentTimeSeries / WaveBuoyTimeSeries


@dataclass
class ValidationDataset:
    """
    Collection of validation points for model evaluation.
    """
    water_level_stations: list[ValidationPoint] = field(default_factory=list)
    current_stations: list[ValidationPoint] = field(default_factory=list)
    wave_stations: list[ValidationPoint] = field(default_factory=list)
    
    @property
    def total_stations(self) -> int:
        return (len(self.water_level_stations) + 
                len(self.current_stations) + 
                len(self.wave_stations))
    
    def summary(self) -> str:
        return (
            f"ValidationDataset\n"
            f"  Water level stations: {len(self.water_level_stations)}\n"
            f"  Current stations: {len(self.current_stations)}\n"
            f"  Wave stations: {len(self.wave_stations)}"
        )


# =============================================================================
# Bundle Data Type
# =============================================================================

@dataclass
class CoastalDataBundle:
    """
    All downloaded data for one coastal region.
    
    This is the return type of CoastalDataManager.download_all().
    """
    config: Any  # RegionConfig (avoid circular import)
    bathymetry: Optional[BathymetryGrid] = None
    shoreline: Optional[Shoreline] = None
    tide_stations: dict[str, TideStation] = field(default_factory=dict)
    water_levels: dict[str, WaterLevelTimeSeries] = field(default_factory=dict)
    currents: dict[str, CurrentTimeSeries] = field(default_factory=dict)
    wave_buoys: dict[str, WaveBuoy] = field(default_factory=dict)
    wave_buoy_data: dict[str, WaveBuoyTimeSeries] = field(default_factory=dict)
    ocean_profiles: Optional[OceanProfileGrid] = None
    
    def summary(self) -> str:
        lines = [f"CoastalDataBundle: {self.config.name}"]
        
        if self.bathymetry:
            lines.append(f"  Bathymetry: {self.bathymetry.shape[0]}×{self.bathymetry.shape[1]} grid")
        else:
            lines.append("  Bathymetry: not loaded")
        
        if self.shoreline:
            lines.append(f"  Shoreline: {self.shoreline.total_points} points")
        else:
            lines.append("  Shoreline: not loaded")
        
        lines.append(f"  Tide stations: {len(self.tide_stations)}")
        lines.append(f"  Water level series: {len(self.water_levels)}")
        lines.append(f"  Current series: {len(self.currents)}")
        lines.append(f"  Wave buoys: {len(self.wave_buoys)}")
        lines.append(f"  Wave data series: {len(self.wave_buoy_data)}")
        
        if self.ocean_profiles:
            lines.append(f"  Ocean profiles: {self.ocean_profiles.shape}")
        else:
            lines.append("  Ocean profiles: not loaded")
        
        return "\n".join(lines)
