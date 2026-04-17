"""
Forcing data builders for coastal simulations.

Converts raw observational data into forcing fields for PDE solvers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np

from ..datatypes import (
    TideStation,
    WaterLevelTimeSeries,
    WaveBuoy,
    WaveBuoyTimeSeries,
    CoastalMesh,
    TidalForcing,
    WaveForcing,
    WindForcing,
    HarmonicConstituent,
)
from .coordinate_utils import great_circle_distance_array

logger = logging.getLogger(__name__)


# Standard tidal constituent angular speeds (degrees per hour)
TIDAL_SPEEDS = {
    'M2': 28.9841042,
    'S2': 30.0,
    'N2': 28.4397295,
    'K2': 30.0821373,
    'K1': 15.0410686,
    'O1': 13.9430356,
    'P1': 14.9589314,
    'Q1': 13.3986609,
    'M4': 57.9682084,
    'MS4': 58.9841042,
    'M6': 86.9523127,
    'S4': 60.0,
    'SA': 0.0410686,
    'SSA': 0.0821373,
}


@dataclass
class ForcingParameters:
    """Parameters for forcing field construction."""
    
    # Interpolation method: 'nearest', 'linear', 'idw'
    interpolation_method: str = 'idw'
    
    # IDW (Inverse Distance Weighting) parameters
    idw_power: float = 2.0
    idw_max_distance_km: float = 100.0
    
    # Number of nearest stations to use for interpolation
    n_nearest_stations: int = 5
    
    # Time interpolation
    time_interp_method: str = 'linear'
    
    # Wave forcing options
    compute_wave_radiation_stress: bool = True
    
    # Wind forcing options
    wind_drag_formula: str = 'large_pond'  # or 'smith_banke', 'garratt'


class ForcingBuilder:
    """
    Build forcing fields from observational data.
    
    Constructs tidal, wave, and wind forcing fields on mesh nodes
    from point observations (tide gauges, wave buoys, etc.).
    
    Example:
        builder = ForcingBuilder(params)
        tidal = builder.build_tidal_forcing(stations, mesh, time_array)
        wave = builder.build_wave_forcing(buoys, mesh, time_array)
    """
    
    def __init__(
        self,
        params: Optional[ForcingParameters] = None,
    ):
        """
        Initialize forcing builder.
        
        Args:
            params: Forcing construction parameters
        """
        self.params = params or ForcingParameters()
    
    def build_tidal_forcing(
        self,
        stations: List[TideStation],
        mesh: CoastalMesh,
        times: Optional[np.ndarray] = None,
        reference_time: Optional[str] = None,
    ) -> TidalForcing:
        """
        Build tidal forcing from tide gauge harmonics.
        
        Interpolates harmonic constituents from stations to mesh nodes,
        then optionally synthesizes time series.
        
        Args:
            stations: List of tide stations with harmonics
            mesh: Coastal mesh
            times: Optional time array (hours from reference) for synthesis
            reference_time: Optional reference time string
            
        Returns:
            TidalForcing with constituents and optionally time series
        """
        logger.info(f"Building tidal forcing from {len(stations)} stations...")
        
        n_nodes = mesh.n_nodes
        vertices = mesh.vertices  # (N, 2) [lon, lat]
        
        # Get stations with harmonic data
        stations_with_harmonics = [s for s in stations if s.harmonics]
        
        if not stations_with_harmonics:
            logger.warning("No stations with harmonic data")
            return self._empty_tidal_forcing(n_nodes)
        
        # Get all unique constituents
        all_constituents = set()
        for station in stations_with_harmonics:
            all_constituents.update(c.name for c in station.harmonics)
        all_constituents = sorted(all_constituents)
        
        logger.info(f"  Found {len(all_constituents)} unique constituents")
        
        # Build interpolation weights
        station_lons = np.array([s.longitude for s in stations_with_harmonics])
        station_lats = np.array([s.latitude for s in stations_with_harmonics])
        
        weights, indices = self._compute_interpolation_weights(
            vertices[:, 0], vertices[:, 1],
            station_lons, station_lats,
        )
        
        # Interpolate each constituent
        node_constituents: List[Dict[str, HarmonicConstituent]] = [
            {} for _ in range(n_nodes)
        ]
        
        for const_name in all_constituents:
            # Get amplitude and phase from each station
            amplitudes = np.full(len(stations_with_harmonics), np.nan)
            phases = np.full(len(stations_with_harmonics), np.nan)
            
            for i, station in enumerate(stations_with_harmonics):
                for h in station.harmonics:
                    if h.name == const_name:
                        amplitudes[i] = h.amplitude
                        phases[i] = h.phase
                        break
            
            # Interpolate amplitude and phase to each node
            for node_idx in range(n_nodes):
                w = weights[node_idx]
                idx = indices[node_idx]
                
                valid = ~np.isnan(amplitudes[idx]) & ~np.isnan(phases[idx])
                if not np.any(valid):
                    continue
                
                # Weighted average of amplitude
                w_valid = w[valid]
                w_valid = w_valid / w_valid.sum()
                
                amp = np.sum(w_valid * amplitudes[idx][valid])
                
                # Phase interpolation (handle wraparound)
                phase_rad = np.radians(phases[idx][valid])
                sin_avg = np.sum(w_valid * np.sin(phase_rad))
                cos_avg = np.sum(w_valid * np.cos(phase_rad))
                phase = np.degrees(np.arctan2(sin_avg, cos_avg)) % 360
                
                # Get angular speed
                speed = TIDAL_SPEEDS.get(const_name, 0.0)
                
                node_constituents[node_idx][const_name] = HarmonicConstituent(
                    name=const_name,
                    amplitude=amp,
                    phase=phase,
                    speed=speed,
                )
        
        # Build amplitude/phase arrays for primary constituents
        primary = ['M2', 'S2', 'K1', 'O1', 'N2']
        n_primary = len(primary)
        
        amplitudes_arr = np.zeros((n_nodes, n_primary))
        phases_arr = np.zeros((n_nodes, n_primary))
        
        for i, const_name in enumerate(primary):
            for node_idx in range(n_nodes):
                if const_name in node_constituents[node_idx]:
                    h = node_constituents[node_idx][const_name]
                    amplitudes_arr[node_idx, i] = h.amplitude
                    phases_arr[node_idx, i] = h.phase
        
        # Synthesize time series if times provided
        elevation = None
        velocity_u = None
        velocity_v = None
        
        if times is not None:
            logger.info(f"  Synthesizing time series at {len(times)} time steps...")
            elevation = self._synthesize_tidal_elevation(
                times, node_constituents, n_nodes
            )
        
        return TidalForcing(
            node_indices=np.arange(n_nodes),
            amplitudes=amplitudes_arr,
            phases=phases_arr,
            constituent_names=primary,
            elevation=elevation,
            velocity_u=velocity_u,
            velocity_v=velocity_v,
            times=times,
            reference_time=reference_time,
        )
    
    def build_wave_forcing(
        self,
        buoys: List[WaveBuoy],
        time_series: Dict[str, WaveBuoyTimeSeries],
        mesh: CoastalMesh,
        times: np.ndarray,
    ) -> WaveForcing:
        """
        Build wave forcing from buoy observations.
        
        Interpolates wave parameters (Hs, Tp, direction) to mesh nodes.
        
        Args:
            buoys: List of wave buoys
            time_series: Dict mapping station_id to time series data
            mesh: Coastal mesh
            times: Target time array (Unix timestamps)
            
        Returns:
            WaveForcing with wave parameters at each node and time
        """
        logger.info(f"Building wave forcing from {len(buoys)} buoys...")
        
        n_nodes = mesh.n_nodes
        n_times = len(times)
        vertices = mesh.vertices
        
        # Get buoys with available time series
        buoys_with_data = [b for b in buoys if b.station_id in time_series]
        
        if not buoys_with_data:
            logger.warning("No buoys with time series data")
            return self._empty_wave_forcing(n_nodes, n_times, times)
        
        # Compute interpolation weights
        buoy_lons = np.array([b.longitude for b in buoys_with_data])
        buoy_lats = np.array([b.latitude for b in buoys_with_data])
        
        weights, indices = self._compute_interpolation_weights(
            vertices[:, 0], vertices[:, 1],
            buoy_lons, buoy_lats,
        )
        
        # Interpolate each buoy's time series to target times
        buoy_hs = []  # (n_buoys, n_times)
        buoy_tp = []
        buoy_dir = []
        
        for buoy in buoys_with_data:
            ts = time_series[buoy.station_id]
            
            # Interpolate to target times
            hs_interp = np.interp(times, ts.times, ts.wave_height, 
                                  left=np.nan, right=np.nan)
            tp_interp = np.interp(times, ts.times, ts.dominant_period,
                                  left=np.nan, right=np.nan)
            
            # Handle direction (circular interpolation)
            if ts.mean_direction is not None:
                # Convert to unit vectors for interpolation
                dir_rad = np.radians(ts.mean_direction)
                dir_x = np.interp(times, ts.times, np.cos(dir_rad),
                                  left=np.nan, right=np.nan)
                dir_y = np.interp(times, ts.times, np.sin(dir_rad),
                                  left=np.nan, right=np.nan)
                dir_interp = np.degrees(np.arctan2(dir_y, dir_x)) % 360
            else:
                dir_interp = np.full(n_times, np.nan)
            
            buoy_hs.append(hs_interp)
            buoy_tp.append(tp_interp)
            buoy_dir.append(dir_interp)
        
        buoy_hs = np.array(buoy_hs)  # (n_buoys, n_times)
        buoy_tp = np.array(buoy_tp)
        buoy_dir = np.array(buoy_dir)
        
        # Interpolate to mesh nodes
        significant_height = np.zeros((n_nodes, n_times))
        peak_period = np.zeros((n_nodes, n_times))
        mean_direction = np.zeros((n_nodes, n_times))
        
        for node_idx in range(n_nodes):
            w = weights[node_idx]
            idx = indices[node_idx]
            
            # Interpolate Hs and Tp
            for t in range(n_times):
                hs_vals = buoy_hs[idx, t]
                tp_vals = buoy_tp[idx, t]
                dir_vals = buoy_dir[idx, t]
                
                valid_hs = ~np.isnan(hs_vals)
                valid_tp = ~np.isnan(tp_vals)
                valid_dir = ~np.isnan(dir_vals)
                
                if np.any(valid_hs):
                    w_norm = w[valid_hs] / w[valid_hs].sum()
                    significant_height[node_idx, t] = np.sum(w_norm * hs_vals[valid_hs])
                
                if np.any(valid_tp):
                    w_norm = w[valid_tp] / w[valid_tp].sum()
                    peak_period[node_idx, t] = np.sum(w_norm * tp_vals[valid_tp])
                
                if np.any(valid_dir):
                    w_norm = w[valid_dir] / w[valid_dir].sum()
                    dir_rad = np.radians(dir_vals[valid_dir])
                    sin_avg = np.sum(w_norm * np.sin(dir_rad))
                    cos_avg = np.sum(w_norm * np.cos(dir_rad))
                    mean_direction[node_idx, t] = np.degrees(np.arctan2(sin_avg, cos_avg)) % 360
        
        # Compute radiation stress if requested
        radiation_stress_xx = None
        radiation_stress_yy = None
        radiation_stress_xy = None
        
        if self.params.compute_wave_radiation_stress:
            radiation_stress_xx, radiation_stress_yy, radiation_stress_xy = \
                self._compute_radiation_stress(
                    significant_height, peak_period, mean_direction, mesh.depths
                )
        
        return WaveForcing(
            node_indices=np.arange(n_nodes),
            significant_height=significant_height,
            peak_period=peak_period,
            mean_direction=mean_direction,
            radiation_stress_xx=radiation_stress_xx,
            radiation_stress_yy=radiation_stress_yy,
            radiation_stress_xy=radiation_stress_xy,
            times=times,
        )
    
    def build_wind_forcing(
        self,
        wind_speed: np.ndarray,
        wind_direction: np.ndarray,
        times: np.ndarray,
        mesh: CoastalMesh,
    ) -> WindForcing:
        """
        Build wind forcing from wind observations.
        
        Computes wind stress from wind speed/direction.
        
        Args:
            wind_speed: (n_times,) or (n_stations, n_times) wind speed m/s
            wind_direction: Wind direction (from) in degrees
            times: Time array (Unix timestamps)
            mesh: Coastal mesh
            
        Returns:
            WindForcing with wind stress at mesh nodes
        """
        logger.info("Building wind forcing...")
        
        n_nodes = mesh.n_nodes
        n_times = len(times)
        
        # Handle scalar (spatially uniform) wind
        if wind_speed.ndim == 1:
            # Broadcast to all nodes
            wind_u = wind_speed * np.cos(np.radians(270 - wind_direction))
            wind_v = wind_speed * np.sin(np.radians(270 - wind_direction))
            
            wind_u_field = np.tile(wind_u, (n_nodes, 1))
            wind_v_field = np.tile(wind_v, (n_nodes, 1))
        else:
            # Spatial wind field - interpolate to nodes
            # For now, just average all stations
            wind_u = np.mean(wind_speed, axis=0) * np.cos(np.radians(270 - wind_direction))
            wind_v = np.mean(wind_speed, axis=0) * np.sin(np.radians(270 - wind_direction))
            
            wind_u_field = np.tile(wind_u, (n_nodes, 1))
            wind_v_field = np.tile(wind_v, (n_nodes, 1))
        
        # Compute wind stress
        speed = np.sqrt(wind_u_field**2 + wind_v_field**2)
        
        rho_air = 1.225  # kg/m³
        cd = self._compute_drag_coefficient(speed)
        
        stress_magnitude = rho_air * cd * speed**2
        
        stress_x = stress_magnitude * (wind_u_field / np.maximum(speed, 0.01))
        stress_y = stress_magnitude * (wind_v_field / np.maximum(speed, 0.01))
        
        return WindForcing(
            node_indices=np.arange(n_nodes),
            stress_x=stress_x,
            stress_y=stress_y,
            wind_u=wind_u_field,
            wind_v=wind_v_field,
            times=times,
        )
    
    def _compute_interpolation_weights(
        self,
        node_lons: np.ndarray,
        node_lats: np.ndarray,
        station_lons: np.ndarray,
        station_lats: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spatial interpolation weights from stations to nodes.
        
        Returns:
            Tuple of (weights, indices) where weights[i] and indices[i]
            give the interpolation weights and station indices for node i.
        """
        n_nodes = len(node_lons)
        n_stations = len(station_lons)
        n_use = min(self.params.n_nearest_stations, n_stations)
        
        weights = np.zeros((n_nodes, n_use))
        indices = np.zeros((n_nodes, n_use), dtype=int)
        
        for i in range(n_nodes):
            # Compute distance to all stations
            distances = great_circle_distance_array(
                node_lats[i], node_lons[i],
                station_lats, station_lons
            )
            
            # Find nearest stations
            sorted_idx = np.argsort(distances)
            nearest_idx = sorted_idx[:n_use]
            nearest_dist = distances[nearest_idx]
            
            indices[i] = nearest_idx
            
            # Compute weights based on method
            if self.params.interpolation_method == 'nearest':
                weights[i, 0] = 1.0
            elif self.params.interpolation_method == 'linear':
                # Simple distance-weighted
                total_dist = np.sum(nearest_dist)
                if total_dist > 0:
                    weights[i] = (total_dist - nearest_dist) / total_dist
                else:
                    weights[i, 0] = 1.0
            else:  # IDW
                # Inverse distance weighting
                mask = nearest_dist <= self.params.idw_max_distance_km
                if not np.any(mask):
                    mask[0] = True  # Use nearest if all too far
                
                # Avoid division by zero
                nearest_dist = np.maximum(nearest_dist, 0.001)
                
                w = 1.0 / (nearest_dist ** self.params.idw_power)
                w[~mask] = 0
                weights[i] = w / w.sum()
        
        return weights, indices
    
    def _synthesize_tidal_elevation(
        self,
        times: np.ndarray,
        node_constituents: List[Dict[str, HarmonicConstituent]],
        n_nodes: int,
    ) -> np.ndarray:
        """
        Synthesize tidal elevation time series from constituents.
        
        Args:
            times: Time array (hours from reference)
            node_constituents: Constituents for each node
            n_nodes: Number of mesh nodes
            
        Returns:
            (n_nodes, n_times) array of water elevation
        """
        n_times = len(times)
        elevation = np.zeros((n_nodes, n_times))
        
        for node_idx in range(n_nodes):
            constituents = node_constituents[node_idx]
            
            for name, h in constituents.items():
                # ζ = A * cos(ωt - φ)
                omega = np.radians(h.speed)  # Convert deg/hr to rad/hr
                phase = np.radians(h.phase)
                
                elevation[node_idx] += h.amplitude * np.cos(omega * times - phase)
        
        return elevation
    
    def _compute_radiation_stress(
        self,
        hs: np.ndarray,
        tp: np.ndarray,
        direction: np.ndarray,
        depths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute wave radiation stress components.
        
        Uses linear wave theory approximations.
        """
        n_nodes, n_times = hs.shape
        
        # Avoid division by zero
        tp_safe = np.maximum(tp, 0.1)
        depths_safe = np.maximum(depths[:, np.newaxis], 0.1)
        
        # Wave number from dispersion relation (deep water approximation for simplicity)
        g = 9.81
        omega = 2 * np.pi / tp_safe
        k = omega**2 / g  # Deep water approximation
        
        # Wave energy density
        rho = 1025.0  # kg/m³
        E = rho * g * hs**2 / 16
        
        # Group velocity factor n = 1/2 * (1 + 2kd/sinh(2kd))
        kd = k * depths_safe
        kd = np.clip(kd, 0.01, 10)  # Avoid overflow
        n_factor = 0.5 * (1 + 2*kd / np.sinh(2*kd))
        
        # Direction in radians (direction wave is going to)
        theta = np.radians(direction + 180)  # Convert "from" to "to"
        
        # Radiation stress components (Longuet-Higgins & Stewart)
        sxx = E * (n_factor * (1 + np.cos(theta)**2) - 0.5)
        syy = E * (n_factor * (1 + np.sin(theta)**2) - 0.5)
        sxy = E * n_factor * np.sin(theta) * np.cos(theta)
        
        return sxx, syy, sxy
    
    def _compute_drag_coefficient(
        self,
        wind_speed: np.ndarray,
    ) -> np.ndarray:
        """
        Compute wind drag coefficient.
        """
        cd = np.zeros_like(wind_speed)
        
        if self.params.wind_drag_formula == 'large_pond':
            # Large & Pond (1981)
            low = wind_speed < 11
            high = wind_speed >= 11
            
            cd[low] = 1.2e-3
            cd[high] = (0.49 + 0.065 * wind_speed[high]) * 1e-3
            
        elif self.params.wind_drag_formula == 'garratt':
            # Garratt (1977)
            cd = (0.75 + 0.067 * wind_speed) * 1e-3
            
        else:  # smith_banke
            # Smith & Banke (1975)
            cd = (0.63 + 0.066 * wind_speed) * 1e-3
        
        return np.clip(cd, 0.5e-3, 3e-3)
    
    def _empty_tidal_forcing(self, n_nodes: int) -> TidalForcing:
        """Create empty tidal forcing."""
        return TidalForcing(
            node_indices=np.arange(n_nodes),
            amplitudes=np.zeros((n_nodes, 5)),
            phases=np.zeros((n_nodes, 5)),
            constituent_names=['M2', 'S2', 'K1', 'O1', 'N2'],
        )
    
    def _empty_wave_forcing(
        self, n_nodes: int, n_times: int, times: np.ndarray
    ) -> WaveForcing:
        """Create empty wave forcing."""
        return WaveForcing(
            node_indices=np.arange(n_nodes),
            significant_height=np.zeros((n_nodes, n_times)),
            peak_period=np.zeros((n_nodes, n_times)),
            mean_direction=np.zeros((n_nodes, n_times)),
            times=times,
        )
