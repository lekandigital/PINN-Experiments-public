"""
Validation data builder for coastal simulations.

Creates matched observation-prediction pairs for model validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np

from ..datatypes import (
    TideStation,
    WaterLevelTimeSeries,
    WaveBuoy,
    WaveBuoyTimeSeries,
    CoastalMesh,
    ValidationPoint,
    ValidationDataset,
)
from .coordinate_utils import great_circle_distance

logger = logging.getLogger(__name__)


@dataclass
class ValidationParameters:
    """Parameters for validation data construction."""
    
    # Maximum distance from mesh node to observation (km)
    max_distance_km: float = 5.0
    
    # Minimum number of valid observations required
    min_observations: int = 100
    
    # Time window for averaging (seconds)
    time_averaging_window: float = 3600.0  # 1 hour
    
    # Variables to include
    include_water_level: bool = True
    include_currents: bool = True
    include_waves: bool = True
    
    # Quality filters
    min_quality_score: float = 0.5


class ValidationBuilder:
    """
    Build validation datasets from observations.
    
    Creates matched pairs of observation points and mesh nodes
    for use in model validation and loss computation.
    
    Example:
        builder = ValidationBuilder(params)
        val_data = builder.build_from_tide_gauges(
            stations, water_levels, mesh
        )
    """
    
    def __init__(
        self,
        params: Optional[ValidationParameters] = None,
    ):
        """
        Initialize validation builder.
        
        Args:
            params: Validation construction parameters
        """
        self.params = params or ValidationParameters()
    
    def build_from_tide_gauges(
        self,
        stations: List[TideStation],
        water_levels: Dict[str, WaterLevelTimeSeries],
        mesh: CoastalMesh,
    ) -> ValidationDataset:
        """
        Build validation dataset from tide gauge water levels.
        
        Args:
            stations: List of tide stations
            water_levels: Dict mapping station_id to water level time series
            mesh: Coastal mesh
            
        Returns:
            ValidationDataset with matched observation points
        """
        logger.info(f"Building validation data from {len(stations)} tide gauges...")
        
        validation_points = []
        
        for station in stations:
            # Check if we have data for this station
            if station.station_id not in water_levels:
                continue
            
            wl = water_levels[station.station_id]
            
            # Find nearest mesh node
            nearest_node, distance_km = self._find_nearest_node(
                station.longitude, station.latitude, mesh
            )
            
            if distance_km > self.params.max_distance_km:
                logger.debug(
                    f"  Station {station.station_id} too far from mesh "
                    f"({distance_km:.1f} km)"
                )
                continue
            
            # Filter valid observations
            valid = ~np.isnan(wl.water_level)
            if np.sum(valid) < self.params.min_observations:
                logger.debug(
                    f"  Station {station.station_id} has too few observations "
                    f"({np.sum(valid)})"
                )
                continue
            
            times = wl.times[valid]
            values = wl.water_level[valid]
            
            # Compute quality score based on data completeness
            data_density = len(values) / max(1, (times[-1] - times[0]) / 3600)
            quality_score = min(1.0, data_density / 1.0)  # Expect ~1 obs/hour
            
            if quality_score < self.params.min_quality_score:
                continue
            
            validation_points.append(ValidationPoint(
                station_id=station.station_id,
                variable='water_level',
                longitude=station.longitude,
                latitude=station.latitude,
                mesh_node_idx=nearest_node,
                distance_to_node=distance_km * 1000,  # Convert to meters
                times=times,
                observations=values,
                observation_error=np.full_like(values, 0.02),  # 2 cm assumed error
                quality_score=quality_score,
            ))
        
        logger.info(f"  Created {len(validation_points)} validation points")
        
        # Compute overall statistics
        all_obs = np.concatenate([vp.observations for vp in validation_points]) \
                  if validation_points else np.array([])
        
        return ValidationDataset(
            points=validation_points,
            variable='water_level',
            n_observations=len(all_obs),
            time_range=(
                (min(vp.times[0] for vp in validation_points),
                 max(vp.times[-1] for vp in validation_points))
                if validation_points else (0.0, 0.0)
            ),
            spatial_coverage=len(validation_points),
            mean_quality_score=np.mean([vp.quality_score for vp in validation_points])
                              if validation_points else 0.0,
        )
    
    def build_from_wave_buoys(
        self,
        buoys: List[WaveBuoy],
        wave_data: Dict[str, WaveBuoyTimeSeries],
        mesh: CoastalMesh,
    ) -> ValidationDataset:
        """
        Build validation dataset from wave buoy observations.
        
        Args:
            buoys: List of wave buoys
            wave_data: Dict mapping station_id to wave time series
            mesh: Coastal mesh
            
        Returns:
            ValidationDataset for wave height validation
        """
        logger.info(f"Building validation data from {len(buoys)} wave buoys...")
        
        validation_points = []
        
        for buoy in buoys:
            if buoy.station_id not in wave_data:
                continue
            
            wd = wave_data[buoy.station_id]
            
            # Find nearest mesh node
            nearest_node, distance_km = self._find_nearest_node(
                buoy.longitude, buoy.latitude, mesh
            )
            
            if distance_km > self.params.max_distance_km:
                continue
            
            # Filter valid observations
            valid = ~np.isnan(wd.wave_height)
            if np.sum(valid) < self.params.min_observations:
                continue
            
            times = wd.times[valid]
            values = wd.wave_height[valid]
            
            # Quality score based on completeness
            data_density = len(values) / max(1, (times[-1] - times[0]) / 3600)
            quality_score = min(1.0, data_density / 1.0)
            
            if quality_score < self.params.min_quality_score:
                continue
            
            # Observation error scales with wave height
            obs_error = 0.1 + 0.05 * values  # 10cm + 5% of Hs
            
            validation_points.append(ValidationPoint(
                station_id=buoy.station_id,
                variable='wave_height',
                longitude=buoy.longitude,
                latitude=buoy.latitude,
                mesh_node_idx=nearest_node,
                distance_to_node=distance_km * 1000,
                times=times,
                observations=values,
                observation_error=obs_error,
                quality_score=quality_score,
            ))
        
        logger.info(f"  Created {len(validation_points)} validation points")
        
        all_obs = np.concatenate([vp.observations for vp in validation_points]) \
                  if validation_points else np.array([])
        
        return ValidationDataset(
            points=validation_points,
            variable='wave_height',
            n_observations=len(all_obs),
            time_range=(
                (min(vp.times[0] for vp in validation_points),
                 max(vp.times[-1] for vp in validation_points))
                if validation_points else (0.0, 0.0)
            ),
            spatial_coverage=len(validation_points),
            mean_quality_score=np.mean([vp.quality_score for vp in validation_points])
                              if validation_points else 0.0,
        )
    
    def build_combined(
        self,
        tide_stations: List[TideStation],
        water_levels: Dict[str, WaterLevelTimeSeries],
        wave_buoys: List[WaveBuoy],
        wave_data: Dict[str, WaveBuoyTimeSeries],
        mesh: CoastalMesh,
    ) -> Dict[str, ValidationDataset]:
        """
        Build validation datasets for all available observation types.
        
        Args:
            tide_stations: List of tide stations
            water_levels: Water level time series
            wave_buoys: List of wave buoys
            wave_data: Wave buoy time series
            mesh: Coastal mesh
            
        Returns:
            Dict mapping variable name to ValidationDataset
        """
        result = {}
        
        if self.params.include_water_level and tide_stations:
            result['water_level'] = self.build_from_tide_gauges(
                tide_stations, water_levels, mesh
            )
        
        if self.params.include_waves and wave_buoys:
            result['wave_height'] = self.build_from_wave_buoys(
                wave_buoys, wave_data, mesh
            )
        
        return result
    
    def compute_validation_metrics(
        self,
        validation: ValidationDataset,
        predictions: Dict[int, np.ndarray],
        prediction_times: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute validation metrics given model predictions.
        
        Args:
            validation: ValidationDataset with observations
            predictions: Dict mapping node_idx to (n_times,) prediction array
            prediction_times: Time array for predictions
            
        Returns:
            Dict of metric names to values
        """
        all_obs = []
        all_pred = []
        all_weights = []
        
        for vp in validation.points:
            if vp.mesh_node_idx not in predictions:
                continue
            
            pred_at_node = predictions[vp.mesh_node_idx]
            
            # Interpolate predictions to observation times
            pred_interp = np.interp(
                vp.times, prediction_times, pred_at_node,
                left=np.nan, right=np.nan
            )
            
            # Mask out invalid
            valid = ~np.isnan(pred_interp) & ~np.isnan(vp.observations)
            
            all_obs.extend(vp.observations[valid])
            all_pred.extend(pred_interp[valid])
            all_weights.extend(1.0 / vp.observation_error[valid]**2)
        
        if not all_obs:
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'bias': np.nan,
                'correlation': np.nan,
                'skill': np.nan,
            }
        
        obs = np.array(all_obs)
        pred = np.array(all_pred)
        weights = np.array(all_weights)
        weights = weights / weights.sum()
        
        # Metrics
        errors = pred - obs
        
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        bias = np.mean(errors)
        
        # Correlation
        if np.std(obs) > 0 and np.std(pred) > 0:
            correlation = np.corrcoef(obs, pred)[0, 1]
        else:
            correlation = np.nan
        
        # Skill score (compared to mean)
        ss_tot = np.sum((obs - np.mean(obs))**2)
        ss_res = np.sum(errors**2)
        skill = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'correlation': correlation,
            'skill': skill,
            'n_observations': len(obs),
        }
    
    def _find_nearest_node(
        self,
        lon: float,
        lat: float,
        mesh: CoastalMesh,
    ) -> Tuple[int, float]:
        """
        Find the nearest mesh node to a point.
        
        Returns:
            Tuple of (node_index, distance_km)
        """
        vertices = mesh.vertices  # (N, 2) [lon, lat]
        
        # Compute distances
        distances = np.array([
            great_circle_distance(lat, lon, vertices[i, 1], vertices[i, 0])
            for i in range(len(vertices))
        ])
        
        nearest = np.argmin(distances)
        return int(nearest), float(distances[nearest])


def build_validation_dataset(
    tide_stations: Optional[List[TideStation]] = None,
    water_levels: Optional[Dict[str, WaterLevelTimeSeries]] = None,
    wave_buoys: Optional[List[WaveBuoy]] = None,
    wave_data: Optional[Dict[str, WaveBuoyTimeSeries]] = None,
    mesh: Optional[CoastalMesh] = None,
    params: Optional[ValidationParameters] = None,
) -> Dict[str, ValidationDataset]:
    """
    Convenience function to build validation datasets.
    
    Args:
        tide_stations: Optional list of tide stations
        water_levels: Optional water level time series
        wave_buoys: Optional list of wave buoys
        wave_data: Optional wave buoy time series
        mesh: Coastal mesh
        params: Validation parameters
        
    Returns:
        Dict mapping variable name to ValidationDataset
    """
    builder = ValidationBuilder(params)
    
    return builder.build_combined(
        tide_stations=tide_stations or [],
        water_levels=water_levels or {},
        wave_buoys=wave_buoys or [],
        wave_data=wave_data or {},
        mesh=mesh,
    )
