"""
NOAA CO-OPS (Center for Operational Oceanographic Products and Services) downloader.

Downloads tide gauge data including:
- Water level observations (6-minute intervals)
- Tidal predictions
- Harmonic constituents
- Datum information
- Current observations

API Documentation: https://api.tidesandcurrents.noaa.gov/api/prod/
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import BoundingBox, RegionConfig
from ..datatypes import (
    TideStation,
    HarmonicConstituent,
    WaterLevelTimeSeries,
    CurrentTimeSeries,
    TIDAL_CONSTITUENT_SPEEDS,
)
from .utils import (
    download_with_retry,
    get_cache_path,
    is_cached,
    date_range_to_months,
    COOPS_RATE_LIMITER,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

COOPS_BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
COOPS_METADATA_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"

# Tidal constituent descriptions
CONSTITUENT_DESCRIPTIONS = {
    "M2": "Principal lunar semidiurnal",
    "S2": "Principal solar semidiurnal",
    "N2": "Larger lunar elliptic semidiurnal",
    "K2": "Lunisolar semidiurnal",
    "K1": "Lunisolar diurnal",
    "O1": "Principal lunar diurnal",
    "P1": "Principal solar diurnal",
    "Q1": "Larger lunar elliptic diurnal",
    "M4": "Shallow water overtide of M2",
    "M6": "Shallow water overtide of M2",
    "S4": "Shallow water overtide of S2",
    "MS4": "Shallow water quarter diurnal",
    "MN4": "Shallow water quarter diurnal",
    "2N2": "Lunar elliptic semidiurnal second-order",
    "S1": "Solar diurnal",
    "Mf": "Lunar fortnightly",
    "Mm": "Lunar monthly",
    "Ssa": "Solar semiannual",
    "Sa": "Solar annual",
}


# =============================================================================
# Downloader Class
# =============================================================================

class COOPSDownloader:
    """
    Downloads NOAA CO-OPS tide gauge and current meter data.
    
    Handles:
    - Station discovery within a bounding box
    - Water level time series (with 31-day chunking for long ranges)
    - Tidal predictions
    - Harmonic constituents
    - Datum information
    - Current velocity time series
    
    Example:
        downloader = COOPSDownloader(cache_dir="data/coastal/cache/tides")
        
        # Discover stations
        stations = downloader.discover_stations(bounds)
        
        # Download data for a station
        station = downloader.download_station_metadata("8638863")
        water_levels = downloader.download_water_levels(
            "8638863", "20240101", "20240630"
        )
    """
    
    def __init__(self, cache_dir: str | Path = "data/coastal/cache/tides"):
        """
        Initialize the CO-OPS downloader.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Station Discovery
    # -------------------------------------------------------------------------
    
    def discover_stations(
        self,
        bounds: BoundingBox,
        station_type: str = "waterlevels",
    ) -> list[dict[str, Any]]:
        """
        Find all CO-OPS stations within a bounding box.
        
        Args:
            bounds: Geographic bounding box
            station_type: Type of station ("waterlevels", "currents", "met")
            
        Returns:
            List of station metadata dicts
        """
        logger.info(f"Discovering CO-OPS {station_type} stations in {bounds}")
        
        # Query the metadata API
        url = f"{COOPS_METADATA_URL}/stations.json"
        params = {"type": station_type}
        
        try:
            data = download_with_retry(
                url, params, rate_limiter=COOPS_RATE_LIMITER
            )
        except Exception as e:
            logger.error(f"Failed to discover stations: {e}")
            return []
        
        # Filter stations by bounding box
        stations = []
        for station in data.get("stations", []):
            try:
                lat = float(station.get("lat", 0))
                lon = float(station.get("lng", 0))
                
                if bounds.contains(lat, lon):
                    stations.append({
                        "id": station.get("id"),
                        "name": station.get("name"),
                        "lat": lat,
                        "lon": lon,
                        "state": station.get("state", ""),
                        "type": station_type,
                    })
            except (TypeError, ValueError):
                continue
        
        logger.info(f"Found {len(stations)} {station_type} stations in bounds")
        return stations
    
    def discover_all_station_types(
        self,
        bounds: BoundingBox,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Discover all types of stations in a bounding box.
        
        Returns:
            Dict mapping station type to list of stations
        """
        result = {}
        for station_type in ["waterlevels", "currents"]:
            result[station_type] = self.discover_stations(bounds, station_type)
        return result
    
    # -------------------------------------------------------------------------
    # Station Metadata
    # -------------------------------------------------------------------------
    
    def download_station_metadata(
        self,
        station_id: str,
        force: bool = False,
    ) -> TideStation:
        """
        Download full metadata for a station including datums and harmonics.
        
        Args:
            station_id: CO-OPS station ID
            force: Force re-download even if cached
            
        Returns:
            TideStation with metadata populated
        """
        cache_path = self.cache_dir / f"station_{station_id}_metadata.json"
        
        # Check cache
        if not force and cache_path.exists():
            logger.debug(f"Loading cached metadata for station {station_id}")
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return self._parse_station_metadata(data)
        
        logger.info(f"Downloading metadata for station {station_id}")
        
        metadata = {"station_id": station_id}
        
        # Get station info
        try:
            info_url = f"{COOPS_METADATA_URL}/stations/{station_id}.json"
            info = download_with_retry(info_url, rate_limiter=COOPS_RATE_LIMITER)
            station_info = info.get("stations", [{}])[0]
            metadata.update({
                "name": station_info.get("name", ""),
                "lat": float(station_info.get("lat", 0)),
                "lon": float(station_info.get("lng", 0)),
                "state": station_info.get("state", ""),
            })
        except Exception as e:
            logger.warning(f"Failed to get station info: {e}")
            metadata.update({"name": "", "lat": 0, "lon": 0, "state": ""})
        
        # Get datums
        try:
            datums_data = download_with_retry(
                COOPS_BASE_URL,
                params={
                    "station": station_id,
                    "product": "datums",
                    "units": "metric",
                    "format": "json",
                },
                rate_limiter=COOPS_RATE_LIMITER,
            )
            datums = {}
            for datum in datums_data.get("datums", []):
                name = datum.get("n")
                value = datum.get("v")
                if name and value is not None:
                    datums[name] = float(value)
            metadata["datums"] = datums
        except Exception as e:
            logger.warning(f"Failed to get datums for station {station_id}: {e}")
            metadata["datums"] = {}
        
        # Get harmonic constituents
        try:
            harmonics_data = download_with_retry(
                COOPS_BASE_URL,
                params={
                    "station": station_id,
                    "product": "harmonic_constituents",
                    "units": "metric",
                    "format": "json",
                },
                rate_limiter=COOPS_RATE_LIMITER,
            )
            harmonics = []
            for h in harmonics_data.get("HarmonicConstituents", []):
                name = h.get("name")
                if name:
                    harmonics.append({
                        "name": name,
                        "amplitude": float(h.get("amplitude", 0)),
                        "phase": float(h.get("phase_GMT", 0)),
                        "speed": float(h.get("speed", TIDAL_CONSTITUENT_SPEEDS.get(name, 0))),
                    })
            metadata["harmonics"] = harmonics
        except Exception as e:
            logger.warning(f"Failed to get harmonics for station {station_id}: {e}")
            metadata["harmonics"] = []
        
        # Determine capabilities
        metadata["has_water_level"] = True  # Assume true if we got this far
        metadata["has_currents"] = False  # Would need separate check
        metadata["has_met"] = False
        
        # Cache the metadata
        with open(cache_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return self._parse_station_metadata(metadata)
    
    def _parse_station_metadata(self, data: dict) -> TideStation:
        """Parse metadata dict into TideStation object."""
        harmonics = []
        for h in data.get("harmonics", []):
            name = h["name"]
            harmonics.append(HarmonicConstituent(
                name=name,
                amplitude_m=h["amplitude"],
                phase_deg=h["phase"],
                speed_deg_hr=h.get("speed", TIDAL_CONSTITUENT_SPEEDS.get(name, 0)),
                description=CONSTITUENT_DESCRIPTIONS.get(name, ""),
            ))
        
        return TideStation(
            station_id=data["station_id"],
            name=data.get("name", ""),
            latitude=data.get("lat", 0),
            longitude=data.get("lon", 0),
            state=data.get("state", ""),
            has_water_level=data.get("has_water_level", True),
            has_currents=data.get("has_currents", False),
            has_met=data.get("has_met", False),
            datums=data.get("datums", {}),
            harmonics=harmonics,
        )
    
    # -------------------------------------------------------------------------
    # Water Level Data
    # -------------------------------------------------------------------------
    
    def download_water_levels(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        datum: str = "MSL",
        force: bool = False,
    ) -> WaterLevelTimeSeries:
        """
        Download observed water level time series.
        
        The CO-OPS API limits requests to 31 days for 6-minute data.
        This method automatically splits longer ranges into monthly chunks.
        
        Args:
            station_id: CO-OPS station ID
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            datum: Vertical datum (MSL, MLLW, NAVD, etc.)
            force: Force re-download
            
        Returns:
            WaterLevelTimeSeries with observations
        """
        cache_path = self.cache_dir / f"station_{station_id}_wl_{start_date}_{end_date}_{datum}.npz"
        
        # Check cache
        if not force and cache_path.exists():
            logger.debug(f"Loading cached water levels for station {station_id}")
            return self._load_water_levels_cache(cache_path, station_id, datum)
        
        logger.info(f"Downloading water levels for station {station_id}: {start_date} to {end_date}")
        
        # Split into monthly chunks
        chunks = date_range_to_months(start_date, end_date)
        
        all_times = []
        all_levels = []
        all_flags = []
        
        for chunk_start, chunk_end in chunks:
            logger.debug(f"  Downloading chunk {chunk_start} to {chunk_end}")
            
            try:
                data = download_with_retry(
                    COOPS_BASE_URL,
                    params={
                        "station": station_id,
                        "begin_date": chunk_start,
                        "end_date": chunk_end,
                        "product": "water_level",
                        "datum": datum,
                        "units": "metric",
                        "time_zone": "gmt",
                        "format": "json",
                    },
                    rate_limiter=COOPS_RATE_LIMITER,
                )
                
                for obs in data.get("data", []):
                    try:
                        time_str = obs.get("t")
                        level = obs.get("v")
                        flag = obs.get("f", "")
                        
                        if time_str and level is not None:
                            # Parse time: "2024-01-01 00:00"
                            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                            all_times.append(np.datetime64(dt))
                            all_levels.append(float(level))
                            all_flags.append(flag)
                    except (ValueError, TypeError):
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to download chunk {chunk_start}-{chunk_end}: {e}")
                continue
        
        if not all_times:
            logger.warning(f"No water level data retrieved for station {station_id}")
            return WaterLevelTimeSeries(
                station_id=station_id,
                times=np.array([], dtype='datetime64[s]'),
                water_level_m=np.array([]),
                datum=datum,
            )
        
        # Convert to arrays
        times = np.array(all_times, dtype='datetime64[s]')
        levels = np.array(all_levels, dtype=np.float64)
        
        # Remove duplicates (can occur at chunk boundaries)
        unique_idx = np.unique(times, return_index=True)[1]
        times = times[unique_idx]
        levels = levels[unique_idx]
        
        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        levels = levels[sort_idx]
        
        # Cache
        np.savez_compressed(
            cache_path,
            times=times.astype('int64'),  # Store as int64 for datetime64
            levels=levels,
            datum=datum,
            station_id=station_id,
        )
        
        return WaterLevelTimeSeries(
            station_id=station_id,
            times=times,
            water_level_m=levels,
            datum=datum,
        )
    
    def _load_water_levels_cache(
        self,
        cache_path: Path,
        station_id: str,
        datum: str,
    ) -> WaterLevelTimeSeries:
        """Load water levels from cache."""
        data = np.load(cache_path)
        times = data['times'].astype('datetime64[s]')
        levels = data['levels']
        
        return WaterLevelTimeSeries(
            station_id=station_id,
            times=times,
            water_level_m=levels,
            datum=datum,
        )
    
    # -------------------------------------------------------------------------
    # Tidal Predictions
    # -------------------------------------------------------------------------
    
    def download_tidal_predictions(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        datum: str = "MSL",
        interval: str = "6",  # minutes: "6", "60", "hilo"
        force: bool = False,
    ) -> WaterLevelTimeSeries:
        """
        Download tidal predictions (from harmonic analysis).
        
        Args:
            station_id: CO-OPS station ID
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            datum: Vertical datum
            interval: Prediction interval ("6" = 6-min, "60" = hourly, "hilo" = high/low only)
            force: Force re-download
            
        Returns:
            WaterLevelTimeSeries with predictions
        """
        cache_path = self.cache_dir / f"station_{station_id}_pred_{start_date}_{end_date}_{datum}_{interval}.npz"
        
        if not force and cache_path.exists():
            logger.debug(f"Loading cached predictions for station {station_id}")
            return self._load_water_levels_cache(cache_path, station_id, datum)
        
        logger.info(f"Downloading tidal predictions for station {station_id}")
        
        # Split into chunks (predictions API also has limits)
        chunks = date_range_to_months(start_date, end_date)
        
        all_times = []
        all_levels = []
        
        for chunk_start, chunk_end in chunks:
            try:
                data = download_with_retry(
                    COOPS_BASE_URL,
                    params={
                        "station": station_id,
                        "begin_date": chunk_start,
                        "end_date": chunk_end,
                        "product": "predictions",
                        "datum": datum,
                        "units": "metric",
                        "time_zone": "gmt",
                        "format": "json",
                        "interval": interval,
                    },
                    rate_limiter=COOPS_RATE_LIMITER,
                )
                
                for pred in data.get("predictions", []):
                    try:
                        time_str = pred.get("t")
                        level = pred.get("v")
                        
                        if time_str and level is not None:
                            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                            all_times.append(np.datetime64(dt))
                            all_levels.append(float(level))
                    except (ValueError, TypeError):
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to download predictions chunk: {e}")
                continue
        
        if not all_times:
            return WaterLevelTimeSeries(
                station_id=station_id,
                times=np.array([], dtype='datetime64[s]'),
                water_level_m=np.array([]),
                datum=datum,
            )
        
        times = np.array(all_times, dtype='datetime64[s]')
        levels = np.array(all_levels, dtype=np.float64)
        
        # Remove duplicates and sort
        unique_idx = np.unique(times, return_index=True)[1]
        times = times[unique_idx]
        levels = levels[unique_idx]
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        levels = levels[sort_idx]
        
        # Cache
        np.savez_compressed(
            cache_path,
            times=times.astype('int64'),
            levels=levels,
        )
        
        return WaterLevelTimeSeries(
            station_id=station_id,
            times=times,
            water_level_m=levels,
            datum=datum,
        )
    
    # -------------------------------------------------------------------------
    # Current Data
    # -------------------------------------------------------------------------
    
    def download_currents(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        force: bool = False,
    ) -> CurrentTimeSeries:
        """
        Download current velocity time series.
        
        Note: Fewer stations have current data than water levels.
        
        Args:
            station_id: CO-OPS station ID
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            force: Force re-download
            
        Returns:
            CurrentTimeSeries with observations
        """
        cache_path = self.cache_dir / f"station_{station_id}_currents_{start_date}_{end_date}.npz"
        
        if not force and cache_path.exists():
            logger.debug(f"Loading cached currents for station {station_id}")
            data = np.load(cache_path)
            return CurrentTimeSeries(
                station_id=station_id,
                times=data['times'].astype('datetime64[s]'),
                speed_m_s=data['speed'],
                direction_deg=data['direction'],
            )
        
        logger.info(f"Downloading currents for station {station_id}")
        
        chunks = date_range_to_months(start_date, end_date)
        
        all_times = []
        all_speeds = []
        all_directions = []
        
        for chunk_start, chunk_end in chunks:
            try:
                data = download_with_retry(
                    COOPS_BASE_URL,
                    params={
                        "station": station_id,
                        "begin_date": chunk_start,
                        "end_date": chunk_end,
                        "product": "currents",
                        "units": "metric",
                        "time_zone": "gmt",
                        "format": "json",
                    },
                    rate_limiter=COOPS_RATE_LIMITER,
                )
                
                for obs in data.get("data", []):
                    try:
                        time_str = obs.get("t")
                        speed = obs.get("s")
                        direction = obs.get("d")
                        
                        if time_str and speed is not None and direction is not None:
                            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                            all_times.append(np.datetime64(dt))
                            all_speeds.append(float(speed))
                            all_directions.append(float(direction))
                    except (ValueError, TypeError):
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to download currents chunk: {e}")
                continue
        
        if not all_times:
            return CurrentTimeSeries(
                station_id=station_id,
                times=np.array([], dtype='datetime64[s]'),
                speed_m_s=np.array([]),
                direction_deg=np.array([]),
            )
        
        times = np.array(all_times, dtype='datetime64[s]')
        speeds = np.array(all_speeds, dtype=np.float64)
        directions = np.array(all_directions, dtype=np.float64)
        
        # Sort
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        speeds = speeds[sort_idx]
        directions = directions[sort_idx]
        
        # Cache
        np.savez_compressed(
            cache_path,
            times=times.astype('int64'),
            speed=speeds,
            direction=directions,
        )
        
        return CurrentTimeSeries(
            station_id=station_id,
            times=times,
            speed_m_s=speeds,
            direction_deg=directions,
        )
    
    # -------------------------------------------------------------------------
    # Batch Download
    # -------------------------------------------------------------------------
    
    def download_all_for_region(
        self,
        config: RegionConfig,
    ) -> tuple[dict[str, TideStation], dict[str, WaterLevelTimeSeries], dict[str, CurrentTimeSeries]]:
        """
        Discover all stations in a region and download all data.
        
        Args:
            config: Region configuration
            
        Returns:
            Tuple of (stations, water_levels, currents) dicts
        """
        logger.info(f"Downloading all CO-OPS data for region {config.name}")
        
        # Discover stations
        if config.tide_stations:
            station_ids = config.tide_stations
        else:
            discovered = self.discover_stations(config.bounds, "waterlevels")
            station_ids = [s["id"] for s in discovered]
        
        logger.info(f"Processing {len(station_ids)} stations")
        
        stations = {}
        water_levels = {}
        currents = {}
        
        start_date, end_date = config.tide_date_range
        
        for station_id in station_ids:
            try:
                # Get metadata
                station = self.download_station_metadata(
                    station_id,
                    force=config.force_redownload,
                )
                stations[station_id] = station
                
                # Get water levels
                if "water_level" in config.tide_products:
                    wl = self.download_water_levels(
                        station_id,
                        start_date,
                        end_date,
                        force=config.force_redownload,
                    )
                    if len(wl.times) > 0:
                        water_levels[station_id] = wl
                
                # Get currents if station has them
                if station.has_currents:
                    curr = self.download_currents(
                        station_id,
                        start_date,
                        end_date,
                        force=config.force_redownload,
                    )
                    if len(curr.times) > 0:
                        currents[station_id] = curr
                        
            except Exception as e:
                logger.error(f"Failed to process station {station_id}: {e}")
                continue
        
        logger.info(
            f"Downloaded data for {len(stations)} stations, "
            f"{len(water_levels)} water level series, "
            f"{len(currents)} current series"
        )
        
        return stations, water_levels, currents
