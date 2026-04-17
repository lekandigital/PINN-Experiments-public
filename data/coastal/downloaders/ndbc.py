"""
NOAA NDBC (National Data Buoy Center) wave buoy downloader.

Downloads wave measurements from moored buoys and coastal stations:
- Significant wave height (Hs)
- Dominant wave period (Tp)
- Average wave period
- Mean wave direction
- Wind speed and direction
- Water temperature

Data access: https://www.ndbc.noaa.gov/
"""

from __future__ import annotations

import gzip
import io
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import BoundingBox, RegionConfig
from ..datatypes import WaveBuoy, WaveBuoyTimeSeries
from .utils import (
    download_with_retry,
    NDBC_RATE_LIMITER,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

NDBC_HISTORICAL_URL = "https://www.ndbc.noaa.gov/view_text_file.php"
NDBC_REALTIME_URL = "https://www.ndbc.noaa.gov/data/realtime2"
NDBC_STATION_LIST_URL = "https://www.ndbc.noaa.gov/activestations.xml"

# NDBC missing value indicators (vary by field)
NDBC_MISSING_VALUES = {
    "WDIR": [999, 999.0],
    "WSPD": [99.0, 99.00],
    "GST": [99.0, 99.00],
    "WVHT": [99.0, 99.00],
    "DPD": [99.0, 99.00],
    "APD": [99.0, 99.00],
    "MWD": [999, 999.0],
    "PRES": [9999.0, 9999.00],
    "ATMP": [999.0, 999.00],
    "WTMP": [999.0, 999.00],
    "DEWP": [999.0, 999.00],
    "VIS": [99.0, 99.00],
    "PTDY": [99.0, 99.00],
    "TIDE": [99.0, 99.00],
}


# =============================================================================
# Downloader Class
# =============================================================================

class NDBCDownloader:
    """
    Downloads NOAA NDBC wave buoy data.
    
    Handles:
    - Station discovery within a bounding box (with padding for offshore buoys)
    - Historical standard meteorological data (yearly text files)
    - Real-time data (last 45 days)
    - Missing value handling across format variations
    
    Example:
        downloader = NDBCDownloader(cache_dir="data/coastal/cache/waves")
        
        # Discover buoys
        buoys = downloader.discover_stations(bounds, padding_degrees=2.0)
        
        # Download historical data
        data = downloader.download_historical("44014", 2024)
    """
    
    def __init__(self, cache_dir: str | Path = "data/coastal/cache/waves"):
        """
        Initialize the NDBC downloader.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._station_cache: Optional[list[dict]] = None
    
    # -------------------------------------------------------------------------
    # Station Discovery
    # -------------------------------------------------------------------------
    
    def get_all_stations(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        """
        Get list of all active NDBC stations.
        
        Args:
            force_refresh: Force download even if cached
            
        Returns:
            List of station metadata dicts
        """
        cache_path = self.cache_dir / "active_stations.xml"
        
        # Check memory cache
        if self._station_cache is not None and not force_refresh:
            return self._station_cache
        
        # Check file cache (valid for 1 day)
        if not force_refresh and cache_path.exists():
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            if age_hours < 24:
                return self._parse_station_xml(cache_path)
        
        logger.info("Downloading NDBC station list")
        
        try:
            xml_content = download_with_retry(
                NDBC_STATION_LIST_URL,
                rate_limiter=NDBC_RATE_LIMITER,
            )
            
            # Cache the XML
            with open(cache_path, 'wb') as f:
                if isinstance(xml_content, str):
                    f.write(xml_content.encode())
                else:
                    f.write(xml_content)
            
            return self._parse_station_xml(cache_path)
            
        except Exception as e:
            logger.error(f"Failed to download station list: {e}")
            
            # Try to use stale cache
            if cache_path.exists():
                logger.warning("Using stale station cache")
                return self._parse_station_xml(cache_path)
            
            return []
    
    def _parse_station_xml(self, xml_path: Path) -> list[dict[str, Any]]:
        """Parse the NDBC active stations XML file."""
        stations = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for station in root.findall('.//station'):
                try:
                    station_id = station.get('id', '')
                    lat = station.get('lat')
                    lon = station.get('lon')
                    name = station.get('name', '')
                    
                    # Some stations are ships or drifters without fixed positions
                    if lat is None or lon is None:
                        continue
                    
                    stations.append({
                        'id': station_id,
                        'name': name,
                        'lat': float(lat),
                        'lon': float(lon),
                        'type': station.get('type', 'buoy'),
                        'owner': station.get('owner', ''),
                        'pgm': station.get('pgm', ''),  # Program
                    })
                except (ValueError, TypeError):
                    continue
            
            self._station_cache = stations
            logger.info(f"Parsed {len(stations)} NDBC stations")
            
        except Exception as e:
            logger.error(f"Failed to parse station XML: {e}")
        
        return stations
    
    def discover_stations(
        self,
        bounds: BoundingBox,
        padding_degrees: float = 2.0,
    ) -> list[WaveBuoy]:
        """
        Find NDBC stations near a region.
        
        Buoys are typically offshore, so we pad the bounding box.
        
        Args:
            bounds: Geographic bounding box of the region
            padding_degrees: Padding to expand search area
            
        Returns:
            List of WaveBuoy objects
        """
        logger.info(f"Discovering NDBC stations near {bounds} (padding={padding_degrees}°)")
        
        all_stations = self.get_all_stations()
        padded_bounds = bounds.pad(padding_degrees)
        
        buoys = []
        for station in all_stations:
            lat = station['lat']
            lon = station['lon']
            
            if padded_bounds.contains(lat, lon):
                buoys.append(WaveBuoy(
                    station_id=station['id'],
                    name=station['name'],
                    latitude=lat,
                    longitude=lon,
                    water_depth_m=None,  # Not in the XML
                ))
        
        logger.info(f"Found {len(buoys)} NDBC stations near region")
        return buoys
    
    # -------------------------------------------------------------------------
    # Historical Data Download
    # -------------------------------------------------------------------------
    
    def download_historical(
        self,
        station_id: str,
        year: int,
        force: bool = False,
    ) -> Optional[WaveBuoyTimeSeries]:
        """
        Download one year of historical standard meteorological data.
        
        Args:
            station_id: NDBC station ID (e.g., "44014")
            year: Year to download
            force: Force re-download
            
        Returns:
            WaveBuoyTimeSeries or None if not available
        """
        cache_path = self.cache_dir / f"{station_id}_stdmet_{year}.npz"
        
        if not force and cache_path.exists():
            logger.debug(f"Loading cached data for {station_id} {year}")
            return self._load_cache(cache_path, station_id)
        
        logger.info(f"Downloading NDBC data for {station_id} year {year}")
        
        # Try gzipped historical file first
        url = NDBC_HISTORICAL_URL
        params = {
            "filename": f"{station_id}h{year}.txt.gz",
            "dir": "data/historical/stdmet/",
        }
        
        try:
            content = download_with_retry(
                url, params,
                rate_limiter=NDBC_RATE_LIMITER,
                timeout=60.0,  # Historical files can be large
            )
            
            # Decompress if gzipped
            if isinstance(content, bytes):
                try:
                    content = gzip.decompress(content).decode('utf-8')
                except gzip.BadGzipFile:
                    content = content.decode('utf-8')
            
            # Parse the data
            data = self._parse_stdmet_file(content, station_id)
            
            if data is not None and len(data.times) > 0:
                self._save_cache(cache_path, data)
                return data
            
        except Exception as e:
            logger.warning(f"Failed to download {station_id} {year}: {e}")
        
        # Try non-gzipped
        params["filename"] = f"{station_id}h{year}.txt"
        try:
            content = download_with_retry(
                url, params,
                rate_limiter=NDBC_RATE_LIMITER,
            )
            
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            data = self._parse_stdmet_file(content, station_id)
            
            if data is not None and len(data.times) > 0:
                self._save_cache(cache_path, data)
                return data
                
        except Exception as e:
            logger.warning(f"Failed to download {station_id} {year} (non-gz): {e}")
        
        return None
    
    def _parse_stdmet_file(
        self,
        content: str,
        station_id: str,
    ) -> Optional[WaveBuoyTimeSeries]:
        """
        Parse NDBC standard meteorological data file.
        
        Format varies by year but generally:
        - First 1-2 rows are headers
        - Columns: YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP DEWP VIS PTDY TIDE
        - Missing values vary by field (99.0, 999.0, 9999.0)
        """
        lines = content.strip().split('\n')
        
        if len(lines) < 3:
            return None
        
        # Find header line (contains column names)
        header_idx = 0
        for i, line in enumerate(lines[:3]):
            if 'YY' in line or 'YYYY' in line or '#YY' in line:
                header_idx = i
                break
        
        # Parse header
        header_line = lines[header_idx].replace('#', '').strip()
        columns = header_line.split()
        
        # Check for units line
        data_start_idx = header_idx + 1
        if data_start_idx < len(lines):
            possible_units = lines[data_start_idx]
            if 'yr' in possible_units.lower() or 'degT' in possible_units or any(
                u in possible_units for u in ['m/s', 'sec', 'hPa', 'degC']
            ):
                data_start_idx += 1
        
        # Map columns to expected names
        col_map = {}
        for i, col in enumerate(columns):
            col_upper = col.upper()
            if col_upper in ['YY', 'YYYY', '#YY']:
                col_map['year'] = i
            elif col_upper == 'MM':
                col_map['month'] = i
            elif col_upper == 'DD':
                col_map['day'] = i
            elif col_upper == 'HH':
                col_map['hour'] = i
            elif col_upper == 'MN' or col_upper == 'MM' and 'minute' not in col_map:
                # MM can be month or minute depending on position
                if 'month' in col_map:
                    col_map['minute'] = i
            elif col_upper == 'WVHT':
                col_map['wvht'] = i
            elif col_upper == 'DPD':
                col_map['dpd'] = i
            elif col_upper == 'APD':
                col_map['apd'] = i
            elif col_upper == 'MWD':
                col_map['mwd'] = i
            elif col_upper == 'WDIR':
                col_map['wdir'] = i
            elif col_upper == 'WSPD':
                col_map['wspd'] = i
            elif col_upper == 'WTMP':
                col_map['wtmp'] = i
        
        # Parse data lines
        times = []
        wvht = []
        dpd = []
        apd = []
        mwd = []
        wspd = []
        wdir = []
        wtmp = []
        
        for line in lines[data_start_idx:]:
            parts = line.split()
            if len(parts) < len(columns):
                continue
            
            try:
                # Parse timestamp
                year = int(parts[col_map.get('year', 0)])
                if year < 100:
                    year += 1900 if year > 50 else 2000
                month = int(parts[col_map.get('month', 1)])
                day = int(parts[col_map.get('day', 2)])
                hour = int(parts[col_map.get('hour', 3)])
                minute = int(parts[col_map.get('minute', 4)]) if 'minute' in col_map else 0
                
                dt = datetime(year, month, day, hour, minute)
                times.append(np.datetime64(dt))
                
                # Parse wave data
                def get_val(key, missing_key):
                    if key in col_map:
                        val = float(parts[col_map[key]])
                        missing = NDBC_MISSING_VALUES.get(missing_key, [99.0])
                        if val in missing or val >= missing[0]:
                            return np.nan
                        return val
                    return np.nan
                
                wvht.append(get_val('wvht', 'WVHT'))
                dpd.append(get_val('dpd', 'DPD'))
                apd.append(get_val('apd', 'APD'))
                mwd.append(get_val('mwd', 'MWD'))
                wspd.append(get_val('wspd', 'WSPD'))
                wdir.append(get_val('wdir', 'WDIR'))
                wtmp.append(get_val('wtmp', 'WTMP'))
                
            except (ValueError, IndexError, KeyError):
                continue
        
        if not times:
            return None
        
        return WaveBuoyTimeSeries(
            station_id=station_id,
            times=np.array(times, dtype='datetime64[s]'),
            significant_wave_height_m=np.array(wvht, dtype=np.float64),
            dominant_period_s=np.array(dpd, dtype=np.float64),
            average_period_s=np.array(apd, dtype=np.float64),
            mean_direction_deg=np.array(mwd, dtype=np.float64),
            wind_speed_m_s=np.array(wspd, dtype=np.float64),
            wind_direction_deg=np.array(wdir, dtype=np.float64),
            water_temp_c=np.array(wtmp, dtype=np.float64),
        )
    
    def _save_cache(self, path: Path, data: WaveBuoyTimeSeries) -> None:
        """Save wave data to cache."""
        np.savez_compressed(
            path,
            times=data.times.astype('int64'),
            wvht=data.significant_wave_height_m,
            dpd=data.dominant_period_s,
            apd=data.average_period_s,
            mwd=data.mean_direction_deg,
            wspd=data.wind_speed_m_s,
            wdir=data.wind_direction_deg,
            wtmp=data.water_temp_c,
        )
    
    def _load_cache(self, path: Path, station_id: str) -> WaveBuoyTimeSeries:
        """Load wave data from cache."""
        data = np.load(path)
        return WaveBuoyTimeSeries(
            station_id=station_id,
            times=data['times'].astype('datetime64[s]'),
            significant_wave_height_m=data['wvht'],
            dominant_period_s=data['dpd'],
            average_period_s=data['apd'],
            mean_direction_deg=data['mwd'],
            wind_speed_m_s=data['wspd'],
            wind_direction_deg=data['wdir'],
            water_temp_c=data['wtmp'],
        )
    
    # -------------------------------------------------------------------------
    # Multi-Year Download
    # -------------------------------------------------------------------------
    
    def download_range(
        self,
        station_id: str,
        start_year: int,
        end_year: int,
        force: bool = False,
    ) -> Optional[WaveBuoyTimeSeries]:
        """
        Download multiple years of data and concatenate.
        
        Args:
            station_id: NDBC station ID
            start_year: First year (inclusive)
            end_year: Last year (inclusive)
            force: Force re-download
            
        Returns:
            Concatenated WaveBuoyTimeSeries or None
        """
        all_data = []
        
        for year in range(start_year, end_year + 1):
            data = self.download_historical(station_id, year, force=force)
            if data is not None:
                all_data.append(data)
        
        if not all_data:
            return None
        
        # Concatenate
        return WaveBuoyTimeSeries(
            station_id=station_id,
            times=np.concatenate([d.times for d in all_data]),
            significant_wave_height_m=np.concatenate([d.significant_wave_height_m for d in all_data]),
            dominant_period_s=np.concatenate([d.dominant_period_s for d in all_data]),
            average_period_s=np.concatenate([d.average_period_s for d in all_data]),
            mean_direction_deg=np.concatenate([d.mean_direction_deg for d in all_data]),
            wind_speed_m_s=np.concatenate([d.wind_speed_m_s for d in all_data]),
            wind_direction_deg=np.concatenate([d.wind_direction_deg for d in all_data]),
            water_temp_c=np.concatenate([d.water_temp_c for d in all_data]),
        )
    
    # -------------------------------------------------------------------------
    # Batch Download
    # -------------------------------------------------------------------------
    
    def download_all_for_region(
        self,
        config: RegionConfig,
    ) -> tuple[dict[str, WaveBuoy], dict[str, WaveBuoyTimeSeries]]:
        """
        Download all wave buoy data for a region.
        
        Args:
            config: Region configuration
            
        Returns:
            Tuple of (buoys, wave_data) dicts
        """
        logger.info(f"Downloading all NDBC data for region {config.name}")
        
        # Discover or use specified stations
        if config.wave_stations:
            buoys = []
            for station_id in config.wave_stations:
                # We don't have full metadata, create placeholder
                buoys.append(WaveBuoy(
                    station_id=station_id,
                    name=station_id,
                    latitude=0,
                    longitude=0,
                ))
        else:
            buoys = self.discover_stations(
                config.bounds,
                padding_degrees=config.wave_buoy_padding_deg,
            )
        
        # Parse date range
        start_date, end_date = config.wave_date_range
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        
        buoy_dict = {}
        wave_data = {}
        
        for buoy in buoys:
            station_id = buoy.station_id
            
            try:
                data = self.download_range(
                    station_id,
                    start_year,
                    end_year,
                    force=config.force_redownload,
                )
                
                if data is not None and len(data.times) > 0:
                    buoy_dict[station_id] = buoy
                    wave_data[station_id] = data
                    logger.info(f"  {station_id}: {len(data.times)} observations")
                    
            except Exception as e:
                logger.warning(f"Failed to download {station_id}: {e}")
                continue
        
        logger.info(f"Downloaded {len(wave_data)} wave buoy datasets")
        
        return buoy_dict, wave_data
