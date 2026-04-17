"""
Shared utilities for NOAA data downloaders.

Provides:
- HTTP request handling with retry and exponential backoff
- Cache management (check, store, retrieve)
- Rate limiting to be polite to NOAA servers
- Progress tracking for large downloads
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlencode

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Simple rate limiter to avoid hammering NOAA servers.
    
    Usage:
        limiter = RateLimiter(min_interval=0.5)
        limiter.wait()  # Call before each request
    """
    
    def __init__(self, min_interval: float = 0.5):
        """
        Args:
            min_interval: Minimum time in seconds between requests
        """
        self.min_interval = min_interval
        self._last_request_time: float = 0.0
    
    def wait(self) -> None:
        """Wait if needed to respect rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def reset(self) -> None:
        """Reset the timer (e.g., after a longer delay)."""
        self._last_request_time = 0.0


# Default rate limiters for each NOAA service
COOPS_RATE_LIMITER = RateLimiter(min_interval=0.5)
NDBC_RATE_LIMITER = RateLimiter(min_interval=1.0)
NCEI_RATE_LIMITER = RateLimiter(min_interval=0.5)


# =============================================================================
# Download with Retry
# =============================================================================

def download_with_retry(
    url: str,
    params: Optional[dict[str, Any]] = None,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    timeout: float = 30.0,
    rate_limiter: Optional[RateLimiter] = None,
    stream: bool = False,
) -> Any:
    """
    Download data from a URL with retry and exponential backoff.
    
    Args:
        url: URL to download from
        params: Query parameters (will be URL-encoded)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (doubles each time)
        timeout: Request timeout in seconds
        rate_limiter: Optional rate limiter to respect
        stream: If True, return raw response for streaming downloads
        
    Returns:
        Response content (bytes if stream=True, else decoded based on content-type)
        
    Raises:
        RuntimeError: If all retries fail
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests package is required: pip install requests")
    
    if rate_limiter:
        rate_limiter.wait()
    
    # Build full URL with params
    if params:
        full_url = f"{url}?{urlencode(params)}"
    else:
        full_url = url
    
    delay = initial_delay
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Downloading: {full_url} (attempt {attempt + 1}/{max_retries})")
            
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                stream=stream,
                headers={
                    'User-Agent': 'PINN-Experiments/1.0 (coastal data infrastructure)',
                }
            )
            response.raise_for_status()
            
            if stream:
                return response
            
            # Decode based on content type
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                return response.json()
            elif 'text/' in content_type:
                return response.text
            else:
                return response.content
                
        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
            last_error = e
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Rate limited
                logger.warning(f"Rate limited, waiting {delay}s")
            elif response.status_code >= 500:  # Server error, retry
                logger.warning(f"Server error {response.status_code}, retrying")
            else:  # Client error, don't retry
                raise RuntimeError(f"HTTP error {response.status_code}: {e}")
            last_error = e
            
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
            last_error = e
            
        except Exception as e:
            logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
            last_error = e
        
        # Wait before retry
        if attempt < max_retries - 1:
            logger.debug(f"Waiting {delay}s before retry")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    
    raise RuntimeError(f"Download failed after {max_retries} attempts: {last_error}")


def download_file(
    url: str,
    output_path: Path,
    params: Optional[dict[str, Any]] = None,
    max_retries: int = 3,
    rate_limiter: Optional[RateLimiter] = None,
    chunk_size: int = 8192,
    show_progress: bool = True,
) -> Path:
    """
    Download a file with progress tracking.
    
    Args:
        url: URL to download from
        output_path: Where to save the file
        params: Query parameters
        max_retries: Maximum retry attempts
        rate_limiter: Optional rate limiter
        chunk_size: Download chunk size
        show_progress: Whether to show progress bar
        
    Returns:
        Path to downloaded file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = download_with_retry(
        url, params, max_retries=max_retries, 
        rate_limiter=rate_limiter, stream=True
    )
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                if show_progress and total_size > 0:
                    pct = 100 * downloaded / total_size
                    logger.info(f"Downloaded {downloaded}/{total_size} bytes ({pct:.1f}%)")
    
    logger.info(f"Downloaded to {output_path}")
    return output_path


# =============================================================================
# Cache Management
# =============================================================================

def get_cache_key(source: str, **kwargs) -> str:
    """
    Generate a unique cache key from parameters.
    
    Args:
        source: Data source name (e.g., "coops", "ndbc")
        **kwargs: Parameters that uniquely identify the data
        
    Returns:
        MD5 hash string
    """
    # Sort kwargs for deterministic ordering
    key_str = f"{source}:" + ":".join(
        f"{k}={v}" for k, v in sorted(kwargs.items())
    )
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def get_cache_path(
    cache_dir: str | Path,
    subdir: str,
    filename: str,
) -> Path:
    """
    Get full cache path, creating directories as needed.
    """
    path = Path(cache_dir) / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def is_cached(cache_path: Path, max_age_days: Optional[float] = None) -> bool:
    """
    Check if a cached file exists and is fresh.
    
    Args:
        cache_path: Path to check
        max_age_days: Maximum age in days (None = no expiry)
        
    Returns:
        True if cache exists and is fresh
    """
    if not cache_path.exists():
        return False
    
    if max_age_days is not None:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_days = (datetime.now() - mtime).total_seconds() / 86400
        if age_days > max_age_days:
            logger.debug(f"Cache expired: {cache_path} ({age_days:.1f} days old)")
            return False
    
    return True


@dataclass
class CacheManager:
    """
    Manages cached data files for a region.
    
    Handles:
    - Cache path generation
    - Freshness checking
    - Metadata storage
    - Cache clearing
    """
    cache_dir: Path
    max_age_days: Optional[float] = None  # None = never expire
    
    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.cache_dir / "cache_metadata.json"
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load cache metadata from disk."""
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self._metadata_path, 'w') as f:
                json.dump(self._metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def get_path(self, subdir: str, filename: str) -> Path:
        """Get cache path for a file."""
        return get_cache_path(self.cache_dir, subdir, filename)
    
    def is_fresh(self, cache_path: Path) -> bool:
        """Check if a cached file is fresh."""
        return is_cached(cache_path, self.max_age_days)
    
    def record(
        self,
        cache_path: Path,
        source: str,
        params: dict[str, Any],
    ) -> None:
        """Record metadata about a cached file."""
        key = str(cache_path.relative_to(self.cache_dir))
        self._metadata[key] = {
            "source": source,
            "params": params,
            "cached_at": datetime.now().isoformat(),
            "size_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
        }
        self._save_metadata()
    
    def status(self) -> dict[str, Any]:
        """Return cache status summary."""
        total_files = 0
        total_size = 0
        by_source = {}
        
        for path, meta in self._metadata.items():
            full_path = self.cache_dir / path
            if full_path.exists():
                total_files += 1
                size = full_path.stat().st_size
                total_size += size
                
                source = meta.get("source", "unknown")
                if source not in by_source:
                    by_source[source] = {"files": 0, "size": 0}
                by_source[source]["files"] += 1
                by_source[source]["size"] += size
        
        return {
            "total_files": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "by_source": by_source,
            "cache_dir": str(self.cache_dir),
        }
    
    def clear(self, subdir: Optional[str] = None) -> int:
        """
        Clear cached files.
        
        Args:
            subdir: If specified, only clear files in this subdirectory
            
        Returns:
            Number of files deleted
        """
        import shutil
        
        count = 0
        if subdir:
            target = self.cache_dir / subdir
            if target.exists():
                for f in target.rglob("*"):
                    if f.is_file():
                        f.unlink()
                        count += 1
        else:
            for f in self.cache_dir.rglob("*"):
                if f.is_file() and f != self._metadata_path:
                    f.unlink()
                    count += 1
            self._metadata = {}
            self._save_metadata()
        
        logger.info(f"Cleared {count} cached files")
        return count


# =============================================================================
# Date Utilities
# =============================================================================

def parse_date(date_str: str) -> datetime:
    """
    Parse date string in various formats.
    
    Supports:
    - YYYYMMDD
    - YYYY-MM-DD
    - YYYY/MM/DD
    """
    for fmt in ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


def date_range_to_months(
    start_date: str,
    end_date: str,
) -> list[tuple[str, str]]:
    """
    Split a date range into monthly chunks.
    
    Used for CO-OPS API which limits requests to 31 days.
    
    Args:
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)
        
    Returns:
        List of (start, end) date tuples in YYYYMMDD format
    """
    from calendar import monthrange
    
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    chunks = []
    current = start
    
    while current <= end:
        # End of current month
        _, days_in_month = monthrange(current.year, current.month)
        month_end = current.replace(day=days_in_month)
        
        chunk_end = min(month_end, end)
        chunks.append((
            current.strftime("%Y%m%d"),
            chunk_end.strftime("%Y%m%d"),
        ))
        
        # Move to first day of next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)
    
    return chunks


# =============================================================================
# Array Utilities
# =============================================================================

def save_array(path: Path, arr: np.ndarray, compress: bool = True) -> None:
    """Save numpy array to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        np.savez_compressed(path.with_suffix('.npz'), data=arr)
    else:
        np.save(path.with_suffix('.npy'), arr)


def load_array(path: Path) -> np.ndarray:
    """Load numpy array from disk."""
    path = Path(path)
    
    if path.suffix == '.npz':
        with np.load(path) as data:
            return data['data']
    else:
        return np.load(path)


def arrays_to_hdf5(path: Path, arrays: dict[str, np.ndarray], attrs: Optional[dict] = None) -> None:
    """Save multiple arrays to HDF5 file."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required: pip install h5py")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(path, 'w') as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=arr, compression='gzip')
        
        if attrs:
            for key, value in attrs.items():
                f.attrs[key] = value


def arrays_from_hdf5(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load arrays and attributes from HDF5 file."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required: pip install h5py")
    
    arrays = {}
    attrs = {}
    
    with h5py.File(path, 'r') as f:
        for name in f.keys():
            arrays[name] = f[name][:]
        
        for key in f.attrs.keys():
            attrs[key] = f.attrs[key]
    
    return arrays, attrs
