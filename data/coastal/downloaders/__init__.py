"""
Downloader package for NOAA coastal data sources.
"""

from .utils import (
    download_with_retry,
    get_cache_path,
    is_cached,
    CacheManager,
    RateLimiter,
)
from .coops import COOPSDownloader
from .ndbc import NDBCDownloader
from .cudem import CUDEMDownloader
from .woa import WOADownloader
from .shoreline import ShorelineDownloader

__all__ = [
    # Utilities
    "download_with_retry",
    "get_cache_path",
    "is_cached",
    "CacheManager",
    "RateLimiter",
    # Downloaders
    "COOPSDownloader",
    "NDBCDownloader", 
    "CUDEMDownloader",
    "WOADownloader",
    "ShorelineDownloader",
]
