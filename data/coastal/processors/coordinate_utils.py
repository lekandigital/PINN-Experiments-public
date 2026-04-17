"""
Coordinate transformation utilities.

Functions for converting between coordinate systems and computing distances.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


# Earth parameters
EARTH_RADIUS_M = 6371000.0
EARTH_RADIUS_KM = 6371.0


def get_utm_zone(lon: float, lat: float) -> Tuple[int, str]:
    """
    Determine the UTM zone for a given longitude/latitude.
    
    Args:
        lon: Longitude in degrees (-180 to 180)
        lat: Latitude in degrees (-90 to 90)
        
    Returns:
        Tuple of (zone_number, hemisphere) where hemisphere is 'N' or 'S'
    """
    # Handle special zones for Norway and Svalbard
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    elif 72 <= lat < 84:
        if 0 <= lon < 9:
            zone = 31
        elif 9 <= lon < 21:
            zone = 33
        elif 21 <= lon < 33:
            zone = 35
        elif 33 <= lon < 42:
            zone = 37
        else:
            zone = int((lon + 180) / 6) + 1
    else:
        zone = int((lon + 180) / 6) + 1
    
    hemisphere = 'N' if lat >= 0 else 'S'
    return zone, hemisphere


def get_utm_epsg(lon: float, lat: float) -> int:
    """
    Get the EPSG code for the UTM zone containing a point.
    
    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees
        
    Returns:
        EPSG code (e.g., 32618 for UTM zone 18N)
    """
    zone, hemisphere = get_utm_zone(lon, lat)
    if hemisphere == 'N':
        return 32600 + zone
    else:
        return 32700 + zone


def latlon_to_utm(
    lat: np.ndarray,
    lon: np.ndarray,
    zone: Optional[int] = None,
    hemisphere: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    Convert lat/lon coordinates to UTM (x, y in meters).
    
    Uses a simple approximation formula. For high accuracy, use pyproj.
    
    Args:
        lat: Latitude(s) in degrees
        lon: Longitude(s) in degrees
        zone: UTM zone number (auto-detected if None)
        hemisphere: 'N' or 'S' (auto-detected if None)
        
    Returns:
        Tuple of (x, y, zone, hemisphere) where x, y are in meters
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    
    # Determine zone from center of data if not specified
    if zone is None or hemisphere is None:
        center_lon = float(np.mean(lon))
        center_lat = float(np.mean(lat))
        zone, hemisphere = get_utm_zone(center_lon, center_lat)
    
    # UTM central meridian for this zone
    central_meridian = (zone - 1) * 6 - 180 + 3
    
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    central_rad = np.radians(central_meridian)
    
    # WGS84 ellipsoid parameters
    a = 6378137.0  # Semi-major axis
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f**2  # First eccentricity squared
    e_prime2 = e2 / (1 - e2)  # Second eccentricity squared
    
    k0 = 0.9996  # UTM scale factor
    
    # Compute intermediate values
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    T = np.tan(lat_rad)**2
    C = e_prime2 * np.cos(lat_rad)**2
    A = (lon_rad - central_rad) * np.cos(lat_rad)
    
    # Compute meridional arc
    M = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * lat_rad
             - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * np.sin(2*lat_rad)
             + (15*e2**2/256 + 45*e2**3/1024) * np.sin(4*lat_rad)
             - (35*e2**3/3072) * np.sin(6*lat_rad))
    
    # Compute x (easting) and y (northing)
    x = k0 * N * (A + (1-T+C)*A**3/6 
                  + (5-18*T+T**2+72*C-58*e_prime2)*A**5/120)
    
    y = k0 * (M + N * np.tan(lat_rad) * (A**2/2 
              + (5-T+9*C+4*C**2)*A**4/24
              + (61-58*T+T**2+600*C-330*e_prime2)*A**6/720))
    
    # Add false easting and northing
    x = x + 500000.0  # False easting
    if hemisphere == 'S':
        y = y + 10000000.0  # False northing for southern hemisphere
    
    return x, y, zone, hemisphere


def utm_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    zone: int,
    hemisphere: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert UTM coordinates to lat/lon.
    
    Uses a simple approximation formula. For high accuracy, use pyproj.
    
    Args:
        x: Easting(s) in meters
        y: Northing(s) in meters
        zone: UTM zone number
        hemisphere: 'N' or 'S'
        
    Returns:
        Tuple of (lat, lon) in degrees
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Remove false easting/northing
    x = x - 500000.0
    if hemisphere == 'S':
        y = y - 10000000.0
    
    # WGS84 parameters
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f**2
    e_prime2 = e2 / (1 - e2)
    
    k0 = 0.9996
    
    # Central meridian
    central_meridian = (zone - 1) * 6 - 180 + 3
    
    # Compute footprint latitude
    M = y / k0
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    
    phi1 = (mu + (3*e1/2 - 27*e1**3/32) * np.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32) * np.sin(4*mu)
            + (151*e1**3/96) * np.sin(6*mu)
            + (1097*e1**4/512) * np.sin(8*mu))
    
    # Compute final latitude and longitude
    N1 = a / np.sqrt(1 - e2 * np.sin(phi1)**2)
    T1 = np.tan(phi1)**2
    C1 = e_prime2 * np.cos(phi1)**2
    R1 = a * (1 - e2) / (1 - e2 * np.sin(phi1)**2)**1.5
    D = x / (N1 * k0)
    
    lat = phi1 - (N1 * np.tan(phi1) / R1) * (D**2/2 
          - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e_prime2) * D**4/24
          + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*e_prime2 - 3*C1**2) * D**6/720)
    
    lon = central_meridian + np.degrees(
        (D - (1 + 2*T1 + C1) * D**3/6
         + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*e_prime2 + 24*T1**2) * D**5/120) / np.cos(phi1)
    )
    
    lat = np.degrees(lat)
    
    return lat, lon


def great_circle_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate great-circle distance between two points in kilometers.
    
    Uses the Haversine formula.
    
    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)
        
    Returns:
        Distance in kilometers
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return EARTH_RADIUS_KM * c


def great_circle_distance_array(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized great-circle distance calculation.
    
    Args:
        lat1, lon1: First point(s) (degrees)
        lat2, lon2: Second point(s) (degrees)
        
    Returns:
        Distance(s) in kilometers
    """
    lat1 = np.asarray(lat1)
    lon1 = np.asarray(lon1)
    lat2 = np.asarray(lat2)
    lon2 = np.asarray(lon2)
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return EARTH_RADIUS_KM * c


def compute_distance_to_shore(
    vertices_lon: np.ndarray,
    vertices_lat: np.ndarray,
    shoreline_segments: list[np.ndarray],
) -> np.ndarray:
    """
    Compute distance from each vertex to the nearest shoreline point.
    
    Args:
        vertices_lon: (V,) array of vertex longitudes
        vertices_lat: (V,) array of vertex latitudes
        shoreline_segments: List of (N, 2) arrays of [lon, lat] shoreline points
        
    Returns:
        (V,) array of distances in meters
    """
    if not shoreline_segments:
        return np.full(len(vertices_lon), np.nan)
    
    # Concatenate all shoreline points
    all_shore = np.vstack(shoreline_segments)
    shore_lon = all_shore[:, 0]
    shore_lat = all_shore[:, 1]
    
    n_vertices = len(vertices_lon)
    distances = np.full(n_vertices, np.inf)
    
    # For each vertex, find distance to nearest shoreline point
    # This is O(V * S) where S is total shoreline points - could be optimized with KD-tree
    for i in range(n_vertices):
        dists = great_circle_distance_array(
            vertices_lat[i], vertices_lon[i],
            shore_lat, shore_lon
        )
        distances[i] = np.min(dists)
    
    # Convert km to meters
    return distances * 1000.0


def point_in_polygon(
    point_lon: float,
    point_lat: float,
    polygon_lon: np.ndarray,
    polygon_lat: np.ndarray,
) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point_lon, point_lat: Point coordinates
        polygon_lon, polygon_lat: Polygon vertices (closed, first==last)
        
    Returns:
        True if point is inside polygon
    """
    n = len(polygon_lon)
    inside = False
    
    j = n - 1
    for i in range(n):
        if ((polygon_lat[i] > point_lat) != (polygon_lat[j] > point_lat)) and \
           (point_lon < (polygon_lon[j] - polygon_lon[i]) * (point_lat - polygon_lat[i]) / 
            (polygon_lat[j] - polygon_lat[i]) + polygon_lon[i]):
            inside = not inside
        j = i
    
    return inside


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial bearing from point 1 to point 2.
    
    Args:
        lat1, lon1: Starting point (degrees)
        lat2, lon2: Ending point (degrees)
        
    Returns:
        Bearing in degrees (0-360, clockwise from north)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    
    bearing_rad = math.atan2(x, y)
    return (math.degrees(bearing_rad) + 360) % 360


def destination_point(
    lat: float,
    lon: float,
    bearing_deg: float,
    distance_km: float,
) -> Tuple[float, float]:
    """
    Calculate destination point given start, bearing, and distance.
    
    Args:
        lat, lon: Starting point (degrees)
        bearing_deg: Bearing in degrees (clockwise from north)
        distance_km: Distance in kilometers
        
    Returns:
        Tuple of (lat, lon) for destination point
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_deg)
    
    angular_dist = distance_km / EARTH_RADIUS_KM
    
    lat2 = math.asin(
        math.sin(lat_rad) * math.cos(angular_dist) +
        math.cos(lat_rad) * math.sin(angular_dist) * math.cos(bearing_rad)
    )
    
    lon2 = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_dist) * math.cos(lat_rad),
        math.cos(angular_dist) - math.sin(lat_rad) * math.sin(lat2)
    )
    
    return math.degrees(lat2), math.degrees(lon2)
