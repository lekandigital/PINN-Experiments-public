"""
Map Projections for Geographic Coordinates
==========================================

Utilities for converting between geographic (lat/lon) and Cartesian coordinates.
Used by the coastal adapter to handle Earth-surface meshes.

Supports:
- Lat/Lon ↔ 3D Cartesian on sphere
- UTM ↔ Lat/Lon
- Great circle distance
- Gnomonic projection (tangent plane)
- Stereographic projection
"""

import numpy as np
from typing import Tuple, Optional


# Earth parameters
EARTH_RADIUS_M = 6371000.0  # Mean Earth radius in meters
EARTH_OMEGA = 7.2921e-5     # Earth rotation rate (rad/s)


def latlon_to_xyz(
    lat: np.ndarray,
    lon: np.ndarray,
    radius: float = EARTH_RADIUS_M,
) -> np.ndarray:
    """
    Convert geographic coordinates to 3D Cartesian on sphere.
    
    Args:
        lat: Latitude in degrees (N positive)
        lon: Longitude in degrees (E positive)
        radius: Sphere radius (default: Earth radius in meters)
        
    Returns:
        xyz: (..., 3) Cartesian coordinates
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    
    return np.stack([x, y, z], axis=-1)


def xyz_to_latlon(
    xyz: np.ndarray,
    radius: float = EARTH_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 3D Cartesian on sphere to geographic coordinates.
    
    Args:
        xyz: (..., 3) Cartesian coordinates
        radius: Sphere radius
        
    Returns:
        lat: Latitude in degrees
        lon: Longitude in degrees
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    
    lat_rad = np.arcsin(np.clip(z / radius, -1, 1))
    lon_rad = np.arctan2(y, x)
    
    return np.degrees(lat_rad), np.degrees(lon_rad)


def great_circle_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    radius: float = EARTH_RADIUS_M,
) -> np.ndarray:
    """
    Compute great-circle (geodesic) distance using Haversine formula.
    
    Args:
        lat1, lon1: Start point(s) in degrees
        lat2, lon2: End point(s) in degrees
        radius: Sphere radius
        
    Returns:
        distance: Distance in same units as radius
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return radius * c


def compute_bearing(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Compute initial bearing (azimuth) from point 1 to point 2.
    
    Returns:
        bearing: Bearing in degrees (0 = north, 90 = east)
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon_rad = np.radians(lon2 - lon1)
    
    x = np.sin(dlon_rad) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
    
    bearing_rad = np.arctan2(x, y)
    
    return (np.degrees(bearing_rad) + 360) % 360


def gnomonic_projection(
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius: float = EARTH_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gnomonic (central/tangent plane) projection.
    
    Projects from sphere center through points to tangent plane at center.
    Great circles (geodesics) become straight lines.
    
    Args:
        lat, lon: Points to project (degrees)
        center_lat, center_lon: Projection center (degrees)
        radius: Sphere radius
        
    Returns:
        x, y: Projected coordinates in same units as radius
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    center_lat_rad = np.radians(center_lat)
    center_lon_rad = np.radians(center_lon)
    
    dlon = lon_rad - center_lon_rad
    
    # Cosine of angular distance from center
    cos_c = (np.sin(center_lat_rad) * np.sin(lat_rad) +
             np.cos(center_lat_rad) * np.cos(lat_rad) * np.cos(dlon))
    
    # Clamp to avoid division issues
    cos_c = np.clip(cos_c, 0.01, 1.0)
    
    x = radius * np.cos(lat_rad) * np.sin(dlon) / cos_c
    y = radius * (np.cos(center_lat_rad) * np.sin(lat_rad) -
                  np.sin(center_lat_rad) * np.cos(lat_rad) * np.cos(dlon)) / cos_c
    
    return x, y


def inverse_gnomonic_projection(
    x: np.ndarray,
    y: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius: float = EARTH_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse gnomonic projection: tangent plane coords to lat/lon.
    
    Args:
        x, y: Tangent plane coordinates
        center_lat, center_lon: Projection center (degrees)
        radius: Sphere radius
        
    Returns:
        lat, lon: Geographic coordinates (degrees)
    """
    center_lat_rad = np.radians(center_lat)
    center_lon_rad = np.radians(center_lon)
    
    rho = np.sqrt(x**2 + y**2)
    c = np.arctan2(rho, radius)
    
    # Handle center point
    at_center = rho < 1e-10
    
    lat_rad = np.where(
        at_center,
        center_lat_rad,
        np.arcsin(np.cos(c) * np.sin(center_lat_rad) +
                  y * np.sin(c) * np.cos(center_lat_rad) / rho)
    )
    
    lon_rad = np.where(
        at_center,
        center_lon_rad,
        center_lon_rad + np.arctan2(
            x * np.sin(c),
            rho * np.cos(center_lat_rad) * np.cos(c) - y * np.sin(center_lat_rad) * np.sin(c)
        )
    )
    
    return np.degrees(lat_rad), np.degrees(lon_rad)


def stereographic_projection(
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius: float = EARTH_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stereographic projection.
    
    Conformal (angle-preserving) projection from opposite pole through
    points to tangent plane at center.
    
    Args:
        lat, lon: Points to project (degrees)
        center_lat, center_lon: Projection center (degrees)
        radius: Sphere radius
        
    Returns:
        x, y: Projected coordinates
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    center_lat_rad = np.radians(center_lat)
    center_lon_rad = np.radians(center_lon)
    
    dlon = lon_rad - center_lon_rad
    
    # Factor k for stereographic projection
    cos_c = (np.sin(center_lat_rad) * np.sin(lat_rad) +
             np.cos(center_lat_rad) * np.cos(lat_rad) * np.cos(dlon))
    k = 2 * radius / (1 + cos_c + 1e-10)
    
    x = k * np.cos(lat_rad) * np.sin(dlon)
    y = k * (np.cos(center_lat_rad) * np.sin(lat_rad) -
             np.sin(center_lat_rad) * np.cos(lat_rad) * np.cos(dlon))
    
    return x, y


def compute_coriolis_parameter(
    lat: np.ndarray,
    omega: float = EARTH_OMEGA,
) -> np.ndarray:
    """
    Compute Coriolis parameter f = 2Ω sin(lat).
    
    Args:
        lat: Latitude in degrees
        omega: Earth rotation rate (rad/s)
        
    Returns:
        f: Coriolis parameter (1/s)
    """
    lat_rad = np.radians(lat)
    return 2 * omega * np.sin(lat_rad)


def utm_to_latlon(
    easting: np.ndarray,
    northing: np.ndarray,
    zone: int,
    hemisphere: str = 'N',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert UTM coordinates to latitude/longitude.
    
    Implements the inverse Transverse Mercator projection.
    
    Args:
        easting: UTM easting in meters
        northing: UTM northing in meters
        zone: UTM zone number (1-60)
        hemisphere: 'N' for northern, 'S' for southern
        
    Returns:
        lat, lon: Geographic coordinates in degrees
    """
    # Try to use pyproj if available
    try:
        from pyproj import Transformer
        
        # Determine EPSG code
        if hemisphere.upper() == 'N':
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone
        
        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)
        return lat, lon
        
    except ImportError:
        pass
    
    # Fallback: manual implementation
    # WGS84 ellipsoid parameters
    a = 6378137.0  # Semi-major axis
    f = 1 / 298.257223563  # Flattening
    k0 = 0.9996  # Scale factor
    
    e = np.sqrt(2 * f - f**2)  # Eccentricity
    e_prime = e / np.sqrt(1 - e**2)
    
    # Central meridian
    lon0 = (zone - 1) * 6 - 180 + 3  # degrees
    
    # Remove false easting
    x = easting - 500000.0
    
    # Remove false northing for southern hemisphere
    if hemisphere.upper() == 'S':
        y = northing - 10000000.0
    else:
        y = northing
    
    # Footpoint latitude
    M = y / k0
    mu = M / (a * (1 - e**2/4 - 3*e**4/64 - 5*e**6/256))
    
    e1 = (1 - np.sqrt(1 - e**2)) / (1 + np.sqrt(1 - e**2))
    
    phi1 = (mu + (3*e1/2 - 27*e1**3/32) * np.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32) * np.sin(4*mu)
            + (151*e1**3/96) * np.sin(6*mu)
            + (1097*e1**4/512) * np.sin(8*mu))
    
    # Compute latitude and longitude
    N1 = a / np.sqrt(1 - e**2 * np.sin(phi1)**2)
    T1 = np.tan(phi1)**2
    C1 = e_prime**2 * np.cos(phi1)**2
    R1 = a * (1 - e**2) / (1 - e**2 * np.sin(phi1)**2)**1.5
    D = x / (N1 * k0)
    
    lat_rad = (phi1 - (N1 * np.tan(phi1) / R1) * 
               (D**2/2 - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e_prime**2) * D**4/24
                + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*e_prime**2 - 3*C1**2) * D**6/720))
    
    lon_rad = (np.radians(lon0) + 
               (D - (1 + 2*T1 + C1) * D**3/6
                + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*e_prime**2 + 24*T1**2) * D**5/120) / np.cos(phi1))
    
    return np.degrees(lat_rad), np.degrees(lon_rad)


def latlon_to_utm(
    lat: np.ndarray,
    lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    Convert latitude/longitude to UTM coordinates.
    
    Auto-determines UTM zone from longitude.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        easting: UTM easting in meters
        northing: UTM northing in meters
        zone: UTM zone number
        hemisphere: 'N' or 'S'
    """
    # Determine zone (simplified - doesn't handle Norway/Svalbard exceptions)
    zone = int((lon + 180) / 6) + 1
    hemisphere = 'N' if np.mean(lat) >= 0 else 'S'
    
    try:
        from pyproj import Transformer
        
        if hemisphere == 'N':
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone
        
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        easting, northing = transformer.transform(lon, lat)
        return easting, northing, zone, hemisphere
        
    except ImportError:
        # Fallback implementation would go here
        raise ImportError(
            "UTM conversion requires pyproj. Install with: pip install pyproj"
        )


def domain_extent_km(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Tuple[float, float]:
    """
    Compute approximate domain extent in kilometers.
    
    Args:
        lat_min, lat_max, lon_min, lon_max: Domain bounds in degrees
        
    Returns:
        width_km: East-west extent
        height_km: North-south extent
    """
    # Height (north-south) is straightforward
    height_m = great_circle_distance(lat_min, 0, lat_max, 0)
    height_km = height_m / 1000
    
    # Width varies with latitude - use center latitude
    center_lat = (lat_min + lat_max) / 2
    width_m = great_circle_distance(center_lat, lon_min, center_lat, lon_max)
    width_km = width_m / 1000
    
    return width_km, height_km


def curvature_scale_factor(
    domain_size_km: float,
    earth_radius_km: float = EARTH_RADIUS_M / 1000,
) -> float:
    """
    Estimate the scale of curvature effects for a domain of given size.
    
    For small domains (< 10 km), curvature is negligible.
    For large domains (> 100 km), curvature becomes significant.
    
    Args:
        domain_size_km: Characteristic domain size in km
        earth_radius_km: Earth radius in km
        
    Returns:
        factor: Approximate relative curvature effect (0 = flat, 1 = significant)
    """
    # Angular size of domain
    theta = domain_size_km / earth_radius_km  # radians
    
    # Deviation from flat: (1 - cos(θ/2)) / (θ/2)²
    # For small θ: ≈ θ²/8
    # For θ = 0.1 rad (637 km): ≈ 0.00125 (0.1%)
    # For θ = 0.01 rad (64 km): ≈ 0.0000125 (0.001%)
    
    curvature_factor = (theta ** 2) / 8
    
    return curvature_factor
