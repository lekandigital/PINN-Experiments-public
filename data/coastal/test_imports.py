#!/usr/bin/env python
"""Test imports for coastal data package."""

import sys
sys.path.insert(0, '.')

print("Testing coastal data package imports...")

try:
    from data.coastal.config import BoundingBox, RegionConfig, REGIONS
    print("  ✓ config")
except Exception as e:
    print(f"  ✗ config: {e}")

try:
    from data.coastal.datatypes import CoastalMesh, BathymetryGrid, TidalForcing
    print("  ✓ datatypes")
except Exception as e:
    print(f"  ✗ datatypes: {e}")

try:
    from data.coastal.downloaders import (
        COOPSDownloader, 
        NDBCDownloader, 
        CUDEMDownloader,
        WOADownloader,
        ShorelineDownloader,
    )
    print("  ✓ downloaders")
except Exception as e:
    print(f"  ✗ downloaders: {e}")

try:
    from data.coastal.processors import (
        CoastalMeshGenerator,
        MeshParameters,
        ForcingBuilder,
        ValidationBuilder,
    )
    print("  ✓ processors")
except Exception as e:
    print(f"  ✗ processors: {e}")

try:
    from data.coastal.manager import CoastalDataManager
    print("  ✓ manager")
except Exception as e:
    print(f"  ✗ manager: {e}")

try:
    from data.coastal import (
        CoastalDataManager,
        REGIONS,
        BoundingBox,
        get_coastal_data,
    )
    print("  ✓ main package")
except Exception as e:
    print(f"  ✗ main package: {e}")

print()
print("Available regions:")
from data.coastal.config import REGIONS
for name, bbox in REGIONS.items():
    print(f"  - {name}: lat=[{bbox.lat_min:.2f}, {bbox.lat_max:.2f}], lon=[{bbox.lon_min:.2f}, {bbox.lon_max:.2f}]")
