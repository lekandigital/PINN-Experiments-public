#!/usr/bin/env python3
"""Quick syntax check for all new diffgeo modules."""

import sys
# Add diffgeo parent to path, not shared (which has torch dependencies)
sys.path.insert(0, '/Users/lekan/Dev/PINN-Experiments/shared')

# Test syntax of all new modules
print('Testing charts module...')
from diffgeo.charts import atlas
from diffgeo.charts import chart_mlp
from diffgeo.charts import projections
print('  ✓ Charts module syntax OK')

print('Testing spectral module...')
from diffgeo.spectral import eigenpairs
from diffgeo.spectral import spectral_conv
from diffgeo.spectral import chebyshev
print('  ✓ Spectral module syntax OK')

print('Testing tangent module...')
from diffgeo.tangent import basis
from diffgeo.tangent import message_passing
print('  ✓ Tangent module syntax OK')

print('Testing projection module...')
from diffgeo import projection
print('  ✓ Projection module syntax OK')

print('Testing geodesic module...')
from diffgeo import geodesic
print('  ✓ Geodesic module syntax OK')

print('Testing adapters...')
from diffgeo.adapters import coastal_adapter
from diffgeo.adapters import cloth_adapter
print('  ✓ Adapters syntax OK')

print()
print('All new modules have valid Python syntax!')
