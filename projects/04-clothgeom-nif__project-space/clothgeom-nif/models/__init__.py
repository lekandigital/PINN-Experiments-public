"""
ClothGeom-NIF Models Package

Neural Implicit Field decoders for cloth geometry reconstruction.
"""

from .siren import (
    SineActivation,
    SirenLinear,
    SirenLayer,
    ModulatedSirenLayer,
    SirenNetwork
)

from .nif_decoder import (
    NIFDecoder,
    NIFDecoderWithHash,
    create_nif_decoder
)

__all__ = [
    # SIREN components
    'SineActivation',
    'SirenLinear',
    'SirenLayer',
    'ModulatedSirenLayer',
    'SirenNetwork',
    # NIF Decoder
    'NIFDecoder',
    'NIFDecoderWithHash',
    'create_nif_decoder'
]
