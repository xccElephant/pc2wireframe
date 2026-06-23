"""PC2Wireframe (WireframeAE branch) model package.

Lightweight modules (``LatentCompressor``, ``UtoniaEncoder``, ``WireframeAE``)
are exported eagerly; the heavy backbone (the frozen Utonia PTv3) is imported
lazily inside ``UtoniaEncoder`` so this package can be imported without the full
dependency set installed.
"""
from .latent_compressor import LatentCompressor
from .utonia_encoder import UtoniaEncoder
from .wireframe_ae import WireframeAE
from .curves import (
    sample_arc,
    sample_bezier,
    sample_curve_by_type,
    sample_line,
)

__all__ = [
    "LatentCompressor",
    "UtoniaEncoder",
    "WireframeAE",
    "sample_line",
    "sample_arc",
    "sample_bezier",
    "sample_curve_by_type",
]
