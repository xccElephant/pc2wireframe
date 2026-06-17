"""PC2Wireframe (Rectified-Flow branch) model package.

Lightweight modules (``LatentCompressor``, ``PCEncoder``,
``RFPointSetVelocity``) are exported eagerly; the heavy vendored stacks (PTv3)
are imported lazily inside those classes so this package can be imported
without the full dependency set installed.
"""
from .latent_compressor import LatentCompressor
from .pc_encoder import PCEncoder
from .rf_pointset import RFPointSetVelocity
from .wireframe_grouper import WireframeGrouper

__all__ = [
    "LatentCompressor",
    "PCEncoder",
    "RFPointSetVelocity",
    "WireframeGrouper",
]
