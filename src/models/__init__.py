"""PC2Wireframe (Rectified-Flow branch) model package.

Lightweight modules (``LatentCompressor``, ``UtoniaEncoder``,
``RFPointSetVelocity``) are exported eagerly; the heavy backbone (the frozen
Utonia PTv3) is imported lazily inside ``UtoniaEncoder`` so this package can be
imported without the full dependency set installed.
"""
from .latent_compressor import LatentCompressor
from .utonia_encoder import UtoniaEncoder
from .rf_pointset import RFPointSetVelocity
from .edge_predictor import EdgePredictor

__all__ = [
    "LatentCompressor",
    "UtoniaEncoder",
    "RFPointSetVelocity",
    "EdgePredictor",
]
