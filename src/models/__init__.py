"""PC2Wireframe model package.

Lightweight modules (``LatentCompressor``, ``PCEncoder``, ``WireframeDecoder``,
``PC2WireframeModel``, ``WireframeCriterion``) are exported eagerly; the heavy
vendored stacks (PTv3, curve VAE) are imported lazily inside those classes so
this package can be imported without the full dependency set installed.
"""
from .criterion import WireframeCriterion
from .latent_compressor import LatentCompressor
from .pc2wireframe import PC2WireframeModel
from .pc_encoder import PCEncoder
from .wireframe_decoder import WireframeDecoder

__all__ = [
    "LatentCompressor",
    "PCEncoder",
    "WireframeDecoder",
    "PC2WireframeModel",
    "WireframeCriterion",
]
