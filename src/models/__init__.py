"""PC2Wireframe model package.

Lightweight modules (``LatentCompressor``, ``PCEncoder``, ``PC2WireframeModel``)
are exported eagerly; the heavy vendored stacks (PTv3, CLR-Wire VAEs) are
imported lazily inside those classes so this package can be imported without
the full dependency set installed.
"""
from .latent_compressor import LatentCompressor
from .pc_encoder import PCEncoder
from .pc2wireframe import ClrWireframeBase, PC2WireframeModel

__all__ = [
    "LatentCompressor",
    "PCEncoder",
    "ClrWireframeBase",
    "PC2WireframeModel",
]
