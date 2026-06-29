"""PC2Wireframe model package (PTv3 encoder + edge-query DETR decoder).

Lightweight modules (``LatentCompressor``, ``PCEncoder``, ``EdgeSetDecoder``,
``EdgeSetCriterion``) are exported eagerly; the heavy vendored stacks (PTv3, the
curve VAE) are imported lazily inside those classes so this package can be
imported without the full dependency set installed.
"""
from .edge_set_criterion import EdgeSetCriterion
from .edge_set_decoder import EdgeSetDecoder
from .latent_compressor import LatentCompressor
from .pc_encoder import PCEncoder

__all__ = [
    "LatentCompressor",
    "PCEncoder",
    "EdgeSetDecoder",
    "EdgeSetCriterion",
]
