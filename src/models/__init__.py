"""PC2Wireframe (VQVAE branch) model package.

Lightweight modules (``LatentCompressor``, ``UtoniaEncoder``,
``MultiScaleResidualVQ``, ``EdgeSetDecoder``, ``EdgeSetCriterion``) are exported
eagerly; the heavy backbone (the frozen Utonia PTv3) is imported lazily inside
``UtoniaEncoder`` and ``vector-quantize-pytorch`` lazily inside the quantizer,
so this package can be imported without the full dependency set installed.
"""
from .latent_compressor import LatentCompressor
from .utonia_encoder import UtoniaEncoder
from .edge_set_decoder import EdgeSetDecoder
from .edge_set_criterion import EdgeSetCriterion
from .quantizer import MultiScaleResidualVQ

__all__ = [
    "LatentCompressor",
    "UtoniaEncoder",
    "EdgeSetDecoder",
    "EdgeSetCriterion",
    "MultiScaleResidualVQ",
]
