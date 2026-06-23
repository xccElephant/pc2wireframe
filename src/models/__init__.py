"""PC2Wireframe (VQVAE branch) model package.

Lightweight modules (``LatentCompressor``, ``UtoniaEncoder``,
``MultiScaleResidualVQ``, ``WireframeGraphDecoder``) are exported eagerly; the
heavy backbone (the frozen Utonia PTv3) is imported lazily inside
``UtoniaEncoder`` and ``vector-quantize-pytorch`` lazily inside the quantizer,
so this package can be imported without the full dependency set installed.
"""
from .latent_compressor import LatentCompressor
from .utonia_encoder import UtoniaEncoder
from .wireframe_graph_decoder import WireframeGraphDecoder, knn_candidate_pairs
from .quantizer import MultiScaleResidualVQ
from .curves import (
    sample_arc,
    sample_bezier,
    sample_curve_by_type,
    sample_line,
)

__all__ = [
    "LatentCompressor",
    "UtoniaEncoder",
    "WireframeGraphDecoder",
    "knn_candidate_pairs",
    "MultiScaleResidualVQ",
    "sample_line",
    "sample_arc",
    "sample_bezier",
    "sample_curve_by_type",
]
