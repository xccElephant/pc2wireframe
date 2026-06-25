"""PC2Wireframe (VQVAE branch) model package.

Lightweight modules (``LatentCompressor``, ``UtoniaEncoder``,
``MultiScaleResidualVQ``, ``JointSetDecoder``, ``JointSetCriterion``) are
exported eagerly; the heavy backbone (the frozen Utonia PTv3) is imported lazily
inside ``UtoniaEncoder`` and ``vector-quantize-pytorch`` lazily inside the
quantizer, so this package can be imported without the full dependency set
installed.
"""
from .latent_compressor import LatentCompressor
from .utonia_encoder import UtoniaEncoder
from .joint_set_decoder import JointSetDecoder
from .joint_set_criterion import JointSetCriterion
from .quantizer import MultiScaleResidualVQ

__all__ = [
    "LatentCompressor",
    "UtoniaEncoder",
    "JointSetDecoder",
    "JointSetCriterion",
    "MultiScaleResidualVQ",
]
