"""VAE stack for the wireframe pipeline: per-curve VAE + curve helpers.

The per-curve VAE is a custom **pure-PyTorch** attention/token model with no
``diffusers`` / ``x_transformers`` dependency:

  * **curve VAE** (``AutoencoderKL1D``) -- encodes a canonicalised curve into a
    small token latent and decodes it at arbitrary parametric ``t`` (see
    ``vae_curve.py``).

It is trained **alone** in stage 1 (``CurveVAEModule``) and then loaded
**frozen** in stage 2 (``PC2WireframeModule``), where the edge-set decoder emits
a per-edge curve latent that this VAE decodes into a polyline. ``curve_packing``
holds the canonicalisation / latent (de)coding helpers shared by the criterion
and the reconstruction.
"""
from __future__ import annotations

from .curve_packing import (
    canonical_curves,
    decode_curve_latent,
    encode_curve_mu,
    normalized_curves_from_batch,
    orient_curves,
)
from .vae_curve import AutoencoderKL1D

__all__ = [
    "AutoencoderKL1D",
    "canonical_curves",
    "decode_curve_latent",
    "encode_curve_mu",
    "normalized_curves_from_batch",
    "orient_curves",
]
