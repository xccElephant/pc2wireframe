"""VAE stack for the joint vertex+edge branch: per-curve VAE + curve helpers.

The per-curve VAE is a custom **pure-PyTorch** attention/token model with no
``diffusers`` / ``x_transformers`` dependency:

  * **curve VAE** (``AutoencoderKL1D``) -- encodes a canonicalised curve into a
    small token latent and decodes it at arbitrary parametric ``t`` (see
    ``vae_curve.py``).

Unlike the original two-stage pipeline, the joint branch trains this VAE
**jointly** with the decoder (no freezing): the edge head learns to emit latents
the shared, trainable decoder recognises, while an autoencoding anchor path
keeps the latent space meaningful (see ``src/models/joint_set_criterion.py``).
``curve_packing`` holds the canonicalisation / latent (de)coding helpers shared
by the criterion and the reconstruction.
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
