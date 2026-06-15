"""VAE stack: the per-curve VAE (``vae_curve``).

The per-curve VAE is a custom **pure-PyTorch** attention/token model with no
``diffusers`` / ``x_transformers`` dependency:

  * **curve VAE** (``AutoencoderKL1D``) -- encodes a canonicalised curve into a
    small token latent and decodes it at arbitrary parametric ``t`` (see
    ``vae_curve.py``). Stage 1 trains it; stage 2 reuses it **frozen** to turn a
    predicted per-edge curve latent back into a (canonical) polyline.
"""
from __future__ import annotations

from .vae_curve import AutoencoderKL1D

__all__ = [
    "AutoencoderKL1D",
]
