"""VAE stack: per-curve VAE (``vae_curve``) + wireframe VAE (``vae_wireframe``).

Both VAEs are now custom **pure-PyTorch** attention/token models with no
``diffusers`` / ``x_transformers`` dependency:

  * **curve VAE** (``AutoencoderKL1D``) -- encodes a canonicalised curve into a
    small token latent and decodes it at arbitrary parametric ``t`` (see
    ``vae_curve.py``).
  * **wireframe VAE** (``AutoencoderKLWireframe``) -- a **graph** VAE that
    autoencodes a wireframe as an attributed graph (node coords + per-edge
    curve latent). The decoder predicts a node set (coord + existence), a
    symmetric inner-product adjacency, and per-edge curve latents, trained with
    Hungarian node matching (see ``vae_wireframe.py``).
"""
from __future__ import annotations

from .vae_curve import AutoencoderKL1D
from .vae_wireframe import AutoencoderKLWireframe

__all__ = [
    "AutoencoderKLWireframe",
    "AutoencoderKL1D",
]
