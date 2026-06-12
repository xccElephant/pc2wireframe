"""VAE stack: per-curve VAE (``vae_curve``) + wireframe VAE (``vae_wireframe``).

Both VAEs are now custom **pure-PyTorch** attention/token models with no
``diffusers`` / ``x_transformers`` dependency:

  * **curve VAE** (``AutoencoderKL1D``) -- encodes a canonicalised curve into a
    small token latent and decodes it at arbitrary parametric ``t`` (see
    ``vae_curve.py``).
  * **wireframe VAE** (``AutoencoderKLWireframe``) -- set-to-set attention VAE
    over a set of curves (endpoints + differential adjacency + curve latent),
    derived from CLR-Wire (https://github.com/qixuema/CLR-Wire, SIGGRAPH 2025)
    but reimplemented with native ``nn.Transformer`` blocks (see
    ``vae_wireframe.py``).
"""
from __future__ import annotations

from .vae_curve import AutoencoderKL1D
from .vae_wireframe import AutoencoderKLWireframe

__all__ = [
    "AutoencoderKLWireframe",
    "AutoencoderKL1D",
]
