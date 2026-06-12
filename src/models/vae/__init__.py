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

Imports stay lazy so pulling in one VAE does not import the other.
"""
from __future__ import annotations

__all__ = [
    "AutoencoderKLWireframe",
    "AutoencoderKL1D",
]

_WIREFRAME = {
    "AutoencoderKLWireframe",
}
_CURVE = {
    "AutoencoderKL1D",
}


def __getattr__(name: str):
    if name in _WIREFRAME:
        from . import vae_wireframe as _m

        return getattr(_m, name)
    if name in _CURVE:
        from . import vae_curve as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
