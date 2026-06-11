"""Vendored CLR-Wire VAE stack (curve VAE + wireframe VAE).

Source: https://github.com/qixuema/CLR-Wire (SIGGRAPH 2025). Only the VAE
modules required to (a) encode a wireframe into the fixed-length latent and
(b) decode that latent back into curves + topology are vendored here. The
flow-matching / diffusion stage is intentionally left out -- in this project
the point-cloud encoder replaces the generative prior over the latent.

Imports are lazy because the VAE relies on ``x_transformers``, ``diffusers``,
``beartype`` and ``torchtyping`` which may not be installed during early
scaffolding.
"""
from __future__ import annotations

__all__ = [
    "AutoencoderKLWireframe",
    "AutoencoderKLWireframeFastEncode",
    "AutoencoderKLWireframeFastDecode",
    "AutoencoderKL1D",
    "AutoencoderKL1DFastEncode",
    "AutoencoderKL1DFastDecode",
]

_WIREFRAME = {
    "AutoencoderKLWireframe",
    "AutoencoderKLWireframeFastEncode",
    "AutoencoderKLWireframeFastDecode",
}
_CURVE = {
    "AutoencoderKL1D",
    "AutoencoderKL1DFastEncode",
    "AutoencoderKL1DFastDecode",
}


def __getattr__(name: str):
    if name in _WIREFRAME:
        from . import vae_wireframe as _m

        return getattr(_m, name)
    if name in _CURVE:
        from . import vae_curve as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
